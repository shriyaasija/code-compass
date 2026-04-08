"""
MCTSSearch: Full Monte Carlo Tree Search algorithm for code retrieval.

Implements the four phases of MCTS:
1. SELECTION:    Walk down the tree following UCB1 (best_child)
2. EXPANSION:    Create MCTSNode children for the reached node
3. SIMULATION:   LLM scores all expanded children (batch call)
4. BACKPROPAGATION: Update value estimates from leaf back to root

Key design decisions:
- max_iterations=50 balances accuracy vs LLM cost (~5 LLM calls per query)
- c_explore=√2 is theoretically optimal for [0,1] rewards
- Early termination when top leaf has >80% of visits (convergence)
- Result extraction collects ALL visited terminal nodes ranked by value
"""

import time
from typing import List, Dict, Any, Optional

from research.mcts.mcts_node import MCTSNode
from research.mcts.simulation import LLMSimulator


class MCTSSearch:
    """
    Monte Carlo Tree Search for code retrieval over AST hierarchies.

    Given a code repository tree (PageIndex JSON) and a natural language query,
    uses MCTS to efficiently navigate the tree and find the most relevant
    leaf nodes (functions, methods, classes).
    """

    def __init__(
        self,
        llm_client,
        max_iterations: int = 50,
        c_explore: float = 1.414,
        convergence_threshold: float = 0.8,
        convergence_check_after: int = 10,
        verbose: bool = True,
    ):
        """
        Args:
            llm_client: OllamaLLM or LMStudioLLM instance.
            max_iterations: Maximum MCTS iterations per query.
            c_explore: UCB1 exploration constant. √2 ≈ 1.414 is default.
            convergence_threshold: Fraction of visits for early termination.
            convergence_check_after: Start checking convergence after N iterations.
            verbose: Whether to print search progress.
        """
        self.simulator = LLMSimulator(llm_client)
        self.max_iterations = max_iterations
        self.c_explore = c_explore
        self.convergence_threshold = convergence_threshold
        self.convergence_check_after = convergence_check_after
        self.verbose = verbose

    def search(
        self,
        tree: Dict[str, Any],
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Run MCTS search on a code tree for a given query.

        Args:
            tree: The PageIndex JSON tree (root node dict).
            query: Natural language query.
            top_k: Maximum number of results to return.

        Returns:
            List of result dicts sorted by MCTS value (descending), each containing:
            - node_id, name, node_type, summary, path
            - similarity_score (average MCTS value)
            - metadata (file_path, start_line, end_line, signature, docstring)
            - mcts_stats (visit_count, average_value)
        """
        start_time = time.time()

        # Reset simulator stats for this query
        self.simulator.reset_stats()

        # Create root MCTSNode
        root = MCTSNode(tree)

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"🔍 MCTS SEARCH: '{query[:80]}'")
            print(f"   Max iterations: {self.max_iterations}, c_explore: {self.c_explore}")
            print(f"{'=' * 70}")

        # Run MCTS iterations
        actual_iterations = 0
        early_terminated = False
        for iteration in range(1, self.max_iterations + 1):
            actual_iterations = iteration
            self._run_one_iteration(root, query, iteration)

            # Check for early termination (convergence)
            if iteration >= self.convergence_check_after:
                if self._check_convergence(root):
                    if self.verbose:
                        print(f"  ✅ Early termination at iteration {iteration} (converged)")
                    early_terminated = True
                    break
        
        # Store for benchmarking tools
        self.actual_iterations = actual_iterations
        self.early_terminated = early_terminated

        # Collect results from all visited terminals
        results = self._collect_results(root)

        elapsed = time.time() - start_time

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"✅ MCTS COMPLETE")
            print(f"   Iterations: {actual_iterations}/{self.max_iterations}")
            print(f"   Leaf nodes found: {len(results)}")
            print(f"   LLM calls: {self.simulator.total_llm_calls}")
            print(f"   Time: {elapsed:.2f}s")
            print(f"{'=' * 70}\n")

        return results[:top_k]

    def _run_one_iteration(self, root: MCTSNode, query: str, iteration: int):
        """
        Execute one full MCTS iteration: Select → Expand → Simulate → Backprop.
        """
        # ── PHASE 1: SELECTION ──
        # Walk down the tree following UCB1 until we reach an unexpanded
        # or terminal node.
        node = root
        while node.is_expanded and not node.is_terminal and node.children:
            node = node.best_child(self.c_explore)
            if node is None:
                return  # Shouldn't happen, but be safe

        # ── PHASE 2: EXPANSION ──
        # If the node is not terminal and not expanded, expand it.
        if not node.is_terminal and not node.is_expanded:
            children = node.expand()

            if not children:
                # No children to explore — treat as terminal
                node.backpropagate(0.0)
                return

            if self.verbose and iteration <= 5:
                name = node.tree_node.get('title', node.tree_node.get('name', '?'))
                print(f"  Iter {iteration}: Expanding '{name}' ({len(children)} children)")

            # ── PHASE 3: SIMULATION ──
            # Score all children in one batch LLM call.
            scores = self.simulator.batch_score_children(
                children, query, node
            )

            # Initialize children with their scores and pick the best
            best_child = None
            best_score = -1.0

            for child in children:
                child_name = child.tree_node.get(
                    'title', child.tree_node.get('name', 'unknown')
                )
                score = scores.get(child_name, 0.5)

                # Store initial score on the child for later use
                child.tree_node['_mcts_initial_score'] = score

                if score > best_score:
                    best_score = score
                    best_child = child

            # ── PHASE 4: BACKPROPAGATION ──
            # Backpropagate the best child's score up through the tree.
            if best_child is not None:
                best_child.backpropagate(best_score)

                # Also give partial credit to other promising children
                # so they accumulate visits for UCB1 exploration
                for child in children:
                    if child is not best_child:
                        child_name = child.tree_node.get(
                            'title', child.tree_node.get('name', 'unknown')
                        )
                        child_score = scores.get(child_name, 0.5)
                        if child_score > 0.3:
                            # Give one visit with their score so UCB1 has info
                            child.visit_count += 1
                            child.total_value += child_score
            else:
                node.backpropagate(0.0)

        elif node.is_terminal:
            # Terminal node reached — simulate its direct relevance
            score = node.tree_node.get('_mcts_initial_score', None)
            if score is None:
                score = self.simulator.simulate_node(node, query)
            node.backpropagate(score)

    def _check_convergence(self, root: MCTSNode) -> bool:
        """
        Check if MCTS has converged (top leaf dominates visit distribution).

        Convergence = the most-visited terminal node has received
        more than `convergence_threshold` fraction of root's visits.
        """
        if root.visit_count == 0:
            return False

        terminals = self._get_all_terminals(root)
        if not terminals:
            return False

        top_terminal = max(terminals, key=lambda n: n.visit_count)
        visit_fraction = top_terminal.visit_count / root.visit_count

        return visit_fraction >= self.convergence_threshold

    def _get_all_terminals(self, root: MCTSNode) -> List[MCTSNode]:
        """Collect all terminal nodes in the MCTS tree."""
        terminals = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.is_terminal and node.visit_count > 0:
                terminals.append(node)
            for child in node.children:
                stack.append(child)
        return terminals

    def _collect_results(self, root: MCTSNode) -> List[Dict[str, Any]]:
        """
        Walk the MCTS tree and collect all visited terminal nodes as results.

        Returns results sorted by average_value (descending).
        """
        results = []
        stack = [root]

        while stack:
            node = stack.pop()

            if node.is_terminal and node.visit_count > 0:
                tn = node.tree_node
                results.append({
                    'node_id': tn.get('node_id', tn.get('title', 'unknown')),
                    'name': tn.get('title', tn.get('name', 'unnamed')),
                    'node_type': tn.get('type', tn.get('node_type', 'unknown')),
                    'summary': tn.get('summary', ''),
                    'path': tn.get('path', tn.get('file_path', '')),
                    'similarity_score': node.average_value,
                    'metadata': {
                        'file_path': tn.get('path', tn.get('file_path', '')),
                        'start_line': tn.get('start_line'),
                        'end_line': tn.get('end_line'),
                        'signature': tn.get('signature', ''),
                        'docstring': tn.get('summary', tn.get('docstring', '')),
                    },
                    'mcts_stats': {
                        'visit_count': node.visit_count,
                        'average_value': node.average_value,
                        'depth': node.depth(),
                        'path_from_root': node.path_from_root(),
                    },
                })

            for child in node.children:
                stack.append(child)

        # Sort by average value (descending)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results

    @property
    def llm_call_count(self) -> int:
        """Total LLM calls made during the last/current search."""
        return self.simulator.total_llm_calls

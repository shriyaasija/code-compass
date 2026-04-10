import math
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

from research.mcts.mcts_node import MCTSNode
from research.mcts.relevance_prior import RelevancePrior


class PUCTSearch:
    """
    PUCT-guided MCTS search over hierarchical code trees.
   
    Lifecycle:
    1. __init__: load prior MLP
    2. search(tree, query): run PUCT-MCTS for one query
       - get query embedding
       - run N iterations of Select→Expand→Simulate→Backprop
       - collect leaf results
       - call online_update to adapt prior
    3. Repeat for next query (prior improves over session)
    """

    def __init__(
        self,
        llm_client,
        prior_path: str = "research/mcts/prior.pt",
        embed_model_name: str = "all-MiniLM-L6-v2",
        max_iterations: int = 30,
        c_puct: float = 1.5,
        online_update: bool = True,
        online_lr: float = 5e-4,
        verbose: bool = True,
    ):
        """
        Args:
            llm_client:       Your LMStudioLLM or OllamaLLM instance
            prior_path:       Path to trained prior.pt
            embed_model_name: Must match what was used to build training data
            max_iterations:   MCTS iterations per query. 30 is good for depth-4 trees.
            c_puct:           Exploration constant. 1.5 works well.
            online_update:    Whether to adapt prior after each query.
            online_lr:        Learning rate for online updates.
            verbose:          Print search progress.
        """
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.c_puct = c_puct
        self.do_online_update = online_update
        self.online_lr = online_lr
        self.verbose = verbose

        # Load embedding model
        from sentence_transformers import SentenceTransformer
        self.embed_model = SentenceTransformer(embed_model_name)

        # Load or initialize prior
        import os
        if os.path.exists(prior_path):
            self.prior = RelevancePrior.load(prior_path)
            if self.verbose:
                print(f"Loaded prior from: {prior_path}")
        else:
            print(f"WARNING: Prior not found at {prior_path}. "
                  f"Using untrained prior (run train_prior.py first).")
            self.prior = RelevancePrior(embed_dim=384)

        self.prior.eval()

        # Stats for the paper
        self.llm_call_count = 0
        self.prior_call_count = 0

    def search(
        self,
        tree: Dict,
        query: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Run PUCT-MCTS search on a tree for a given query.
       
        Returns list of result dicts in same format as original MCTSSearch,
        ranked by average_value (best first).
        """
        start_time = time.time()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"PUCT SEARCH: '{query[:60]}'")
            print(f"  max_iterations={self.max_iterations}, c_puct={self.c_puct}")
            print(f"{'='*60}")

        # Embed the query once, reuse for all scoring
        query_emb = self.embed_model.encode(query, show_progress_bar=False)

        # Reset per-query stats
        self.llm_call_count = 0
        self.prior_call_count = 0

        # Build root MCTS node
        root = MCTSNode(tree_node=tree, parent=None)

        # Run MCTS iterations
        all_visited_nodes = []  # track for online update

        for iteration in range(self.max_iterations):
            # ── SELECT ────────────────────────────────────────────────
            node = self._select(root)

            # ── EXPAND ────────────────────────────────────────────────
            if not node.is_terminal and not node.is_expanded:
                children = node.expand()
                self._assign_priors(children, query_emb)

            # ── SIMULATE ──────────────────────────────────────────────
            value = self._simulate(node, query, query_emb)

            # ── BACKPROPAGATE ─────────────────────────────────────────
            node.backpropagate(value)

            # Track visited nodes for online update
            all_visited_nodes.append(node)

            # Early termination: if root has a clearly dominant child
            if iteration > 10 and root.children:
                top_child = max(root.children, key=lambda c: c.visit_count)
                if top_child.visit_count > 0.85 * root.visit_count:
                    if self.verbose:
                        print(f"  Early termination at iteration {iteration+1}")
                    break

        elapsed = time.time() - start_time

        if self.verbose:
            print(f"\n  Done: {iteration+1} iterations, "
                  f"{self.llm_call_count} LLM calls, "
                  f"{self.prior_call_count} prior calls, "
                  f"{elapsed:.1f}s")

        # ── ONLINE PRIOR UPDATE ───────────────────────────────────────
        if self.do_online_update:
            self._online_update(query_emb, root)

        # ── COLLECT RESULTS ───────────────────────────────────────────
        results = self._collect_results(root)
        return results[:top_k]

    # ──────────────────────────────────────────────────────────────────────────
    # Internal MCTS phases
    # ──────────────────────────────────────────────────────────────────────────

    def _select(self, root: MCTSNode) -> MCTSNode:
        """
        Walk down tree from root, always picking the child with the
        highest PUCT score, until we reach an unexpanded or terminal node.
        """
        node = root
        while node.is_expanded and not node.is_terminal:
            if not node.children:
                break
            # Use PUCT if priors are set, otherwise UCB1
            node = max(node.children, key=lambda c: c.puct_score(self.c_puct))
        return node

    def _assign_priors(self, children: List[MCTSNode], query_emb: np.ndarray):
        """
        Score all children using the prior MLP and assign P(s,a) to each.
        This is the key replacement: was an LLM call, now is a 0.1ms MLP call.
        """
        child_dicts = [c.tree_node for c in children]
        scores = self.prior.score_children(query_emb, child_dicts)
        self.prior_call_count += 1

        for child, score in zip(children, scores):
            child.set_prior(score)

    def _simulate(
        self,
        node: MCTSNode,
        query: str,
        query_emb: np.ndarray,
    ) -> float:
        """
        Estimate the value of a node.
       
        For INTERNAL nodes (folders, files): use the prior MLP score.
        For LEAF nodes (functions, methods): call the LLM for a real score.
       
        This is the crucial efficiency trick: LLM only called at leaves.
        """
        if node.is_terminal:
            # Leaf node — call LLM for accurate relevance score
            return self._llm_score_leaf(node.tree_node, query)
        else:
            # Internal node — use prior MLP score (already computed if expanded)
            if node.prior is not None:
                return node.prior
            else:
                # Node not yet scored — score it now
                node_emb = node.tree_node.get('embedding')
                if node_emb is None:
                    return 0.3  # No embedding: return neutral score
                import torch
                q_tensor = torch.tensor(query_emb, dtype=torch.float32)
                n_tensor = torch.tensor(node_emb, dtype=torch.float32)
                with torch.no_grad():
                    score = self.prior(q_tensor, n_tensor)
                self.prior_call_count += 1
                return float(score)

    def _llm_score_leaf(self, node: Dict, query: str) -> float:
        """
        Call the LLM to score a leaf node's relevance to the query.
       
        This is the ONLY place the LLM is called.
        Uses a simple, deterministic prompt (temperature=0).
        """
        self.llm_call_count += 1

        title   = node.get('title', node.get('name', 'unknown'))
        summary = node.get('summary', '')
        path    = node.get('path', '')

        prompt = f"""Rate the relevance of this code function to the query.
Respond with ONLY a decimal number between 0.0 and 1.0. No explanation.

Query: {query}

Function: {title}
File: {path}
Description: {summary[:300]}

Relevance score (0.0 to 1.0):"""

        try:
            messages = [
                {"role": "system", "content": "You are a code relevance scorer. "
                 "Respond with only a number between 0.0 and 1.0."},
                {"role": "user", "content": prompt}
            ]
            # temperature=0 for deterministic scoring
            response = self.llm.chat(messages, temperature=0.0, max_tokens=10)
            if response:
                # Parse the number from response
                import re
                numbers = re.findall(r'\d+\.?\d*', response.strip())
                if numbers:
                    score = float(numbers[0])
                    return max(0.0, min(1.0, score))
        except Exception as e:
            if self.verbose:
                print(f"    LLM error for {title}: {e}")

        # Fallback: use embedding similarity
        node_emb = node.get('embedding')
        if node_emb is not None:
            from sentence_transformers.util import cos_sim
            import torch
            q_emb = self.embed_model.encode(query)
            sim = cos_sim(q_emb, np.array(node_emb))[0][0].item()
            return (sim + 1) / 2  # map [-1,1] to [0,1]

        return 0.0

    def _online_update(self, query_emb: np.ndarray, root: MCTSNode):
        """
        After search completes, use visit counts to update the prior.
        Collects all nodes from the search tree and uses their visit counts
        as soft training labels.
        """
        # Collect all nodes that were expanded during this search
        all_nodes = []
        all_counts = []

        stack = [root]
        while stack:
            node = stack.pop()
            if node.is_expanded:
                for child in node.children:
                    all_nodes.append(child.tree_node)
                    all_counts.append(child.visit_count)
                    stack.append(child)

        if not all_nodes:
            return

        self.prior.online_update(
            query_emb=query_emb,
            visited_nodes=all_nodes,
            visit_counts=all_counts,
            lr=self.online_lr,
        )

    def _collect_results(self, root: MCTSNode) -> List[Dict]:
        """
        Walk the full search tree and collect all terminal (leaf) nodes
        that were visited, ranked by average_value.
        """
        results = []
        stack = [root]

        while stack:
            node = stack.pop()

            if node.is_terminal and node.visit_count > 0:
                tree_node = node.tree_node
                results.append({
                    'node_id':         tree_node.get('node_id', ''),
                    'name':            tree_node.get('title', tree_node.get('name', '')),
                    'node_type':       tree_node.get('type', ''),
                    'summary':         tree_node.get('summary', ''),
                    'path':            tree_node.get('path', ''),
                    'similarity_score': node.average_value,
                    'metadata': {
                        'file_path':  tree_node.get('path', ''),
                        'start_line': tree_node.get('start_line'),
                        'end_line':   tree_node.get('end_line'),
                        'signature':  tree_node.get('signature', ''),
                        'docstring':  tree_node.get('summary', ''),
                    },
                    'mcts_stats': {
                        'visit_count':   node.visit_count,
                        'average_value': node.average_value,
                        'prior':         node.prior,
                    }
                })

            for child in node.children:
                stack.append(child)

        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results

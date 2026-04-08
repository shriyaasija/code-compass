"""
MCTS-Based Code Search
━━━━━━━━━━━━━━━━━━━━━━
Replaces the old greedy TreeBasedSearch with Monte Carlo Tree Search.

Instead of picking ONE branch (which fails 93% of the time), MCTS:
  1. Explores multiple branches using UCB1 (exploit + explore)
  2. Uses the LLM to score node relevance (cached)
  3. Simulates rollouts with keyword heuristic (no extra LLM calls)
  4. Returns top-k nodes by LLM relevance score
=
API is identical to TreeBasedSearch and PageIndexSemanticSearch:
  - load_repository_tree(repo_id, json_path)
  - search(repo_id, query, top_k=5)
"""

import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# MCTS Node
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MCTSNode:
    """A node in the MCTS search tree, wrapping a PageIndex tree node."""
    tree_node:    dict
    parent:       Optional["MCTSNode"] = None
    children:     list = field(default_factory=list)
    visits:       int = 0
    total_reward: float = 0.0
    reward:       float = 0.0
    expanded:     bool = False

    @property
    def name(self) -> str:
        return self.tree_node.get("name", self.tree_node.get("title", "?"))

    @property
    def node_type(self) -> str:
        return self.tree_node.get("node_type", self.tree_node.get("type", "unknown"))

    def ucb1(self, C: float = 1.4) -> float:
        """Upper Confidence Bound for tree exploration."""
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return float("inf")
        exploit = self.total_reward / self.visits
        explore = C * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore


# ═══════════════════════════════════════════════════════════════════════════════
# MCTS Search Engine
# ═══════════════════════════════════════════════════════════════════════════════

class MCTSSearch:
    """
    Monte Carlo Tree Search over the PageIndex code tree.

    Uses LLM to score relevance at each node, UCB1 to balance
    exploration vs exploitation, and keyword heuristic for rollouts.
    """

    def __init__(self, llm_client, n_simulations: int = 30,
                 ucb_c: float = 1.4, max_rollout_depth: int = 5,
                 verbose: bool = False):
        """
        Args:
            llm_client: OllamaLLM or LMStudioLLM instance
            n_simulations: Number of MCTS iterations per query
            ucb_c: UCB1 exploration constant (higher = more exploration)
            max_rollout_depth: Max depth for simulation rollouts
            verbose: Print detailed MCTS progress
        """
        self.llm = llm_client
        self.n_simulations = n_simulations
        self.ucb_c = ucb_c
        self.max_rollout_depth = max_rollout_depth
        self.verbose = verbose
        self.repositories = {}  # repo_id -> {tree, json_path}

        # Track LLM calls for benchmarking
        self.llm_call_count = 0

    def load_repository_tree(self, repo_id: str, json_tree_path: str):
        """Load PageIndex JSON tree. Same API as PageIndexSemanticSearch."""
        print(f"\n📂 Loading repository tree: {repo_id}")
        print(f"   From: {json_tree_path}")

        with open(json_tree_path, 'r') as f:
            tree = json.load(f)

        self.repositories[repo_id] = {
            'tree': tree,
            'json_path': json_tree_path
        }

        node_count = self._count_nodes(tree)
        print(f"✅ Loaded tree with ~{node_count} nodes")

    def search(self, repo_id: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        MCTS-based search. Same return format as PageIndexSemanticSearch.search().

        Returns list of dicts with: node_id, name, node_type, summary, path,
        similarity_score, metadata
        """
        if repo_id not in self.repositories:
            raise ValueError(f"Repository '{repo_id}' not loaded.")

        tree = self.repositories[repo_id]['tree']

        # Reset per-query state
        self.llm_call_count = 0
        score_cache = {}
        all_visited = []  # (reward, mcts_node) for every scored node

        # Create MCTS root
        root = MCTSNode(tree_node=tree)
        root.reward = self._score_node(root, query, score_cache)
        all_visited.append((root.reward, root))
        root.visits = 1
        root.total_reward = root.reward

        if self.verbose:
            print(f"\n🌲 MCTS Search: '{query[:60]}...'")
            print(f"   Simulations: {self.n_simulations}, UCB_C: {self.ucb_c}")

        # ── MCTS Main Loop ───────────────────────────────────────────────
        for sim in range(self.n_simulations):
            # SELECT: walk down tree using UCB1
            leaf = self._select(root)

            # EXPAND: if leaf has children, expand them (score with LLM)
            if leaf.tree_node.get("children"):
                self._expand(leaf, query, score_cache, all_visited)
                # Pick best child to simulate from
                if leaf.children:
                    leaf = max(leaf.children, key=lambda c: c.reward)

            # SIMULATE: random rollout with keyword heuristic
            reward = self._simulate(leaf, query, score_cache)

            # BACKPROPAGATE: update visit counts and rewards up the tree
            self._backprop(leaf, reward)

            if self.verbose and (sim + 1) % 10 == 0:
                print(f"   [sim {sim+1}/{self.n_simulations}] "
                      f"scored {len(score_cache)} nodes, "
                      f"LLM calls: {self.llm_call_count}")

        # ── Collect top-k results ────────────────────────────────────────
        seen = set()
        ranked = []
        for score, mnode in sorted(all_visited, key=lambda x: -x[0]):
            nid = self._node_key(mnode.tree_node)
            if nid not in seen:
                seen.add(nid)
                ranked.append((score, mnode))
            if len(ranked) >= top_k:
                break

        if self.verbose:
            print(f"   ✅ Found {len(ranked)} results, {self.llm_call_count} LLM calls")

        # Format results to match PageIndexSemanticSearch output
        results = []
        for score, mnode in ranked:
            node = mnode.tree_node
            results.append({
                'node_id': node.get('node_id', ''),
                'name': mnode.name,
                'node_type': mnode.node_type,
                'summary': node.get('summary', node.get('repository_summary', '')),
                'path': node.get('path', node.get('file_path', '')),
                'similarity_score': score,
                'metadata': {
                    'file_path': node.get('file_path', node.get('path', '')),
                    'signature': node.get('signature', ''),
                    'start_line': node.get('start_line', 0),
                    'end_line': node.get('end_line', 0),
                    'docstring': node.get('docstring', node.get('summary', '')),
                },
            })

        return results

    # ── LLM Relevance Scoring ────────────────────────────────────────────

    def _score_node(self, mnode: MCTSNode, query: str, cache: dict) -> float:
        """Ask the LLM to score how relevant this node is to the query."""
        nid = self._node_key(mnode.tree_node)
        if nid in cache:
            return cache[nid]

        node_text = self._node_summary_text(mnode.tree_node)

        prompt = (
            f"Query: {query}\n\n"
            f"Node summary: {node_text[:800]}\n\n"
            "Rate how relevant this node is to the query. "
            "Reply with ONLY a single decimal number between 0.0 and 1.0. Nothing else. /no_think"
        )

        try:
            self.llm_call_count += 1
            messages = [
                {"role": "system", "content": "You are a relevance scorer. Output only a number between 0.0 and 1.0. Do not explain."},
                {"role": "user", "content": prompt},
            ]
            raw = self.llm.chat(
                messages=messages,
                max_tokens=500,  # reasoning models need room for thinking tokens
                temperature=0.3,
                retry_on_empty=False,  # don't retry — just use fallback score
            )

            match = re.search(r"\d+\.?\d*", raw)
            if match:
                score = max(0.0, min(1.0, float(match.group())))
            else:
                score = 0.0
        except Exception as e:
            if self.verbose:
                print(f"     ⚠️ LLM scoring failed: {e}")
            score = 0.0

        cache[nid] = score
        return score

    # ── MCTS Phases ──────────────────────────────────────────────────────

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Walk down tree using UCB1 until we reach an unexpanded node."""
        while node.expanded and node.children:
            node = max(node.children, key=lambda c: c.ucb1(self.ucb_c))
        return node

    def _expand(self, node: MCTSNode, query: str, cache: dict, all_visited: list):
        """Expand node: create MCTS children and score each with LLM."""
        if node.expanded:
            return
        for child_data in node.tree_node.get("children", []):
            child = MCTSNode(tree_node=child_data, parent=node)
            child.reward = self._score_node(child, query, cache)
            all_visited.append((child.reward, child))
            node.children.append(child)
        node.expanded = True

    def _simulate(self, node: MCTSNode, query: str, cache: dict) -> float:
        """
        Lightweight rollout: walk a random path to a leaf.
        Uses keyword-overlap heuristic (no extra LLM calls) for unvisited nodes.
        """
        current = node.tree_node
        best = node.reward
        depth = 0

        while current.get("children") and depth < self.max_rollout_depth:
            current = random.choice(current["children"])
            cid = self._node_key(current)

            if cid in cache:
                s = cache[cid]
            else:
                # Keyword overlap heuristic (free — no LLM call)
                qwords = set(query.lower().split())
                nwords = set(self._node_summary_text(current).lower().split())
                s = len(qwords & nwords) / (len(qwords) + 1)
                s = min(s, 1.0)

            best = max(best, s)
            depth += 1

        return best

    def _backprop(self, node: MCTSNode, reward: float):
        """Propagate reward up to root."""
        while node:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    # ── Helpers ──────────────────────────────────────────────────────────

    def _node_key(self, node: dict) -> tuple:
        """Stable cache key for a tree node."""
        return (
            node.get("name", ""),
            node.get("type", node.get("node_type", "")),
            node.get("title", ""),
            node.get("node_id", ""),
        )

    def _node_summary_text(self, node: dict) -> str:
        """Extract best text representation for scoring."""
        parts = []
        if node.get("summary"):
            parts.append(node["summary"])
        if node.get("repository_summary"):
            parts.append(node["repository_summary"])
        if node.get("name"):
            parts.append(f"name: {node['name']}")
        if node.get("type") or node.get("node_type"):
            parts.append(f"type: {node.get('type', node.get('node_type', ''))}")
        if node.get("docstring"):
            parts.append(node["docstring"][:400])
        return "\n".join(parts)

    def _count_nodes(self, node: dict, count: int = 0) -> int:
        """Recursively count all nodes in tree."""
        count += 1
        for child in node.get("children", node.get("nodes", [])):
            count = self._count_nodes(child, count)
        return count

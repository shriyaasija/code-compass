import numpy as np 
from typing import Dict, List, Optional, Tuple, Any

from research.rl_index.tree_mutations import (
    get_children, tree_depth, avg_branching_factor,
    count_leaves, collect_all_nodes, get_node_embedding,
    cosine_similarity,
)

def gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient of a distribution."""
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0

    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)

    return float((2 * np.sum(index * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1) / n)

def compute_avg_sibling_similarity(root: Dict) -> float:
    """Compute average cosine similarity between sibling nodes."""
    similarities = []
    stack = [root]

    while stack:
        node = stack.pop()
        children = get_children(node)
        if len(children) >= 2:
            # Compare all pairs of siblings
            for i in range(len(children)):
                emb_i = get_node_embedding(children[i])
                if emb_i is None:
                    continue
                for j in range(i + 1, len(children)):
                    emb_j = get_node_embedding(children[j])
                    if emb_j is None:
                        continue
                    similarities.append(cosine_similarity(emb_i, emb_j))
        for child in children:
            stack.append(child)

    if not similarities:
        return 0.5  # Default when no embeddings
    return float(np.mean(similarities))

def compute_leaf_depths(root: Dict) -> Dict[str, int]:
    """Map leaf node titles to their depths."""
    depths = {}

    def _walk(node, depth):
        children = get_children(node)
        if not children:
            title = node.get('title', node.get('name', str(id(node))))
            depths[title] = depth
        else:
            for child in children:
                _walk(child, depth + 1)

    _walk(root, 0)
    return depths

class TreeStateExtractor:
    """
    Extracts the 8-dim observation vector for the RL environment.

    Tracks initial tree stats for normalization and maintains a buffer
    of recent query results for computing MRR and hit statistics.
    """

    STATE_DIM = 8  # Observation space dimension

    def __init__(self, initial_tree: Dict):
        """
        Initialize with the original (un-mutated) tree for normalization.

        Args:
            initial_tree: The original PageIndex tree before any mutations.
        """
        self.initial_depth = max(tree_depth(initial_tree), 1)
        self.initial_leaf_count = max(count_leaves(initial_tree), 1)

        # Query buffer: list of {query, ground_truth, results, retrieved_depths}
        self.query_buffer: List[Dict] = []
        self.buffer_size = 100  # Rolling window

        # Leaf hit counts for Gini coefficient
        self.leaf_hits: Dict[str, int] = {}

    def extract_state(self, tree: Dict) -> np.ndarray:
        """
        Extract 8-dimensional state vector from current tree + query buffer.

        Returns:
            np.ndarray of shape (8,) with values in [0, 1].
        """
        # [0] Normalized depth
        current_depth = tree_depth(tree)
        norm_depth = min(current_depth / self.initial_depth, 2.0) / 2.0

        # [1] Average branching factor (clamped to [0, 1] via /20)
        avg_bf, bf_var = avg_branching_factor(tree)
        norm_avg_bf = min(avg_bf / 20.0, 1.0)

        # [2] Branching variance (clamped)
        norm_bf_var = min(bf_var / 100.0, 1.0)

        # [3] Normalized leaf count
        current_leaves = count_leaves(tree)
        norm_leaves = min(current_leaves / self.initial_leaf_count, 2.0) / 2.0

        # [4] Hit skew (Gini coefficient of leaf hits)
        if self.leaf_hits:
            hit_values = np.array(list(self.leaf_hits.values()), dtype=np.float64)
            hit_skew = gini_coefficient(hit_values)
        else:
            hit_skew = 0.0

        # [5] Average sibling similarity
        avg_sim = compute_avg_sibling_similarity(tree)
        # Cosine sim is in [-1, 1], shift to [0, 1]
        norm_sim = (avg_sim + 1.0) / 2.0

        # [6] Average retrieval depth
        if self.query_buffer:
            all_depths = []
            for q in self.query_buffer:
                all_depths.extend(q.get('retrieved_depths', []))
            if all_depths:
                avg_ret_depth = np.mean(all_depths)
                norm_ret_depth = min(avg_ret_depth / self.initial_depth, 2.0) / 2.0
            else:
                norm_ret_depth = 0.5
        else:
            norm_ret_depth = 0.5

        # [7] Buffer MRR
        buffer_mrr = self._compute_buffer_mrr()

        state = np.array([
            norm_depth,
            norm_avg_bf,
            norm_bf_var,
            norm_leaves,
            hit_skew,
            norm_sim,
            norm_ret_depth,
            buffer_mrr,
        ], dtype=np.float32)

        # Clamp to [0, 1]
        state = np.clip(state, 0.0, 1.0)
        return state

    def record_query(self, query: str, ground_truth: str,
                     ranked_results: List[str], tree: Dict):
        """
        Record a query result into the rolling buffer.

        Args:
            query: The query string.
            ground_truth: The correct function/node name.
            ranked_results: List of result names in ranked order.
            tree: Current tree (for computing retrieval depths).
        """
        # Compute depths of retrieved results
        leaf_depths = compute_leaf_depths(tree)
        retrieved_depths = [leaf_depths.get(name, 0) for name in ranked_results[:10]]

        # Update buffer
        self.query_buffer.append({
            'query': query,
            'ground_truth': ground_truth,
            'results': ranked_results,
            'retrieved_depths': retrieved_depths,
        })

        # Trim to buffer size
        if len(self.query_buffer) > self.buffer_size:
            self.query_buffer = self.query_buffer[-self.buffer_size:]

        # Update hit counts
        for name in ranked_results[:10]:
            self.leaf_hits[name] = self.leaf_hits.get(name, 0) + 1

        # Also count ground truth
        self.leaf_hits[ground_truth] = self.leaf_hits.get(ground_truth, 0) + 1

    def _compute_buffer_mrr(self) -> float:
        """Compute MRR across the query buffer."""
        if not self.query_buffer:
            return 0.0

        reciprocal_ranks = []
        for q in self.query_buffer:
            gt = q['ground_truth']
            results = q['results']
            try:
                rank = results.index(gt) + 1
                reciprocal_ranks.append(1.0 / rank)
            except ValueError:
                reciprocal_ranks.append(0.0)

        return float(np.mean(reciprocal_ranks))

    def reset(self):
        """Clear query buffer and hit counts (for environment reset)."""
        self.query_buffer.clear()
        self.leaf_hits.clear()
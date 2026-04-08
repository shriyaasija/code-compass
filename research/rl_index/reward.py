"""
References:
  - Reward shaping: Ng et al., "Policy invariance under reward transformations" (ICML 1999)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from research.rl_index.tree_mutations import tree_depth, count_leaves


class RewardComputer:
    """
    Computes reward for a single RL step (mutation).

    Tracks pre/post mutation metrics and computes the shaped reward.
    """

    def __init__(
        self,
        lambda_mrr: float = 1.0,
        lambda_depth: float = 0.1,
        lambda_validity: float = 1.0,
        initial_depth: int = 1,
    ):

        self.lambda_mrr = lambda_mrr
        self.lambda_depth = lambda_depth
        self.lambda_validity = lambda_validity
        self.initial_depth = max(initial_depth, 1)

        # Tracked metrics
        self.prev_mrr: float = 0.0
        self.prev_depth: int = initial_depth

    def compute(
        self,
        tree: Dict,
        current_mrr: float,
        mutation_valid: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute reward after a mutation step."""
        current_depth = tree_depth(tree)

        # Component 1: MRR improvement
        delta_mrr = current_mrr - self.prev_mrr
        mrr_reward = self.lambda_mrr * delta_mrr

        # Component 2: Depth penalty
        delta_depth = (current_depth - self.prev_depth) / self.initial_depth
        depth_penalty = self.lambda_depth * delta_depth

        # Component 3: Validity bonus
        if mutation_valid:
            validity_bonus = self.lambda_validity * 0.01  # Small positive
        else:
            validity_bonus = self.lambda_validity * (-0.1)  # Penalty

        # Total reward
        reward = mrr_reward - depth_penalty + validity_bonus

        # Update tracked state
        self.prev_mrr = current_mrr
        self.prev_depth = current_depth

        info = {
            'delta_mrr': delta_mrr,
            'mrr_reward': mrr_reward,
            'delta_depth': delta_depth,
            'depth_penalty': depth_penalty,
            'validity_bonus': validity_bonus,
            'total_reward': reward,
            'current_mrr': current_mrr,
            'current_depth': current_depth,
        }

        return reward, info
    
    def reset(self, tree: Dict, initial_mrr: float = 0.0):
        """Reset tracked metrics for a new episode."""
        self.prev_mrr = initial_mrr
        self.prev_depth = tree_depth(tree)


class ProxyMRREstimator:
    def __init__(self, embedding_model=None):
        """
        Args:
            embedding_model: SentenceTransformer model for encoding queries.
                           If None, falls back to summary-based text matching.
        """
        self.model = embedding_model

    def estimate_mrr(
        self,
        tree: Dict,
        query_buffer: List[Dict],
    ) -> float:
        """
        Estimate MRR on a query buffer using embedding similarity.

        Args:
            tree: Current tree.
            query_buffer: List of {query, ground_truth} dicts.

        Returns:
            Estimated MRR value in [0, 1].
        """
        if not query_buffer:
            return 0.0

        # Collect all leaf nodes with their embeddings
        leaves = self._collect_leaves(tree)
        if not leaves:
            return 0.0

        reciprocal_ranks = []
        for q_info in query_buffer:
            ground_truth = q_info['ground_truth']

            # Rank leaves by embedding similarity to query
            if self.model is not None:
                query_emb = self.model.encode(q_info['query'])
                ranked = self._rank_by_embedding(leaves, query_emb)
            else:
                # Fallback: rank by text overlap with query
                ranked = self._rank_by_text(leaves, q_info['query'])

            # Find rank of ground truth
            try:
                rank = [leaf['title'] for leaf in ranked].index(ground_truth) + 1
                reciprocal_ranks.append(1.0 / rank)
            except ValueError:
                reciprocal_ranks.append(0.0)

        return float(np.mean(reciprocal_ranks))

    def _collect_leaves(self, root: Dict) -> List[Dict]:
        """Collect all leaf nodes from tree."""
        from research.rl_index.tree_mutations import get_children
        leaves = []
        stack = [root]
        while stack:
            node = stack.pop()
            children = get_children(node)
            if not children:
                leaves.append(node)
            else:
                for child in children:
                    stack.append(child)
        return leaves

    def _rank_by_embedding(self, leaves: List[Dict], query_emb: np.ndarray) -> List[Dict]:
        """Rank leaves by cosine similarity to query embedding."""
        from research.rl_index.tree_mutations import get_node_embedding, cosine_similarity
        scored = []
        for leaf in leaves:
            emb = get_node_embedding(leaf)
            if emb is not None:
                sim = cosine_similarity(query_emb, emb)
            else:
                sim = 0.0
            scored.append((sim, leaf))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [leaf for _, leaf in scored]

    def _rank_by_text(self, leaves: List[Dict], query: str) -> List[Dict]:
        """Fallback ranking by word overlap."""
        query_words = set(query.lower().split())
        scored = []
        for leaf in leaves:
            summary = leaf.get('summary', '') + ' ' + leaf.get('title', '')
            leaf_words = set(summary.lower().split())
            overlap = len(query_words & leaf_words)
            scored.append((overlap, leaf))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [leaf for _, leaf in scored]    
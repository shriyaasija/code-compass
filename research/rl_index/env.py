"""
References:
  - Huang et al., "A Closer Look at Invalid Action Masking in RL" (2022)
"""

import copy
import gymnasium as gym
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from research.rl_index.tree_mutations import (
    merge, split, reparent,
    can_merge, can_split, can_reparent,
    get_children, count_leaves, tree_depth,
    deep_copy_tree, validate_tree,
    collect_sibling_pairs, collect_internal_nodes,
    collect_all_nodes,
)
from research.rl_index.tree_state import TreeStateExtractor
from research.rl_index.reward import RewardComputer, ProxyMRREstimator


# Action indices
ACTION_MERGE = 0
ACTION_SPLIT = 1
ACTION_REPARENT = 2
ACTION_NOOP = 3
NUM_ACTIONS = 4

ACTION_NAMES = {
    ACTION_MERGE: "Merge",
    ACTION_SPLIT: "Split",
    ACTION_REPARENT: "Reparent",
    ACTION_NOOP: "NoOp",
}


class TreeIndexEnv(gym.Env):
    """
    RL environment for adaptive tree index optimization.

    At each step, the agent selects a mutation type. The environment
    automatically selects the best target nodes for that mutation
    (using heuristics like highest co-query correlation or lowest
    sibling similarity).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        tree: Dict,
        query_buffer: List[Dict],
        max_steps: int = 50,
        lambda_mrr: float = 1.0,
        lambda_depth: float = 0.1,
        merge_sim_threshold: float = 0.3,
        split_min_children: int = 4,
        split_k: int = 2,
        embedding_model=None,
        seed: int = 42,
    ):
        """
        Args:
            tree: The initial PageIndex tree (will be deep-copied).
            query_buffer: List of {query, ground_truth} for MRR evaluation.
            max_steps: Maximum mutations per episode.
            lambda_mrr: Reward weight for MRR improvement.
            lambda_depth: Reward weight for depth penalty.
            merge_sim_threshold: Min cosine similarity for merge.
            split_min_children: Min children for split eligibility.
            split_k: Number of clusters for split.
            embedding_model: SentenceTransformer for proxy MRR.
            seed: Random seed.
        """
        super().__init__()

        # Store original tree for reset
        self._original_tree = deep_copy_tree(tree)
        self._query_buffer = query_buffer
        self._max_steps = max_steps
        self._merge_sim_threshold = merge_sim_threshold
        self._split_min_children = split_min_children
        self._split_k = split_k
        self._seed = seed

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(TreeStateExtractor.STATE_DIM,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        # Components (initialized in reset)
        self._tree: Optional[Dict] = None
        self._state_extractor: Optional[TreeStateExtractor] = None
        self._reward_computer: Optional[RewardComputer] = None
        self._proxy_mrr = ProxyMRREstimator(embedding_model)
        self._step_count = 0
        self._initial_leaf_count = count_leaves(tree)
        self._mutation_log: List[Dict] = []

        # Pre-compute action mask cache
        self._cached_merge_targets: List[Tuple] = []
        self._cached_split_targets: List[Dict] = []
        self._cached_reparent_targets: List[Tuple] = []

    def reset(self, seed=None, options=None):
        """Reset environment to initial tree state."""
        super().reset(seed=seed)

        self._tree = deep_copy_tree(self._original_tree)
        self._state_extractor = TreeStateExtractor(self._tree)
        self._reward_computer = RewardComputer(
            initial_depth=tree_depth(self._tree)
        )

        # Seed the query buffer into state extractor
        initial_mrr = self._proxy_mrr.estimate_mrr(self._tree, self._query_buffer)
        self._reward_computer.reset(self._tree, initial_mrr=initial_mrr)

        # Record initial queries into state extractor
        for q in self._query_buffer[:20]:  # Use first 20 for initialization
            self._state_extractor.record_query(
                q['query'], q['ground_truth'], [], self._tree
            )

        self._step_count = 0
        self._mutation_log = []
        self._update_action_targets()

        obs = self._state_extractor.extract_state(self._tree)
        return obs, {}

    def step(self, action: int):
        """
        Execute one mutation step.

        Args:
            action: 0=Merge, 1=Split, 2=Reparent, 3=NoOp

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        action = int(action)  # model.predict() returns numpy array
        self._step_count += 1
        mutation_valid = False
        action_name = ACTION_NAMES.get(action, "Unknown")
        mutation_msg = ""

        if action == ACTION_MERGE:
            mutation_valid, mutation_msg = self._execute_merge()
        elif action == ACTION_SPLIT:
            mutation_valid, mutation_msg = self._execute_split()
        elif action == ACTION_REPARENT:
            mutation_valid, mutation_msg = self._execute_reparent()
        elif action == ACTION_NOOP:
            mutation_valid = True
            mutation_msg = "NoOp"

        # Compute proxy MRR on current tree
        current_mrr = self._proxy_mrr.estimate_mrr(self._tree, self._query_buffer)

        # Compute reward
        reward, reward_info = self._reward_computer.compute(
            self._tree, current_mrr, mutation_valid
        )

        # Update action targets for next step
        self._update_action_targets()

        # Check termination
        valid_tree, valid_msg = validate_tree(self._tree, self._initial_leaf_count)
        terminated = not valid_tree
        truncated = self._step_count >= self._max_steps

        # Build observation
        obs = self._state_extractor.extract_state(self._tree)

        # Info dict
        info = {
            'action': action_name,
            'mutation_valid': mutation_valid,
            'mutation_msg': mutation_msg,
            'step': self._step_count,
            **reward_info,
        }
        self._mutation_log.append(info)

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """
        Return boolean mask of valid actions.

        Required by sb3-contrib's MaskablePPO.
        True = action is allowed, False = action is masked.
        """
        mask = np.array([
            len(self._cached_merge_targets) > 0,    # Merge
            len(self._cached_split_targets) > 0,     # Split
            len(self._cached_reparent_targets) > 0,  # Reparent
            True,                                     # NoOp always valid
        ], dtype=bool)
        return mask

    # Alias for compatibility
    def get_action_mask(self) -> np.ndarray:
        return self.action_masks()

    def _execute_merge(self) -> Tuple[bool, str]:
        """Execute the best available merge."""
        if not self._cached_merge_targets:
            return False, "No valid merge targets"

        # Pick the merge pair with highest similarity
        parent, node_a, node_b = self._cached_merge_targets[0]
        result = merge(self._tree, node_a, node_b, self._merge_sim_threshold)
        return result.success, result.message

    def _execute_split(self) -> Tuple[bool, str]:
        """Execute the best available split."""
        if not self._cached_split_targets:
            return False, "No valid split targets"

        # Pick the node with most children (most bloated)
        target = self._cached_split_targets[0]
        result = split(self._tree, target, k=self._split_k,
                       min_children=self._split_min_children)
        return result.success, result.message

    def _execute_reparent(self) -> Tuple[bool, str]:
        """Execute the best available reparent."""
        if not self._cached_reparent_targets:
            return False, "No valid reparent targets"

        node, new_parent = self._cached_reparent_targets[0]
        result = reparent(self._tree, node, new_parent)
        return result.success, result.message

    def _update_action_targets(self):
        """Pre-compute valid targets for each action type."""
        # Merge targets: sibling pairs sorted by similarity (highest first)
        self._cached_merge_targets = []
        try:
            pairs = collect_sibling_pairs(self._tree)
            scored_pairs = []
            for parent, a, b in pairs:
                valid, _ = can_merge(self._tree, a, b, self._merge_sim_threshold)
                if valid:
                    from research.rl_index.tree_mutations import (
                        get_node_embedding, cosine_similarity
                    )
                    emb_a = get_node_embedding(a)
                    emb_b = get_node_embedding(b)
                    sim = 0.5
                    if emb_a is not None and emb_b is not None:
                        sim = cosine_similarity(emb_a, emb_b)
                    scored_pairs.append((sim, parent, a, b))
            scored_pairs.sort(key=lambda x: x[0], reverse=True)
            self._cached_merge_targets = [(p, a, b) for _, p, a, b in scored_pairs[:5]]
        except Exception:
            self._cached_merge_targets = []

        # Split targets: internal nodes sorted by child count (most children first)
        self._cached_split_targets = []
        try:
            internals = collect_internal_nodes(self._tree)
            splittable = []
            for node in internals:
                valid, _ = can_split(node, self._split_min_children)
                if valid:
                    splittable.append(node)
            splittable.sort(key=lambda n: len(get_children(n)), reverse=True)
            self._cached_split_targets = splittable[:5]
        except Exception:
            self._cached_split_targets = []

        # Reparent targets: find nodes that could benefit from reparenting
        self._cached_reparent_targets = []
        try:
            all_nodes = collect_all_nodes(self._tree)
            internals = [n for n in all_nodes if get_children(n)]
            # Find leaf nodes and try to reparent them to semantically similar subtrees
            leaves = [n for n in all_nodes if not get_children(n)]
            for leaf in leaves[:10]:  # Limit search
                for target in internals[:10]:
                    valid, _ = can_reparent(self._tree, leaf, target)
                    if valid:
                        self._cached_reparent_targets.append((leaf, target))
                        break  # One reparent option per leaf is enough
            self._cached_reparent_targets = self._cached_reparent_targets[:5]
        except Exception:
            self._cached_reparent_targets = []

    def get_mutation_log(self) -> List[Dict]:
        """Get the log of all mutations performed this episode."""
        return self._mutation_log

    @property
    def current_tree(self) -> Dict:
        """Get the current (possibly mutated) tree."""
        return self._tree
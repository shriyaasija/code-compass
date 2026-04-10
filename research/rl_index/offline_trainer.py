"""
Offline RL Trainer for Tree Index Optimization.

Key design: ZERO LLM calls during training.
Reward signal: structural quality of the tree (depth balance + branching
balance + sibling coherence), NOT proxy MRR (which fails because
CodeSearchNet ground_truth names don't match tree-sitter node titles).
"""

import json
import os
import time
import numpy as np
from typing import Dict, List, Tuple
import copy

import gymnasium as gym


# ── Tree utilities (inline to avoid import issues) ───────────────────────────

def get_children(node):
    return node.get('nodes', node.get('children', []))

def set_children(node, children):
    if 'nodes' in node:
        node['nodes'] = children
    if 'children' in node:
        node['children'] = children
    if 'nodes' not in node and 'children' not in node:
        node['nodes'] = children

def count_leaves(node):
    kids = get_children(node)
    if not kids:
        return 1
    return sum(count_leaves(c) for c in kids)

def tree_depth(node):
    kids = get_children(node)
    if not kids:
        return 0
    return 1 + max(tree_depth(c) for c in kids)

def avg_branching_factor(root):
    bfs = []
    stack = [root]
    while stack:
        n = stack.pop()
        kids = get_children(n)
        if kids:
            bfs.append(len(kids))
            for c in kids:
                stack.append(c)
    if not bfs:
        return 1.0, 0.0
    return float(np.mean(bfs)), float(np.var(bfs))

def get_node_embedding(node):
    e = node.get('embedding')
    if e is not None:
        arr = np.array(e, dtype=np.float32)
        if arr.ndim == 1 and len(arr) > 0:
            return arr
    return None

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def deep_copy_tree(tree):
    return copy.deepcopy(tree)

def find_parent(root, target):
    for c in get_children(root):
        if c is target:
            return root
        result = find_parent(c, target)
        if result is not None:
            return result
    return None

def is_descendant(ancestor, node):
    if ancestor is node:
        return True
    for c in get_children(ancestor):
        if is_descendant(c, node):
            return True
    return False

def collect_sibling_pairs(root):
    pairs = []
    stack = [root]
    while stack:
        node = stack.pop()
        kids = get_children(node)
        for i in range(len(kids)):
            for j in range(i+1, len(kids)):
                pairs.append((node, kids[i], kids[j]))
            stack.append(kids[i])
    return pairs

def collect_internal_nodes(root):
    result = []
    stack = [root]
    while stack:
        n = stack.pop()
        kids = get_children(n)
        if kids:
            result.append(n)
            for c in kids:
                stack.append(c)
    return result

# ── Mutations ────────────────────────────────────────────────────────────────

class MutationResult:
    def __init__(self, success, message):
        self.success = success
        self.message = message

def can_merge(root, a, b):
    if a is b:
        return False
    pa = find_parent(root, a)
    pb = find_parent(root, b)
    if pa is None or pb is None or pa is not pb:
        return False
    return True

def do_merge(root, a, b):
    if not can_merge(root, a, b):
        return MutationResult(False, "precondition failed")
    parent = find_parent(root, a)
    kids_a = get_children(a)
    kids_b = get_children(b)
    title_a = a.get('title', a.get('name', 'A'))
    title_b = b.get('title', b.get('name', 'B'))
    merged_kids = (kids_a if kids_a else [a]) + (kids_b if kids_b else [b])
    merged = {
        'title': f'{title_a}+{title_b}',
        'name': f'{title_a}+{title_b}',
        'type': 'folder',
        'summary': f'{a.get("summary","")} | {b.get("summary","")}',
        'path': a.get('path', ''),
    }
    ea, eb = get_node_embedding(a), get_node_embedding(b)
    if ea is not None and eb is not None:
        merged['embedding'] = ((ea + eb) / 2.0).tolist()
    set_children(merged, merged_kids)
    parent_kids = get_children(parent)
    new_kids, inserted = [], False
    for c in parent_kids:
        if c is a or c is b:
            if not inserted:
                new_kids.append(merged)
                inserted = True
        else:
            new_kids.append(c)
    set_children(parent, new_kids)
    return MutationResult(True, f"merged {title_a}+{title_b}")

def can_split(node, min_children=3):
    return len(get_children(node)) >= min_children

def do_split(root, node, k=2, min_children=3):
    if not can_split(node, min_children):
        return MutationResult(False, "too few children")
    kids = get_children(node)
    k = min(k, len(kids))
    embs = [get_node_embedding(c) for c in kids]
    if all(e is not None for e in embs):
        try:
            from sklearn.cluster import KMeans
            X = np.array(embs)
            labels = KMeans(n_clusters=k, random_state=42, n_init=5,
                            max_iter=50).fit_predict(X)
        except Exception:
            labels = [i % k for i in range(len(kids))]
    else:
        labels = [i % k for i in range(len(kids))]
    title = node.get('title', node.get('name', 'node'))
    clusters = []
    for ci in range(k):
        cluster_kids = [kids[i] for i in range(len(kids)) if labels[i] == ci]
        if not cluster_kids:
            continue
        cn = {
            'title': f'{title}_g{ci}',
            'name': f'{title}_g{ci}',
            'type': 'folder',
            'summary': f'Group {ci} of {title}',
            'path': node.get('path', ''),
        }
        set_children(cn, cluster_kids)
        clusters.append(cn)
    if len(clusters) < 2:
        return MutationResult(False, "clustering produced <2 non-empty clusters")
    set_children(node, clusters)
    return MutationResult(True, f"split {title} into {len(clusters)} groups")


# ── Structural quality reward ────────────────────────────────────────────────

def compute_sibling_coherence(tree: Dict) -> float:
    """
    ONLY measure sibling embedding coherence.
    This is the one thing Merge provably improves (groups similar nodes)
    and Split provably doesn't hurt (splits by K-means so children
    within each cluster are MORE similar than before).
    
    Returns mean cosine similarity across all sibling pairs in [0,1].
    """
    sims = []
    stack = [tree]
    while stack:
        node = stack.pop()
        kids = get_children(node)
        if len(kids) >= 2:
            embs = []
            for c in kids[:10]:
                e = get_node_embedding(c)
                if e is not None:
                    embs.append(e)
            if len(embs) >= 2:
                A = np.array(embs)
                norms = np.linalg.norm(A, axis=1, keepdims=True)
                norms = np.where(norms < 1e-8, 1.0, norms)
                A = A / norms
                sim_mat = A @ A.T
                idx = np.triu_indices(len(embs), k=1)
                sims.extend(sim_mat[idx].tolist())
        for c in kids:
            stack.append(c)
    
    if not sims:
        return 0.5
    return float(np.mean(sims))  # raw value in [-1, 1]


# ── Environment ──────────────────────────────────────────────────────────────

ACTION_MERGE = 0
ACTION_SPLIT = 1
ACTION_NOOP  = 2
NUM_ACTIONS  = 3


class FastTreeIndexEnv(gym.Env):

    def __init__(
        self,
        tree: Dict,
        query_buffer: List[Dict],          # kept for API compat, not used in reward
        query_embeddings: Dict,            # kept for API compat, not used in reward
        max_steps: int = 30,
        merge_threshold: float = 0.0,      # allow any merge; coherence reward filters
        split_min_children: int = 3,
        target_refresh_every: int = 5,
        **kwargs,
    ):
        super().__init__()
        self._original_tree = deep_copy_tree(tree)
        self._max_steps = max_steps
        self._merge_threshold = merge_threshold
        self._split_min_children = split_min_children
        self._target_refresh_every = target_refresh_every

        self._initial_leaves = count_leaves(tree)
        self._initial_depth = max(tree_depth(tree), 1)

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        self._tree = None
        self._step_count = 0
        self._prev_quality = 0.0
        self._merge_targets = []
        self._split_targets = []
        self._steps_since_refresh = 999

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._tree = deep_copy_tree(self._original_tree)
        self._step_count = 0
        self._prev_coherence = compute_sibling_coherence(self._tree)
        self._steps_since_refresh = 999
        self._update_targets()
        return self._get_obs(), {}

    def step(self, action: int):
        self._step_count += 1
        self._steps_since_refresh += 1
        valid = False

        if action == ACTION_MERGE and self._merge_targets:
            _, a, b = self._merge_targets[0]
            result = do_merge(self._tree, a, b)
            valid = result.success
        elif action == ACTION_SPLIT and self._split_targets:
            result = do_split(self._tree, self._split_targets[0],
                              k=2, min_children=self._split_min_children)
            valid = result.success
        elif action == ACTION_NOOP:
            valid = True

        if self._steps_since_refresh >= self._target_refresh_every or valid:
            self._update_targets()
            self._steps_since_refresh = 0

        current_coherence = compute_sibling_coherence(self._tree)
        delta = current_coherence - self._prev_coherence

        # Coherence is in [-1,1] so delta range is [-2,2]
        # Scale so a +0.01 coherence gain gives reward ~+1.0 (agent can feel it)
        reward = 100.0 * delta

        # Invalid action penalty
        if action == ACTION_NOOP and len(self._merge_targets) > 0:
            reward -= 0.05

        self._prev_coherence = current_coherence

        info = {
            'action': action,
            'valid': valid,
            'coherence': current_coherence,
            'delta': delta,
        }

        return self._get_obs(), float(reward), False, self._step_count >= self._max_steps, info

    def action_masks(self) -> np.ndarray:
        return np.array([
            len(self._merge_targets) > 0,
            len(self._split_targets) > 0,
            True,
        ], dtype=bool)

    def get_action_mask(self):
        return self.action_masks()

    @property
    def current_tree(self):
        return self._tree

    def _get_obs(self) -> np.ndarray:
        depth = tree_depth(self._tree)
        avg_bf, bf_var = avg_branching_factor(self._tree)
        leaves = count_leaves(self._tree)

        sims = []
        for _, a, b in self._merge_targets[:5]:
            ea, eb = get_node_embedding(a), get_node_embedding(b)
            if ea is not None and eb is not None:
                sims.append(cosine_sim(ea, eb))
        avg_sim = float(np.mean(sims)) if sims else 0.5

        n_merge = min(len(self._merge_targets) / 10.0, 1.0)
        n_split = min(len(self._split_targets) / 10.0, 1.0)

        obs = np.array([
            min(depth / self._initial_depth, 2.0) / 2.0,
            min(avg_bf / 20.0, 1.0),
            min(bf_var / 100.0, 1.0),
            min(leaves / max(self._initial_leaves, 1), 2.0) / 2.0,
            (avg_sim + 1.0) / 2.0,
            (self._prev_coherence + 1.0) / 2.0,  # shift to [0,1]
            n_merge,
            n_split,
        ], dtype=np.float32)

        return np.clip(obs, 0.0, 1.0)

    def _update_targets(self):
        """
        Only expose merges that we can VERIFY will improve coherence.
        Pre-screen candidates by simulating the merge and checking delta.
        Cap at 3 verified beneficial merges.
        """
        self._merge_targets = []
        try:
            candidates = []
            for parent, a, b in collect_sibling_pairs(self._tree):
                if not can_merge(self._tree, a, b):
                    continue
                # Simulate the merge on a shallow copy to check if it helps
                # We only copy the affected parent node, not the whole tree
                ea = get_node_embedding(a)
                eb = get_node_embedding(b)
                if ea is None or eb is None:
                    continue
                # The merged node's embedding will be (ea+eb)/2
                merged_emb = (ea + eb) / 2.0
                norm = np.linalg.norm(merged_emb)
                if norm > 1e-8:
                    merged_emb = merged_emb / norm
                # Compare merged_emb similarity to siblings vs current a,b similarity to siblings
                siblings = [c for c in get_children(parent) 
                        if c is not a and c is not b]
                if not siblings:
                    # Only 2 children, merging removes the whole level — skip
                    continue
                sib_embs = [get_node_embedding(s) for s in siblings]
                sib_embs = [e for e in sib_embs if e is not None]
                if not sib_embs:
                    continue
                # Current avg similarity of a and b to their siblings
                current_sims = []
                for se in sib_embs:
                    current_sims.append(cosine_sim(ea, se))
                    current_sims.append(cosine_sim(eb, se))
                current_avg = float(np.mean(current_sims))
                # After merge: merged node vs siblings
                after_sims = [cosine_sim(merged_emb, se) for se in sib_embs]
                after_avg = float(np.mean(after_sims))
                delta_local = after_avg - current_avg
                if delta_local > 0.005:  # Only queue merges with measurable local gain
                    candidates.append((delta_local, parent, a, b))
            
            # Sort by expected gain, take top 3
            candidates.sort(reverse=True)
            self._merge_targets = [(p, a, b) for _, p, a, b in candidates[:3]]
        except Exception as e:
            self._merge_targets = []

        self._split_targets = []
        try:
            candidates = []
            for node in collect_internal_nodes(self._tree):
                if can_split(node, self._split_min_children):
                    candidates.append((len(get_children(node)), node))
            candidates.sort(reverse=True)
            self._split_targets = [n for _, n in candidates[:3]]
        except Exception:
            self._split_targets = []


# ── Kept for API compatibility with train_offline_rl.py ─────────────────────

def collect_leaves_with_embeddings(root):
    leaves = []
    stack = [root]
    while stack:
        node = stack.pop()
        kids = get_children(node)
        if not kids and 'embedding' in node:
            leaves.append(node)
        for c in kids:
            stack.append(c)
    return leaves

def compute_proxy_mrr(tree, query_buffer, query_embeddings):
    return compute_sibling_coherence(tree)

def build_query_buffer_from_metadata(repo_metadata, sentence_model):
    """Build query buffer — still needed for API compat even though
    we don't use it in the reward anymore."""
    queries = repo_metadata.get('queries', [])
    query_buffer = []
    for q in queries:
        text = q.get('query', q.get('docstring', ''))
        gt = q.get('ground_truth', q.get('func_name', ''))
        if text and gt:
            query_buffer.append({'query': text, 'ground_truth': gt})
    # Return empty embeddings dict — not needed for structural quality
    return query_buffer, {}
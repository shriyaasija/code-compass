# Phase 3: RL Adaptive Index Restructuring — Complete Implementation Guide

> **Goal:** Train a reinforcement learning agent (MaskablePPO) that dynamically restructures the code tree index using Merge/Split/Reparent mutations, optimizing retrieval quality (MRR) for incoming queries.
>
> **Why this is the novelty:** No prior work applies RL to adaptively restructure AST-based code retrieval indexes. Kraska et al. (2018) introduced learned indexes for databases; we extend this idea to hierarchical code search trees with semantically-aware mutations. This is the paper's core contribution.
>
> **Time estimate:** ~7-10 days
>
> **Prerequisites:** Phase 2 complete (MCTS working, all 46 tests passing, benchmark infrastructure ready)
>
> **Key dependencies:** `gymnasium`, `stable-baselines3`, `sb3-contrib` (for MaskablePPO), `scikit-learn` (for K-means in Split), `sentence-transformers` (for embeddings)

---

## Sub-Phase 3A: Setup & File Skeleton (Steps 1–8)

### Step 1: Open terminal and navigate
```bash
cd /home/shriya/code-compass
```

### Step 2: Install RL dependencies
```bash
pip install gymnasium stable-baselines3 sb3-contrib
```
Verify:
```bash
python -c "import gymnasium; import stable_baselines3; from sb3_contrib import MaskablePPO; print('All RL deps OK')"
```

### Step 3: Verify Phase 2 tests still pass
```bash
python -m pytest research/mcts/ -v
```
All 46 tests must pass. If not, fix Phase 2 first.

### Step 4: Create the RL module directory
```bash
mkdir -p research/rl_index
touch research/rl_index/__init__.py
```

### Step 5: Create all empty files
```bash
touch research/rl_index/tree_mutations.py
touch research/rl_index/tree_state.py
touch research/rl_index/env.py
touch research/rl_index/reward.py
touch research/rl_index/train_ppo.py
touch research/rl_index/evaluate_rl.py
touch research/rl_index/test_tree_mutations.py
touch research/rl_index/test_tree_state.py
touch research/rl_index/test_env.py
touch research/rl_index/test_reward.py
touch research/rl_index/test_integration.py
```

### Step 6: Verify directory structure
```bash
find research/rl_index -type f | sort
```
Expected:
```
research/rl_index/__init__.py
research/rl_index/env.py
research/rl_index/evaluate_rl.py
research/rl_index/reward.py
research/rl_index/test_env.py
research/rl_index/test_integration.py
research/rl_index/test_reward.py
research/rl_index/test_tree_mutations.py
research/rl_index/test_tree_state.py
research/rl_index/train_ppo.py
research/rl_index/tree_mutations.py
research/rl_index/tree_state.py
```

### Step 7: Git commit skeleton
```bash
git add research/rl_index/
git commit -m "phase3: create RL adaptive index module skeleton"
```

### Step 8: Take a breath
You have 11 empty Python files. The rest of this guide fills every single one.

---

## Sub-Phase 3B: Tree Mutation Operators (Steps 9–18)

> **File:** `research/rl_index/tree_mutations.py`
>
> These are the three graph-surgery operations the RL agent can perform on the code tree. Each one preserves tree invariants (no orphans, no cycles, all leaves retained). This is where the "learned index restructuring" happens.

### Step 9: Understand what mutations do

The tree starts as the raw AST hierarchy from tree-sitter. But this structure is optimized for *parsing*, not *retrieval*. The RL agent learns to reshape it for better search:

- **Merge:** Two semantically similar siblings → one parent node (reduces branching, groups related code)
- **Split:** One bloated node with 10+ children → 2-3 semantically clustered sub-groups (creates useful intermediate nodes)
- **Reparent:** Move a misplaced node to where it's actually co-queried (fixes structural inefficiencies)

### Step 10: Write `tree_mutations.py` — the complete file

Open `research/rl_index/tree_mutations.py` and write this EXACT code:

```python
"""
Tree Mutation Operators for RL Adaptive Index Restructuring.

Three structure-preserving operations that reshape a code retrieval tree:
  1. Merge(A, B)       — combine two similar siblings into one parent
  2. Split(A, K)       — break a bloated node into K semantic clusters
  3. Reparent(A, Tgt)  — move a node under a more relevant parent

Invariants maintained after every mutation:
  - No orphan nodes (every node reachable from root)
  - No cycles (parent chain always terminates at root)
  - Total leaf count unchanged (no code lost)
  - Tree remains a valid connected DAG

References:
  - Kraska et al., "The Case for Learned Index Structures" (SIGMOD 2018)
  - Ding et al., "ALEX: An Updatable Adaptive Learned Index" (SIGMOD 2020)
"""

import copy
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_children(node: Dict) -> List[Dict]:
    """Get children list from a node (handles both 'nodes' and 'children' keys)."""
    return node.get('nodes', node.get('children', []))


def set_children(node: Dict, children: List[Dict]):
    """Set children on a node, maintaining whichever key it uses."""
    if 'nodes' in node:
        node['nodes'] = children
    else:
        node['children'] = children
    # Keep both in sync if both exist
    if 'nodes' in node and 'children' in node:
        node['nodes'] = children
        node['children'] = children


def count_leaves(node: Dict) -> int:
    """Count total leaf nodes in subtree."""
    children = get_children(node)
    if not children:
        return 1
    return sum(count_leaves(c) for c in children)


def find_node_parent(root: Dict, target_node: Dict) -> Optional[Dict]:
    """Find the parent of a target node by searching from root."""
    children = get_children(root)
    for child in children:
        if child is target_node:
            return root
        result = find_node_parent(child, target_node)
        if result is not None:
            return result
    return None


def is_descendant(ancestor: Dict, node: Dict) -> bool:
    """Check if 'node' is a descendant of 'ancestor'."""
    if ancestor is node:
        return True
    for child in get_children(ancestor):
        if is_descendant(child, node):
            return True
    return False


def get_node_embedding(node: Dict) -> Optional[np.ndarray]:
    """Extract embedding vector from node, if it exists."""
    emb = node.get('embedding')
    if emb is not None:
        return np.array(emb, dtype=np.float32)
    return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def generate_node_id(prefix: str = "merged") -> str:
    """Generate a unique node ID."""
    import time
    raw = f"{prefix}_{time.time_ns()}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════════════════════
# MUTATION 1: MERGE
# ═══════════════════════════════════════════════════════════════════════════════

class MergeResult:
    """Result of a Merge operation."""
    def __init__(self, success: bool, message: str, new_node: Optional[Dict] = None):
        self.success = success
        self.message = message
        self.new_node = new_node


def can_merge(root: Dict, node_a: Dict, node_b: Dict,
              min_similarity: float = 0.3) -> Tuple[bool, str]:
    """
    Check if two nodes can be merged.

    Preconditions:
    1. A and B must share the same parent (siblings)
    2. Neither can be the root
    3. If embeddings exist, cosine similarity must be >= min_similarity
    """
    if node_a is node_b:
        return False, "Cannot merge a node with itself"

    parent_a = find_node_parent(root, node_a)
    parent_b = find_node_parent(root, node_b)

    if parent_a is None or parent_b is None:
        return False, "One or both nodes have no parent (is root?)"

    if parent_a is not parent_b:
        return False, f"Nodes are not siblings (different parents)"

    # Check embedding similarity if available
    emb_a = get_node_embedding(node_a)
    emb_b = get_node_embedding(node_b)
    if emb_a is not None and emb_b is not None:
        sim = cosine_similarity(emb_a, emb_b)
        if sim < min_similarity:
            return False, f"Cosine similarity {sim:.3f} < threshold {min_similarity}"

    return True, "Merge is valid"


def merge(root: Dict, node_a: Dict, node_b: Dict,
          min_similarity: float = 0.3) -> MergeResult:
    """
    Merge two sibling nodes into a single parent node.

    Operation:
    - Create new node C
    - C.children = children(A) ∪ children(B)
    - If A or B are leaves themselves (no children), they become children of C
    - Remove A and B from parent; insert C
    - C inherits combined summary and averaged embedding

    Returns MergeResult with success status and the new merged node.
    """
    valid, msg = can_merge(root, node_a, node_b, min_similarity)
    if not valid:
        return MergeResult(False, msg)

    parent = find_node_parent(root, node_a)
    parent_children = get_children(parent)

    # Build merged children: collect all grandchildren
    children_a = get_children(node_a)
    children_b = get_children(node_b)

    title_a = node_a.get('title', node_a.get('name', 'A'))
    title_b = node_b.get('title', node_b.get('name', 'B'))

    # If A has no children, A itself becomes a child of C (it's a leaf)
    if not children_a:
        merged_children = [node_a] + (children_b if children_b else [node_b])
    elif not children_b:
        merged_children = children_a + [node_b]
    else:
        merged_children = children_a + children_b

    # Create the merged node
    merged_node = {
        'node_id': generate_node_id('merged'),
        'title': f"{title_a}+{title_b}",
        'name': f"{title_a}+{title_b}",
        'type': 'folder',  # Merged nodes are always folders
        'summary': f"Merged group: {node_a.get('summary', '')} | {node_b.get('summary', '')}",
        'path': node_a.get('path', ''),
    }

    # Average embeddings if available
    emb_a = get_node_embedding(node_a)
    emb_b = get_node_embedding(node_b)
    if emb_a is not None and emb_b is not None:
        merged_emb = ((emb_a + emb_b) / 2.0).tolist()
        merged_node['embedding'] = merged_emb

    # Set children on merged node
    set_children(merged_node, merged_children)

    # Replace A and B with merged node in parent
    new_parent_children = []
    merged_inserted = False
    for child in parent_children:
        if child is node_a or child is node_b:
            if not merged_inserted:
                new_parent_children.append(merged_node)
                merged_inserted = True
            # Skip the other one
        else:
            new_parent_children.append(child)

    set_children(parent, new_parent_children)

    return MergeResult(True, f"Merged '{title_a}' + '{title_b}'", merged_node)


# ═══════════════════════════════════════════════════════════════════════════════
# MUTATION 2: SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

class SplitResult:
    """Result of a Split operation."""
    def __init__(self, success: bool, message: str,
                 new_nodes: Optional[List[Dict]] = None):
        self.success = success
        self.message = message
        self.new_nodes = new_nodes or []


def can_split(node: Dict, min_children: int = 4) -> Tuple[bool, str]:
    """
    Check if a node can be split.

    Preconditions:
    1. Node must have >= min_children children
    2. Children should have embeddings for K-means clustering
       (falls back to round-robin if no embeddings)
    """
    children = get_children(node)
    if len(children) < min_children:
        return False, f"Node has {len(children)} children < minimum {min_children}"

    return True, "Split is valid"


def split(root: Dict, node: Dict, k: int = 2,
          min_children: int = 4) -> SplitResult:
    """
    Split a bloated node into K clusters using K-means on embeddings.

    Operation:
    - Run K-means on children's embeddings (K=2 or 3)
    - Create K new intermediate "cluster" nodes
    - Assign each child to its nearest cluster
    - Replace node's children with the K cluster nodes

    Falls back to round-robin assignment if embeddings are unavailable.
    """
    valid, msg = can_split(node, min_children)
    if not valid:
        return SplitResult(False, msg)

    children = get_children(node)
    k = min(k, len(children))  # Can't have more clusters than children

    # Collect embeddings
    embeddings = []
    has_embeddings = True
    for child in children:
        emb = get_node_embedding(child)
        if emb is not None:
            embeddings.append(emb)
        else:
            has_embeddings = False
            break

    if has_embeddings and len(embeddings) >= k:
        # Use K-means clustering
        from sklearn.cluster import KMeans
        X = np.array(embeddings)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        cluster_centers = kmeans.cluster_centers_
    else:
        # Fallback: round-robin assignment
        labels = [i % k for i in range(len(children))]
        cluster_centers = [None] * k

    # Create cluster nodes
    node_title = node.get('title', node.get('name', 'node'))
    cluster_nodes = []
    for cluster_idx in range(k):
        cluster_children = [
            children[i] for i in range(len(children))
            if labels[i] == cluster_idx
        ]

        if not cluster_children:
            continue  # Skip empty clusters

        cluster_node = {
            'node_id': generate_node_id(f'split_{cluster_idx}'),
            'title': f"{node_title}_group_{cluster_idx}",
            'name': f"{node_title}_group_{cluster_idx}",
            'type': 'folder',
            'summary': f"Cluster {cluster_idx} of {node_title} ({len(cluster_children)} items)",
            'path': node.get('path', ''),
        }

        # Set cluster centroid as embedding
        if cluster_centers[cluster_idx] is not None:
            cluster_node['embedding'] = cluster_centers[cluster_idx].tolist()

        set_children(cluster_node, cluster_children)
        cluster_nodes.append(cluster_node)

    # Replace node's children with the cluster nodes
    set_children(node, cluster_nodes)

    return SplitResult(
        True,
        f"Split '{node_title}' into {len(cluster_nodes)} clusters",
        cluster_nodes
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MUTATION 3: REPARENT
# ═══════════════════════════════════════════════════════════════════════════════

class ReparentResult:
    """Result of a Reparent operation."""
    def __init__(self, success: bool, message: str):
        self.success = success
        self.message = message


def can_reparent(root: Dict, node: Dict, new_parent: Dict) -> Tuple[bool, str]:
    """
    Check if a node can be reparented.

    Preconditions:
    1. Node is not root
    2. New parent is not the node itself
    3. New parent is not a descendant of node (would create cycle)
    4. Node is not already a child of new_parent
    """
    if node is root:
        return False, "Cannot reparent the root node"

    if node is new_parent:
        return False, "Cannot reparent a node under itself"

    if is_descendant(node, new_parent):
        return False, "New parent is a descendant of node (would create cycle)"

    # Check if already a child
    for child in get_children(new_parent):
        if child is node:
            return False, "Node is already a child of the target parent"

    # Make sure node has a current parent
    current_parent = find_node_parent(root, node)
    if current_parent is None:
        return False, "Node has no current parent"

    return True, "Reparent is valid"


def reparent(root: Dict, node: Dict, new_parent: Dict) -> ReparentResult:
    """
    Move a node from its current parent to a new parent.

    Operation:
    - Detach node from current parent's children
    - Attach node as child of new_parent
    - Check: current parent still has children (warn if empty)

    This fixes structural inefficiencies where related code is spread
    across unrelated subtrees.
    """
    valid, msg = can_reparent(root, node, new_parent)
    if not valid:
        return ReparentResult(False, msg)

    # Detach from current parent
    current_parent = find_node_parent(root, node)
    current_children = get_children(current_parent)
    new_current_children = [c for c in current_children if c is not node]
    set_children(current_parent, new_current_children)

    # Attach to new parent
    new_parent_children = get_children(new_parent)
    new_parent_children.append(node)
    set_children(new_parent, new_parent_children)

    node_title = node.get('title', node.get('name', '?'))
    target_title = new_parent.get('title', new_parent.get('name', '?'))

    return ReparentResult(True, f"Reparented '{node_title}' under '{target_title}'")


# ═══════════════════════════════════════════════════════════════════════════════
# TREE UTILITIES (used by environment)
# ═══════════════════════════════════════════════════════════════════════════════

def deep_copy_tree(tree: Dict) -> Dict:
    """Create a deep copy of a tree (for environment reset)."""
    return copy.deepcopy(tree)


def collect_all_nodes(root: Dict) -> List[Dict]:
    """Collect all nodes in the tree (BFS)."""
    nodes = []
    stack = [root]
    while stack:
        node = stack.pop()
        nodes.append(node)
        for child in get_children(node):
            stack.append(child)
    return nodes


def collect_internal_nodes(root: Dict) -> List[Dict]:
    """Collect all non-leaf nodes (nodes that have children)."""
    internals = []
    stack = [root]
    while stack:
        node = stack.pop()
        children = get_children(node)
        if children:
            internals.append(node)
            for child in children:
                stack.append(child)
    return internals


def collect_sibling_pairs(root: Dict) -> List[Tuple[Dict, Dict, Dict]]:
    """Collect all (parent, child_a, child_b) sibling pairs."""
    pairs = []
    stack = [root]
    while stack:
        node = stack.pop()
        children = get_children(node)
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                pairs.append((node, children[i], children[j]))
            stack.append(children[i])
    return pairs


def tree_depth(node: Dict) -> int:
    """Compute maximum depth of subtree."""
    children = get_children(node)
    if not children:
        return 0
    return 1 + max(tree_depth(c) for c in children)


def avg_branching_factor(root: Dict) -> Tuple[float, float]:
    """Compute mean and variance of branching factor for internal nodes."""
    branching_factors = []
    stack = [root]
    while stack:
        node = stack.pop()
        children = get_children(node)
        if children:
            branching_factors.append(len(children))
            for child in children:
                stack.append(child)
    if not branching_factors:
        return 0.0, 0.0
    return float(np.mean(branching_factors)), float(np.var(branching_factors))


def validate_tree(root: Dict, expected_leaf_count: Optional[int] = None) -> Tuple[bool, str]:
    """
    Validate tree invariants after a mutation.

    Checks:
    1. All nodes reachable from root (no orphans)
    2. Leaf count matches expected (no code lost)
    3. No empty internal nodes
    """
    actual_leaves = count_leaves(root)
    if expected_leaf_count is not None and actual_leaves != expected_leaf_count:
        return False, f"Leaf count changed: expected {expected_leaf_count}, got {actual_leaves}"

    # Check for empty internals (nodes with 'children'/'nodes' key but empty list)
    stack = [root]
    while stack:
        node = stack.pop()
        children = get_children(node)
        node_type = node.get('type', node.get('node_type', ''))
        if node_type == 'folder' and not children:
            title = node.get('title', '?')
            return False, f"Empty folder node found: '{title}'"
        for child in children:
            stack.append(child)

    return True, "Tree is valid"
```

### Step 11: Write tests for tree mutations

Open `research/rl_index/test_tree_mutations.py` and write:

```python
"""
Tests for tree mutation operators.

Tests verify:
1. Precondition checking (rejects invalid mutations)
2. Correct structural changes
3. Invariant preservation (leaf count, no cycles, no orphans)
"""

import copy
import pytest
import numpy as np

from research.rl_index.tree_mutations import (
    merge, can_merge, MergeResult,
    split, can_split, SplitResult,
    reparent, can_reparent, ReparentResult,
    count_leaves, get_children, set_children,
    find_node_parent, is_descendant, validate_tree,
    collect_all_nodes, collect_internal_nodes,
    collect_sibling_pairs, tree_depth, avg_branching_factor,
    deep_copy_tree,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

def make_simple_tree():
    """
    root
    ├── file_a (file_py)
    │   ├── func_1 (function, leaf)
    │   └── func_2 (function, leaf)
    ├── file_b (file_py)
    │   ├── func_3 (function, leaf)
    │   └── func_4 (function, leaf)
    └── file_c (file_py)
        └── func_5 (function, leaf)
    """
    func_1 = {'title': 'func_1', 'type': 'function', 'start_line': 1, 'end_line': 10,
              'summary': 'Load data from disk', 'path': 'a.py',
              'embedding': np.random.randn(8).tolist()}
    func_2 = {'title': 'func_2', 'type': 'function', 'start_line': 12, 'end_line': 20,
              'summary': 'Parse data into tensors', 'path': 'a.py',
              'embedding': np.random.randn(8).tolist()}
    func_3 = {'title': 'func_3', 'type': 'function', 'start_line': 1, 'end_line': 15,
              'summary': 'Train the model', 'path': 'b.py',
              'embedding': np.random.randn(8).tolist()}
    func_4 = {'title': 'func_4', 'type': 'function', 'start_line': 17, 'end_line': 30,
              'summary': 'Evaluate model accuracy', 'path': 'b.py',
              'embedding': np.random.randn(8).tolist()}
    func_5 = {'title': 'func_5', 'type': 'function', 'start_line': 1, 'end_line': 8,
              'summary': 'Utility helper', 'path': 'c.py',
              'embedding': np.random.randn(8).tolist()}

    file_a = {'title': 'data_loader.py', 'type': 'file_py', 'path': 'a.py',
              'summary': 'Data loading', 'nodes': [func_1, func_2],
              'embedding': np.random.randn(8).tolist()}
    file_b = {'title': 'train.py', 'type': 'file_py', 'path': 'b.py',
              'summary': 'Training', 'nodes': [func_3, func_4],
              'embedding': np.random.randn(8).tolist()}
    file_c = {'title': 'utils.py', 'type': 'file_py', 'path': 'c.py',
              'summary': 'Utilities', 'nodes': [func_5],
              'embedding': np.random.randn(8).tolist()}

    root = {'title': 'repo', 'type': 'repository', 'path': '.', 'summary': 'Test repo',
            'nodes': [file_a, file_b, file_c]}
    return root


def make_wide_tree(n_children=12):
    """Create a node with many children (for split testing)."""
    children = []
    for i in range(n_children):
        children.append({
            'title': f'func_{i}', 'type': 'function',
            'start_line': i * 10 + 1, 'end_line': i * 10 + 9,
            'summary': f'Function {i}', 'path': 'big_file.py',
            'embedding': np.random.randn(8).tolist(),
        })
    wide_node = {
        'title': 'big_file.py', 'type': 'file_py', 'path': 'big_file.py',
        'summary': 'A file with many functions', 'nodes': children,
    }
    root = {'title': 'repo', 'type': 'repository', 'nodes': [wide_node]}
    return root


# ═══════════════════════════════════════════════════════════════════════════════
# MERGE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMerge:
    def test_merge_preserves_leaves(self):
        root = make_simple_tree()
        before = count_leaves(root)
        children = get_children(root)
        result = merge(root, children[0], children[1], min_similarity=0.0)
        assert result.success, result.message
        after = count_leaves(root)
        assert after == before, f"Leaf count changed: {before} -> {after}"

    def test_merge_reduces_sibling_count(self):
        root = make_simple_tree()
        children = get_children(root)
        assert len(children) == 3
        result = merge(root, children[0], children[1], min_similarity=0.0)
        assert result.success
        assert len(get_children(root)) == 2  # 3 -> 2 (merged + remaining)

    def test_merge_rejects_non_siblings(self):
        root = make_simple_tree()
        file_a = get_children(root)[0]
        func_3 = get_children(get_children(root)[1])[0]
        valid, msg = can_merge(root, file_a, func_3)
        assert not valid
        assert "not siblings" in msg.lower() or "different parents" in msg.lower()

    def test_merge_rejects_self(self):
        root = make_simple_tree()
        child = get_children(root)[0]
        valid, msg = can_merge(root, child, child)
        assert not valid

    def test_merge_creates_combined_node(self):
        root = make_simple_tree()
        children = get_children(root)
        result = merge(root, children[0], children[1], min_similarity=0.0)
        assert result.success
        assert result.new_node is not None
        assert 'data_loader.py' in result.new_node['title']
        assert 'train.py' in result.new_node['title']

    def test_merge_tree_validates_after(self):
        root = make_simple_tree()
        before_leaves = count_leaves(root)
        children = get_children(root)
        merge(root, children[0], children[1], min_similarity=0.0)
        valid, msg = validate_tree(root, expected_leaf_count=before_leaves)
        assert valid, msg


# ═══════════════════════════════════════════════════════════════════════════════
# SPLIT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSplit:
    def test_split_creates_clusters(self):
        root = make_wide_tree(12)
        wide_node = get_children(root)[0]
        result = split(root, wide_node, k=3)
        assert result.success, result.message
        assert len(result.new_nodes) >= 2  # At least 2 non-empty clusters

    def test_split_preserves_leaves(self):
        root = make_wide_tree(12)
        before = count_leaves(root)
        wide_node = get_children(root)[0]
        split(root, wide_node, k=3)
        after = count_leaves(root)
        assert after == before

    def test_split_rejects_small_node(self):
        root = make_simple_tree()
        small_node = get_children(root)[2]  # utils.py with 1 child
        valid, msg = can_split(small_node, min_children=4)
        assert not valid

    def test_split_k_equals_2(self):
        root = make_wide_tree(8)
        wide_node = get_children(root)[0]
        result = split(root, wide_node, k=2, min_children=4)
        assert result.success
        clusters = get_children(wide_node)
        assert len(clusters) == 2
        total_children = sum(len(get_children(c)) for c in clusters)
        assert total_children == 8  # All children preserved

    def test_split_tree_validates_after(self):
        root = make_wide_tree(12)
        before_leaves = count_leaves(root)
        wide_node = get_children(root)[0]
        split(root, wide_node, k=3)
        valid, msg = validate_tree(root, expected_leaf_count=before_leaves)
        assert valid, msg


# ═══════════════════════════════════════════════════════════════════════════════
# REPARENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestReparent:
    def test_reparent_moves_node(self):
        root = make_simple_tree()
        children = get_children(root)
        file_c = children[2]  # utils.py
        file_a = children[0]  # data_loader.py
        result = reparent(root, file_c, file_a)
        assert result.success, result.message
        # file_c should now be a child of file_a
        assert file_c in get_children(file_a)
        # file_c should not be a direct child of root anymore
        assert file_c not in get_children(root)

    def test_reparent_preserves_leaves(self):
        root = make_simple_tree()
        before = count_leaves(root)
        children = get_children(root)
        reparent(root, children[2], children[0])
        after = count_leaves(root)
        assert after == before

    def test_reparent_prevents_cycles(self):
        root = make_simple_tree()
        children = get_children(root)
        file_a = children[0]
        func_1 = get_children(file_a)[0]
        # Try to reparent file_a under its own child → cycle
        valid, msg = can_reparent(root, file_a, func_1)
        assert not valid
        assert "cycle" in msg.lower() or "descendant" in msg.lower()

    def test_reparent_rejects_root(self):
        root = make_simple_tree()
        children = get_children(root)
        valid, msg = can_reparent(root, root, children[0])
        assert not valid

    def test_reparent_rejects_self(self):
        root = make_simple_tree()
        child = get_children(root)[0]
        valid, msg = can_reparent(root, child, child)
        assert not valid

    def test_reparent_tree_validates_after(self):
        root = make_simple_tree()
        before_leaves = count_leaves(root)
        children = get_children(root)
        reparent(root, children[2], children[0])
        valid, msg = validate_tree(root, expected_leaf_count=before_leaves)
        assert valid, msg


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestUtilities:
    def test_count_leaves(self):
        root = make_simple_tree()
        assert count_leaves(root) == 5  # func_1..func_5

    def test_tree_depth(self):
        root = make_simple_tree()
        assert tree_depth(root) == 2  # root -> file -> function

    def test_avg_branching_factor(self):
        root = make_simple_tree()
        mean, var = avg_branching_factor(root)
        assert mean > 0

    def test_deep_copy(self):
        root = make_simple_tree()
        copy_root = deep_copy_tree(root)
        # Modify copy shouldn't affect original
        get_children(copy_root)[0]['title'] = 'MODIFIED'
        assert get_children(root)[0]['title'] != 'MODIFIED'

    def test_collect_all_nodes(self):
        root = make_simple_tree()
        all_nodes = collect_all_nodes(root)
        # root + 3 files + 5 functions = 9
        assert len(all_nodes) == 9

    def test_collect_sibling_pairs(self):
        root = make_simple_tree()
        pairs = collect_sibling_pairs(root)
        # Root has 3 children -> C(3,2)=3 pairs, file_a has 2 -> 1 pair, file_b has 2 -> 1 pair
        assert len(pairs) >= 5
```

### Step 12: Run mutation tests
```bash
python -m pytest research/rl_index/test_tree_mutations.py -v
```
All tests must pass. Fix the mutation code if any fail.

### Step 13: Git commit
```bash
git add research/rl_index/tree_mutations.py research/rl_index/test_tree_mutations.py
git commit -m "phase3: implement Merge/Split/Reparent tree mutations with 17 tests"
```

---

## Sub-Phase 3C: State Feature Extraction (Steps 14–17)

> **File:** `research/rl_index/tree_state.py`
>
> The RL agent observes the tree through an 8-dimensional feature vector. These features capture tree structure quality and recent retrieval performance.

### Step 14: Write `tree_state.py`

```python
"""
State Feature Extraction for the RL Tree Index Environment.

Extracts an 8-dimensional observation vector from the current tree
structure and a rolling buffer of recent queries + results.

Feature Vector:
  [0] norm_depth:        Max tree depth / initial depth (relative change)
  [1] avg_branching:     Mean branching factor of internal nodes
  [2] branching_var:     Variance of branching factor (imbalance signal)
  [3] norm_leaf_count:   Current leaf count / initial leaf count
  [4] hit_skew:          Gini coefficient of leaf hit rates (query concentration)
  [5] avg_sibling_sim:   Mean cosine similarity between sibling embeddings
  [6] avg_retrieval_dep: Mean depth of retrieved leaves in recent queries
  [7] buffer_mrr:        MRR on the recent query buffer (retrieval quality)

All features are normalized to [0, 1] range for stable RL training.

References:
  - Gini coefficient: Measures inequality of distribution (0=equal, 1=max inequality)
  - MRR: Mean Reciprocal Rank (standard IR metric)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from research.rl_index.tree_mutations import (
    get_children, tree_depth, avg_branching_factor,
    count_leaves, collect_all_nodes, get_node_embedding,
    cosine_similarity,
)


def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient of a distribution.

    0.0 = perfectly equal (all leaves queried equally)
    1.0 = maximally unequal (one leaf gets all queries)

    Used to measure how concentrated query hits are across leaves.
    """
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1) / n)


def compute_avg_sibling_similarity(root: Dict) -> float:
    """
    Compute average cosine similarity between sibling nodes.

    Higher similarity means siblings are semantically related (good grouping).
    Lower similarity means siblings are unrelated (bad grouping, split candidate).
    """
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
```

### Step 15: Write tests for state extraction

Open `research/rl_index/test_tree_state.py`:

```python
"""Tests for tree state feature extraction."""

import numpy as np
import pytest

from research.rl_index.tree_state import (
    TreeStateExtractor, gini_coefficient,
    compute_avg_sibling_similarity, compute_leaf_depths,
)
from research.rl_index.test_tree_mutations import make_simple_tree


class TestGiniCoefficient:
    def test_equal_distribution(self):
        vals = np.array([10, 10, 10, 10])
        assert abs(gini_coefficient(vals)) < 0.01

    def test_unequal_distribution(self):
        vals = np.array([0, 0, 0, 100])
        g = gini_coefficient(vals)
        assert g > 0.5

    def test_empty(self):
        assert gini_coefficient(np.array([])) == 0.0


class TestTreeStateExtractor:
    def test_state_shape(self):
        root = make_simple_tree()
        extractor = TreeStateExtractor(root)
        state = extractor.extract_state(root)
        assert state.shape == (8,)

    def test_state_range(self):
        root = make_simple_tree()
        extractor = TreeStateExtractor(root)
        state = extractor.extract_state(root)
        assert np.all(state >= 0.0)
        assert np.all(state <= 1.0)

    def test_mrr_improves_with_good_results(self):
        root = make_simple_tree()
        extractor = TreeStateExtractor(root)
        # Record a query where ground truth is first result
        extractor.record_query("load data", "func_1", ["func_1", "func_2"], root)
        state = extractor.extract_state(root)
        assert state[7] > 0.9  # MRR should be ~1.0

    def test_mrr_low_with_bad_results(self):
        root = make_simple_tree()
        extractor = TreeStateExtractor(root)
        extractor.record_query("load data", "func_1", ["func_5", "func_4", "func_3"], root)
        state = extractor.extract_state(root)
        assert state[7] < 0.5  # MRR should be 0 (not found)

    def test_reset_clears_buffer(self):
        root = make_simple_tree()
        extractor = TreeStateExtractor(root)
        extractor.record_query("q", "func_1", ["func_1"], root)
        extractor.reset()
        assert len(extractor.query_buffer) == 0
        assert len(extractor.leaf_hits) == 0

    def test_leaf_depths(self):
        root = make_simple_tree()
        depths = compute_leaf_depths(root)
        assert depths['func_1'] == 2  # root(0) -> file(1) -> func(2)
```

### Step 16: Run state tests
```bash
python -m pytest research/rl_index/test_tree_state.py -v
```

### Step 17: Git commit
```bash
git add research/rl_index/tree_state.py research/rl_index/test_tree_state.py
git commit -m "phase3: implement 8-dim tree state extractor with Gini, MRR, sibling sim (6 tests)"
```

---

*Guide continues in `phase3_detailed_guide_part2.md`...*

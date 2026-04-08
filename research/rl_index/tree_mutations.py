"""
References:
  - Kraska et al., "The Case for Learned Index Structures" (SIGMOD 2018)
  - Ding et al., "ALEX: An Updatable Adaptive Learned Index" (SIGMOD 2020)
"""

import copy
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any

import numpy as np 


def get_children(node: Dict) -> List[Dict]:
    """Get children list from a node"""
    return node.get('nodes', node.get('children', []))

def set_children(node: Dict, children: List[Dict]):
    """Set children on a node"""
    if 'nodes' in node: 
        node['nodes'] = children
    else:
        node['children'] = children

    if 'nodes' in node and 'children' in node:
        node['nodes'] = children
        node['children'] = children

def count_leaves(node: Dict) -> int:
    """Count total leaf nodes in subtree"""
    children = get_children(node)

    if not children: 
        return 1

    return sum(count_leaves(c) for c in children)

def find_node_parent(root: Dict, target_node: Dict) -> Optional[Dict]:
    """Find parent of a node in the tree"""
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

class MergeResult:
    """Result of a Merge operation."""
    def __init__(self, success: bool, message: str, new_node: Optional[Dict] = None):
        self.success = success
        self.message = message
        self.new_node = new_node

def can_merge(root: Dict, node_a: Dict, node_b: Dict,
              min_similarity: float = 0.3) -> Tuple[bool, str]:
    """Check if two nodes can be merged."""
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

def merge(root: Dict, node_a: Dict, node_b: Dict, min_similarity: float = 0.3) -> MergeResult:
    """Merge two sibling nodes into a single parent node."""
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


class SplitResult:
    """Result of a Split operation."""
    def __init__(self, success: bool, message: str,
                 new_nodes: Optional[List[Dict]] = None):
        self.success = success
        self.message = message
        self.new_nodes = new_nodes or []

def can_split(node: Dict, min_children: int = 4) -> Tuple[bool, str]:
    """Check if a node can be split."""
    children = get_children(node)
    if len(children) < min_children:
        return False, f"Node has {len(children)} children < minimum {min_children}"

    return True, "Split is valid"

def split(root: Dict, node: Dict, k: int = 2, min_children: int = 4) -> SplitResult:
    """Split a bloated node into K clusters using K-means on embeddings."""
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


class ReparentResult:
    """Result of a Reparent operation."""
    def __init__(self, success: bool, message: str):
        self.success = success
        self.message = message

def can_reparent(root: Dict, node: Dict, new_parent: Dict) -> Tuple[bool, str]:
    """Check if a node can be reparented."""
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
    """Move a node from its current parent to a new parent."""
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
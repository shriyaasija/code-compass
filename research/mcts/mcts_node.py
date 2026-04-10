"""
MCTSNode: A single node in the MCTS search tree.

Each MCTSNode wraps a tree node from the code AST/PageIndex and tracks:
- Visit count N(s)
- Total value Q(s)
- Children (lazily expanded)
- Parent pointer for backpropagation

Mathematical guarantees:
- UCB1 converges to optimal action as N → ∞ (Auer et al., 2002)
- Regret bound: O(√(K ln N)) where K = branching factor

The UCB1 formula:
    UCB1(s,a) = V̄(s,a) + c * √(ln N(parent) / N(s,a))

Where:
    V̄(s,a) = Q(s,a) / N(s,a)  (average value = exploitation term)
    c = exploration constant (default √2 ≈ 1.414, theoretically optimal for [0,1] rewards)
    N(parent) = visit count of parent node
    N(s,a) = visit count of this node
"""

import math
from typing import List, Dict, Optional, Any
import numpy as np


class MCTSNode:
    """
    A node in the MCTS search tree that wraps a code tree node.
    
    Attributes:
        tree_node: The raw JSON dict from the PageIndex tree
        parent: Parent MCTSNode (None for root)
        children: List of MCTSNode children (lazily created via expand())
        visit_count: N(s) — number of times this node has been visited
        total_value: Q(s) — sum of all rollout/simulation values
        is_expanded: Whether expand() has been called
        is_terminal: Whether this is a leaf node (function/method with line numbers)
    """

    def __init__(self, tree_node: Dict[str, Any], parent: Optional['MCTSNode'] = None):
        self.tree_node = tree_node
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.is_expanded: bool = False
        self.is_terminal: bool = self._check_terminal()
        self.prior: Optional[float] = None  # P(s) from RelevancePrior MLP

    @property
    def average_value(self) -> float:
        """V̄(s) = Q(s) / N(s). Returns 0.0 for unvisited nodes."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb1_score(self, c_explore: float = 1.414) -> float:
        """
        Compute Upper Confidence Bound 1 (UCB1) score.

        UCB1(s) = V̄(s) + c * √(ln N(parent) / N(s))

        Args:
            c_explore: Exploration constant. √2 ≈ 1.414 is theoretically optimal
                       for rewards in [0,1] (Auer et al., 2002).

        Returns:
            UCB1 score. Returns infinity for unvisited nodes (optimistic initialization).
        """
        if self.visit_count == 0:
            return float('inf')

        if self.parent is None or self.parent.visit_count == 0:
            return self.average_value

        exploitation = self.average_value
        exploration = c_explore * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        return exploitation + exploration
    
    def puct_score(self, c_puct: float = 1.5) -> float:
        """
        PUCT (Polynomial Upper Confidence Trees) score.
    
        This is the AlphaZero selection formula:
        PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(parent)) / (1 + N(s,a))
    
        Where P(s,a) is the prior from our RelevancePrior MLP.
        P is set during node expansion via set_prior().
    
        Key difference from UCB1:
        - UCB1 treats all children equally at first (exploration term only)  
        - PUCT uses the prior to immediately bias toward promising nodes
        - This means we need fewer MCTS iterations to find the right branch
    
        Args:
            c_puct: Exploration constant. Higher = more exploration.
                    1.0-2.0 works well for retrieval (vs 5.0 for games).
    
        Returns:
            float PUCT score. Returns prior alone for unvisited nodes.
        """
        if self.prior is None:
            # No prior set: fall back to UCB1 behavior
            return self.ucb1_score(c_puct)
    
        if self.parent is None:
            return self.average_value
    
        exploitation = self.average_value
    
        # PUCT exploration term: scales with sqrt(parent visits),
        # decays as this node is visited more
        exploration = (
            c_puct
            * self.prior
            * (self.parent.visit_count ** 0.5)
            / (1 + self.visit_count)
        )
    
        return exploitation + exploration

    def expand(self) -> List['MCTSNode']:
        """
        Expand this node by creating MCTSNode children from tree children.

        Idempotent: calling expand() multiple times returns the same children.

        Returns:
            List of MCTSNode children.
        """
        if self.is_expanded or self.is_terminal:
            return self.children

        tree_children = self.tree_node.get('nodes', self.tree_node.get('children', []))
        self.children = [MCTSNode(child, parent=self) for child in tree_children]
        self.is_expanded = True
        return self.children

    def backpropagate(self, value: float):
        """
        Backpropagate a simulation value up to the root.

        Updates visit_count and total_value for this node and all ancestors.

        Args:
            value: The simulation result (relevance score ∈ [0, 1]).
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent

    def best_child(self, c_explore: float = 1.414) -> Optional['MCTSNode']:
        """
        Select the child with the highest UCB1 score.

        Used during the SELECTION phase of MCTS.

        Args:
            c_explore: Exploration constant for UCB1.

        Returns:
            The child MCTSNode with the highest UCB1 score, or None if no children.
        """
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb1_score(c_explore))

    def most_visited_child(self) -> Optional['MCTSNode']:
        """
        Select the child with the highest visit count.

        Used for FINAL action selection (after all MCTS iterations complete).
        Most-visited is preferred over highest-value because it's more robust.

        Returns:
            The child MCTSNode with the most visits, or None if no children.
        """
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visit_count)

    def _check_terminal(self) -> bool:
        """
        Check if this node is a terminal (leaf) in the code tree.

        A node is terminal if:
        1. It's a function/method/class with start_line (has code location), OR
        2. It's a file with no children (unparsed file like README, Dockerfile)

        This mirrors _is_leaf() in backend/code_index.py (lines 151-165).
        """
        node_type = self.tree_node.get('type', self.tree_node.get('node_type', ''))

        # AST elements with line numbers are always leaves
        if node_type in ('function', 'method', 'class', 'struct', 'impl', 'module'):
            if 'start_line' in self.tree_node:
                return True

        # Files without children are leaves (unparsed files)
        if node_type.startswith('file'):
            children = self.tree_node.get('nodes', self.tree_node.get('children', []))
            if not children:
                return True

        return False

    def get_tree_children_raw(self) -> List[Dict[str, Any]]:
        """Get the raw tree children dicts (before wrapping in MCTSNode)."""
        return self.tree_node.get('nodes', self.tree_node.get('children', []))

    def depth(self) -> int:
        """Compute depth of this node (root = 0)."""
        d = 0
        node = self.parent
        while node is not None:
            d += 1
            node = node.parent
        return d

    def path_from_root(self) -> List[str]:
        """Get the path of node names from root to this node."""
        path = []
        node = self
        while node is not None:
            name = node.tree_node.get('title', node.tree_node.get('name', '?'))
            path.append(name)
            node = node.parent
        return list(reversed(path))

    def __repr__(self) -> str:
        name = self.tree_node.get('title', self.tree_node.get('name', '?'))
        return (
            f"MCTSNode({name}, visits={self.visit_count}, "
            f"val={self.average_value:.3f}, terminal={self.is_terminal})"
        )
    
    def set_prior(self, prior: float):
        """Set the prior probability P(s) for this node from the RelevancePrior MLP."""
        self.prior = float(np.clip(prior, 1e-6, 1.0))  

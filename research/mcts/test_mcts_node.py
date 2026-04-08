"""
Unit tests for MCTSNode: UCB1 computation, backpropagation, expansion, terminal detection.

These tests validate the mathematical correctness and structural invariants
of the MCTSNode class — the fundamental data structure of our MCTS search.
"""

import math
import pytest

from research.mcts.mcts_node import MCTSNode


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES: Reusable test tree structures
# ═══════════════════════════════════════════════════════════════════════════════

def make_simple_node(title="test", node_type="folder", children=None, **kwargs):
    """Create a simple tree node dict for testing."""
    node = {"title": title, "type": node_type, "node_id": f"test_{title}"}
    if children is not None:
        node["children"] = children
    node.update(kwargs)
    return node


def make_function_node(name="my_func", start_line=10, end_line=20):
    """Create a terminal function node."""
    return {
        "title": name,
        "name": name,
        "type": "function",
        "node_id": f"func_{name}",
        "start_line": start_line,
        "end_line": end_line,
        "summary": f"Function {name} that does something",
        "path": f"src/{name}.py",
    }


def make_three_level_tree():
    """Create a 3-level tree: root -> [file_a, file_b] -> functions."""
    func1 = make_function_node("func1", 1, 10)
    func2 = make_function_node("func2", 11, 20)
    func3 = make_function_node("func3", 1, 15)

    file_a = make_simple_node("file_a.py", "file_py", children=[func1, func2])
    file_b = make_simple_node("file_b.py", "file_py", children=[func3])

    root = make_simple_node("repo_root", "repository", children=[file_a, file_b])
    return root


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: New node initialization
# ═══════════════════════════════════════════════════════════════════════════════

class TestMCTSNodeInit:
    def test_new_node_has_zero_visits(self):
        node = MCTSNode(make_simple_node("test", "folder", children=[]))
        assert node.visit_count == 0
        assert node.total_value == 0.0
        assert node.average_value == 0.0

    def test_new_node_is_not_expanded(self):
        node = MCTSNode(make_simple_node("test", "folder", children=[{"title": "child"}]))
        assert node.is_expanded is False
        assert len(node.children) == 0

    def test_root_has_no_parent(self):
        node = MCTSNode(make_simple_node())
        assert node.parent is None


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: UCB1 score computation
# ═══════════════════════════════════════════════════════════════════════════════

class TestUCB1:
    def test_unvisited_node_is_infinity(self):
        parent = MCTSNode(make_simple_node("parent"))
        child_tree = make_simple_node("child")
        child = MCTSNode(child_tree, parent=parent)
        assert child.ucb1_score() == float('inf')

    def test_ucb1_with_zero_explore_is_pure_exploitation(self):
        parent = MCTSNode(make_simple_node("parent"))
        parent.visit_count = 100

        child = MCTSNode(make_simple_node("child"), parent=parent)
        child.visit_count = 10
        child.total_value = 7.0  # average = 0.7

        score = child.ucb1_score(c_explore=0.0)
        assert abs(score - 0.7) < 1e-9

    def test_ucb1_includes_exploration_term(self):
        parent = MCTSNode(make_simple_node("parent"))
        parent.visit_count = 100

        child = MCTSNode(make_simple_node("child"), parent=parent)
        child.visit_count = 10
        child.total_value = 7.0  # average = 0.7

        score_with_explore = child.ucb1_score(c_explore=1.414)
        score_without_explore = child.ucb1_score(c_explore=0.0)

        assert score_with_explore > score_without_explore

    def test_ucb1_exploration_decreases_with_visits(self):
        parent = MCTSNode(make_simple_node("parent"))
        parent.visit_count = 100

        child = MCTSNode(make_simple_node("child"), parent=parent)

        # Few visits → high UCB1 (exploration dominates)
        child.visit_count = 1
        child.total_value = 0.5
        score_few_visits = child.ucb1_score(1.414)

        # Many visits → lower UCB1 (exploitation dominates)
        child.visit_count = 50
        child.total_value = 25.0  # same average (0.5)
        score_many_visits = child.ucb1_score(1.414)

        assert score_few_visits > score_many_visits

    def test_ucb1_no_parent_returns_average(self):
        node = MCTSNode(make_simple_node("orphan"))
        node.visit_count = 10
        node.total_value = 7.0
        assert abs(node.ucb1_score() - 0.7) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Backpropagation
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackpropagation:
    def test_backpropagate_updates_single_node(self):
        node = MCTSNode(make_simple_node())
        node.backpropagate(0.8)
        assert node.visit_count == 1
        assert abs(node.total_value - 0.8) < 1e-9

    def test_backpropagate_updates_ancestors(self):
        root = MCTSNode(make_simple_node("root"))
        child = MCTSNode(make_simple_node("child"), parent=root)
        grandchild = MCTSNode(make_simple_node("grandchild"), parent=child)

        grandchild.backpropagate(0.8)

        assert grandchild.visit_count == 1
        assert child.visit_count == 1
        assert root.visit_count == 1

        assert abs(grandchild.total_value - 0.8) < 1e-9
        assert abs(child.total_value - 0.8) < 1e-9
        assert abs(root.total_value - 0.8) < 1e-9

    def test_backpropagate_accumulates(self):
        root = MCTSNode(make_simple_node("root"))
        child = MCTSNode(make_simple_node("child"), parent=root)

        child.backpropagate(0.8)
        child.backpropagate(0.6)

        assert child.visit_count == 2
        assert abs(child.total_value - 1.4) < 1e-9
        assert abs(child.average_value - 0.7) < 1e-9

        assert root.visit_count == 2
        assert abs(root.total_value - 1.4) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: Expansion
# ═══════════════════════════════════════════════════════════════════════════════

class TestExpansion:
    def test_expand_creates_children(self):
        tree = make_simple_node("root", "folder", children=[
            make_simple_node("a"),
            make_simple_node("b"),
            make_simple_node("c"),
        ])
        node = MCTSNode(tree)
        children = node.expand()

        assert len(children) == 3
        assert node.is_expanded is True
        assert all(isinstance(c, MCTSNode) for c in children)

    def test_expand_is_idempotent(self):
        tree = make_simple_node("root", "folder", children=[
            make_simple_node("a"),
        ])
        node = MCTSNode(tree)

        children1 = node.expand()
        children2 = node.expand()

        assert children1 is children2  # Same list object

    def test_expand_sets_parent(self):
        tree = make_simple_node("root", "folder", children=[
            make_simple_node("child"),
        ])
        node = MCTSNode(tree)
        children = node.expand()

        assert children[0].parent is node

    def test_expand_terminal_returns_empty(self):
        func = make_function_node("my_func")
        node = MCTSNode(func)
        children = node.expand()

        assert len(children) == 0
        assert node.is_terminal is True

    def test_expand_handles_nodes_key(self):
        """Test that expansion works with 'nodes' key (alternative to 'children')."""
        tree = {"title": "root", "type": "folder", "nodes": [
            make_simple_node("a"),
            make_simple_node("b"),
        ]}
        node = MCTSNode(tree)
        children = node.expand()
        assert len(children) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: Terminal detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestTerminalDetection:
    def test_function_with_lines_is_terminal(self):
        func = make_function_node("my_func", 10, 20)
        node = MCTSNode(func)
        assert node.is_terminal is True

    def test_folder_is_not_terminal(self):
        folder = make_simple_node("src", "folder", children=[make_simple_node("a")])
        node = MCTSNode(folder)
        assert node.is_terminal is False

    def test_file_without_children_is_terminal(self):
        file_node = make_simple_node("README.md", "file_md", children=[])
        node = MCTSNode(file_node)
        assert node.is_terminal is True

    def test_file_with_children_is_not_terminal(self):
        file_node = make_simple_node("model.py", "file_py", children=[
            make_function_node("forward"),
        ])
        node = MCTSNode(file_node)
        assert node.is_terminal is False

    def test_method_with_lines_is_terminal(self):
        method = {"title": "forward", "type": "method", "start_line": 5, "end_line": 20}
        node = MCTSNode(method)
        assert node.is_terminal is True

    def test_class_with_lines_is_terminal(self):
        cls = {"title": "Model", "type": "class", "start_line": 1, "end_line": 50}
        node = MCTSNode(cls)
        assert node.is_terminal is True


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: Child selection
# ═══════════════════════════════════════════════════════════════════════════════

class TestChildSelection:
    def test_best_child_selects_highest_ucb(self):
        parent = MCTSNode(make_simple_node("parent"))
        parent.visit_count = 100

        # Create 3 children with different values
        c1 = MCTSNode(make_simple_node("c1"), parent=parent)
        c2 = MCTSNode(make_simple_node("c2"), parent=parent)
        c3 = MCTSNode(make_simple_node("c3"), parent=parent)

        c1.visit_count, c1.total_value = 10, 3.0   # avg=0.3
        c2.visit_count, c2.total_value = 10, 9.0   # avg=0.9 (highest)
        c3.visit_count, c3.total_value = 10, 5.0   # avg=0.5

        parent.children = [c1, c2, c3]

        best = parent.best_child(c_explore=0.0)  # Pure exploitation
        assert best is c2

    def test_most_visited_child(self):
        parent = MCTSNode(make_simple_node("parent"))

        c1 = MCTSNode(make_simple_node("c1"), parent=parent)
        c2 = MCTSNode(make_simple_node("c2"), parent=parent)
        c3 = MCTSNode(make_simple_node("c3"), parent=parent)

        c1.visit_count = 5
        c2.visit_count = 50  # Most visited
        c3.visit_count = 20

        parent.children = [c1, c2, c3]

        most_visited = parent.most_visited_child()
        assert most_visited is c2

    def test_best_child_no_children_returns_none(self):
        parent = MCTSNode(make_simple_node("parent"))
        assert parent.best_child() is None

    def test_most_visited_no_children_returns_none(self):
        parent = MCTSNode(make_simple_node("parent"))
        assert parent.most_visited_child() is None


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: Utility methods
# ═══════════════════════════════════════════════════════════════════════════════

class TestUtilities:
    def test_depth_root_is_zero(self):
        root = MCTSNode(make_simple_node("root"))
        assert root.depth() == 0

    def test_depth_child_is_one(self):
        root = MCTSNode(make_simple_node("root"))
        child = MCTSNode(make_simple_node("child"), parent=root)
        assert child.depth() == 1

    def test_path_from_root(self):
        root = MCTSNode(make_simple_node("root"))
        child = MCTSNode(make_simple_node("src"), parent=root)
        grandchild = MCTSNode(make_simple_node("model.py"), parent=child)

        path = grandchild.path_from_root()
        assert path == ["root", "src", "model.py"]

    def test_repr(self):
        node = MCTSNode(make_simple_node("my_node"))
        node.visit_count = 5
        node.total_value = 3.5
        r = repr(node)
        assert "my_node" in r
        assert "visits=5" in r

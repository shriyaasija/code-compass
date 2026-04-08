"""
Mathematical correctness tests for UCB1 formula.

These tests verify that the UCB1 implementation matches the theoretical
formula from Auer et al. (2002) "Finite-time Analysis of the Multiarmed
Bandit Problem":

    UCB1(s,a) = V̄(s,a) + c * √(ln N(parent) / N(s,a))

We test:
1. Manual computation against known values
2. Exploration dominance with few visits
3. Exploitation dominance with many visits
4. c=0 reduces to pure exploitation
5. Convergence behavior over many pulls
"""

import math
import pytest

from research.mcts.mcts_node import MCTSNode


def make_node(title="test", parent=None):
    """Quick helper to create an MCTSNode."""
    tree = {"title": title, "type": "folder", "children": []}
    return MCTSNode(tree, parent=parent)


class TestUCB1MathematicalCorrectness:

    def test_ucb1_formula_manual_computation(self):
        """
        Verify UCB1 against hand-computed expected values.

        Parent: N=100
        Child:  N=10, Q=7.0 (avg=0.7), c=1.414

        Expected:
            exploitation = 0.7
            exploration  = 1.414 * sqrt(ln(100) / 10)
                         = 1.414 * sqrt(4.60517 / 10)
                         = 1.414 * sqrt(0.460517)
                         = 1.414 * 0.67862
                         ≈ 0.95957
            UCB1 = 0.7 + 0.95957 ≈ 1.65957
        """
        parent = make_node("parent")
        parent.visit_count = 100

        child = make_node("child", parent=parent)
        child.visit_count = 10
        child.total_value = 7.0

        expected_exploitation = 0.7
        expected_exploration = 1.414 * math.sqrt(math.log(100) / 10)
        expected_ucb1 = expected_exploitation + expected_exploration

        actual = child.ucb1_score(c_explore=1.414)

        assert abs(actual - expected_ucb1) < 0.001, \
            f"UCB1 mismatch: expected {expected_ucb1:.5f}, got {actual:.5f}"

    def test_exploration_dominates_with_few_visits(self):
        """
        With few visits, unexplored/rarely-visited nodes should have
        higher UCB1 than frequently-visited high-value nodes.

        This ensures the algorithm tries all branches before committing.
        """
        parent = make_node("parent")
        parent.visit_count = 100

        # Rarely visited, low value
        rare_child = make_node("rare", parent=parent)
        rare_child.visit_count = 1
        rare_child.total_value = 0.3  # avg = 0.3

        # Frequently visited, high value
        frequent_child = make_node("frequent", parent=parent)
        frequent_child.visit_count = 50
        frequent_child.total_value = 45.0  # avg = 0.9

        rare_ucb = rare_child.ucb1_score(1.414)
        frequent_ucb = frequent_child.ucb1_score(1.414)

        assert rare_ucb > frequent_ucb, \
            f"Exploration should dominate: rare={rare_ucb:.3f} should > frequent={frequent_ucb:.3f}"

    def test_exploitation_dominates_with_many_visits(self):
        """
        With very many visits, the exploration term becomes negligible.
        The higher-value node should win.
        """
        parent = make_node("parent")
        parent.visit_count = 100000

        # Low value, many visits
        low_child = make_node("low", parent=parent)
        low_child.visit_count = 10000
        low_child.total_value = 3000.0  # avg = 0.3

        # High value, many visits
        high_child = make_node("high", parent=parent)
        high_child.visit_count = 10000
        high_child.total_value = 9000.0  # avg = 0.9

        low_ucb = low_child.ucb1_score(1.414)
        high_ucb = high_child.ucb1_score(1.414)

        assert high_ucb > low_ucb, \
            f"Exploitation should dominate: high={high_ucb:.3f} should > low={low_ucb:.3f}"

    def test_c_zero_is_pure_exploitation(self):
        """With c=0, UCB1 should equal the average value exactly."""
        parent = make_node("parent")
        parent.visit_count = 100

        child = make_node("child", parent=parent)
        child.visit_count = 20
        child.total_value = 14.0  # avg = 0.7

        ucb_score = child.ucb1_score(c_explore=0.0)
        assert abs(ucb_score - 0.7) < 1e-9

    def test_ucb1_convergence_prefers_better_arm(self):
        """
        Simulate pulling a "good" arm (avg=0.8) and "bad" arm (avg=0.3)
        many times. After convergence, good arm should have higher UCB1.
        """
        parent = make_node("parent")
        parent.visit_count = 2000

        good_arm = make_node("good", parent=parent)
        good_arm.visit_count = 1000
        good_arm.total_value = 800.0  # avg = 0.8

        bad_arm = make_node("bad", parent=parent)
        bad_arm.visit_count = 1000
        bad_arm.total_value = 300.0  # avg = 0.3

        good_ucb = good_arm.ucb1_score(1.414)
        bad_ucb = bad_arm.ucb1_score(1.414)

        assert good_ucb > bad_ucb, \
            f"After convergence, good arm ({good_ucb:.3f}) should beat bad arm ({bad_ucb:.3f})"

    def test_exploration_term_is_correct_formula(self):
        """
        Directly verify: exploration_term = c * sqrt(ln(N_parent) / N_child)
        """
        parent = make_node("parent")
        parent.visit_count = 50

        child = make_node("child", parent=parent)
        child.visit_count = 5
        child.total_value = 2.5  # avg = 0.5

        c = 2.0
        ucb = child.ucb1_score(c_explore=c)

        expected_exploit = 0.5
        expected_explore = c * math.sqrt(math.log(50) / 5)
        expected = expected_exploit + expected_explore

        assert abs(ucb - expected) < 1e-9, \
            f"Expected {expected:.6f}, got {ucb:.6f}"

    def test_ucb1_ordering_with_three_arms(self):
        """
        Test that UCB1 correctly orders 3 arms with different characteristics.
        """
        parent = make_node("parent")
        parent.visit_count = 30

        # Arm A: high value, many visits (exploitation)
        arm_a = make_node("a", parent=parent)
        arm_a.visit_count = 20
        arm_a.total_value = 16.0  # avg = 0.8

        # Arm B: low value, many visits
        arm_b = make_node("b", parent=parent)
        arm_b.visit_count = 9
        arm_b.total_value = 2.7  # avg = 0.3

        # Arm C: unvisited (should be highest due to infinity)
        arm_c = make_node("c", parent=parent)
        arm_c.visit_count = 0

        a_ucb = arm_a.ucb1_score(1.414)
        b_ucb = arm_b.ucb1_score(1.414)
        c_ucb = arm_c.ucb1_score(1.414)

        # Unvisited should always be first
        assert c_ucb == float('inf')
        assert c_ucb > a_ucb > b_ucb

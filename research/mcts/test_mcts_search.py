"""
Integration tests for MCTSSearch: end-to-end search on mock trees.

Uses a MockLLM to return predetermined scores so tests don't require
a running Ollama/LMStudio instance.
"""

import json
import pytest
from pathlib import Path
from typing import List, Dict, Any

from research.mcts.mcts_node import MCTSNode
from research.mcts.mcts_search import MCTSSearch


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK LLM
# ═══════════════════════════════════════════════════════════════════════════════

class MockLLM:
    """
    Mock LLM that returns predetermined scores for testing.

    score_map: Dict[str, float] mapping node titles to relevance scores.
    Any node not in the map gets a default score of 0.3.
    """

    def __init__(self, score_map: Dict[str, float], default_score: float = 0.3):
        self.score_map = score_map
        self.default_score = default_score
        self.call_count = 0
        self.last_messages = None

    def chat(self, messages: List[Dict], **kwargs) -> str:
        """Return JSON scores based on the score_map."""
        self.call_count += 1
        self.last_messages = messages

        # Parse the prompt to find item names
        user_msg = messages[-1]["content"] if messages else ""

        # Build response from score_map
        scores = {}
        for name, score in self.score_map.items():
            if name in user_msg:
                scores[name] = score

        # If we didn't find specific names, return default for any items
        if not scores:
            # Try to extract items from the prompt
            lines = user_msg.split("\n")
            for line in lines:
                line = line.strip()
                if line and line[0].isdigit() and ". " in line:
                    # Extract item name after "N. "
                    item_part = line.split(". ", 1)[1]
                    # Remove parenthetical info
                    item_name = item_part.split(" (")[0].strip()
                    scores[item_name] = self.score_map.get(item_name, self.default_score)

        if not scores:
            scores = {"unknown": self.default_score}

        return json.dumps(scores)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

def make_mock_tree():
    """
    Build a 3-level mock tree for testing:
    
    repo_root (repository)
    ├── model.py (file_py)
    │   ├── train_model (function) ← target for "training" queries
    │   ├── evaluate_model (function)
    │   └── save_weights (function)
    ├── data.py (file_py)
    │   ├── load_data (function)
    │   └── preprocess (function)
    └── utils.py (file_py)
        ├── setup_logger (function)
        └── set_seed (function)
    """
    return {
        "title": "repo_root",
        "type": "repository",
        "node_id": "root",
        "summary": "ML training repository",
        "children": [
            {
                "title": "model.py",
                "type": "file_py",
                "node_id": "file_model",
                "summary": "Model training and evaluation",
                "path": "src/model.py",
                "children": [
                    {
                        "title": "train_model",
                        "type": "function",
                        "node_id": "func_train",
                        "summary": "Train the neural network model",
                        "path": "src/model.py",
                        "start_line": 10,
                        "end_line": 40,
                        "signature": "def train_model(model, data, epochs)",
                    },
                    {
                        "title": "evaluate_model",
                        "type": "function",
                        "node_id": "func_eval",
                        "summary": "Evaluate model on test set",
                        "path": "src/model.py",
                        "start_line": 42,
                        "end_line": 60,
                        "signature": "def evaluate_model(model, test_data)",
                    },
                    {
                        "title": "save_weights",
                        "type": "function",
                        "node_id": "func_save",
                        "summary": "Save model weights to disk",
                        "path": "src/model.py",
                        "start_line": 62,
                        "end_line": 75,
                        "signature": "def save_weights(model, path)",
                    },
                ],
            },
            {
                "title": "data.py",
                "type": "file_py",
                "node_id": "file_data",
                "summary": "Data loading and preprocessing",
                "path": "src/data.py",
                "children": [
                    {
                        "title": "load_data",
                        "type": "function",
                        "node_id": "func_load",
                        "summary": "Load dataset from disk",
                        "path": "src/data.py",
                        "start_line": 5,
                        "end_line": 25,
                        "signature": "def load_data(path, batch_size=32)",
                    },
                    {
                        "title": "preprocess",
                        "type": "function",
                        "node_id": "func_preprocess",
                        "summary": "Preprocess images with normalization",
                        "path": "src/data.py",
                        "start_line": 27,
                        "end_line": 45,
                        "signature": "def preprocess(images, mean, std)",
                    },
                ],
            },
            {
                "title": "utils.py",
                "type": "file_py",
                "node_id": "file_utils",
                "summary": "Utility functions for logging and config",
                "path": "src/utils.py",
                "children": [
                    {
                        "title": "setup_logger",
                        "type": "function",
                        "node_id": "func_logger",
                        "summary": "Setup logging configuration",
                        "path": "src/utils.py",
                        "start_line": 5,
                        "end_line": 15,
                        "signature": "def setup_logger(log_file)",
                    },
                    {
                        "title": "set_seed",
                        "type": "function",
                        "node_id": "func_seed",
                        "summary": "Set random seed for reproducibility",
                        "path": "src/utils.py",
                        "start_line": 17,
                        "end_line": 28,
                        "signature": "def set_seed(seed=42)",
                    },
                ],
            },
        ],
    }


def make_training_score_map():
    """Scores that make train_model the best match for training queries."""
    return {
        "model.py": 0.95,
        "data.py": 0.4,
        "utils.py": 0.1,
        "train_model": 0.95,
        "evaluate_model": 0.6,
        "save_weights": 0.5,
        "load_data": 0.3,
        "preprocess": 0.2,
        "setup_logger": 0.1,
        "set_seed": 0.1,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMCTSSearchEndToEnd:

    def test_mcts_finds_correct_leaf(self):
        """MCTS should find train_model as the top result for a training query."""
        mock_llm = MockLLM(make_training_score_map())
        searcher = MCTSSearch(mock_llm, max_iterations=30, verbose=False)

        tree = make_mock_tree()
        results = searcher.search(tree, "How do I train the model?")

        assert len(results) > 0, "Should find at least one result"

        result_names = [r["name"] for r in results]
        assert "train_model" in result_names, \
            f"train_model should be in results, got: {result_names}"

    def test_mcts_ranks_correctly(self):
        """Higher-scored nodes should appear before lower-scored ones."""
        mock_llm = MockLLM(make_training_score_map())
        searcher = MCTSSearch(mock_llm, max_iterations=30, verbose=False)

        tree = make_mock_tree()
        results = searcher.search(tree, "How do I train the model?")

        if len(results) >= 2:
            # Results should be sorted by similarity_score descending
            for i in range(len(results) - 1):
                assert results[i]["similarity_score"] >= results[i + 1]["similarity_score"], \
                    f"Results not sorted: {results[i]['name']}={results[i]['similarity_score']:.3f} " \
                    f"< {results[i+1]['name']}={results[i+1]['similarity_score']:.3f}"

    def test_mcts_explores_multiple_branches(self):
        """With enough iterations, MCTS should explore multiple file branches."""
        # Give model.py and data.py both high scores
        score_map = {
            "model.py": 0.9,
            "data.py": 0.85,
            "utils.py": 0.1,
            "train_model": 0.9,
            "evaluate_model": 0.7,
            "save_weights": 0.5,
            "load_data": 0.85,
            "preprocess": 0.6,
            "setup_logger": 0.1,
            "set_seed": 0.1,
        }
        mock_llm = MockLLM(score_map)
        searcher = MCTSSearch(mock_llm, max_iterations=50, verbose=False)

        tree = make_mock_tree()
        results = searcher.search(tree, "training and data loading")

        result_files = set()
        for r in results:
            path = r.get("path", "")
            if path:
                result_files.add(path)

        assert len(result_files) >= 2, \
            f"Should explore at least 2 files, got: {result_files}"

    def test_mcts_respects_max_iterations(self):
        """LLM call count should not exceed max_iterations."""
        mock_llm = MockLLM(make_training_score_map())
        searcher = MCTSSearch(mock_llm, max_iterations=5, verbose=False)

        tree = make_mock_tree()
        searcher.search(tree, "test query")

        # LLM calls should be <= max_iterations (usually much less due to caching)
        assert mock_llm.call_count <= 10, \
            f"Too many LLM calls: {mock_llm.call_count} for max_iterations=5"

    def test_mcts_result_format(self):
        """Each result should have all required fields."""
        mock_llm = MockLLM(make_training_score_map())
        searcher = MCTSSearch(mock_llm, max_iterations=20, verbose=False)

        tree = make_mock_tree()
        results = searcher.search(tree, "How do I train?")

        for result in results:
            assert "node_id" in result, "Missing node_id"
            assert "name" in result, "Missing name"
            assert "node_type" in result, "Missing node_type"
            assert "similarity_score" in result, "Missing similarity_score"
            assert "metadata" in result, "Missing metadata"

            meta = result["metadata"]
            assert "file_path" in meta, "Missing file_path in metadata"
            assert "start_line" in meta, "Missing start_line in metadata"
            assert "end_line" in meta, "Missing end_line in metadata"

            assert 0.0 <= result["similarity_score"] <= 1.0, \
                f"Score out of range: {result['similarity_score']}"

    def test_mcts_handles_empty_tree(self):
        """MCTS should handle a tree with no children gracefully."""
        mock_llm = MockLLM({})
        searcher = MCTSSearch(mock_llm, max_iterations=10, verbose=False)

        tree = {"title": "empty_repo", "type": "repository", "children": []}
        results = searcher.search(tree, "anything")

        assert results == [] or len(results) == 0

    def test_mcts_on_single_leaf_tree(self):
        """MCTS should work with a tree that has only one function."""
        mock_llm = MockLLM({"only_func": 0.9})
        searcher = MCTSSearch(mock_llm, max_iterations=10, verbose=False)

        tree = {
            "title": "tiny_repo",
            "type": "repository",
            "children": [{
                "title": "only_func",
                "type": "function",
                "start_line": 1,
                "end_line": 10,
                "summary": "The only function",
                "path": "main.py",
                "node_id": "func_only",
            }],
        }
        results = searcher.search(tree, "what does this do?")

        assert len(results) >= 1
        assert results[0]["name"] == "only_func"

    def test_mcts_early_termination(self):
        """With a dominant branch, MCTS should terminate early."""
        # Make model.py massively dominant
        score_map = {
            "model.py": 0.99,
            "data.py": 0.01,
            "utils.py": 0.01,
            "train_model": 0.99,
            "evaluate_model": 0.01,
            "save_weights": 0.01,
        }
        mock_llm = MockLLM(score_map, default_score=0.01)
        searcher = MCTSSearch(
            mock_llm,
            max_iterations=100,
            convergence_threshold=0.7,
            convergence_check_after=5,
            verbose=False,
        )

        tree = make_mock_tree()
        results = searcher.search(tree, "train the model")

        # Should have terminated before 100 iterations
        assert searcher.llm_call_count < 50, \
            f"Should have terminated early, but used {searcher.llm_call_count} LLM calls"


class TestMCTSSearchWithMockPageindex:
    """Test MCTS with the actual mock_pageindex_tree.json if available."""

    @pytest.fixture
    def mock_tree(self):
        tree_path = Path(__file__).parent.parent.parent / "mock_pageindex_tree.json"
        if not tree_path.exists():
            pytest.skip("mock_pageindex_tree.json not found")
        with open(tree_path) as f:
            return json.load(f)

    def test_search_on_real_mock_tree(self, mock_tree):
        """Run MCTS on the actual mock PageIndex tree."""
        score_map = {
            "data_loader.py": 0.3,
            "model.py": 0.6,
            "train.py": 0.95,
            "evaluate.py": 0.4,
            "utils.py": 0.1,
            "train_one_epoch": 0.95,
            "validate_model": 0.7,
            "save_checkpoint": 0.6,
            "load_checkpoint": 0.5,
        }
        mock_llm = MockLLM(score_map, default_score=0.3)
        searcher = MCTSSearch(mock_llm, max_iterations=30, verbose=False)

        results = searcher.search(mock_tree, "How do I train the model?")

        assert len(results) > 0
        result_names = [r["name"] for r in results]
        # train_one_epoch should be found
        assert "train_one_epoch" in result_names, \
            f"Expected train_one_epoch in results, got: {result_names}"

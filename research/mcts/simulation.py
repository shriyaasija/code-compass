"""
LLMSimulator: Bridge between MCTS and the LLM scoring oracle.

In classical MCTS (e.g., AlphaGo), the simulation phase plays random moves
until the game ends. In Code Compass, "simulation" means asking the LLM:
    "Given this query, how relevant is this code node?"

Key design decisions:
- Batch scoring: All siblings are scored in ONE LLM call (not one per child)
- Caching: Results are cached by (node_id, query) to avoid redundant LLM calls
- Prompt compatibility: Uses the same prompt format as code_index.py's _score_siblings
"""

import json
import hashlib
from typing import List, Dict, Any, Optional

from research.mcts.mcts_node import MCTSNode


class LLMSimulator:
    """
    LLM-based simulation oracle for MCTS.
    
    Scores code tree nodes for relevance to a natural language query.
    Wraps the LLM client (Ollama/LMStudio) with caching and batch scoring.
    """

    def __init__(self, llm_client, cache_size: int = 2000):
        """
        Args:
            llm_client: An OllamaLLM or LMStudioLLM instance (ignored now, kept for signature).
            cache_size: Maximum number of cached score results.
        """
        self.llm = llm_client
        self.cache: Dict[str, Dict[str, float]] = {}
        self.cache_size = cache_size
        self.total_llm_calls: int = 0
        from sentence_transformers import CrossEncoder
        print("🔄 Loading cross-encoder model for MCTS simulation...")
        self.scorer = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("✅ Cross-encoder loaded")

    def batch_score_children(
        self,
        children: List[MCTSNode],
        query: str,
        parent_node: MCTSNode,
        trajectory: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Score all sibling nodes in a single LLM call.

        This mirrors _score_siblings() from backend/code_index.py.
        All children of a parent are scored together for efficiency.

        Args:
            children: List of MCTSNode children to score.
            query: The user's natural language query.
            parent_node: The parent MCTSNode (for context path).
            trajectory: Path from root to parent (for context).

        Returns:
            Dict mapping child title/name to relevance score ∈ [0, 1].
        """
        if not children:
            return {}

        # Check cache
        cache_key = self._cache_key(parent_node, query)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Build context path
        if trajectory is None:
            trajectory = parent_node.path_from_root()
        context_path = " → ".join(trajectory) if trajectory else "root"
        current_location = parent_node.tree_node.get(
            'title', parent_node.tree_node.get('name', 'root')
        )

        # Build pairs for cross-encoder mapping (query, item_text)
        pairs = []
        titles = []
        for child in children:
            node = child.tree_node
            title = node.get('title', node.get('name', 'unknown'))
            node_type = node.get('type', node.get('node_type', 'unknown'))
            summary = node.get('summary', '')
            
            doc = f"{title} ({node_type})"
            if node_type == 'folder':
                num_items = len(node.get('nodes', node.get('children', [])))
                doc += f" with {num_items} items"
            elif node_type.startswith('file_'):
                doc += f" (file)"
                
            if summary:
                doc += f": {summary}"
                
            pairs.append((query, doc))
            titles.append(title)

        try:
            import numpy as np
            raw_scores = self.scorer.predict(pairs)
            # ms-marco scores are logits, normalize to 0-1 via sigmoid
            normalized = 1 / (1 + np.exp(-raw_scores))
            
            scores = {t: float(s) for t, s in zip(titles, normalized)}
            self.total_llm_calls += 1  # Track as a scoring hit
        except Exception as e:
            print(f"⚠️ Cross-encoder scoring failed: {e}")
            scores = self._default_scores(children)

        # Cache result
        if len(self.cache) >= self.cache_size:
            # Evict oldest entry (FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[cache_key] = scores

        return scores

    def simulate_node(self, node: MCTSNode, query: str) -> float:
        """
        Score a single node's relevance to the query.

        Used for terminal nodes or when we need a quick estimate.

        Args:
            node: The MCTSNode to score.
            query: The user's query.

        Returns:
            Relevance score ∈ [0, 1].
        """
        summary = node.tree_node.get('summary', '')
        name = node.tree_node.get('title', node.tree_node.get('name', ''))
        node_type = node.tree_node.get('type', node.tree_node.get('node_type', ''))
        
        doc = f"{name} ({node_type}): {summary}" if summary else f"{name} ({node_type})"

        if not summary and not name:
            return 0.5

        import numpy as np
        try:
            score = self.scorer.predict([(query, doc)])[0]
            normalized = float(1 / (1 + np.exp(-score)))
            self.total_llm_calls += 1
            return normalized
        except Exception:
            return 0.5

    def _parse_scores(
        self, llm_response: str, children: List[MCTSNode]
    ) -> Dict[str, float]:
        """
        Parse LLM JSON response into a scores dictionary.

        Handles: markdown code blocks, extra text around JSON, invalid scores.
        Mirrors _parse_scores() from backend/code_index.py.
        """
        parsed = self._extract_json(llm_response)

        if not isinstance(parsed, dict):
            return self._default_scores(children)

        validated_scores = {}
        for child in children:
            title = child.tree_node.get('title', child.tree_node.get('name', 'unknown'))
            score = parsed.get(title, 0.5)

            try:
                score = float(score)
                score = max(0.0, min(1.0, score))
            except (ValueError, TypeError):
                score = 0.5

            validated_scores[title] = score

        return validated_scores

    def _extract_json(self, text: str) -> Any:
        """Extract JSON from LLM response, handling markdown blocks and extra text."""
        response_clean = text.strip()

        if not response_clean:
            return None

        # Remove markdown code blocks
        if response_clean.startswith("```"):
            lines = response_clean.split("\n")
            start_idx = 1
            end_idx = len(lines) - 1
            for i, line in enumerate(lines):
                if i > 0 and line.strip().startswith("```"):
                    end_idx = i
                    break
            response_clean = "\n".join(lines[start_idx:end_idx])

        response_clean = response_clean.strip()

        # Extract JSON object
        if '{' in response_clean and '}' in response_clean:
            start = response_clean.index('{')
            end = response_clean.rindex('}') + 1
            response_clean = response_clean[start:end]

        try:
            return json.loads(response_clean)
        except json.JSONDecodeError:
            return None

    def _default_scores(self, children: List[MCTSNode]) -> Dict[str, float]:
        """Return moderate default scores (0.5) when LLM scoring fails."""
        return {
            child.tree_node.get('title', child.tree_node.get('name', 'unknown')): 0.5
            for child in children
        }

    def _cache_key(self, parent_node: MCTSNode, query: str) -> str:
        """Generate a cache key from parent node ID and query."""
        node_id = parent_node.tree_node.get(
            'node_id',
            parent_node.tree_node.get('title', str(id(parent_node)))
        )
        raw = f"{node_id}::{query}"
        return hashlib.md5(raw.encode()).hexdigest()

    def reset_stats(self):
        """Reset LLM call counter (for benchmarking)."""
        self.total_llm_calls = 0

    def clear_cache(self):
        """Clear the score cache."""
        self.cache.clear()

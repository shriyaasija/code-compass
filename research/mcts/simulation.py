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
            llm_client: An OllamaLLM or LMStudioLLM instance with .chat() method.
            cache_size: Maximum number of cached score results.
        """
        self.llm = llm_client
        self.cache: Dict[str, Dict[str, float]] = {}
        self.cache_size = cache_size
        self.total_llm_calls: int = 0

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

        # Build scoring prompt (same format as code_index.py _score_siblings)
        prompt = f"""Query: "{query}"

Location: {context_path} → {current_location}

Rate relevance (0.0 to 1.0) for each item:
- 1.0 = Definitely needed to answer the query
- 0.7-0.9 = Likely relevant
- 0.4-0.6 = Possibly relevant
- 0.0-0.3 = Not relevant

Items:
"""

        for i, child in enumerate(children, 1):
            node = child.tree_node
            title = node.get('title', node.get('name', 'unknown'))
            node_type = node.get('type', node.get('node_type', 'unknown'))
            summary = node.get('summary', '')

            prompt += f"\n{i}. {title}"

            if node_type == 'folder':
                num_items = len(node.get('nodes', node.get('children', [])))
                prompt += f" (folder, {num_items} items)"
            elif node_type.startswith('file_'):
                prompt += f" ({node_type.replace('file_', '.')} file)"
            elif node_type in ('function', 'method'):
                start = node.get('start_line', '?')
                end = node.get('end_line', '?')
                prompt += f" (function, lines {start}-{end})"
            elif node_type == 'class':
                num_methods = len(node.get('nodes', node.get('children', [])))
                prompt += f" (class, {num_methods} methods)"

            if summary:
                short_summary = summary[:100] + "..." if len(summary) > 100 else summary
                prompt += f"\n   {short_summary}"

        prompt += """

Respond with ONLY a JSON object:
{"item_name": score, ...}

Example: {"auth.py": 0.9, "utils.py": 0.2}
"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a code search assistant. Rate the relevance of "
                        "code elements to answer user queries. Respond ONLY with valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            response = self.llm.chat(messages, temperature=0.1, max_tokens=500)
            self.total_llm_calls += 1

            if not response or not response.strip():
                scores = self._default_scores(children)
            else:
                scores = self._parse_scores(response, children)

        except Exception as e:
            print(f"⚠️ LLM scoring failed: {e}")
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

        if not summary and not name:
            return 0.5

        # For terminal nodes, we can use a simpler prompt
        prompt = f"""Query: "{query}"

Code element: {name}
Summary: {summary}

Rate the relevance of this code element to the query on a scale of 0.0 to 1.0.
Respond with ONLY a JSON object: {{"score": <number>}}
"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a code search assistant. Respond ONLY with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ]

            response = self.llm.chat(messages, temperature=0.1, max_tokens=100)
            self.total_llm_calls += 1

            if not response or not response.strip():
                return 0.5

            parsed = self._extract_json(response)
            if isinstance(parsed, dict) and 'score' in parsed:
                score = float(parsed['score'])
                return max(0.0, min(1.0, score))

            return 0.5

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

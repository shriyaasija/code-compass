import json
import math
from pathlib import Path
from typing import List, Dict, Any

class MCTSNode:
    def __init__(self, node_data, parent=None):
        self.data = node_data
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0.0
        self.expanded = False

    def ucb1(self, exploration=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.score / self.visits) + exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

class TreeBasedSearch:
    """
    Tree-based search that maintains compatibility with semantic search API.
    Uses LLM to score and traverse the code tree.
    """
    def __init__(self, llm_client, threshold: float = 0.5):
        self.llm = llm_client
        self.threshold = threshold
        self.repositories = {}  # repo_id -> {tree, json_path}
        
    def load_repository_tree(self, repo_id: str, json_tree_path: str):
        """
        Load PageIndex JSON tree for a repository.
        Compatible with PageIndexSemanticSearch API.
        """
        print(f"\n📂 Loading repository tree: {repo_id}")
        print(f"   From: {json_tree_path}")
        
        with open(json_tree_path, 'r') as f:
            tree = json.load(f)
        
        self.repositories[repo_id] = {
            'tree': tree,
            'json_path': json_tree_path
        }
        
        # Count nodes for statistics
        node_count = self._count_nodes(tree)
        print(f"✅ Loaded tree with ~{node_count} nodes")
        
    def _count_nodes(self, node: Dict, count: int = 0) -> int:
        """Recursively count all nodes in tree."""
        count += 1
        if 'nodes' in node:
            for child in node['nodes']:
                count = self._count_nodes(child, count)
        elif 'children' in node:
            for child in node['children']:
                count = self._count_nodes(child, count)
        return count

    def generate_tree_summaries(self, repo_id: str, repo_path: str):
        """Recursively generate short summaries for tree nodes that lack them."""
        if repo_id not in self.repositories:
            return
            
        tree = self.repositories[repo_id]['tree']
        print(f"🚀 Starting fast bottom-up LLM summarization for {repo_id}...")
        
        self._summarize_node_bottom_up(tree, repo_path)
        
        # Save back to disk
        json_path = self.repositories[repo_id]['json_path']
        with open(json_path, 'w') as f:
            json.dump(tree, f, indent=2)
        print(f"✅ Summarization complete. Updated JSON dumped to {json_path}")

    def _summarize_node_bottom_up(self, node: Dict, repo_path: str) -> str:
        # Process children
        children = node.get('nodes', node.get('children', []))
        child_summaries = []
        for child in children:
            c_sum = self._summarize_node_bottom_up(child, repo_path)
            if c_sum:
                title = child.get('title', child.get('name', 'unknown'))
                child_summaries.append(f"{title}: {c_sum}")

        # Skip if already has summary
        if node.get('summary', '').strip():
            return node['summary']

        node_type = node.get('type', node.get('node_type', ''))
        title = node.get('title', node.get('name', ''))
        
        prompt = ""
        # 1. Leaf node code fetching
        if node_type in ['function', 'method', 'class', 'struct', 'impl']:
            file_path = node.get('path', node.get('file_path', ''))
            start = node.get('start_line')
            end = node.get('end_line')
            code_snippet = ""
            if file_path and start is not None and end is not None:
                try:
                    import os
                    from pathlib import Path
                    full_path = str(Path(repo_path) / file_path) if not file_path.startswith('/') else file_path
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        # Usually line numbers are 0-indexed in our parser
                        snippet_lines = lines[max(0, start):end+1]
                        code_snippet = "".join(snippet_lines)[:800] # Cap length
                except Exception:
                    pass
            prompt = f"Summarize this {node_type} code in MAX 8 WORDS:\n{code_snippet}"
        
        # 2. Branch node fetching
        elif node_type.startswith('file') or node_type == 'folder':
            if not child_summaries:
                return "Empty"
            children_text = "\n".join(child_summaries)[:1000]
            prompt = f"Summarize this {node_type} named '{title}' which contains:\n{children_text}\nMAX 8 WORDS."
        else:
            return ""

        # Query LLM
        try:
            messages = [
                {"role": "system", "content": "You are a fast code summarizer. Output ONLY the short summary, nothing else. Be extremely brief (max 8 words). Do NOT use markdown."},
                {"role": "user", "content": prompt}
            ]
            response = self.llm.chat(messages, temperature=0.1, max_tokens=20)
            summary = response.strip().replace('\n', ' ')
            # Remove any quotes or weird chars
            if summary.startswith('"') and summary.endswith('"'):
                summary = summary[1:-1]
                
            node['summary'] = summary
            print(f"   [Summarized {title}] -> {summary}")
            return summary
        except Exception as e:
            print(f"   [Failed to summarize {title}]: {e}")
            return ""
    
    def search(self, 
               repo_id: str, 
               query: str, 
               top_k: int = None,
               min_similarity: float = None,
               node_type_filter: str = None) -> List[Dict[str, Any]]:
        """
        Perform tree-based search on repository.
        Returns ALL leaf nodes found during traversal (top_k is ignored).
        """
        if repo_id not in self.repositories:
            raise ValueError(f"Repository '{repo_id}' not loaded. Call load_repository_tree() first.")
        
        tree = self.repositories[repo_id]['tree']
        
        # Use min_similarity as threshold if provided, otherwise use default
        search_threshold = min_similarity if min_similarity is not None else self.threshold
        
        # Perform tree search
        results = []
        llm_call_count = [0]
        
        print(f"\n{'='*70}")
        print(f"🔍 TREE SEARCH: '{query}'")
        print(f"   Threshold: {search_threshold}")
        print(f"{'='*70}")
        
        self._recursive_search(tree, query, [], results, llm_call_count, search_threshold)
        
        print(f"\n{'='*70}")
        print(f"✅ SEARCH COMPLETE")
        print(f"   Found {len(results)} leaf nodes")
        print(f"   LLM calls: {llm_call_count[0]}")
        print(f"{'='*70}\n")
        
        # Apply node type filter if specified
        if node_type_filter:
            results = [r for r in results if r['node_type'] == node_type_filter]
        
        # Sort by score (descending) but return ALL results
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results
    
    def mcts_search(self, repo_id: str, query: str, n_simulations=15, threshold=None):
        if repo_id not in self.repositories:
            raise ValueError(f"Repository '{repo_id}' not loaded. Call load_repository_tree() first.")
        
        tree = self.repositories[repo_id]['tree']
        search_threshold = threshold if threshold is not None else self.threshold
        
        # PRE-COMPUTE: Bottom-up keyword score propagation
        query_words = set(w for w in query.lower().replace('?','').replace(',','').replace('.','').split() if len(w) > 2)
        self._propagate_keyword_scores(tree, query_words)
        
        root = MCTSNode(tree)
        
        print(f"\n{'='*70}")
        print(f"🔍 MCTS SEARCH: '{query}'")
        print(f"   Simulations: {n_simulations}, Threshold: {search_threshold}")
        print(f"{'='*70}")
        
        for _ in range(n_simulations):
            # Step 1: Selection - UCB1 traversal
            node = self._select(root)
            
            # Step 2: Expansion - load & score children via LLM
            if not node.expanded:
                self._expand(node, query)
            
            # Step 3: Simulation - keyword heuristic rollout, NO LLM
            reward = self._simulate(node, query)
            
            # Step 4: Backpropagation
            self._backpropagate(node, reward)
        
        results = self._collect_results(root, search_threshold)
        print(f"\n{'='*70}")
        print(f"✅ MCTS SEARCH COMPLETE")
        print(f"   Found {len(results)} leaf nodes")
        print(f"{'='*70}\n")
        return results

    def _select(self, node):
        while node.children:
            node = max(node.children, key=lambda n: n.ucb1())
        return node

    def _expand(self, node, query):
        children_data = node.data.get('nodes', node.data.get('children', []))
        if not children_data:
            node.expanded = True
            return
            
        trajectory = []
        curr = node
        while curr and curr.data:
            title = curr.data.get('title', curr.data.get('name', 'root'))
            trajectory.insert(0, title)
            curr = curr.parent

        scores = self._score_siblings(children_data, query, node.data, trajectory)
        
        for child_data in children_data:
            child = MCTSNode(child_data, parent=node)
            title = child_data.get('title', child_data.get('name', 'unknown'))
            # Give it an initial score from LLM guidance
            llm_score = scores.get(title, 0.0)
            keyword_score = child_data.get('_keyword_score', 0.0)
            
            # Blend: if a child deep down has strong keyword match, don't prune it!
            child.score = max(llm_score, keyword_score)
            
            child_data['_score'] = child.score
            node.children.append(child)
            
        node.expanded = True

    def _simulate(self, node, query):
        # The propagated keyword score is the maximum keyword match in the entire subtree!
        # This provides an O(1) rollout score of the best possible leaf in this branch.
        return node.data.get('_keyword_score', 0.0)
        
    def _propagate_keyword_scores(self, node: Dict, query_words: set) -> float:
        """Compute keyword score and propagate max child score bottom-up."""
        title = node.get('title', node.get('name', '')).lower()
        summary = node.get('summary', '').lower()
        text = f"{title} {summary}"
        
        score = 0.0
        if query_words:
            overlap = sum(1 for w in query_words if w in text)
            score = overlap / len(query_words)
            
        children = node.get('nodes', node.get('children', []))
        for child in children:
            child_score = self._propagate_keyword_scores(child, query_words)
            score = max(score, child_score)
            
        node['_keyword_score'] = score
        return score

    def _backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.score += reward
            node = node.parent

    def _collect_results(self, root, threshold):
        results = []
        def walk(node):
            if self._is_leaf(node.data) and getattr(node, 'score', 0) > threshold:
                # Add score to data so it's accessible downstream
                node.data['_score'] = node.score
                results.append(node.data)
            for child in node.children:
                walk(child)
        walk(root)
        
        # Optionally format slightly to keep retrieval happy if some attributes are missing
        formatted_results = []
        for res in results:
            formatted_results.append({
                'name': res.get('title', res.get('name', 'unnamed')),
                'node_type': res.get('type', 'unknown'),
                'file_path': res.get('path', ''),
                'path': res.get('path', ''),
                'start_line': res.get('start_line'),
                'end_line': res.get('end_line'),
                'docstring': res.get('summary', ''),
                '_score': res.get('_score', 0),
                'relevance_score': res.get('_score', 0)
            })
            
        return sorted(formatted_results, key=lambda x: x.get('_score', 0), reverse=True)
    
    def _is_leaf(self, node: Dict) -> bool:
        """Check if node is a leaf (has code location or is an unparsed file)."""
        node_type = node.get('type', node.get('node_type', ''))
        
        # AST elements are always leaves
        if node_type in ['function', 'method', 'class', 'struct', 'impl', 'module'] and 'start_line' in node:
            return True
            
        # Files without children (unparsed files like Dockerfile, README) can be leaves
        if node_type.startswith('file'):
            children = node.get('nodes', node.get('children', []))
            if not children:
                return True
                
        return False
    
    def _score_siblings(self, children: List[Dict], query: str, 
                       parent_node: Dict, trajectory: List[str]) -> Dict[str, float]:
        """Score all sibling nodes using LLM."""
        context_path = " → ".join(trajectory) if trajectory else "root"
        current_location = parent_node.get('title', parent_node.get('name', 'root'))
        
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
            title = child.get('title', child.get('name', 'unknown'))
            node_type = child.get('type', child.get('node_type', 'unknown'))
            summary = child.get('summary', '')
            
            prompt += f"\n{i}. {title}"
            
            if node_type == 'folder':
                num_items = len(child.get('nodes', child.get('children', [])))
                prompt += f" (folder, {num_items} items)"
            elif node_type.startswith('file_'):
                prompt += f" ({node_type.replace('file_', '.')} file)"
            elif node_type in ['function', 'method']:
                start = child.get('start_line', '?')
                end = child.get('end_line', '?')
                prompt += f" (function, lines {start}-{end})"
            elif node_type == 'class':
                num_methods = len(child.get('nodes', child.get('children', [])))
                prompt += f" (class, {num_methods} methods)"
            
            if summary:
                short_summary = summary[:100] + "..." if len(summary) > 100 else summary
                prompt += f"\n   {short_summary}"
                
            # Add algorithmic hint for LLM if branch contains match
            kw_score = child.get('_keyword_score', 0.0)
            if kw_score > 0.4:
                prompt += f" [HINT: Contains exact query match inside folder! Score it high!]"
        
        prompt += f"""

Respond with ONLY a JSON object:
{{"item_name": score, ...}}

Example: {{"auth.py": 0.9, "utils.py": 0.2}}
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a code search assistant. Rate the relevance of code elements to answer user queries. Respond ONLY with valid JSON."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm.chat(messages, temperature=0.1, max_tokens=500)
            
            # Debug: print what LLM returned
            if not response or not response.strip():
                print(f"⚠️ LLM returned empty response")
                print(f"   Prompt length: {len(prompt)} chars")
                # Return moderate scores as fallback
                return {child.get('title', child.get('name', 'unknown')): 0.5 for child in children}
            
            return self._parse_scores(response, children)
            
        except Exception as e:
            print(f"⚠️ LLM scoring failed: {e}")
            print(f"   Response was: {response[:200] if 'response' in locals() else 'No response'}")
            # Return moderate scores as fallback to continue search
            return {child.get('title', child.get('name', 'unknown')): 0.5 for child in children}
    
    def _parse_scores(self, llm_response: str, children: List[Dict]) -> Dict[str, float]:
        """Parse LLM response into scores dictionary."""
        try:
            response_clean = llm_response.strip()
            
            # Check if empty
            if not response_clean:
                print(f"⚠️ Empty LLM response")
                return {child.get('title', child.get('name', 'unknown')): 0.5 for child in children}
            
            # Remove markdown code blocks if present
            if response_clean.startswith("```"):
                lines = response_clean.split("\n")
                # Find first and last ``` markers
                start_idx = 1
                end_idx = len(lines) - 1
                for i, line in enumerate(lines):
                    if i > 0 and line.strip().startswith("```"):
                        end_idx = i
                        break
                response_clean = "\n".join(lines[start_idx:end_idx])
            
            # Try to find JSON in response
            response_clean = response_clean.strip()
            
            # Sometimes LLM adds extra text, try to extract JSON
            if '{' in response_clean and '}' in response_clean:
                start = response_clean.index('{')
                end = response_clean.rindex('}') + 1
                response_clean = response_clean[start:end]
            
            # Parse JSON
            scores_dict = json.loads(response_clean)
            
            # Validate and clamp scores
            validated_scores = {}
            for child in children:
                title = child.get('title', child.get('name', 'unknown'))
                score = scores_dict.get(title, 0.5)  # Default to 0.5 if not found
                
                try:
                    score = float(score)
                    score = max(0.0, min(1.0, score))
                except (ValueError, TypeError):
                    score = 0.5
                
                validated_scores[title] = score
            
            return validated_scores
            
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse LLM scores as JSON: {e}")
            print(f"   Response was: {llm_response[:300]}")
            # Return moderate scores (0.5) as fallback to continue exploring
            return {child.get('title', child.get('name', 'unknown')): 0.5 for child in children}
        except Exception as e:
            print(f"⚠️ Unexpected error parsing scores: {e}")
            return {child.get('title', child.get('name', 'unknown')): 0.5 for child in children}
    
    def search_and_format_for_chatbot(self,
                                      repo_id: str,
                                      query: str,
                                      top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search and format results for ProductionChatbot compatibility.
        Returns ALL leaf nodes found during tree traversal.
        
        Args:
            repo_id: Repository identifier
            query: User's natural language query
            top_k: Ignored - returns all leaf nodes found
        """
        search_results = self.search(repo_id, query, top_k=None)  # Get all results
        
        filtered_functions = []
        for result in search_results:
            meta = result['metadata']
            
            if 'file_path' in meta and 'start_line' in meta and 'end_line' in meta:
                filtered_functions.append({
                    'name': result['name'],
                    'signature': meta.get('signature', ''),
                    'file_path': meta['file_path'],
                    'start_line': int(meta['start_line']) if meta.get('start_line') is not None else None,
                    'end_line': int(meta['end_line']) if meta.get('end_line') is not None else None,
                    'docstring': meta.get('docstring', result['summary']),
                    'relevance_score': result['similarity_score']
                })
        
        return filtered_functions
    
    def list_loaded_repos(self) -> List[str]:
        """List all loaded repository IDs."""
        return list(self.repositories.keys())
    
    def get_repo_info(self, repo_id: str) -> Dict:
        """Get information about a loaded repository."""
        if repo_id not in self.repositories:
            return None
        
        repo = self.repositories[repo_id]
        tree = repo['tree']
        
        return {
            'repo_id': repo_id,
            'json_path': repo['json_path'],
            'total_nodes': self._count_nodes(tree),
            'node_types': {'tree_based': 'dynamic traversal'}
        }
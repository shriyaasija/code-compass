import json
from pathlib import Path
from typing import List, Dict, Any

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
    
    def _recursive_search(self, node: Dict, query: str, trajectory: List[str], 
                         results: List[Dict], llm_call_count: List[int], threshold: float):
        """Recursive tree traversal with LLM scoring - PageIndex style."""
        
        node_title = node.get('title', node.get('name', 'unknown'))
        node_type = node.get('type', 'unknown')
        depth = len(trajectory)
        indent = "  " * depth
        
        print(f"{indent}📂 [{node_type}] {node_title}")
        
        # Check if this is a leaf node (has code location)
        if self._is_leaf(node):
            score = node.get('_score', 0.8)
            print(f"{indent}  ✅ LEAF NODE (score: {score:.2f}) - COLLECTED")
            results.append({
                'node_id': node.get('node_id', node.get('title', 'unknown')),
                'name': node.get('title', node.get('name', 'unnamed')),
                'node_type': node_type,
                'summary': node.get('summary', ''),
                'path': node.get('path', ''),
                'similarity_score': score,
                'metadata': {
                    'file_path': node.get('path', ''),
                    'start_line': node.get('start_line'),
                    'end_line': node.get('end_line'),
                    'signature': node.get('signature', ''),
                    'docstring': node.get('summary', '')
                }
            })
            return
        
        # Get children
        children = node.get('nodes', node.get('children', []))
        if not children:
            print(f"{indent}  ⚠️  No children to explore")
            return
        
        print(f"{indent}  🎯 Scoring {len(children)} children...")
        
        # Score all siblings in one LLM call
        scores = self._score_siblings(children, query, node, trajectory)
        llm_call_count[0] += 1
        
        # Show scores and decide which to explore
        explored_count = 0
        for child in children:
            child_title = child.get('title', child.get('name', 'unknown'))
            child_score = scores.get(child_title, 0.0)
            
            if child_score >= threshold:
                print(f"{indent}    ✓ {child_title}: {child_score:.2f} → EXPLORE")
                explored_count += 1
                child['_score'] = child_score
                new_trajectory = trajectory + [node_title]
                self._recursive_search(child, query, new_trajectory, results, llm_call_count, threshold)
            else:
                print(f"{indent}    ✗ {child_title}: {child_score:.2f} → SKIP")
        
        if explored_count == 0:
            print(f"{indent}  ⛔ No children passed threshold - stopping here")
    
    def _is_leaf(self, node: Dict) -> bool:
        """Check if node is a leaf (has code location)."""
        node_type = node.get('type', node.get('node_type', ''))
        return (
            node_type in ['function', 'method', 'class', 'struct', 'impl', 'module']
            and 'start_line' in node 
            and 'end_line' in node
        )
    
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
                    'start_line': int(meta['start_line']),
                    'end_line': int(meta['end_line']),
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
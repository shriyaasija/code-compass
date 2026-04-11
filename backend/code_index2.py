import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class TreeBasedSearch:
    """
    Tree-based search that maintains compatibility with semantic search API.
    Uses LLM to score and traverse the code tree.
    """
    def __init__(self, llm_client, threshold: float = 0.5):
        self.llm = llm_client
        self.threshold = threshold
        self.repositories = {}  # repo_id -> {tree, json_path}
        # Load cross-encoder once (~80MB model, downloads on first use)
        print("🔄 Loading cross-encoder model for relevance scoring...")
        self.scorer = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("✅ Cross-encoder loaded")
        
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
        """Score all sibling nodes using cross-encoder — ~10x faster than LLM, no JSON parsing."""
        pairs = []
        titles = []
        for child in children:
            title = child.get('title', child.get('name', 'unknown'))
            summary = child.get('summary', '')
            node_type = child.get('type', child.get('node_type', ''))
            doc = f"{title} ({node_type}): {summary}" if summary else f"{title} ({node_type})"
            pairs.append((query, doc))
            titles.append(title)
        
        try:
            scores = self.scorer.predict(pairs)
            # ms-marco scores are logits, normalize to 0-1 via sigmoid
            normalized = 1 / (1 + np.exp(-scores))
            return {t: float(s) for t, s in zip(titles, normalized)}
        except Exception as e:
            print(f"⚠️ Cross-encoder scoring failed: {e}")
            # Fallback to moderate scores
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

from research.mcts.mcts_search import MCTSSearch

class MCTSTreeSearch:
    """
    MCTS-based search that maintains compatibility with semantic search API.
    """
    def __init__(self, llm_client, threshold: float = 0.5, max_iterations: int = 50, c_explore: float = 1.414):
        self.llm = llm_client
        self.threshold = threshold
        self.repositories = {}
        self.mcts = MCTSSearch(llm_client, max_iterations=max_iterations, c_explore=c_explore, verbose=False)
        
    def load_repository_tree(self, repo_id: str, json_tree_path: str):
        print(f"\n📂 Loading repository tree: {repo_id}")
        import json
        with open(json_tree_path, 'r') as f:
            tree = json.load(f)
        self.repositories[repo_id] = {'tree': tree, 'json_path': json_tree_path}
        
    def _count_nodes(self, node: dict, count: int = 0) -> int:
        count += 1
        if 'nodes' in node:
            for child in node['nodes']:
                count = self._count_nodes(child, count)
        elif 'children' in node:
            for child in node['children']:
                count = self._count_nodes(child, count)
        return count

    def search(self, repo_id: str, query: str, top_k: int = None, min_similarity: float = None, node_type_filter: str = None) -> list:
        if repo_id not in self.repositories:
            raise ValueError(f"Repository '{repo_id}' not loaded.")
        
        tree = self.repositories[repo_id]['tree']
        results = self.mcts.search(tree, query, top_k=top_k if top_k else 50)
        
        if node_type_filter:
            results = [r for r in results if r['node_type'] == node_type_filter]
            
        return results

    def search_and_format_for_chatbot(self, repo_id: str, query: str, top_k: int = None) -> list:
        search_results = self.search(repo_id, query, top_k=top_k)
        
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

    def list_loaded_repos(self) -> list:
        return list(self.repositories.keys())
    
    def get_repo_info(self, repo_id: str) -> dict:
        if repo_id not in self.repositories:
            return None
        repo = self.repositories[repo_id]
        return {
            'repo_id': repo_id,
            'json_path': repo['json_path'],
            'total_nodes': self._count_nodes(repo['tree']),
            'node_types': {'mcts_based': 'mcts traversal'}
        }

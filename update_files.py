import sys
import shutil
from pathlib import Path

# 1. Copy code_index.py to code_index2.py
shutil.copy('backend/code_index.py', 'backend/code_index2.py')

# 2. Append MCTSTreeSearch to code_index2.py
mcts_class = """
from research.mcts.mcts_search import MCTSSearch

class MCTSTreeSearch:
    \"\"\"
    MCTS-based search that maintains compatibility with semantic search API.
    \"\"\"
    def __init__(self, llm_client, threshold: float = 0.5, max_iterations: int = 50, c_explore: float = 1.414):
        self.llm = llm_client
        self.threshold = threshold
        self.repositories = {}
        self.mcts = MCTSSearch(llm_client, max_iterations=max_iterations, c_explore=c_explore, verbose=False)
        
    def load_repository_tree(self, repo_id: str, json_tree_path: str):
        print(f"\\n📂 Loading repository tree: {repo_id}")
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
"""
with open('backend/code_index2.py', 'a') as f:
    f.write("\n" + mcts_class)

# 3. We also need to add run_mcts_search into benchmark.py, but it's complex to inject. Let's do it via multi_replace_file_content tool after this script runs.

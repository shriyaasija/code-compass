import json
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pathlib import Path


class PageIndexSemanticSearch:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        print(f"🔄 Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"✅ Model loaded (dimension: {self.embedding_model.get_sentence_embedding_dimension()})")

        # Storage for loaded repositories
        self.repositories = {}  # repo_id -> {tree, nodes}

    def load_repository_tree(self, repo_id: str, json_tree_path: str):
        """
        Load PageIndex JSON tree for a repository.

        Args:
            repo_id: Unique identifier for repository
            json_tree_path: Path to JSON file with embedded tree structure
        """
        print(f"\n📂 Loading repository tree: {repo_id}")
        print(f"   From: {json_tree_path}")

        # Load JSON
        with open(json_tree_path, 'r') as f:
            tree = json.load(f)

        # Flatten tree into searchable nodes
        nodes = self._flatten_tree(tree)

        # Store
        self.repositories[repo_id] = {
            'tree': tree,
            'nodes': nodes,
            'json_path': json_tree_path
        }

        print(f"Loaded {len(nodes)} nodes from repository")

        # Print statistics
        node_types = {}
        for node in nodes:
            node_type = node['node_type']
            node_types[node_type] = node_types.get(node_type, 0) + 1

        print(f"\nNode Distribution:")
        for ntype, count in sorted(node_types.items()):
            print(f"   - {ntype}: {count}")

    def _flatten_tree(self, node: Dict, parent_path: str = "", nodes_list: List = None) -> List[Dict]:
        """
        Recursively flatten hierarchical tree into flat list of nodes.

        Each node must have:
        - summary: text description
        - embedding: vector representation (optional, will generate if missing)
        - metadata: file_path, start_line, end_line, etc.
        """
        if nodes_list is None:
            nodes_list = []

        # Only process nodes with summaries
        if 'summary' in node or 'repository_summary' in node:
            node_info = {
                'node_id': node.get('node_id', node.get('name', 'unknown')),
                'node_type': node.get('node_type', node.get('type', 'unknown')),
                'name': node.get('name', node.get('file_name', node.get('repository_name', 'unnamed'))),
                'summary': node.get('summary', node.get('repository_summary', '')),
                'path': parent_path,
                'metadata': {}
            }

            # Get or generate embedding
            if 'embedding' in node:
                node_info['embedding'] = np.array(node['embedding'])
            else:
                # Generate embedding from summary if not present
                print(f"Generating embedding for {node_info['name']}")
                node_info['embedding'] = self.embedding_model.encode(node_info['summary'])

            # Extract metadata
            if 'file_path' in node:
                node_info['metadata']['file_path'] = node['file_path']
            if 'signature' in node:
                node_info['metadata']['signature'] = node['signature']
            if 'start_line' in node:
                node_info['metadata']['start_line'] = node['start_line']
            if 'end_line' in node:
                node_info['metadata']['end_line'] = node['end_line']
            if 'docstring' in node:
                node_info['metadata']['docstring'] = node['docstring']

            nodes_list.append(node_info)

        # Recursively process children
        if 'children' in node:
            current_name = node.get('name', node.get('file_name', ''))
            current_path = f"{parent_path}/{current_name}" if parent_path else current_name

            for child in node['children']:
                self._flatten_tree(child, current_path, nodes_list)

        return nodes_list

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Returns value between -1 and 1 (1 = identical)
        """
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def search(self, 
               repo_id: str, 
               query: str, 
               top_k: int = 5,
               min_similarity: float = 0.0,
               node_type_filter: str = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search on repository.

        Args:
            repo_id: Repository identifier
            query: User's natural language query
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold
            node_type_filter: Filter by node type (e.g., 'function', 'class')

        Returns:
            List of matched nodes formatted for ProductionChatbot
        """
        # Check if repository is loaded
        if repo_id not in self.repositories:
            raise ValueError(f"Repository '{repo_id}' not loaded. Call load_repository_tree() first.")

        nodes = self.repositories[repo_id]['nodes']

        # Step 1: Generate query embedding
        query_embedding = self.embedding_model.encode(query)

        # Step 2: Calculate similarities with all nodes
        results = []

        for node in nodes:
            # Apply node type filter if specified
            if node_type_filter and node['node_type'] != node_type_filter:
                continue

            # Calculate similarity
            similarity = self.cosine_similarity(query_embedding, node['embedding'])

            # Filter by threshold
            if similarity < min_similarity:
                continue

            # Format result
            result = {
                'node_id': node['node_id'],
                'name': node['name'],
                'node_type': node['node_type'],
                'summary': node['summary'],
                'path': node['path'],
                'similarity_score': float(similarity),
                'metadata': node['metadata']
            }

            results.append(result)

        # Step 3: Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)

        # Step 4: Return top-k
        return results[:top_k]

    def search_and_format_for_chatbot(self,
                                      repo_id: str,
                                      query: str,
                                      top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search and format results for ProductionChatbot compatibility.

        Returns list in format expected by retrieval.py:
        [
            {
                'name': str,
                'signature': str,
                'file_path': str,
                'start_line': int,
                'end_line': int,
                'docstring': str
            }
        ]
        """
        # Perform search
        search_results = self.search(repo_id, query, top_k)

        # Format for chatbot
        filtered_functions = []

        for result in search_results:
            meta = result['metadata']

            # Only include nodes with code location (file_path + line numbers)
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
        return {
            'repo_id': repo_id,
            'json_path': repo['json_path'],
            'total_nodes': len(repo['nodes']),
            'node_types': self._count_node_types(repo['nodes'])
        }

    def _count_node_types(self, nodes: List[Dict]) -> Dict[str, int]:
        """Count nodes by type."""
        counts = {}
        for node in nodes:
            ntype = node['node_type']
            counts[ntype] = counts.get(ntype, 0) + 1
        return counts
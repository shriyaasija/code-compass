"""
PageIndex Semantic Search - CORRECT IMPLEMENTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

How it works:
1. Load JSON tree
2. Flatten tree into list of nodes (each has summary + embedding)
3. Search: Compare query embedding with each node's SUMMARY embedding
4. Return top-k most similar nodes

IMPORTANT: We ONLY search using summary embeddings, nothing else
"""

import json
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer


class PageIndexSemanticSearch:
    """Semantic search on PageIndex JSON tree"""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        print(f"🔄 Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded (dimension: {self.embedding_dim})")

        self.repositories = {}  # repo_id -> {tree, nodes}

    def load_repository_tree(self, repo_id: str, json_tree_path: str):
        """
        Load PageIndex JSON tree and flatten it into searchable nodes

        Args:
            repo_id: Unique identifier for this repository
            json_tree_path: Path to the PageIndex JSON file
        """
        print(f"\n📂 Loading repository: {repo_id}")
        print(f"   From: {json_tree_path}")

        # Load JSON
        with open(json_tree_path, 'r') as f:
            tree = json.load(f)

        # Flatten tree: Walk through all nodes and extract those with embeddings
        nodes = []
        self._extract_nodes(tree, nodes)

        print(f"✅ Extracted {len(nodes)} searchable nodes")

        # Store
        self.repositories[repo_id] = {
            'tree': tree,
            'nodes': nodes,
            'json_path': json_tree_path
        }

    def _extract_nodes(self, node: Dict, nodes: List, parent_path: str = ""):
        """
        Recursively walk tree and extract nodes with embeddings

        Process:
        1. Check if node has 'summary' field (or 'repository_summary')
        2. Check if node has 'embedding' field
        3. If both exist, extract it as a searchable node
        4. Recursively process children

        Args:
            node: Current node in tree
            nodes: List to append extracted nodes to
            parent_path: Path from root to this node
        """

        # Check if this node is searchable (has summary AND embedding)
        has_summary = 'summary' in node or 'repository_summary' in node
        has_embedding = 'embedding' in node

        if has_summary and has_embedding:
            # Extract node information
            node_info = {
                'node_id': node.get('node_id', f"node_{len(nodes)}"),
                'node_type': node.get('node_type', 'unknown'),
                'name': node.get('name', node.get('file_name', node.get('repository_name', 'unnamed'))),
                'summary': node.get('summary', node.get('repository_summary', '')),
                'embedding': node['embedding'],  # The 384-dimensional vector
                'path': parent_path,
                'metadata': {}
            }

            # Extract metadata (for code retrieval later)
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

            # Add to searchable nodes list
            nodes.append(node_info)

        # Recursively process children
        if 'children' in node:
            # Build path for children
            current_name = node.get('name', node.get('file_name', ''))
            current_path = f"{parent_path}/{current_name}" if parent_path else current_name

            for child in node['children']:
                self._extract_nodes(child, nodes, current_path)

    def search(self, repo_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search: Find nodes whose SUMMARY embeddings are similar to query

        Process:
        1. Convert query to embedding using SentenceTransformer
        2. Compare query embedding with EACH node's summary embedding using cosine similarity
        3. Sort by similarity (highest first)
        4. Return top-k results

        Args:
            repo_id: Which repository to search
            query: User's search query
            top_k: How many results to return

        Returns:
            List of top-k matching nodes with similarity scores
        """
        if repo_id not in self.repositories:
            raise ValueError(f"Repository '{repo_id}' not loaded. Call load_repository_tree() first.")

        nodes = self.repositories[repo_id]['nodes']

        print(f"\n🔍 Searching {len(nodes)} nodes for: '{query}'")

        # STEP 1: Convert query to embedding
        query_embedding = self.embedding_model.encode(query)
        print(f"   Generated query embedding: shape={query_embedding.shape}")

        # STEP 2: Calculate cosine similarity with each node's embedding
        results = []
        for node in nodes:
            node_embedding = np.array(node['embedding'])

            # Cosine similarity formula: cos(θ) = (A·B) / (||A|| × ||B||)
            dot_product = np.dot(query_embedding, node_embedding)
            norm_query = np.linalg.norm(query_embedding)
            norm_node = np.linalg.norm(node_embedding)

            if norm_query == 0 or norm_node == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm_query * norm_node)

            results.append({
                'node': node,
                'similarity': float(similarity)
            })

        # STEP 3: Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)

        print(f"   Top result: {results[0]['node']['name']} (similarity: {results[0]['similarity']:.3f})")

        # STEP 4: Format and return top-k
        top_results = []
        for r in results[:top_k]:
            result = {
                'node_id': r['node']['node_id'],
                'name': r['node']['name'],
                'node_type': r['node']['node_type'],
                'summary': r['node']['summary'],
                'path': r['node']['path'],
                'similarity_score': r['similarity'],
                'metadata': r['node']['metadata']
            }
            top_results.append(result)

        print(f"✅ Returning {len(top_results)} results")
        return top_results

    def search_and_format_for_chatbot(self, repo_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search and format results for ProductionChatbot

        Returns results in format expected by chatbot:
        - name: function/class name
        - signature: code signature
        - file_path: where the code is
        - start_line, end_line: line numbers
        - docstring: description (for display, NOT search)
        - relevance_score: similarity score
        """
        # Do semantic search
        search_results = self.search(repo_id, query, top_k)

        # Format for chatbot
        formatted_results = []
        for result in search_results:
            meta = result['metadata']

            # Build formatted result
            formatted = {
                'name': result['name'],
                'signature': meta.get('signature', f"# {result['node_type']}: {result['name']}"),
                'file_path': meta.get('file_path', result['path']),
                'start_line': meta.get('start_line', 0),
                'end_line': meta.get('end_line', 0),
                'docstring': meta.get('docstring', result['summary']),  # Use summary as fallback
                'relevance_score': result['similarity_score']
            }

            formatted_results.append(formatted)

        return formatted_results

    def list_loaded_repos(self) -> List[str]:
        """Return list of loaded repository IDs"""
        return list(self.repositories.keys())

    def get_repo_info(self, repo_id: str) -> Dict:
        """Get statistics about a loaded repository"""
        if repo_id not in self.repositories:
            return None

        repo = self.repositories[repo_id]
        nodes = repo['nodes']

        # Count node types
        node_types = {}
        for node in nodes:
            ntype = node['node_type']
            node_types[ntype] = node_types.get(ntype, 0) + 1

        return {
            'repo_id': repo_id,
            'json_path': repo['json_path'],
            'total_nodes': len(nodes),
            'node_types': node_types
        }

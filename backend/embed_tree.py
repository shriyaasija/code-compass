"""
Embed all node summaries in a tree using SentenceTransformer.

After TreeSummarizer generates summaries, this adds embedding vectors
to every node so the tree is ready for MCTS + RL.
"""

import time
from typing import Dict
from sentence_transformers import SentenceTransformer


class TreeEmbedder:
    """Adds embedding vectors to all nodes with summaries."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"🔄 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded (dim={self.dim})")
    
    def embed_tree(self, tree: Dict) -> Dict:
        """Add embeddings to all nodes with summaries. Modifies in-place."""
        count = 0
        start = time.time()
        
        def _walk(node):
            nonlocal count
            summary = node.get('summary', '')
            if summary:
                node['embedding'] = self.model.encode(summary).tolist()
                count += 1
            
            children = node.get('nodes', node.get('children', []))
            for child in children:
                _walk(child)
        
        _walk(tree)
        elapsed = time.time() - start
        print(f"✅ Embedded {count} nodes in {elapsed:.1f}s")
        return tree
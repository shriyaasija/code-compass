
"""
Test semantic search with mock data
Run this to see if the problem is in the semantic search or API
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

print("Testing Semantic Search...")
print("="*70)

# Load the mock JSON
print("\n1. Loading mock_pageindex_tree.json...")
try:
    with open('mock_pageindex_tree.json', 'r') as f:
        tree = json.load(f)
    print(f"✅ Loaded JSON")
except Exception as e:
    print(f"❌ Failed to load JSON: {e}")
    exit(1)

# Check if embeddings exist
print("\n2. Checking for embeddings...")
has_embedding = 'embedding' in tree
print(f"Root has embedding: {has_embedding}")
if has_embedding:
    embedding = tree['embedding']
    print(f"Embedding type: {type(embedding)}")
    print(f"Embedding length: {len(embedding) if isinstance(embedding, list) else 'N/A'}")
    print(f"First 5 values: {embedding[:5] if isinstance(embedding, list) else 'N/A'}")

# Count nodes with embeddings
def count_nodes_with_embeddings(node, count=0):
    if 'embedding' in node:
        count += 1
    if 'children' in node:
        for child in node['children']:
            count = count_nodes_with_embeddings(child, count)
    return count

total_with_emb = count_nodes_with_embeddings(tree)
print(f"\nNodes with embeddings: {total_with_emb}")

# Initialize sentence transformer
print("\n3. Loading SentenceTransformer...")
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# Generate query embedding
print("\n4. Generating query embedding...")
query = "How do I train the model?"
query_embedding = model.encode(query)
print(f"Query: {query}")
print(f"Query embedding shape: {query_embedding.shape}")
print(f"First 5 values: {query_embedding[:5]}")

# Test cosine similarity with first node
print("\n5. Testing cosine similarity...")
if has_embedding:
    node_emb = np.array(tree['embedding'])
    query_emb = query_embedding

    # Calculate cosine similarity
    dot_product = np.dot(query_emb, node_emb)
    norm_query = np.linalg.norm(query_emb)
    norm_node = np.linalg.norm(node_emb)
    similarity = dot_product / (norm_query * norm_node)

    print(f"Similarity with root node: {similarity:.4f}")

    if similarity < 0.5:
        print("⚠️  WARNING: Low similarity! This might be why nothing is found.")
    else:
        print("✅ Similarity looks reasonable")

# Flatten tree and check all nodes
print("\n6. Checking all nodes...")
def flatten_tree(node, nodes=[]):
    if 'summary' in node or 'repository_summary' in node:
        node_info = {
            'name': node.get('name', node.get('file_name', 'unknown')),
            'summary': node.get('summary', node.get('repository_summary', '')),
            'has_embedding': 'embedding' in node
        }
        nodes.append(node_info)

    if 'children' in node:
        for child in node['children']:
            flatten_tree(child, nodes)

    return nodes

nodes = flatten_tree(tree, [])
print(f"Total nodes found: {len(nodes)}")
print(f"Nodes with embeddings: {sum(1 for n in nodes if n['has_embedding'])}")

print("\nFirst 5 nodes:")
for i, node in enumerate(nodes[:5], 1):
    print(f"{i}. {node['name']}: has_embedding={node['has_embedding']}")

print("\n" + "="*70)
print("DIAGNOSIS:")
if total_with_emb == 0:
    print("❌ NO EMBEDDINGS FOUND IN JSON!")
    print("   The mock_pageindex_tree.json doesn't have embeddings.")
    print("   You need to regenerate it with embeddings.")
elif total_with_emb < len(nodes):
    print(f"⚠️  PARTIAL EMBEDDINGS: {total_with_emb}/{len(nodes)} nodes have embeddings")
    print("   Some nodes are missing embeddings.")
else:
    print(f"✅ All {total_with_emb} nodes have embeddings")
    print("   The issue might be in the semantic search logic.")

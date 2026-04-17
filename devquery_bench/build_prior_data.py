"""
Build prior training data from DevQuery-Bench annotations.

For each (query, ground_truth_function) pair in the TRAINING repos:
1. Find the path from tree root to the ground-truth function
2. At each level of the path, create:
   - Positive pair: (query_emb, on_path_node_emb, label=1)
   - Negative pairs: (query_emb, each_sibling_emb, label=0)

This teaches the prior to navigate the tree correctly.
"""
import json
import os
import sys
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_path_to_node(root: Dict, target_title: str) -> Optional[List[Dict]]:
    """DFS to find path from root to node with matching title."""
    def dfs(node, path):
        current_title = node.get('title', node.get('name', ''))
        if current_title == target_title:
            return path + [node]
        for child in node.get('nodes', node.get('children', [])):
            result = dfs(child, path + [node])
            if result is not None:
                return result
        return None
    return dfs(root, [])


def get_embedding(node: Dict) -> Optional[np.ndarray]:
    """Extract embedding from node."""
    emb = node.get('embedding')
    if emb is None:
        return None
    arr = np.array(emb, dtype=np.float32)
    if arr.ndim != 1 or len(arr) < 10:
        return None
    return arr


def main():
    print("=" * 60)
    print("BUILDING PRIOR TRAINING DATA (DevQuery-Bench)")
    print("=" * 60)

    # Load data
    with open('devquery_bench/train_test_split.json') as f:
        split = json.load(f)
    train_repos = set(split['train'])

    with open('devquery_bench/devquery_bench.json') as f:
        bench = json.load(f)

    # Filter to training repos only
    train_entries = [e for e in bench if e['repo_id'] in train_repos]
    print(f"\nTraining entries: {len(train_entries)} (from {len(train_repos)} repos)")
    print(f"Test entries: {len(bench) - len(train_entries)} (held out)")

    # Load embedding model
    print(f"\nLoading embedding model...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embed_dim = embed_model.get_sentence_embedding_dimension()
    print(f"Embedding dim: {embed_dim}")

    all_query_embs = []
    all_node_embs = []
    all_labels = []

    used = 0
    skipped = 0

    # Cache loaded trees
    tree_cache = {}

    for entry in train_entries:
        repo_id = entry['repo_id']
        query = entry['query']
        ground_truth = entry['ground_truth']

        # Load tree (cached)
        if repo_id not in tree_cache:
            tree_path = f"devquery_bench/trees/{repo_id}.json"
            if not os.path.exists(tree_path):
                skipped += 1
                continue
            with open(tree_path) as f:
                tree_cache[repo_id] = json.load(f)

        tree = tree_cache[repo_id]

        # Embed the query once
        query_emb = embed_model.encode(query, show_progress_bar=False)
        found_any = False

        # Iterate over all ground truth targets
        for target_title in ground_truth:
            # Find path from root to this specific ground truth
            path = find_path_to_node(tree, target_title)
            if path is None or len(path) < 2:
                continue
            
            found_any = True

            # For each level in the path, create training pairs
            for i in range(1, len(path)):
                on_path_node = path[i]
                parent = path[i - 1]
                siblings = parent.get('nodes', parent.get('children', []))

                if not siblings:
                    continue

                on_path_title = on_path_node.get('title', '')

                for sibling in siblings:
                    node_emb = get_embedding(sibling)
                    if node_emb is None:
                        continue

                    sibling_title = sibling.get('title', '')
                    label = 1.0 if sibling_title == on_path_title else 0.0

                    all_query_embs.append(query_emb)
                    all_node_embs.append(node_emb)
                    all_labels.append(label)

        if found_any:
            used += 1
        else:
            skipped += 1

    if not all_query_embs:
        print("\n❌ No training pairs generated!")
        print("   Check that trees have 'embedding' fields.")
        return

    print(f"\nResults:")
    print(f"  Used queries:    {used}")
    print(f"  Skipped queries: {skipped}")
    print(f"  Total pairs:     {len(all_labels)}")
    print(f"  Positives:       {sum(1 for l in all_labels if l == 1.0)}")
    print(f"  Negatives:       {sum(1 for l in all_labels if l == 0.0)}")
    pos_rate = sum(1 for l in all_labels if l == 1.0) / len(all_labels) * 100
    print(f"  Positive rate:   {pos_rate:.1f}%")

    # Save as tensors
    data = {
        'query_embs': torch.tensor(np.array(all_query_embs), dtype=torch.float32),
        'node_embs': torch.tensor(np.array(all_node_embs), dtype=torch.float32),
        'labels': torch.tensor(all_labels, dtype=torch.float32),
        'embed_dim': embed_dim,
    }

    output_path = 'devquery_bench/prior_training_data.pt'
    torch.save(data, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Saved to: {output_path} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
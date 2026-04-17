"""
Build prior training data from DevQuery-Bench annotations.

Simplified data format — only stores raw embeddings. The model computes
its own projected features internally (bottleneck architecture).

Includes ALL ground-truth paths (no ambiguity filtering) to maximise
training signal. Adds repo_ids for repo-based validation splitting.

For each (query, ground_truth_function) pair in TRAINING repos:
  1. Find path from tree root to ground-truth function
  2. At each level of the path, create:
     - Positive pair: (query_emb, on_path_node_emb, label=1)
     - Negative pairs: (query_emb, each_sibling_emb, label=0)
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

    with open('devquery_bench/train_test_split.json') as f:
        split = json.load(f)
    train_repos = set(split['train'])

    with open('devquery_bench/devquery_bench.json') as f:
        bench = json.load(f)

    train_entries = [e for e in bench if e['repo_id'] in train_repos]
    print(f"\nTraining entries: {len(train_entries)} (from {len(train_repos)} repos)")
    print(f"Test entries: {len(bench) - len(train_entries)} (held out)")

    print(f"\nLoading embedding model...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embed_dim = embed_model.get_sentence_embedding_dimension()
    print(f"Embedding dim: {embed_dim}")

    all_query_embs  = []
    all_node_embs   = []
    all_labels      = []
    all_query_ids   = []  # group id per (query, target, level)
    all_repo_ids    = []  # repo int id for repo-based val split

    used       = 0
    skipped    = 0
    query_id   = 0

    tree_cache   = {}
    repo_to_int  = {}
    next_repo_id = 0

    for entry in train_entries:
        repo_id      = entry['repo_id']
        query        = entry['query']
        ground_truth = entry['ground_truth']

        # Normalise: ground_truth is always a list, but guard anyway
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]

        # Assign integer id to repo
        if repo_id not in repo_to_int:
            repo_to_int[repo_id] = next_repo_id
            next_repo_id += 1
        repo_int = repo_to_int[repo_id]

        if repo_id not in tree_cache:
            tree_path = f"devquery_bench/trees/{repo_id}.json"
            if not os.path.exists(tree_path):
                skipped += 1
                continue
            with open(tree_path) as f:
                tree_cache[repo_id] = json.load(f)

        tree = tree_cache[repo_id]

        query_emb = embed_model.encode(query, show_progress_bar=False)
        found_any_path = False

        # Include ALL ground truth paths (no ambiguity filtering)
        for target_title in ground_truth:
            path = find_path_to_node(tree, target_title)
            if path is None or len(path) < 2:
                continue

            found_any_path = True

            for i in range(1, len(path)):
                on_path_node = path[i]
                parent       = path[i - 1]
                siblings     = parent.get('nodes', parent.get('children', []))

                if not siblings:
                    continue

                on_path_title = on_path_node.get('title', '')

                level_pairs  = []
                has_positive = False

                for sibling in siblings:
                    node_emb = get_embedding(sibling)
                    if node_emb is None:
                        continue

                    sibling_title = sibling.get('title', '')
                    label = 1.0 if sibling_title == on_path_title else 0.0
                    if label == 1.0:
                        has_positive = True

                    level_pairs.append((query_emb, node_emb, label))

                if not has_positive or len(level_pairs) < 2:
                    continue

                for qe, ne, label in level_pairs:
                    all_query_embs.append(qe)
                    all_node_embs.append(ne)
                    all_labels.append(label)
                    all_query_ids.append(query_id)
                    all_repo_ids.append(repo_int)

                query_id += 1

        if found_any_path:
            used += 1
        else:
            skipped += 1

    if not all_query_embs:
        print("\n❌ No training pairs generated!")
        print("   Check that trees have 'embedding' fields.")
        return

    n_pos = sum(1 for l in all_labels if l == 1.0)
    n_neg = sum(1 for l in all_labels if l == 0.0)
    pos_rate = n_pos / len(all_labels) * 100

    print(f"\nResults:")
    print(f"  Used queries:    {used}")
    print(f"  Skipped queries: {skipped}")
    print(f"  Total pairs:     {len(all_labels)}")
    print(f"  Positives:       {n_pos}")
    print(f"  Negatives:       {n_neg}")
    print(f"  Positive rate:   {pos_rate:.1f}%")
    print(f"  Query groups:    {query_id}")
    print(f"  Repos:           {list(repo_to_int.keys())}")

    data = {
        'query_embs':  torch.tensor(np.array(all_query_embs),  dtype=torch.float32),
        'node_embs':   torch.tensor(np.array(all_node_embs),   dtype=torch.float32),
        'labels':      torch.tensor(all_labels,                dtype=torch.float32),
        'query_ids':   torch.tensor(all_query_ids,             dtype=torch.long),
        'repo_ids':    torch.tensor(all_repo_ids,              dtype=torch.long),
        'embed_dim':   embed_dim,
        'n_groups':    query_id,
        'repo_names':  list(repo_to_int.keys()),
    }

    output_path = 'devquery_bench/prior_training_data.pt'
    torch.save(data, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Saved to: {output_path} ({size_mb:.1f} MB)")
    print(f"   Data format: query_embs + node_embs + labels + query_ids + repo_ids")


if __name__ == '__main__':
    main()
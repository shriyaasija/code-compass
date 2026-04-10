import json
import os
import glob
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ─── Helper: find the path from root to a named leaf ──────────────────────────

def find_path_to_node(root: Dict, target_title: str) -> Optional[List[Dict]]:
    """
    DFS to find the path from root to the node whose title matches target_title.
    Returns list of nodes [root, ..., target] or None if not found.
    """
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


def collect_siblings_at_each_level(path: List[Dict], root: Dict) -> List[Tuple[Dict, List[Dict]]]:
    """
    For each node in the path (except root), collect all siblings.
    Returns list of (on_path_node, [all siblings including on_path_node]).
    """
    result = []
    for i in range(1, len(path)):
        on_path_node = path[i]
        parent = path[i - 1]
        siblings = parent.get('nodes', parent.get('children', []))
        if siblings:
            result.append((on_path_node, siblings))
    return result


def get_embedding(node: Dict) -> Optional[np.ndarray]:
    """Get pre-computed embedding from node, or None."""
    emb = node.get('embedding')
    if emb is None:
        return None
    arr = np.array(emb, dtype=np.float32)
    # Some embeddings might have wrong shape — skip them
    if arr.ndim != 1 or len(arr) < 10:
        return None
    return arr


# ─── Main builder ─────────────────────────────────────────────────────────────

def build_training_data(
    train_repo_ids: List[str],
    metadata_list: List[Dict],
    proper_trees_dir: str = "benchmark_results/proper_trees",
    output_path: str = "benchmark_results/prior_training_data.pt",
    embed_model_name: str = "all-MiniLM-L6-v2",
):
    print("=" * 60)
    print("BUILDING PRIOR TRAINING DATA")
    print("=" * 60)

    print(f"\nLoading embedding model: {embed_model_name}")
    embed_model = SentenceTransformer(embed_model_name)
    embed_dim = embed_model.get_sentence_embedding_dimension()
    print(f"Embedding dim: {embed_dim}")

    all_query_embs = []
    all_node_embs = []
    all_labels = []

    skipped_queries = 0
    used_queries = 0

    for repo_id in train_repo_ids:
        tree_path = os.path.join(proper_trees_dir, f"{repo_id}.json")
        if not os.path.exists(tree_path):
            print(f"  Skipping {repo_id}: tree not found")
            continue

        # Load tree
        with open(tree_path) as f:
            tree = json.load(f)

        # Find metadata for this repo
        meta = next((m for m in metadata_list if m['repo_id'] == repo_id), None)
        if not meta:
            print(f"  Skipping {repo_id}: no metadata")
            continue

        queries = meta.get('queries', [])
        if not queries:
            print(f"  Skipping {repo_id}: no queries")
            continue

        repo_pairs = 0

        for q_info in queries:
            query_text = q_info.get('query', q_info.get('docstring', ''))
            ground_truth = q_info.get('ground_truth', q_info.get('func_name', ''))

            if not query_text or not ground_truth:
                skipped_queries += 1
                continue

            # Find path from root to the correct function
            path = find_path_to_node(tree, ground_truth)
            if path is None or len(path) < 2:
                # Ground truth function not found in tree — skip
                skipped_queries += 1
                continue

            # Embed the query
            query_emb = embed_model.encode(query_text, show_progress_bar=False)

            # For each level in the path, create training pairs:
            # - The on-path node: label 1.0
            # - All its siblings: label 0.0
            level_pairs = collect_siblings_at_each_level(path, tree)

            for on_path_node, siblings in level_pairs:
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
                    repo_pairs += 1

            used_queries += 1

        print(f"  {repo_id}: {used_queries} queries, {repo_pairs} pairs generated")

    if not all_query_embs:
        print("\nERROR: No training pairs generated!")
        print("Check that your trees have 'embedding' fields and metadata has 'queries'.")
        return

    print(f"\nTotal pairs: {len(all_labels)}")
    print(f"Positive (on-path): {sum(1 for l in all_labels if l == 1.0)}")
    print(f"Negative (off-path): {sum(1 for l in all_labels if l == 0.0)}")
    print(f"Skipped queries: {skipped_queries}")

    # Convert to tensors and save
    data = {
        'query_embs': torch.tensor(np.array(all_query_embs), dtype=torch.float32),
        'node_embs': torch.tensor(np.array(all_node_embs), dtype=torch.float32),
        'labels': torch.tensor(all_labels, dtype=torch.float32),
        'embed_dim': embed_dim,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)
    print(f"\nSaved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    return data


if __name__ == "__main__":
    # Load split and metadata
    split = json.load(open('benchmark_results/train_test_split.json'))
    metadata_list = json.load(open('benchmark_results/proper_benchmark_metadata.json'))

    build_training_data(
        train_repo_ids=split['train'],
        metadata_list=metadata_list,
    )

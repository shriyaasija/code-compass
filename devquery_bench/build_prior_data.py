"""
Build prior training data from DevQuery-Bench annotations.

Changes vs previous version:
  - Removed cosine similarity feature. The whole point of this benchmark is
    that query-to-code cosine similarity is LOW (high alpha). Including it as
    a feature teaches the MLP to rely on the same signal that fails on
    naturalistic queries. The interaction feature (elem_prod) captures
    cross-attention between the two embeddings without leaking the scalar.
  - Multi-target GT filtering: for queries with multiple ground-truth targets,
    only emit training pairs for targets whose path is "unambiguous" — i.e.
    no other GT target shares the same parent node at that level. Conflicting
    supervision (two GT targets in different children of the same parent, both
    labelled positive at the parent level) was creating contradictory gradients.

Input to MLP is now: query_emb || node_emb || elem_prod = 384+384+384 = 1152 dims.

For each (query, ground_truth_function) pair in TRAINING repos:
  1. Find path from tree root to ground-truth function
  2. At each level of the path, if the level is unambiguous, create:
     - Positive pair: (query_emb, on_path_node_emb, label=1)
     - Negative pairs: (query_emb, each_sibling_emb, label=0)
"""
import json
import os
import sys
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Set

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


def get_unambiguous_paths(tree: Dict, gt_list: List[str]) -> List[List[Dict]]:
    """
    For a multi-target GT query, return only the paths that are unambiguous
    at every level.

    A level is ambiguous if two different GT targets have a different on-path
    node at that level (i.e. the parent would need to point to two different
    children as positive). We discard the entire path for any target that
    causes such a conflict at any level.

    For single-target GT this just returns the one path as-is.
    """
    paths = []
    for target in gt_list:
        path = find_path_to_node(tree, target)
        if path is not None and len(path) >= 2:
            paths.append(path)

    if len(paths) <= 1:
        return paths  # nothing to filter

    # Build a map: level_index -> set of on-path node titles across all GT targets
    max_depth = max(len(p) for p in paths)
    level_titles: List[Set[str]] = [set() for _ in range(max_depth)]
    for path in paths:
        for i, node in enumerate(path):
            level_titles[i].add(node.get('title', node.get('name', '')))

    # A level is ambiguous if >1 distinct title appears across GT paths at that level
    ambiguous_levels: Set[int] = {
        i for i, titles in enumerate(level_titles) if len(titles) > 1
    }

    # Keep only paths that do NOT pass through any ambiguous level
    # (i.e. every level in this path has the same on-path node as all other
    # kept paths — meaning they converge at that level)
    clean_paths = []
    for path in paths:
        path_ambiguous = any(
            i in ambiguous_levels for i in range(len(path))
        )
        if not path_ambiguous:
            clean_paths.append(path)

    # If everything got filtered (all paths conflict), fall back to the
    # single longest path so we don't lose the query entirely
    if not clean_paths:
        clean_paths = [max(paths, key=len)]

    return clean_paths


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
    all_elem_prods  = []  # element-wise product: query_emb * node_emb (384,)
    all_labels      = []
    all_query_ids   = []  # group id per (query, target, level) for pairwise loss

    used       = 0
    skipped    = 0
    filtered_ambiguous = 0
    query_id   = 0

    tree_cache = {}

    for entry in train_entries:
        repo_id      = entry['repo_id']
        query        = entry['query']
        ground_truth = entry['ground_truth']

        # Normalise: ground_truth is always a list in the bench JSON, but guard anyway
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]

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

        # Get unambiguous paths only
        clean_paths = get_unambiguous_paths(tree, ground_truth)
        n_original_paths = sum(
            1 for t in ground_truth
            if find_path_to_node(tree, t) is not None
        )
        filtered_ambiguous += max(0, n_original_paths - len(clean_paths))

        for path in clean_paths:
            found_any_path = True

            for i in range(1, len(path)):
                on_path_node  = path[i]
                parent        = path[i - 1]
                siblings      = parent.get('nodes', parent.get('children', []))

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

                    elem_prod = query_emb * node_emb  # (384,)
                    level_pairs.append((query_emb, node_emb, elem_prod, label))

                if not has_positive or len(level_pairs) < 2:
                    continue

                for qe, ne, ep, label in level_pairs:
                    all_query_embs.append(qe)
                    all_node_embs.append(ne)
                    all_elem_prods.append(ep)
                    all_labels.append(label)
                    all_query_ids.append(query_id)

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
    print(f"  Used queries:          {used}")
    print(f"  Skipped queries:       {skipped}")
    print(f"  Ambiguous paths pruned:{filtered_ambiguous}")
    print(f"  Total pairs:           {len(all_labels)}")
    print(f"  Positives:             {n_pos}")
    print(f"  Negatives:             {n_neg}")
    print(f"  Positive rate:         {pos_rate:.1f}%")
    print(f"  Query groups:          {query_id}  (for pairwise ranking loss)")

    data = {
        'query_embs':  torch.tensor(np.array(all_query_embs),  dtype=torch.float32),
        'node_embs':   torch.tensor(np.array(all_node_embs),   dtype=torch.float32),
        'elem_prods':  torch.tensor(np.array(all_elem_prods),  dtype=torch.float32),
        'labels':      torch.tensor(all_labels,                dtype=torch.float32),
        'query_ids':   torch.tensor(all_query_ids,             dtype=torch.long),
        'embed_dim':   embed_dim,
        'n_groups':    query_id,
    }

    output_path = 'devquery_bench/prior_training_data.pt'
    torch.save(data, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Saved to: {output_path} ({size_mb:.1f} MB)")
    print(f"   MLP input dim will be: {embed_dim} + {embed_dim} + {embed_dim} = {embed_dim * 3}")


if __name__ == '__main__':
    main()
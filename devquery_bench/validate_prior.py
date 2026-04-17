"""
validate_prior.py — test the bottleneck RelevancePrior on held-out repos.

On the TEST repos, check if the prior ranks the correct path higher
than all sibling nodes at each tree level.

Metric: per-level ranking accuracy
  = fraction of (query, level) pairs where the on-path node scores #1

For multi-target GT queries, a level counts as correct if the on-path
node for ANY of the ground truth targets ranks #1 at that level.
"""
import json
import os
import sys
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.mcts.relevance_prior import RelevancePrior


def find_path(root, target):
    def dfs(node, path):
        title = node.get('title', node.get('name', ''))
        if title == target:
            return path + [node]
        for c in node.get('nodes', node.get('children', [])):
            r = dfs(c, path + [node])
            if r:
                return r
        return None
    return dfs(root, [])


def score_siblings(prior, query_emb, siblings):
    """
    Score all siblings that have embeddings.
    Returns list of (title, score) pairs.
    """
    q_tensor = torch.tensor(query_emb, dtype=torch.float32)
    results  = []

    for sib in siblings:
        emb = sib.get('embedding')
        if emb is None:
            continue
        n_tensor = torch.tensor(np.array(emb, dtype=np.float32), dtype=torch.float32)

        with torch.no_grad():
            score = prior(q_tensor, n_tensor).item()

        results.append((sib.get('title', sib.get('name', '')), score))

    return results


def main():
    prior_path = 'devquery_bench/prior.pt'
    if not os.path.exists(prior_path):
        print(f"❌ {prior_path} not found. Run train_prior.py first.")
        return

    prior = RelevancePrior.load(prior_path)
    prior.eval()
    total_params = sum(p.numel() for p in prior.parameters())
    print(f"Loaded prior: {total_params:,} params, proj_dim={prior.proj_dim}")

    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    with open('devquery_bench/train_test_split.json') as f:
        split = json.load(f)
    with open('devquery_bench/devquery_bench.json') as f:
        bench = json.load(f)

    test_entries = [e for e in bench if e['repo_id'] in split['test']]
    print(f"Validating on {len(test_entries)} test entries...")

    correct_levels = 0
    total_levels   = 0
    tree_cache     = {}

    for entry in test_entries:
        repo_id = entry['repo_id']
        if repo_id not in tree_cache:
            tp = f"devquery_bench/trees/{repo_id}.json"
            if not os.path.exists(tp):
                continue
            with open(tp) as f:
                tree_cache[repo_id] = json.load(f)

        tree      = tree_cache[repo_id]
        query_emb = embed_model.encode(entry['query'], show_progress_bar=False)

        gt_list = entry['ground_truth']
        if isinstance(gt_list, str):
            gt_list = [gt_list]

        # For each tree level, check if ANY gt target's on-path node ranks #1
        # Group by (parent_title, level_index) so we don't double-count
        evaluated_levels = set()

        for target_title in gt_list:
            path = find_path(tree, target_title)
            if not path or len(path) < 2:
                continue

            for i in range(1, len(path)):
                parent       = path[i - 1]
                on_path      = path[i]
                parent_title = parent.get('title', parent.get('name', ''))
                level_key    = (parent_title, i)

                if level_key in evaluated_levels:
                    continue

                siblings = parent.get('nodes', parent.get('children', []))
                if len(siblings) < 2:
                    continue

                scored = score_siblings(prior, query_emb, siblings)
                if not scored:
                    continue

                on_path_title = on_path.get('title', '')
                best_title    = max(scored, key=lambda x: x[1])[0]

                evaluated_levels.add(level_key)
                total_levels += 1
                if best_title == on_path_title:
                    correct_levels += 1

    if total_levels > 0:
        acc = correct_levels / total_levels * 100
        print(f"\n✅ Per-level accuracy: {correct_levels}/{total_levels} = {acc:.1f}%")
        print(f"   (Prior correctly ranks the on-path node #1 among siblings)")
        if acc > 50:
            print(f"   ABOVE random chance — prior is learning!")
        if acc >= 60:
            print(f"   ≥60% — good enough for PUCT to benefit from the prior")
        if acc >= 70:
            print(f"   ≥70% — strong prior, should give clear PUCT wins")
    else:
        print("❌ Could not evaluate any levels. Check paths and embeddings.")


if __name__ == '__main__':
    main()
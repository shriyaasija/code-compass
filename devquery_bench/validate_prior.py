"""
validate_prior.py  —  updated for ImprovedPrior

On the TEST repos, check if the prior ranks the correct path higher
than all sibling nodes at each tree level.

Metric: per-level ranking accuracy
  = fraction of (query, level) pairs where the on-path node scores #1

This is the same metric validate_prior.py always computed, but now
loads the ImprovedPrior and passes the extra features it needs.
"""
import json
import os
import sys
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── import the new model class ──
# It lives in devquery_bench/train_prior.py so we load it from there.
# If you moved it to research/mcts/relevance_prior.py, adjust accordingly.
from devquery_bench.train_prior import ImprovedPrior


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


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


def main():
    prior_path = 'devquery_bench/prior.pt'
    if not os.path.exists(prior_path):
        print(f"❌ {prior_path} not found. Run train_prior.py first.")
        return

    prior = ImprovedPrior.load(prior_path)
    prior.eval()

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

    for entry in test_entries[:50]:   # up to 50 for speed; raise to len() for full eval
        repo_id = entry['repo_id']
        if repo_id not in tree_cache:
            tp = f"devquery_bench/trees/{repo_id}.json"
            if not os.path.exists(tp):
                continue
            with open(tp) as f:
                tree_cache[repo_id] = json.load(f)

        tree = tree_cache[repo_id]
        
        # Embed the query once
        query_emb   = embed_model.encode(entry['query'], show_progress_bar=False)
        q_tensor    = torch.tensor(query_emb, dtype=torch.float32)

        # Iterate over all ground truth targets
        for target_title in entry['ground_truth']:
            path = find_path(tree, target_title)
            if not path or len(path) < 2:
                continue

            for i in range(1, len(path)):
                parent      = path[i - 1]
                on_path     = path[i]
                siblings    = parent.get('nodes', parent.get('children', []))
                if len(siblings) < 2:
                    continue

                on_path_title  = on_path.get('title', '')
                scores         = []
                on_path_score  = None

                for sib in siblings:
                    emb = sib.get('embedding')
                    if emb is None:
                        continue

                    node_emb   = np.array(emb, dtype=np.float32)
                    n_tensor   = torch.tensor(node_emb, dtype=torch.float32)
                    ep_tensor  = torch.tensor(query_emb * node_emb, dtype=torch.float32)
                    cos_tensor = torch.tensor([[cosine_sim(query_emb, node_emb)]], dtype=torch.float32)

                    with torch.no_grad():
                        score = prior(
                            q_tensor.unsqueeze(0),
                            n_tensor.unsqueeze(0),
                            ep_tensor.unsqueeze(0),
                            cos_tensor,
                        ).item()

                    scores.append(score)
                    if sib.get('title', '') == on_path_title:
                        on_path_score = score

                if on_path_score is not None and scores:
                    rank = sum(1 for s in scores if s > on_path_score) + 1
                    if rank == 1:
                        correct_levels += 1
                    total_levels += 1

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
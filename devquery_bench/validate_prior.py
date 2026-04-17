"""
Quick validation: on the TEST repos, check if the prior ranks
the correct path higher than random paths.
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
            if r: return r
        return None
    return dfs(root, [])


def main():
    prior = RelevancePrior.load('devquery_bench/prior.pt')
    prior.eval()
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    with open('devquery_bench/train_test_split.json') as f:
        split = json.load(f)
    with open('devquery_bench/devquery_bench.json') as f:
        bench = json.load(f)

    test_entries = [e for e in bench if e['repo_id'] in split['test']]
    print(f"Validating on {len(test_entries)} test entries...")

    correct_levels = 0
    total_levels = 0
    tree_cache = {}

    for entry in test_entries[:30]:  # Sample 30 for speed
        repo_id = entry['repo_id']
        if repo_id not in tree_cache:
            tp = f"devquery_bench/trees/{repo_id}.json"
            if not os.path.exists(tp): continue
            with open(tp) as f:
                tree_cache[repo_id] = json.load(f)

        tree = tree_cache[repo_id]
        path = find_path(tree, entry['ground_truth'])
        if not path or len(path) < 2: continue

        query_emb = embed_model.encode(entry['query'], show_progress_bar=False)
        q_tensor = torch.tensor(query_emb, dtype=torch.float32)

        for i in range(1, len(path)):
            parent = path[i - 1]
            on_path = path[i]
            siblings = parent.get('nodes', parent.get('children', []))
            if len(siblings) < 2: continue

            on_path_title = on_path.get('title', '')
            scores = []
            on_path_score = None

            for sib in siblings:
                emb = sib.get('embedding')
                if emb is None: continue
                n_tensor = torch.tensor(emb, dtype=torch.float32)
                with torch.no_grad():
                    score = prior(q_tensor, n_tensor).item()
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
            print(f"   This is ABOVE random chance — prior is learning!")
        if acc > 70:
            print(f"   This is really good — prior should help PUCT significantly")
    else:
        print(f"❌ Could not evaluate any levels. Check paths and embeddings.")


if __name__ == '__main__':
    main()
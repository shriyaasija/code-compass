# Day 3 Guide: Main Benchmark Evaluation (Experiment 1)

**Goal:** Run ALL baselines and PUCT-MCTS on the 5 held-out DevQuery-Bench repos. This produces **Table 1** in the paper — the primary results.

**Time estimate:** 6–10 hours (mostly LLM inference time — start early and let it run)

**Prerequisites:** Day 1 + Day 2 complete. Trees built, prior trained.

**LM Studio must be running with a model loaded.**

---

## Step 0: Verify Prerequisites (5 minutes)

```bash
cd ~/code-compass
source venv/bin/activate

python3 -c "
import json, os

# Check prior exists
assert os.path.exists('devquery_bench/prior.pt'), '❌ prior.pt missing (run Day 2)'
print('✅ prior.pt exists')

# Check test repos have trees
with open('devquery_bench/train_test_split.json') as f:
    split = json.load(f)
for repo_id in split['test']:
    tp = f'devquery_bench/trees/{repo_id}.json'
    assert os.path.exists(tp), f'❌ Tree missing: {tp}'
print(f'✅ All {len(split[\"test\"])} test repo trees exist')

# Check benchmark has test entries
with open('devquery_bench/devquery_bench.json') as f:
    bench = json.load(f)
test_entries = [e for e in bench if e['repo_id'] in split['test']]
print(f'✅ {len(test_entries)} test queries ready')

# Check LM Studio
import requests
try:
    r = requests.get('http://localhost:1234/v1/models', timeout=5)
    models = r.json().get('data', [])
    print(f'✅ LM Studio running, model: {models[0][\"id\"] if models else \"none\"}')
except:
    print('❌ LM Studio not running! Start it before proceeding.')
"
```

---

## Step 1: Create the Experiment 1 Runner (20 minutes)

This script runs all 7 methods on all test repos and collects every metric.

```bash
cat > experiments/run_day3_main_eval.py << 'PYEOF'
"""
Experiment 1: Main DevQuery-Bench Evaluation

Runs 7 methods on 5 held-out test repos:
1. Dense (MiniLM cosine)
2. BM25  (rank_bm25 on function bodies)
3. Dense + BM25 hybrid (lambda=0.5)
4. Greedy-Tree (LLM at each level, no backtrack)
5. Baseline MCTS (UCB1, LLM everywhere)
6. PUCT-MCTS (prior at internal, LLM at leaves)
7. PUCT-MCTS + Online Adaptation

Collects: R@1, R@5, MRR, LLM calls/query, latency, per-query details
"""
import json
import os
import sys
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Metrics ──────────────────────────────────────────────────────────────────

def recall_at_k(ranked, gt, k):
    return 1.0 if any(r == gt or r.split('.')[-1] == gt.split('.')[-1] 
                       for r in ranked[:k]) else 0.0

def mrr_score(ranked, gt):
    for i, r in enumerate(ranked):
        if r == gt or r.split('.')[-1] == gt.split('.')[-1]:
            return 1.0 / (i + 1)
    return 0.0

def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    """Compute 95% bootstrap confidence interval."""
    rng = np.random.RandomState(42)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=len(scores), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return float(lower), float(upper)


# ─── Method 1: Dense Retrieval ───────────────────────────────────────────────

def run_dense(tree, query, embed_model, top_k=10):
    """Flat cosine similarity over all leaf functions."""
    from sentence_transformers.util import cos_sim
    
    query_emb = embed_model.encode(query, show_progress_bar=False)
    
    # Collect all leaf functions with embeddings
    leaves = []
    def walk(node):
        ntype = node.get('type', node.get('node_type', ''))
        children = node.get('nodes', node.get('children', []))
        if ntype in ('function', 'method') and 'embedding' in node:
            leaves.append(node)
        for c in children:
            walk(c)
    walk(tree)
    
    # Score by cosine similarity
    scored = []
    for leaf in leaves:
        emb = np.array(leaf['embedding'], dtype=np.float32)
        sim = float(cos_sim(query_emb, emb)[0][0])
        scored.append((sim, leaf))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:top_k]]


# ─── Method 2: BM25 ─────────────────────────────────────────────────────────

def run_bm25(tree, query, top_k=10):
    """BM25 over function names + summaries."""
    from rank_bm25 import BM25Okapi
    
    leaves = []
    def walk(node):
        ntype = node.get('type', node.get('node_type', ''))
        children = node.get('nodes', node.get('children', []))
        if ntype in ('function', 'method'):
            leaves.append(node)
        for c in children:
            walk(c)
    walk(tree)
    
    if not leaves:
        return []
    
    # Build corpus from function info
    corpus = []
    for leaf in leaves:
        text = f"{leaf.get('title', '')} {leaf.get('summary', '')} {leaf.get('path', '')}"
        corpus.append(text.lower().split())
    
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.lower().split())
    
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    
    return [leaves[i] for i, _ in indexed[:top_k]]


# ─── Method 3: Hybrid Dense + BM25 ──────────────────────────────────────────

def run_hybrid(tree, query, embed_model, lambda_val=0.5, top_k=10):
    """Interpolate Dense and BM25 scores."""
    from sentence_transformers.util import cos_sim
    from rank_bm25 import BM25Okapi
    
    query_emb = embed_model.encode(query, show_progress_bar=False)
    
    leaves = []
    def walk(node):
        ntype = node.get('type', node.get('node_type', ''))
        children = node.get('nodes', node.get('children', []))
        if ntype in ('function', 'method'):
            leaves.append(node)
        for c in children:
            walk(c)
    walk(tree)
    
    if not leaves:
        return []
    
    # Dense scores
    dense_scores = []
    for leaf in leaves:
        emb = leaf.get('embedding')
        if emb is not None:
            sim = float(cos_sim(query_emb, np.array(emb, dtype=np.float32))[0][0])
        else:
            sim = 0.0
        dense_scores.append(sim)
    
    # BM25 scores
    corpus = [f"{l.get('title','')} {l.get('summary','')} {l.get('path','')}".lower().split() 
              for l in leaves]
    bm25 = BM25Okapi(corpus)
    bm25_raw = bm25.get_scores(query.lower().split())
    
    # Normalize BM25 scores to [0,1]
    bm25_max = max(bm25_raw) if max(bm25_raw) > 0 else 1
    bm25_scores = [s / bm25_max for s in bm25_raw]
    
    # Hybrid: lambda * dense + (1-lambda) * bm25
    hybrid_scores = [lambda_val * d + (1 - lambda_val) * b 
                     for d, b in zip(dense_scores, bm25_scores)]
    
    indexed = list(enumerate(hybrid_scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    
    return [leaves[i] for i, _ in indexed[:top_k]]


# ─── Method 4: Greedy-Tree ──────────────────────────────────────────────────

def run_greedy_tree(tree, query, llm, top_k=10):
    """LLM-scored greedy tree traversal (no backtracking)."""
    llm_calls = 0
    
    def score_node(node, query_text):
        nonlocal llm_calls
        llm_calls += 1
        title = node.get('title', node.get('name', ''))
        summary = node.get('summary', '')
        prompt = f"""Rate relevance of this code element to the query (0.0-1.0).
Query: {query_text}
Element: {title}
Description: {summary[:200]}
Respond with ONLY a number."""
        try:
            resp = llm.chat([
                {"role": "system", "content": "Respond with only a number between 0.0 and 1.0."},
                {"role": "user", "content": prompt}
            ], temperature=0.0, max_tokens=10)
            import re
            nums = re.findall(r'\d+\.?\d*', resp.strip())
            if nums:
                return min(1.0, max(0.0, float(nums[0])))
        except:
            pass
        return 0.0
    
    # Traverse greedily
    current = tree
    visited_leaves = []
    
    for _ in range(10):  # max depth
        children = current.get('nodes', current.get('children', []))
        if not children:
            break
        
        # Check if any children are leaves
        leaf_children = []
        internal_children = []
        for c in children:
            ntype = c.get('type', c.get('node_type', ''))
            if ntype in ('function', 'method'):
                leaf_children.append(c)
            elif c.get('nodes', c.get('children', [])):
                internal_children.append(c)
        
        # Score and collect leaves
        for leaf in leaf_children:
            s = score_node(leaf, query)
            visited_leaves.append((s, leaf))
        
        # Score internal children and pick best
        if internal_children:
            best_score = -1
            best_child = internal_children[0]
            for c in internal_children:
                s = score_node(c, query)
                if s > best_score:
                    best_score = s
                    best_child = c
            current = best_child
        else:
            break
    
    visited_leaves.sort(key=lambda x: x[0], reverse=True)
    return [l for _, l in visited_leaves[:top_k]], llm_calls


# ─── Main evaluation loop ────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EXPERIMENT 1: Main DevQuery-Bench Evaluation")
    print("=" * 70)
    
    # Load everything
    with open('devquery_bench/train_test_split.json') as f:
        split = json.load(f)
    with open('devquery_bench/devquery_bench.json') as f:
        bench = json.load(f)
    
    test_entries_by_repo = defaultdict(list)
    for e in bench:
        if e['repo_id'] in split['test']:
            test_entries_by_repo[e['repo_id']].append(e)
    
    print(f"\nTest repos: {list(test_entries_by_repo.keys())}")
    print(f"Total test queries: {sum(len(v) for v in test_entries_by_repo.values())}")
    
    # Load embedding model
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load LLM
    from backend.lmstudio_client import LMStudioLLM
    llm = LMStudioLLM()
    
    # Results storage
    all_results = {}
    method_names = ['dense', 'bm25', 'hybrid', 'greedy_tree', 
                    'baseline_mcts', 'puct_mcts', 'puct_adapted']
    
    for method in method_names:
        all_results[method] = {
            'per_query': [],
            'per_repo': {},
        }
    
    for repo_id, entries in test_entries_by_repo.items():
        tree_path = f"devquery_bench/trees/{repo_id}.json"
        if not os.path.exists(tree_path):
            print(f"\n⚠️ Skipping {repo_id} (no tree)")
            continue
        
        with open(tree_path) as f:
            tree = json.load(f)
        
        print(f"\n{'━'*70}")
        print(f"  REPO: {repo_id} ({len(entries)} queries)")
        print(f"{'━'*70}")
        
        for method in method_names:
            print(f"\n  Running {method}...")
            r1s, r5s, mrrs, lats, llm_counts = [], [], [], [], []
            
            # Reset PUCT prior for adapted variant
            puct_searcher = None
            if method in ('puct_mcts', 'puct_adapted'):
                from research.mcts.puct_search import PUCTSearch
                puct_searcher = PUCTSearch(
                    llm_client=llm,
                    prior_path='devquery_bench/prior.pt',
                    max_iterations=30,
                    c_puct=1.0,
                    online_update=(method == 'puct_adapted'),
                    verbose=False,
                )
            
            if method == 'baseline_mcts':
                from backend.code_index2 import MCTSTreeSearch
                mcts_searcher = MCTSTreeSearch(llm_client=llm)
                # Need to save tree to temp file for MCTSTreeSearch
                import tempfile
                tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False, 
                                                  dir='devquery_bench/trees')
                json.dump(tree, open(tmp.name, 'w'))
                mcts_searcher.load_repository_tree(repo_id, tmp.name)
            
            for qi, entry in enumerate(entries):
                query = entry['query']
                gt = entry['ground_truth']
                
                t0 = time.time()
                llm_call_count = 0
                
                try:
                    if method == 'dense':
                        results = run_dense(tree, query, embed_model)
                    elif method == 'bm25':
                        results = run_bm25(tree, query)
                    elif method == 'hybrid':
                        results = run_hybrid(tree, query, embed_model)
                    elif method == 'greedy_tree':
                        results, llm_call_count = run_greedy_tree(tree, query, llm)
                    elif method == 'baseline_mcts':
                        results = mcts_searcher.search(repo_id, query, top_k=10)
                        if hasattr(mcts_searcher, 'mcts') and hasattr(mcts_searcher.mcts, 'llm_call_count'):
                            llm_call_count = mcts_searcher.mcts.llm_call_count
                    elif method in ('puct_mcts', 'puct_adapted'):
                        results = puct_searcher.search(tree, query, top_k=10)
                        llm_call_count = puct_searcher.llm_call_count
                except Exception as e:
                    print(f"    ⚠️ Error on query {qi}: {e}")
                    results = []
                    llm_call_count = 0
                
                lat = time.time() - t0
                
                ranked_names = [r.get('title', r.get('name', '')) for r in results]
                
                r1 = recall_at_k(ranked_names, gt, 1)
                r5 = recall_at_k(ranked_names, gt, 5)
                m = mrr_score(ranked_names, gt)
                
                r1s.append(r1)
                r5s.append(r5)
                mrrs.append(m)
                lats.append(lat)
                llm_counts.append(llm_call_count)
                
                all_results[method]['per_query'].append({
                    'repo_id': repo_id,
                    'query': query,
                    'ground_truth': gt,
                    'r1': r1, 'r5': r5, 'mrr': m,
                    'latency_ms': round(lat * 1000, 1),
                    'llm_calls': llm_call_count,
                    'ranked': ranked_names[:5],
                })
            
            avg_mrr = np.mean(mrrs) if mrrs else 0
            avg_r1 = np.mean(r1s) if r1s else 0
            avg_llm = np.mean(llm_counts) if llm_counts else 0
            avg_lat = np.mean(lats) * 1000 if lats else 0
            
            all_results[method]['per_repo'][repo_id] = {
                'R@1': round(float(avg_r1), 4),
                'R@5': round(float(np.mean(r5s)), 4) if r5s else 0,
                'MRR': round(float(avg_mrr), 4),
                'n_queries': len(mrrs),
                'avg_llm_calls': round(float(avg_llm), 2),
                'avg_latency_ms': round(float(avg_lat), 1),
            }
            
            print(f"    R@1={avg_r1:.3f}  MRR={avg_mrr:.3f}  "
                  f"LLM={avg_llm:.1f}  lat={avg_lat:.0f}ms")
    
    # ─── Compute aggregates and bootstrap CIs ────────────────────────────────
    print(f"\n\n{'='*70}")
    print("AGGREGATED RESULTS")
    print(f"{'='*70}\n")
    
    summary_table = []
    for method in method_names:
        pq = all_results[method]['per_query']
        if not pq:
            continue
        
        mrrs_all = [q['mrr'] for q in pq]
        r1s_all = [q['r1'] for q in pq]
        r5s_all = [q['r5'] for q in pq]
        llm_all = [q['llm_calls'] for q in pq]
        lat_all = [q['latency_ms'] for q in pq]
        
        mrr_ci = bootstrap_ci(mrrs_all) if len(mrrs_all) > 5 else (0, 0)
        
        row = {
            'method': method,
            'R@1': round(float(np.mean(r1s_all)), 4),
            'R@5': round(float(np.mean(r5s_all)), 4),
            'MRR': round(float(np.mean(mrrs_all)), 4),
            'MRR_CI_lower': round(mrr_ci[0], 4),
            'MRR_CI_upper': round(mrr_ci[1], 4),
            'avg_llm_calls': round(float(np.mean(llm_all)), 2),
            'avg_latency_ms': round(float(np.mean(lat_all)), 1),
            'n_queries': len(pq),
        }
        summary_table.append(row)
        
        print(f"  {method:>20}: R@1={row['R@1']:.3f}  R@5={row['R@5']:.3f}  "
              f"MRR={row['MRR']:.3f} [{row['MRR_CI_lower']:.3f}, {row['MRR_CI_upper']:.3f}]  "
              f"LLM={row['avg_llm_calls']:.1f}  lat={row['avg_latency_ms']:.0f}ms")
    
    # ─── Save everything ─────────────────────────────────────────────────────
    output = {
        'summary': summary_table,
        'per_method': {m: all_results[m] for m in method_names},
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    os.makedirs('experiments/results', exist_ok=True)
    output_path = 'experiments/results/experiment1_main_eval.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Full results saved to: {output_path}")
    
    # Also print as markdown table for paper
    print(f"\n\n{'='*70}")
    print("MARKDOWN TABLE (paste into paper)")
    print(f"{'='*70}\n")
    print("| Method | R@1 | R@5 | MRR | 95% CI | LLM/q | Latency |")
    print("|--------|-----|-----|-----|--------|-------|---------|")
    for row in summary_table:
        ci = f"[{row['MRR_CI_lower']:.3f}, {row['MRR_CI_upper']:.3f}]"
        print(f"| {row['method']} | {row['R@1']:.3f} | {row['R@5']:.3f} | "
              f"{row['MRR']:.3f} | {ci} | {row['avg_llm_calls']:.1f} | "
              f"{row['avg_latency_ms']:.0f}ms |")


if __name__ == '__main__':
    main()
PYEOF

echo "✅ Created experiments/run_day3_main_eval.py"
```

---

## Step 2: Run It (4–8 hours)

```bash
cd ~/code-compass
source venv/bin/activate

# Make sure LM Studio is running!
# This is a long run — use nohup or tmux/screen to prevent losing progress

# Option A: Run directly (keep terminal open)
python experiments/run_day3_main_eval.py 2>&1 | tee experiments/day3_log.txt

# Option B: Run in background with nohup
nohup python experiments/run_day3_main_eval.py > experiments/day3_log.txt 2>&1 &
echo "Started in background. Check progress with: tail -f experiments/day3_log.txt"
```

---

## Step 3: Check Results (after run completes)

```bash
# Check results
python3 -c "
import json
with open('experiments/results/experiment1_main_eval.json') as f:
    results = json.load(f)
print('SUMMARY:')
for row in results['summary']:
    print(f\"  {row['method']:>20}: MRR={row['MRR']:.3f}  LLM={row['avg_llm_calls']:.1f}\")
"
```

---

## Day 3 Outputs Checklist

- [ ] `experiments/run_day3_main_eval.py` — the runner script
- [ ] `experiments/results/experiment1_main_eval.json` — full results with per-query details
- [ ] `experiments/day3_log.txt` — execution log
- [ ] All 7 methods evaluated on all 5 test repos
- [ ] Bootstrap CIs computed for MRR

**Git checkpoint:**
```bash
cd ~/code-compass
git add experiments/
git commit -m "Day 3: Experiment 1 - Main DevQuery-Bench evaluation (7 methods × 5 repos)"
```

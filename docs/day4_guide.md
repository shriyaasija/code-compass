# Day 4 Guide: Online Adaptation Trajectory (Experiment 2)

**Goal:** Show that PUCT-MCTS gets better over a session. Run 20 queries in sequence on 3 repos, compare "static prior" vs "adapting prior". Produce the adaptation trajectory figure.

**Time estimate:** 3–4 hours

**Prerequisites:** Days 1–3 complete.

---

## Step 0: Verify Prerequisites (2 minutes)

```bash
cd ~/code-compass
source venv/bin/activate

python3 -c "
import os
assert os.path.exists('devquery_bench/prior.pt'), '❌ prior.pt missing'
assert os.path.exists('devquery_bench/devquery_bench.json'), '❌ benchmark missing'
print('✅ All prerequisites met')
"
```

Make sure LM Studio is running.

---

## Step 1: Create the Adaptation Experiment Script (10 minutes)

```bash
cat > experiments/run_day4_adaptation.py << 'PYEOF'
"""
Experiment 2: Online Adaptation Trajectory

For each of 3 test repos:
1. Run PUCT-MCTS (static prior) on queries 1-20 sequentially
2. Run PUCT-MCTS (online adaptation enabled) on queries 1-20 sequentially
3. Record MRR and LLM call count after EACH query

The adaptation version should show improving MRR over the session.
"""
import json
import os
import sys
import time
import copy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.mcts.puct_search import PUCTSearch
from backend.lmstudio_client import LMStudioLLM


def recall_at_k(ranked, gt, k):
    return 1.0 if any(r == gt or r.split('.')[-1] == gt.split('.')[-1] 
                       for r in ranked[:k]) else 0.0

def mrr_score(ranked, gt):
    for i, r in enumerate(ranked):
        if r == gt or r.split('.')[-1] == gt.split('.')[-1]:
            return 1.0 / (i + 1)
    return 0.0


def run_session(tree, queries, llm, prior_path, online_adapt, max_queries=20):
    """
    Run a sequence of queries with the same PUCTSearch instance.
    Returns per-query results.
    """
    searcher = PUCTSearch(
        llm_client=llm,
        prior_path=prior_path,
        max_iterations=30,
        c_puct=1.0,
        online_update=online_adapt,
        online_lr=5e-4,
        verbose=False,
    )
    
    per_query = []
    cumulative_mrr = 0.0
    
    for qi, entry in enumerate(queries[:max_queries]):
        query = entry['query']
        gt = entry['ground_truth']
        
        t0 = time.time()
        results = searcher.search(tree, query, top_k=10)
        lat = time.time() - t0
        
        ranked = [r.get('title', r.get('name', '')) for r in results]
        m = mrr_score(ranked, gt)
        r1 = recall_at_k(ranked, gt, 1)
        cumulative_mrr += m
        
        per_query.append({
            'query_index': qi + 1,
            'query': query,
            'ground_truth': gt,
            'mrr': m,
            'r1': r1,
            'cumulative_mrr': cumulative_mrr / (qi + 1),
            'llm_calls': searcher.llm_call_count,
            'prior_calls': searcher.prior_call_count,
            'latency_ms': round(lat * 1000, 1),
            'top_result': ranked[0] if ranked else '',
        })
        
        mode = "ADAPT" if online_adapt else "STATIC"
        print(f"    [{mode}] q{qi+1:>2}: MRR={m:.3f}  cumMRR={cumulative_mrr/(qi+1):.3f}  "
              f"LLM={searcher.llm_call_count}  r1={r1}")
    
    return per_query


def main():
    print("=" * 70)
    print("EXPERIMENT 2: Online Adaptation Trajectory")
    print("=" * 70)
    
    llm = LMStudioLLM()
    
    with open('devquery_bench/train_test_split.json') as f:
        split = json.load(f)
    with open('devquery_bench/devquery_bench.json') as f:
        bench = json.load(f)
    
    # Pick 3 test repos (the largest ones for most signal)
    test_repos = split['test'][:3]
    print(f"\nTest repos: {test_repos}")
    
    all_results = {}
    
    for repo_id in test_repos:
        print(f"\n{'━'*70}")
        print(f"  REPO: {repo_id}")
        print(f"{'━'*70}")
        
        tree_path = f"devquery_bench/trees/{repo_id}.json"
        if not os.path.exists(tree_path):
            print(f"  ⚠️ No tree, skipping")
            continue
        
        with open(tree_path) as f:
            tree = json.load(f)
        
        entries = [e for e in bench if e['repo_id'] == repo_id]
        if len(entries) < 10:
            print(f"  ⚠️ Only {len(entries)} queries, need 10+")
            continue
        
        # Run STATIC session
        print(f"\n  ── STATIC PRIOR (no adaptation) ──")
        static_results = run_session(
            tree, entries, llm, 
            prior_path='devquery_bench/prior.pt',
            online_adapt=False,
            max_queries=20,
        )
        
        # Run ADAPTED session (uses same initial prior but adapts)
        print(f"\n  ── ADAPTED PRIOR (online updates) ──")
        adapted_results = run_session(
            tree, entries, llm,
            prior_path='devquery_bench/prior.pt',
            online_adapt=True,
            max_queries=20,
        )
        
        all_results[repo_id] = {
            'static': static_results,
            'adapted': adapted_results,
        }
        
        # Print comparison
        static_final = static_results[-1]['cumulative_mrr'] if static_results else 0
        adapted_final = adapted_results[-1]['cumulative_mrr'] if adapted_results else 0
        improvement = adapted_final - static_final
        print(f"\n  Final cumulative MRR — Static: {static_final:.3f}, "
              f"Adapted: {adapted_final:.3f}, Δ={improvement:+.3f}")
    
    # Save results
    os.makedirs('experiments/results', exist_ok=True)
    output_path = 'experiments/results/experiment2_adaptation.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to: {output_path}")
    
    # Generate the figure
    generate_figure(all_results)


def generate_figure(results):
    """Generate the adaptation trajectory plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4), 
                                  sharey=True)
        if len(results) == 1:
            axes = [axes]
        
        colors = {'static': '#e74c3c', 'adapted': '#2ecc71'}
        
        for ax, (repo_id, data) in zip(axes, results.items()):
            for mode, label, color in [('static', 'Static Prior', colors['static']),
                                        ('adapted', 'Adapted Prior', colors['adapted'])]:
                queries = data[mode]
                x = [q['query_index'] for q in queries]
                y = [q['cumulative_mrr'] for q in queries]
                ax.plot(x, y, label=label, color=color, linewidth=2)
            
            ax.set_xlabel('Query Index', fontsize=11)
            ax.set_title(repo_id.replace('__', '/'), fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        axes[0].set_ylabel('Cumulative MRR', fontsize=11)
        
        plt.suptitle('Online Adaptation Trajectory', fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        os.makedirs('figures', exist_ok=True)
        fig_path = 'figures/adaptation_trajectory.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✅ Figure saved to: {fig_path}")
    except ImportError:
        print("⚠️ matplotlib not installed, skipping figure generation")


if __name__ == '__main__':
    main()
PYEOF

echo "✅ Created experiments/run_day4_adaptation.py"
```

---

## Step 2: Run the Experiment (2–3 hours)

```bash
cd ~/code-compass
source venv/bin/activate

python experiments/run_day4_adaptation.py 2>&1 | tee experiments/day4_log.txt
```

---

## Step 3: Check Results

```bash
python3 -c "
import json
with open('experiments/results/experiment2_adaptation.json') as f:
    r = json.load(f)
for repo, data in r.items():
    s_final = data['static'][-1]['cumulative_mrr'] if data['static'] else 0
    a_final = data['adapted'][-1]['cumulative_mrr'] if data['adapted'] else 0
    print(f'{repo}: static={s_final:.3f}, adapted={a_final:.3f}, Δ={a_final-s_final:+.3f}')
"

# View the figure
ls -la figures/adaptation_trajectory.png
```

---

## Day 4 Outputs Checklist

- [ ] `experiments/run_day4_adaptation.py` — experiment script
- [ ] `experiments/results/experiment2_adaptation.json` — per-query results for both modes
- [ ] `figures/adaptation_trajectory.png` — the trajectory plot for the paper
- [ ] Adapted prior shows MRR improvement by query 10–15

**Git checkpoint:**
```bash
cd ~/code-compass
git add experiments/ figures/
git commit -m "Day 4: Experiment 2 - Online adaptation trajectory (static vs adapted prior)"
```

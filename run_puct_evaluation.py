import argparse
import json
import os
import time
import numpy as np
from typing import Dict, List


SPLIT_PATH       = "benchmark_results/train_test_split.json"
PROPER_TREES_DIR = "benchmark_results/proper_trees"
METADATA_PATH    = "benchmark_results/proper_benchmark_metadata.json"
RESULTS_DIR      = "benchmark_results/eval_results"
PRIOR_PATH       = "research/mcts/prior.pt"


def recall_at_k(ranked: List[str], gt: str, k: int) -> float:
    return 1.0 if gt in ranked[:k] else 0.0

def mrr(ranked: List[str], gt: str) -> float:
    try:
        return 1.0 / (ranked.index(gt) + 1)
    except ValueError:
        return 0.0


def run_one_method(searcher, tree, queries, method_name, max_queries=15):
    """Run a searcher on queries and return metrics dict."""
    r1_scores, r5_scores, mrr_scores, latencies = [], [], [], []
    llm_calls_list = []

    queries = queries[:max_queries]

    for q_info in queries:
        query = q_info.get('query', q_info.get('docstring', ''))
        gt    = q_info.get('ground_truth', q_info.get('func_name', ''))
        if not query or not gt:
            continue

        t0 = time.time()
        results = searcher.search(tree, query, top_k=10)
        latency = time.time() - t0

        titles = [r.get('name', '') for r in results]
        r1_scores.append(recall_at_k(titles, gt, 1))
        r5_scores.append(recall_at_k(titles, gt, 5))
        mrr_scores.append(mrr(titles, gt))
        latencies.append(latency)

        if hasattr(searcher, 'llm_call_count'):
            llm_calls_list.append(searcher.llm_call_count)

    if not mrr_scores:
        return {'error': 'no valid queries'}

    return {
        'method':       method_name,
        'n_queries':    len(mrr_scores),
        'R@1':          round(float(np.mean(r1_scores)), 4),
        'R@5':          round(float(np.mean(r5_scores)), 4),
        'MRR':          round(float(np.mean(mrr_scores)), 4),
        'latency_ms':   round(float(np.mean(latencies)) * 1000, 1),
        'avg_llm_calls': round(float(np.mean(llm_calls_list)), 2) if llm_calls_list else None,
    }


def main(provider='lmstudio', model=None, max_queries=15):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    split         = json.load(open(SPLIT_PATH))
    test_repo_ids = split['test']
    metadata_list = json.load(open(METADATA_PATH))

    # Load LLM
    if provider == 'lmstudio':
        from backend.lmstudio_client import LMStudioLLM
        llm = LMStudioLLM()
    else:
        from backend.ollama_client import OllamaLLM
        llm = OllamaLLM(model=model or 'qwen2.5:7b')

    print("=" * 70)
    print(f"EVALUATION: PUCT-MCTS vs Baseline MCTS")
    print(f"Test repos: {len(test_repo_ids)}")
    print("=" * 70)

    all_results = {}

    for repo_id in test_repo_ids:
        meta = next((m for m in metadata_list if m['repo_id'] == repo_id), None)
        if not meta:
            print(f"  Skipping {repo_id}: no metadata")
            continue

        tree_path = os.path.join(PROPER_TREES_DIR, f"{repo_id}.json")
        if not os.path.exists(tree_path):
            print(f"  Skipping {repo_id}: no tree")
            continue

        tree    = json.load(open(tree_path))
        queries = meta.get('queries', [])

        print(f"\n{'─'*70}")
        print(f"  Repo: {repo_id}  ({len(queries[:max_queries])} queries)")
        print(f"{'─'*70}")

        repo_results = {}

        # ── Method 1: PUCT-MCTS (prior MLP + LLM at leaves only) ──────
        print(f"  Running PUCT-MCTS...")
        from research.mcts.puct_search import PUCTSearch
        puct_searcher = PUCTSearch(
            llm_client=llm,
            prior_path=PRIOR_PATH,
            max_iterations=30,
            c_puct=1.5,
            online_update=True,
            verbose=False,
        )
        puct_metrics = run_one_method(puct_searcher, tree, queries, 'PUCT-MCTS', max_queries)
        repo_results['puct'] = puct_metrics
        print(f"    R@1={puct_metrics.get('R@1')}, MRR={puct_metrics.get('MRR')}, "
              f"LLM calls/q={puct_metrics.get('avg_llm_calls')}, "
              f"latency={puct_metrics.get('latency_ms')}ms")

        # ── Method 2: Baseline MCTS (LLM at every node) ────────────────
        print(f"  Running Baseline MCTS...")
        from research.mcts.mcts_search import MCTSSearch
        baseline_searcher = MCTSSearch(
            llm_client=llm,
            max_iterations=30,
            c_explore=1.414,
        )
        base_metrics = run_one_method(baseline_searcher, tree, queries, 'Baseline-MCTS', max_queries)
        repo_results['baseline'] = base_metrics
        print(f"    R@1={base_metrics.get('R@1')}, MRR={base_metrics.get('MRR')}, "
              f"LLM calls/q={base_metrics.get('avg_llm_calls')}, "
              f"latency={base_metrics.get('latency_ms')}ms")

        all_results[repo_id] = repo_results

        # Save after each repo (don't lose progress)
        json.dump(all_results,
                  open(os.path.join(RESULTS_DIR, 'puct_eval.json'), 'w'),
                  indent=2)

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Repo':<25} {'PUCT R@1':>10} {'Base R@1':>10} "
          f"{'PUCT MRR':>10} {'Base MRR':>10} "
          f"{'LLM↓':>8}")
    print("─" * 75)

    puct_mrrs, base_mrrs = [], []
    puct_calls, base_calls = [], []

    for repo_id, res in all_results.items():
        p = res.get('puct', {})
        b = res.get('baseline', {})
        pm = p.get('MRR', 0); bm = b.get('MRR', 0)
        pc = p.get('avg_llm_calls', 0); bc = b.get('avg_llm_calls', 0)
        puct_mrrs.append(pm); base_mrrs.append(bm)
        if pc: puct_calls.append(pc)
        if bc: base_calls.append(bc)
        llm_reduction = f"{(1 - pc/max(bc,1))*100:.0f}%" if pc and bc else "N/A"
        print(f"  {repo_id:<23} "
              f"{p.get('R@1',0):>10.4f} {b.get('R@1',0):>10.4f} "
              f"{pm:>10.4f} {bm:>10.4f} "
              f"{llm_reduction:>8}")

    if puct_mrrs:
        print("─" * 75)
        avg_puct = np.mean(puct_mrrs); avg_base = np.mean(base_mrrs)
        avg_pc = np.mean(puct_calls) if puct_calls else 0
        avg_bc = np.mean(base_calls) if base_calls else 0
        reduction = f"{(1 - avg_pc/max(avg_bc,1))*100:.0f}%"
        print(f"  {'Average':<23} "
              f"{'':>10} {'':>10} "
              f"{avg_puct:>10.4f} {avg_base:>10.4f} "
              f"{reduction:>8}")
        print(f"\n  PUCT-MRR: {avg_puct:.4f}  Baseline-MRR: {avg_base:.4f}  "
              f"Delta: {avg_puct - avg_base:+.4f}")
        print(f"  Avg LLM calls — PUCT: {avg_pc:.1f}  Baseline: {avg_bc:.1f}  "
              f"Reduction: {reduction}")

    print(f"\nFull results: {RESULTS_DIR}/puct_eval.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--provider",    default="lmstudio")
    p.add_argument("--model",       default=None)
    p.add_argument("--max-queries", type=int, default=15)
    args = p.parse_args()
    main(provider=args.provider, model=args.model, max_queries=args.max_queries)
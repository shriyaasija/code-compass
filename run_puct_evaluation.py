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


def _match(a: str, b: str) -> bool:
    """Match by exact or suffix — handles module.func vs func."""
    if a == b:
        return True
    return a.split('.')[-1] == b.split('.')[-1] and a.split('.')[-1] != ''

def recall_at_k(ranked: List[str], gt: str, k: int) -> float:
    return 1.0 if any(_match(r, gt) for r in ranked[:k]) else 0.0

def mrr(ranked: List[str], gt: str) -> float:
    for i, r in enumerate(ranked):
        if _match(r, gt):
            return 1.0 / (i + 1)
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
            print(f"  Skipping {repo_id}: no tree at {tree_path}")
            continue

        queries = meta.get('queries', [])[:max_queries]
        print(f"\n{'─'*70}")
        print(f"  Repo: {repo_id}  ({len(queries)} queries)")

        repo_results = {}

        # Load raw tree for PUCT
        import json
        with open(tree_path, 'r') as f:
            raw_tree = json.load(f)

        def run_search(searcher, search_type, is_puct=False):
            r1, r5, mrr_scores, lats, llm_calls = [], [], [], [], []
            for q_info in queries:
                query = q_info.get('query', q_info.get('docstring', ''))
                gt    = q_info.get('ground_truth', q_info.get('func_name', ''))
                if not query or not gt:
                    continue
                t0 = time.time()
                if is_puct:
                    results = searcher.search(raw_tree, query, top_k=10)
                else:
                    results = searcher.search(repo_id, query, top_k=10)
                lat = time.time() - t0
                
                titles = [r.get('name', '') for r in results]
                r1.append(recall_at_k(titles, gt, 1))
                r5.append(recall_at_k(titles, gt, 5))
                mrr_scores.append(mrr(titles, gt))
                lats.append(lat)
                
                if hasattr(searcher, 'llm_call_count'):
                    llm_calls.append(searcher.llm_call_count)
                elif hasattr(searcher, 'mcts') and hasattr(searcher.mcts, 'llm_call_count'):
                    llm_calls.append(searcher.mcts.llm_call_count)

            repo_results[search_type] = {
                'R@1': round(float(np.mean(r1)), 4) if r1 else 0,
                'R@5': round(float(np.mean(r5)), 4) if r5 else 0,
                'MRR': round(float(np.mean(mrr_scores)), 4) if mrr_scores else 0,
                'latency_ms': round(float(np.mean(lats)) * 1000, 1) if lats else 0,
                'n_queries': len(mrr_scores),
                'avg_llm_calls': round(float(np.mean(llm_calls)), 2) if llm_calls else 0,
            }
            m = repo_results[search_type]
            print(f"    [{search_type}] R@1={m['R@1']}  R@5={m['R@5']}  MRR={m['MRR']}  lat={m['latency_ms']}ms  llm_calls={m['avg_llm_calls']}")

        # ── Baseline MCTS ──────
        print(f"  Running Baseline MCTS...")
        from backend.code_index2 import MCTSTreeSearch
        base_searcher = MCTSTreeSearch(llm_client=llm)
        base_searcher.load_repository_tree(repo_id, tree_path)
        run_search(base_searcher, 'baseline', is_puct=False)

        # ── PUCT MCTS ──────
        print(f"  Running PUCT MCTS...")
        from research.mcts.puct_search import PUCTSearch
        puct_searcher = PUCTSearch(llm_client=llm, prior_path="research/mcts/prior.pt")
        run_search(puct_searcher, 'puct', is_puct=True)

        all_results[repo_id] = repo_results
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
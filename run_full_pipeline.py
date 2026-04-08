#!/usr/bin/env python3
"""
Full Pipeline: MCTS Tree Search + RL Dynamic Optimization on CodeSearchNet.

Usage:
  # Step 1: Prepare data (if not already done)
  python run_full_pipeline.py --step prepare

  # Step 2: Run MCTS baseline on original trees
  python run_full_pipeline.py --step mcts-baseline --provider lmstudio

  # Step 3: Train RL agent to optimize each tree
  python run_full_pipeline.py --step rl-train --timesteps 5000

  # Step 4: Run MCTS on RL-optimized trees
  python run_full_pipeline.py --step mcts-optimized --provider lmstudio

  # Step 5: Compare results
  python run_full_pipeline.py --step compare

  # Or run everything at once (no LLM needed for RL training):
  python run_full_pipeline.py --step all --provider lmstudio --timesteps 5000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "benchmark_dir": "benchmark_results",
    "metadata_path": "benchmark_results/proper_benchmark_metadata.json",
    "rl_output_dir": "research/rl_index/checkpoints",
    "optimized_trees_dir": "benchmark_results/optimized_trees",
    "results_dir": "pipeline_results",
    "num_repos": 25,
    "queries_per_repo": 15,
    "rl_timesteps": 5000,
    "rl_max_steps": 50,
    "rl_lambda_mrr": 1.0,
    "rl_lambda_depth": 0.1,
    "mcts_max_iterations": 50,
    "mcts_c_explore": 1.414,
    "seed": 42,
}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: DATA PREPARATION (reuses existing benchmark.py)
# ═══════════════════════════════════════════════════════════════════════════════

def step_prepare(config: Dict):
    """Prepare CodeSearchNet data — download, build trees, generate queries."""
    meta_path = Path(config["benchmark_dir"]) / "benchmark_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        print(f"✅ Data already prepared: {len(metadata)} repos in {config['benchmark_dir']}/")
        print(f"   To re-prepare, delete {meta_path} and run again.")
        return metadata

    print("📦 Running data preparation via benchmark.py...")
    os.system(
        f"python benchmark.py --mode prepare "
        f"--num-repos {config['num_repos']} "
        f"--queries-per-repo {config['queries_per_repo']} "
        f"--seed {config['seed']} "
        f"--output-dir {config['benchmark_dir']}"
    )

    with open(meta_path) as f:
        metadata = json.load(f)
    return metadata


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: RL TRAINING — optimize each repo's tree
# ═══════════════════════════════════════════════════════════════════════════════

def step_rl_train(config: Dict, metadata: List[Dict]):
    """Train RL agent on each repo's tree to produce optimized trees."""
    from research.rl_index.env import TreeIndexEnv
    from research.rl_index.tree_mutations import deep_copy_tree, count_leaves, tree_depth

    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from sb3_contrib.common.maskable.utils import get_action_masks
    except ImportError:
        print("❌ sb3-contrib not installed. Run: pip install stable-baselines3 sb3-contrib")
        sys.exit(1)

    opt_dir = Path(config["optimized_trees_dir"])
    opt_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("🧠 STEP 2: RL ADAPTIVE INDEX TRAINING")
    print("=" * 70)
    print(f"   Repos: {len(metadata)}")
    print(f"   Timesteps per repo: {config['rl_timesteps']}")
    print(f"   Max steps/episode: {config['rl_max_steps']}")

    rl_results = []

    for repo_idx, repo in enumerate(metadata, 1):
        repo_id = repo["repo_id"]
        repo_name = repo["repo_name"]
        tree_path = repo["tree_path"]

        print(f"\n  [{repo_idx}/{len(metadata)}] {repo_name}")

        # Load tree
        with open(tree_path) as f:
            tree = json.load(f)

        # Build query buffer from repo's queries
        query_buffer = []
        for q in repo.get("queries", []):
            query_buffer.append({
                "query": q["query"][:200],  # Truncate long queries
                "ground_truth": q["ground_truth"],
            })

        if not query_buffer:
            print(f"     ⚠️  No queries for this repo, skipping RL training")
            # Just copy original tree as "optimized"
            opt_path = opt_dir / f"{repo_id}.json"
            with open(opt_path, 'w') as f:
                json.dump(tree, f)
            rl_results.append({
                "repo_id": repo_id, "repo_name": repo_name,
                "trained": False, "reason": "no queries",
            })
            continue

        before_depth = tree_depth(tree)
        before_leaves = count_leaves(tree)

        # Create environment
        env = TreeIndexEnv(
            tree=tree,
            query_buffer=query_buffer,
            max_steps=config["rl_max_steps"],
            lambda_mrr=config["rl_lambda_mrr"],
            lambda_depth=config["rl_lambda_depth"],
            seed=config["seed"],
        )

        def mask_fn(env):
            return env.get_action_mask()

        wrapped_env = ActionMasker(env, mask_fn)

        # Train
        try:
            model = MaskablePPO(
                "MlpPolicy", wrapped_env,
                learning_rate=3e-4,
                n_steps=min(128, config["rl_timesteps"]),
                batch_size=min(64, config["rl_timesteps"] // 2),
                n_epochs=10,
                gamma=0.99,
                ent_coef=0.01,
                seed=config["seed"],
                verbose=0,
            )

            start_time = time.time()
            model.learn(total_timesteps=config["rl_timesteps"])
            train_time = time.time() - start_time

            # Run inference to get optimized tree
            obs, _ = wrapped_env.reset()
            for step in range(config["rl_max_steps"]):
                action_masks = get_action_masks(wrapped_env)
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                if terminated or truncated:
                    break

            # Extract optimized tree
            inner_env = wrapped_env
            while hasattr(inner_env, 'env'):
                inner_env = inner_env.env
            optimized_tree = deep_copy_tree(inner_env.current_tree)

            after_depth = tree_depth(optimized_tree)
            after_leaves = count_leaves(optimized_tree)

            # Save optimized tree
            opt_path = opt_dir / f"{repo_id}.json"
            with open(opt_path, 'w') as f:
                json.dump(optimized_tree, f)

            print(f"     ✅ Trained in {train_time:.1f}s | "
                  f"Depth: {before_depth}→{after_depth} | "
                  f"Leaves: {before_leaves}→{after_leaves}")

            rl_results.append({
                "repo_id": repo_id, "repo_name": repo_name,
                "trained": True, "train_time": train_time,
                "depth_before": before_depth, "depth_after": after_depth,
                "leaves_before": before_leaves, "leaves_after": after_leaves,
                "optimized_tree_path": str(opt_path),
            })

        except Exception as e:
            print(f"     ❌ Training failed: {e}")
            # Save original tree as fallback
            opt_path = opt_dir / f"{repo_id}.json"
            with open(opt_path, 'w') as f:
                json.dump(tree, f)
            rl_results.append({
                "repo_id": repo_id, "repo_name": repo_name,
                "trained": False, "reason": str(e),
            })

    # Save RL results
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    rl_results_path = results_dir / "rl_training_results.json"
    with open(rl_results_path, 'w') as f:
        json.dump(rl_results, f, indent=2)
    print(f"\n  📁 RL results saved to: {rl_results_path}")

    return rl_results


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: MCTS SEARCH (on original or optimized trees)
# ═══════════════════════════════════════════════════════════════════════════════

def step_mcts_search(config: Dict, metadata: List[Dict],
                     provider: str, model_name: str,
                     use_optimized: bool = False, label: str = "baseline"):
    """Run MCTS search on original or RL-optimized trees."""
    sys.path.insert(0, str(Path(__file__).parent / "backend"))
    from code_index2 import MCTSTreeSearch

    # Init LLM
    if provider == "lmstudio":
        from backend.lmstudio_client import LMStudioLLM
        llm = LMStudioLLM(model=model_name)
    else:
        from backend.ollama_client import OllamaLLM
        llm = OllamaLLM(model=model_name)

    print(f"\n{'=' * 70}")
    print(f"🔍 STEP {'4' if use_optimized else '3'}: MCTS SEARCH ({label})")
    print(f"{'=' * 70}")
    print(f"   Provider: {provider} | Model: {model_name}")
    print(f"   Trees: {'RL-optimized' if use_optimized else 'Original'}")
    print(f"   Repos: {len(metadata)}")

    search_engine = MCTSTreeSearch(
        llm,
        max_iterations=config["mcts_max_iterations"],
        c_explore=config["mcts_c_explore"],
    )

    all_results = []

    for repo_idx, repo in enumerate(metadata, 1):
        repo_id = repo["repo_id"]
        repo_name = repo["repo_name"]
        queries = repo.get("queries", [])

        if use_optimized:
            tree_path = str(Path(config["optimized_trees_dir"]) / f"{repo_id}.json")
            if not os.path.exists(tree_path):
                print(f"  [{repo_idx}] ⚠️  No optimized tree for {repo_name}, using original")
                tree_path = repo["tree_path"]
        else:
            tree_path = repo["tree_path"]

        print(f"\n  [{repo_idx}/{len(metadata)}] {repo_name} ({len(queries)} queries)")

        search_engine.load_repository_tree(repo_id, tree_path)

        # Sample queries
        sampled = queries[:config["queries_per_repo"]] if queries else []

        repo_metrics = []
        for q_idx, q_info in enumerate(sampled):
            query = q_info["query"]
            gt = q_info["ground_truth"]

            try:
                search_engine.mcts.verbose = False
                results = search_engine.search(repo_id, query, top_k=50)
                ranked = [r["name"] for r in results]

                # Compute metrics
                mrr = 0.0
                try:
                    rank = ranked.index(gt) + 1
                    mrr = 1.0 / rank
                except ValueError:
                    mrr = 0.0

                r_at_1 = 1.0 if gt in ranked[:1] else 0.0
                r_at_5 = 1.0 if gt in ranked[:5] else 0.0
                r_at_10 = 1.0 if gt in ranked[:10] else 0.0

                repo_metrics.append({
                    "query": query[:100],
                    "ground_truth": gt,
                    "mrr": mrr,
                    "recall_at_1": r_at_1,
                    "recall_at_5": r_at_5,
                    "recall_at_10": r_at_10,
                    "llm_calls": search_engine.mcts.llm_call_count,
                })

            except Exception as e:
                print(f"     ⚠️  Query {q_idx} failed: {e}")

            if (q_idx + 1) % 5 == 0:
                print(f"     Completed {q_idx + 1}/{len(sampled)} queries")

        if repo_metrics:
            avg_mrr = np.mean([m["mrr"] for m in repo_metrics])
            avg_r1 = np.mean([m["recall_at_1"] for m in repo_metrics])
            avg_r5 = np.mean([m["recall_at_5"] for m in repo_metrics])
            avg_r10 = np.mean([m["recall_at_10"] for m in repo_metrics])
            print(f"     MRR={avg_mrr:.3f}  R@1={avg_r1:.3f}  R@5={avg_r5:.3f}  R@10={avg_r10:.3f}")

        all_results.append({
            "repo_id": repo_id,
            "repo_name": repo_name,
            "tree_type": "optimized" if use_optimized else "original",
            "num_queries": len(repo_metrics),
            "per_query": repo_metrics,
        })

        # Free memory
        if repo_id in search_engine.repositories:
            del search_engine.repositories[repo_id]

    # Save results
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"mcts_{label}_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  📁 MCTS {label} results saved to: {out_path}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def step_compare(config: Dict):
    """Compare MCTS baseline vs MCTS on RL-optimized trees."""
    results_dir = Path(config["results_dir"])

    baseline_path = results_dir / "mcts_baseline_results.json"
    optimized_path = results_dir / "mcts_optimized_results.json"
    rl_path = results_dir / "rl_training_results.json"

    print(f"\n{'=' * 70}")
    print("📊 RESULTS COMPARISON: MCTS (Original) vs MCTS (RL-Optimized)")
    print("=" * 70)

    # Load RL training results
    if rl_path.exists():
        with open(rl_path) as f:
            rl_results = json.load(f)
        trained = [r for r in rl_results if r.get("trained")]
        print(f"\n  RL Training: {len(trained)}/{len(rl_results)} repos trained")
        if trained:
            avg_time = np.mean([r["train_time"] for r in trained])
            depth_changes = [r["depth_after"] - r["depth_before"] for r in trained]
            print(f"  Avg training time: {avg_time:.1f}s")
            print(f"  Avg depth change: {np.mean(depth_changes):+.1f}")

    # Load MCTS results
    for label, path in [("Baseline (original trees)", baseline_path),
                        ("Optimized (RL trees)", optimized_path)]:
        if not path.exists():
            print(f"\n  ⚠️  {label}: No results found at {path}")
            continue

        with open(path) as f:
            results = json.load(f)

        all_mrr, all_r1, all_r5, all_r10, all_llm = [], [], [], [], []
        for repo in results:
            for q in repo.get("per_query", []):
                all_mrr.append(q["mrr"])
                all_r1.append(q["recall_at_1"])
                all_r5.append(q["recall_at_5"])
                all_r10.append(q["recall_at_10"])
                all_llm.append(q.get("llm_calls", 0))

        if all_mrr:
            print(f"\n  {label}:")
            print(f"    Total queries: {len(all_mrr)}")
            print(f"    MRR:     {np.mean(all_mrr):.4f} ± {np.std(all_mrr):.4f}")
            print(f"    R@1:     {np.mean(all_r1):.4f}")
            print(f"    R@5:     {np.mean(all_r5):.4f}")
            print(f"    R@10:    {np.mean(all_r10):.4f}")
            print(f"    LLM/q:   {np.mean(all_llm):.1f}")

    print(f"\n{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Full Pipeline: MCTS + RL Optimization on CodeSearchNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps (run in order, or use --step all):
  prepare         Download CodeSearchNet data and build trees
  rl-train        Train RL agent to optimize each tree (no LLM needed)
  mcts-baseline   Run MCTS search on original trees (needs LLM)
  mcts-optimized  Run MCTS search on RL-optimized trees (needs LLM)
  compare         Print comparison of baseline vs optimized

Examples:
  python run_full_pipeline.py --step rl-train --timesteps 5000
  python run_full_pipeline.py --step mcts-baseline --provider lmstudio --num-repos 5
  python run_full_pipeline.py --step all --provider lmstudio --timesteps 5000
        """
    )
    parser.add_argument("--step", required=True,
                        choices=["prepare", "rl-train", "mcts-baseline",
                                 "mcts-optimized", "compare", "all"])
    parser.add_argument("--provider", default="lmstudio", choices=["lmstudio", "ollama"])
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--num-repos", type=int, default=25)
    parser.add_argument("--queries-per-repo", type=int, default=15)
    parser.add_argument("--timesteps", type=int, default=5000,
                        help="RL training timesteps per repo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Build config
    config = DEFAULT_CONFIG.copy()
    config["num_repos"] = args.num_repos
    config["queries_per_repo"] = args.queries_per_repo
    config["rl_timesteps"] = args.timesteps
    config["seed"] = args.seed

    if args.model is None:
        model_name = "local-model" if args.provider == "lmstudio" else "qwen3:8b"
    else:
        model_name = args.model

    print("=" * 70)
    print("🧭 CODE COMPASS: FULL PIPELINE")
    print("=" * 70)
    print(f"  Step:       {args.step}")
    print(f"  Provider:   {args.provider}")
    print(f"  Repos:      {config['num_repos']}")
    print(f"  RL steps:   {config['rl_timesteps']}")
    print("=" * 70)

    start = time.time()

    # Load metadata (most steps need it)
    meta_path = Path(config["benchmark_dir"]) / "benchmark_metadata.json"
    metadata = None
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        # Limit to num_repos
        metadata = metadata[:config["num_repos"]]

    # Execute step(s)
    if args.step in ("prepare", "all"):
        metadata = step_prepare(config)
        metadata = metadata[:config["num_repos"]]

    if metadata is None:
        print("❌ No benchmark data found. Run with --step prepare first.")
        sys.exit(1)

    if args.step in ("rl-train", "all"):
        step_rl_train(config, metadata)

    if args.step in ("mcts-baseline", "all"):
        step_mcts_search(config, metadata, args.provider, model_name,
                         use_optimized=False, label="baseline")

    if args.step in ("mcts-optimized", "all"):
        step_mcts_search(config, metadata, args.provider, model_name,
                         use_optimized=True, label="optimized")

    if args.step in ("compare", "all"):
        step_compare(config)

    elapsed = time.time() - start
    print(f"\n⏱️  Pipeline step '{args.step}' completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()

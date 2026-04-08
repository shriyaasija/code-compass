#!/usr/bin/env python3
"""
Code Compass Benchmark
━━━━━━━━━━━━━━━━━━━━━━
Evaluate TreeBasedSearch vs PageIndexSemanticSearch using CodeSearchNet data.

Metrics:
  Primary:   Recall@1, Recall@5, Recall@10, MRR
  Secondary: NDCG@10, LLM calls per query

Usage:
  python benchmark.py --mode prepare         # Download data & build trees (no LLM needed)
  python benchmark.py --mode dense-only      # Evaluate dense baseline only
  python benchmark.py --mode full            # Evaluate both methods (requires Ollama/LMStudio)
  python benchmark.py --mode full --provider lmstudio   # Use LM Studio instead of Ollama

Options:
  --num-repos N           Number of repos to benchmark (default: 25)
  --queries-per-repo N    Queries per repo, 0=all (default: 15)
  --provider              LLM provider: ollama or lmstudio (default: ollama)
  --model                 LLM model name (default: qwen3:8b for ollama)
  --embedding-model       Sentence transformer model (default: all-MiniLM-L6-v2)
  --threshold             Tree search threshold (default: 0.5)
  --seed                  Random seed (default: 42)
  --output-dir            Output directory (default: benchmark_results)
  --skip-prepare          Skip data preparation if trees already exist
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsComputer:
    """Computes standard IR metrics for code search evaluation."""

    @staticmethod
    def recall_at_k(ranked_names: List[str], ground_truth_name: str, k: int) -> float:
        """
        Is the correct function in the top-k results?
        Returns 1.0 if yes, 0.0 if no.
        """
        top_k = ranked_names[:k]
        return 1.0 if ground_truth_name in top_k else 0.0

    @staticmethod
    def reciprocal_rank(ranked_names: List[str], ground_truth_name: str) -> float:
        """
        Reciprocal rank: 1/position of the correct result.
        Returns 0.0 if not found.
        """
        for i, name in enumerate(ranked_names):
            if name == ground_truth_name:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(ranked_names: List[str], ground_truth_name: str, k: int) -> float:
        """
        Normalized Discounted Cumulative Gain @ k.
        Since we have one relevant item, ideal DCG = 1.0 (relevant at position 1).
        """
        top_k = ranked_names[:k]

        # DCG: sum of rel_i / log2(i+1)
        dcg = 0.0
        for i, name in enumerate(top_k):
            if name == ground_truth_name:
                dcg += 1.0 / math.log2(i + 2)  # i+2 because i is 0-indexed

        # Ideal DCG (relevant item at position 1)
        idcg = 1.0 / math.log2(2)  # = 1.0

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def compute_all(ranked_names: List[str], ground_truth_name: str) -> Dict[str, float]:
        """Compute all metrics for a single query."""
        mc = MetricsComputer
        return {
            "recall_at_1": mc.recall_at_k(ranked_names, ground_truth_name, 1),
            "recall_at_5": mc.recall_at_k(ranked_names, ground_truth_name, 5),
            "recall_at_10": mc.recall_at_k(ranked_names, ground_truth_name, 10),
            "mrr": mc.reciprocal_rank(ranked_names, ground_truth_name),
            "ndcg_at_10": mc.ndcg_at_k(ranked_names, ground_truth_name, 10),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

class BenchmarkDataPreparer:
    """
    Downloads CodeSearchNet Python split and builds PageIndex-compatible JSON trees.

    Each CodeSearchNet entry has:
      - func_name: function name
      - func_code_string: source code
      - func_documentation_string: docstring (used as query)
      - repository_name: owner/repo
      - path: file path within the repo
    """

    def __init__(self, output_dir: str, embedding_model_name: str = "all-MiniLM-L6-v2",
                 num_repos: int = 25, min_funcs: int = 30, max_funcs: int = 200,
                 seed: int = 42):
        self.output_dir = Path(output_dir)
        self.trees_dir = self.output_dir / "trees"
        self.embedding_model_name = embedding_model_name
        self.num_repos = num_repos
        self.min_funcs = min_funcs
        self.max_funcs = max_funcs
        self.seed = seed

        self.trees_dir.mkdir(parents=True, exist_ok=True)

    def prepare(self) -> List[Dict[str, Any]]:
        """
        Full preparation pipeline.
        Returns list of repo metadata: [{repo_id, tree_path, num_functions, queries}]
        """
        print("\n" + "=" * 70)
        print("📦 PHASE 1: DATA PREPARATION")
        print("=" * 70)

        # Step 1: Download and group by repo
        repo_functions = self._download_and_group()

        # Step 2: Filter and select repos
        selected_repos = self._select_repos(repo_functions)

        # Step 3: Build trees with embeddings
        repo_metadata = self._build_trees(selected_repos)

        # Save metadata
        meta_path = self.output_dir / "benchmark_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(repo_metadata, f, indent=2)
        print(f"\n✅ Saved metadata to {meta_path}")

        return repo_metadata

    def _download_and_group(self) -> Dict[str, List[Dict]]:
        """Download CodeSearchNet Python and group by repository using streaming."""
        print("\n📥 Streaming CodeSearchNet Python split (no full download)...")

        try:
            from datasets import load_dataset
        except ImportError:
            print("❌ 'datasets' library not installed. Run: pip install datasets")
            sys.exit(1)

        # Use streaming to avoid downloading the entire ~700MB dataset
        dataset = load_dataset("code_search_net", "python", split="train",
                               streaming=True)

        # Group by repository
        print("   Streaming and grouping by repository...")
        repo_functions = defaultdict(list)
        entry_count = 0
        eligible_count = 0

        # We need repos with 30-200 functions, so keep streaming until we have
        # enough eligible repos (target: 3x the number we need, for good selection)
        target_eligible = self.num_repos * 3
        max_entries = 200_000  # Safety cap to avoid streaming forever

        for entry in dataset:
            entry_count += 1

            repo_name = entry.get("repository_name", "")
            func_name = entry.get("func_name", "")
            docstring = entry.get("func_documentation_string", "")
            code = entry.get("func_code_string", "")
            file_path = entry.get("path", "")

            # Skip entries without proper docstrings (too short to be useful queries)
            if not docstring or len(docstring.strip()) < 20:
                continue

            # Skip entries without function names
            if not func_name:
                continue

            repo_functions[repo_name].append({
                "func_name": func_name,
                "docstring": docstring.strip(),
                "code": code,
                "file_path": file_path,
            })

            # Progress reporting
            if entry_count % 20_000 == 0:
                eligible_count = sum(1 for funcs in repo_functions.values()
                                     if self.min_funcs <= len(funcs) <= self.max_funcs)
                print(f"   ... scanned {entry_count:,} entries, "
                      f"{len(repo_functions)} repos, {eligible_count} eligible "
                      f"(need {self.num_repos})")

                # Early stop if we have enough eligible repos
                if eligible_count >= target_eligible:
                    print(f"   ✅ Found enough eligible repos, stopping early")
                    break

            if entry_count >= max_entries:
                print(f"   ⏹️  Reached {max_entries:,} entry scan limit")
                break

        print(f"   Scanned {entry_count:,} entries → {len(repo_functions)} unique repositories")
        return dict(repo_functions)

    def _select_repos(self, repo_functions: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Filter repos by function count and select N repos."""
        print(f"\n🎯 Selecting {self.num_repos} repos with {self.min_funcs}-{self.max_funcs} functions...")

        # Filter by function count
        eligible = {
            repo: funcs for repo, funcs in repo_functions.items()
            if self.min_funcs <= len(funcs) <= self.max_funcs
        }
        print(f"   {len(eligible)} repos have {self.min_funcs}-{self.max_funcs} functions")

        if len(eligible) < self.num_repos:
            print(f"   ⚠️  Only {len(eligible)} eligible repos. Using all of them.")
            selected_names = list(eligible.keys())
        else:
            random.seed(self.seed)
            selected_names = random.sample(list(eligible.keys()), self.num_repos)

        selected = {name: eligible[name] for name in selected_names}

        for repo, funcs in selected.items():
            print(f"   ✓ {repo}: {len(funcs)} functions")

        return selected

    def _build_trees(self, selected_repos: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """Build PageIndex-compatible JSON trees with embeddings."""
        print(f"\n🌳 Building PageIndex trees with embeddings...")
        print(f"   Embedding model: {self.embedding_model_name}")

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.embedding_model_name)
        print(f"   ✅ Model loaded (dim={model.get_sentence_embedding_dimension()})")

        repo_metadata = []

        for idx, (repo_name, functions) in enumerate(selected_repos.items(), 1):
            safe_name = repo_name.replace("/", "__")
            print(f"\n   [{idx}/{len(selected_repos)}] {repo_name} ({len(functions)} functions)")

            # Deduplicate functions by name within same file
            seen = set()
            unique_functions = []
            for func in functions:
                key = f"{func['file_path']}::{func['func_name']}"
                if key not in seen:
                    seen.add(key)
                    unique_functions.append(func)

            # Group functions by file
            files = defaultdict(list)
            for func in unique_functions:
                files[func["file_path"]].append(func)

            # Build tree
            tree = self._build_repo_tree(repo_name, files, model)

            # Save tree (use relative path for portability across machines)
            tree_abs_path = self.trees_dir / f"{safe_name}.json"
            tree_rel_path = os.path.relpath(str(tree_abs_path), str(Path(__file__).parent))
            with open(tree_abs_path, "w") as f:
                json.dump(tree, f)
            print(f"      Saved tree to {tree_abs_path}")

            # Build query list (docstring → function name pairs)
            queries = []
            for func in unique_functions:
                queries.append({
                    "query": func["docstring"],
                    "ground_truth": func["func_name"],
                    "file_path": func["file_path"],
                })

            repo_metadata.append({
                "repo_id": safe_name,
                "repo_name": repo_name,
                "tree_path": tree_rel_path,
                "num_functions": len(unique_functions),
                "num_files": len(files),
                "queries": queries,
            })

        return repo_metadata

    def _build_repo_tree(self, repo_name: str, files: Dict[str, List[Dict]],
                         model) -> Dict:
        """Build a single repo tree in PageIndex format with embeddings."""

        def embed(text: str) -> List[float]:
            return model.encode(text).tolist()

        # Build repo summary from file names and function names
        all_func_names = []
        for file_funcs in files.values():
            all_func_names.extend([f["func_name"] for f in file_funcs])

        repo_summary = f"Repository {repo_name} containing {len(files)} files with functions: {', '.join(all_func_names[:20])}"
        if len(all_func_names) > 20:
            repo_summary += f" and {len(all_func_names) - 20} more"

        tree = {
            "node_id": "repo_root",
            "node_type": "repository",
            "repository_name": repo_name,
            "name": repo_name.split("/")[-1] if "/" in repo_name else repo_name,
            "title": repo_name.split("/")[-1] if "/" in repo_name else repo_name,
            "type": "repository",
            "repository_summary": repo_summary,
            "summary": repo_summary,
            "embedding": embed(repo_summary),
            "children": [],
            "nodes": [],
        }

        # Build file nodes
        for file_idx, (file_path, funcs) in enumerate(files.items()):
            file_name = os.path.basename(file_path)

            # File-level summary
            func_names_str = ", ".join([f["func_name"] for f in funcs])
            file_summary = f"File {file_name} containing functions: {func_names_str}"

            file_node = {
                "node_id": f"file_{file_idx}",
                "node_type": "file",
                "file_name": file_name,
                "name": file_name,
                "title": file_name,
                "type": "file_py",
                "file_path": file_path,
                "path": file_path,
                "summary": file_summary,
                "embedding": embed(file_summary),
                "children": [],
                "nodes": [],
            }

            # Build function nodes
            for func_idx, func in enumerate(funcs):
                # Use first 200 chars of docstring as summary
                summary = func["docstring"][:200]
                if len(func["docstring"]) > 200:
                    summary += "..."

                func_node = {
                    "node_id": f"func_{file_idx}_{func_idx}",
                    "node_type": "function",
                    "name": func["func_name"],
                    "title": func["func_name"],
                    "type": "function",
                    "file_path": file_path,
                    "path": file_path,
                    "signature": f"def {func['func_name']}(...)",
                    "start_line": func_idx * 20 + 1,  # Synthetic line numbers
                    "end_line": func_idx * 20 + 19,
                    "summary": summary,
                    "embedding": embed(summary),
                    "docstring": func["docstring"],
                }

                file_node["children"].append(func_node)
                file_node["nodes"].append(func_node)

            tree["children"].append(file_node)
            tree["nodes"].append(file_node)

        return tree


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

class BenchmarkRunner:
    """Runs search methods and collects metrics."""

    def __init__(self, output_dir: str, embedding_model: str = "all-MiniLM-L6-v2",
                 queries_per_repo: int = 15, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.embedding_model = embedding_model
        self.queries_per_repo = queries_per_repo
        self.seed = seed
        self.metrics = MetricsComputer()

    def run_dense_baseline(self, repo_metadata: List[Dict]) -> Dict[str, Any]:
        """Evaluate PageIndexSemanticSearch on all repos."""
        print("\n" + "=" * 70)
        print("📊 EVALUATING: Dense Baseline (PageIndexSemanticSearch)")
        print("=" * 70)

        # Import search engine
        sys.path.insert(0, str(Path(__file__).parent / "backend"))
        from pageindex_semantic_search import PageIndexSemanticSearch

        search_engine = PageIndexSemanticSearch(self.embedding_model)

        all_results = []
        per_repo_results = []

        for repo_idx, repo in enumerate(repo_metadata, 1):
            repo_id = repo["repo_id"]
            # Resolve relative path back to absolute from script directory
            tree_path = str(Path(__file__).parent / repo["tree_path"])
            queries = repo["queries"]

            print(f"\n  [{repo_idx}/{len(repo_metadata)}] {repo['repo_name']}")

            # Load tree
            search_engine.load_repository_tree(repo_id, tree_path)

            # Sample queries
            sampled_queries = self._sample_queries(queries)
            print(f"     Running {len(sampled_queries)} queries...")

            repo_metrics = defaultdict(list)
            query_details = []

            for q_idx, query_info in enumerate(sampled_queries):
                query_text = query_info["query"]
                ground_truth = query_info["ground_truth"]

                # Run search (get top 20 to compute Recall@10 etc.)
                try:
                    results = search_engine.search(repo_id, query_text, top_k=20)

                    # Extract ranked names
                    ranked_names = [r["name"] for r in results]

                    # Compute metrics
                    m = self.metrics.compute_all(ranked_names, ground_truth)

                    for key, val in m.items():
                        repo_metrics[key].append(val)

                    query_details.append({
                        "query": query_text[:100],
                        "ground_truth": ground_truth,
                        "top_5_results": ranked_names[:5],
                        "metrics": m,
                    })

                except Exception as e:
                    print(f"     ⚠️  Query {q_idx} failed: {e}")

            # Aggregate repo metrics
            avg_metrics = {key: sum(vals) / len(vals) if vals else 0.0
                           for key, vals in repo_metrics.items()}

            per_repo_results.append({
                "repo_id": repo_id,
                "repo_name": repo["repo_name"],
                "num_queries": len(sampled_queries),
                "num_functions": repo["num_functions"],
                "avg_metrics": avg_metrics,
                "query_details": query_details,
            })

            all_results.extend(
                [detail["metrics"] for detail in query_details]
            )

            # Print summary
            print(f"     R@1={avg_metrics.get('recall_at_1', 0):.3f}  "
                  f"R@5={avg_metrics.get('recall_at_5', 0):.3f}  "
                  f"R@10={avg_metrics.get('recall_at_10', 0):.3f}  "
                  f"MRR={avg_metrics.get('mrr', 0):.3f}")

            # Unload repo to save memory
            if repo_id in search_engine.repositories:
                del search_engine.repositories[repo_id]

        # Compute global averages
        global_metrics = self._aggregate_metrics(all_results)

        return {
            "method": "PageIndexSemanticSearch (Dense Baseline)",
            "embedding_model": self.embedding_model,
            "global_metrics": global_metrics,
            "per_repo": per_repo_results,
            "total_queries": len(all_results),
            "llm_calls_per_query": 0,  # No LLM calls for dense baseline
        }

    def run_tree_search(self, repo_metadata: List[Dict],
                        provider: str = "ollama",
                        model: str = "qwen3:8b",
                        threshold: float = 0.5) -> Dict[str, Any]:
        """Evaluate TreeBasedSearch on all repos."""
        print("\n" + "=" * 70)
        print("🌲 EVALUATING: Tree-Based Search (LLM-guided)")
        print(f"   Provider: {provider}, Model: {model}, Threshold: {threshold}")
        print("=" * 70)

        sys.path.insert(0, str(Path(__file__).parent / "backend"))
        from code_index import TreeBasedSearch

        # Initialize LLM client
        llm_client = self._init_llm(provider, model)
        if llm_client is None:
            print("❌ Could not connect to LLM. Skipping tree search.")
            return None

        search_engine = TreeBasedSearch(llm_client, threshold=threshold)

        all_results = []
        all_llm_calls = []
        per_repo_results = []

        for repo_idx, repo in enumerate(repo_metadata, 1):
            repo_id = repo["repo_id"]
            # Resolve relative path back to absolute from script directory
            tree_path = str(Path(__file__).parent / repo["tree_path"])
            queries = repo["queries"]

            print(f"\n  [{repo_idx}/{len(repo_metadata)}] {repo['repo_name']}")

            # Load tree
            search_engine.load_repository_tree(repo_id, tree_path)

            # Sample queries
            sampled_queries = self._sample_queries(queries)
            print(f"     Running {len(sampled_queries)} queries...")

            repo_metrics = defaultdict(list)
            repo_llm_calls = []
            query_details = []

            for q_idx, query_info in enumerate(sampled_queries):
                query_text = query_info["query"]
                ground_truth = query_info["ground_truth"]

                try:
                    # Count LLM calls by intercepting _recursive_search
                    call_counter = [0]
                    original_score = search_engine._score_siblings

                    def counting_score(*args, **kwargs):
                        call_counter[0] += 1
                        return original_score(*args, **kwargs)

                    search_engine._score_siblings = counting_score

                    # Run search
                    results = search_engine.search(repo_id, query_text)

                    # Restore original method
                    search_engine._score_siblings = original_score

                    # Extract ranked names
                    ranked_names = [r["name"] for r in results]

                    # Compute metrics
                    m = self.metrics.compute_all(ranked_names, ground_truth)
                    m["llm_calls"] = call_counter[0]

                    for key, val in m.items():
                        if key != "llm_calls":
                            repo_metrics[key].append(val)

                    repo_llm_calls.append(call_counter[0])
                    all_llm_calls.append(call_counter[0])

                    query_details.append({
                        "query": query_text[:100],
                        "ground_truth": ground_truth,
                        "top_5_results": ranked_names[:5],
                        "metrics": m,
                        "llm_calls": call_counter[0],
                    })

                    if (q_idx + 1) % 5 == 0:
                        print(f"       Completed {q_idx + 1}/{len(sampled_queries)} queries")

                except Exception as e:
                    print(f"     ⚠️  Query {q_idx} failed: {e}")
                    search_engine._score_siblings = original_score

            # Aggregate repo metrics
            avg_metrics = {key: sum(vals) / len(vals) if vals else 0.0
                           for key, vals in repo_metrics.items()}
            avg_llm_calls = sum(repo_llm_calls) / len(repo_llm_calls) if repo_llm_calls else 0

            per_repo_results.append({
                "repo_id": repo_id,
                "repo_name": repo["repo_name"],
                "num_queries": len(sampled_queries),
                "num_functions": repo["num_functions"],
                "avg_metrics": avg_metrics,
                "avg_llm_calls": avg_llm_calls,
                "query_details": query_details,
            })

            all_results.extend(
                [detail["metrics"] for detail in query_details]
            )

            # Print summary
            print(f"     R@1={avg_metrics.get('recall_at_1', 0):.3f}  "
                  f"R@5={avg_metrics.get('recall_at_5', 0):.3f}  "
                  f"R@10={avg_metrics.get('recall_at_10', 0):.3f}  "
                  f"MRR={avg_metrics.get('mrr', 0):.3f}  "
                  f"LLM calls/q={avg_llm_calls:.1f}")

            # Unload repo
            if repo_id in search_engine.repositories:
                del search_engine.repositories[repo_id]

        # Compute global averages
        global_metrics = self._aggregate_metrics(all_results)
        avg_global_llm = sum(all_llm_calls) / len(all_llm_calls) if all_llm_calls else 0

        return {
            "method": "TreeBasedSearch (LLM-guided)",
            "provider": provider,
            "model": model,
            "threshold": threshold,
            "global_metrics": global_metrics,
            "per_repo": per_repo_results,
            "total_queries": len(all_results),
            "llm_calls_per_query": avg_global_llm,
            "llm_calls_distribution": {
                "min": min(all_llm_calls) if all_llm_calls else 0,
                "max": max(all_llm_calls) if all_llm_calls else 0,
                "mean": avg_global_llm,
                "median": float(np.median(all_llm_calls)) if all_llm_calls else 0,
                "std": float(np.std(all_llm_calls)) if all_llm_calls else 0,
            },
        }

    def run_mcts_search(self, repo_metadata: List[Dict],
                        provider: str = "ollama",
                        model: str = "qwen3:8b",
                        max_iterations: int = 50,
                        c_explore: float = 1.414) -> Dict[str, Any]:
        """Evaluate MCTSTreeSearch on all repos."""
        print("\n" + "=" * 70)
        print("🌳 EVALUATING: MCTS Search")
        print(f"   Provider: {provider}, Model: {model}")
        print(f"   Iterations: {max_iterations}, c_explore: {c_explore}")
        print("=" * 70)

        sys.path.insert(0, str(Path(__file__).parent / "backend"))
        from code_index2 import MCTSTreeSearch

        # Initialize LLM client
        llm_client = self._init_llm(provider, model)
        if llm_client is None:
            print("❌ Could not connect to LLM. Skipping MCTS search.")
            return None

        search_engine = MCTSTreeSearch(llm_client, max_iterations=max_iterations, c_explore=c_explore)

        all_results = []
        all_llm_calls = []
        per_repo_results = []

        for repo_idx, repo in enumerate(repo_metadata, 1):
            repo_id = repo["repo_id"]
            tree_path = str(Path(__file__).parent / repo["tree_path"])
            queries = repo["queries"]

            print(f"\n  [{repo_idx}/{len(repo_metadata)}] {repo['repo_name']}")
            search_engine.load_repository_tree(repo_id, tree_path)
            sampled_queries = self._sample_queries(queries)
            print(f"     Running {len(sampled_queries)} queries...")

            repo_metrics = defaultdict(list)
            repo_llm_calls = []
            query_details = []

            for q_idx, query_info in enumerate(sampled_queries):
                query_text = query_info["query"]
                ground_truth = query_info["ground_truth"]

                try:
                    search_engine.mcts.verbose = False
                    results = search_engine.search(repo_id, query_text, top_k=50)
                    ranked_names = [r["name"] for r in results]

                    m = self.metrics.compute_all(ranked_names, ground_truth)
                    call_count = search_engine.mcts.llm_call_count
                    m["llm_calls"] = call_count
                    
                    # Capture MCTS stats
                    iters_used = getattr(search_engine.mcts, "actual_iterations", max_iterations)
                    early_term = 1 if getattr(search_engine.mcts, "early_terminated", False) else 0
                    m["iterations_used"] = iters_used
                    m["early_terminations"] = early_term

                    for key, val in m.items():
                        if key != "llm_calls":
                            repo_metrics[key].append(val)

                    repo_llm_calls.append(call_count)
                    all_llm_calls.append(call_count)

                    query_details.append({
                        "query": query_text[:100],
                        "ground_truth": ground_truth,
                        "top_5_results": ranked_names[:5],
                        "metrics": m,
                        "llm_calls": call_count,
                        "iterations": iters_used,
                        "early_terminated": early_term,
                    })

                    if (q_idx + 1) % 5 == 0:
                        print(f"       Completed {q_idx + 1}/{len(sampled_queries)} queries")

                except Exception as e:
                    print(f"     ⚠️  Query {q_idx} failed: {e}")

            avg_metrics = {key: sum(vals) / len(vals) if vals else 0.0
                           for key, vals in repo_metrics.items()}
            avg_llm_calls = sum(repo_llm_calls) / len(repo_llm_calls) if repo_llm_calls else 0

            per_repo_results.append({
                "repo_id": repo_id,
                "repo_name": repo["repo_name"],
                "num_queries": len(sampled_queries),
                "num_functions": repo["num_functions"],
                "avg_metrics": avg_metrics,
                "avg_llm_calls": avg_llm_calls,
                "query_details": query_details,
            })

            all_results.extend([detail["metrics"] for detail in query_details])

            print(f"     R@1={avg_metrics.get('recall_at_1', 0):.3f}  "
                  f"R@5={avg_metrics.get('recall_at_5', 0):.3f}  "
                  f"R@10={avg_metrics.get('recall_at_10', 0):.3f}  "
                  f"MRR={avg_metrics.get('mrr', 0):.3f}  "
                  f"LLM calls/q={avg_llm_calls:.1f}  "
                  f"Iters/q={avg_metrics.get('iterations_used', 0):.1f}  "
                  f"Early Term rate={avg_metrics.get('early_terminations', 0):.2f}")

            if repo_id in search_engine.repositories:
                del search_engine.repositories[repo_id]

        global_metrics = self._aggregate_metrics(all_results)
        avg_global_llm = sum(all_llm_calls) / len(all_llm_calls) if all_llm_calls else 0

        return {
            "method": "MCTSTreeSearch",
            "provider": provider,
            "model": model,
            "max_iterations": max_iterations,
            "global_metrics": global_metrics,
            "per_repo": per_repo_results,
            "total_queries": len(all_results),
            "llm_calls_per_query": avg_global_llm,
            "llm_calls_distribution": {
                "min": min(all_llm_calls) if all_llm_calls else 0,
                "max": max(all_llm_calls) if all_llm_calls else 0,
                "mean": avg_global_llm,
                "median": float(np.median(all_llm_calls)) if all_llm_calls else 0,
                "std": float(np.std(all_llm_calls)) if all_llm_calls else 0,
            },
        }

    def _init_llm(self, provider: str, model: str):
        """Initialize LLM client."""
        try:
            if provider == "ollama":
                from ollama_client import OllamaLLM
                return OllamaLLM(model=model)
            elif provider == "lmstudio":
                from lmstudio_client import LMStudioLLM
                return LMStudioLLM(model=model)
            else:
                print(f"❌ Unknown provider: {provider}")
                return None
        except Exception as e:
            print(f"❌ Failed to init LLM ({provider}): {e}")
            return None

    def _sample_queries(self, queries: List[Dict]) -> List[Dict]:
        """Sample N queries per repo, or return all if N=0."""
        if self.queries_per_repo == 0 or len(queries) <= self.queries_per_repo:
            return queries

        random.seed(self.seed)
        return random.sample(queries, self.queries_per_repo)

    def _aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, float]:
        """Compute global averages from per-query metrics."""
        if not all_metrics:
            return {}

        keys = [k for k in all_metrics[0].keys() if k != "llm_calls"]
        return {
            key: sum(m.get(key, 0) for m in all_metrics) / len(all_metrics)
            for key in keys
        }


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """Generates comparison reports in markdown and JSON."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, dense_results: Dict, tree_results: Optional[Dict] = None):
        """Generate full comparison report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw JSON results
        results = {
            "timestamp": timestamp,
            "dense_baseline": dense_results,
        }
        if tree_results:
            results["tree_search"] = tree_results

        json_path = self.output_dir / f"results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📄 Raw results saved to: {json_path}")

        # Generate markdown report
        md_path = self.output_dir / f"report_{timestamp}.md"
        self._generate_markdown(dense_results, tree_results, md_path)
        print(f"📊 Report saved to: {md_path}")

        # Print summary to console
        self._print_summary(dense_results, tree_results)

    def _generate_markdown(self, dense: Dict, tree: Optional[Dict], path: Path):
        """Generate a markdown report."""
        lines = [
            "# Code Compass Benchmark Report",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Global Metrics Summary",
            "",
        ]

        # Comparison table header
        if tree:
            lines.extend([
                "| Metric | Dense Baseline | Tree Search | Δ (Tree - Dense) |",
                "|--------|---------------|-------------|-------------------|",
            ])

            dm = dense["global_metrics"]
            tm = tree["global_metrics"]

            for metric_name, display_name in [
                ("recall_at_1", "Recall@1"),
                ("recall_at_5", "Recall@5"),
                ("recall_at_10", "Recall@10"),
                ("mrr", "MRR"),
                ("ndcg_at_10", "NDCG@10"),
            ]:
                dv = dm.get(metric_name, 0)
                tv = tm.get(metric_name, 0)
                delta = tv - dv
                sign = "+" if delta > 0 else ""
                lines.append(
                    f"| {display_name} | {dv:.4f} | {tv:.4f} | {sign}{delta:.4f} |"
                )

            lines.extend([
                "",
                f"| LLM calls/query | 0 | {tree['llm_calls_per_query']:.1f} | - |",
                f"| Total queries | {dense['total_queries']} | {tree['total_queries']} | - |",
            ])

            # LLM calls distribution
            if "llm_calls_distribution" in tree:
                dist = tree["llm_calls_distribution"]
                lines.extend([
                    "",
                    "## LLM Call Distribution (Tree Search)",
                    "",
                    f"- **Min:** {dist['min']}",
                    f"- **Max:** {dist['max']}",
                    f"- **Mean:** {dist['mean']:.1f}",
                    f"- **Median:** {dist['median']:.1f}",
                    f"- **Std Dev:** {dist['std']:.1f}",
                    "",
                    "This should follow O(log n) if the tree pruning is effective.",
                ])
        else:
            lines.extend([
                "| Metric | Dense Baseline |",
                "|--------|---------------|",
            ])
            dm = dense["global_metrics"]
            for metric_name, display_name in [
                ("recall_at_1", "Recall@1"),
                ("recall_at_5", "Recall@5"),
                ("recall_at_10", "Recall@10"),
                ("mrr", "MRR"),
                ("ndcg_at_10", "NDCG@10"),
            ]:
                lines.append(f"| {display_name} | {dm.get(metric_name, 0):.4f} |")

        # Per-repo breakdown
        lines.extend([
            "",
            "## Per-Repository Breakdown",
            "",
        ])

        # Dense per-repo
        lines.append("### Dense Baseline")
        lines.append("")
        lines.append("| Repository | Funcs | Queries | R@1 | R@5 | R@10 | MRR |")
        lines.append("|-----------|-------|---------|-----|-----|------|-----|")
        for repo in dense["per_repo"]:
            m = repo["avg_metrics"]
            lines.append(
                f"| {repo['repo_name'][:40]} | {repo['num_functions']} | "
                f"{repo['num_queries']} | {m.get('recall_at_1', 0):.3f} | "
                f"{m.get('recall_at_5', 0):.3f} | {m.get('recall_at_10', 0):.3f} | "
                f"{m.get('mrr', 0):.3f} |"
            )

        # Tree per-repo
        if tree:
            lines.append("")
            lines.append("### Tree Search")
            lines.append("")
            lines.append("| Repository | Funcs | Queries | R@1 | R@5 | R@10 | MRR | LLM/q |")
            lines.append("|-----------|-------|---------|-----|-----|------|-----|-------|")
            for repo in tree["per_repo"]:
                m = repo["avg_metrics"]
                llm_q = repo.get("avg_llm_calls", 0)
                lines.append(
                    f"| {repo['repo_name'][:40]} | {repo['num_functions']} | "
                    f"{repo['num_queries']} | {m.get('recall_at_1', 0):.3f} | "
                    f"{m.get('recall_at_5', 0):.3f} | {m.get('recall_at_10', 0):.3f} | "
                    f"{m.get('mrr', 0):.3f} | {llm_q:.1f} |"
                )

        with open(path, "w") as f:
            f.write("\n".join(lines))

    def _print_summary(self, dense: Dict, tree: Optional[Dict]):
        """Print summary to console."""
        print("\n" + "=" * 70)
        print("📊 BENCHMARK RESULTS SUMMARY")
        print("=" * 70)

        dm = dense["global_metrics"]
        print(f"\n  Dense Baseline ({dense['total_queries']} queries):")
        print(f"    Recall@1:  {dm.get('recall_at_1', 0):.4f}")
        print(f"    Recall@5:  {dm.get('recall_at_5', 0):.4f}")
        print(f"    Recall@10: {dm.get('recall_at_10', 0):.4f}")
        print(f"    MRR:       {dm.get('mrr', 0):.4f}")
        print(f"    NDCG@10:   {dm.get('ndcg_at_10', 0):.4f}")

        if tree:
            tm = tree["global_metrics"]
            print(f"\n  Tree Search ({tree['total_queries']} queries):")
            print(f"    Recall@1:  {tm.get('recall_at_1', 0):.4f}")
            print(f"    Recall@5:  {tm.get('recall_at_5', 0):.4f}")
            print(f"    Recall@10: {tm.get('recall_at_10', 0):.4f}")
            print(f"    MRR:       {tm.get('mrr', 0):.4f}")
            print(f"    NDCG@10:   {tm.get('ndcg_at_10', 0):.4f}")
            print(f"    LLM calls/query: {tree['llm_calls_per_query']:.1f}")

            print(f"\n  Δ (Tree - Dense):")
            for metric in ["recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"]:
                delta = tm.get(metric, 0) - dm.get(metric, 0)
                sign = "+" if delta > 0 else ""
                display = metric.replace("_", "@").replace("at@", "@").upper()
                print(f"    {display}: {sign}{delta:.4f}")

        print("\n" + "=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Code Compass Benchmark: Evaluate search methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --mode prepare
  python benchmark.py --mode dense-only
  python benchmark.py --mode full --provider ollama --model qwen3:8b
  python benchmark.py --mode full --provider lmstudio
        """
    )

    parser.add_argument("--mode", required=True,
                        choices=["prepare", "dense-only", "full", "mcts"],
                        help="Benchmark mode")
    parser.add_argument("--num-repos", type=int, default=25,
                        help="Number of repos to benchmark (default: 25)")
    parser.add_argument("--queries-per-repo", type=int, default=15,
                        help="Queries per repo, 0=all (default: 15)")
    parser.add_argument("--provider", default="ollama",
                        choices=["ollama", "lmstudio"],
                        help="LLM provider (default: ollama)")
    parser.add_argument("--model", default=None,
                        help="LLM model name (default: qwen3:8b for ollama)")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                        help="Sentence transformer model (default: all-MiniLM-L6-v2)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Tree search threshold (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output-dir", default="benchmark_results",
                        help="Output directory (default: benchmark_results)")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip data preparation if trees already exist")

    args = parser.parse_args()

    # Default model per provider
    if args.model is None:
        args.model = "qwen3:8b" if args.provider == "ollama" else "local-model"

    output_dir = str(Path(__file__).parent / args.output_dir)

    print("=" * 70)
    print("🧭 CODE COMPASS BENCHMARK")
    print("=" * 70)
    print(f"  Mode:            {args.mode}")
    print(f"  Repos:           {args.num_repos}")
    print(f"  Queries/repo:    {args.queries_per_repo}")
    print(f"  Embedding model: {args.embedding_model}")
    print(f"  Output dir:      {output_dir}")
    if args.mode == "full":
        print(f"  LLM provider:    {args.provider}")
        print(f"  LLM model:       {args.model}")
        print(f"  Threshold:       {args.threshold}")
    print("=" * 70)

    start_time = time.time()

    # ── DATA PREPARATION ──
    meta_path = Path(output_dir) / "benchmark_metadata.json"

    if args.mode == "prepare" or (not args.skip_prepare and not meta_path.exists()):
        preparer = BenchmarkDataPreparer(
            output_dir=output_dir,
            embedding_model_name=args.embedding_model,
            num_repos=args.num_repos,
            seed=args.seed,
        )
        repo_metadata = preparer.prepare()

        if args.mode == "prepare":
            elapsed = time.time() - start_time
            print(f"\n✅ Data preparation complete in {elapsed:.1f}s")
            print(f"   {len(repo_metadata)} repos with trees saved to {output_dir}/trees/")
            print(f"   Run benchmark with: python benchmark.py --mode dense-only --skip-prepare")
            return
    else:
        print(f"\n📂 Loading existing metadata from {meta_path}")
        with open(meta_path, "r") as f:
            repo_metadata = json.load(f)
        print(f"   Found {len(repo_metadata)} repos")

    # ── BENCHMARK EVALUATION ──
    runner = BenchmarkRunner(
        output_dir=output_dir,
        embedding_model=args.embedding_model,
        queries_per_repo=args.queries_per_repo,
        seed=args.seed,
    )

    # Run dense baseline
    dense_results = runner.run_dense_baseline(repo_metadata)

    # Run tree search (if mode=full)
    tree_results = None
    if args.mode == "full":
        tree_results = runner.run_tree_search(
            repo_metadata,
            provider=args.provider,
            model=args.model,
            threshold=args.threshold,
        )
        
    mcts_results = None
    if args.mode == "mcts":
        mcts_results = runner.run_mcts_search(
            repo_metadata,
            provider=args.provider,
            model=args.model,
        )

    # ── REPORT ──
    reporter = ReportGenerator(output_dir)
    reporter.generate(dense_results, tree_results or mcts_results)

    elapsed = time.time() - start_time
    print(f"\n⏱️  Total benchmark time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"📁 All results in: {output_dir}/")


if __name__ == "__main__":
    main()

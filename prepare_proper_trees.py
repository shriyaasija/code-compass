"""
Prepare Proper Trees for Benchmarking.

Replaces the flat CodeSearchNet trees with real repo structure:
  1. Clone repos from GitHub
  2. Parse with tree-sitter (real folder/file/class/function hierarchy)
  3. LLM bottom-up summarization (each node gets a meaningful summary)
  4. Embed all summaries with SentenceTransformer
  5. Save proper trees + metadata for MCTS + RL benchmarking

Usage:
  python prepare_proper_trees.py --provider lmstudio --num-repos 25
  python prepare_proper_trees.py --provider lmstudio --num-repos 3  # quick test
"""

import argparse
import json
import os
import subprocess
import sys
import time
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

from backend.tree_builder import build_directory_tree
from backend.code_parser import CodeParser, enrich_tree_with_code_structure
from backend.summarizer import TreeSummarizer
from backend.embed_tree import TreeEmbedder


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CLONE_DIR = "benchmark_results/cloned_repos"
PROPER_TREES_DIR = "benchmark_results/proper_trees"
METADATA_PATH = "benchmark_results/benchmark_metadata.json"
PROPER_METADATA_PATH = "benchmark_results/proper_benchmark_metadata.json"


def clone_repo(repo_name: str, target_dir: str) -> str:
    """Clone a GitHub repo (shallow). Returns path to cloned repo."""
    repo_url = f"https://github.com/{repo_name}.git"
    safe_name = repo_name.replace("/", "__")
    repo_path = os.path.join(target_dir, safe_name)
    
    if os.path.exists(repo_path):
        print(f"   ✅ Already cloned: {safe_name}")
        return repo_path
    
    print(f"   📥 Cloning {repo_name}...")
    try:
        subprocess.run(
            ['git', 'clone', '--depth', '1', repo_url, repo_path],
            check=True, capture_output=True, timeout=120
        )
        print(f"   ✅ Cloned to {repo_path}")
        return repo_path
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Clone failed: {e.stderr.decode()[:200]}")
        return None
    except subprocess.TimeoutExpired:
        print(f"   ❌ Clone timed out (120s)")
        return None


def count_tree_nodes(node: Dict) -> Dict[str, int]:
    """Count nodes by type in a tree."""
    counts = defaultdict(int)
    
    def _walk(n):
        ntype = n.get('type', n.get('node_type', 'unknown'))
        counts[ntype] += 1
        for child in n.get('nodes', n.get('children', [])):
            _walk(child)
    
    _walk(node)
    return dict(counts)


def tree_depth(node: Dict) -> int:
    """Compute max depth of tree."""
    children = node.get('nodes', node.get('children', []))
    if not children:
        return 0
    return 1 + max(tree_depth(c) for c in children)


def prepare_proper_trees(
    provider: str = "lmstudio",
    model_name: str = None,
    num_repos: int = 25,
    skip_clone: bool = False,
    skip_summarize: bool = False,
):
    """Full pipeline: clone → parse → summarize → embed → save."""
    
    print("\n" + "=" * 70)
    print("🌳 PROPER TREE PREPARATION PIPELINE")
    print("=" * 70)
    
    # ─── Load existing metadata (has repo names + queries) ───
    if not os.path.exists(METADATA_PATH):
        print(f"❌ No metadata found at {METADATA_PATH}")
        print("   Run 'python benchmark.py --mode prepare' first")
        sys.exit(1)
    
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    metadata = metadata[:num_repos]
    print(f"\n  Repos to process: {len(metadata)}")
    
    # ─── Create directories ───
    os.makedirs(CLONE_DIR, exist_ok=True)
    os.makedirs(PROPER_TREES_DIR, exist_ok=True)
    
    # ─── Initialize tools ───
    print("\n🔧 Initializing tools...")
    parser = CodeParser()
    
    if not skip_summarize:
        if provider == "lmstudio":
            from backend.lmstudio_client import LMStudioLLM
            llm = LMStudioLLM(model=model_name or "local-model")
        else:
            from backend.ollama_client import OllamaLLM
            llm = OllamaLLM(model=model_name or "qwen3:8b")
        summarizer = TreeSummarizer(llm, verbose=True)
    
    embedder = TreeEmbedder()
    
    # ─── Process each repo ───
    proper_metadata = []
    
    for idx, repo in enumerate(metadata, 1):
        repo_name = repo["repo_name"]
        repo_id = repo["repo_id"]
        
        print(f"\n{'─'*70}")
        print(f"  [{idx}/{len(metadata)}] {repo_name}")
        print(f"{'─'*70}")
        
        # STEP 1: Clone
        if not skip_clone:
            repo_path = clone_repo(repo_name, CLONE_DIR)
            if repo_path is None:
                print(f"   ⚠️ Skipping {repo_name} (clone failed)")
                continue
        else:
            repo_path = os.path.join(CLONE_DIR, repo_id)
            if not os.path.exists(repo_path):
                print(f"   ⚠️ Skipping {repo_name} (not cloned)")
                continue
        
        # STEP 2: Build directory tree
        print(f"   🌳 Building directory tree...")
        tree_obj = build_directory_tree(repo_path)
        
        # STEP 3: Parse with tree-sitter
        print(f"   🧩 Parsing code structure...")
        tree_obj = enrich_tree_with_code_structure(tree_obj, parser)
        
        # Convert to dict
        tree_dict = tree_obj.to_dict()
        
        # Count what we found
        node_counts = count_tree_nodes(tree_dict)
        depth = tree_depth(tree_dict)
        print(f"   📊 Tree: depth={depth}, nodes={node_counts}")
        
        # STEP 4: LLM bottom-up summarization
        if not skip_summarize:
            print(f"   📝 Generating LLM summaries (bottom-up)...")
            tree_dict = summarizer.summarize_tree(tree_dict, repo_path)
        
        # STEP 5: Embed all summaries
        print(f"   🔢 Embedding summaries...")
        tree_dict = embedder.embed_tree(tree_dict)
        
        # STEP 6: Save proper tree
        tree_path = os.path.join(PROPER_TREES_DIR, f"{repo_id}.json")
        with open(tree_path, 'w') as f:
            json.dump(tree_dict, f)
        tree_size_mb = os.path.getsize(tree_path) / (1024 * 1024)
        print(f"   💾 Saved: {tree_path} ({tree_size_mb:.1f} MB)")
        
        # Build proper metadata entry (keep queries from original)
        proper_metadata.append({
            "repo_id": repo_id,
            "repo_name": repo_name,
            "tree_path": tree_path,
            "repo_path": repo_path,
            "num_files": node_counts.get('file_py', 0) + node_counts.get('file_js', 0),
            "num_functions": node_counts.get('function', 0) + node_counts.get('method', 0),
            "num_classes": node_counts.get('class', 0),
            "tree_depth": depth,
            "total_nodes": sum(node_counts.values()),
            "queries": repo.get("queries", []),  # Keep original queries
        })
    
    # ─── Save proper metadata ───
    with open(PROPER_METADATA_PATH, 'w') as f:
        json.dump(proper_metadata, f, indent=2)
    print(f"\n✅ Proper metadata saved to: {PROPER_METADATA_PATH}")
    print(f"   {len(proper_metadata)} repos processed")
    
    # ─── Print summary ───
    print(f"\n{'='*70}")
    print("📊 SUMMARY")
    print(f"{'='*70}")
    for r in proper_metadata:
        print(f"  {r['repo_name']}: depth={r['tree_depth']}, "
              f"funcs={r['num_functions']}, files={r['num_files']}, "
              f"nodes={r['total_nodes']}")


def main():
    p = argparse.ArgumentParser(description="Prepare proper benchmark trees")
    p.add_argument("--provider", default="lmstudio", choices=["lmstudio", "ollama"])
    p.add_argument("--model", default=None, help="LLM model name")
    p.add_argument("--num-repos", type=int, default=25)
    p.add_argument("--skip-clone", action="store_true",
                   help="Skip git clone (use existing clones)")
    p.add_argument("--skip-summarize", action="store_true",
                   help="Skip LLM summarization (just parse + embed)")
    args = p.parse_args()
    
    prepare_proper_trees(
        provider=args.provider,
        model_name=args.model,
        num_repos=args.num_repos,
        skip_clone=args.skip_clone,
        skip_summarize=args.skip_summarize,
    )


if __name__ == "__main__":
    main()
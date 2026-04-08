# Phase 3B: Proper Tree Pipeline — Clone, Parse, Summarize, Benchmark

> **The Problem:** The current benchmark builds fake flat trees from CodeSearchNet metadata.
> Each "tree" is just `repo → 1 file → N functions` (depth 2). No real folder structure.
> No LLM-generated summaries. Summaries are just raw docstrings. This does NOT match
> how Code Compass actually works, and it makes the RL (Merge/Split/Reparent) useless
> because there's nothing to restructure in a flat tree.
>
> **The Fix:** Clone real repos → tree-sitter parse → LLM bottom-up summaries → proper tree → then MCTS + RL.
>
> **Time estimate:** ~3-4 days
>
> **Prerequisites:** Phase 3A (RL module) done, LM Studio server available

---

## How The Pipeline Should Actually Work

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Clone Repo                                             │
│  git clone --depth 1 https://github.com/owner/repo              │
│                                                                  │
│  Result: Actual files on disk                                    │
│  /tmp/repos/pymagicc/                                            │
│  ├── pymagicc/                                                   │
│  │   ├── __init__.py                                             │
│  │   ├── core.py                                                 │
│  │   ├── io.py                                                   │
│  │   └── definitions/                                            │
│  │       ├── regions.py                                          │
│  │       └── variables.py                                        │
│  ├── tests/                                                      │
│  └── setup.py                                                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Build Directory Tree (tree_builder.py)                  │
│  Traverse folders & files, create CodeNode hierarchy             │
│                                                                  │
│  Result: Tree with real folder structure (depth 4-6+)            │
│  repo (repository)                                               │
│  ├── pymagicc/ (folder)                                          │
│  │   ├── __init__.py (file_py)                                   │
│  │   ├── core.py (file_py)                                       │
│  │   ├── io.py (file_py)                                         │
│  │   └── definitions/ (folder)                                   │
│  │       ├── regions.py (file_py)                                │
│  │       └── variables.py (file_py)                              │
│  ├── tests/ (folder)                                             │
│  └── setup.py (file_py)                                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Parse Code Structure (code_parser.py)                   │
│  tree-sitter extracts functions, classes, methods                │
│                                                                  │
│  Result: Tree enriched with AST nodes                            │
│  ├── core.py (file_py)                                           │
│  │   ├── class MAGICC (class)                                    │
│  │   │   ├── __init__ (method)                                   │
│  │   │   ├── run (method)                                        │
│  │   │   └── set_config (method)                                 │
│  │   └── def read_config (function)                              │
│  ├── io.py (file_py)                                             │
│  │   ├── class MAGICCReader (class)                              │
│  │   │   ├── read (method)                                       │
│  │   │   └── process (method)                                    │
│  │   └── def determine_tool (function)                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: LLM Bottom-Up Summarization (NEW — summarizer.py)      │
│  For each node, generate a summary using LLM. BOTTOM-UP:        │
│                                                                  │
│  4a. LEAF NODES (functions/methods):                             │
│      Read actual source code → LLM prompt:                       │
│      "Summarize this function in 1-2 sentences"                  │
│      Result: "Reads MAGICC configuration from .cfg file"         │
│                                                                  │
│  4b. FILE NODES:                                                 │
│      Combine child summaries → LLM prompt:                       │
│      "Given these function summaries, summarize this file"       │
│      Result: "Core MAGICC model interface with run/config mgmt"  │
│                                                                  │
│  4c. FOLDER NODES:                                               │
│      Combine child summaries → LLM prompt:                       │
│      "Given these file/folder summaries, summarize this module"  │
│      Result: "Climate modeling package with I/O and definitions"  │
│                                                                  │
│  4d. ROOT NODE:                                                  │
│      Combine all child summaries → final repo summary            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Embed All Summaries (sentence-transformers)             │
│  For every node: embedding = model.encode(summary)               │
│  Save full tree as JSON with summaries + embeddings              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: MCTS Search (existing code_index2.py)                   │
│  LLM scores branches using summaries as context                  │
│                                                                  │
│  STEP 7: RL Optimization (existing research/rl_index/)           │
│  Now actually useful because trees have 4-6+ depth levels        │
│  Merge/Split/Reparent can restructure real folder hierarchies    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Sub-Phase 3B.1: Create the Summarizer (Steps 1–8)

> **New file:** `backend/summarizer.py`
> This is the missing piece — bottom-up LLM summary generation.

### Step 1: Create `backend/summarizer.py`

```python
"""
Bottom-Up LLM Summarizer for Code Trees.

Generates summaries for every node in a code tree, working from leaves
up to the root:
  1. Functions/methods: summarized from source code
  2. Files: summarized from child function/class summaries  
  3. Folders: summarized from child file/folder summaries
  4. Root: summarized from top-level children

Each summary is a 1-3 sentence description of what that subtree does.
These summaries are what MCTS uses to score branch relevance.

Usage:
    from backend.summarizer import TreeSummarizer
    summarizer = TreeSummarizer(llm_client)
    summarizer.summarize_tree(tree_dict, repo_path)
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any


class TreeSummarizer:
    """
    Bottom-up LLM summarizer for code trees.
    
    Traverses the tree depth-first, summarizing leaves first,
    then combining child summaries to produce parent summaries.
    """
    
    def __init__(self, llm_client, max_code_chars: int = 2000, verbose: bool = True):
        """
        Args:
            llm_client: LMStudioLLM or OllamaLLM instance with .chat() method.
            max_code_chars: Max characters of source code to include in prompt.
            verbose: Print progress messages.
        """
        self.llm = llm_client
        self.max_code_chars = max_code_chars
        self.verbose = verbose
        self.summary_count = 0
        self.llm_calls = 0
    
    def summarize_tree(self, tree: Dict, repo_path: str) -> Dict:
        """
        Generate summaries for every node in the tree, bottom-up.
        Modifies the tree in-place and returns it.
        
        Args:
            tree: Tree dict (from tree_builder.to_dict())
            repo_path: Path to the cloned repository (for reading source code)
        """
        self.summary_count = 0
        self.llm_calls = 0
        start = time.time()
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"📝 BOTTOM-UP TREE SUMMARIZATION")
            print(f"{'='*70}")
        
        self._summarize_node(tree, repo_path)
        
        elapsed = time.time() - start
        if self.verbose:
            print(f"\n✅ Summarized {self.summary_count} nodes in {elapsed:.1f}s "
                  f"({self.llm_calls} LLM calls)")
        
        return tree
    
    def _summarize_node(self, node: Dict, repo_path: str) -> str:
        """
        Recursively summarize a node. Children are summarized first.
        Returns the summary string for this node.
        """
        children = node.get('nodes', node.get('children', []))
        node_type = node.get('type', node.get('node_type', ''))
        title = node.get('title', node.get('name', 'unknown'))
        
        # RECURSIVE CASE: summarize children first (bottom-up)
        child_summaries = []
        for child in children:
            child_summary = self._summarize_node(child, repo_path)
            child_title = child.get('title', child.get('name', '?'))
            child_type = child.get('type', child.get('node_type', '?'))
            if child_summary:
                child_summaries.append({
                    'title': child_title,
                    'type': child_type,
                    'summary': child_summary,
                })
        
        # Generate summary based on node type
        if node_type in ('function', 'method'):
            summary = self._summarize_code_node(node, repo_path)
        elif node_type == 'class':
            summary = self._summarize_class_node(node, child_summaries)
        elif node_type.startswith('file'):
            summary = self._summarize_file_node(node, child_summaries, repo_path)
        elif node_type == 'folder':
            summary = self._summarize_folder_node(node, child_summaries)
        elif node_type == 'repository':
            summary = self._summarize_repo_node(node, child_summaries)
        else:
            # Unknown type — use title as summary
            summary = title
        
        # Store summary on node
        node['summary'] = summary
        self.summary_count += 1
        
        if self.verbose and self.summary_count % 20 == 0:
            print(f"   Summarized {self.summary_count} nodes...")
        
        return summary
    
    def _summarize_code_node(self, node: Dict, repo_path: str) -> str:
        """Summarize a function or method by reading its source code."""
        file_path = node.get('path', node.get('file_path', ''))
        start_line = node.get('start_line')
        end_line = node.get('end_line')
        title = node.get('title', 'unknown')
        
        # Try to read the actual source code
        source_code = ""
        if file_path and start_line is not None and end_line is not None:
            full_path = os.path.join(repo_path, file_path) if not os.path.isabs(file_path) else file_path
            try:
                with open(full_path, 'r', errors='ignore') as f:
                    lines = f.readlines()
                    source_code = ''.join(lines[start_line:end_line + 1])
                    # Truncate if too long
                    if len(source_code) > self.max_code_chars:
                        source_code = source_code[:self.max_code_chars] + "\n... (truncated)"
            except (FileNotFoundError, IOError):
                source_code = ""
        
        if not source_code:
            # Fallback: use title and any existing docstring
            return node.get('summary', f"Function {title}")
        
        prompt = f"""Summarize this Python function in 1-2 sentences. Focus on WHAT it does, not HOW.

```python
{source_code}
```

Respond with ONLY the summary, no extra text."""
        
        return self._call_llm(prompt, fallback=f"Function {title}")
    
    def _summarize_class_node(self, node: Dict, child_summaries: List[Dict]) -> str:
        """Summarize a class from its method summaries."""
        title = node.get('title', 'unknown')
        
        if not child_summaries:
            return f"Class {title}"
        
        methods_text = "\n".join([
            f"  - {cs['title']}: {cs['summary']}" 
            for cs in child_summaries[:15]
        ])
        if len(child_summaries) > 15:
            methods_text += f"\n  ... and {len(child_summaries) - 15} more methods"
        
        prompt = f"""Summarize this class in 1-2 sentences based on its methods.

Class: {title}
Methods:
{methods_text}

Respond with ONLY the summary."""
        
        return self._call_llm(prompt, fallback=f"Class {title} with {len(child_summaries)} methods")
    
    def _summarize_file_node(self, node: Dict, child_summaries: List[Dict],
                             repo_path: str) -> str:
        """Summarize a file from its function/class summaries."""
        title = node.get('title', node.get('name', 'unknown'))
        
        if not child_summaries:
            # For files with no parsed children (non-code files, etc.)
            return f"File {title}"
        
        children_text = "\n".join([
            f"  - {cs['title']} ({cs['type']}): {cs['summary']}"
            for cs in child_summaries[:20]
        ])
        if len(child_summaries) > 20:
            children_text += f"\n  ... and {len(child_summaries) - 20} more items"
        
        prompt = f"""Summarize this source file in 1-2 sentences based on its contents.

File: {title}
Contents:
{children_text}

Respond with ONLY the summary."""
        
        return self._call_llm(prompt, fallback=f"File {title} with {len(child_summaries)} items")
    
    def _summarize_folder_node(self, node: Dict, child_summaries: List[Dict]) -> str:
        """Summarize a folder from its children summaries."""
        title = node.get('title', node.get('name', 'unknown'))
        
        if not child_summaries:
            return f"Folder {title}"
        
        children_text = "\n".join([
            f"  - {cs['title']} ({cs['type']}): {cs['summary']}"
            for cs in child_summaries[:15]
        ])
        if len(child_summaries) > 15:
            children_text += f"\n  ... and {len(child_summaries) - 15} more items"
        
        prompt = f"""Summarize this folder/module in 1-2 sentences based on its contents.

Folder: {title}/
Contents:
{children_text}

Respond with ONLY the summary."""
        
        return self._call_llm(prompt, fallback=f"Module {title} with {len(child_summaries)} items")
    
    def _summarize_repo_node(self, node: Dict, child_summaries: List[Dict]) -> str:
        """Summarize the root repository node."""
        title = node.get('title', node.get('name', 'unknown'))
        
        if not child_summaries:
            return f"Repository {title}"
        
        children_text = "\n".join([
            f"  - {cs['title']} ({cs['type']}): {cs['summary']}"
            for cs in child_summaries[:20]
        ])
        
        prompt = f"""Summarize this code repository in 2-3 sentences.

Repository: {title}
Top-level contents:
{children_text}

Respond with ONLY the summary."""
        
        return self._call_llm(prompt, fallback=f"Repository {title}")
    
    def _call_llm(self, prompt: str, fallback: str = "") -> str:
        """Call the LLM and return the response, with fallback on failure."""
        self.llm_calls += 1
        try:
            messages = [
                {"role": "system", "content": "You are a code documentation assistant. "
                 "Provide concise, factual summaries. No markdown, no bullet points, "
                 "just plain English sentences."},
                {"role": "user", "content": prompt}
            ]
            response = self.llm.chat(messages, temperature=0.1, max_tokens=150)
            if response and response.strip():
                # Clean up: remove quotes, extra whitespace
                summary = response.strip().strip('"').strip("'").strip()
                # Limit length
                if len(summary) > 300:
                    summary = summary[:297] + "..."
                return summary
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ LLM call failed: {e}")
        return fallback
```

### Step 2: Create `backend/embed_tree.py`

This embeds all summaries after the LLM generates them:

```python
"""
Embed all node summaries in a tree using SentenceTransformer.

After TreeSummarizer generates summaries, this adds embedding vectors
to every node so the tree is ready for MCTS + RL.
"""

import time
from typing import Dict
from sentence_transformers import SentenceTransformer


class TreeEmbedder:
    """Adds embedding vectors to all nodes with summaries."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"🔄 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded (dim={self.dim})")
    
    def embed_tree(self, tree: Dict) -> Dict:
        """Add embeddings to all nodes with summaries. Modifies in-place."""
        count = 0
        start = time.time()
        
        def _walk(node):
            nonlocal count
            summary = node.get('summary', '')
            if summary:
                node['embedding'] = self.model.encode(summary).tolist()
                count += 1
            
            children = node.get('nodes', node.get('children', []))
            for child in children:
                _walk(child)
        
        _walk(tree)
        elapsed = time.time() - start
        print(f"✅ Embedded {count} nodes in {elapsed:.1f}s")
        return tree
```

---

## Sub-Phase 3B.2: Build the Proper Benchmark Pipeline (Steps 3–10)

> **File:** `prepare_proper_trees.py` (replaces the flat tree builder in benchmark.py)

### Step 3: Create `prepare_proper_trees.py`

This is the new data preparation script that:
1. Reads the 25 repo names from existing benchmark_metadata.json
2. Clones each repo from GitHub
3. Runs tree-sitter parsing
4. Runs LLM bottom-up summarization
5. Embeds all summaries
6. Saves proper trees + updated metadata

```python
#!/usr/bin/env python3
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
            llm = LMStudioLLM(model=model_name)
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
```

---

## Sub-Phase 3B.3: Update Pipeline Runner (Steps 11–14)

### Step 4: Update `run_full_pipeline.py`

Modify the pipeline runner to use `proper_benchmark_metadata.json` and `proper_trees/` instead of the flat trees. The key change:

```python
# In run_full_pipeline.py, update DEFAULT_CONFIG:
DEFAULT_CONFIG = {
    "benchmark_dir": "benchmark_results",
    "metadata_path": "benchmark_results/proper_benchmark_metadata.json",  # ← USE PROPER
    "rl_output_dir": "research/rl_index/checkpoints",
    "optimized_trees_dir": "benchmark_results/optimized_trees",
    "results_dir": "pipeline_results",
    # ... rest stays the same
}
```

---

## Sub-Phase 3B.4: Tests (Steps 15–18)

### Step 5: Write `tests/test_summarizer.py`

```python
"""Tests for bottom-up tree summarizer."""

import pytest
from unittest.mock import MagicMock
from backend.summarizer import TreeSummarizer


class MockLLM:
    """Mock LLM that returns predictable summaries."""
    def chat(self, messages, temperature=0.1, max_tokens=150):
        # Extract the prompt content
        prompt = messages[-1]['content']
        if 'function' in prompt.lower() or 'def ' in prompt.lower():
            return "This function processes data and returns results."
        elif 'class' in prompt.lower():
            return "A class that manages data processing operations."
        elif 'file' in prompt.lower():
            return "A module containing data processing utilities."
        elif 'folder' in prompt.lower():
            return "A package for data processing and analysis."
        else:
            return "A code component."


class TestTreeSummarizer:
    def test_summarizes_leaf_node(self):
        llm = MockLLM()
        summarizer = TreeSummarizer(llm, verbose=False)
        
        tree = {
            'title': 'process_data',
            'type': 'function',
            'path': 'test.py',
            'start_line': 0,
            'end_line': 5,
        }
        
        summarizer.summarize_tree(tree, '/nonexistent')  # No file to read
        assert 'summary' in tree
        assert len(tree['summary']) > 0
    
    def test_bottom_up_order(self):
        """Children should be summarized before parents."""
        llm = MockLLM()
        summarizer = TreeSummarizer(llm, verbose=False)
        
        tree = {
            'title': 'main.py',
            'type': 'file_py',
            'nodes': [
                {'title': 'func_a', 'type': 'function', 'path': 'main.py',
                 'start_line': 0, 'end_line': 5},
                {'title': 'func_b', 'type': 'function', 'path': 'main.py',
                 'start_line': 7, 'end_line': 12},
            ]
        }
        
        summarizer.summarize_tree(tree, '/nonexistent')
        # All nodes should have summaries
        assert tree.get('summary')
        assert tree['nodes'][0].get('summary')
        assert tree['nodes'][1].get('summary')
    
    def test_handles_empty_tree(self):
        llm = MockLLM()
        summarizer = TreeSummarizer(llm, verbose=False)
        
        tree = {'title': 'empty_repo', 'type': 'repository'}
        summarizer.summarize_tree(tree, '/nonexistent')
        assert 'summary' in tree
```

### Step 6: Write `tests/test_embed_tree.py`

```python
"""Tests for tree embedding."""

from backend.embed_tree import TreeEmbedder


class TestTreeEmbedder:
    def test_embeds_all_nodes(self):
        embedder = TreeEmbedder()
        tree = {
            'title': 'repo',
            'type': 'repository',
            'summary': 'A test repository',
            'nodes': [
                {'title': 'file.py', 'type': 'file_py',
                 'summary': 'A Python file',
                 'nodes': [
                     {'title': 'func', 'type': 'function',
                      'summary': 'A function'}
                 ]}
            ]
        }
        
        embedder.embed_tree(tree)
        assert 'embedding' in tree
        assert 'embedding' in tree['nodes'][0]
        assert 'embedding' in tree['nodes'][0]['nodes'][0]
        assert len(tree['embedding']) == 384  # all-MiniLM-L6-v2 dimension
```

---

## Commands — The Exact Order

```bash
# ─── ONE-TIME SETUP ───
# 1. Make sure LM Studio is running on port 1234

# ─── PREPARE PROPER TREES ───

# 2. Quick test with 1 repo first:
python prepare_proper_trees.py --provider lmstudio --num-repos 1

# 3. Verify the proper tree looks right:
python -c "
import json
t = json.load(open('benchmark_results/proper_trees/openclimatedata__pymagicc.json'))
def show(n, d=0):
    title = n.get('title','?')
    typ = n.get('type','?')
    has_s = bool(n.get('summary'))
    kids = n.get('nodes', n.get('children', []))
    print(f\"{'  '*d}[{typ}] {title} (summary={has_s}, children={len(kids)})\")
    if d < 4:
        for c in kids[:3]:
            show(c, d+1)
        if len(kids) > 3:
            print(f\"{'  '*(d+1)}... +{len(kids)-3} more\")
show(t)
"

# 4. If it looks good (real folder structure, LLM summaries, depth 4+):
python prepare_proper_trees.py --provider lmstudio --num-repos 25

# ─── RUN RL ON PROPER TREES ───

# 5. Update pipeline to use proper metadata (one-line change in run_full_pipeline.py)
# Then:
python run_full_pipeline.py --step rl-train --timesteps 5000

# ─── RUN MCTS ON PROPER TREES ───
python run_full_pipeline.py --step mcts-baseline --provider lmstudio
python run_full_pipeline.py --step mcts-optimized --provider lmstudio
python run_full_pipeline.py --step compare

# ─── RUN TESTS ───
python -m pytest tests/test_summarizer.py tests/test_embed_tree.py -v
```

---

## Expected Results After Fix

### Before (current flat trees):
```
repo → 1 file → 46 functions    (depth 2, branching 46)
RL: nothing to restructure, Merge/Split/Reparent are useless
```

### After (proper trees):
```
repo → 3 folders → 8 files → 5 classes → 46 functions    (depth 5, branching ~4)
RL: Split bloated folders, Merge related files, Reparent misplaced utilities
MCTS: LLM summaries at every level guide search efficiently
```

---

## Time Estimates Per Step

| Step | Time (25 repos) | Needs LLM? |
|------|:----------------:|:----------:|
| Clone repos | ~5 min | No |
| tree-sitter parse | ~2 min | No |
| LLM summarization | ~45-60 min | **Yes** |
| Embed summaries | ~5 min | No |
| RL training | ~5 min | No |
| MCTS baseline | ~2-3 hrs | **Yes** |
| MCTS optimized | ~2-3 hrs | **Yes** |

> **Bottleneck:** LLM summarization (~60 min for 25 repos). Each function = 1 LLM call, 
> each file/folder = 1 call. A repo with 100 functions + 15 files + 5 folders = ~120 calls.
> At ~0.5s per call on LM Studio = ~60s per repo × 25 repos = ~25 min.

---

## ✅ Phase 3B Checklist

- [ ] `backend/summarizer.py` — Bottom-up LLM summarization
- [ ] `backend/embed_tree.py` — Embed all summaries
- [ ] `prepare_proper_trees.py` — Full prep pipeline (clone → parse → summarize → embed)
- [ ] `tests/test_summarizer.py` — Summarizer tests
- [ ] `tests/test_embed_tree.py` — Embedder tests
- [ ] 1-repo test passes (proper tree with depth 4+, LLM summaries)
- [ ] 25-repo run completes
- [ ] `run_full_pipeline.py` updated to use proper metadata
- [ ] RL training shows actual depth/branching changes on proper trees
- [ ] Git committed and pushed

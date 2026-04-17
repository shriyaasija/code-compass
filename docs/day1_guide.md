# Day 1 Guide: DevQuery-Bench Construction

**Goal:** Build the benchmark dataset that replaces CodeSearchNet. By end of day you'll have `devquery_bench/devquery_bench.json` containing 300 (query, target_function, repo, alpha) triples across 15 Python repos.

**Time estimate:** 8–10 hours (3h repo setup, 2h query generation, 3h annotation)

---

## Step 0: Environment Setup (15 minutes)

Open a terminal. Everything below assumes you're in the `code-compass` directory.

```bash
# Go to your project
cd ~/code-compass

# Activate your virtual environment
source venv/bin/activate

# Verify Python
python --version
# Should print Python 3.10+ 

# Install new dependencies needed this week
pip install rank_bm25 nltk matplotlib seaborn scipy

# Download NLTK data (for BLEU score computation)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Create directories for this week's work
mkdir -p devquery_bench
mkdir -p devquery_bench/cloned_repos
mkdir -p devquery_bench/trees
mkdir -p experiments
mkdir -p figures

# Quick sanity check — make sure existing code imports work
python -c "from backend.code_parser import CodeParser; print('✅ code_parser OK')"
python -c "from backend.tree_builder import build_directory_tree; print('✅ tree_builder OK')"
python -c "from backend.summarizer import TreeSummarizer; print('✅ summarizer OK')"
python -c "from backend.embed_tree import TreeEmbedder; print('✅ embed_tree OK')"
python -c "from research.mcts.puct_search import PUCTSearch; print('✅ puct_search OK')"
```

If any of those fail, fix the import error before proceeding.

---

## Step 1: Create the Repository List (10 minutes)

Create the file `devquery_bench/repo_list.json`. This is the list of 15 repos you'll be benchmarking.

**Why these repos?** We need a range of sizes. The paper's thesis is that MCTS beats dense retrieval on large repos with naturalistic queries. If we only test on small repos, we can't prove the thesis.

Create this file:

```bash
cat > devquery_bench/repo_list.json << 'JSONEOF'
{
  "repos": [
    {
      "name": "django/django",
      "url": "https://github.com/django/django.git",
      "category": "large",
      "description": "Python web framework"
    },
    {
      "name": "scikit-learn/scikit-learn",
      "url": "https://github.com/scikit-learn/scikit-learn.git",
      "category": "large",
      "description": "Machine learning library"
    },
    {
      "name": "psf/requests",
      "url": "https://github.com/psf/requests.git",
      "category": "large",
      "description": "HTTP library for Python"
    },
    {
      "name": "encode/httpx",
      "url": "https://github.com/encode/httpx.git",
      "category": "large",
      "description": "Async HTTP client"
    },
    {
      "name": "pallets/flask",
      "url": "https://github.com/pallets/flask.git",
      "category": "large",
      "description": "Micro web framework"
    },
    {
      "name": "sqlalchemy/sqlalchemy",
      "url": "https://github.com/sqlalchemy/sqlalchemy.git",
      "category": "medium",
      "description": "SQL toolkit and ORM"
    },
    {
      "name": "pydantic/pydantic",
      "url": "https://github.com/pydantic/pydantic.git",
      "category": "medium",
      "description": "Data validation library"
    },
    {
      "name": "tiangolo/fastapi",
      "url": "https://github.com/tiangolo/fastapi.git",
      "category": "medium",
      "description": "Modern web API framework"
    },
    {
      "name": "aio-libs/aiohttp",
      "url": "https://github.com/aio-libs/aiohttp.git",
      "category": "medium",
      "description": "Async HTTP client/server"
    },
    {
      "name": "pytest-dev/pytest",
      "url": "https://github.com/pytest-dev/pytest.git",
      "category": "medium",
      "description": "Python testing framework"
    },
    {
      "name": "pallets/click",
      "url": "https://github.com/pallets/click.git",
      "category": "small",
      "description": "CLI creation toolkit"
    },
    {
      "name": "docopt/docopt",
      "url": "https://github.com/docopt/docopt.git",
      "category": "small",
      "description": "CLI argument parser"
    },
    {
      "name": "keleshev/schema",
      "url": "https://github.com/keleshev/schema.git",
      "category": "small",
      "description": "Data validation library"
    },
    {
      "name": "pytoolz/toolz",
      "url": "https://github.com/pytoolz/toolz.git",
      "category": "small",
      "description": "Functional utilities"
    },
    {
      "name": "kennethreitz/records",
      "url": "https://github.com/kennethreitz/records.git",
      "category": "small",
      "description": "SQL for humans"
    }
  ]
}
JSONEOF

echo "✅ Created devquery_bench/repo_list.json"
```

---

## Step 2: Clone All 15 Repos (20–40 minutes)

We use `--depth 1` (shallow clone) to save disk space and time.

```bash
cd ~/code-compass

# Create the clone script
python3 << 'PYEOF'
import json, subprocess, os

with open('devquery_bench/repo_list.json') as f:
    repos = json.load(f)['repos']

clone_dir = 'devquery_bench/cloned_repos'
os.makedirs(clone_dir, exist_ok=True)

for i, repo in enumerate(repos, 1):
    name = repo['name']
    url = repo['url']
    safe_name = name.replace('/', '__')
    target = os.path.join(clone_dir, safe_name)
    
    if os.path.exists(target):
        print(f"[{i}/{len(repos)}] ✅ Already cloned: {safe_name}")
        continue
    
    print(f"[{i}/{len(repos)}] 📥 Cloning {name}...")
    try:
        subprocess.run(
            ['git', 'clone', '--depth', '1', url, target],
            check=True, capture_output=True, timeout=300
        )
        print(f"  ✅ Done")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

print("\n✅ All repos cloned!")
PYEOF
```

**Verify clones:**
```bash
ls devquery_bench/cloned_repos/ | wc -l
# Should print 15

# Check a big one worked
ls devquery_bench/cloned_repos/django__django/ | head -5
```

> **Troubleshooting:**
> - If `git clone` fails with timeout: try without `--depth 1` (full clone) — `git clone https://github.com/django/django.git devquery_bench/cloned_repos/django__django`
> - If disk space is low: `df -h .` to check. The 15 repos will use ~2–3GB total.
> - If a specific repo fails repeatedly: remove it from `repo_list.json` and continue with 14.

---

## Step 3: Parse All Repos with tree-sitter (30–60 minutes)

This uses the existing `backend/code_parser.py` and `backend/tree_builder.py` to build hierarchical trees.

Create the script:

```bash
cat > devquery_bench/build_trees.py << 'PYEOF'
"""
Build tree-sitter trees for all DevQuery-Bench repos.
Reuses the existing prepare_proper_trees.py pipeline.
"""
import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.tree_builder import build_directory_tree
from backend.code_parser import CodeParser, enrich_tree_with_code_structure
from backend.embed_tree import TreeEmbedder
from collections import defaultdict


def count_nodes(node, counts=None):
    if counts is None:
        counts = defaultdict(int)
    ntype = node.get('type', node.get('node_type', 'unknown'))
    counts[ntype] += 1
    for child in node.get('nodes', node.get('children', [])):
        count_nodes(child, counts)
    return dict(counts)


def tree_depth(node):
    children = node.get('nodes', node.get('children', []))
    if not children:
        return 0
    return 1 + max(tree_depth(c) for c in children)


def main():
    with open('devquery_bench/repo_list.json') as f:
        repos = json.load(f)['repos']

    clone_dir = 'devquery_bench/cloned_repos'
    trees_dir = 'devquery_bench/trees'
    os.makedirs(trees_dir, exist_ok=True)

    parser = CodeParser()
    embedder = TreeEmbedder()

    results = []

    for i, repo in enumerate(repos, 1):
        name = repo['name']
        safe_name = name.replace('/', '__')
        repo_path = os.path.join(clone_dir, safe_name)
        tree_path = os.path.join(trees_dir, f"{safe_name}.json")

        print(f"\n{'─'*70}")
        print(f"[{i}/{len(repos)}] {name}")

        if not os.path.exists(repo_path):
            print(f"  ⚠️ Not cloned, skipping")
            continue

        if os.path.exists(tree_path):
            print(f"  ✅ Tree already exists, loading stats...")
            with open(tree_path) as f:
                tree_dict = json.load(f)
            counts = count_nodes(tree_dict)
            depth = tree_depth(tree_dict)
            n_funcs = counts.get('function', 0) + counts.get('method', 0)
            results.append({
                'repo_id': safe_name, 'repo_name': name,
                'category': repo['category'],
                'tree_path': tree_path, 'repo_path': repo_path,
                'num_functions': n_funcs, 'tree_depth': depth,
                'total_nodes': sum(counts.values()),
            })
            print(f"  depth={depth}, funcs={n_funcs}, total={sum(counts.values())}")
            continue

        # Step 1: Build directory tree
        print(f"  🌳 Building directory tree...")
        t0 = time.time()
        tree_obj = build_directory_tree(repo_path)

        # Step 2: Parse with tree-sitter
        print(f"  🧩 Parsing code structure...")
        tree_obj = enrich_tree_with_code_structure(tree_obj, parser)
        tree_dict = tree_obj.to_dict()

        counts = count_nodes(tree_dict)
        depth = tree_depth(tree_dict)
        n_funcs = counts.get('function', 0) + counts.get('method', 0)
        print(f"  📊 depth={depth}, funcs={n_funcs}, total={sum(counts.values())}")

        # Step 3: Embed (NO LLM summarization yet — we'll do that after
        # query generation so we know which repos are in train/test split)
        print(f"  🔢 Embedding node titles/names...")
        # For now, use node titles as summaries (fast, no LLM needed)
        def add_title_summaries(node):
            title = node.get('title', node.get('name', ''))
            ntype = node.get('type', node.get('node_type', ''))
            path = node.get('path', '')
            # Create a basic summary from available info
            node['summary'] = f"{ntype}: {title}" + (f" at {path}" if path else "")
            for child in node.get('nodes', node.get('children', [])):
                add_title_summaries(child)
        add_title_summaries(tree_dict)

        tree_dict = embedder.embed_tree(tree_dict)

        # Step 4: Save
        with open(tree_path, 'w') as f:
            json.dump(tree_dict, f)
        size_mb = os.path.getsize(tree_path) / (1024*1024)
        elapsed = time.time() - t0
        print(f"  💾 Saved: {tree_path} ({size_mb:.1f} MB, {elapsed:.0f}s)")

        results.append({
            'repo_id': safe_name, 'repo_name': name,
            'category': repo['category'],
            'tree_path': tree_path, 'repo_path': repo_path,
            'num_functions': n_funcs, 'tree_depth': depth,
            'total_nodes': sum(counts.values()),
        })

    # Save metadata
    with open('devquery_bench/repo_metadata.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for r in sorted(results, key=lambda x: x['num_functions'], reverse=True):
        cat = r.get('category', '?')
        print(f"  [{cat:>6}] {r['repo_name']:>40}: "
              f"funcs={r['num_functions']:>5}, depth={r['tree_depth']}")

    total_funcs = sum(r['num_functions'] for r in results)
    print(f"\n  Total: {len(results)} repos, {total_funcs} functions")


if __name__ == '__main__':
    main()
PYEOF

echo "✅ Created devquery_bench/build_trees.py"
```

Now run it:

```bash
cd ~/code-compass
source venv/bin/activate

python devquery_bench/build_trees.py
```

This will take 30–60 minutes depending on disk speed. The big repos (django, scikit-learn) will take the longest.

**Check the output:**
```bash
cat devquery_bench/repo_metadata.json | python3 -m json.tool | head -30
```

You should see repos with `num_functions` counts like:
- django: 5000–8000 functions
- scikit-learn: 2000–4000
- flask: 200–400 
- docopt: 30–80

> **If a repo has 0 functions:** tree-sitter may have failed to parse it. Check if the repo contains Python files: `find devquery_bench/cloned_repos/REPO_NAME -name "*.py" | wc -l`. If it has .py files but 0 functions, there may be a parser issue — remove that repo from the list and continue.

---

## Step 4: Generate Naturalistic Queries (1–2 hours)

This is where DevQuery-Bench differs from CodeSearchNet. Instead of using docstrings (which are trivially easy for embeddings), we generate queries that sound like real developers.

**You need LM Studio running for this step.**

1. Open LM Studio
2. Load a model (any 7B+ model works)
3. Go to Local Server tab → Start Server
4. Verify: `curl http://localhost:1234/v1/models`

Create the query generation script:

```bash
cat > devquery_bench/generate_queries.py << 'PYEOF'
"""
Generate naturalistic developer queries for DevQuery-Bench.

Uses LLM to generate queries that sound like what a developer would actually type,
NOT docstrings or function descriptions.
"""
import json
import os
import sys
import time
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.lmstudio_client import LMStudioLLM


QUERIES_PER_REPO = 20

QUERY_GEN_PROMPT = """You are generating naturalistic developer queries for a code retrieval benchmark.

I will give you:
1. A Python repository name and description
2. The top-level directory listing
3. A sample of function names from the repo (for context only — do NOT use these in queries)

Your task: Generate exactly {n} queries that a real developer might ask when navigating this codebase for the first time.

CRITICAL RULES:
- Do NOT use any function names, method names, class names, or variable names from the codebase
- Do NOT repeat vocabulary from docstrings
- Each query should express a developer's *intent* in natural language
- Queries should vary in specificity:
  - Some broad: "how does caching work?"
  - Some narrow: "what validates the API key before a request is processed?"
- Each query should ideally have ONE correct answer function in the codebase
- Make queries sound like what someone would type into a search bar

Repository: {repo_name}
Description: {description}
Top-level directories: {dir_listing}
Sample functions (DO NOT use these names in queries): {sample_functions}

Respond with ONLY a JSON list of {n} query strings. Example format:
["query 1", "query 2", ...]
"""


def get_dir_listing(repo_path):
    """Get top-level directory listing."""
    items = []
    try:
        for entry in sorted(os.listdir(repo_path)):
            if entry.startswith('.'):
                continue
            full = os.path.join(repo_path, entry)
            if os.path.isdir(full):
                items.append(f"{entry}/")
            else:
                items.append(entry)
    except Exception:
        pass
    return items[:30]  # limit


def get_sample_functions(tree, max_samples=20):
    """Get a random sample of function names from the tree."""
    funcs = []
    def walk(node):
        ntype = node.get('type', node.get('node_type', ''))
        if ntype in ('function', 'method'):
            name = node.get('title', node.get('name', ''))
            if name and not name.startswith('_'):
                funcs.append(name)
        for child in node.get('nodes', node.get('children', [])):
            walk(child)
    walk(tree)
    import random
    random.seed(42)
    return random.sample(funcs, min(max_samples, len(funcs)))


def parse_json_list(text):
    """Parse a JSON list from LLM output, handling common issues."""
    # Try direct parse
    try:
        result = json.loads(text.strip())
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array in the text
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Last resort: split by newlines and clean up
    lines = [l.strip().strip('",').strip('"').strip("'") 
             for l in text.strip().split('\n') 
             if l.strip() and not l.strip().startswith(('[', ']', '#', '//'))]
    return [l for l in lines if len(l) > 10]


def main():
    llm = LMStudioLLM()

    with open('devquery_bench/repo_metadata.json') as f:
        repos = json.load(f)

    all_queries = {}

    for i, repo in enumerate(repos, 1):
        repo_name = repo['repo_name']
        repo_id = repo['repo_id']
        repo_path = repo['repo_path']
        tree_path = repo['tree_path']

        print(f"\n{'─'*70}")
        print(f"[{i}/{len(repos)}] Generating queries for {repo_name}")

        # Check if already done
        output_path = f"devquery_bench/queries_{repo_id}.json"
        if os.path.exists(output_path):
            print(f"  ✅ Already generated, loading...")
            with open(output_path) as f:
                all_queries[repo_id] = json.load(f)
            continue

        # Load tree for function sampling
        with open(tree_path) as f:
            tree = json.load(f)

        dir_listing = get_dir_listing(repo_path)
        sample_funcs = get_sample_functions(tree)

        # Find description from repo_list
        with open('devquery_bench/repo_list.json') as f:
            repo_info = json.load(f)
        desc = next((r['description'] for r in repo_info['repos'] 
                     if r['name'] == repo_name), repo_name)

        prompt = QUERY_GEN_PROMPT.format(
            n=QUERIES_PER_REPO,
            repo_name=repo_name,
            description=desc,
            dir_listing=', '.join(dir_listing),
            sample_functions=', '.join(sample_funcs),
        )

        print(f"  🤖 Calling LLM to generate {QUERIES_PER_REPO} queries...")
        try:
            messages = [
                {"role": "system", "content": "You generate benchmark queries for code retrieval research. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ]
            response = llm.chat(messages, temperature=0.7, max_tokens=2000)
            queries = parse_json_list(response)
        except Exception as e:
            print(f"  ❌ LLM call failed: {e}")
            queries = []

        if len(queries) < 10:
            print(f"  ⚠️ Only got {len(queries)} queries, retrying...")
            try:
                response = llm.chat(messages, temperature=0.8, max_tokens=2000)
                queries = parse_json_list(response)
            except Exception:
                pass

        print(f"  ✅ Generated {len(queries)} queries")
        for j, q in enumerate(queries[:5]):
            print(f"    {j+1}. {q}")
        if len(queries) > 5:
            print(f"    ... and {len(queries) - 5} more")

        # Save per-repo queries
        with open(output_path, 'w') as f:
            json.dump(queries, f, indent=2)

        all_queries[repo_id] = queries
        time.sleep(1)  # Be nice to LM Studio

    # Save combined
    with open('devquery_bench/all_queries.json', 'w') as f:
        json.dump(all_queries, f, indent=2)

    total = sum(len(q) for q in all_queries.values())
    print(f"\n{'='*70}")
    print(f"✅ Generated {total} queries across {len(all_queries)} repos")
    print(f"   Saved to: devquery_bench/all_queries.json")


if __name__ == '__main__':
    main()
PYEOF

echo "✅ Created devquery_bench/generate_queries.py"
```

Run it:

```bash
cd ~/code-compass
source venv/bin/activate

# Make sure LM Studio is running!
python devquery_bench/generate_queries.py
```

**Check the output:**
```bash
# Count total queries
python3 -c "
import json
with open('devquery_bench/all_queries.json') as f:
    q = json.load(f)
total = sum(len(v) for v in q.values())
print(f'Total queries: {total}')
for repo, queries in q.items():
    print(f'  {repo}: {len(queries)} queries')
"
```

You should have ~300 queries (20 per repo × 15 repos). If some repos got fewer than 20, that's fine — aim for 250+ total.

---

## Step 5: Annotate Ground Truth (2–3 hours — MANUAL)

This is the most time-intensive step. For each query, you need to find the correct target function in the repo.

**Create the annotation helper script:**

```bash
cat > devquery_bench/annotate.py << 'PYEOF'
"""
Interactive ground truth annotation tool.

For each query, shows candidate functions from the repo.
You type the correct function name (or skip with 's').

Usage:
  python devquery_bench/annotate.py --repo django__django
  python devquery_bench/annotate.py --repo django__django --start 5  # resume from query 5
"""
import json
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def collect_functions(node, funcs=None):
    """Collect all functions with their paths."""
    if funcs is None:
        funcs = []
    ntype = node.get('type', node.get('node_type', ''))
    if ntype in ('function', 'method'):
        funcs.append({
            'name': node.get('title', node.get('name', '')),
            'path': node.get('path', ''),
            'summary': node.get('summary', ''),
            'start_line': node.get('start_line'),
        })
    for child in node.get('nodes', node.get('children', [])):
        collect_functions(child, funcs)
    return funcs


def search_functions(funcs, query_words):
    """Simple keyword search over functions."""
    query_words_lower = set(w.lower() for w in query_words if len(w) > 2)
    scored = []
    for func in funcs:
        text = f"{func['name']} {func['path']} {func['summary']}".lower()
        score = sum(1 for w in query_words_lower if w in text)
        if score > 0:
            scored.append((score, func))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in scored[:20]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True, help='Repo ID like django__django')
    p.add_argument('--start', type=int, default=0, help='Start from query index')
    args = p.parse_args()

    repo_id = args.repo
    queries_path = f"devquery_bench/queries_{repo_id}.json"
    tree_path = f"devquery_bench/trees/{repo_id}.json"
    output_path = f"devquery_bench/annotations_{repo_id}.json"

    if not os.path.exists(queries_path):
        print(f"❌ No queries found at {queries_path}")
        sys.exit(1)
    if not os.path.exists(tree_path):
        print(f"❌ No tree found at {tree_path}")
        sys.exit(1)

    with open(queries_path) as f:
        queries = json.load(f)
    with open(tree_path) as f:
        tree = json.load(f)

    funcs = collect_functions(tree)
    print(f"Loaded {len(funcs)} functions from {repo_id}")

    # Load existing annotations if resuming
    annotations = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            annotations = json.load(f)
        print(f"Loaded {len(annotations)} existing annotations")

    for i, query in enumerate(queries):
        if i < args.start:
            continue
        if i < len(annotations):
            print(f"\n[{i+1}/{len(queries)}] Already annotated, skipping")
            continue

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(queries)}] QUERY: {query}")
        print(f"{'='*70}")

        # Show candidate functions
        words = query.split()
        candidates = search_functions(funcs, words)

        if candidates:
            print(f"\n  Top candidates:")
            for j, func in enumerate(candidates[:10]):
                print(f"    {j+1:>2}. {func['name']:>40} | {func['path']}")
        else:
            print(f"\n  No obvious candidates found by keyword search.")
            print(f"  You can type a function name directly, or 'ls <keyword>' to search.")

        while True:
            answer = input(f"\n  Enter function name (or number 1-10, 's' to skip, 'ls <word>' to search): ").strip()

            if answer.lower() == 's':
                annotations.append({
                    'query': query,
                    'ground_truth': None,
                    'skipped': True,
                })
                break
            elif answer.lower().startswith('ls '):
                keyword = answer[3:].strip()
                results = [f for f in funcs if keyword.lower() in 
                          f"{f['name']} {f['path']} {f['summary']}".lower()]
                for j, func in enumerate(results[:15]):
                    print(f"    {func['name']:>40} | {func['path']}")
                continue
            elif answer.isdigit() and 1 <= int(answer) <= len(candidates):
                func_name = candidates[int(answer)-1]['name']
                print(f"  ✅ Selected: {func_name}")
                annotations.append({
                    'query': query,
                    'ground_truth': func_name,
                    'skipped': False,
                })
                break
            elif answer:
                # Check if it exists
                matches = [f for f in funcs if f['name'] == answer]
                if matches:
                    print(f"  ✅ Found: {answer}")
                    annotations.append({
                        'query': query,
                        'ground_truth': answer,
                        'skipped': False,
                    })
                    break
                else:
                    # Fuzzy check
                    close = [f for f in funcs if answer.lower() in f['name'].lower()]
                    if close:
                        print(f"  ⚠️ Exact match not found. Did you mean one of:")
                        for f in close[:5]:
                            print(f"    → {f['name']}")
                    else:
                        print(f"  ❌ '{answer}' not found in the repo. Try again.")

        # Save after each annotation (in case you need to stop)
        with open(output_path, 'w') as f:
            json.dump(annotations, f, indent=2)

    annotated = sum(1 for a in annotations if not a.get('skipped'))
    print(f"\n✅ Done! {annotated}/{len(queries)} queries annotated for {repo_id}")
    print(f"   Saved to: {output_path}")


if __name__ == '__main__':
    main()
PYEOF

echo "✅ Created devquery_bench/annotate.py"
```

**How to annotate:**

Run this for each repo. Start with the smaller ones to get the hang of it:

```bash
cd ~/code-compass
source venv/bin/activate

# Start with small repos (faster to annotate)
python devquery_bench/annotate.py --repo pallets__click
python devquery_bench/annotate.py --repo docopt__docopt
python devquery_bench/annotate.py --repo keleshev__schema
python devquery_bench/annotate.py --repo pytoolz__toolz
python devquery_bench/annotate.py --repo kennethreitz__records

# Then medium
python devquery_bench/annotate.py --repo pydantic__pydantic
python devquery_bench/annotate.py --repo tiangolo__fastapi
python devquery_bench/annotate.py --repo pytest-dev__pytest
python devquery_bench/annotate.py --repo aio-libs__aiohttp
python devquery_bench/annotate.py --repo sqlalchemy__sqlalchemy

# Then large (these are harder — more functions to sift through)
python devquery_bench/annotate.py --repo django__django
python devquery_bench/annotate.py --repo scikit-learn__scikit-learn
python devquery_bench/annotate.py --repo psf__requests
python devquery_bench/annotate.py --repo encode__httpx
python devquery_bench/annotate.py --repo pallets__flask

# If you need to resume from where you left off:
# python devquery_bench/annotate.py --repo django__django --start 12
```

> **Tips for faster annotation:**
> - Use `ls <keyword>` to search within the annotation tool
> - If a query has no good answer, skip it with `s` — it's fine, not every query needs an answer
> - For large repos, use grep in another terminal: `grep -r "def rate_limit" devquery_bench/cloned_repos/django__django/`
> - Aim for at least 15 annotated queries per repo (some skips are OK)
> - Save frequently — the tool auto-saves after each annotation

---

## Step 6: Compute Naturalism Scores (15 minutes)

After annotation, compute how "naturalistic" each query is using BLEU-1 score.

```bash
cat > devquery_bench/compute_naturalism.py << 'PYEOF'
"""
Compute naturalism score alpha for each query-function pair.
alpha = 1 - BLEU_1(query, docstring(target_function))
High alpha = naturalistic query (good for our benchmark)
Low alpha = docstring-like query (too easy for embeddings)
"""
import json
import os
import sys
import glob
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tokenize(text):
    """Simple whitespace + punctuation tokenizer."""
    import re
    return re.findall(r'\w+', text.lower())


def bleu1(reference, hypothesis):
    """Compute smoothed BLEU-1 (unigram precision)."""
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    if not hyp_tokens or not ref_tokens:
        return 0.0

    ref_counts = Counter(ref_tokens)
    hyp_counts = Counter(hyp_tokens)

    clipped = sum(min(hyp_counts[w], ref_counts[w]) for w in hyp_counts)
    # Smoothed: add 1 to denominator to avoid division by zero
    return clipped / (len(hyp_tokens) + 1)


def get_function_docstring(tree, func_name):
    """Find a function's summary/docstring in the tree."""
    def walk(node):
        ntype = node.get('type', node.get('node_type', ''))
        title = node.get('title', node.get('name', ''))
        if ntype in ('function', 'method') and title == func_name:
            return node.get('summary', title)
        for child in node.get('nodes', node.get('children', [])):
            result = walk(child)
            if result:
                return result
        return None
    return walk(tree) or func_name


def main():
    annotation_files = sorted(glob.glob('devquery_bench/annotations_*.json'))
    if not annotation_files:
        print("❌ No annotation files found. Run annotate.py first.")
        return

    all_entries = []
    alphas = []

    for ann_file in annotation_files:
        repo_id = ann_file.replace('devquery_bench/annotations_', '').replace('.json', '')
        tree_path = f"devquery_bench/trees/{repo_id}.json"

        with open(ann_file) as f:
            annotations = json.load(f)

        if not os.path.exists(tree_path):
            print(f"  ⚠️ No tree for {repo_id}, skipping")
            continue

        with open(tree_path) as f:
            tree = json.load(f)

        for ann in annotations:
            if ann.get('skipped') or not ann.get('ground_truth'):
                continue

            query = ann['query']
            gt = ann['ground_truth']
            docstring = get_function_docstring(tree, gt)

            b1 = bleu1(docstring, query)
            alpha = round(1.0 - b1, 4)
            alphas.append(alpha)

            all_entries.append({
                'query': query,
                'ground_truth': gt,
                'repo_id': repo_id,
                'docstring': docstring,
                'bleu1': round(b1, 4),
                'alpha': alpha,
            })

    # Save final benchmark
    with open('devquery_bench/devquery_bench.json', 'w') as f:
        json.dump(all_entries, f, indent=2)

    # Print stats
    import numpy as np
    alphas = np.array(alphas)
    print(f"\n{'='*70}")
    print(f"DEVQUERY-BENCH STATISTICS")
    print(f"{'='*70}")
    print(f"  Total annotated queries: {len(all_entries)}")
    print(f"  Mean alpha (naturalism): {alphas.mean():.3f}")
    print(f"  Std alpha:               {alphas.std():.3f}")
    print(f"  Min alpha:               {alphas.min():.3f}")
    print(f"  Max alpha:               {alphas.max():.3f}")
    print(f"\n  For reference:")
    print(f"    CodeSearchNet alpha ≈ 0.05 (docstring queries)")
    print(f"    DevQuery-Bench alpha should be > 0.5")
    print(f"\n  Saved to: devquery_bench/devquery_bench.json")


if __name__ == '__main__':
    main()
PYEOF

echo "✅ Created devquery_bench/compute_naturalism.py"
```

Run it:

```bash
cd ~/code-compass
source venv/bin/activate

python devquery_bench/compute_naturalism.py
```

---

## Step 7: LLM Summarization for Trees (2–4 hours, can run overnight)

Now that trees exist, we need proper LLM summaries (not just titles). This is slow because it calls LM Studio for every node.

> **Important:** Make sure LM Studio is running and has a model loaded.

```bash
cat > devquery_bench/summarize_trees.py << 'PYEOF'
"""
Add LLM-generated summaries to all DevQuery-Bench trees.
This replaces the title-only summaries from build_trees.py with
proper bottom-up LLM summaries.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.summarizer import TreeSummarizer
from backend.embed_tree import TreeEmbedder
from backend.lmstudio_client import LMStudioLLM


def main():
    llm = LMStudioLLM()
    summarizer = TreeSummarizer(llm, verbose=True)
    embedder = TreeEmbedder()

    with open('devquery_bench/repo_metadata.json') as f:
        repos = json.load(f)

    for i, repo in enumerate(repos, 1):
        repo_id = repo['repo_id']
        tree_path = repo['tree_path']
        repo_path = repo['repo_path']

        # Check if already summarized (look for a marker)
        summarized_marker = f"devquery_bench/trees/.{repo_id}_summarized"
        if os.path.exists(summarized_marker):
            print(f"[{i}/{len(repos)}] ✅ {repo_id} already summarized")
            continue

        print(f"\n[{i}/{len(repos)}] Summarizing {repo['repo_name']}...")
        print(f"  Functions: {repo['num_functions']}, Depth: {repo['tree_depth']}")

        with open(tree_path) as f:
            tree = json.load(f)

        # LLM summarize (this is slow — ~1-2s per node)
        t0 = time.time()
        tree = summarizer.summarize_tree(tree, repo_path)
        summary_time = time.time() - t0

        # Re-embed with new summaries
        tree = embedder.embed_tree(tree)

        # Save
        with open(tree_path, 'w') as f:
            json.dump(tree, f)

        # Mark as done
        with open(summarized_marker, 'w') as f:
            f.write(f"summarized at {time.strftime('%Y-%m-%d %H:%M')}")

        print(f"  ✅ Done in {summary_time:.0f}s")

    print(f"\n{'='*70}")
    print("✅ All trees summarized and re-embedded!")


if __name__ == '__main__':
    main()
PYEOF

echo "✅ Created devquery_bench/summarize_trees.py"
```

Run it (this is slow — if you have many repos, start it and let it run):

```bash
cd ~/code-compass
source venv/bin/activate

python devquery_bench/summarize_trees.py
```

> **Tip:** For very large repos (django with 5000+ functions), this can take 1-2 hours per repo. If you're short on time, start with the smaller repos and run the large ones overnight.

---

## Step 8: Create the Train/Test Split (5 minutes)

The 5 largest repos are held out for testing. The other 10 are used for prior training.

```bash
cat > devquery_bench/create_split.py << 'PYEOF'
"""Create train/test split for DevQuery-Bench."""
import json

# Test repos: the 5 largest (most interesting for the paper)
TEST_REPOS = [
    "django__django",
    "scikit-learn__scikit-learn", 
    "psf__requests",
    "encode__httpx",
    "pallets__flask",
]

with open('devquery_bench/repo_metadata.json') as f:
    repos = json.load(f)

all_ids = [r['repo_id'] for r in repos]
train_ids = [r for r in all_ids if r not in TEST_REPOS]

split = {
    'train': train_ids,
    'test': TEST_REPOS,
}

with open('devquery_bench/train_test_split.json', 'w') as f:
    json.dump(split, f, indent=2)

print(f"Train repos ({len(train_ids)}): {train_ids}")
print(f"Test repos  ({len(TEST_REPOS)}): {TEST_REPOS}")
print(f"Saved to: devquery_bench/train_test_split.json")
PYEOF

python devquery_bench/create_split.py
```

---

## Step 9: Verify Everything (10 minutes)

Run this final check to make sure Day 1 output is complete:

```bash
python3 << 'PYEOF'
import json, os, glob

print("=" * 60)
print("DAY 1 VERIFICATION CHECKLIST")
print("=" * 60)

# 1. Repos cloned
clones = os.listdir('devquery_bench/cloned_repos') if os.path.exists('devquery_bench/cloned_repos') else []
print(f"\n✅ Repos cloned: {len(clones)}/15")

# 2. Trees built
trees = glob.glob('devquery_bench/trees/*.json')
print(f"✅ Trees built: {len(trees)}/15")

# 3. Queries generated
if os.path.exists('devquery_bench/all_queries.json'):
    with open('devquery_bench/all_queries.json') as f:
        queries = json.load(f)
    total_q = sum(len(v) for v in queries.values())
    print(f"✅ Queries generated: {total_q} (target: ~300)")
else:
    print(f"❌ Queries not generated yet")

# 4. Annotations
ann_files = glob.glob('devquery_bench/annotations_*.json')
total_ann = 0
for af in ann_files:
    with open(af) as f:
        anns = json.load(f)
    annotated = sum(1 for a in anns if not a.get('skipped'))
    total_ann += annotated
print(f"✅ Annotations: {total_ann} across {len(ann_files)} repos")

# 5. DevQuery-Bench
if os.path.exists('devquery_bench/devquery_bench.json'):
    with open('devquery_bench/devquery_bench.json') as f:
        bench = json.load(f)
    alphas = [e['alpha'] for e in bench]
    import numpy as np
    print(f"✅ DevQuery-Bench: {len(bench)} entries, mean α={np.mean(alphas):.3f}")
else:
    print(f"❌ devquery_bench.json not created yet")

# 6. Train/test split
if os.path.exists('devquery_bench/train_test_split.json'):
    with open('devquery_bench/train_test_split.json') as f:
        split = json.load(f)
    print(f"✅ Split: {len(split['train'])} train, {len(split['test'])} test")
else:
    print(f"❌ Split not created yet")

print(f"\n{'='*60}")
PYEOF
```

---

## Day 1 Outputs Checklist

By end of Day 1, you should have:

- [ ] `devquery_bench/repo_list.json` — 15 repos defined
- [ ] `devquery_bench/cloned_repos/` — 15 repos cloned
- [ ] `devquery_bench/trees/` — 15 parsed tree JSON files
- [ ] `devquery_bench/repo_metadata.json` — repo stats (function counts, depths)
- [ ] `devquery_bench/all_queries.json` — ~300 naturalistic queries
- [ ] `devquery_bench/annotations_*.json` — ground truth labels (15 files)
- [ ] `devquery_bench/devquery_bench.json` — final benchmark (300 triples with alpha)
- [ ] `devquery_bench/train_test_split.json` — 10 train / 5 test repos

**Git checkpoint:**
```bash
cd ~/code-compass
git add devquery_bench/
git commit -m "Day 1: DevQuery-Bench construction - 15 repos, 300 queries, ground truth annotations"
```

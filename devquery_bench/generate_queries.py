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
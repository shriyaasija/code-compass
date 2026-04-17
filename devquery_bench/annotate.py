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
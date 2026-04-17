"""
Interactive ground truth annotation tool.

For each query, shows candidate functions/classes from the repo.
You can select MULTIPLE ground truth nodes per query (since a query
can be answered by several functions/classes).

Node types collected:
  - function, method   (always terminal in MCTS)
  - class              (terminal in MCTS when it has start_line)

Usage:
  python devquery_bench/annotate.py --repo django__django
  python devquery_bench/annotate.py --repo django__django --start 5  # resume from query 5
"""
import json
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def collect_terminal_nodes(node, nodes=None):
    """Collect all terminal nodes (functions, methods, classes) with their paths.
    
    These are the nodes that MCTS can actually return as results:
      - function/method with start_line  → always terminal
      - class with start_line            → terminal (MCTS won't expand into methods)
    """
    if nodes is None:
        nodes = []
    ntype = node.get('type', node.get('node_type', ''))
    has_start = 'start_line' in node

    if ntype in ('function', 'class') and has_start:
        nodes.append({
            'name': node.get('title', node.get('name', '')),
            'type': ntype,
            'path': node.get('path', ''),
            'summary': node.get('summary', ''),
            'start_line': node.get('start_line'),
        })

    for child in node.get('nodes', node.get('children', [])):
        collect_terminal_nodes(child, nodes)
    return nodes


def search_nodes(nodes, query_words):
    """Simple keyword search over nodes."""
    query_words_lower = set(w.lower() for w in query_words if len(w) > 2)
    scored = []
    for n in nodes:
        text = f"{n['name']} {n['path']} {n['summary']}".lower()
        score = sum(1 for w in query_words_lower if w in text)
        if score > 0:
            scored.append((score, n))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [n for _, n in scored[:20]]


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

    nodes = collect_terminal_nodes(tree)
    func_count = sum(1 for n in nodes if n['type'] in ('function', 'method'))
    class_count = sum(1 for n in nodes if n['type'] == 'class')
    print(f"Loaded {len(nodes)} terminal nodes from {repo_id} "
          f"({func_count} functions/methods, {class_count} classes)")

    # Load existing annotations if resuming
    annotations = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            annotations = json.load(f)
        print(f"Loaded {len(annotations)} existing annotations")

    print(f"\n  Commands:")
    print(f"    <number>       — select a candidate by number (toggles on/off)")
    print(f"    <name>         — type a node name directly")
    print(f"    ls <keyword>   — search all nodes")
    print(f"    done           — finish selecting for this query")
    print(f"    s              — skip this query")
    print(f"    show           — show current selections")

    for i, query in enumerate(queries):
        if i < args.start:
            continue
        if i < len(annotations):
            print(f"\n[{i+1}/{len(queries)}] Already annotated, skipping")
            continue

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(queries)}] QUERY: {query}")
        print(f"{'='*70}")

        # Show candidate nodes
        words = query.split()
        candidates = search_nodes(nodes, words)

        if candidates:
            print(f"\n  Top candidates:")
            for j, n in enumerate(candidates[:15]):
                tag = f"[{n['type'][:3]}]"
                print(f"    {j+1:>2}. {tag} {n['name']:<40} | {n['path']}")
        else:
            print(f"\n  No obvious candidates found by keyword search.")
            print(f"  Type a name directly, or 'ls <keyword>' to search.")

        # Collect multiple ground truths
        selected = []

        while True:
            sel_display = ', '.join(selected) if selected else '(none)'
            answer = input(f"\n  [{sel_display}] > ").strip()

            if answer.lower() == 's':
                annotations.append({
                    'query': query,
                    'ground_truth': [],
                    'skipped': True,
                })
                break

            elif answer.lower() == 'done':
                if not selected:
                    confirm = input("  ⚠️  No nodes selected. Skip? (y/n): ").strip()
                    if confirm.lower() == 'y':
                        annotations.append({
                            'query': query,
                            'ground_truth': [],
                            'skipped': True,
                        })
                        break
                    continue
                print(f"  ✅ Ground truth: {selected}")
                annotations.append({
                    'query': query,
                    'ground_truth': selected,
                    'skipped': False,
                })
                break

            elif answer.lower() == 'show':
                if selected:
                    print(f"  Current selections: {selected}")
                else:
                    print(f"  Nothing selected yet.")
                continue

            elif answer.lower().startswith('ls '):
                keyword = answer[3:].strip()
                results = [n for n in nodes if keyword.lower() in
                          f"{n['name']} {n['path']} {n['summary']}".lower()]
                for j, n in enumerate(results[:20]):
                    tag = f"[{n['type'][:3]}]"
                    print(f"    {tag} {n['name']:<40} | {n['path']}")
                if not results:
                    print(f"    No results for '{keyword}'")
                continue

            elif answer.isdigit() and 1 <= int(answer) <= len(candidates):
                name = candidates[int(answer)-1]['name']
                if name in selected:
                    selected.remove(name)
                    print(f"  ➖ Removed: {name}")
                else:
                    selected.append(name)
                    print(f"  ➕ Added: {name}")
                continue

            elif answer:
                # Direct name input — check if it exists
                matches = [n for n in nodes if n['name'] == answer]
                if matches:
                    if answer in selected:
                        selected.remove(answer)
                        print(f"  ➖ Removed: {answer}")
                    else:
                        selected.append(answer)
                        ntype = matches[0]['type']
                        print(f"  ➕ Added [{ntype}]: {answer}")
                else:
                    close = [n for n in nodes if answer.lower() in n['name'].lower()]
                    if close:
                        print(f"  ⚠️ Exact match not found. Did you mean:")
                        for n in close[:5]:
                            tag = f"[{n['type'][:3]}]"
                            print(f"    → {tag} {n['name']}")
                    else:
                        print(f"  ❌ '{answer}' not found. Try 'ls <keyword>' to search.")

        # Save after each annotation
        with open(output_path, 'w') as f:
            json.dump(annotations, f, indent=2)

    annotated = sum(1 for a in annotations if not a.get('skipped'))
    print(f"\n✅ Done! {annotated}/{len(queries)} queries annotated for {repo_id}")
    print(f"   Saved to: {output_path}")


if __name__ == '__main__':
    main()
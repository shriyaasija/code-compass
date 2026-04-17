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
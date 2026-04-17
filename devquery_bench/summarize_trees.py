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

    with open('devquery_bench/repo_metadata_2.json') as f:
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
import argparse
import json
import os
import sys
from pathlib import Path

from research.rl_index.tree_mutations import count_leaves, tree_depth, avg_branching_factor


def compare_trees(original_path: str, optimized_path: str):
    """Print structural comparison between original and optimized trees."""
    with open(original_path) as f:
        original = json.load(f)
    with open(optimized_path) as f:
        optimized = json.load(f)

    print("\n" + "=" * 70)
    print("📊 TREE STRUCTURE COMPARISON")
    print("=" * 70)

    metrics = [
        ("Leaf count", count_leaves(original), count_leaves(optimized)),
        ("Tree depth", tree_depth(original), tree_depth(optimized)),
    ]

    orig_bf, orig_var = avg_branching_factor(original)
    opt_bf, opt_var = avg_branching_factor(optimized)
    metrics.append(("Avg branching", f"{orig_bf:.2f}", f"{opt_bf:.2f}"))
    metrics.append(("Branching var", f"{orig_var:.2f}", f"{opt_var:.2f}"))

    print(f"\n  {'Metric':<20} {'Original':>12} {'Optimized':>12} {'Change':>12}")
    print(f"  {'─' * 56}")
    for name, orig, opt in metrics:
        if isinstance(orig, (int, float)) and isinstance(opt, (int, float)):
            delta = opt - orig
            sign = "+" if delta > 0 else ""
            print(f"  {name:<20} {orig:>12} {opt:>12} {sign}{delta:>11}")
        else:
            print(f"  {name:<20} {str(orig):>12} {str(opt):>12}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--optimized", required=True)
    args = parser.parse_args()
    compare_trees(args.original, args.optimized)


if __name__ == "__main__":
    main()
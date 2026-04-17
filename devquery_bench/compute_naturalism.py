"""
Compute naturalism score alpha for each query-function pair.
alpha = 1 - BLEU_1(query, docstring(target_function))
High alpha = naturalistic query (good for our benchmark)
Low alpha = docstring-like query (too easy for embeddings)

This version handles ground_truth as a list of terminal nodes.
"""
import json
import os
import sys
import glob
from collections import Counter
import re
import numpy as np

# Add parent directory to path if needed for relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def tokenize(text):
    """Simple whitespace + punctuation tokenizer."""
    if not isinstance(text, str):
        text = str(text)
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

def get_function_docstring(tree, target_name):
    """Find a specific node's summary/docstring in the MCTS tree."""
    def walk(node):
        ntype = node.get('type', node.get('node_type', ''))
        title = node.get('title', node.get('name', ''))
        
        # In your MCTS, classes, functions, and files are terminal
        if ntype in ('function', 'method', 'class', 'file_py', 'file_static') and title == target_name:
            # Prefer summary if available, otherwise fallback to the node name
            return node.get('summary', node.get('docstring', title))
            
        for child in node.get('nodes', node.get('children', [])):
            result = walk(child)
            if result:
                return result
        return None
    
    result = walk(tree)
    # Return found docstring or the name itself as a string fallback
    return str(result) if result else str(target_name)

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

        if not os.path.exists(tree_path):
            print(f"  ⚠️ No tree for {repo_id}, skipping")
            continue

        with open(ann_file) as f:
            try:
                annotations = json.load(f)
            except json.JSONDecodeError:
                print(f"  ❌ Error parsing {ann_file}")
                continue

        with open(tree_path) as f:
            try:
                tree = json.load(f)
            except json.JSONDecodeError:
                print(f"  ❌ Error parsing {tree_path}")
                continue

        print(f"  Processing {repo_id}...")

        for ann in annotations:
            # Skip entries marked as skipped or with no ground truth
            if ann.get('skipped') or not ann.get('ground_truth'):
                continue

            query = ann['query']
            gt_list = ann['ground_truth']
            
            # Ensure gt_list is treated as a list even if it's a single string
            if isinstance(gt_list, str):
                gt_list = [gt_list]

            # 1. Retrieve docstrings for every terminal node in ground truth
            docstrings = []
            for gt_name in gt_list:
                ds = get_function_docstring(tree, gt_name)
                docstrings.append(ds)

            # 2. Compute BLEU-1 for all pairs and find the max overlap (worst case)
            bleu_scores = [bleu1(ds, query) for ds in docstrings]
            b1 = max(bleu_scores) if bleu_scores else 0.0
            
            # 3. Final alpha calculation
            alpha = round(1.0 - b1, 4)
            alphas.append(alpha)

            # Store the entry with the most relevant docstring for debugging
            best_ds_idx = bleu_scores.index(b1) if bleu_scores else 0
            
            all_entries.append({
                'query': query,
                'ground_truth': gt_list,
                'repo_id': repo_id,
                'target_docstring': docstrings[best_ds_idx] if docstrings else "",
                'bleu1': round(b1, 4),
                'alpha': alpha,
            })

    # Save final benchmark dataset
    output_path = 'devquery_bench/devquery_bench.json'
    with open(output_path, 'w') as f:
        json.dump(all_entries, f, indent=2)

    # Calculate and display performance metrics
    if alphas:
        alphas_np = np.array(alphas)
        print(f"\n{'='*70}")
        print(f"DEVQUERY-BENCH NATURALISM METRICS")
        print(f"{'='*70}")
        print(f"  Total annotated queries: {len(all_entries)}")
        print(f"  Mean alpha (naturalism): {alphas_np.mean():.3f}")
        print(f"  Std alpha:               {alphas_np.std():.3f}")
        print(f"  Min alpha:               {alphas_np.min():.3f}")
        print(f"  Max alpha:               {alphas_np.max():.3f}")
        print(f"\n  Threshold Guide:")
        print(f"    < 0.3: High leakage (too similar to source code)")
        print(f"    > 0.6: Strong naturalistic query")
        print(f"\n  Final JSON saved to: {output_path}")
    else:
        print("\n  ⚠️ No valid entries processed. Check ground_truth fields.")

if __name__ == '__main__':
    main()
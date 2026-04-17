import json
import time
import numpy as np
import sys
import os

# Add root directory to path to allow importing backend
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.code_index2 import MCTSTreeSearch
from backend.lmstudio_client import LMStudioLLM

def calculate_metrics(ranked_results, ground_truths, k_values=[1, 5, 10]):
    """Calculate R@K and MRR for a single query."""
    result_names = [res.get('name', '') for res in ranked_results]
    
    # Calculate Recall at K
    recalls = {}
    for k in k_values:
        top_k = result_names[:k]
        # 1 if ANY ground truth is in the top K, else 0
        hit = any(gt in top_k for gt in ground_truths)
        recalls[f'R@{k}'] = 1.0 if hit else 0.0
        
    # Calculate MRR
    mrr = 0.0
    for rank, name in enumerate(result_names, 1):
        if name in ground_truths:
            mrr = 1.0 / rank
            break
            
    return recalls, mrr

def main():
    repo_id = "django__django"
    annotations_path = "devquery_bench/annotations_django__django.json"
    
    # Assuming the tree was built in devquery_bench/trees/
    tree_path = f"devquery_bench/trees/{repo_id}.json"
    
    if not os.path.exists(annotations_path):
        print(f"❌ Annotations not found at {annotations_path}")
        return
        
    if not os.path.exists(tree_path):
        # Fallback to the proper_trees directory if available
        tree_path = f"benchmark_results/proper_trees/{repo_id}.json"
        if not os.path.exists(tree_path):
            print(f"❌ Tree file not found for {repo_id}")
            return

    # Load annotations
    with open(annotations_path, 'r') as f:
        data = json.load(f)
        
    # Filter out skipped queries
    valid_queries = [q for q in data if not q.get('skipped', False) and q.get('ground_truth')]
    
    if not valid_queries:
        print("No valid queries to evaluate.")
        return

    print(f"🚀 Initializing LM Studio for MCTS...")
    try:
        llm = LMStudioLLM()
    except Exception as e:
        print(f"❌ Failed to connect to LM Studio (Ensure server is running!): {e}")
        llm = None
        # Could fallback to OllamaLLM if needed

    print(f"🚀 Initializing MCTS Tree Search...")
    searcher = MCTSTreeSearch(llm_client=llm, max_iterations=50) # Setting to standard 50 limit
    searcher.load_repository_tree(repo_id, tree_path)

    
    all_metrics = {'R@1': [], 'R@5': [], 'R@10': [], 'MRR': [], 'latency': []}
    
    print(f"\n📊 Evaluating {len(valid_queries)} queries for {repo_id}...")
    print("-" * 80)
    
    for i, item in enumerate(valid_queries, 1):
        query = item['query']
        ground_truths = item['ground_truth']
        
        print(f"\n[{i}/{len(valid_queries)}] Q: {query[:70]}...")
        print(f"   Targets: {ground_truths}")
        
        start_time = time.time()
        # Perform search
        results = searcher.search(repo_id, query, top_k=None)
        latency = time.time() - start_time
        
        # Calculate metrics
        recalls, mrr = calculate_metrics(results, ground_truths, k_values=[1, 5, 10])
        
        # Log result
        r1, r5, r10 = recalls['R@1'], recalls['R@5'], recalls['R@10']
        print(f"   Found {len(results)} nodes. MRR: {mrr:.4f} | R@1: {r1:.0f} | R@5: {r5:.0f} | R@10: {r10:.0f} | Time: {latency:.2f}s")
        
        # Show what the MCTS actually found
        top_names = [res.get('name', 'unknown') for res in results[:10]]
        print(f"   Top 10 found: {top_names}")
        
        # Record
        all_metrics['R@1'].append(r1)
        all_metrics['R@5'].append(r5)
        all_metrics['R@10'].append(r10)
        all_metrics['MRR'].append(mrr)
        all_metrics['latency'].append(latency)
        
    # Final Report
    print("\n" + "=" * 50)
    print(f"FINAL METRICS ({repo_id}): MCTSTreeSearch")
    print("=" * 50)
    print(f"Total Queries Evaluated: {len(valid_queries)}")
    print(f"Recall@1:  {np.mean(all_metrics['R@1']):.4f}")
    print(f"Recall@5:  {np.mean(all_metrics['R@5']):.4f}")
    print(f"Recall@10: {np.mean(all_metrics['R@10']):.4f}")
    print(f"MRR:       {np.mean(all_metrics['MRR']):.4f}")
    print(f"Avg Latency: {np.mean(all_metrics['latency']):.2f}s per query")
    print("=" * 50)

if __name__ == "__main__":
    main()

import os
import json
from pathlib import Path

from backend.tree_builder import build_directory_tree
from backend.code_parser import CodeParser, enrich_tree_with_code_structure
from backend.code_index import TreeBasedSearch
from backend.lmstudio_client import LMStudioLLM

def main():
    # 1. Create test_output folder
    out_dir = Path("test_output")
    out_dir.mkdir(exist_ok=True)
    
    print("🚀 Starting Sanity Test...")
    
    # 2. STEP 1: INITIAL PARSING (NO SUMMARIES)
    print("\n--- STEP 1: Initial Parsing (No Summaries) ---")
    repo_path = "backend"  # Parse a subset of the local repo to save time
    parser = CodeParser()
    tree = build_directory_tree(repo_path)
    tree = enrich_tree_with_code_structure(tree, parser)
    
    raw_tree_dict = tree.to_dict()
    step1_path = out_dir / "1_initial_parsed_tree.json"
    with open(step1_path, "w") as f:
        json.dump(raw_tree_dict, f, indent=2)
    print(f"✅ Saved raw parsed tree to {step1_path}")

    # 3. STEP 2: JSON WITH SUMMARIES
    print("\n--- STEP 2: JSON After Summaries Embedded ---")
    # We will use your existing final.json which holds the pre-computed summaries.
    # If it is massive, we only output a subset for the sanity test logic.
    final_json_path = Path("final.json")
    if final_json_path.exists():
        with open(final_json_path, "r") as f:
            summary_tree = json.load(f)
            
        step2_path = out_dir / "2_with_summaries.json"
        with open(step2_path, "w") as f:
            json.dump(summary_tree, f, indent=2)
        print(f"✅ Saved summary embedded tree to {step2_path}")
    else:
        print("⚠️ final.json not found in root. Using raw tree for Step 2.")
        summary_tree = raw_tree_dict
        
    # 4. STEP 3: MCTS traversal & RELEVANCE SCORES
    print("\n--- STEP 3: MCTS Traversal & Relevance Scores ---")
    try:
        llm = LMStudioLLM()
        tree_search = TreeBasedSearch(llm_client=llm)
        
        # Load the tree manually into engine
        repo_id = "sanity_test_repo"
        tree_search.repositories[repo_id] = {
            'tree': summary_tree,
            'json_path': str(final_json_path)
        }
        
        # Run a quick MCTS tree search (3 simulations to hit a few nodes)
        print("Running MCTS simulations... (this alters the tree in-place by adding _score parameters)")
        results = tree_search.mcts_search(repo_id=repo_id, query="How does tree search work?", n_simulations=3)
        
        # The MCTS nodes mutate the original summary_tree in-place with `_score` values.
        step3_path = out_dir / "3_with_relevance_scores.json"
        with open(step3_path, "w") as f:
            json.dump(summary_tree, f, indent=2)
            
        print(f"✅ Saved mutated tree with LLM relevance scores to {step3_path}")
        print(f"🎉 Sanity test complete. Found {len(results)} thresholded results.")
        
    except Exception as e:
        print(f"❌ Failed to run LLM scoring: {e}")

if __name__ == "__main__":
    main()

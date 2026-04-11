import json
import argparse
import sys
from pathlib import Path

def count_nodes(node):
    """Recursively count this node and all its children."""
    if not isinstance(node, dict):
        return 0
    
    count = 1  # Count self
    
    # Children might be under 'nodes', 'children', etc.
    children = node.get('nodes', node.get('children', []))
    
    for child in children:
        count += count_nodes(child)
        
    return count

def main():
    parser = argparse.ArgumentParser(description="Count total nodes in a AST JSON tree")
    parser.add_argument("json_file", nargs='?', default="output_view/03_final.json", help="Path to the JSON file to parse")
    
    args = parser.parse_args()
    
    file_path = Path(args.json_file)
    
    if not file_path.exists():
        # Fallbacks to look for other common paths
        if Path("output_view/02_parsed.json").exists():
            file_path = Path("output_view/02_parsed.json")
            print(f"⚠️ {args.json_file} not found, falling back to {file_path}")
        elif Path("output_view/01_initial.json").exists():
            file_path = Path("output_view/01_initial.json")
            print(f"⚠️ Using early stage tree: {file_path}")
        else:
            print(f"❌ Could not find {args.json_file} or any fallback JSON trees.")
            sys.exit(1)
            
    print(f"Analyzing tree: {file_path} ...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = json.load(f)
            
        total = count_nodes(tree)
        
        print("\n" + "="*40)
        print(f"TOTAL NODES PRODUCED: {total:,}")
        print("="*40 + "\n")
        
    except json.JSONDecodeError:
        print(f"Error: {file_path} contains invalid JSON.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error parsing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

import json
import os
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_code(entry):
    """Extract code for a single JSON entry."""
    file_path = entry["path"]
    sline = entry["start_line"]
    eline = entry["end_line"]
    try:
        with open(file_path, 'r') as f:
            code = ''.join(islice(f, sline - 1, eline))
            return {**entry, "extracted_text": code}
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return {**entry, "extracted_text": None}


def process_json(json_file, max_workers=4):
    """Read JSON and extract code blocks in parallel."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(get_code, entry) for entry in data]
        for future in as_completed(futures):
            results.append(future.result())

    return results


if __name__ == "__main__":
    # Example usage:
    json_path = "input.json"  # replace with your file path
    output = process_json(json_path)

    # Save to file or print
    with open("output.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Extraction complete. Results saved to output.json")

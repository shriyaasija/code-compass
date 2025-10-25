import os
from itertools import islice
import subprocess
import json

def get_code(entry):
    file_path = entry["path"]
    sline = entry["start_line"]
    eline = entry["end_line"]
    try:
        with open(file_path, 'r') as f:
            return ''.join(islice(f, sline - 1, eline))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def clone(glink):
    comd = ["git", "clone", glink, "target-dir"]
    try:
        subprocess.run(comd, check=True)
        print("Repo cloned to target-dir")
        return "target-dir"
    except subprocess.CalledProcessError as e:
        print(f"Git clone failed: {e}")
        return None


def main():
    glink = input("Enter git link to clone: ").strip()
    location = clone(glink)
    if not location:
        return

    with open(input("json input here"),'r') as f:
        data=json.load(f)
    results=[]
    for entry in data:
        results.append(get_code(entry))

    if results:
        print("\n--- Extracted Code ---\n")
        print(results)
    else:
        print("No code extracted.")


if __name__ == "__main__":
    main()

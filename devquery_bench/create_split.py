"""Create train/test split for DevQuery-Bench."""
import json

# Test repos: the 5 largest (most interesting for the paper)
TEST_REPOS = [
    "django__django",
    "scikit-learn__scikit-learn", 
    "psf__requests",
    "encode__httpx",
    "pallets__flask",
]

with open('devquery_bench/repo_metadata.json') as f:
    repos = json.load(f)

all_ids = [r['repo_id'] for r in repos]
train_ids = [r for r in all_ids if r not in TEST_REPOS]

split = {
    'train': train_ids,
    'test': TEST_REPOS,
}

with open('devquery_bench/train_test_split.json', 'w') as f:
    json.dump(split, f, indent=2)

print(f"Train repos ({len(train_ids)}): {train_ids}")
print(f"Test repos  ({len(TEST_REPOS)}): {TEST_REPOS}")
print(f"Saved to: devquery_bench/train_test_split.json")
"""Create train/test split for DevQuery-Bench."""
import json

TEST_REPOS = [
    "pallets__flask",
    "aio-libs__aiohttp",
    "pytest-dev__pytest"
]

TRAIN_REPOS = [
    "pallets__click",
    "keleshev__schema",
    "pytoolz__toolz",
    "kennethreitz__records",
    "docopt__docopt",
    "psf__requests",
    "encode__httpx",
]

# with open('devquery_bench/repo_metadata.json') as f:
#     repos = json.load(f)

# all_ids = [r['repo_id'] for r in repos]
# train_ids = [r for r in all_ids if r not in TEST_REPOS]

split = {
    'train': TRAIN_REPOS,
    'test': TEST_REPOS,
}

with open('devquery_bench/train_test_split.json', 'w') as f:
    json.dump(split, f, indent=2)

print(f"Train repos ({len(TRAIN_REPOS)}): {TRAIN_REPOS}")
print(f"Test repos  ({len(TEST_REPOS)}): {TEST_REPOS}")
print(f"Saved to: devquery_bench/train_test_split.json")
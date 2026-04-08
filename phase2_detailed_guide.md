# Phase 2: MCTS Implementation — Stupidly Detailed Step-by-Step Guide

> **Goal:** Replace the naive greedy threshold traversal in `code_index.py` with a mathematically rigorous MCTS search algorithm.
>
> **Time estimate:** ~5-7 days
>
> **Prerequisites:** Phase 1 complete (conda env exists, deps installed, directory structure created)

---

## Sub-Phase 2A: Get Into Position (Steps 1–6)

### Step 1: Open a terminal
Open your terminal. Yes, literally open it.

### Step 2: Navigate to the project
```bash
cd /home/shriya/code-compass
```

### Step 3: Activate the conda environment
```bash
conda activate compass-paper
```
Verify you see `(compass-paper)` in your prompt. If not, something went wrong in Phase 1.

### Step 4: Make sure you're on the right branch
```bash
git branch
```
You should see `* research/neurips-submission`. If you're on `main`, switch:
```bash
git checkout research/neurips-submission
```

### Step 5: Pull latest (in case you worked on another machine)
```bash
git pull origin research/neurips-submission 2>/dev/null || echo "No remote branch yet, that's fine"
```

### Step 6: Verify Ollama is running (you'll need it for integration tests later)
```bash
curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; print([m['name'] for m in json.load(sys.stdin)['models']])" 2>/dev/null || echo "Ollama not running - start it later with: ollama serve"
```
You don't need Ollama for steps 7–30 (pure Python math), but you'll need it from step 31 onward.

---

## Sub-Phase 2B: Create the File Skeleton (Steps 7–12)

### Step 7: Create the MCTS module directory
```bash
mkdir -p research/mcts
```

### Step 8: Create the `__init__.py` for the module
```bash
touch research/__init__.py
touch research/mcts/__init__.py
```
This makes `research.mcts` importable as a Python package.

### Step 9: Create all empty files you'll be writing
```bash
touch research/mcts/mcts_node.py
touch research/mcts/mcts_search.py
touch research/mcts/simulation.py
touch research/mcts/test_mcts_node.py
touch research/mcts/test_mcts_search.py
touch research/mcts/test_ucb1.py
```

### Step 10: Verify your directory looks right
```bash
find research/mcts -type f | sort
```
Expected output:
```
research/mcts/__init__.py
research/mcts/mcts_node.py
research/mcts/mcts_search.py
research/mcts/simulation.py
research/mcts/test_mcts_node.py
research/mcts/test_mcts_search.py
research/mcts/test_ucb1.py
```

### Step 11: Git commit the skeleton
```bash
git add research/
git commit -m "phase2: create MCTS module skeleton (empty files)"
```

### Step 12: Take a breath
You now have 6 empty files. The rest of this guide tells you what goes in each one, in what order, and how to test each piece before moving on.

---

## Sub-Phase 2C: Write MCTSNode (Steps 13–20)

> **File:** `research/mcts/mcts_node.py`
>
> This is the fundamental data structure. Everything else builds on it. Write this FIRST.

### Step 13: Understand what MCTSNode represents
Open `backend/code_index.py` and re-read `_recursive_search()` (lines 89–149). The current system maintains NO state between siblings — it scores them, picks the ones above threshold, and dives in. MCTSNode will wrap each tree node and track visit counts and value estimates so the search can learn which branches are more promising.

### Step 14: Write the MCTSNode class
Open `research/mcts/mcts_node.py` and write the class. It needs these attributes and methods:

**Attributes:**
- `tree_node: Dict` — the raw JSON node from the PageIndex tree (has `title`, `summary`, `children`/`nodes`, `type`, etc.)
- `parent: Optional[MCTSNode]` — pointer to parent (None for root)
- `children: List[MCTSNode]` — MCTS children (NOT the same as tree_node's children — these are lazily created)
- `visit_count: int` — initialized to 0. This is $N(s)$ in the math.
- `total_value: float` — initialized to 0.0. This is $Q(s)$ in the math.
- `is_expanded: bool` — False until we create MCTSNode children
- `is_terminal: bool` — True if the tree_node is a leaf (function/method with line numbers)

**Properties:**
- `average_value` → `float`: Returns `total_value / visit_count` (handle division by zero → return 0.0)

**Methods:**
- `ucb1_score(c_explore: float = 1.414) -> float`:
  - If `visit_count == 0`: return `float('inf')` (unvisited nodes get infinite priority — this is the optimistic initialization trick)
  - If no parent or parent has 0 visits: return `average_value`
  - Otherwise: return `average_value + c_explore * sqrt(ln(parent.visit_count) / visit_count)`
  - You need `import math` for `math.sqrt` and `math.log`

- `expand() -> List[MCTSNode]`:
  - If already expanded or is terminal: return existing children
  - Get tree children from `tree_node.get('nodes', tree_node.get('children', []))`
  - Create an MCTSNode for each child, with `parent=self`
  - Set `is_expanded = True`
  - Return the children list

- `backpropagate(value: float)`:
  - Walk up from this node to root
  - At each node: increment `visit_count` by 1, add `value` to `total_value`
  - Use a while loop: `node = self; while node is not None: ... node = node.parent`

- `best_child(c_explore: float = 1.414) -> Optional[MCTSNode]`:
  - Return the child with the highest `ucb1_score(c_explore)`
  - Return None if no children

- `most_visited_child() -> Optional[MCTSNode]`:
  - Return child with highest `visit_count`
  - This is used for the FINAL selection (not during search — during search we use UCB1)

- `_check_terminal() -> bool`:
  - Look at `tree_node.get('type')` or `tree_node.get('node_type')`
  - Terminal if: type is `function`/`method`/`class` AND `start_line` exists in the node
  - Also terminal if: type starts with `file` AND node has no children
  - This logic should match `_is_leaf()` in `code_index.py` (line 151–165)

### Step 15: Add a `__repr__` method
Add `__repr__` so you can print nodes during debugging:
```python
def __repr__(self):
    name = self.tree_node.get('title', self.tree_node.get('name', '?'))
    return f"MCTSNode({name}, visits={self.visit_count}, val={self.average_value:.3f})"
```

### Step 16: Write tests BEFORE running anything
Open `research/mcts/test_mcts_node.py`. Write these tests:

**Test 1: `test_new_node_has_zero_visits`**
- Create a MCTSNode with a simple dict `{"title": "test", "type": "folder", "children": []}`
- Assert `visit_count == 0`, `total_value == 0.0`, `average_value == 0.0`

**Test 2: `test_ucb1_unvisited_is_infinity`**
- Create a parent node and a child node
- Assert `child.ucb1_score() == float('inf')`

**Test 3: `test_ucb1_decreases_with_visits`**
- Create parent (visit_count=10) and child (visit_count=1, total_value=0.5)
- Compute UCB1
- Now set child.visit_count=5, total_value=2.5
- Compute UCB1 again
- Assert the exploration term decreased (because N(child) increased)

**Test 4: `test_backpropagate_updates_ancestors`**
- Create a chain: root → child → grandchild
- Call `grandchild.backpropagate(0.8)`
- Assert all three nodes have `visit_count == 1` and `total_value == 0.8`

**Test 5: `test_expand_creates_children`**
- Create a node with 3 children in its `tree_node`
- Call `expand()`
- Assert `len(children) == 3` and `is_expanded == True`
- Call `expand()` again — assert it returns the same children (idempotent)

**Test 6: `test_terminal_detection`**
- Create a node with `{"type": "function", "start_line": 10, "end_line": 20}` → assert `is_terminal == True`
- Create a node with `{"type": "folder", "children": [...]}` → assert `is_terminal == False`
- Create a node with `{"type": "file_py", "children": []}` → assert `is_terminal == True` (file with no children = leaf)

**Test 7: `test_best_child_selects_highest_ucb`**
- Create a parent with 3 children, give them different visit counts and values
- Assert `best_child()` returns the one with highest UCB1 score

**Test 8: `test_most_visited_child`**
- Create a parent with 3 children, give them different visit counts
- Assert `most_visited_child()` returns the one with most visits

### Step 17: Run the tests
```bash
python -m pytest research/mcts/test_mcts_node.py -v
```
All 8 should pass. If any fail, fix the MCTSNode code, NOT the tests (tests encode the math spec).

### Step 18: Write the UCB1 mathematical correctness tests
Open `research/mcts/test_ucb1.py`. These tests verify the MATH is right, not just the code.

**Test 1: `test_ucb1_formula_manual_computation`**
- Parent: N=100. Child: N=10, Q=7.0 (avg=0.7). c=1.414
- Expected: 0.7 + 1.414 * sqrt(ln(100)/10) = 0.7 + 1.414 * sqrt(4.605/10) = 0.7 + 1.414 * 0.6786 ≈ 1.6595
- Assert `abs(child.ucb1_score(1.414) - expected) < 0.001`

**Test 2: `test_ucb1_exploration_dominates_early`**
- With N_parent=100, create two children: one with (N=1, Q=0.3) and one with (N=50, Q=45.0 → avg=0.9)
- Assert the LESS visited child has a higher UCB1 score (exploration dominates)

**Test 3: `test_ucb1_exploitation_dominates_late`**
- With N_parent=100000, create two children: one with (N=10000, Q=3000 → avg=0.3) and one with (N=10000, Q=9000 → avg=0.9)
- Assert the HIGHER value child has a higher UCB1 score (exploitation dominates when N is large)

**Test 4: `test_ucb1_c_zero_is_pure_exploitation`**
- Set c_explore=0
- Assert UCB1 equals average_value exactly

**Test 5: `test_ucb1_convergence`**
- Simulate pulling a "good" arm 1000 times vs a "bad" arm
- Assert the good arm's UCB1 score is higher

### Step 19: Run UCB1 tests
```bash
python -m pytest research/mcts/test_ucb1.py -v
```

### Step 20: Git commit
```bash
git add research/mcts/mcts_node.py research/mcts/test_mcts_node.py research/mcts/test_ucb1.py
git commit -m "phase2: implement MCTSNode with UCB1, backpropagation, expansion (8+5 tests passing)"
```

---

## Sub-Phase 2D: Write the Simulation Module (Steps 21–25)

> **File:** `research/mcts/simulation.py`
>
> The simulation module is the bridge between MCTS and the LLM. It takes an unexplored node and returns a score ∈ [0,1] estimating how relevant that subtree is to the query.

### Step 21: Understand what simulation does in MCTS
In classical MCTS (e.g., AlphaGo), simulation means playing random moves until the game ends. In our case, "simulation" means asking the LLM: *"Given this query, how relevant is this code node?"*

The key insight: we ALREADY have this logic in `code_index.py` → `_score_siblings()` (lines 167–239). The simulation module wraps this existing functionality with caching and batching.

### Step 22: Design the simulation module
The module should have one main class `LLMSimulator` with these methods:

**`__init__(self, llm_client, cache_size=1000)`**
- Store the LLM client (Ollama or LMStudio)
- Create a cache dict `{}` to store previous scores
- Cache key = `(node_id, query_hash)` → avoids re-scoring the same node for the same query

**`simulate(self, node: MCTSNode, query: str) -> float`**
- If node is terminal: score it directly with the LLM (single node relevance)
- If node has children (but hasn't been expanded in MCTS yet): score all children's summaries in one batch LLM call (this is the existing `_score_siblings` approach)
- Return a score ∈ [0,1]
- Check cache first; store result in cache

**`batch_score_children(self, children: List[MCTSNode], query: str, parent: MCTSNode) -> Dict[str, float]`**
- This is essentially `_score_siblings()` from `code_index.py`
- Takes a list of tree nodes, builds the scoring prompt, calls LLM, parses JSON response
- Returns dict of `{node_title: score}`
- IMPORTANT: Copy the prompt format and JSON parsing logic from the existing `_score_siblings` — don't reinvent it

**`_cache_key(self, node_id: str, query: str) -> str`**
- Return a hash of `(node_id, query)` for cache lookup

### Step 23: Handle the scoring prompt
Copy the prompt from `code_index.py` lines 173–216 into your simulation module. Don't change the prompt format — it already works and has been tested. Just wrap it in the `LLMSimulator` class.

The prompt looks like:
```
Query: "{query}"
Location: root → src → model.py
Rate relevance (0.0 to 1.0) for each item:
...
Respond with ONLY a JSON object: {"item_name": score, ...}
```

### Step 24: Handle the JSON parsing
Also copy `_parse_scores()` from `code_index.py` lines 241–298. Same logic — handles markdown code blocks, extracts JSON, validates and clamps scores to [0,1].

### Step 25: Git commit
```bash
git add research/mcts/simulation.py
git commit -m "phase2: implement LLMSimulator (wraps _score_siblings with caching)"
```

No tests yet for this file — it requires a running LLM. We'll test it during integration (step 35).

---

## Sub-Phase 2E: Write the Full MCTS Search Algorithm (Steps 26–33)

> **File:** `research/mcts/mcts_search.py`
>
> This is the core algorithm. It orchestrates Selection → Expansion → Simulation → Backpropagation.

### Step 26: Understand the 4 phases of MCTS
Before writing code, draw this on paper or a whiteboard:

```
     ROOT (repo)
    /     \
  src/    tests/     ← Selection: pick highest UCB1
  / \      |
 a.py b.py  ...      ← Expansion: create MCTSNode children
  |
 func1, func2        ← Simulation: LLM scores these
                     ← Backpropagation: update values up to root
```

One "iteration" of MCTS does all 4 phases once. We repeat for N iterations (e.g., 50), then extract results.

### Step 27: Design the MCTSSearch class
Create a class `MCTSSearch` with these pieces:

**`__init__(self, llm_client, max_iterations=50, c_explore=1.414, max_depth=None)`**
- `max_iterations`: How many MCTS iterations per query (more = better but slower). Start with 50.
- `c_explore`: UCB1 exploration constant. Default √2 ≈ 1.414.
- `max_depth`: Optional depth limit. None = no limit (go to leaves).
- Create an `LLMSimulator` internally.

**`search(self, tree: Dict, query: str, top_k: int = 10) -> List[Dict]`**
The main search method. Steps:
1. Create root MCTSNode from the tree
2. Run `max_iterations` iterations of MCTS
3. After all iterations: collect all terminal (leaf) nodes that were visited
4. Rank them by average_value (descending)
5. Return top_k as list of result dicts (same format as `code_index.py` returns)

**`_run_one_iteration(self, root: MCTSNode, query: str)`**
One MCTS iteration:
1. **SELECTION:** Start at root. While current node is expanded and not terminal: pick `best_child(c_explore)`. This walks down the tree following UCB1.
2. **EXPANSION:** When we reach an unexpanded non-terminal node: call `expand()` to create MCTSNode children.
3. **SIMULATION:** Use `LLMSimulator.batch_score_children()` to score all newly expanded children. Pick the child with the highest score.
4. **BACKPROPAGATION:** Call `selected_child.backpropagate(score)`.

### Step 28: Handle the "first visit" edge case
On iteration 1, the root hasn't been expanded yet. So:
- Iteration 1: Selection goes nowhere (root not expanded), Expansion creates root's children, Simulation scores them, Backprop updates root + best child.
- Iteration 2: Selection follows UCB1 from root to best child. If that child isn't expanded, expand it. Score its children. Backprop.
- Iteration 3+: Selection goes deeper as the tree gets explored.

This means the tree is explored top-down, breadth-first at first, then focusing on promising branches.

### Step 29: Handle batched LLM calls carefully
The key optimization: score ALL siblings in ONE LLM call, not one call per child. This is how the current `_score_siblings` works. In MCTS terms:
- When we expand a node, we get N children
- We call `batch_score_children()` ONCE to get scores for all N
- We store these scores as initial values for each child
- We pick the best-scoring child for deeper exploration
- This costs 1 LLM call per tree level, same as the current greedy approach

The MCTS advantage comes from **revisiting** — on later iterations, we may re-explore a previously low-scoring branch if its UCB1 exploration term gets high enough.

### Step 30: Implement result extraction
After all iterations complete, walk the entire MCTSNode tree and collect all terminal nodes:
```python
def _collect_results(self, root: MCTSNode) -> List[Dict]:
    results = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_terminal and node.visit_count > 0:
            results.append({
                'node_id': node.tree_node.get('node_id', ''),
                'name': node.tree_node.get('title', node.tree_node.get('name', '')),
                'node_type': node.tree_node.get('type', ''),
                'summary': node.tree_node.get('summary', ''),
                'path': node.tree_node.get('path', ''),
                'similarity_score': node.average_value,
                'metadata': {
                    'file_path': node.tree_node.get('path', ''),
                    'start_line': node.tree_node.get('start_line'),
                    'end_line': node.tree_node.get('end_line'),
                    'signature': node.tree_node.get('signature', ''),
                    'docstring': node.tree_node.get('summary', '')
                },
                'mcts_stats': {
                    'visit_count': node.visit_count,
                    'average_value': node.average_value,
                }
            })
        for child in node.children:
            stack.append(child)
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results
```

### Step 31: Add early termination
Add a convergence check: if the top leaf has received >80% of all visits in the last 10 iterations, the search has converged and we can stop early.
```python
# Inside the iteration loop:
if iteration > 10:
    top_leaf = max(all_terminals, key=lambda n: n.visit_count)
    if top_leaf.visit_count > 0.8 * root.visit_count:
        print(f"  Early termination at iteration {iteration} (converged)")
        break
```

### Step 32: Add logging/printing
Add print statements like the current `code_index.py` does (lines 68–79, 97–98, etc.):
```
======================================================================
🔍 MCTS SEARCH: 'How do I train the model?'
   Max iterations: 50, c_explore: 1.414
======================================================================
  Iteration 1: Expanding root (3 children)
  Iteration 2: Exploring src/ (5 children)
  Iteration 5: Exploring model.py (4 children)
  ...
  ✅ MCTS COMPLETE: 23 iterations, 8 leaf nodes found, 5 LLM calls
======================================================================
```

### Step 33: Write tests for MCTSSearch (mock LLM)
Open `research/mcts/test_mcts_search.py`. For these tests, create a **mock LLM** that returns predetermined scores (no actual Ollama needed).

**Mock LLM class:**
```python
class MockLLM:
    def __init__(self, score_map: Dict[str, float]):
        """score_map: {"node_title": score} for predetermined scoring"""
        self.score_map = score_map
        self.call_count = 0
    
    def chat(self, messages, **kwargs):
        self.call_count += 1
        # Parse the prompt to find items, return scores from map
        # Return JSON string like: {"model.py": 0.9, "README.md": 0.1}
        ...
```

**Test 1: `test_mcts_finds_correct_leaf`**
- Build a 3-level mock tree: root → [file_a, file_b] → [func1, func2, func3, func4]
- Set mock scores so func2 in file_a is the "correct" answer (score=0.95)
- Run MCTS search
- Assert func2 is in the top-1 result

**Test 2: `test_mcts_explores_multiple_branches`**
- Build a tree where two branches have high-scoring leaves
- Run MCTS with enough iterations
- Assert BOTH leaves appear in results (not just the first one found)

**Test 3: `test_mcts_respects_max_iterations`**
- Set max_iterations=5
- Run search
- Assert LLM call count <= 5 (one call per iteration)

**Test 4: `test_mcts_early_termination`**
- Set max_iterations=100 but make one branch obviously dominant
- Assert actual iterations < 100 (early termination triggered)

**Test 5: `test_mcts_result_format`**
- Run search, check each result has: `node_id`, `name`, `similarity_score`, `metadata` with `file_path`, `start_line`, `end_line`

**Test 6: `test_mcts_on_mock_pageindex_tree`**
- Load the actual `mock_pageindex_tree.json` from the project root
- Use MockLLM with sensible scores
- Run MCTS and verify results make sense

Run tests:
```bash
python -m pytest research/mcts/test_mcts_search.py -v
```

### Git commit:
```bash
git add research/mcts/mcts_search.py research/mcts/test_mcts_search.py
git commit -m "phase2: implement full MCTS search algorithm with 4 phases + early termination (6 tests passing)"
```

---

## Sub-Phase 2F: Integrate with Existing code_index.py (Steps 34–38)

### Step 34: Read code_index.py one more time
Open `backend/code_index.py` and study the public API:
- `load_repository_tree(repo_id, json_tree_path)` — loads JSON
- `search(repo_id, query, top_k, min_similarity, node_type_filter)` — returns results
- `search_and_format_for_chatbot(repo_id, query, top_k)` — formats for retrieval.py

Your MCTS class needs to expose the EXACT same interface so it's a drop-in replacement.

### Step 35: Create MCTSTreeSearch class in code_index.py
Add a NEW class `MCTSTreeSearch` in `backend/code_index.py` (or in a new file `backend/mcts_code_index.py` if you prefer). This class:

- Has the same `__init__(self, llm_client, ...)` signature
- Has the same `load_repository_tree()` method
- Has `search()` that internally uses `MCTSSearch` from `research/mcts/mcts_search.py`
- Has `search_and_format_for_chatbot()` that wraps search results

**DO NOT modify the existing `TreeBasedSearch` class.** Keep it intact — it's baseline B4 ("Greedy-Tree") in the paper.

### Step 36: Test with the actual mock_pageindex_tree.json
Start Ollama if not running:
```bash
# In a separate terminal:
ollama serve
```

Then test:
```bash
python -c "
from backend.code_index import MCTSTreeSearch  # or wherever you put it
from backend.ollama_client import OllamaLLM

llm = OllamaLLM(model='qwen2.5:7b')  # or qwen3:8b
mcts = MCTSTreeSearch(llm, max_iterations=30, c_explore=1.414)
mcts.load_repository_tree('mock', './mock_pageindex_tree.json')
results = mcts.search('mock', 'How do I train the model?')
for r in results[:5]:
    print(f\"  {r['name']}: score={r['similarity_score']:.3f}\")
"
```

Expected output should include `train_one_epoch`, `validate_model`, `save_checkpoint` near the top.

### Step 37: Compare with greedy search (sanity check)
Run the same query through the old TreeBasedSearch:
```bash
python -c "
from backend.code_index import TreeBasedSearch
from backend.ollama_client import OllamaLLM

llm = OllamaLLM(model='qwen2.5:7b')
greedy = TreeBasedSearch(llm, threshold=0.5)
greedy.load_repository_tree('mock', './mock_pageindex_tree.json')
results = greedy.search('mock', 'How do I train the model?')
for r in results[:5]:
    print(f\"  {r['name']}: score={r['similarity_score']:.3f}\")
"
```

Both should find similar functions, but MCTS might find different ones or rank them differently. That's fine — the benchmark will prove which is better.

### Step 38: Git commit the integration
```bash
git add backend/
git commit -m "phase2: integrate MCTSTreeSearch as drop-in replacement for TreeBasedSearch"
```

---

## Sub-Phase 2G: Wire into Benchmark Framework (Steps 39–43)

### Step 39: Open benchmark.py and study run_tree_search()
Look at lines 511–655. This method:
1. Creates a `TreeBasedSearch` instance
2. Loads trees for each repo
3. Runs queries and collects metrics
4. Counts LLM calls

You need to add a parallel method: `run_mcts_search()`.

### Step 40: Add MCTS mode to benchmark.py
Add a new mode `--mode mcts` to argparse (line 892). Add a new method `run_mcts_search()` that mirrors `run_tree_search()` but uses `MCTSTreeSearch`.

Key differences from `run_tree_search()`:
- Uses `MCTSTreeSearch` instead of `TreeBasedSearch`
- Tracks `max_iterations` and `c_explore` as hyperparameters in the output
- Logs MCTS-specific stats (iterations used, early termination count)

### Step 41: Test benchmark with MCTS mode
First, make sure benchmark data is prepared (should be from Phase 1):
```bash
ls benchmark_results/benchmark_metadata.json
```
If it doesn't exist, run:
```bash
python benchmark.py --mode prepare --num-repos 3 --queries-per-repo 5
```

Then test MCTS on a small subset:
```bash
python benchmark.py --mode mcts --num-repos 1 --queries-per-repo 3 --skip-prepare
```

### Step 42: Run a comparison (3 repos, quick test)
```bash
# Dense baseline
python benchmark.py --mode dense-only --num-repos 3 --queries-per-repo 5 --skip-prepare

# MCTS
python benchmark.py --mode mcts --num-repos 3 --queries-per-repo 5 --skip-prepare
```

Compare the output reports. Don't worry if MCTS is worse on 3 repos — the real benchmark is on 25 repos.

### Step 43: Git commit
```bash
git add benchmark.py
git commit -m "phase2: add --mode mcts to benchmark.py for MCTS evaluation"
```

---

## Sub-Phase 2H: Final Cleanup & Phase 2 Merge (Steps 44–47)

### Step 44: Run ALL tests one final time
```bash
python -m pytest research/mcts/ -v
```
All tests must pass. Fix any failures.

### Step 45: Check for linting issues
```bash
python -m py_compile research/mcts/mcts_node.py
python -m py_compile research/mcts/mcts_search.py
python -m py_compile research/mcts/simulation.py
```
No output = no syntax errors.

### Step 46: Write a brief docstring at the top of each file
Every file should start with a module docstring explaining what it does, the key algorithm/math, and how it fits into the system. Reviewers (and future you) will thank you.

### Step 47: Final git commit and tag
```bash
git add -A
git commit -m "phase2: MCTS implementation complete — MCTSNode, LLMSimulator, MCTSSearch, benchmark integration, 19 tests passing"
git tag phase2-complete
git push origin research/neurips-submission
git push origin phase2-complete
```

---

## ✅ Phase 2 Completion Checklist

Before moving to Phase 3, verify:

- [ ] `research/mcts/mcts_node.py` — MCTSNode class with UCB1, backprop, expand
- [ ] `research/mcts/simulation.py` — LLMSimulator wrapping `_score_siblings` with caching
- [ ] `research/mcts/mcts_search.py` — Full MCTS algorithm (Select/Expand/Simulate/Backprop)
- [ ] `research/mcts/test_mcts_node.py` — 8 tests passing
- [ ] `research/mcts/test_ucb1.py` — 5 tests passing
- [ ] `research/mcts/test_mcts_search.py` — 6 tests passing
- [ ] `MCTSTreeSearch` integrated (same API as `TreeBasedSearch`)
- [ ] `benchmark.py --mode mcts` works
- [ ] Compared MCTS vs Greedy on mock_pageindex_tree.json (both return sensible results)
- [ ] All tests: `python -m pytest research/mcts/ -v` → 19 passing
- [ ] Git tagged `phase2-complete`
- [ ] Pushed to remote

---

> **Next:** Phase 3 (RL Adaptive Index) — ask me for the detailed guide when ready.

# Analysis: What Has Been Built So Far

**Last updated:** 2026-04-17

This document is a brutally honest audit of every component in the Code Compass codebase — what works, what doesn't, what's missing, and what needs to change for the DevQuery-Bench paper rewrite.

---

## 1. The Application Layer (WORKS — No Changes Needed)

### What exists
The original Code Compass product: a FastAPI backend + Streamlit frontend that lets you ask natural language questions about a codebase.

| File | Status | What it does |
|------|--------|-------------|
| `backend/api.py` | ✅ Working | FastAPI server, endpoints for `/initialize`, `/query`, `/search`, `/health` |
| `backend/code_index.py` | ✅ Working | Greedy tree search — the original LLM-at-every-level traversal |
| `backend/code_index2.py` | ✅ Working | MCTS drop-in replacement for `code_index.py` |
| `backend/lmstudio_client.py` | ✅ Working | OpenAI-compatible wrapper for LM Studio on `localhost:1234` |
| `backend/ollama_client.py` | ✅ Working | OpenAI-compatible wrapper for Ollama on `localhost:11434` |
| `backend/code_parser.py` | ✅ Working | tree-sitter parsing — extracts functions/classes/methods with line numbers |
| `backend/tree_builder.py` | ✅ Working | Walks a directory and builds the folder/file hierarchy dict |
| `backend/summarizer.py` | ✅ Working | Bottom-up LLM summarization (leaves first, then parents) |
| `backend/embed_tree.py` | ✅ Working | Embeds all node summaries with `all-MiniLM-L6-v2` (384-dim) |
| `backend/retrieval.py` | ✅ Working | Orchestrates search → LLM response generation |
| `backend/semantic_search.py` | ✅ Working | Embedding-based flat search |
| `backend/pageindex_semantic_search.py` | ✅ Working | Dense baseline (PageIndex) |
| `frontend/app.py` | ✅ Working | Streamlit chat UI |

**Verdict:** This layer is untouched by the paper rewrite. It stays as-is.

---

## 2. The MCTS Research Engine (WORKS — Needs Extension)

### What exists

| File | Status | What it does |
|------|--------|-------------|
| `research/mcts/mcts_node.py` | ✅ Working | `MCTSNode` class with UCB1 + PUCT scoring, backpropagation, expansion. 247 lines. |
| `research/mcts/mcts_search.py` | ✅ Working | Baseline MCTS — LLM scoring at every node expansion |
| `research/mcts/puct_search.py` | ✅ Working | PUCT-MCTS — prior MLP at internal nodes, LLM at leaves only. 334 lines. Has online adaptation (`_online_update`). |
| `research/mcts/relevance_prior.py` | ✅ Working | 200K-param MLP: Linear(768→256)→LN→ReLU→Drop→Linear(256→64)→ReLU→Linear(64→1)→Sigmoid |
| `research/mcts/prior.pt` | ✅ Exists | Trained weights from CodeSearchNet 19-repo training split |
| `research/mcts/simulation.py` | ✅ Working | LLM simulation bridge |
| `research/mcts/test_*.py` | ✅ Passing | 19 unit tests for node math, PUCT, UCB1 |

### What needs to change for the paper
- **`puct_search.py` needs an `online_update` toggle** — already has `self.do_online_update` flag ✅
- **Need to add ablation support**: random prior mode, UCB1-with-prior mode, `c_puct` sweep mode
- **Need to track per-query metrics** in sequence (for the adaptation trajectory experiment)
- **`prior.pt` must be retrained** on DevQuery-Bench training split (new repos, new queries)

---

## 3. The RL Index Restructuring (PARTIALLY WORKS — Not Used in Paper)

### What exists

| File | Status | What it does |
|------|--------|-------------|
| `research/rl_index/env.py` | ✅ Working | Gymnasium `TreeIndexEnv` — RL agent performs Merge/Split/Reparent |
| `research/rl_index/tree_mutations.py` | ✅ Working | 414 lines. Merge, Split (KMeans), Reparent, validate, utility functions |
| `research/rl_index/tree_state.py` | ✅ Working | Feature extraction for RL observation space |
| `research/rl_index/reward.py` | ✅ Working | `RewardComputer` (delta MRR + depth penalty) + `ProxyMRREstimator` |
| `research/rl_index/train_ppo.py` | ✅ Working | MaskablePPO training via Stable Baselines 3 |
| `research/rl_index/offline_trainer.py` | ✅ Working | Batch RL training pipeline |
| `research/rl_index/evaluate_rl.py` | ✅ Working | RL evaluation script |

### Problems with current RL results
The RL agent learned **NoOp on 4/5 test repos** — it did nothing:
```
OpenAccess_EPUB  → 0 Merge, 30 NoOp, delta_quality = 0.000
ramses           → 29 Merge, 1 NoOp, delta_quality = +0.005  ← only useful one
uqbar            → 0 Merge, 30 NoOp, delta_quality = 0.000
mrcrowbar        → 0 Merge, 30 NoOp, delta_quality = 0.000
finnsyll         → 0 Merge, 30 NoOp, delta_quality = 0.000
```

**Root cause:** The merge similarity threshold (0.3) is too conservative, and the proxy MRR reward doesn't correlate strongly with real LLM-based MRR.

### Verdict for the paper
RL restructuring is **not part of the main paper results**. It's mentioned in §7 as a proposed extension (Thompson Sampling bandit over mutations). This is the right call — the RL results are too weak to include.

---

## 4. The Data Pipeline (WORKS — Needs New Data)

### What exists

| File/Dir | Status | What it does |
|----------|--------|-------------|
| `benchmark.py` | ✅ Working | Multi-mode benchmark: `--mode prepare`, `--mode dense-only`, `--mode full`, `--mode mcts` |
| `prepare_proper_trees.py` | ✅ Working | Clone → tree-sitter → LLM summarize → embed → save JSON |
| `build_prior_training_data.py` | ✅ Working | Generates (query_emb, node_emb, label) triples for prior training |
| `train_prior.py` | ✅ Working | Trains the RelevancePrior MLP. 15 epochs, weighted BCE. |
| `run_puct_evaluation.py` | ✅ Working | Runs PUCT vs Baseline MCTS on test repos |
| `run_full_pipeline.py` | ✅ Working | End-to-end orchestrator |
| `benchmark_results/cloned_repos/` | ✅ Has data | 24 cloned CodeSearchNet repos |
| `benchmark_results/proper_trees/` | ✅ Has data | 23 summarized + embedded JSON trees |
| `benchmark_results/prior_training_data.pt` | ✅ Has data | ~35MB training tensor |
| `benchmark_results/train_test_split.json` | ✅ Has data | 19 train / 5 test repos |

### What needs to change
**Everything in `benchmark_results/` is for CodeSearchNet.** The paper rewrite replaces CodeSearchNet with DevQuery-Bench. This means:
- New repos need to be cloned (django, scikit-learn, requests, httpx, flask, etc.)
- New trees need to be built
- New queries need to be generated (naturalistic, not docstrings)
- New training data needs to be generated
- New prior needs to be trained
- All evaluation needs to be re-run

**The good news**: all the *scripts* are reusable. The pipeline (`prepare_proper_trees.py` → `build_prior_training_data.py` → `train_prior.py` → `run_puct_evaluation.py`) is battle-tested and just needs to be pointed at new data.

---

## 5. Current Benchmark Results (HONEST ASSESSMENT)

### CodeSearchNet Results (25 repos, clean docstring queries)

| Method | R@1 | R@5 | MRR | LLM Calls/q |
|--------|-----|-----|-----|-------------|
| Dense Baseline | **0.952** | **1.000** | **0.975** | 0 |
| Greedy-Tree | 0.576 | 0.720 | 0.650 | 2.0 |
| Baseline MCTS | 0.040 | 0.133 | 0.075 | 7.56 |
| PUCT-MCTS | 0.013 | 0.027 | 0.018 | **6.17** |

### PUCT vs Baseline MCTS (5 test repos)

| Repo | PUCT MRR | Base MRR | PUCT LLM | Base LLM |
|------|----------|----------|----------|----------|
| OpenAccess_EPUB | 0.000 | 0.000 | 6.13 | 8.40 |
| ramses | 0.022 | 0.272 | 7.87 | 8.33 |
| uqbar | 0.067 | 0.000 | 6.27 | 7.67 |
| mrcrowbar | 0.000 | 0.100 | 6.40 | 3.40 |
| finnsyll | — | — | — | — |

### Why the results are bad
1. **CodeSearchNet queries ARE docstrings** — cosine similarity between query and function embedding is ~0.95. Dense retrieval is essentially cheating.
2. **MCTS/PUCT on small repos (30-120 functions)** — tree depth is only 2-3 levels. There's no structural advantage to exploit.
3. **The prior was trained on the same small-repo regime** — it learned nothing useful because there was nothing useful to learn.

### What the paper rewrite fixes
DevQuery-Bench uses:
- **Larger repos** (300–8000 functions) where flat retrieval drowns in noise
- **Naturalistic queries** (α > 0.5) where embedding similarity is low
- This is exactly the regime where MCTS + prior should win

---

## 6. Dependencies & Environment

### Installed (in requirements.txt)
```
fastapi, uvicorn, streamlit, requests, python-dotenv, pydantic
openai, gitpython
tree-sitter + 9 language bindings (python, js, ts, cpp, c, java, go, rust, ruby, php)
```

### Installed (but not in requirements.txt — need to be added)
```
torch, sentence-transformers
stable-baselines3, sb3-contrib (for MaskablePPO)
gymnasium
scikit-learn (for KMeans in tree_mutations.py Split)
numpy
rank_bm25 (TBD — needed for BM25 baseline)
nltk (TBD — needed for BLEU naturalism score)
```

### Environment
- Python 3.10+ in a `venv`
- LM Studio running locally on `localhost:1234`
- 1 GPU available

---

## 7. Gap Analysis: What's Missing for the Paper

| What's needed | Status | Effort |
|--------------|--------|--------|
| DevQuery-Bench repos (15 large Python repos) | ❌ Not started | Day 1 |
| Naturalistic query generation (300 queries) | ❌ Not started | Day 1 |
| Ground truth annotation (300 function labels) | ❌ Not started | Day 1 |
| Naturalism score computation (BLEU-1) | ❌ Not started | Day 1 |
| New tree construction (15 repos) | ❌ Not started | Day 1 |
| Prior retraining on DevQuery-Bench | ❌ Not started | Day 2 |
| BM25 baseline implementation | ❌ Not started | Day 3 |
| Dense + BM25 hybrid baseline | ❌ Not started | Day 3 |
| Full benchmark run (7 methods × 5 repos) | ❌ Not started | Day 3 |
| Bootstrap CI computation | ❌ Not started | Day 3 |
| Online adaptation trajectory logging | ❌ Not started | Day 4 |
| Ablation variants (random prior, no-update, UCB1) | ❌ Not started | Day 5 |
| c_puct sweep | ❌ Not started | Day 5 |
| Regime boundary grid evaluation | ❌ Not started | Day 6 |
| Contour plot generation | ❌ Not started | Day 6 |
| Paper placeholder filling | ❌ Not started | Day 7 |

---

## 8. Files That Can Be Reused As-Is

These existing files need **zero changes** for the paper rewrite:

- `backend/code_parser.py` — tree-sitter parsing
- `backend/tree_builder.py` — directory tree construction
- `backend/summarizer.py` — bottom-up LLM summarization
- `backend/embed_tree.py` — embedding with MiniLM
- `backend/lmstudio_client.py` — LLM interface
- `research/mcts/mcts_node.py` — MCTSNode with UCB1 + PUCT
- `research/mcts/relevance_prior.py` — prior MLP architecture
- `research/mcts/puct_search.py` — PUCT search (minor extensions for ablations)
- `train_prior.py` — prior training loop

## 9. Files That Need Modification

| File | Change needed |
|------|--------------|
| `prepare_proper_trees.py` | Point at DevQuery-Bench repo list instead of CodeSearchNet metadata |
| `build_prior_training_data.py` | Point at new metadata format (DevQuery-Bench JSON) |
| `run_puct_evaluation.py` | Add per-query logging, adaptation trajectory, ablation modes |
| `benchmark.py` | Add BM25 mode, hybrid mode, bootstrap CI |

## 10. New Files That Need to Be Created

| File | Purpose |
|------|---------|
| `devquery_bench/generate_queries.py` | LLM-based naturalistic query generation |
| `devquery_bench/compute_naturalism.py` | BLEU-1 naturalism score computation |
| `devquery_bench/devquery_bench.json` | The final benchmark file (300 triples) |
| `devquery_bench/repo_list.json` | The 15 repos with GitHub URLs |
| `experiments/run_day3_main_eval.py` | Orchestrator for all 7 baselines |
| `experiments/run_day4_adaptation.py` | Online adaptation trajectory experiment |
| `experiments/run_day5_ablations.py` | All ablation variants |
| `experiments/run_day6_regime.py` | Regime boundary grid evaluation |
| `experiments/generate_figures.py` | matplotlib figure generation for paper |

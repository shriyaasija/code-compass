# 🧭 Code Compass

**Repository-level code retrieval via learned tree search.**

Code Compass navigates codebases the way developers do — by searching the hierarchy. It applies **Monte Carlo Tree Search (MCTS)** over tree-sitter ASTs, guided by a lightweight learned prior that replaces expensive LLM calls at internal nodes. The LLM is invoked only at leaf nodes for final verification, reducing token consumption by up to 56%.

> 📄 **Research paper:** [`paper.md`](paper.md) — *Code Compass: Navigating Repository-Scale Codebases via Learned Tree Search*

---

## Architecture

```
Query: "How do I authenticate?"
         │
         ▼
┌──────────────────────────────────────┐
│        PUCT-MCTS Search Engine       │
│                                      │
│  Selection ──→ Expansion ──→ Simulation
│  (PUCT)       (tree-sitter)   Prior MLP (internal nodes)
│                                LLM (leaf nodes only)
│       ↑                              │
│  Backpropagation ←───────────────────┘
│       │
│  Online Prior Adaptation             │
│  (1 gradient step from visit counts) │
└──────────────────────────────────────┘
         │
         ▼
  Ranked code functions with file paths + line numbers
```

### Key Idea

| Component | Role | Cost |
|-----------|------|------|
| **Relevance Prior** (MLP, 200K params) | Scores internal nodes (folders, files, classes) | ~0.1ms |
| **LLM** (7B, local) | Scores leaf nodes (functions, methods) | ~2–5s |
| **MCTS** | Balances exploration vs exploitation | UCB1/PUCT |
| **Online Adaptation** | Updates prior after each query from visit counts | ~0.1ms |

---

## Project Structure

```
code-compass/
├── backend/
│   ├── api.py                        # FastAPI server
│   ├── code_index.py                 # Greedy tree search (baseline)
│   ├── code_index2.py                # MCTS tree search integration
│   ├── lmstudio_client.py            # LM Studio LLM client
│   ├── ollama_client.py              # Ollama LLM client
│   ├── code_parser.py                # tree-sitter code parsing
│   ├── tree_builder.py               # Directory → tree construction
│   ├── retrieval.py                  # Code retrieval + LLM response
│   ├── pageindex_semantic_search.py  # Dense baseline (PageIndex)
│   └── semantic_search.py            # Embedding-based search
│
├── research/
│   ├── mcts/
│   │   ├── mcts_node.py              # MCTSNode with UCB1 + PUCT scoring
│   │   ├── mcts_search.py            # Baseline MCTS (LLM at every node)
│   │   ├── puct_search.py            # PUCT-MCTS with learned prior
│   │   ├── relevance_prior.py        # Prior MLP (200K params)
│   │   ├── simulation.py             # LLM simulation bridge
│   │   ├── test_mcts_node.py         # Unit tests for MCTSNode
│   │   ├── test_mcts_search.py       # Integration tests for MCTS
│   │   ├── test_ucb1.py              # Mathematical correctness tests
│   │   └── prior.pt                  # Trained prior weights
│   │
│   └── rl_index/
│       ├── env.py                    # Gymnasium environment for tree mutations
│       ├── tree_mutations.py         # Merge / Split / Reparent operations
│       ├── tree_state.py             # State extraction for RL agent
│       ├── reward.py                 # MRR-based reward computation
│       ├── train_ppo.py              # PPO training script
│       └── offline_trainer.py        # Batch RL training pipeline
│
├── frontend/
│   └── app.py                        # Streamlit chat UI
│
├── benchmark.py                      # Full benchmark pipeline (Dense + Tree + MCTS)
├── run_puct_evaluation.py            # PUCT vs Baseline MCTS evaluation
├── run_full_pipeline.py              # End-to-end: prepare → RL train → MCTS eval
├── prepare_proper_trees.py           # Clone repos → tree-sitter → summarize → embed
├── build_prior_training_data.py      # Generate (query, node, label) training pairs
├── train_prior.py                    # Train the relevance prior MLP
├── train_offline_rl.py               # Offline RL for tree restructuring
├── paper.md                          # Research paper draft
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- An LLM provider:
  - [LM Studio](https://lmstudio.ai/) (recommended — GUI, local, OpenAI-compatible)
  - [Ollama](https://ollama.com/) (CLI-based)

### Setup

```bash
git clone https://github.com/shriyaasija/code-compass.git
cd code-compass

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Run the Application (Chat UI)

```bash
# Terminal 1: Start LM Studio and load a model, then start its server on localhost:1234

# Terminal 2: Start backend
LLM_PROVIDER=lmstudio python -m backend.api

# Terminal 3: Start frontend
streamlit run frontend/app.py
```

Open `http://localhost:8501`, initialize with `./mock_repository` and `./mock_pageindex_tree.json`, and ask questions.

---

## Research Pipeline

### Step 1: Prepare Benchmark Data

Downloads CodeSearchNet repos, parses with tree-sitter, generates LLM summaries, embeds.

```bash
# Prepare flat trees (for Dense + Greedy baselines)
python benchmark.py --mode prepare --num-repos 25 --provider lmstudio

# Build proper trees with full AST hierarchy
python prepare_proper_trees.py --provider lmstudio --num-repos 25
```

### Step 2: Train the Relevance Prior

```bash
# Generate training pairs from the 19 training repos
python build_prior_training_data.py

# Train the MLP (15 epochs, ~2 minutes on GPU)
python train_prior.py --epochs 15 --lr 1e-3
```

### Step 3: Evaluate

```bash
# Dense + Greedy-Tree baselines
python benchmark.py --mode full --provider lmstudio --skip-prepare

# PUCT-MCTS vs Baseline MCTS (on 5 test repos)
python run_puct_evaluation.py --provider lmstudio --max-queries 15

# Full pipeline: RL training + MCTS comparison
python run_full_pipeline.py --step all --provider lmstudio --timesteps 5000
```

---

## Results Summary

Evaluated on 5 held-out test repositories (75 queries) from CodeSearchNet:

| Method | MRR | LLM Calls/q | Latency |
|--------|-----|-------------|---------|
| Dense Baseline | **0.975** | 0 | ~50ms |
| Greedy-Tree (LLM) | 0.650 | 2.0 | ~800ms |
| Baseline MCTS (LLM) | 0.075 | 7.56 | 31.6s |
| **PUCT-MCTS (Ours)** | 0.018 | **6.17** | **15.9s** |

**Key finding:** PUCT-MCTS reduces LLM calls by 18% and latency by 49% vs Baseline MCTS. Dense retrieval dominates on CodeSearchNet's clean docstring queries; we expect structural search to show advantage on larger repos with naturalistic queries (see paper §7).

---

## Tests

```bash
# Run MCTS unit tests (19 tests)
python -m pytest research/mcts/ -v

# Key test files:
#   test_mcts_node.py    — UCB1 math, tree operations, PUCT scoring
#   test_mcts_search.py  — Integration tests with MockLLM
#   test_ucb1.py         — Mathematical correctness of exploration formula
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/initialize` | Load repository + PageIndex tree |
| `POST` | `/query` | Ask a question about a loaded repo |
| `GET` | `/search/{repo_id}?query=...` | Search without LLM response |
| `GET` | `/health` | Health check + provider status |
| `GET` | `/repos` | List loaded repositories |

---

## Configuration

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `LLM_PROVIDER` | `ollama`, `lmstudio` | `ollama` | LLM backend |

**LM Studio:** Load any model → Local Server → Start Server (port 1234).  
**Ollama:** `ollama serve` + `ollama pull qwen3:8b`.

---

## Tech Stack

- **Search:** MCTS with UCB1/PUCT selection
- **Prior:** PyTorch MLP (200K params, all-MiniLM-L6-v2 embeddings)
- **Parsing:** tree-sitter (Python, extensible to JS/Java/Go/Rust)
- **Backend:** FastAPI + Uvicorn
- **Frontend:** Streamlit
- **LLM:** LM Studio / Ollama (local inference, OpenAI-compatible)
- **RL:** Gymnasium + Stable Baselines 3 (sb3-contrib for MaskablePPO)
- **Evaluation:** CodeSearchNet (Python partition)

---

## Citation

```bibtex
@article{codecompass2026,
  title={Code Compass: Navigating Repository-Scale Codebases via Learned Tree Search},
  author={Anonymous},
  year={2026},
  note={Under review}
}
```

---

## License

This project is for educational and research purposes.

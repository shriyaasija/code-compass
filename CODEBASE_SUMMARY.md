# Code Compass Codebase Summary

Code Compass is an intelligent, repository-scale code retrieval and Q&A system. It maps codebase navigation as a **tree search problem** and uses **Monte Carlo Tree Search (MCTS)** backed by a **learned relevance prior** and **Reinforcement Learning (RL)** to navigate Abstract Syntax Trees (ASTs) efficiently. This significantly reduces necessary LLM inferences compared to greedy tree searches or exhaustive flat evaluations.

This document serves as a comprehensive guide to understanding the physical architecture, core modules, and data workflows of the codebase.

---

## 1. High-Level Architecture

The project splits into two major functional domains:
1.  **The Application**: A FastAPI backend and Streamlit frontend allowing end-users to query codebases in real-time using local LLMs (Ollama/LM Studio).
2.  **The Research Engine**: An experimental pipeline used to evaluate MCTS navigation strategies, train neural relevance priors, and implement RL-based dynamic AST restructuring.

**Standard Flow:**
1. **Preparation**: Target repositories are cloned and parsed via `tree-sitter`.
2. **Summarization & Embedding**: Bottom-up LLM summaries are generated for all tree nodes and embedded via `all-MiniLM-L6-v2`.
3. **Retrieval**: Natural language queries are processed via PUCT-guided MCTS, navigating the tree to locate the most relevant code function (leaf node).
4. **Generation**: The retrieved code is passed to the LLM to generate the final chat response.

---

## 2. Directory & Module Breakdown

### `backend/` - The Core Application Server

Handles parsing, API bridging, and the main retrieval pipeline.

*   **`api.py`**: The FastAPI server entry point. Configures endpoints (e.g., `/initialize`, `/query`, `/search`). It instantiates the LLM client (Ollama/LMStudio) dynamically based on the `LLM_PROVIDER` environment variable.
*   **`code_parser.py`**: Wraps the `tree-sitter` parser to convert raw repository language files (Python focus currently) into semantic nodes (classes, methods, functions) keeping line references intact.
*   **`tree_builder.py`**: Constructs the hierarchical tree matching the folder/file/class layout.
*   **`code_index.py` & `code_index2.py`**: Holds the baseline greedy tree search structure and integrates the newer MCTS tree-search drop-in replacement (`code_index2.py`).
*   **`semantic_search.py` & `pageindex_semantic_search.py`**: Semantic matching modules using Hugging Face `sentence-transformers`. Implements the dense retrieval baselines.
*   **`retrieval.py`**: Given a query, orchestrates the search mechanism and sends the context + prompt back to the local LLM.
*   **`lmstudio_client.py` & `ollama_client.py`**: OpenAI-compatible wrappers for interfacing with LM Studio or Ollama seamlessly.

### `frontend/` - User Interface

*   **`app.py`**: A Streamlit chat UI. Handles session states, API calls to the backend, and displays markdown formatted AI responses against selected local repositories.

### `research/` - The Algorithmic Engine

Contains the advanced AI components preparing the project for academic submission.

#### `research/mcts/` (Tree Search & Learned Prior)
*   **`mcts_node.py`**: Defines `MCTSNode`, a wrapper around `tree-sitter` nodes tracking visit counts ($N$), average values ($\bar{V}$), and prior scores ($P$). Implements the `ucb1_score` and `puct_score` formulas for selection.
*   **`mcts_search.py`**: Baseline MCTS running standard UCB1 and requiring expensive LLM simulations per node expansion.
*   **`puct_search.py`**: The optimized Predictor Upper Confidence Trees (PUCT) variant. Limits LLM calls to literal tree leaves, scoring internal node navigation using a pre-trained `RelevancePrior`.
*   **`relevance_prior.py`**: A ~200k parameter PyTorch MLP mapping concatenated (`query_embedding`, `node_embedding`) vectors to a probability relevance score. Includes an `online_update` method to adapt dynamically.
*   **`prior.pt`**: Resulting serialized weights for the trained MLP.
*   **`test_*.py`**: Comprehensive unit tests covering node math (UCB1 guarantees), traversal boundaries, and simulation bridging.

#### `research/rl_index/` (Reinforcement Learning Restructuring)
*   **`env.py`**: A Gymnasium-based `TreeIndexEnv` configuring the state space and defining discrete environment steps.
*   **`tree_mutations.py`**: Contains `Merge`, `Split`, and `Reparent` logic to physically restructure the AST index, clustering bloated namespaces or collapsing sparse trees.
*   **`tree_state.py`**: Encodes the tree constraints into feature vectors suitable for the RL agent.
*   **`reward.py`**: Computes standard Proxy MRR (embedding distance of retrieved leaves) against tree depth penalties for the agent.
*   **`train_ppo.py` & `offline_trainer.py`**: Employs `sb3-contrib` MaskablePPO to train an offline agent on the mutations across parsed datasets.

### Root Scripts - Orchestration & Pipelines

*   **`benchmark.py`**: The multi-mode evaluation suite tracking metrics (Recall@K, MRR, latency, LLM calls) across Flat Dense, Greedy Tree, and MCTS strategies. Tests against the `CodeSearchNet` dataset.
*   **`run_puct_evaluation.py`**: An isolated script to test `puct_search.py` (with the MLP) tightly against baseline MCTS.
*   **`run_full_pipeline.py`**: The mega-orchestrator. Allows `python run_full_pipeline.py --step all` to automatically: parse repos, summarize via LLM, embed, train RL, run MCTS eval, and format MD reports.
*   **`prepare_proper_trees.py`**: Pre-processing module to download from CodeSearchNet, apply `tree-sitter`, generate hierarchical summaries bottom-up, and cache the dense vectors in `/proper_trees`.
*   **`build_prior_training_data.py` / `train_prior.py`**: Fetches correct paths from the parsed trees to build the positive/negative classification samples for the `RelevancePrior`, then trains the `prior.pt` artifact.

---

## 3. Data Flow & Evaluation Caches

The `/benchmark_results/` directory stores massive generated datasets:
*   `cloned_repos/`: Ephemeral Github clones from `CodeSearchNet`.
*   `proper_trees/`: Serialized JSON files corresponding to one repository perfectly summarized and embedded up the tree.
*   `optimized_trees/`: JSON trees that have passed through the RL agent mutations (`Merge`, `Split`).
*   `prior_training_data.pt`: Cached torch dataset tracking the positive/negative nodes to ground-truth queries.
*   `eval_results/`: Direct JSON dumps detailing the specific LLM invocations and latency per test query.

---

## 4. Notable Design Patterns & Mechanics

*   **Online Prior Adaptation (`puct_search.py`)**: After a query terminates, the normalized node visitation counts are used as soft-labels. A single stochastic gradient descent step tunes the prior MLP. No retraining pipeline is invoked, enabling few-shot structural alignment specific to the developer session.
*   **Maskable PPO (`train_ppo.py`)**: AST mutation operations are restricted (e.g., cannot split a node with < 4 children). Maskable PPO is used instead of standard PPO to provide zero probabilities to invalid mutation actions in the discrete action space.
*   **Cost Efficiency Architecture**: The fundamental constraint determining the codebase architecture is the ratio of LLM latency compared to dense embedding retrieval. Tree extraction and graph navigation strictly defer LLM usage to verification roles.

---

## 5. Entry Points

*   **To run the Dev Chat**: `LLM_PROVIDER=lmstudio python -m backend.api` & `streamlit run frontend/app.py`.
*   **To run Neural Training**: `python train_prior.py`
*   **To run Benchmark Evals**: `python run_puct_evaluation.py`
*   **To run End-to-End Pipeline**: `python run_full_pipeline.py --step all`

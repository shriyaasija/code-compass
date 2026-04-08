# Code Compass → NeurIPS/ICML/ICLR Research Paper: Exhaustive Implementation Plan

> **Paper Title (Working):** *"Adaptive Tree Search with Learned Index Restructuring for Repository-Level Code Retrieval"*

---

## 1. Current Codebase Audit & Findings

### 1.1 What We Have (Assets)

| Component | File | Lines | What It Does | Research Value |
|-----------|------|-------|-------------|----------------|
| **Tree Search Engine** | [code_index.py](file:///home/shriya/code-compass/backend/code_index.py) | 349 | Greedy LLM-scored traversal with threshold pruning | **Core prototype** — needs MCTS formalization |
| **Dense Baseline** | [pageindex_semantic_search.py](file:///home/shriya/code-compass/backend/pageindex_semantic_search.py) | 245 | Flat cosine similarity over summary embeddings | **Baseline B1** — already implemented |
| **AST Parser** | [code_parser.py](file:///home/shriya/code-compass/backend/code_parser.py) | 398 | Multi-language tree-sitter parsing (9 languages) | **Strong asset** — real AST, not heuristic |
| **Tree Builder** | [tree_builder.py](file:///home/shriya/code-compass/backend/tree_builder.py) | 157 | Directory → hierarchical CodeNode tree | Foundation for index structure |
| **Benchmark Framework** | [benchmark.py](file:///home/shriya/code-compass/benchmark.py) | 995 | CodeSearchNet data prep, metrics (R@1/5/10, MRR, NDCG@10) | **Strong asset** — needs expansion |
| **LLM Clients** | [ollama_client.py](file:///home/shriya/code-compass/backend/ollama_client.py) | 282 | Ollama chat/generate with retry logic | Simulation oracle for MCTS |
| **Embedding Generator** | [generate_real_embeddings.py](file:///home/shriya/code-compass/generate_real_embeddings.py) | 324 | SentenceTransformer embedding pipeline | Embedding infrastructure |

### 1.2 Existing Benchmark Results (Dense Baseline Only — 3 repos)

```
Recall@1:  0.8889
Recall@5:  1.0000
Recall@10: 1.0000
MRR:       0.9341
NDCG@10:   0.9507
```

> [!WARNING]
> These results are on only 3 repos with 15 queries each (45 total queries). This is **far too small** for a top conference. We need 25+ repos, 375+ queries, and statistical significance tests.

### 1.3 Critical Gaps for Publication

| Gap | Severity | Resolution |
|-----|----------|------------|
| No MCTS — current search is greedy threshold | 🔴 Critical | Phase 2: Full MCTS with UCB1 |
| No RL adaptive indexing | 🔴 Critical | Phase 3: Gymnasium + PPO |
| Only 3 repos benchmarked | 🔴 Critical | Phase 5: Scale to 25+ repos |
| No baselines (BM25, CodeBERT, UniXcoder) | 🔴 Critical | Phase 4: Implement 5 baselines |
| No ablation studies | 🔴 Critical | Phase 6: Systematic ablations |
| No statistical significance tests | 🟡 High | Phase 6: Paired t-tests, bootstrap CI |
| No latency/efficiency analysis | 🟡 High | Phase 5: Wall-clock + LLM call tracking |
| No cross-language evaluation | 🟡 Medium | Phase 5: JS/TS/Java/Go repos |
| No LaTeX paper | 🟡 Medium | Phase 7: Full NeurIPS template |
| No reproducibility package | 🟡 Medium | Phase 8: Docker + scripts |

---

## 2. Competitive Landscape & How We Beat SOTA

### 2.1 Current SOTA Systems We Must Beat

| System | Approach | Strengths | Weaknesses (Our Advantage) |
|--------|----------|-----------|---------------------------|
| **Ranger** (2024) | MCTS over dense knowledge graphs | Strong retrieval + generation | Requires expensive KG construction; static graph |
| **RepoHyper** (2024) | Hypergraph + GNN for repo-level completion | Captures cross-file deps | Heavy GNN training; no adaptive indexing |
| **CodeBERT/UniXcoder** | Bi-encoder dense retrieval | Fast inference | No structure awareness; flat ranking |
| **GraphCodeBERT** | Data-flow graph + pre-training | Code structure in embeddings | Pre-training cost; no online adaptation |
| **BM25 + Reranker** | Sparse retrieval + cross-encoder | Simple, strong baseline | No hierarchy; no semantic tree |

### 2.2 Our Three-Pillar Novelty (Why Reviewers Will Accept)

> [!IMPORTANT]
> **Pillar 1: MCTS over Strict AST Hierarchies**
> Unlike Ranger's dense knowledge graphs, we use lightweight, parser-derived AST trees. This is fundamentally different: no graph construction cost, no embedding-based edges — just tree-sitter output + MCTS. We prove that structure alone (without learned graph weights) is sufficient for competitive retrieval.

> [!IMPORTANT]
> **Pillar 2: RL-Driven Adaptive Index Restructuring**
> This is the *key technical novelty*. No prior code retrieval system learns to reshape its index structure online. Borrowing from "The Case for Learned Index Structures" (Kraska et al., SIGMOD 2018), we train a PPO agent that performs `Merge`, `Split`, and `Reparent` operations on the AST tree to minimize future retrieval depth. This creates a self-optimizing retrieval system.

> [!IMPORTANT]
> **Pillar 3: Hybrid Retrieval with Semantic Gating**
> We combine dense embedding pre-filtering with LLM-guided MCTS traversal. A learned gating mechanism decides when to use cheap dense retrieval vs. expensive but accurate MCTS, creating an efficiency-accuracy Pareto frontier.

### 2.3 Target Metrics to Beat SOTA

| Metric | Current Dense Baseline | Target (MCTS) | Target (MCTS+RL) | SOTA Reference |
|--------|----------------------|---------------|-------------------|----------------|
| R@1 | 0.889 | >0.90 | >0.93 | ~0.85-0.90 |
| R@5 | 1.000 | >0.95 | >0.97 | ~0.90-0.95 |
| MRR | 0.934 | >0.94 | >0.96 | ~0.88-0.93 |
| LLM calls/query | 0 (dense) | <5 avg | <3 avg (after RL) | N/A |
| Latency (ms) | ~50ms | <2000ms | <1500ms | Varies |

---

## 3. Mathematical Formulation (Paper-Ready)

### 3.1 Problem Definition

Given a code repository $\mathcal{R}$ parsed into an AST-derived tree $\mathcal{T} = (V, E)$ where $V$ are nodes (repos, folders, files, classes, functions) and $E$ are parent-child edges, and a natural language query $q$, find the set of leaf nodes $L^* \subseteq \text{Leaves}(\mathcal{T})$ most relevant to $q$.

### 3.2 MCTS Formulation

**State:** $s_t = (n_t, q, H_t)$ where $n_t \in V$ is the current tree node, $q$ is the query, and $H_t$ is the traversal history.

**Action:** $a_t \in \text{Children}(n_t)$ — select a child to explore.

**Selection (UCB1):**
$$a^* = \arg\max_{a \in \text{Children}(n_t)} \left[ \bar{V}(n_t, a) + c \sqrt{\frac{\ln N(n_t)}{N(n_t, a)}} \right]$$

where $\bar{V}$ is average value, $N$ is visit count, and $c$ is the exploration constant.

**Simulation (LLM Rollout):** An SLM scores the relevance of unexplored subtrees:
$$r(n, q) = \text{LLM}_\theta\left(\text{prompt}(n.\text{summary}, q)\right) \in [0, 1]$$

**Backpropagation:** Update value estimates along the path:
$$\bar{V}(n, a) \leftarrow \bar{V}(n, a) + \frac{r - \bar{V}(n, a)}{N(n, a) + 1}$$

### 3.3 RL Index Restructuring (PPO MDP)

**State:** $\mathbf{s} = [\text{depth}_{\max}, \sigma^2_{\text{branch}}, \text{skew}_{\text{hit}}, \bar{\cos}_{\text{sib}}, |\mathcal{T}|]$

**Actions:** $\mathcal{A} = \{\texttt{Merge}(A,B), \texttt{Split}(A,K), \texttt{Reparent}(A, T), \texttt{NoOp}\}$

**Reward:**
$$R_t = \lambda_1 \left(\text{MRR}_t - \text{MRR}_{t-1}\right) - \lambda_2 \cdot \text{AvgDepth}_t - \lambda_3 \cdot \mathbb{1}[\text{invalid\_mutation}]$$

**Policy:** $\pi_\phi(a|s)$ parameterized by MLP, trained with PPO (clip ratio $\epsilon = 0.2$).

---

## 4. Phase 1: Environment & Infrastructure Setup

### 4.1 Git Branch & Environment

```bash
# Step 1: Create research branch
cd /home/shriya/code-compass
git checkout main && git pull
git checkout -b research/neurips-submission

# Step 2: Create conda environment
conda create -n compass-paper python=3.10 -y
conda activate compass-paper

# Step 3: Install base + research dependencies
pip install -r requirements.txt
pip install -r requirements2.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install stable-baselines3[extra] gymnasium
pip install sentence-transformers datasets wandb
pip install networkx matplotlib seaborn pandas scipy
pip install rank_bm25  # BM25 baseline
pip install scikit-learn  # K-means for Split action

# Step 4: Initialize W&B
wandb login
```

### 4.2 Directory Structure (Every Single File)

```
code-compass/
├── research/
│   ├── __init__.py
│   ├── mcts/
│   │   ├── __init__.py
│   │   ├── mcts_node.py          # MCTSNode class with UCB1
│   │   ├── mcts_search.py        # Full MCTS search algorithm
│   │   ├── simulation.py         # LLM simulation/rollout
│   │   ├── test_mcts_node.py     # Unit tests for MCTSNode
│   │   ├── test_mcts_search.py   # Integration tests for search
│   │   └── test_ucb1.py          # Mathematical correctness of UCB1
│   ├── rl_index/
│   │   ├── __init__.py
│   │   ├── tree_mutations.py     # Merge, Split, Reparent operations
│   │   ├── tree_state.py         # State feature extraction
│   │   ├── env.py                # Gymnasium environment
│   │   ├── reward.py             # Reward computation
│   │   ├── train_ppo.py          # PPO training loop
│   │   ├── test_mutations.py     # Invariant tests for mutations
│   │   ├── test_env.py           # Environment step/reset tests
│   │   └── test_reward.py        # Reward signal tests
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── bm25_baseline.py      # BM25 sparse retrieval
│   │   ├── dense_baseline.py     # Bi-encoder (MiniLM, CodeBERT)
│   │   ├── greedy_baseline.py    # Current greedy tree search
│   │   ├── random_baseline.py    # Random walk baseline
│   │   └── oracle_baseline.py    # Perfect retrieval upper bound
│   ├── experiments/
│   │   ├── configs/
│   │   │   ├── baselines.yaml
│   │   │   ├── mcts_sweep.yaml   # c_explore, max_iterations sweep
│   │   │   ├── rl_training.yaml
│   │   │   └── ablation.yaml
│   │   ├── run_baselines.py
│   │   ├── run_mcts.py
│   │   ├── run_rl_training.py
│   │   ├── run_ablations.py
│   │   ├── run_significance.py   # Statistical tests
│   │   └── results/              # Output JSONs
│   ├── analysis/
│   │   ├── plot_main_results.py  # Table 1 + bar charts
│   │   ├── plot_rl_convergence.py # RL training curves
│   │   ├── plot_ablations.py     # Ablation bar charts
│   │   ├── plot_efficiency.py    # Latency vs accuracy
│   │   ├── plot_tree_evolution.py # Tree structure before/after RL
│   │   └── generate_latex_tables.py
│   └── paper/
│       ├── main.tex
│       ├── neurips_2025.sty
│       ├── references.bib
│       └── figures/              # PDF figures
```

### 4.3 Configuration Management

**File: `research/experiments/configs/baselines.yaml`**
```yaml
experiment:
  name: "baseline_comparison"
  seed: 42
  num_repos: 25
  queries_per_repo: 15
  embedding_model: "all-MiniLM-L6-v2"
  output_dir: "research/experiments/results"

baselines:
  - name: "BM25"
    type: "sparse"
  - name: "Dense-MiniLM"
    type: "dense"
    model: "all-MiniLM-L6-v2"
  - name: "Dense-CodeBERT"
    type: "dense"
    model: "microsoft/codebert-base"
  - name: "Greedy-Tree"
    type: "tree_greedy"
    threshold: 0.5
    llm_model: "qwen2.5:7b"
  - name: "Random-Walk"
    type: "random"
```

---

## 5. Phase 2: MCTS Implementation (The Search Algorithm)

### 5.1 MCTSNode Class

**File: `research/mcts/mcts_node.py`**

```python
"""
MCTSNode: A single node in the MCTS search tree.

Each MCTSNode wraps a tree node from the code AST and tracks:
- Visit count N(s,a)
- Total value Q(s,a)  
- Children (lazily expanded)
- Parent pointer for backpropagation

Mathematical guarantees:
- UCB1 converges to optimal action as N → ∞ (Auer et al., 2002)
- Regret bound: O(√(K ln N)) where K = branching factor
"""

import math
from typing import List, Dict, Optional, Any

class MCTSNode:
    def __init__(self, tree_node: Dict, parent: Optional['MCTSNode'] = None):
        self.tree_node = tree_node  # The actual AST/code tree node
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visit_count: int = 0  # N(s)
        self.total_value: float = 0.0  # Sum of all rollout values
        self.is_expanded: bool = False
        self.is_terminal: bool = self._check_terminal()
    
    @property
    def average_value(self) -> float:
        """V̄(s) = Q(s) / N(s)"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    def ucb1_score(self, c_explore: float = 1.414) -> float:
        """
        UCB1(s,a) = V̄(s,a) + c * √(ln N(parent) / N(s,a))
        
        Args:
            c_explore: Exploration constant. √2 ≈ 1.414 is theoretically optimal
                       for rewards in [0,1] (Auer et al., 2002).
                       
        Returns:
            UCB1 score. Returns infinity for unvisited nodes (optimistic init).
        """
        if self.visit_count == 0:
            return float('inf')  # Optimistic initialization
        
        if self.parent is None or self.parent.visit_count == 0:
            return self.average_value
        
        exploitation = self.average_value
        exploration = c_explore * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        return exploitation + exploration
    
    def _check_terminal(self) -> bool:
        """A node is terminal if it's a leaf in the code tree."""
        node_type = self.tree_node.get('type', self.tree_node.get('node_type', ''))
        has_children = bool(self.tree_node.get('nodes', self.tree_node.get('children', [])))
        
        # Terminal = function/method/class with line numbers OR file without children
        if node_type in ['function', 'method', 'class'] and 'start_line' in self.tree_node:
            return True
        if node_type.startswith('file') and not has_children:
            return True
        return False
    
    def expand(self) -> List['MCTSNode']:
        """Expand this node by creating MCTSNode children from tree children."""
        if self.is_expanded or self.is_terminal:
            return self.children
        
        tree_children = self.tree_node.get('nodes', self.tree_node.get('children', []))
        self.children = [MCTSNode(child, parent=self) for child in tree_children]
        self.is_expanded = True
        return self.children
    
    def backpropagate(self, value: float):
        """Backpropagate value up to root."""
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent
    
    def best_child(self, c_explore: float = 1.414) -> Optional['MCTSNode']:
        """Select child with highest UCB1 score."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb1_score(c_explore))
    
    def most_visited_child(self) -> Optional['MCTSNode']:
        """Select child with highest visit count (for final selection)."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visit_count)
```

### 5.2 Full MCTS Search Algorithm

**File: `research/mcts/mcts_search.py`**

Core algorithm implementing all 4 phases: Selection → Expansion → Simulation → Backpropagation.

Key design decisions:
- **Max iterations**: 50 per query (balances accuracy vs. LLM cost)
- **Max depth**: Tree depth (typically 3-5 for code repos)
- **Top-K extraction**: Collect all leaves with visit_count > threshold
- **Early termination**: Stop if top leaf has >90% of visits (convergence)

### 5.3 LLM Simulation Module

**File: `research/mcts/simulation.py`**

The simulation phase uses the LLM as a discriminator:
- Batch-scores all children of a node in ONE LLM call (inherits from current `_score_siblings`)
- Returns normalized scores ∈ [0,1]
- Caches results to avoid redundant LLM calls
- Supports both Ollama and LM Studio backends

### 5.4 Integration with Existing Code

**Modify: [code_index.py](file:///home/shriya/code-compass/backend/code_index.py)**

- Add `MCTSSearch` class alongside existing `TreeBasedSearch`
- Both share the same `load_repository_tree()` and `_score_siblings()` interfaces
- `MCTSSearch.search()` replaces `_recursive_search()` with MCTS
- Keep `TreeBasedSearch` intact as "Greedy-Tree" baseline

### 5.5 Unit Tests (Phase 2)

| Test File | Tests | What It Validates |
|-----------|-------|-------------------|
| `test_mcts_node.py` | 8 tests | UCB1 math, backprop, expansion, terminal detection |
| `test_ucb1.py` | 5 tests | UCB1 converges; unvisited = ∞; exploration decreases with visits |
| `test_mcts_search.py` | 6 tests | End-to-end search on mock tree; correct leaf extraction; early termination |

---

*Continued in Part 2 (Phases 3-8)...*

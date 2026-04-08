# Updated Competitive Landscape & Baselines

> This document **replaces Section 2 and Phase 4** in the original implementation plan.

---

## 1. Seminal Works We MUST Compare Against

### Tier 1: Foundational Code Encoders (Pre-trained Bi-Encoders)

These are the bedrock baselines in every code search paper. Omitting any of them = instant reviewer rejection.

| System | Paper | Venue | Approach | CodeSearchNet Python MRR | How to Compare |
|--------|-------|-------|----------|------------------------|----------------|
| **CodeBERT** | Feng et al. 2020 | EMNLP 2020 | Bimodal pre-training on NL-PL pairs, bi-encoder retrieval | ~0.882 | Use `microsoft/codebert-base` from HuggingFace; encode query + code summaries, cosine similarity |
| **GraphCodeBERT** | Guo et al. 2021 | ICLR 2021 | Adds data-flow graph edges to pre-training; structure-aware embeddings | ~0.897 | Use `microsoft/graphcodebert-base`; same bi-encoder protocol |
| **UniXcoder** | Guo et al. 2022 | ACL 2022 | Unified cross-modal pre-training (AST + comments + code); SOTA on code search | ~0.943 | Use `microsoft/unixcoder-base`; this is the strongest bi-encoder baseline |
| **CodeT5+** | Wang et al. 2023 | EMNLP 2023 | Encoder-decoder with flexible task heads; instruction-tuned | ~0.950+ | Use `Salesforce/codet5p-110m-embedding`; currently at/near top of CodeSearchNet |

> [!IMPORTANT]
> **UniXcoder and CodeT5+** are the baselines reviewers will immediately check. If we don't beat them on at least one axis (accuracy, latency, or structure-awareness), the paper will be rejected.

**Implementation (File: `research/baselines/pretrained_encoders.py`):**
```python
"""
Baseline B1-B4: Pre-trained code encoder bi-encoders.

For each model:
1. Load from HuggingFace
2. Encode all function summaries/docstrings in the repo
3. Encode query
4. Rank by cosine similarity
5. Compute R@1, R@5, R@10, MRR, NDCG@10

These are FLAT retrievers — no tree structure, no hierarchy.
This is the key weakness we exploit: they treat a 10,000-function repo
the same as a 10-function repo.
"""
```

### Tier 2: Hierarchical / Tree-Structured Retrieval

These are the most directly comparable systems — they also use tree structures for retrieval.

| System | Paper | Venue | Approach | Key Result | Our Advantage |
|--------|-------|-------|----------|-----------|---------------|
| **RAPTOR** | Sarthi et al. 2024 | ICLR 2024 | Bottom-up tree: cluster text chunks → summarize → embed at each level. Retrieves from multiple abstraction layers simultaneously | 20% absolute accuracy improvement on QuALITY over flat retrieval | RAPTOR builds trees from **text chunks** via clustering. We use **real AST structure** from tree-sitter. Our trees are syntactically grounded, not statistically constructed. Also: RAPTOR is static; our RL agent adapts the tree. |
| **Greedy-Tree (Ours v0)** | Current code_index.py | This work | LLM scores siblings, threshold-based pruning, greedy top-down traversal | R@1=0.889, MRR=0.934 (3 repos) | This is our ablation baseline — shows improvement from MCTS + RL |

> [!IMPORTANT]
> **RAPTOR is the most important hierarchical baseline.** It's from Stanford (Manning's group), published at ICLR 2024, and is the current gold standard for tree-structured retrieval. We MUST implement and compare against it.

**Implementation (File: `research/baselines/raptor_baseline.py`):**
```python
"""
Baseline B5: RAPTOR-style tree retrieval adapted for code.

Implementation:
1. Take all function summaries/docstrings as leaf text chunks
2. Embed each chunk with SentenceTransformer
3. Cluster using GMM (as in RAPTOR paper) — form tree levels
4. Summarize each cluster using LLM (or concatenate summaries)
5. Embed cluster summaries → these become internal tree nodes
6. Repeat recursively until single root
7. At query time: traverse tree from root, retrieve from all levels
   (RAPTOR uses "collapsed tree" retrieval — searching all nodes flat)

This gives RAPTOR the benefit of hierarchical summarization but
WITHOUT real code structure (AST). This is our key differentiator.
"""
```

### Tier 3: Repository-Level Code Generation/Completion Systems

These systems tackle the same problem domain (repo-level understanding) but from the code completion angle. We compare retrieval components.

| System | Paper | Venue | Approach | Key Innovation | How to Compare |
|--------|-------|-------|----------|---------------|----------------|
| **RepoCoder** | Zhang et al. 2023 | EMNLP 2023 | Iterative retrieval-generation: retrieve → generate → use generation as new query → retrieve again | Iterative refinement of context | Implement single-iteration retrieval as baseline; compare retrieval recall |
| **RepoHyper** | Phan et al. 2024 | arXiv 2024 | Builds a Repo-level Semantic Hypergraph (RSG) capturing file imports, function calls, class inheritance → GNN-based retrieval | Hypergraph captures cross-file dependencies | Simulate by building call-graph edges from our AST; compare with/without cross-file info |
| **CoCoMIC** | Ding et al. 2023 | NeurIPS 2023 | Joint modeling of in-file and cross-file context via two-stage retrieval: BM25 for cross-file, then neural reranking | Separates in-file vs cross-file retrieval | Use their two-stage approach as baseline |

**Implementation (File: `research/baselines/repo_level_baselines.py`):**
```python
"""
Baselines B6-B8: Repository-level systems.

B6 - RepoCoder (single iteration):
  1. BM25 retrieval over all function signatures + summaries
  2. Top-20 results fed to LLM as context
  3. Evaluate which ground-truth functions appear in retrieval

B7 - RepoHyper (simplified):
  1. Build import graph from AST (which files import which)
  2. Build call graph (which functions call which)
  3. For query, retrieve by embedding similarity + graph proximity
  4. GNN not tractable to reimplement; use graph-walk heuristic instead
  Note: In paper, clearly state this is a "RepoHyper-inspired" baseline

B8 - CoCoMIC (two-stage):
  1. Stage 1: BM25 retrieval of cross-file candidates
  2. Stage 2: Neural reranking with UniXcoder
  3. Compare retrieval quality before/after reranking
"""
```

### Tier 4: Search-Based / MCTS Methods for Code

These use tree search or MCTS specifically for code tasks — the most directly competing approaches.

| System | Paper | Venue | Approach | Our Advantage |
|--------|-------|-------|----------|---------------|
| **Ranger** | Cao et al. 2024 | arXiv 2024 | MCTS over a dense **Code Knowledge Graph** (CKG) with function-call edges, import edges, type edges. Uses LLM as policy for tree search expansion | Ranger's CKG requires expensive static analysis + embedding of all edges. Our approach uses **raw AST hierarchy** (zero-cost to construct from tree-sitter). We also add **RL adaptive restructuring** which Ranger lacks entirely. |
| **R2C2-Coder** | arXiv 2024 | arXiv 2024 | Repository-level code completion via retrieve-then-rerank-then-complete pipeline with contextual retrieval | Multi-stage pipeline; our single-pass MCTS is more efficient |
| **SWE-Agent** | Yang et al. 2024 | arXiv 2024 | Agent that uses file browsing + search tools to navigate repos for bug fixing | Uses LLM agent for navigation (expensive); we use principled MCTS |

**Implementation (File: `research/baselines/search_baselines.py`):**
```python
"""
Baselines B9-B10: Search-based methods.

B9 - Ranger-Lite (our reimplementation):
  CRITICAL: We cannot fully reimplement Ranger's CKG (it requires 
  static analysis tooling they haven't open-sourced). Instead:
  1. Build a function call graph from AST (who-calls-whom)
  2. Build import graph (file-level dependencies)
  3. Combine into a "Code Knowledge Graph"
  4. Run MCTS over this graph (same UCB1 as ours)
  5. Use same LLM simulation function as our method

  This gives Ranger its core advantage (graph structure) while using
  our MCTS implementation. Fair comparison: same search algorithm,
  different index structure (graph vs tree).

B10 - Agent-Walk (SWE-Agent inspired):
  1. Start at repository root
  2. At each step, LLM chooses: list_files, open_file, search_text, done
  3. Count total LLM calls to find target function
  4. Compare efficiency (LLM calls) and accuracy (did it find it?)
"""
```

### Tier 5: Classic IR Baselines + Bounds

| System | Approach | Purpose |
|--------|----------|---------|
| **BM25** | Sparse TF-IDF retrieval on function signatures + docstrings | Lower bound for neural methods |
| **Random Walk** | Random child selection at each tree level | Lower bound for tree traversal |
| **Oracle** | Perfect retrieval (ground truth always rank 1) | Upper bound |

---

## 2. Complete Baseline Table (12 Systems)

| ID | System | Type | Structure | Requires LLM | Published Venue |
|----|--------|------|-----------|---------------|-----------------|
| B1 | BM25 | Sparse | Flat | No | Classic IR |
| B2 | CodeBERT | Dense Bi-encoder | Flat | No | EMNLP 2020 |
| B3 | GraphCodeBERT | Dense + Data Flow | Flat | No | ICLR 2021 |
| B4 | UniXcoder | Dense Cross-modal | Flat | No | ACL 2022 |
| B5 | CodeT5+ | Dense Encoder-Decoder | Flat | No | EMNLP 2023 |
| B6 | RAPTOR | Hierarchical Clustering | Clustered Tree | Yes (summaries) | ICLR 2024 |
| B7 | RepoCoder (1-iter) | Iterative BM25 | Flat → Refined | No | EMNLP 2023 |
| B8 | RepoHyper-Lite | Hypergraph + Retrieval | Graph | No | arXiv 2024 |
| B9 | CoCoMIC (2-stage) | BM25 + Neural Rerank | Flat → Reranked | No | NeurIPS 2023 |
| B10 | Ranger-Lite | MCTS over Knowledge Graph | Dense Graph | Yes | arXiv 2024 |
| B11 | Random Walk | Random tree traversal | Tree | No | N/A |
| B12 | Oracle | Ground truth | N/A | No | N/A |

### Our Methods (3 variants):

| ID | Method | Description |
|----|--------|-------------|
| **M1** | MCTS-AST | MCTS with UCB1 over tree-sitter AST hierarchy |
| **M2** | MCTS-AST + RL | MCTS on RL-restructured tree |
| **M3** | Hybrid (Dense gate + MCTS) | Learned gating between dense retrieval and MCTS |

---

## 3. Why We Beat Each System (Reviewer Argument Map)

| Baseline | Their Weakness | Our Exploit |
|----------|---------------|-------------|
| CodeBERT/UniXcoder | Flat retrieval — no structure awareness. Treats 10,000-node repo like a bag of functions | Our tree structure reduces search space from O(N) to O(log N) with provable bounds |
| GraphCodeBERT | Only uses data-flow within a function, not cross-file structure | Our repo-level AST captures file → class → function hierarchy |
| RAPTOR | Trees built by statistical clustering — no syntactic grounding. Tree is static once built | Our trees are **syntactically grounded** (tree-sitter) AND **dynamically adapted** (RL) |
| Ranger | Requires expensive knowledge graph construction; graph is static | We prove that **strict hierarchies** (cheaper to build) + **RL adaptation** (dynamic) achieve comparable or better results |
| RepoCoder | Iterative retrieval adds latency; no structure | Single-pass MCTS with structured index is faster |
| RepoHyper | Heavy GNN training required; hypergraph construction is expensive | Lightweight tree-sitter + PPO is orders of magnitude cheaper |

---

## 4. Implementation Priority & Effort Estimates

| Priority | Baseline | Effort | Notes |
|----------|----------|--------|-------|
| 🔴 P0 | BM25 | 2 hrs | `rank_bm25` library, trivial |
| 🔴 P0 | UniXcoder | 3 hrs | HuggingFace model, bi-encoder protocol |
| 🔴 P0 | RAPTOR | 8 hrs | Must build bottom-up clustering tree + LLM summaries |
| 🔴 P0 | Ranger-Lite | 10 hrs | Build call/import graph, run MCTS over it |
| 🟡 P1 | CodeBERT | 2 hrs | Same as UniXcoder, different model |
| 🟡 P1 | GraphCodeBERT | 3 hrs | Same protocol |
| 🟡 P1 | CodeT5+ | 3 hrs | Salesforce model, embedding API |
| 🟡 P1 | RepoCoder (1-iter) | 4 hrs | BM25 + LLM context stuffing |
| 🟢 P2 | RepoHyper-Lite | 6 hrs | Graph construction from imports/calls |
| 🟢 P2 | CoCoMIC (2-stage) | 5 hrs | BM25 + neural reranker |
| ⚪ Auto | Random Walk | 1 hr | Trivial random selection |
| ⚪ Auto | Oracle | 0.5 hr | Return ground truth |

**Total new baseline code: ~48 hours of implementation**

---

## 5. Updated Related Work Section for Paper

```latex
\section{Related Work}

\paragraph{Code Search with Pre-trained Models.}
Pre-trained code models have driven the state of the art in natural language 
code search. CodeBERT~\cite{feng2020codebert} introduced bimodal pre-training 
on NL-PL pairs. GraphCodeBERT~\cite{guo2021graphcodebert} incorporated 
data-flow edges into pre-training. UniXcoder~\cite{guo2022unixcoder} unified 
AST, comments, and code tokens in a cross-modal framework, achieving strong 
results on CodeSearchNet. CodeT5+~\cite{wang2023codet5p} further improved 
with instruction-tuned encoder-decoder architectures. However, all these 
methods perform \emph{flat retrieval}: they embed entire functions and 
rank by similarity, ignoring the hierarchical structure of repositories.

\paragraph{Hierarchical and Tree-Structured Retrieval.}
RAPTOR~\cite{sarthi2024raptor} constructs retrieval trees by recursively 
clustering and summarizing text chunks, achieving state-of-the-art on 
multi-step QA. However, RAPTOR's trees are \emph{statistically constructed} 
via GMM clustering, not grounded in the syntactic structure of the content. 
Our approach builds trees directly from AST parse results, ensuring that 
the hierarchy reflects actual code organization.

\paragraph{Repository-Level Code Understanding.}
RepoCoder~\cite{zhang2023repocoder} uses iterative retrieval-generation 
cycles for cross-file context. RepoHyper~\cite{phan2024repohyper} 
constructs semantic hypergraphs. CoCoMIC~\cite{ding2023cocomic} jointly 
models in-file and cross-file context. These systems build rich contextual 
representations but require expensive graph construction or multi-stage 
pipelines.

\paragraph{MCTS for Code.}
Most closely related to our work, Ranger~\cite{cao2024ranger} applies 
MCTS over a Code Knowledge Graph for repository-level code generation. 
While effective, Ranger requires construction of a dense knowledge graph 
with function-call, import, and type edges --- a costly preprocessing step. 
We demonstrate that MCTS over \emph{strict AST hierarchies} (derived 
at near-zero cost from tree-sitter) achieves competitive or superior 
retrieval performance. Furthermore, our RL-based index restructuring 
--- inspired by learned index structures~\cite{kraska2018case} --- is 
entirely absent from Ranger and all prior code retrieval systems.
```

---

## 6. What Exactly Changes in the Plan

> [!IMPORTANT]
> **Phase 4 expands from 5 baselines → 12 baselines.** The four critical additions are:
> 1. **RAPTOR** (ICLR 2024) — hierarchical clustering retrieval
> 2. **Ranger-Lite** — MCTS over knowledge graph (our reimplementation)
> 3. **UniXcoder** (ACL 2022) — strongest bi-encoder baseline  
> 4. **CodeT5+** (EMNLP 2023) — current CodeSearchNet SOTA

> [!WARNING]
> **Timeline impact:** Baseline implementation increases from ~1 week to ~2 weeks. RAPTOR and Ranger-Lite are complex. Recommend starting these in parallel with MCTS development.

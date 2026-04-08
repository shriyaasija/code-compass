# Implementation Plan — Part 2: Phases 3–8, Risk Mitigation & Timeline

> Continues from Part 1. Read [implementation_plan.md](file:///home/shriya/.gemini/antigravity/brain/d61b8c5d-333f-41f8-a6f5-dfb297b33866/implementation_plan.md) first.

---

## 6. Phase 3: RL Adaptive Index Restructuring

### 6.1 Tree Mutation Operators

**File: `research/rl_index/tree_mutations.py`**

Three graph-mutating operations that preserve tree invariants:

#### 6.1.1 `Merge(A, B)` — Combine two sibling nodes
- **Precondition:** A and B share the same parent; their cosine similarity > 0.7
- **Operation:** Create new node C with children = Children(A) ∪ Children(B); replace A,B with C under parent
- **Invariant checks:** No orphans, no cycles, total leaf count unchanged
- **Example:** Two utility files `string_utils.py` and `text_helpers.py` → merged `text_utils/` folder

#### 6.1.2 `Split(A, K)` — Split a bloated node into K clusters
- **Precondition:** |Children(A)| > 10 (node is too broad)
- **Operation:** Run K-means (K=2 or 3) on children's embeddings; create K new intermediate nodes
- **Invariant checks:** All original children preserved; K new internal nodes added
- **Example:** A folder with 30 functions → split into 3 semantic groups of ~10

#### 6.1.3 `Reparent(A, Target)` — Move node A under a new parent
- **Precondition:** A is frequently co-queried with Target's subtree (hit-rate correlation > 0.5)
- **Operation:** Detach A from current parent; attach as child of Target
- **Invariant checks:** No cycles (check Target is not a descendant of A); tree remains connected
- **Example:** A utility function used only by one module gets moved into that module's subtree

### 6.2 State Feature Extraction

**File: `research/rl_index/tree_state.py`**

```python
def extract_state(tree: Dict, query_buffer: List[Dict]) -> np.ndarray:
    """
    Extract 8-dimensional state vector from current tree + recent queries.
    
    Features:
    [0] max_depth: Maximum tree depth (normalized by initial depth)
    [1] avg_branching: Average branching factor
    [2] branching_var: Variance of branching factor (imbalance indicator)
    [3] leaf_count: Number of leaf nodes (normalized)
    [4] hit_skew: Gini coefficient of leaf hit rates (from query buffer)
    [5] avg_sibling_sim: Average cosine similarity between siblings
    [6] avg_retrieval_depth: Average depth of retrieved leaves (from buffer)
    [7] buffer_mrr: MRR computed on the current query buffer
    """
```

### 6.3 Gymnasium Environment

**File: `research/rl_index/env.py`**

```python
class TreeIndexEnv(gymnasium.Env):
    """
    RL environment for adaptive tree index optimization.
    
    Observation space: Box(8,) — tree state features
    Action space: Discrete(4) — {Merge, Split, Reparent, NoOp}
    
    Episode structure:
    - Each episode = 50 steps (mutations)
    - After each mutation, replay last 100 queries from buffer
    - Reward = change in MRR - depth penalty
    - Done when 50 steps reached OR tree becomes invalid
    """
    
    # Key design decisions:
    # - Action masking: Actions with violated preconditions are masked
    # - Query buffer: Rolling window of 100 recent (query, ground_truth) pairs
    # - Mutation targets: Selected by heuristic (highest co-query correlation)
    # - Reset: Restores tree to original parsed state
```

### 6.4 PPO Training Loop

**File: `research/rl_index/train_ppo.py`**

```python
# Training configuration
PPO_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 128,          # Steps per rollout
    "batch_size": 64,
    "n_epochs": 10,          # PPO epochs per update
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,        # Encourage exploration early
    "total_timesteps": 50000,
    "seed": 42
}

# Logging: W&B tracks reward, MRR, tree depth, action distribution
```

### 6.5 Unit Tests (Phase 3)

| Test | What It Validates |
|------|-------------------|
| `test_merge_preserves_leaves` | Merge doesn't lose any leaf nodes |
| `test_merge_precondition` | Merge rejects non-sibling nodes |
| `test_split_clustering` | Split produces K groups with correct children |
| `test_split_small_node` | Split rejects nodes with <10 children |
| `test_reparent_no_cycles` | Reparent detects and blocks cycles |
| `test_reparent_connectivity` | Tree remains connected after reparent |
| `test_env_reset` | Environment resets to original tree |
| `test_env_step` | Step returns valid (obs, reward, done, info) |
| `test_env_action_masking` | Invalid actions are masked |
| `test_reward_positive_on_improvement` | MRR increase → positive reward |
| `test_reward_depth_penalty` | Deeper trees → lower reward |

---

## 7. Phase 4: Competitive Baselines Implementation

### 7.1 Five Baselines (Every One Implemented)

| # | Baseline | File | Method | Embedding |
|---|----------|------|--------|-----------|
| B1 | BM25 (Sparse) | `baselines/bm25_baseline.py` | `rank_bm25` on function docstrings + names | None |
| B2 | Dense-MiniLM | `baselines/dense_baseline.py` | Cosine similarity, `all-MiniLM-L6-v2` | 384-dim |
| B3 | Dense-CodeBERT | `baselines/dense_baseline.py` | Cosine similarity, `microsoft/codebert-base` | 768-dim |
| B4 | Greedy-Tree (Ours, no MCTS) | `baselines/greedy_baseline.py` | Current `_recursive_search` from `code_index.py` | LLM-scored |
| B5 | Random Walk | `baselines/random_baseline.py` | Random child selection at each level | None |
| B6 | Oracle (Upper Bound) | `baselines/oracle_baseline.py` | Perfect retrieval (ground truth) | N/A |

### 7.2 Our Methods

| # | Method | Description |
|---|--------|-------------|
| M1 | **MCTS-Search** | Full MCTS with UCB1 + LLM simulation (Phase 2 output) |
| M2 | **MCTS + RL Index** | MCTS search on RL-optimized tree (Phase 3 output) |
| M3 | **Hybrid (Dense + MCTS)** | Dense pre-filter → MCTS on top candidates |

---

## 8. Phase 5: Large-Scale Benchmarking

### 8.1 Datasets

| Dataset | Repos | Queries | Languages | Purpose |
|---------|-------|---------|-----------|---------|
| **CodeSearchNet (Python)** | 25 repos | 375 queries | Python | Primary evaluation |
| **CodeSearchNet (Multi)** | 10 repos | 150 queries | JS, Java, Go | Cross-language robustness |
| **SWE-bench-lite** | 12 repos | 60 queries | Python | Real developer queries |
| **Custom-Large** | 5 repos | 50 queries | Python | Repos with 500+ functions (stress test) |
| **Custom-Adversarial** | 5 repos | 25 queries | Mixed | Deliberately ambiguous queries |

**Total: 57 repos, 660 queries minimum**

### 8.2 Evaluation Metrics

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **Recall@K** (K=1,5,10) | $\frac{1}{|Q|} \sum_{q} \mathbb{1}[\text{gt} \in \text{top-K}]$ | Can we find the right code? |
| **MRR** | $\frac{1}{|Q|} \sum_{q} \frac{1}{\text{rank}(q)}$ | How high is the correct result? |
| **NDCG@10** | Already implemented | Ranking quality |
| **LLM Calls/Query** | Count of `_score_siblings` invocations | Computational cost |
| **Wall-Clock Latency** | `time.perf_counter()` per query | Real-world speed |
| **Tree Mutations/Episode** | Count of non-NoOp RL actions | RL agent activity |

### 8.3 Statistical Rigor

For every comparison between methods:
1. **Paired t-test** across per-repo MRR scores (p < 0.05)
2. **Bootstrap confidence intervals** (1000 resamples, 95% CI)
3. **Effect size** (Cohen's d)
4. **Multiple comparison correction** (Bonferroni or Holm-Bonferroni)

### 8.4 Benchmark Execution Plan

```bash
# Step 1: Prepare data (no LLM needed, ~30 min)
python benchmark.py --mode prepare --num-repos 25 --queries-per-repo 15 --output-dir research/experiments/results/csn_python

# Step 2: Run all baselines (parallelizable)
python research/experiments/run_baselines.py --config research/experiments/configs/baselines.yaml

# Step 3: Run MCTS (requires Ollama, ~4-8 hours for 375 queries)
python research/experiments/run_mcts.py --config research/experiments/configs/mcts_sweep.yaml

# Step 4: Train RL agent (requires previous query logs, ~2-4 hours)
python research/experiments/run_rl_training.py --timesteps 50000 --track_wandb true

# Step 5: Run MCTS on RL-optimized trees
python research/experiments/run_mcts.py --config research/experiments/configs/mcts_rl.yaml

# Step 6: Statistical significance tests
python research/experiments/run_significance.py --results_dir research/experiments/results
```

---

## 9. Phase 6: Ablation Studies (12 Experiments)

| # | Ablation | What We Remove/Change | Expected Finding |
|---|----------|-----------------------|-----------------|
| A1 | No UCB1 (random selection) | Replace UCB1 with uniform random | UCB1 crucial for efficiency |
| A2 | No exploration ($c=0$) | Set exploration constant to 0 | Pure exploitation misses relevant code |
| A3 | High exploration ($c=3.0$) | Triple exploration constant | Over-exploration wastes LLM calls |
| A4 | $c$ sweep | $c \in \{0.5, 1.0, 1.414, 2.0, 3.0\}$ | Sweet spot around $\sqrt{2}$ |
| A5 | No LLM simulation | Replace LLM scores with embedding cosine sim | LLM provides superior discrimination |
| A6 | No backpropagation | Skip backprop step | Search doesn't converge |
| A7 | Max iterations sweep | $\{10, 25, 50, 100, 200\}$ | Diminishing returns after ~50 |
| A8 | RL: Merge only | Disable Split + Reparent | Merge alone insufficient |
| A9 | RL: Split only | Disable Merge + Reparent | Split alone insufficient |
| A10 | RL: No reward shaping | Remove depth penalty | Agent ignores efficiency |
| A11 | RL: Reward $\lambda$ sweep | Vary $\lambda_1, \lambda_2$ ratios | Optimal balance of MRR vs depth |
| A12 | Dense vs. Tree structure | Flat search vs hierarchical | Structure provides 10-15% MRR gain |

---

## 10. Phase 7: LaTeX Paper Structure

### 10.1 Section-by-Section Outline

```
1. Introduction (1.5 pages)
   - Problem: Repository-level code retrieval is hard
   - Limitation: Flat retrieval ignores code structure
   - Our approach: MCTS + RL adaptive indexing (1 paragraph each)
   - Contributions bullet list (3 items)
   - Figure 1: System overview diagram

2. Related Work (1 page)
   - 2.1 Code Search & Retrieval (CodeBERT, UniXcoder, GraphCodeBERT)
   - 2.2 Structure-Aware Code Understanding (AST-based methods)
   - 2.3 MCTS in Code (Ranger, code generation)
   - 2.4 Learned Index Structures (Kraska et al.)

3. Methodology (2.5 pages)
   - 3.1 Problem Formulation
   - 3.2 AST-Based Code Tree Construction (tree-sitter)
   - 3.3 MCTS-Guided Search (UCB1, Simulation, Backprop)
     - Algorithm 1: MCTS Search pseudocode
   - 3.4 RL Adaptive Index Restructuring (State, Actions, Reward)
     - Algorithm 2: PPO Training pseudocode
     - Figure 2: Mutation operators visualization
   - 3.5 Hybrid Dense + MCTS Retrieval

4. Experimental Setup (1 page)
   - 4.1 Datasets (Table 1: CodeSearchNet + extensions)
   - 4.2 Baselines (5 systems)
   - 4.3 Metrics
   - 4.4 Implementation Details (hardware, models, hyperparameters)

5. Results (2 pages)
   - 5.1 Main Results (Table 2: All methods × all metrics)
   - 5.2 RL Index Optimization (Figure 3: MRR over training, Figure 4: tree evolution)
   - 5.3 Efficiency Analysis (Figure 5: LLM calls vs accuracy Pareto)
   - 5.4 Cross-Language Generalization (Table 3)
   - 5.5 Statistical Significance (p-values, CIs)

6. Ablation Studies (1 page)
   - 6.1 MCTS Components (Table 4: A1-A7)
   - 6.2 RL Components (Table 5: A8-A12)
   - 6.3 Exploration-Exploitation Tradeoff (Figure 6)

7. Conclusion (0.5 page)
   - Summary, limitations, future work

Appendix (supplementary):
   - A: Full per-repo results
   - B: Hyperparameter sensitivity
   - C: Example search trajectories
   - D: Reproducibility checklist
```

### 10.2 Figures to Generate (7 Total)

| Fig # | Type | Content | Tool |
|-------|------|---------|------|
| 1 | System diagram | Architecture overview with MCTS + RL loop | draw.io / TikZ |
| 2 | Diagram | Merge/Split/Reparent mutation operators | TikZ |
| 3 | Line plot | MRR improvement over RL training steps | matplotlib + scienceplots |
| 4 | Tree visualization | Tree structure before vs after RL | networkx + graphviz |
| 5 | Scatter plot | LLM calls vs MRR (Pareto frontier) | matplotlib |
| 6 | Line plot | UCB1 $c$ sweep (exploration vs exploitation) | matplotlib |
| 7 | Bar chart | Main results comparison across methods | matplotlib |

---

## 11. Phase 8: Reproducibility & Submission

### 11.1 Reproducibility Package
- `Dockerfile` with exact Python/CUDA versions
- `run_all.sh` — single script to reproduce all results
- `requirements-research.txt` — pinned versions
- Anonymous GitHub repo for double-blind review

### 11.2 Camera-Ready Checklist
- [ ] All figures are PDF (vector graphics)
- [ ] References use `\citep{}` and `\citet{}` correctly
- [ ] NeurIPS formatting check passes (`neurips_2025.sty`)
- [ ] Appendix includes reproducibility checklist
- [ ] Code link (anonymous) in abstract footnote
- [ ] Supplementary materials uploaded

---

## 12. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| MCTS doesn't beat dense baseline | Medium | Fatal | Hybrid approach as fallback; tune $c$ extensively |
| RL doesn't converge | Medium | High | Start with simpler reward; use imitation learning warmup |
| LLM costs too high for benchmarks | High | Medium | Cache all LLM calls; use smaller model (Qwen 2.5 7B) |
| CodeSearchNet trees too shallow | Low | Medium | Add custom large repos with deep hierarchies |
| Reviewers say "not novel enough" | Medium | Fatal | Emphasize RL adaptive indexing (totally novel in code retrieval) |
| Wall-clock latency too high | Medium | Medium | Report amortized cost; show RL reduces future queries |

---

## 13. Timeline (Aggressive but Achievable)

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Setup + MCTS core | Environment, MCTSNode, basic search working |
| 2 | MCTS integration + tests | MCTS integrated into code_index.py, all tests pass |
| 3 | RL mutations + env | Tree mutations working, Gymnasium env stepping |
| 4 | RL training + baselines | PPO training loop, all 5 baselines implemented |
| 5 | Large-scale benchmarks | 25-repo evaluation complete, all metrics computed |
| 6 | Ablations + analysis | 12 ablations run, all figures generated |
| 7 | Paper writing | Full LaTeX draft complete |
| 8 | Polish + submission | Camera-ready, reproducibility package, submit |

---

## Open Questions for Your Review

> [!IMPORTANT]
> 1. **Target venue**: NeurIPS 2025 (deadline ~May), ICML 2025 (deadline ~Jan, passed), or ICLR 2026 (deadline ~Oct)? This affects the timeline.

> [!IMPORTANT]  
> 2. **LLM for MCTS simulation**: Should we use Qwen 2.5 7B (fast, local) or a larger model? Larger = better scores but slower benchmarks.

> [!WARNING]
> 3. **Compute budget**: RL training + 660 queries × LLM calls is ~2000-5000 LLM calls total. Are you running on GPU? Local Ollama should handle this in 8-12 hours.

> [!NOTE]
> 4. **Co-authors / advisor**: Top conferences heavily weigh institutional affiliation. Do you have an advisor or lab connection to list?

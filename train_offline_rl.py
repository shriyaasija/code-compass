"""
Offline RL Training Script.

Trains on 80% train repos, saves one optimized tree per test repo.
Runtime: ~10-20 minutes on M3 (no LLM calls during training).

Usage:
  python train_offline_rl.py
  python train_offline_rl.py --timesteps 20000  # faster, less training
  python train_offline_rl.py --timesteps 50000  # more training, better results
"""

import argparse
import json
import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer

from research.rl_index.offline_trainer import (
    FastTreeIndexEnv,
    build_query_buffer_from_metadata,
    compute_proxy_mrr,
    collect_leaves_with_embeddings,
)
from research.rl_index.tree_mutations import count_leaves, tree_depth, deep_copy_tree


SPLIT_PATH = "benchmark_results/train_test_split.json"
PROPER_TREES_DIR = "benchmark_results/proper_trees"
PROPER_METADATA_PATH = "benchmark_results/proper_benchmark_metadata.json"
CHECKPOINT_DIR = "research/rl_index/checkpoints"
OPTIMIZED_TREES_DIR = "benchmark_results/optimized_trees"


def load_repo_data(repo_id: str, metadata_list: list):
    """Load tree JSON and metadata for a repo."""
    tree_path = os.path.join(PROPER_TREES_DIR, f"{repo_id}.json")
    if not os.path.exists(tree_path):
        return None, None
    with open(tree_path) as f:
        tree = json.load(f)
    meta = next((m for m in metadata_list if m['repo_id'] == repo_id), None)
    return tree, meta


def train(timesteps: int = 30000, max_steps_per_episode: int = 30):
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    print("=" * 70)
    print("OFFLINE RL TRAINING (zero LLM calls)")
    print("=" * 70)

    # Load split and metadata
    split = json.load(open(SPLIT_PATH))
    train_repo_ids = split['train']
    test_repo_ids  = split['test']
    metadata_list  = json.load(open(PROPER_METADATA_PATH))

    print(f"\nTrain repos: {len(train_repo_ids)}")
    print(f"Test repos:  {len(test_repo_ids)}")

    # Load embedding model once
    print("\nLoading embedding model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # ─── Training phase ───
    # We train one PPO model, cycling through all train repos
    # Each "environment reset" picks a random train repo
    # This gives the agent diversity

    print("\nPreparing training environments...")
    train_envs = []
    for repo_id in train_repo_ids:
        tree, meta = load_repo_data(repo_id, metadata_list)
        if tree is None or meta is None:
            continue
        buf, q_embs = build_query_buffer_from_metadata(meta, embed_model)
        if len(buf) < 3:
            continue
        initial_mrr = compute_proxy_mrr(tree, buf, q_embs)
        print(f"  {repo_id}: {count_leaves(tree)} leaves, "
              f"depth={tree_depth(tree)}, "
              f"{len(buf)} queries, initial_mrr={initial_mrr:.3f}")
        train_envs.append({
            'tree': tree, 'buf': buf, 'q_embs': q_embs,
            'repo_id': repo_id, 'initial_mrr': initial_mrr,
        })

    if not train_envs:
        print("ERROR: No valid training environments found. Check your proper_trees/ dir.")
        return

    print(f"\nUsing {len(train_envs)} training environments")

    # Use the first train env to build the initial PPO model
    # (We'll cycle repos by resetting to different trees)
    first = train_envs[0]
    env = FastTreeIndexEnv(
        tree=first['tree'],
        query_buffer=first['buf'],
        query_embeddings=first['q_embs'],
        max_steps=max_steps_per_episode,
    )

    def mask_fn(e):
        return e.get_action_mask()

    wrapped = ActionMasker(env, mask_fn)

    print("\nCreating MaskablePPO agent...")
    model = MaskablePPO(
        "MlpPolicy",
        wrapped,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=5,
        gamma=0.95,
        clip_range=0.2,
        ent_coef=0.03,    # was 0.15 — lower so agent learns to be selective
        verbose=0,
        seed=42,
    )

    # Train in chunks, cycling through repos
    chunk = max(timesteps // max(len(train_envs), 1), 3000)
    total_trained = 0
    start_time = time.time()

    print(f"\nTraining for {timesteps} total timesteps...")
    print(f"  ({chunk} timesteps per repo, cycling through {len(train_envs)} repos)")

    for i, env_data in enumerate(train_envs * 10): 
        if total_trained >= timesteps:
            break

        # Swap the underlying tree
        inner_env = wrapped.env if hasattr(wrapped, 'env') else wrapped
        while hasattr(inner_env, 'env'):
            inner_env = inner_env.env

        inner_env._original_tree = deep_copy_tree(env_data['tree'])
        inner_env._query_buffer = env_data['buf']
        inner_env._query_embeddings = env_data['q_embs']
        inner_env._initial_leaves = count_leaves(env_data['tree'])
        inner_env._initial_depth = max(tree_depth(env_data['tree']), 1)

        steps_this_round = min(chunk, timesteps - total_trained)
        model.learn(
            total_timesteps=steps_this_round,
            reset_num_timesteps=(total_trained == 0),
            progress_bar=False,
        )
        total_trained += steps_this_round

        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"  {total_trained}/{timesteps} timesteps ({elapsed:.0f}s)")

    elapsed = time.time() - start_time
    print(f"\nTraining complete: {total_trained} timesteps in {elapsed:.1f}s")

    # Save model
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model_path = os.path.join(CHECKPOINT_DIR, "ppo_tree_index")
    model.save(model_path)
    print(f"Model saved: {model_path}")

    # ─── Apply RL to test repos ───
    print("\n" + "=" * 70)
    print("APPLYING RL TO TEST REPOS")
    print("=" * 70)

    os.makedirs(OPTIMIZED_TREES_DIR, exist_ok=True)
    results = []

    for repo_id in test_repo_ids:
        tree, meta = load_repo_data(repo_id, metadata_list)
        if tree is None or meta is None:
            print(f"  Skipping {repo_id} (not found)")
            continue

        buf, q_embs = build_query_buffer_from_metadata(meta, embed_model)

        before_quality = compute_proxy_mrr(tree, buf, q_embs)  # now returns structural quality
        before_depth = tree_depth(optimized_tree := deep_copy_tree(tree))
        before_leaves = count_leaves(tree)

        # Run one episode with trained policy
        from sb3_contrib.common.maskable.utils import get_action_masks

        test_env = FastTreeIndexEnv(
            tree=tree, query_buffer=buf, query_embeddings=q_embs,
            max_steps=max_steps_per_episode,
        )
        wrapped_test = ActionMasker(test_env, lambda e: e.get_action_mask())

        obs, _ = wrapped_test.reset()
        action_log = []
        for _ in range(max_steps_per_episode):
            masks = get_action_masks(wrapped_test)
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, info = wrapped_test.step(int(action))
            action_log.append(info)
            if terminated or truncated:
                break

        optimized_tree = deep_copy_tree(test_env.current_tree)
        after_quality = compute_proxy_mrr(optimized_tree, buf, q_embs)
        after_depth = tree_depth(optimized_tree)

        # Count mutations
        action_names = {0: 'Merge', 1: 'Split', 2: 'NoOp'}
        mutation_counts = {'Merge': 0, 'Split': 0, 'NoOp': 0}
        for step_info in action_log:
            if step_info.get('valid', False):
                mutation_counts[action_names.get(step_info['action'], 'NoOp')] += 1

        # Save optimized tree
        opt_path = os.path.join(OPTIMIZED_TREES_DIR, f"{repo_id}.json")
        with open(opt_path, 'w') as f:
            json.dump(optimized_tree, f)

        result = {
            'repo_id': repo_id,
            'before_quality': round(before_quality, 4),
            'after_quality':  round(after_quality, 4),
            'delta_quality':  round(after_quality - before_quality, 4),
            'before_depth': before_depth,
            'after_depth':  after_depth,
            'before_leaves': before_leaves,
            'after_leaves':  count_leaves(optimized_tree),
            'mutations': mutation_counts,
        }

        results.append(result)
        print(f"\n  {repo_id}:")
        print(f"    Structural quality: {before_quality:.3f} → {after_quality:.3f} "
              f"(Δ={after_quality - before_quality:+.3f})")
        print(f"    Depth: {before_depth} → {after_depth}")
        print(f"    Mutations: {mutation_counts}")

    # Save RL results
    rl_results_path = "benchmark_results/rl_optimization_results.json"
    json.dump(results, open(rl_results_path, 'w'), indent=2)
    print(f"\nRL results saved: {rl_results_path}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=30000)
    p.add_argument("--max-steps", type=int, default=30)
    args = p.parse_args()
    train(timesteps=args.timesteps, max_steps_per_episode=args.max_steps)


if __name__ == "__main__":
    main()
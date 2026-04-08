import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from research.rl_index.env import TreeIndexEnv
from research.rl_index.tree_mutations import deep_copy_tree, count_leaves, tree_depth


def load_tree_and_queries(
    tree_path: str,
    queries_path: str = None,
    num_queries: int = 20,
) -> tuple:
    """
    Load a tree and build a query buffer.

    If no queries_path is given, auto-generates queries from leaf node summaries.
    """
    with open(tree_path, 'r') as f:
        tree = json.load(f)

    if queries_path and os.path.exists(queries_path):
        with open(queries_path, 'r') as f:
            queries = json.load(f)
    else:
        # Auto-generate queries from leaf summaries
        queries = auto_generate_queries(tree, num_queries)

    return tree, queries


def auto_generate_queries(tree: Dict, n: int = 20) -> List[Dict]:
    """
    Generate synthetic queries from leaf node summaries.

    Each query is the leaf's summary, and ground_truth is the leaf's title.
    This gives us a query buffer without needing an LLM.
    """
    from research.rl_index.tree_mutations import get_children
    leaves = []
    stack = [tree]
    while stack:
        node = stack.pop()
        children = get_children(node)
        if not children:
            summary = node.get('summary', '')
            title = node.get('title', node.get('name', ''))
            if summary and title:
                leaves.append({'query': summary, 'ground_truth': title})
        else:
            for child in children:
                stack.append(child)

    # Take up to n leaves
    if len(leaves) > n:
        import random
        random.seed(42)
        leaves = random.sample(leaves, n)

    return leaves


def train(
    tree_path: str,
    output_dir: str = "research/rl_index/checkpoints",
    total_timesteps: int = 10000,
    max_steps_per_episode: int = 50,
    learning_rate: float = 3e-4,
    n_steps: int = 128,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    lambda_mrr: float = 1.0,
    lambda_depth: float = 0.1,
    seed: int = 42,
    verbose: int = 1,
):
    """
    Train a MaskablePPO agent to optimize tree index structure.
    """
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    print("=" * 70)
    print("🧠 RL ADAPTIVE INDEX TRAINING")
    print("=" * 70)

    # Load data
    tree, queries = load_tree_and_queries(tree_path)
    print(f"  Tree: {tree_path}")
    print(f"  Leaves: {count_leaves(tree)}")
    print(f"  Depth: {tree_depth(tree)}")
    print(f"  Queries: {len(queries)}")
    print(f"  Timesteps: {total_timesteps}")

    # Create environment
    env = TreeIndexEnv(
        tree=tree,
        query_buffer=queries,
        max_steps=max_steps_per_episode,
        lambda_mrr=lambda_mrr,
        lambda_depth=lambda_depth,
        seed=seed,
    )

    # Wrap with ActionMasker for MaskablePPO
    def mask_fn(env):
        return env.get_action_mask()

    env = ActionMasker(env, mask_fn)

    # Create agent
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        seed=seed,
        verbose=verbose,
    )

    print(f"\n  Training MaskablePPO...")
    start_time = time.time()

    model.learn(total_timesteps=total_timesteps)

    elapsed = time.time() - start_time
    print(f"\n  ✅ Training complete in {elapsed:.1f}s")

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "ppo_tree_index")
    model.save(model_path)
    print(f"  Model saved to: {model_path}")

    # Run one final episode to get the optimized tree
    optimized_tree = run_inference(model, env, tree, queries, max_steps_per_episode)

    # Save optimized tree
    opt_tree_path = os.path.join(output_dir, "optimized_tree.json")
    with open(opt_tree_path, 'w') as f:
        json.dump(optimized_tree, f, indent=2)
    print(f"  Optimized tree saved to: {opt_tree_path}")

    return model, optimized_tree


def run_inference(model, wrapped_env, tree, queries, max_steps):
    """Run one episode with the trained model to produce optimized tree."""
    from sb3_contrib.common.maskable.utils import get_action_masks

    obs, _ = wrapped_env.reset()
    for step in range(max_steps):
        action_masks = get_action_masks(wrapped_env)
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        if terminated or truncated:
            break

    # Extract the optimized tree from the unwrapped env
    inner_env = wrapped_env.unwrapped if hasattr(wrapped_env, 'unwrapped') else wrapped_env
    # Navigate through wrappers
    while hasattr(inner_env, 'env'):
        inner_env = inner_env.env
    return deep_copy_tree(inner_env.current_tree)


def main():
    parser = argparse.ArgumentParser(description="Train RL agent for tree index optimization")
    parser.add_argument("--tree", required=True, help="Path to PageIndex JSON tree")
    parser.add_argument("--queries", default=None, help="Path to query buffer JSON (optional)")
    parser.add_argument("--output-dir", default="research/rl_index/checkpoints")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-mrr", type=float, default=1.0)
    parser.add_argument("--lambda-depth", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    train(
        tree_path=args.tree,
        output_dir=args.output_dir,
        total_timesteps=args.timesteps,
        max_steps_per_episode=args.max_steps,
        learning_rate=args.lr,
        lambda_mrr=args.lambda_mrr,
        lambda_depth=args.lambda_depth,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
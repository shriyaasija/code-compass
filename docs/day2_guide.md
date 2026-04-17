# Day 2 Guide: Prior Training on DevQuery-Bench

**Goal:** Train the 200K-parameter MLP prior on the 10-repo training split of DevQuery-Bench. By end of day you'll have a new `prior.pt` that knows how to navigate the larger repos.

**Time estimate:** 3–4 hours (1h data generation, 2h training, 1h validation)

**Prerequisites:** Day 1 complete. You should have `devquery_bench/devquery_bench.json` and all trees summarized + embedded.

---

## Step 0: Verify Day 1 Output (5 minutes)

```bash
cd ~/code-compass
source venv/bin/activate

# Quick check
python3 -c "
import json
with open('devquery_bench/devquery_bench.json') as f:
    bench = json.load(f)
print(f'DevQuery-Bench entries: {len(bench)}')

with open('devquery_bench/train_test_split.json') as f:
    split = json.load(f)
print(f'Train repos: {len(split[\"train\"])}')
print(f'Test repos: {len(split[\"test\"])}')
"
```

Expected: 200+ entries, 10 train repos, 5 test repos. If the numbers are off, go back and finish Day 1.

---

## Step 1: Build Prior Training Data (30–60 minutes)

This is the same concept as `build_prior_training_data.py` but adapted for DevQuery-Bench format.

```bash
cat > devquery_bench/build_prior_data.py << 'PYEOF'
"""
Build prior training data from DevQuery-Bench annotations.

For each (query, ground_truth_function) pair in the TRAINING repos:
1. Find the path from tree root to the ground-truth function
2. At each level of the path, create:
   - Positive pair: (query_emb, on_path_node_emb, label=1)
   - Negative pairs: (query_emb, each_sibling_emb, label=0)

This teaches the prior to navigate the tree correctly.
"""
import json
import os
import sys
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_path_to_node(root: Dict, target_title: str) -> Optional[List[Dict]]:
    """DFS to find path from root to node with matching title."""
    def dfs(node, path):
        current_title = node.get('title', node.get('name', ''))
        if current_title == target_title:
            return path + [node]
        for child in node.get('nodes', node.get('children', [])):
            result = dfs(child, path + [node])
            if result is not None:
                return result
        return None
    return dfs(root, [])


def get_embedding(node: Dict) -> Optional[np.ndarray]:
    """Extract embedding from node."""
    emb = node.get('embedding')
    if emb is None:
        return None
    arr = np.array(emb, dtype=np.float32)
    if arr.ndim != 1 or len(arr) < 10:
        return None
    return arr


def main():
    print("=" * 60)
    print("BUILDING PRIOR TRAINING DATA (DevQuery-Bench)")
    print("=" * 60)

    # Load data
    with open('devquery_bench/train_test_split.json') as f:
        split = json.load(f)
    train_repos = set(split['train'])

    with open('devquery_bench/devquery_bench.json') as f:
        bench = json.load(f)

    # Filter to training repos only
    train_entries = [e for e in bench if e['repo_id'] in train_repos]
    print(f"\nTraining entries: {len(train_entries)} (from {len(train_repos)} repos)")
    print(f"Test entries: {len(bench) - len(train_entries)} (held out)")

    # Load embedding model
    print(f"\nLoading embedding model...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embed_dim = embed_model.get_sentence_embedding_dimension()
    print(f"Embedding dim: {embed_dim}")

    all_query_embs = []
    all_node_embs = []
    all_labels = []

    used = 0
    skipped = 0

    # Cache loaded trees
    tree_cache = {}

    for entry in train_entries:
        repo_id = entry['repo_id']
        query = entry['query']
        ground_truth = entry['ground_truth']

        # Load tree (cached)
        if repo_id not in tree_cache:
            tree_path = f"devquery_bench/trees/{repo_id}.json"
            if not os.path.exists(tree_path):
                skipped += 1
                continue
            with open(tree_path) as f:
                tree_cache[repo_id] = json.load(f)

        tree = tree_cache[repo_id]

        # Find path from root to ground truth
        path = find_path_to_node(tree, ground_truth)
        if path is None or len(path) < 2:
            skipped += 1
            continue

        # Embed the query
        query_emb = embed_model.encode(query, show_progress_bar=False)

        # For each level in the path, create training pairs
        for i in range(1, len(path)):
            on_path_node = path[i]
            parent = path[i - 1]
            siblings = parent.get('nodes', parent.get('children', []))

            if not siblings:
                continue

            on_path_title = on_path_node.get('title', '')

            for sibling in siblings:
                node_emb = get_embedding(sibling)
                if node_emb is None:
                    continue

                sibling_title = sibling.get('title', '')
                label = 1.0 if sibling_title == on_path_title else 0.0

                all_query_embs.append(query_emb)
                all_node_embs.append(node_emb)
                all_labels.append(label)

        used += 1

    if not all_query_embs:
        print("\n❌ No training pairs generated!")
        print("   Check that trees have 'embedding' fields.")
        return

    print(f"\nResults:")
    print(f"  Used queries:    {used}")
    print(f"  Skipped queries: {skipped}")
    print(f"  Total pairs:     {len(all_labels)}")
    print(f"  Positives:       {sum(1 for l in all_labels if l == 1.0)}")
    print(f"  Negatives:       {sum(1 for l in all_labels if l == 0.0)}")
    pos_rate = sum(1 for l in all_labels if l == 1.0) / len(all_labels) * 100
    print(f"  Positive rate:   {pos_rate:.1f}%")

    # Save as tensors
    data = {
        'query_embs': torch.tensor(np.array(all_query_embs), dtype=torch.float32),
        'node_embs': torch.tensor(np.array(all_node_embs), dtype=torch.float32),
        'labels': torch.tensor(all_labels, dtype=torch.float32),
        'embed_dim': embed_dim,
    }

    output_path = 'devquery_bench/prior_training_data.pt'
    torch.save(data, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Saved to: {output_path} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
PYEOF

echo "✅ Created devquery_bench/build_prior_data.py"
```

Run it:

```bash
cd ~/code-compass
source venv/bin/activate

python devquery_bench/build_prior_data.py
```

**Expected output:** Something like:
```
Total pairs:     50000+
Positives:       6000+
Negatives:       44000+
Positive rate:   ~12%
```

If you get less than 5000 total pairs, check that your trees have embeddings (`python3 -c "import json; t=json.load(open('devquery_bench/trees/pallets__click.json')); print('embedding' in str(t)[:500])"`)

---

## Step 2: Train the Prior MLP (1–2 hours)

We reuse the existing `train_prior.py` but point it at the new data.

```bash
cat > devquery_bench/train_prior.py << 'PYEOF'
"""
Train the relevance prior MLP on DevQuery-Bench training data.

Same architecture as the original train_prior.py but uses the
DevQuery-Bench training data instead of CodeSearchNet.
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.mcts.relevance_prior import RelevancePrior


DATA_PATH  = "devquery_bench/prior_training_data.pt"
MODEL_PATH = "devquery_bench/prior.pt"


def train(epochs=20, lr=1e-3, batch_size=512, val_fraction=0.1):
    print("=" * 60)
    print("TRAINING RELEVANCE PRIOR (DevQuery-Bench)")
    print("=" * 60)

    # Load data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Training data not found at {DATA_PATH}")
        print("   Run devquery_bench/build_prior_data.py first")
        return

    data = torch.load(DATA_PATH, weights_only=True)
    query_embs = data['query_embs']
    node_embs  = data['node_embs']
    labels     = data['labels']
    embed_dim  = int(data['embed_dim'])

    print(f"\nData:")
    print(f"  Total pairs:    {len(labels)}")
    print(f"  Positives:      {labels.sum().int()} ({100*labels.mean():.1f}%)")
    print(f"  Embedding dim:  {embed_dim}")

    # Class imbalance handling
    n_pos = labels.sum().item()
    n_neg = (1 - labels).sum().item()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    print(f"  Positive weight: {pos_weight.item():.2f}x")

    # Dataset split
    dataset = TensorDataset(query_embs, node_embs, labels)
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
    print(f"\n  Train: {n_train} | Val: {n_val}")

    # Model
    model = RelevancePrior(embed_dim=embed_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("  Using CUDA GPU")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("  Using CPU")

    model = model.to(device)
    pos_weight = pos_weight.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Weighted BCE loss
    def weighted_bce(preds, targets):
        eps = 1e-7
        preds = preds.clamp(eps, 1 - eps)
        loss = -(pos_weight[0] * targets * torch.log(preds)
                 + (1 - targets) * torch.log(1 - preds))
        return loss.mean()

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    best_val_loss = float('inf')
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for q_batch, n_batch, l_batch in train_loader:
            q_batch = q_batch.to(device)
            n_batch = n_batch.to(device)
            l_batch = l_batch.to(device)

            preds = model(q_batch, n_batch)
            loss = weighted_bce(preds, l_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(l_batch)

        train_loss /= n_train
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for q_batch, n_batch, l_batch in val_loader:
                q_batch = q_batch.to(device)
                n_batch = n_batch.to(device)
                l_batch = l_batch.to(device)

                preds = model(q_batch, n_batch)
                loss = weighted_bce(preds, l_batch)
                val_loss += loss.item() * len(l_batch)

                predicted = (preds > 0.5).float()
                correct += (predicted == l_batch).sum().item()
                total += len(l_batch)

        val_loss /= n_val
        val_acc = correct / total * 100

        print(f"  Epoch {epoch:2d}/{epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.1f}%  "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            model_cpu = model.to('cpu')
            model_cpu.save(MODEL_PATH)
            model = model.to(device)

    print(f"\n✅ Best: val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.1f}%")
    print(f"   Model saved to: {MODEL_PATH}")

    # Also copy to the standard location for puct_search.py
    import shutil
    shutil.copy(MODEL_PATH, "research/mcts/prior.pt")
    print(f"   Also copied to: research/mcts/prior.pt")

    # Sanity check
    print("\nSanity check on 5 examples:")
    model.eval()
    model = model.to('cpu')
    for i in range(min(5, len(val_set))):
        q, n, label = val_set[i]
        with torch.no_grad():
            pred = model(q, n)
        status = "✅" if (pred.item() > 0.5) == (label.item() > 0.5) else "❌"
        print(f"  {status} label={label.item():.0f}, pred={pred.item():.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch-size", type=int,   default=512)
    args = p.parse_args()
    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
```
PYEOF

echo "✅ Created devquery_bench/train_prior.py"
```

Run it:

```bash
cd ~/code-compass
source venv/bin/activate

python devquery_bench/train_prior.py --epochs 20 --lr 1e-3 --batch-size 512
```

**What to expect:**
- GPU: ~5–10 minutes for 20 epochs
- CPU: ~30–60 minutes for 20 epochs
- You should see validation accuracy climbing to **>70%** (ideally >75%)
- If val_acc stays below 60%, something is wrong with the training data

**Watch for:**
```
  Epoch  1/20  train_loss=0.8432  val_loss=0.6521  val_acc=63.2%
  Epoch  5/20  train_loss=0.4123  val_loss=0.3891  val_acc=72.8%
  Epoch 10/20  train_loss=0.2456  val_loss=0.2789  val_acc=78.1%  ← good
  Epoch 20/20  train_loss=0.1234  val_loss=0.2654  val_acc=79.5%  ← great
```

> **Troubleshooting:**
> - val_acc stuck at ~50%: the training data might have all same labels. Check with `python3 -c "import torch; d=torch.load('devquery_bench/prior_training_data.pt', weights_only=True); print(d['labels'].mean())"` — should be ~0.1 to 0.2
> - val_acc oscillating: reduce lr to 5e-4
> - Training very slow on CPU: reduce batch-size to 128 and epochs to 10

---

## Step 3: Validate the Prior Works (30 minutes)

Quick sanity check: does the prior actually prefer on-path nodes over off-path nodes?

```bash
cat > devquery_bench/validate_prior.py << 'PYEOF'
"""
Quick validation: on the TEST repos, check if the prior ranks
the correct path higher than random paths.
"""
import json
import os
import sys
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.mcts.relevance_prior import RelevancePrior


def find_path(root, target):
    def dfs(node, path):
        title = node.get('title', node.get('name', ''))
        if title == target:
            return path + [node]
        for c in node.get('nodes', node.get('children', [])):
            r = dfs(c, path + [node])
            if r: return r
        return None
    return dfs(root, [])


def main():
    prior = RelevancePrior.load('devquery_bench/prior.pt')
    prior.eval()
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    with open('devquery_bench/train_test_split.json') as f:
        split = json.load(f)
    with open('devquery_bench/devquery_bench.json') as f:
        bench = json.load(f)

    test_entries = [e for e in bench if e['repo_id'] in split['test']]
    print(f"Validating on {len(test_entries)} test entries...")

    correct_levels = 0
    total_levels = 0
    tree_cache = {}

    for entry in test_entries[:30]:  # Sample 30 for speed
        repo_id = entry['repo_id']
        if repo_id not in tree_cache:
            tp = f"devquery_bench/trees/{repo_id}.json"
            if not os.path.exists(tp): continue
            with open(tp) as f:
                tree_cache[repo_id] = json.load(f)

        tree = tree_cache[repo_id]
        path = find_path(tree, entry['ground_truth'])
        if not path or len(path) < 2: continue

        query_emb = embed_model.encode(entry['query'], show_progress_bar=False)
        q_tensor = torch.tensor(query_emb, dtype=torch.float32)

        for i in range(1, len(path)):
            parent = path[i - 1]
            on_path = path[i]
            siblings = parent.get('nodes', parent.get('children', []))
            if len(siblings) < 2: continue

            on_path_title = on_path.get('title', '')
            scores = []
            on_path_score = None

            for sib in siblings:
                emb = sib.get('embedding')
                if emb is None: continue
                n_tensor = torch.tensor(emb, dtype=torch.float32)
                with torch.no_grad():
                    score = prior(q_tensor, n_tensor).item()
                scores.append(score)
                if sib.get('title', '') == on_path_title:
                    on_path_score = score

            if on_path_score is not None and scores:
                rank = sum(1 for s in scores if s > on_path_score) + 1
                if rank == 1:
                    correct_levels += 1
                total_levels += 1

    if total_levels > 0:
        acc = correct_levels / total_levels * 100
        print(f"\n✅ Per-level accuracy: {correct_levels}/{total_levels} = {acc:.1f}%")
        print(f"   (Prior correctly ranks the on-path node #1 among siblings)")
        if acc > 50:
            print(f"   This is ABOVE random chance — prior is learning!")
        if acc > 70:
            print(f"   This is really good — prior should help PUCT significantly")
    else:
        print(f"❌ Could not evaluate any levels. Check paths and embeddings.")


if __name__ == '__main__':
    main()
PYEOF

python devquery_bench/validate_prior.py
```

**Target:** Per-level accuracy > 60%. If you get > 70%, the prior is strong.

---

## Day 2 Outputs Checklist

- [ ] `devquery_bench/prior_training_data.pt` — training tensor (should be several MB)
- [ ] `devquery_bench/prior.pt` — trained prior weights
- [ ] `research/mcts/prior.pt` — copy of trained prior (for puct_search.py)
- [ ] Prior validation accuracy > 70% on per-level binary classification
- [ ] Prior per-level ranking accuracy > 60% on test repos

**Git checkpoint:**
```bash
cd ~/code-compass
git add devquery_bench/build_prior_data.py devquery_bench/train_prior.py devquery_bench/validate_prior.py
git add devquery_bench/prior_training_data.pt devquery_bench/prior.pt research/mcts/prior.pt
git commit -m "Day 2: Prior trained on DevQuery-Bench (val_acc=XX%, per-level=XX%)"
```

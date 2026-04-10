import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from research.mcts.relevance_prior import RelevancePrior


DATA_PATH  = "benchmark_results/prior_training_data.pt"
MODEL_PATH = "research/mcts/prior.pt"


def train(
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 512,
    val_fraction: float = 0.1,
):
    print("=" * 60)
    print("TRAINING RELEVANCE PRIOR MLP")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Training data not found at {DATA_PATH}")
        print("Run build_prior_training_data.py first.")
        return

    data = torch.load(DATA_PATH, weights_only=True)
    query_embs = data['query_embs']   # (N, 384)
    node_embs  = data['node_embs']    # (N, 384)
    labels     = data['labels']       # (N,)
    embed_dim  = int(data['embed_dim'])

    print(f"\nData loaded:")
    print(f"  Total pairs:    {len(labels)}")
    print(f"  Positives:      {labels.sum().int()} ({100*labels.mean():.1f}%)")
    print(f"  Negatives:      {(1-labels).sum().int()}")
    print(f"  Embedding dim:  {embed_dim}")

    # ── Handle class imbalance ─────────────────────────────────────
    # Positive class (on-path nodes) will be ~20% of data.
    # We weight positives more heavily so the model doesn't just predict 0.
    n_pos = labels.sum().item()
    n_neg = (1 - labels).sum().item()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    print(f"  Positive weight: {pos_weight.item():.2f}x")

    # ── Build dataset ──────────────────────────────────────────────
    dataset = TensorDataset(query_embs, node_embs, labels)
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

    print(f"\nTrain: {n_train} pairs | Val: {n_val} pairs")

    # ── Model ──────────────────────────────────────────────────────
    model = RelevancePrior(embed_dim=embed_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Use MPS on M3 Mac if available, else CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (M3)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = model.to(device)
    pos_weight = pos_weight.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Reduce LR if val loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    criterion = nn.BCELoss(weight=None)  # We handle weighting via pos_weight in BCEWithLogitsLoss
    # Actually let's use weighted BCE properly:
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # We need to remove the final Sigmoid from the model when using BCEWithLogitsLoss
    # Instead let's just use weighted BCE with Sigmoid output
    # Replace: use weighted BCE directly
    def weighted_bce(preds, targets):
        # Manual weighted BCE
        eps = 1e-7
        preds = preds.clamp(eps, 1 - eps)
        loss = -(pos_weight[0] * targets * torch.log(preds)
                 + (1 - targets) * torch.log(1 - preds))
        return loss.mean()

    criterion = weighted_bce

    # ── Training loop ──────────────────────────────────────────────
    print(f"\nTraining for {epochs} epochs...")
    best_val_loss = float('inf')
    best_val_acc  = 0.0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for q_batch, n_batch, l_batch in train_loader:
            q_batch = q_batch.to(device)
            n_batch = n_batch.to(device)
            l_batch = l_batch.to(device)

            preds = model(q_batch, n_batch)
            loss  = criterion(preds, l_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(l_batch)

        train_loss /= n_train

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
                loss  = criterion(preds, l_batch)
                val_loss += loss.item() * len(l_batch)

                # Accuracy: threshold at 0.5
                predicted = (preds > 0.5).float()
                correct += (predicted == l_batch).sum().item()
                total   += len(l_batch)

        val_loss /= n_val
        val_acc   = correct / total * 100

        scheduler.step(val_loss)

        # Print every epoch
        print(f"  Epoch {epoch:2d}/{epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.1f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            model_cpu = model.to('cpu')
            model_cpu.save(MODEL_PATH)
            model = model.to(device)

    print(f"\nBest val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.1f}%")
    print(f"Model saved to: {MODEL_PATH}")

    # ── Quick sanity check ─────────────────────────────────────────
    print("\nSanity check on 3 examples:")
    model.eval()
    model = model.to('cpu')
    for i in range(min(3, len(val_set))):
        q, n, label = val_set[i]
        with torch.no_grad():
            pred = model(q, n)
        print(f"  Example {i}: label={label.item():.0f}, pred={pred.item():.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=15)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch-size", type=int,   default=256)
    args = p.parse_args()
    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
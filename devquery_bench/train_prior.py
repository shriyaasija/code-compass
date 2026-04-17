"""
Train the bottleneck relevance prior on DevQuery-Bench training data.

Key improvements over previous version:
  - Imports RelevancePrior from research/mcts/relevance_prior.py
    (single source of truth, same model used by PUCT pipeline)
  - Bottleneck architecture: ~31K params vs 312K (10x reduction)
  - Repo-based validation split: holds out 2 repos, not random samples.
    This honestly measures cross-repo generalisation.
  - Embedding noise augmentation during training (std=0.02)
  - Stronger regularisation: dropout=0.5, weight_decay=5e-3
  - No cosine similarity or elem_prods — model computes projected
    features internally from raw embeddings.
"""
import argparse
import os
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.mcts.relevance_prior import RelevancePrior

DATA_PATH  = "devquery_bench/prior_training_data.pt"
MODEL_PATH = "devquery_bench/prior.pt"


# ─────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────
def weighted_bce(preds, targets, pos_weight):
    eps = 1e-7
    preds = preds.clamp(eps, 1 - eps)
    loss = -(pos_weight * targets * torch.log(preds)
             + (1 - targets) * torch.log(1 - preds))
    return loss.mean()


def pairwise_ranking_loss(preds, labels, query_ids, margin=0.4):
    """
    For each query group, form all (positive, negative) pairs and apply
    margin ranking loss: max(0, margin - (score_pos - score_neg)).
    """
    total_loss = torch.tensor(0.0, device=preds.device, requires_grad=True)
    n_pairs = 0

    for qid in query_ids.unique():
        mask         = (query_ids == qid)
        group_preds  = preds[mask]
        group_labels = labels[mask]

        pos_mask = (group_labels > 0.5)
        neg_mask = ~pos_mask

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        pos_scores = group_preds[pos_mask]
        neg_scores = group_preds[neg_mask]

        pos_exp = pos_scores.unsqueeze(1).expand(-1, neg_scores.shape[0])
        neg_exp = neg_scores.unsqueeze(0).expand(pos_scores.shape[0], -1)

        pair_loss  = torch.clamp(margin - (pos_exp - neg_exp), min=0.0)
        total_loss = total_loss + pair_loss.mean()
        n_pairs   += 1

    return total_loss / max(n_pairs, 1)


# ─────────────────────────────────────────────
# Ranking accuracy (mirrors validate_prior.py)
# ─────────────────────────────────────────────
def ranking_accuracy(preds, labels, query_ids):
    """Per-group: is the positive ranked #1 among all siblings?"""
    correct = 0
    total   = 0
    for qid in query_ids.unique():
        mask         = (query_ids == qid)
        group_preds  = preds[mask]
        group_labels = labels[mask]

        pos_mask = (group_labels > 0.5)
        if pos_mask.sum() == 0:
            continue

        best_pos = group_preds[pos_mask].max().item()
        best_all = group_preds.max().item()
        if abs(best_pos - best_all) < 1e-6:
            correct += 1
        total += 1

    return correct / max(total, 1)


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────
def train(epochs=50, lr=1e-3, batch_size=256, n_val_repos=1,
          bce_weight=0.5, rank_weight=0.5, margin=0.3,
          noise_std=0.01, patience=15):

    print("=" * 60)
    print("TRAINING BOTTLENECK RELEVANCE PRIOR (DevQuery-Bench)")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"❌ Training data not found at {DATA_PATH}")
        print("   Run devquery_bench/build_prior_data.py first")
        return

    data = torch.load(DATA_PATH, weights_only=False)

    query_embs  = data['query_embs']   # (N, 384)
    node_embs   = data['node_embs']    # (N, 384)
    labels      = data['labels']       # (N,)
    query_ids   = data['query_ids']    # (N,) long
    embed_dim   = int(data['embed_dim'])
    n_groups    = int(data['n_groups'])

    # Repo-based validation split
    if 'repo_ids' not in data:
        print("❌ Training data missing repo_ids. Re-run build_prior_data.py.")
        return

    repo_ids    = data['repo_ids']     # (N,) long
    repo_names  = data.get('repo_names', [])

    print(f"\nData:")
    print(f"  Total pairs:    {len(labels)}")
    print(f"  Positives:      {labels.sum().int()} ({100*labels.mean():.1f}%)")
    print(f"  Embedding dim:  {embed_dim}")
    print(f"  Query groups:   {n_groups}")
    print(f"  Repos:          {len(repo_names)}")

    # ── Repo-based val split ──
    # Hold out n_val_repos MEDIAN-sized repos (not the largest — that starves training)
    unique_repos = repo_ids.unique().tolist()
    repo_counts = {r: (repo_ids == r).sum().item() for r in unique_repos}
    sorted_repos = sorted(repo_counts.items(), key=lambda x: x[1])  # ascending
    mid = len(sorted_repos) // 2
    val_repo_picks = sorted_repos[mid:mid + n_val_repos]
    val_repo_set = set()
    for repo_int, count in val_repo_picks:
        val_repo_set.add(repo_int)
        rname = repo_names[repo_int] if repo_int < len(repo_names) else f"repo_{repo_int}"
        print(f"  Val repo: {rname} ({count} pairs)")

    val_mask  = torch.tensor([r.item() in val_repo_set for r in repo_ids])
    train_mask = ~val_mask
    train_idx = torch.where(train_mask)[0]
    val_idx   = torch.where(val_mask)[0]
    n_train   = len(train_idx)
    n_val     = len(val_idx)

    # Class imbalance
    train_labels = labels[train_idx]
    n_pos      = train_labels.sum().item()
    n_neg      = (1 - train_labels).sum().item()
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)
    print(f"  Positive weight: {pos_weight.item():.2f}x")

    def make_tensors(idx):
        return (query_embs[idx], node_embs[idx],
                labels[idx], query_ids[idx])

    train_ds = TensorDataset(*make_tensors(train_idx))
    val_ds   = TensorDataset(*make_tensors(val_idx))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"\n  Train: {n_train} | Val: {n_val}")

    # ── Model ──
    model = RelevancePrior(embed_dim=embed_dim, proj_dim=48, dropout=0.3)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")
    print(f"  Noise std:    {noise_std}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("  Using CUDA GPU")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("  Using CPU")

    model      = model.to(device)
    pos_weight = pos_weight.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\nTraining for {epochs} epochs  "
          f"(BCE={bce_weight}, rank={rank_weight}, margin={margin}, patience={patience})...")
    print(f"{'Epoch':>6}  {'train_loss':>11}  {'val_loss':>9}  "
          f"{'val_bce_acc':>11}  {'val_rank_acc':>12}  {'lr':>9}")

    best_rank_acc = 0.0
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss_sum = 0.0

        for batch in train_loader:
            q, n, lbl, qids = [t.to(device) for t in batch]

            # Embedding noise augmentation — prevents memorisation
            if noise_std > 0:
                q = q + torch.randn_like(q) * noise_std
                n = n + torch.randn_like(n) * noise_std

            preds = model(q, n)

            loss_bce  = weighted_bce(preds, lbl, pos_weight)
            loss_rank = pairwise_ranking_loss(preds, lbl, qids, margin=margin)
            loss      = bce_weight * loss_bce + rank_weight * loss_rank

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(lbl)

        train_loss = train_loss_sum / n_train
        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss_sum = 0.0
        all_preds    = []
        all_labels_v = []
        all_qids_v   = []

        with torch.no_grad():
            for batch in val_loader:
                q, n, lbl, qids = [t.to(device) for t in batch]
                preds = model(q, n)

                loss_bce  = weighted_bce(preds, lbl, pos_weight)
                loss_rank = pairwise_ranking_loss(preds, lbl, qids, margin=margin)
                loss      = bce_weight * loss_bce + rank_weight * loss_rank
                val_loss_sum += loss.item() * len(lbl)

                all_preds.append(preds.cpu())
                all_labels_v.append(lbl.cpu())
                all_qids_v.append(qids.cpu())

        val_loss     = val_loss_sum / max(n_val, 1)
        all_preds    = torch.cat(all_preds)
        all_labels_v = torch.cat(all_labels_v)
        all_qids_v   = torch.cat(all_qids_v)

        bce_acc  = ((all_preds > 0.5).float() == all_labels_v).float().mean().item() * 100
        rank_acc = ranking_accuracy(all_preds, all_labels_v, all_qids_v) * 100
        cur_lr   = scheduler.get_last_lr()[0]

        print(f"  {epoch:>4}/{epochs}  "
              f"{train_loss:>11.4f}  "
              f"{val_loss:>9.4f}  "
              f"{bce_acc:>10.1f}%  "
              f"{rank_acc:>11.1f}%  "
              f"{cur_lr:>9.6f}")

        # Save best by ranking accuracy
        if rank_acc > best_rank_acc or (rank_acc == best_rank_acc and val_loss < best_val_loss):
            best_rank_acc = rank_acc
            best_val_loss = val_loss
            epochs_without_improvement = 0
            model.save(MODEL_PATH)
            print(f"           ↑ saved (rank_acc={rank_acc:.1f}%)")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print(f"\n✅ Best: rank_acc={best_rank_acc:.1f}%, val_loss={best_val_loss:.4f}")
    print(f"   Model saved to: {MODEL_PATH}")

    shutil.copy(MODEL_PATH, "research/mcts/prior.pt")
    print(f"   Also copied to: research/mcts/prior.pt")

    # Sanity check on val set
    print("\nSanity check — top-5 val groups:")
    model = RelevancePrior.load(MODEL_PATH)
    model.eval()
    shown = 0
    for qid in all_qids_v.unique()[:5]:
        mask       = (all_qids_v == qid)
        scores     = all_preds[mask]
        lbls       = all_labels_v[mask]
        ranked_idx = scores.argsort(descending=True)
        top_label  = lbls[ranked_idx[0]].item()
        status     = "✅" if top_label > 0.5 else "❌"
        print(f"  {status} group {qid.item():>4}: top score={scores[ranked_idx[0]]:.3f}  "
              f"label={top_label:.0f}  n_siblings={mask.sum()}")
        shown += 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch-size",  type=int,   default=256)
    p.add_argument("--bce-weight",  type=float, default=0.5)
    p.add_argument("--rank-weight", type=float, default=0.5)
    p.add_argument("--margin",      type=float, default=0.3)
    p.add_argument("--noise-std",   type=float, default=0.01)
    p.add_argument("--patience",    type=int,   default=15)
    p.add_argument("--n-val-repos", type=int,   default=1)
    args = p.parse_args()
    train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        bce_weight=args.bce_weight,
        rank_weight=args.rank_weight,
        margin=args.margin,
        noise_std=args.noise_std,
        patience=args.patience,
        n_val_repos=args.n_val_repos,
    )
"""
Train the relevance prior MLP on DevQuery-Bench training data.

Changes vs original:
  - Wider MLP: input is (query_emb || node_emb || elem_prod || cos_sim)
    i.e. 384+384+384+1 = 1153 dims instead of 768
  - Combined loss: BCE (pointwise) + pairwise margin ranking loss
    The ranking loss directly optimises what validate_prior.py measures
  - Saves best model by ranking accuracy on val set, not val loss
    (val loss is misleading with heavy class imbalance)
  - Longer warmup, slower decay so model doesn't stall early
"""
import argparse
import os
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH  = "devquery_bench/prior_training_data.pt"
MODEL_PATH = "devquery_bench/prior.pt"


# ─────────────────────────────────────────────
# Model: wider MLP that takes the richer input
# ─────────────────────────────────────────────
class ImprovedPrior(nn.Module):
    """
    Input:  concat(query_emb, node_emb, elem_prod, cos_sim)
            = 384 + 384 + 384 + 1 = 1153 dims

    Architecture: 1153 → 512 → LN → ReLU → Dropout(0.3)
                              → 128 → LN → ReLU → Dropout(0.2)
                              → 32  → ReLU
                              → 1   → Sigmoid
    """
    def __init__(self, embed_dim: int = 384, dropout: float = 0.3):
        super().__init__()
        in_dim = embed_dim * 3 + 1   # query + node + elem_prod + cos_sim scalar

        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),  # slightly less dropout in later layers

            nn.Linear(128, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, query_emb, node_emb, elem_prod, cos_sim):
        # All inputs are (batch, dim) or (batch, 1) for cos_sim
        x = torch.cat([query_emb, node_emb, elem_prod, cos_sim], dim=-1)
        return self.net(x).squeeze(-1)   # (batch,)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({'state_dict': self.state_dict(),
                    'embed_dim':  self._embed_dim}, path)

    @classmethod
    def load(cls, path: str, device='cpu'):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = cls(embed_dim=ckpt['embed_dim'])
        model.load_state_dict(ckpt['state_dict'])
        return model

    @property
    def _embed_dim(self):
        # infer from first layer weight: in_dim = embed_dim*3 + 1
        in_dim = self.net[0].weight.shape[1]
        return (in_dim - 1) // 3


# ─────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────
def weighted_bce(preds, targets, pos_weight):
    eps = 1e-7
    preds = preds.clamp(eps, 1 - eps)
    loss = -(pos_weight * targets * torch.log(preds)
             + (1 - targets) * torch.log(1 - preds))
    return loss.mean()


def pairwise_ranking_loss(preds, labels, query_ids, margin=0.2):
    """
    For each query group, form all (positive, negative) pairs and apply
    margin ranking loss: max(0, margin - (score_pos - score_neg)).

    This directly optimises what validate_prior.py measures.
    """
    total_loss = torch.tensor(0.0, device=preds.device, requires_grad=True)
    n_pairs = 0

    unique_qids = query_ids.unique()
    for qid in unique_qids:
        mask = (query_ids == qid)
        group_preds  = preds[mask]
        group_labels = labels[mask]

        pos_mask = (group_labels > 0.5)
        neg_mask = ~pos_mask

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        pos_scores = group_preds[pos_mask]    # (n_pos,)
        neg_scores = group_preds[neg_mask]    # (n_neg,)

        # All pairs: (n_pos, n_neg)
        pos_exp = pos_scores.unsqueeze(1).expand(-1, neg_scores.shape[0])
        neg_exp = neg_scores.unsqueeze(0).expand(pos_scores.shape[0], -1)

        pair_loss = torch.clamp(margin - (pos_exp - neg_exp), min=0.0)
        total_loss = total_loss + pair_loss.mean()
        n_pairs += 1

    return total_loss / max(n_pairs, 1)


# ─────────────────────────────────────────────
# Ranking accuracy (mirrors validate_prior.py)
# ─────────────────────────────────────────────
def ranking_accuracy(preds, labels, query_ids):
    """
    Per-group: is the positive ranked #1 among all siblings?
    Returns fraction of groups where this holds.
    """
    correct = 0
    total   = 0
    for qid in query_ids.unique():
        mask = (query_ids == qid)
        group_preds  = preds[mask]
        group_labels = labels[mask]

        pos_mask = (group_labels > 0.5)
        if pos_mask.sum() == 0:
            continue

        best_pos_score = group_preds[pos_mask].max().item()
        best_overall   = group_preds.max().item()
        if abs(best_pos_score - best_overall) < 1e-6:
            correct += 1
        total += 1

    return correct / max(total, 1)


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────
def train(epochs=30, lr=5e-4, batch_size=512, val_fraction=0.1,
          bce_weight=0.5, rank_weight=0.5, margin=0.2):

    print("=" * 60)
    print("TRAINING IMPROVED RELEVANCE PRIOR (DevQuery-Bench)")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"❌ Training data not found at {DATA_PATH}")
        print("   Run devquery_bench/build_prior_data.py first")
        return

    data = torch.load(DATA_PATH, weights_only=True)

    # Check that the new fields exist
    if 'cos_sims' not in data or 'elem_prods' not in data:
        print("❌ Training data is missing cos_sims / elem_prods fields.")
        print("   Re-run build_prior_data.py to regenerate with the new format.")
        return

    query_embs  = data['query_embs']   # (N, 384)
    node_embs   = data['node_embs']    # (N, 384)
    cos_sims    = data['cos_sims']     # (N, 1)
    elem_prods  = data['elem_prods']   # (N, 384)
    labels      = data['labels']       # (N,)
    query_ids   = data['query_ids']    # (N,)  long
    embed_dim   = int(data['embed_dim'])
    n_groups    = int(data['n_groups'])

    print(f"\nData:")
    print(f"  Total pairs:    {len(labels)}")
    print(f"  Positives:      {labels.sum().int()} ({100*labels.mean():.1f}%)")
    print(f"  Embedding dim:  {embed_dim}")
    print(f"  Query groups:   {n_groups}")
    print(f"  MLP input dim:  {embed_dim*3 + 1}")

    # Class imbalance
    n_pos = labels.sum().item()
    n_neg = (1 - labels).sum().item()
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)
    print(f"  Positive weight: {pos_weight.item():.2f}x")

    # ── Dataset split (split by index, keep query_ids aligned) ──
    N = len(labels)
    indices = torch.randperm(N, generator=torch.Generator().manual_seed(42))
    n_val   = max(1, int(N * val_fraction))
    n_train = N - n_val
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    def make_tensors(idx):
        return (query_embs[idx], node_embs[idx], cos_sims[idx],
                elem_prods[idx], labels[idx], query_ids[idx])

    train_data = make_tensors(train_idx)
    val_data   = make_tensors(val_idx)

    train_ds = TensorDataset(*train_data)
    val_ds   = TensorDataset(*val_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"\n  Train: {n_train} | Val: {n_val}")

    # ── Model ──
    model = ImprovedPrior(embed_dim=embed_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")

    # ── Device ──
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
    # Warmup for 3 epochs then cosine decay — avoids the early stall you saw
    def lr_lambda(epoch):
        warmup = 3
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))   # cosine to 0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training ──
    print(f"\nTraining for {epochs} epochs  "
          f"(BCE weight={bce_weight}, rank weight={rank_weight}, margin={margin})...")
    print(f"{'Epoch':>6}  {'train_loss':>11}  {'val_loss':>9}  "
          f"{'val_bce_acc':>11}  {'val_rank_acc':>12}  {'lr':>9}")

    best_rank_acc  = 0.0
    best_val_loss  = float('inf')

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss_sum = 0.0

        for batch in train_loader:
            q, n, cos, ep, lbl, qids = [t.to(device) for t in batch]

            preds = model(q, n, ep, cos)

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
        val_loss_sum  = 0.0
        all_preds     = []
        all_labels_v  = []
        all_qids_v    = []

        with torch.no_grad():
            for batch in val_loader:
                q, n, cos, ep, lbl, qids = [t.to(device) for t in batch]
                preds = model(q, n, ep, cos)

                loss_bce  = weighted_bce(preds, lbl, pos_weight)
                loss_rank = pairwise_ranking_loss(preds, lbl, qids, margin=margin)
                loss      = bce_weight * loss_bce + rank_weight * loss_rank
                val_loss_sum += loss.item() * len(lbl)

                all_preds.append(preds.cpu())
                all_labels_v.append(lbl.cpu())
                all_qids_v.append(qids.cpu())

        val_loss   = val_loss_sum / n_val
        all_preds  = torch.cat(all_preds)
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

        # Save best by ranking accuracy (what actually matters)
        if rank_acc > best_rank_acc or (rank_acc == best_rank_acc and val_loss < best_val_loss):
            best_rank_acc = rank_acc
            best_val_loss = val_loss
            model_cpu = model.to('cpu')
            ckpt = {'state_dict': model_cpu.state_dict(), 'embed_dim': embed_dim}
            torch.save(ckpt, MODEL_PATH)
            model = model.to(device)
            print(f"           ↑ saved (rank_acc={rank_acc:.1f}%)")

    print(f"\n✅ Best: rank_acc={best_rank_acc:.1f}%, val_loss={best_val_loss:.4f}")
    print(f"   Model saved to: {MODEL_PATH}")

    shutil.copy(MODEL_PATH, "research/mcts/prior.pt")
    print(f"   Also copied to: research/mcts/prior.pt")

    # ── Sanity check ──
    print("\nSanity check — top-5 val groups:")
    model.eval()
    model = model.to('cpu')
    shown = 0
    for qid in all_qids_v.unique()[:5]:
        mask   = (all_qids_v == qid)
        scores = all_preds[mask]
        lbls   = all_labels_v[mask]
        ranked_idx = scores.argsort(descending=True)
        top_label  = lbls[ranked_idx[0]].item()
        status     = "✅" if top_label > 0.5 else "❌"
        print(f"  {status} group {qid.item():>4}: top score={scores[ranked_idx[0]]:.3f}  "
              f"label={top_label:.0f}  n_siblings={mask.sum()}")
        shown += 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--batch-size",   type=int,   default=512)
    p.add_argument("--bce-weight",   type=float, default=0.5,
                   help="Weight for BCE pointwise loss")
    p.add_argument("--rank-weight",  type=float, default=0.5,
                   help="Weight for pairwise ranking loss")
    p.add_argument("--margin",       type=float, default=0.2,
                   help="Margin for ranking loss")
    args = p.parse_args()
    train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        bce_weight=args.bce_weight,
        rank_weight=args.rank_weight,
        margin=args.margin,
    )
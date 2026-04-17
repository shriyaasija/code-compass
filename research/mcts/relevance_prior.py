"""
Bottleneck Relevance Prior for PUCT-guided MCTS.

Architecture: Projects 384-d embeddings to 32-d via learned projections,
then uses a small head. Total ~31K params (10x smaller than previous 312K).

The bottleneck prevents memorisation of repo-specific embedding patterns,
forcing the model to learn generalizable relevance features.

No cosine similarity feature — naturalistic queries have intentionally low
cosine with code summaries (high alpha). The bottleneck learns a task-specific
projection that captures relevance differently from raw cosine.

Interface:
    forward(query_emb, node_emb) -> score in [0, 1]
    score_children(query_emb, children_dicts) -> np.ndarray of scores
    online_update(query_emb, visited_nodes, visit_counts)
    save(path) / load(path)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List


class RelevancePrior(nn.Module):
    """
    Bottleneck prior for PUCT-guided MCTS tree search.

    Input:  query_emb (384-d), node_emb (384-d)
    Output: relevance score in [0, 1]

    Architecture:
        query_emb (384) -> Linear(384, 32) -> ReLU -> q  (32)
        node_emb  (384) -> Linear(384, 32) -> ReLU -> n  (32)
        interaction = q * n                            (32)
        concat(q, n, interaction) = 96
        -> Linear(96, 64) -> LayerNorm -> GELU -> Dropout(0.5)
        -> Linear(64, 1)  -> Sigmoid

    Total params: ~31K (vs 312K before)
    """

    def __init__(self, embed_dim: int = 384, proj_dim: int = 32, dropout: float = 0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim

        # Bottleneck projections — force generalisation
        self.query_proj = nn.Linear(embed_dim, proj_dim)
        self.node_proj = nn.Linear(embed_dim, proj_dim)

        # Head: proj_query + proj_node + proj_interaction = proj_dim * 3
        head_in = proj_dim * 3
        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        query_emb: torch.Tensor,
        node_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_emb: shape (D,) or (N, D)
            node_emb:  shape (D,) or (N, D)
        Returns:
            scalar or (N,) tensor of prior probabilities in [0, 1]
        """
        # Handle both single and batch inputs
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        if node_emb.dim() == 1:
            node_emb = node_emb.unsqueeze(0)

        # If query is (1, D) and node is (N, D), broadcast query
        if query_emb.shape[0] == 1 and node_emb.shape[0] > 1:
            query_emb = query_emb.expand(node_emb.shape[0], -1)

        # Project to bottleneck dimension
        q = F.relu(self.query_proj(query_emb))
        n = F.relu(self.node_proj(node_emb))
        interaction = q * n  # element-wise in projected space

        x = torch.cat([q, n, interaction], dim=-1)
        out = self.head(x).squeeze(-1)

        # If original input was 1D, return scalar
        if out.shape[0] == 1:
            return out.squeeze(0)
        return out

    def score_children(
        self,
        query_emb: np.ndarray,
        children: List[dict],
    ) -> np.ndarray:
        """
        Score a list of tree node dicts for a given query embedding.
        Returns numpy array of prior scores, shape (N,).
        Used during MCTS expansion.
        """
        scores = []
        q_tensor = torch.tensor(query_emb, dtype=torch.float32)

        for child in children:
            node_emb = child.get('embedding')
            if node_emb is None:
                # No embedding: assign a uniform prior of 0.5
                scores.append(0.5)
                continue
            n_tensor = torch.tensor(node_emb, dtype=torch.float32)
            with torch.no_grad():
                score = self.forward(q_tensor, n_tensor)
            scores.append(float(score))

        return np.array(scores, dtype=np.float32)

    def online_update(
        self,
        query_emb: np.ndarray,
        visited_nodes: List[dict],
        visit_counts: List[int],
        lr: float = 5e-4,
    ):
        """
        One gradient step using MCTS visit counts as a soft supervision signal.
        Called after each query completes.

        visited_nodes: list of tree node dicts that were visited during search
        visit_counts:  corresponding visit counts

        Nodes with high visit counts acted as good intermediaries — treat them
        as positive examples. Nodes with zero visits are negative examples.
        """
        if not visited_nodes or sum(visit_counts) == 0:
            return

        q_tensor = torch.tensor(query_emb, dtype=torch.float32)
        visit_array = np.array(visit_counts, dtype=np.float32)

        # Normalize visit counts to get soft targets in [0, 1]
        targets = visit_array / (visit_array.max() + 1e-8)
        targets = torch.tensor(targets, dtype=torch.float32)

        # Build node embedding tensor
        node_embs = []
        valid_mask = []
        for node in visited_nodes:
            emb = node.get('embedding')
            if emb is not None:
                node_embs.append(emb)
                valid_mask.append(True)
            else:
                node_embs.append(np.zeros(self.embed_dim, dtype=np.float32))
                valid_mask.append(False)

        node_embs_tensor = torch.tensor(np.array(node_embs), dtype=torch.float32)
        valid_tensor = torch.tensor(valid_mask, dtype=torch.bool)

        if not valid_tensor.any():
            return

        node_embs_tensor = node_embs_tensor[valid_tensor]
        targets = targets[valid_tensor]
        q_expanded = q_tensor.unsqueeze(0).expand(node_embs_tensor.shape[0], -1)

        preds = self.forward(q_expanded, node_embs_tensor)
        loss = nn.BCELoss()(preds, targets)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

    def save(self, path: str):
        """Save model weights and config."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'embed_dim': self.embed_dim,
            'proj_dim': self.proj_dim,
        }, path)

    @classmethod
    def load(cls, path: str, device='cpu') -> 'RelevancePrior':
        """Load model from saved checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(
            embed_dim=checkpoint['embed_dim'],
            proj_dim=checkpoint.get('proj_dim', 32),
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model
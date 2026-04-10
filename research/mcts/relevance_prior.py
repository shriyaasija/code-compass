import math
import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple


class RelevancePrior(nn.Module):
    """
    Lightweight relevance prior for PUCT-guided MCTS.
   
    This replaces the LLM for node scoring at internal tree nodes.
    The LLM is only called at leaf nodes (functions/methods) for final verification.
    """

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        input_dim = embed_dim * 2  # query + node embeddings concatenated

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Store embed_dim for inference
        self.embed_dim = embed_dim

        # Initialize weights with Xavier for stable training
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

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
            scalar or (N,) tensor of prior probabilities
        """
        # Handle both single and batch inputs
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        if node_emb.dim() == 1:
            node_emb = node_emb.unsqueeze(0)

        # If query is (1, D) and node is (N, D), broadcast query
        if query_emb.shape[0] == 1 and node_emb.shape[0] > 1:
            query_emb = query_emb.expand(node_emb.shape[0], -1)

        x = torch.cat([query_emb, node_emb], dim=-1)
        out = self.net(x).squeeze(-1)

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
        # Top-visited nodes get target close to 1, unvisited get target 0
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
                # Placeholder for nodes without embeddings — will be masked
                node_embs.append(np.zeros(self.embed_dim, dtype=np.float32))
                valid_mask.append(False)

        node_embs_tensor = torch.tensor(np.array(node_embs), dtype=torch.float32)
        valid_tensor = torch.tensor(valid_mask, dtype=torch.bool)

        if not valid_tensor.any():
            return

        # Only update on nodes that have embeddings
        node_embs_tensor = node_embs_tensor[valid_tensor]
        targets = targets[valid_tensor]
        q_expanded = q_tensor.unsqueeze(0).expand(node_embs_tensor.shape[0], -1)

        # Forward pass
        preds = self.forward(q_expanded, node_embs_tensor)

        # BCE loss
        loss = nn.BCELoss()(preds, targets)

        # Single gradient step — we DON'T want to overfit to one query
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent large updates from unusual queries
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

    def save(self, path: str):
        """Save model weights and config."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'embed_dim': self.embed_dim,
        }, path)
        print(f"Prior saved: {path}")

    @classmethod
    def load(cls, path: str) -> 'RelevancePrior':
        """Load model from saved checkpoint."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = cls(embed_dim=checkpoint['embed_dim'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model
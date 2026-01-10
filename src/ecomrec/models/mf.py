"""PyTorch Bayesian Personalized Ranking matrix factorization."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 32) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (
            (self.user_emb(users) * self.item_emb(items)).sum(dim=-1)
            + self.user_bias(users).squeeze(-1)
            + self.item_bias(items).squeeze(-1)
        )

    def bpr_loss(self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        pos_scores = self.score(users, pos)
        neg_scores = self.score(users, neg)
        return -F.logsigmoid(pos_scores - neg_scores).mean()

    def user_item_scores(self, user_idx: int) -> torch.Tensor:
        u = self.user_emb.weight[user_idx]
        return self.item_emb.weight @ u + self.item_bias.weight.squeeze(-1) + self.user_bias.weight[user_idx]

    def item_similarity(self, item_idx: int) -> torch.Tensor:
        q = self.item_emb.weight[item_idx]
        return self.item_emb.weight @ q

"""Recall@K / NDCG@K / HitRate@K versus a popularity baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ecomrec.models.mf import BPRMF


def _dcg(ranks: np.ndarray) -> float:
    return float(np.sum(1.0 / np.log2(ranks + 1.0)))


def _metrics_from_ranks(ranks: list[int], k: int) -> dict[str, float]:
    if not ranks:
        return {"recall": 0.0, "ndcg": 0.0, "hit_rate": 0.0, "n_eval_users": 0}
    hits = np.array([r <= k for r in ranks], dtype=np.float64)
    ndcgs = []
    for r in ranks:
        if r <= k:
            ndcgs.append(_dcg(np.array([r], dtype=np.float64)) / 1.0)
        else:
            ndcgs.append(0.0)
    return {
        "recall": float(hits.mean()),
        "ndcg": float(np.mean(ndcgs)),
        "hit_rate": float(hits.mean()),
        "n_eval_users": int(len(ranks)),
    }


@torch.no_grad()
def evaluate_model(
    model: BPRMF,
    test: pd.DataFrame,
    user_pos: dict[int, set[int]],
    n_items: int,
    k: int = 10,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Leave-one-out ranking: hide train items, rank the held-out positive."""
    model.eval()
    device = device or torch.device("cpu")
    model = model.to(device)
    item_emb = model.item_emb.weight
    item_bias = model.item_bias.weight.squeeze(-1)
    ranks: list[int] = []
    recommended: set[int] = set()
    grouped = test.groupby("user_idx")["item_idx"].apply(list)
    for user_idx, pos_items in grouped.items():
        u = int(user_idx)
        target = int(pos_items[0])
        scores = item_emb @ model.user_emb.weight[u] + item_bias + model.user_bias.weight[u].squeeze(-1)
        seen = user_pos.get(u, set())
        for s in seen:
            scores[s] = -1e9
        order = torch.argsort(scores, descending=True)
        rank = int((order == target).nonzero(as_tuple=True)[0].item()) + 1
        ranks.append(rank)
        top = order[:k].detach().cpu().tolist()
        recommended.update(int(i) for i in top)
    metrics = _metrics_from_ranks(ranks, k)
    metrics["coverage"] = float(len(recommended) / max(n_items, 1))
    return metrics


def popularity_ranks(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_items: int,
    k: int = 10,
) -> dict[str, float]:
    pop = np.zeros(n_items, dtype=np.float64)
    for i, w in zip(train["item_idx"].to_numpy(), train["weight"].to_numpy()):
        pop[int(i)] += float(w)
    seen = train.groupby("user_idx")["item_idx"].apply(lambda s: set(int(x) for x in s)).to_dict()
    ranks: list[int] = []
    recommended: set[int] = set()
    grouped = test.groupby("user_idx")["item_idx"].apply(list)
    for user_idx, pos_items in grouped.items():
        u = int(user_idx)
        target = int(pos_items[0])
        scores = pop.copy()
        for s in seen.get(u, set()):
            scores[s] = -1e18
        order = np.argsort(-scores)
        rank = int(np.where(order == target)[0][0]) + 1
        ranks.append(rank)
        recommended.update(int(i) for i in order[:k])
    metrics = _metrics_from_ranks(ranks, k)
    metrics["coverage"] = float(len(recommended) / max(n_items, 1))
    return metrics

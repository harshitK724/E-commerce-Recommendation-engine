"""Train BPR-MF and export embeddings."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ecomrec.config import Settings, settings as default_settings
from ecomrec.models.evaluate import evaluate_model, popularity_ranks
from ecomrec.models.mf import BPRMF
from ecomrec.tables import read_df, write_df


class BPRPairDataset(Dataset):
    def __init__(
        self,
        user_idx: np.ndarray,
        item_idx: np.ndarray,
        n_items: int,
        user_pos: dict[int, set[int]],
        n_neg: int = 4,
        seed: int = 0,
    ) -> None:
        self.users = user_idx.astype(np.int64)
        self.items = item_idx.astype(np.int64)
        self.n_items = n_items
        self.user_pos = user_pos
        self.n_neg = n_neg
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, i: int) -> tuple[int, int, int]:
        u = int(self.users[i])
        pos = int(self.items[i])
        seen = self.user_pos[u]
        neg = int(self.rng.integers(0, self.n_items))
        while neg in seen:
            neg = int(self.rng.integers(0, self.n_items))
        return u, pos, neg


def _user_pos_sets(train: pd.DataFrame) -> dict[int, set[int]]:
    pos: dict[int, set[int]] = defaultdict(set)
    for u, i in zip(train["user_idx"].tolist(), train["item_idx"].tolist()):
        pos[int(u)].add(int(i))
    return pos


def load_splits(processed_dir: Path) -> dict[str, pd.DataFrame]:
    return {name: read_df(processed_dir / f"{name}.pkl") for name in ("train", "val", "test")}


def train_bpr(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    cfg: Settings | None = None,
    val: pd.DataFrame | None = None,
) -> tuple[BPRMF, list[float]]:
    cfg = cfg or default_settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_pos = _user_pos_sets(train)
    dataset = BPRPairDataset(
        train["user_idx"].to_numpy(),
        train["item_idx"].to_numpy(),
        n_items=n_items,
        user_pos=user_pos,
        n_neg=cfg.negatives,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    model = BPRMF(n_users, n_items, dim=cfg.embedding_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    losses: list[float] = []
    best_state = None
    best_recall = -1.0
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for users, pos, neg in loader:
            users = users.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            opt.zero_grad()
            loss = model.bpr_loss(users, pos, neg)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
        epoch_loss = running / max(n_batches, 1)
        losses.append(epoch_loss)
        print(f"epoch {epoch + 1}/{cfg.epochs} loss={epoch_loss:.4f}")
        if val is not None and len(val):
            metrics = evaluate_model(model, val, user_pos, n_items, k=cfg.eval_k, device=device)
            if metrics["recall"] > best_recall:
                best_recall = metrics["recall"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), losses


def export_artifacts(
    model: BPRMF,
    processed_dir: Path,
    artifacts_dir: Path,
    train: pd.DataFrame,
    metrics: dict | None = None,
) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "n_users": model.n_users,
        "n_items": model.n_items,
        "dim": model.dim,
    }
    path = artifacts_dir / "mf.pt"
    torch.save(ckpt, path)
    for name in ("user_map", "item_map", "catalog"):
        src = processed_dir / f"{name}.pkl"
        if src.exists():
            write_df(read_df(src), artifacts_dir / f"{name}.pkl")
    seen = train.groupby("user_idx")["item_idx"].apply(lambda s: [int(x) for x in s]).to_dict()
    pop = (
        train.groupby("item_idx")["weight"]
        .sum()
        .rename("score")
        .reset_index()
        .sort_values("score", ascending=False)
    )
    write_df(pop, artifacts_dir / "popularity.pkl")
    torch.save(seen, artifacts_dir / "seen.pt")
    if metrics is not None:
        (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def run_training(cfg: Settings | None = None) -> dict:
    cfg = cfg or default_settings
    splits = load_splits(cfg.processed_dir)
    user_map = read_df(cfg.processed_dir / "user_map.pkl")
    item_map = read_df(cfg.processed_dir / "item_map.pkl")
    n_users, n_items = len(user_map), len(item_map)
    model, losses = train_bpr(splits["train"], n_users, n_items, cfg, val=splits["val"])
    user_pos = _user_pos_sets(splits["train"])
    device = torch.device("cpu")
    mf_metrics = evaluate_model(model, splits["test"], user_pos, n_items, k=cfg.eval_k, device=device)
    pop_metrics = popularity_ranks(splits["train"], splits["test"], n_items, k=cfg.eval_k)
    metrics = {
        "mf": mf_metrics,
        "popularity": pop_metrics,
        "train_loss": losses,
        "n_users": n_users,
        "n_items": n_items,
        "beats_popularity": mf_metrics["recall"] >= pop_metrics["recall"],
    }
    export_artifacts(model, cfg.processed_dir, cfg.artifacts_dir, splits["train"], metrics)
    return metrics

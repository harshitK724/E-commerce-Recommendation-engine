"""In-process recommendation service used by FastAPI and MCP."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch

from ecomrec.config import Settings, settings as default_settings
from ecomrec.models.mf import BPRMF
from ecomrec.serving.db import CatalogStore, FileCatalogStore, PostgresCatalogStore
from ecomrec.tables import read_df

REASON_COLLAB = "collaborative"
REASON_POP = "popularity_fallback"


@dataclass
class RecItem:
    product_id: int
    score: float
    title: str | None = None
    brand: str | None = None
    category_code: str | None = None
    price: float | None = None
    reason: str = REASON_COLLAB

    def as_dict(self) -> dict:
        return {
            "product_id": int(self.product_id),
            "score": round(float(self.score), 6),
            "title": self.title,
            "brand": self.brand,
            "category_code": self.category_code,
            "price": self.price,
            "reason": self.reason,
        }


@dataclass
class RecommendationService:
    model: BPRMF
    user_map: pd.DataFrame
    item_map: pd.DataFrame
    catalog: pd.DataFrame
    popularity: pd.DataFrame
    seen: dict[int, list[int]]
    store: CatalogStore | None = None
    user_to_idx: dict[int, int] = field(init=False)
    item_to_idx: dict[int, int] = field(init=False)
    idx_to_item: dict[int, int] = field(init=False)
    catalog_by_id: dict[int, dict] = field(init=False)

    def __post_init__(self) -> None:
        self.user_to_idx = dict(
            zip(self.user_map["user_id"].astype(int), self.user_map["user_idx"].astype(int))
        )
        self.item_to_idx = dict(
            zip(self.item_map["product_id"].astype(int), self.item_map["item_idx"].astype(int))
        )
        self.idx_to_item = {v: k for k, v in self.item_to_idx.items()}
        self.catalog_by_id = {}
        for row in self.catalog.to_dict("records"):
            pid = int(row["product_id"])
            cleaned = {
                key: value
                for key, value in row.items()
                if value is not None and not (isinstance(value, float) and pd.isna(value))
            }
            self.catalog_by_id[pid] = cleaned
        self.model.eval()

    @classmethod
    def load(cls, cfg: Settings | None = None, store: CatalogStore | None = None) -> "RecommendationService":
        cfg = cfg or default_settings
        artifacts = cfg.artifacts_dir
        ckpt = torch.load(artifacts / "mf.pt", map_location="cpu", weights_only=False)
        model = BPRMF(ckpt["n_users"], ckpt["n_items"], dim=ckpt["dim"])
        model.load_state_dict(ckpt["state_dict"])
        if store is None:
            store = _default_store(cfg)
        return cls(
            model=model,
            user_map=read_df(artifacts / "user_map.pkl"),
            item_map=read_df(artifacts / "item_map.pkl"),
            catalog=read_df(artifacts / "catalog.pkl"),
            popularity=read_df(artifacts / "popularity.pkl"),
            seen=torch.load(artifacts / "seen.pt", map_location="cpu", weights_only=False),
            store=store,
        )

    def _decorate(self, product_id: int, score: float, reason: str) -> RecItem:
        meta = self.catalog_by_id.get(int(product_id), {})
        return RecItem(
            product_id=int(product_id),
            score=float(score),
            title=meta.get("title"),
            brand=meta.get("brand"),
            category_code=meta.get("category_code"),
            price=float(meta["price"]) if meta.get("price") is not None else None,
            reason=reason,
        )

    def _topk_from_scores(
        self,
        scores: torch.Tensor,
        k: int,
        exclude: set[int],
        reason: str,
    ) -> list[RecItem]:
        masked = scores.clone()
        for idx in exclude:
            if 0 <= idx < masked.numel():
                masked[idx] = -1e9
        k = min(k, int(masked.numel()))
        values, indices = torch.topk(masked, k)
        pairs = sorted(
            zip(values.tolist(), indices.tolist()),
            key=lambda t: (-t[0], self.idx_to_item.get(int(t[1]), 0)),
        )
        items: list[RecItem] = []
        for score, idx in pairs:
            pid = self.idx_to_item.get(int(idx))
            if pid is None:
                continue
            items.append(self._decorate(pid, score, reason))
        return items

    def recommend(self, user_id: int, k: int = 5, exclude_seen: bool = True) -> list[RecItem]:
        k = int(max(1, min(k, 20)))
        uid = int(user_id)
        if uid not in self.user_to_idx:
            return self.popular(k)
        uidx = self.user_to_idx[uid]
        with torch.no_grad():
            scores = self.model.user_item_scores(uidx)
        exclude: set[int] = set()
        if exclude_seen:
            exclude = set(self.seen.get(uidx, []))
        return self._topk_from_scores(scores, k, exclude, REASON_COLLAB)

    def popular(self, k: int = 5) -> list[RecItem]:
        k = int(max(1, min(k, 20)))
        rows = self.popularity.head(k)
        items: list[RecItem] = []
        for row in rows.itertuples(index=False):
            pid = self.idx_to_item.get(int(row.item_idx))
            if pid is None:
                continue
            items.append(self._decorate(pid, float(row.score), REASON_POP))
        return items

    def similar_items(self, product_id: int, k: int = 5) -> list[RecItem]:
        k = int(max(1, min(k, 20)))
        pid = int(product_id)
        if pid not in self.item_to_idx:
            return []
        idx = self.item_to_idx[pid]
        with torch.no_grad():
            scores = self.model.item_similarity(idx)
        return self._topk_from_scores(scores, k, {idx}, REASON_COLLAB)

    def history(self, user_id: int, n: int = 10) -> list[dict]:
        if self.store is not None:
            return self.store.user_history(int(user_id), n)
        return []

    def get_product(self, product_id: int) -> dict | None:
        if self.store is not None:
            found = self.store.get_product(int(product_id))
            if found is not None:
                return found
        meta = self.catalog_by_id.get(int(product_id))
        return dict(meta) if meta else None


def _default_store(cfg: Settings) -> CatalogStore:
    try:
        from ecomrec.data.ingest import postgres_available

        if postgres_available(cfg.database_url):
            return PostgresCatalogStore(cfg.database_url)
    except Exception:
        pass
    events = cfg.processed_dir / "user_events.pkl"
    catalog = cfg.artifacts_dir / "catalog.pkl"
    if not catalog.exists():
        catalog = cfg.processed_dir / "catalog.pkl"
    return FileCatalogStore(catalog, events if events.exists() else None)

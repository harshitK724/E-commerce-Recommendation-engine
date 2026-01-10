from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from ecomrec.config import Settings
from ecomrec.data.features import attach_indices, build_catalog, build_id_maps
from ecomrec.data.splits import time_leave_last_out
from ecomrec.data.synthetic import generate_synthetic_events
from ecomrec.models.infer import RecommendationService
from ecomrec.models.train import export_artifacts, train_bpr
from ecomrec.serving.db import MemoryCatalogStore
from ecomrec.tables import read_df, write_df


@pytest.fixture
def toy_events() -> pd.DataFrame:
    return generate_synthetic_events(n_rows=4_000, n_users=80, n_items=40, seed=0)


@pytest.fixture
def toy_settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")


def _prepare_splits(events: pd.DataFrame):
    from ecomrec.data.clean import pandas_transform

    interactions, stats = pandas_transform(events)
    user_map, item_map = build_id_maps(interactions)
    indexed = attach_indices(interactions, user_map, item_map)
    catalog = build_catalog(interactions)
    splits = time_leave_last_out(indexed)
    return interactions, user_map, item_map, catalog, splits, stats


@pytest.fixture
def toy_service(tmp_path: Path, toy_events: pd.DataFrame) -> RecommendationService:
    settings = Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        epochs=3,
        embedding_dim=8,
        batch_size=256,
        negatives=2,
    )
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    _, user_map, item_map, catalog, splits, _ = _prepare_splits(toy_events)
    for name, frame in splits.items():
        write_df(frame, settings.processed_dir / f"{name}.pkl")
    write_df(user_map, settings.processed_dir / "user_map.pkl")
    write_df(item_map, settings.processed_dir / "item_map.pkl")
    write_df(catalog, settings.processed_dir / "catalog.pkl")
    model, _ = train_bpr(
        splits["train"],
        n_users=len(user_map),
        n_items=len(item_map),
        cfg=settings,
        val=None,
    )
    export_artifacts(model, settings.processed_dir, settings.artifacts_dir, splits["train"], metrics=None)
    events = splits["train"][["user_id", "product_id", "weight"]].copy()
    store = MemoryCatalogStore(catalog, events)
    return RecommendationService(
        model=model,
        user_map=user_map,
        item_map=item_map,
        catalog=catalog,
        popularity=read_df(settings.artifacts_dir / "popularity.pkl"),
        seen=torch.load(settings.artifacts_dir / "seen.pt", map_location="cpu", weights_only=False),
        store=store,
    )

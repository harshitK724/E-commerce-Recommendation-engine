"""End-to-end ingest + SQL/Pandas transform + time-based splits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import psycopg

from ecomrec.config import Settings, settings as default_settings
from ecomrec.data.clean import pandas_transform
from ecomrec.data.features import attach_indices, build_catalog, build_id_maps
from ecomrec.data.ingest import (
    connect,
    copy_events_frame,
    create_tables,
    postgres_available,
    replace_serving_tables,
    truncate_raw_events,
)
from ecomrec.data.sql import PIPELINE_STATS_SQL, TRANSFORM_SQL
from ecomrec.data.splits import time_leave_last_out
from ecomrec.data.synthetic import generate_synthetic_events
from ecomrec.tables import write_df


def _ensure_dirs(cfg: Settings) -> None:
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)


def load_source_frame(csv_path: Path | None, cfg: Settings) -> pd.DataFrame:
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        if len(df) > cfg.sample_rows:
            df = df.sample(n=cfg.sample_rows, random_state=42)
        return df
    n_users = min(8_000, max(200, cfg.sample_rows // 150))
    n_items = min(3_000, max(80, cfg.sample_rows // 400))
    return generate_synthetic_events(
        n_rows=cfg.sample_rows,
        n_users=n_users,
        n_items=n_items,
        seed=42,
    )


def run_sql_transform(conn: psycopg.Connection, cfg: Settings) -> tuple[pd.DataFrame, dict]:
    sql = TRANSFORM_SQL.format(
        bot_events_per_hour=int(cfg.bot_events_per_hour),
        min_user_events=int(cfg.min_user_events),
        min_item_events=int(cfg.min_item_events),
    )
    with conn.cursor() as cur:
        for stmt in (chunk.strip() for chunk in sql.split(";") if chunk.strip()):
            cur.execute(stmt)
        cur.execute(PIPELINE_STATS_SQL)
        stats_row = cur.fetchone()
        stats_cols = [d.name for d in cur.description]
        cur.execute("SELECT * FROM interactions")
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
    conn.commit()
    stats = {k: int(v) for k, v in zip(stats_cols, stats_row or [])}
    stats["transform"] = "postgres"
    interactions = pd.DataFrame(rows, columns=cols)
    if "last_event_time" in interactions.columns:
        interactions["last_event_time"] = pd.to_datetime(interactions["last_event_time"], utc=True)
    return interactions, stats


def persist_splits(splits: dict[str, pd.DataFrame], processed_dir: Path) -> None:
    for name, frame in splits.items():
        write_df(frame, processed_dir / f"{name}.pkl")


def run_pipeline(
    csv_path: Path | None = None,
    *,
    events: pd.DataFrame | None = None,
    cfg: Settings | None = None,
    use_postgres: bool | None = None,
) -> dict:
    cfg = cfg or default_settings
    _ensure_dirs(cfg)
    source = events if events is not None else load_source_frame(csv_path or cfg.raw_csv_path, cfg)

    conn = None
    if use_postgres is None:
        use_postgres = postgres_available(cfg.database_url)
    if use_postgres:
        conn = connect(cfg.database_url)
        create_tables(conn)
        truncate_raw_events(conn)
        copy_events_frame(conn, source)
        interactions, stats = run_sql_transform(conn, cfg)
    else:
        interactions, stats = pandas_transform(source, cfg)

    if interactions.empty:
        raise ValueError("Pipeline produced no interactions. Relax filters or add more events.")

    user_map, item_map = build_id_maps(interactions)
    indexed = attach_indices(interactions, user_map, item_map)
    catalog = build_catalog(interactions)
    splits = time_leave_last_out(indexed)

    processed = cfg.processed_dir
    persist_splits(splits, processed)
    write_df(indexed, processed / "interactions.pkl")
    write_df(user_map, processed / "user_map.pkl")
    write_df(item_map, processed / "item_map.pkl")
    write_df(catalog, processed / "catalog.pkl")

    history_cols = [c for c in ("user_id", "product_id", "weight", "last_event_time") if c in indexed.columns]
    history = indexed[history_cols].rename(columns={"last_event_time": "event_time"}).copy()
    history["event_type"] = history["weight"].map({1: "view", 3: "cart", 5: "purchase"})
    write_df(history, processed / "user_events.pkl")

    if conn is not None:
        replace_serving_tables(conn, catalog, history)
        conn.close()

    stats.update(
        {
            "train_rows": int(len(splits["train"])),
            "val_rows": int(len(splits["val"])),
            "test_rows": int(len(splits["test"])),
            "n_users_mapped": int(len(user_map)),
            "n_items_mapped": int(len(item_map)),
        }
    )
    (cfg.artifacts_dir / "pipeline_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats

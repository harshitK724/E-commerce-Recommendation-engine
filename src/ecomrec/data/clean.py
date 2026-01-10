"""Pandas cleansing and implicit-weight aggregation."""

from __future__ import annotations

import pandas as pd

from ecomrec.config import Settings, settings as default_settings

EVENT_WEIGHTS = {"view": 1, "cart": 3, "purchase": 5}


def validate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    required = {"user_id", "product_id", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"interactions missing columns: {sorted(missing)}")
    if df[["user_id", "product_id", "weight"]].isna().any().any():
        raise ValueError("interactions contain null user_id, product_id, or weight")
    return df


def normalize_events(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["event_time"] = pd.to_datetime(out["event_time"], utc=True, errors="coerce")
    out["event_type"] = out["event_type"].astype(str).str.lower().str.strip()
    out["product_id"] = pd.to_numeric(out["product_id"], errors="coerce")
    out["user_id"] = pd.to_numeric(out["user_id"], errors="coerce")
    if "price" in out.columns:
        out["price"] = pd.to_numeric(out["price"], errors="coerce")
    if "category_id" in out.columns:
        out["category_id"] = pd.to_numeric(out["category_id"], errors="coerce")
    for col in ("brand", "category_code", "user_session"):
        if col in out.columns:
            out[col] = out[col].astype("string")
    return out


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df["user_id"].notna()
        & df["product_id"].notna()
        & df["event_time"].notna()
        & df["event_type"].isin(EVENT_WEIGHTS)
    )
    out = df.loc[mask].copy()
    out["user_id"] = out["user_id"].astype("int64")
    out["product_id"] = out["product_id"].astype("int64")
    return out.reset_index(drop=True)


def filter_bots(df: pd.DataFrame, events_per_hour: int) -> tuple[pd.DataFrame, pd.Index]:
    hour = df["event_time"].dt.floor("h")
    counts = df.groupby(["user_id", hour], sort=False).size().rename("n")
    bots = counts[counts > events_per_hour].index.get_level_values(0).unique()
    return df.loc[~df["user_id"].isin(bots)].reset_index(drop=True), bots


def filter_inactive(
    df: pd.DataFrame,
    min_user_events: int,
    min_item_events: int,
) -> pd.DataFrame:
    user_n = df.groupby("user_id").size()
    item_n = df.groupby("product_id").size()
    keep_users = user_n[user_n >= min_user_events].index
    keep_items = item_n[item_n >= min_item_events].index
    return df[df["user_id"].isin(keep_users) & df["product_id"].isin(keep_items)].reset_index(
        drop=True
    )


def aggregate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["weight"] = tmp["event_type"].map(EVENT_WEIGHTS).astype("int64")
    agg: dict[str, str] = {"weight": "max", "event_time": "max"}
    for col in ("category_id", "category_code", "brand", "price"):
        if col in tmp.columns:
            agg[col] = "last"
    grouped = tmp.sort_values("event_time").groupby(["user_id", "product_id"], as_index=False).agg(agg)
    grouped = grouped.rename(columns={"event_time": "last_event_time"})
    validate_interactions(grouped)
    return grouped


def pandas_transform(df: pd.DataFrame, cfg: Settings | None = None) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or default_settings
    raw_rows = int(len(df))
    events = drop_invalid_rows(normalize_events(df))
    events, bots = filter_bots(events, cfg.bot_events_per_hour)
    events = filter_inactive(events, cfg.min_user_events, cfg.min_item_events)
    interactions = aggregate_interactions(events)
    stats = {
        "raw_rows": raw_rows,
        "clean_rows": int(len(drop_invalid_rows(normalize_events(df)))),
        "bot_users": int(len(bots)),
        "interaction_rows": int(len(interactions)),
        "n_users": int(interactions["user_id"].nunique()) if len(interactions) else 0,
        "n_items": int(interactions["product_id"].nunique()) if len(interactions) else 0,
        "transform": "pandas",
    }
    return interactions, stats

"""Time-based leave-last-out splits."""

from __future__ import annotations

import pandas as pd


def time_leave_last_out(indexed: pd.DataFrame, min_events: int = 3) -> dict[str, pd.DataFrame]:
    """Last event -> test, second-last -> val, remainder -> train.

    Users with fewer than `min_events` interactions stay in train only so
    evaluation is not scored on users with no history.
    """
    ordered = indexed.sort_values(["user_idx", "last_event_time", "product_id"]).copy()
    ordered["rev_rank"] = ordered.groupby("user_idx").cumcount(ascending=False)
    counts = ordered.groupby("user_idx")["product_id"].transform("size")

    test_mask = (ordered["rev_rank"] == 0) & (counts >= min_events)
    val_mask = (ordered["rev_rank"] == 1) & (counts >= min_events)
    train_mask = ~(test_mask | val_mask)

    splits = {
        "train": ordered.loc[train_mask].drop(columns=["rev_rank"]).reset_index(drop=True),
        "val": ordered.loc[val_mask].drop(columns=["rev_rank"]).reset_index(drop=True),
        "test": ordered.loc[test_mask].drop(columns=["rev_rank"]).reset_index(drop=True),
    }
    return splits

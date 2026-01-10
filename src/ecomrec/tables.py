"""Pickle tables so we do not need pyarrow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)


def read_df(path: Path) -> pd.DataFrame:
    return pd.read_pickle(path)

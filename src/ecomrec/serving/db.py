"""Catalog and history lookups: Postgres when available, pickle otherwise."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd
from psycopg.rows import dict_row

from ecomrec.data.ingest import connect
from ecomrec.tables import read_df


class CatalogStore(Protocol):
    def user_history(self, user_id: int, n: int = 10) -> list[dict]: ...

    def get_product(self, product_id: int) -> dict | None: ...


class PostgresCatalogStore:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def user_history(self, user_id: int, n: int = 10) -> list[dict]:
        sql = """
            SELECT e.user_id, e.product_id, e.event_type, e.event_time, e.weight,
                   p.brand, p.category_code, p.price, p.title
            FROM user_events e
            LEFT JOIN products p ON p.product_id = e.product_id
            WHERE e.user_id = %s
            ORDER BY e.event_time DESC NULLS LAST
            LIMIT %s
        """
        with connect(self.database_url) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, (int(user_id), int(n)))
                return [dict(r) for r in cur.fetchall()]

    def get_product(self, product_id: int) -> dict | None:
        with connect(self.database_url) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT * FROM products WHERE product_id = %s", (int(product_id),))
                row = cur.fetchone()
        return dict(row) if row else None


class FileCatalogStore:
    def __init__(self, catalog_path: Path, events_path: Path | None) -> None:
        self.catalog = read_df(catalog_path) if catalog_path.exists() else pd.DataFrame()
        self.events = read_df(events_path) if events_path and events_path.exists() else pd.DataFrame()

    def user_history(self, user_id: int, n: int = 10) -> list[dict]:
        if self.events.empty:
            return []
        subset = self.events[self.events["user_id"].astype(int) == int(user_id)]
        if "event_time" in subset.columns:
            subset = subset.sort_values("event_time", ascending=False)
        subset = subset.head(n)
        if not self.catalog.empty:
            subset = subset.merge(self.catalog, on="product_id", how="left", suffixes=("", "_cat"))
        return subset.to_dict("records")

    def get_product(self, product_id: int) -> dict | None:
        if self.catalog.empty:
            return None
        hit = self.catalog[self.catalog["product_id"].astype(int) == int(product_id)]
        if hit.empty:
            return None
        return hit.iloc[0].to_dict()


class MemoryCatalogStore:
    def __init__(self, catalog: pd.DataFrame, events: pd.DataFrame | None = None) -> None:
        self.catalog = catalog
        self.events = events if events is not None else pd.DataFrame()

    def user_history(self, user_id: int, n: int = 10) -> list[dict]:
        if self.events.empty:
            return []
        subset = self.events[self.events["user_id"].astype(int) == int(user_id)].head(n)
        return subset.to_dict("records")

    def get_product(self, product_id: int) -> dict | None:
        hit = self.catalog[self.catalog["product_id"].astype(int) == int(product_id)]
        if hit.empty:
            return None
        return hit.iloc[0].to_dict()

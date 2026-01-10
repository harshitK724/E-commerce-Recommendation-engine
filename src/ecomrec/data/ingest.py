"""PostgreSQL ingest via COPY FROM STDIN (psycopg, no ORM)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import psycopg

from ecomrec.data.sql import COPY_COLUMNS, CREATE_RAW_EVENTS, CREATE_SERVING_TABLES


def dsn(database_url: str) -> str:
    return database_url.replace("postgresql+psycopg://", "postgresql://")


def connect(database_url: str) -> psycopg.Connection:
    return psycopg.connect(dsn(database_url))


def postgres_available(database_url: str) -> bool:
    try:
        with connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception:
        return False


def _run_script(conn: psycopg.Connection, sql: str) -> None:
    with conn.cursor() as cur:
        for stmt in (chunk.strip() for chunk in sql.split(";") if chunk.strip()):
            cur.execute(stmt)
    conn.commit()


def create_tables(conn: psycopg.Connection) -> None:
    _run_script(conn, CREATE_RAW_EVENTS)
    _run_script(conn, CREATE_SERVING_TABLES)


def truncate_raw_events(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE raw_events")
    conn.commit()


def copy_frame(
    conn: psycopg.Connection,
    table: str,
    df: pd.DataFrame,
    columns: Sequence[str],
) -> int:
    payload = df.reindex(columns=list(columns)).copy()
    if "event_time" in payload.columns:
        payload["event_time"] = pd.to_datetime(payload["event_time"], utc=True).dt.strftime(
            "%Y-%m-%d %H:%M:%S%z"
        )
    csv_bytes = payload.to_csv(index=False).encode("utf-8")
    copy_sql = f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT csv, HEADER true)"
    with conn.cursor() as cur:
        with cur.copy(copy_sql) as copy:
            copy.write(csv_bytes)
    conn.commit()
    return int(len(payload))


def copy_events_frame(conn: psycopg.Connection, df: pd.DataFrame) -> int:
    return copy_frame(conn, "raw_events", df, COPY_COLUMNS)


def copy_events_csv(conn: psycopg.Connection, path: Path) -> int:
    return copy_events_frame(conn, pd.read_csv(path))


def replace_serving_tables(
    conn: psycopg.Connection,
    catalog: pd.DataFrame,
    events: pd.DataFrame,
) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE products")
        cur.execute("TRUNCATE TABLE user_events")
    conn.commit()
    product_cols = [c for c in ("product_id", "category_id", "category_code", "brand", "price", "title") if c in catalog.columns]
    copy_frame(conn, "products", catalog, product_cols)
    event_cols = [c for c in ("user_id", "product_id", "event_type", "event_time", "weight") if c in events.columns]
    copy_frame(conn, "user_events", events, event_cols)

"""Synthetic events matching the Kaggle multi-category store schema."""

from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_TYPES = ("view", "cart", "purchase")
CATEGORIES = (
    "electronics.smartphone",
    "electronics.audio.headphone",
    "computers.notebook",
    "appliances.kitchen.washer",
    "apparel.shoes",
    "furniture.bedroom.bed",
    "auto.accessories",
    "kids.toys",
)
BRANDS = (
    "samsung",
    "apple",
    "xiaomi",
    "sony",
    "lg",
    "adidas",
    "nike",
    "bosch",
    "lenovo",
    "huawei",
)


def generate_synthetic_events(
    n_rows: int = 5_000,
    n_users: int = 250,
    n_items: int = 120,
    seed: int = 42,
    start: str = "2019-10-01",
) -> pd.DataFrame:
    """Power-law implicit feedback logs with view/cart/purchase funnel events."""
    rng = np.random.default_rng(seed)
    user_ids = np.arange(1, n_users + 1, dtype=np.int64)
    product_ids = np.arange(100_000, 100_000 + n_items, dtype=np.int64)

    user_p = (n_users - np.arange(n_users)).astype(np.float64)
    user_p /= user_p.sum()
    item_p = (n_items - np.arange(n_items)).astype(np.float64)
    item_p /= item_p.sum()

    users = rng.choice(user_ids, size=n_rows, p=user_p)
    products = rng.choice(product_ids, size=n_rows, p=item_p)
    types = rng.choice(EVENT_TYPES, size=n_rows, p=[0.72, 0.18, 0.10])

    start_ts = pd.Timestamp(start)
    seconds = rng.integers(0, 14 * 24 * 3600, size=n_rows)
    times = start_ts + pd.to_timedelta(seconds, unit="s")

    item_meta = {
        int(pid): {
            "category_id": 1000 + (i % len(CATEGORIES)),
            "category_code": CATEGORIES[i % len(CATEGORIES)],
            "brand": BRANDS[i % len(BRANDS)],
            "price": round(float(10 + (i * 7) % 900) + 0.99, 2),
        }
        for i, pid in enumerate(product_ids)
    }

    rows = {
        "event_time": times,
        "event_type": types,
        "product_id": products,
        "category_id": [item_meta[int(p)]["category_id"] for p in products],
        "category_code": [item_meta[int(p)]["category_code"] for p in products],
        "brand": [item_meta[int(p)]["brand"] for p in products],
        "price": [item_meta[int(p)]["price"] for p in products],
        "user_id": users,
        "user_session": [f"s-{u}-{t}" for u, t in zip(users, seconds)],
    }
    df = pd.DataFrame(rows)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    return df

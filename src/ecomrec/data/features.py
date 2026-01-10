"""ID maps and catalog. Interaction matrix is the sparse (user_idx, item_idx, weight) table."""

from __future__ import annotations

import pandas as pd


def build_id_maps(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    users = (
        pd.DataFrame({"user_id": sorted(interactions["user_id"].unique())})
        .reset_index()
        .rename(columns={"index": "user_idx"})
    )
    items = (
        pd.DataFrame({"product_id": sorted(interactions["product_id"].unique())})
        .reset_index()
        .rename(columns={"index": "item_idx"})
    )
    return users, items


def attach_indices(
    interactions: pd.DataFrame,
    user_map: pd.DataFrame,
    item_map: pd.DataFrame,
) -> pd.DataFrame:
    return interactions.merge(user_map, on="user_id", how="inner").merge(
        item_map, on="product_id", how="inner"
    )


def build_catalog(interactions: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("product_id", "category_id", "category_code", "brand", "price") if c in interactions.columns]
    catalog = interactions[cols].drop_duplicates("product_id").copy()
    brand = catalog["brand"].fillna("unknown").astype(str) if "brand" in catalog.columns else "item"
    category = (
        catalog["category_code"].fillna("general").astype(str)
        if "category_code" in catalog.columns
        else "general"
    )
    catalog["title"] = (
        brand + " " + category.str.replace(".", " ", regex=False) + " #" + catalog["product_id"].astype(str)
    )
    return catalog.reset_index(drop=True)

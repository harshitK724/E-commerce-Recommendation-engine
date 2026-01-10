import pandas as pd

from ecomrec.data.clean import aggregate_interactions, drop_invalid_rows, pandas_transform
from ecomrec.data.features import attach_indices, build_id_maps
from ecomrec.data.pipeline import run_pipeline
from ecomrec.data.splits import time_leave_last_out


def test_pandas_pipeline_filters_and_splits(toy_events, toy_settings):
    stats = run_pipeline(events=toy_events, cfg=toy_settings, use_postgres=False)
    assert stats["raw_rows"] == len(toy_events)
    assert stats["interaction_rows"] > 0
    assert stats["n_users"] > 1
    assert stats["train_rows"] > stats["test_rows"]
    assert (toy_settings.processed_dir / "train.pkl").exists()
    assert (toy_settings.artifacts_dir / "pipeline_stats.json").exists()


def test_invalid_rows_dropped():
    df = pd.DataFrame(
        {
            "event_time": ["2019-10-01T00:00:00Z", "2019-10-01T00:00:01Z", None],
            "event_type": ["view", "click", "purchase"],
            "product_id": [1, 2, 3],
            "user_id": [10, None, 12],
        }
    )
    cleaned = drop_invalid_rows(df.assign(event_time=pd.to_datetime(df["event_time"], utc=True)))
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["event_type"] == "view"


def test_max_weight_not_click_count():
    ts = pd.Timestamp("2019-10-01", tz="UTC")
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "product_id": [9, 9, 9],
            "event_type": ["view", "view", "purchase"],
            "event_time": [ts, ts, ts],
        }
    )
    agg = aggregate_interactions(df)
    assert len(agg) == 1
    assert int(agg.iloc[0]["weight"]) == 5


def test_time_split_is_leave_last_out(toy_events):
    interactions, _ = pandas_transform(toy_events)
    user_map, item_map = build_id_maps(interactions)
    indexed = attach_indices(interactions, user_map, item_map)
    splits = time_leave_last_out(indexed)
    test_users = set(splits["test"]["user_idx"])
    for user in list(test_users)[:10]:
        user_rows = indexed[indexed["user_idx"] == user].sort_values("last_event_time")
        last = user_rows.iloc[-1]
        held = splits["test"][splits["test"]["user_idx"] == user].iloc[0]
        assert last["product_id"] == held["product_id"]


def test_index_bounds(toy_events):
    interactions, _ = pandas_transform(toy_events)
    user_map, item_map = build_id_maps(interactions)
    indexed = attach_indices(interactions, user_map, item_map)
    assert indexed["user_idx"].max() < len(user_map)
    assert indexed["item_idx"].max() < len(item_map)
    assert len(indexed) == len(interactions)

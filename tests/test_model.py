import torch

from ecomrec.config import Settings
from ecomrec.data.clean import pandas_transform
from ecomrec.data.features import attach_indices, build_id_maps
from ecomrec.data.splits import time_leave_last_out
from ecomrec.models.evaluate import evaluate_model, popularity_ranks
from ecomrec.models.mf import BPRMF
from ecomrec.models.train import _user_pos_sets, train_bpr


def test_bpr_loss_drops_on_toy_matrix():
    torch.manual_seed(0)
    n_users, n_items = 20, 25
    pairs = []
    for u in range(n_users):
        for offset in range(4):
            pairs.append((u, (u + offset) % n_items, 5))
    import pandas as pd

    train = pd.DataFrame(pairs, columns=["user_idx", "item_idx", "weight"])
    cfg = Settings(epochs=6, embedding_dim=8, batch_size=32, negatives=1, lr=0.1)
    model, losses = train_bpr(train, n_users, n_items, cfg, val=None)
    assert losses[-1] < losses[0]
    model.eval()
    u = torch.tensor([0])
    pos = torch.tensor([0])
    neg = torch.tensor([10])
    assert float(model.score(u, pos).detach()) > float(model.score(u, neg).detach())


def test_evaluate_returns_metrics(toy_events):
    cfg = Settings(epochs=2, embedding_dim=8, batch_size=128, negatives=2)
    interactions, _ = pandas_transform(toy_events)
    user_map, item_map = build_id_maps(interactions)
    indexed = attach_indices(interactions, user_map, item_map)
    splits = time_leave_last_out(indexed)
    model, _ = train_bpr(splits["train"], len(user_map), len(item_map), cfg, val=None)
    pos = _user_pos_sets(splits["train"])
    metrics = evaluate_model(model, splits["test"], pos, len(item_map), k=10)
    pop = popularity_ranks(splits["train"], splits["test"], len(item_map), k=10)
    assert "recall" in metrics and "ndcg" in metrics and "hit_rate" in metrics
    assert metrics["n_eval_users"] > 0
    assert 0.0 <= pop["recall"] <= 1.0


def test_mf_forward_shape():
    model = BPRMF(5, 7, dim=4)
    users = torch.tensor([0, 1])
    items = torch.tensor([2, 3])
    out = model.score(users, items)
    assert out.shape == (2,)

import numpy as np

from evaluation import ndcg_at_k, precision_recall_at_k, rmse_mae


def test_precision_recall_basic():
    p, r = precision_recall_at_k(["a", "b", "c"], {"b", "c", "d"}, 3)
    assert p == 2 / 3
    assert r == 2 / 3


def test_ndcg_nonzero():
    n = ndcg_at_k(["x", "y", "z"], {"x", "z"}, 3)
    assert n > 0


def test_rmse_mae():
    rmse, mae = rmse_mae(np.array([3.0, 4.0]), np.array([2.0, 5.0]))
    assert rmse > 0
    assert mae == 1.0

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import train
from artifacts import CFArtifacts
from recommender import item_based_cf_recommend, svd_predict_matrix


def test_train_main_calls_init_db(monkeypatch, tmp_path: Path):
    called = {"init": 0}

    def fake_init_db():
        called["init"] += 1

    def fake_load_core_data():
        return (
            pd.DataFrame({"item_id": ["i1"], "title": ["t1"]}),
            pd.DataFrame({"user_id": ["u1"], "item_id": ["i1"], "rating": [4.0]}),
            pd.DataFrame(columns=["item_id", "review_text"]),
        )

    monkeypatch.setattr(train, "init_db", fake_init_db)
    monkeypatch.setattr(train, "load_core_data", fake_load_core_data)
    monkeypatch.setattr(train, "read_table", lambda _: pd.DataFrame(columns=["item_id", "review_text"]))
    train.main(tmp_path)
    assert called["init"] == 1


def test_item_cf_with_stale_artifacts_does_not_crash():
    ratings = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rating": 5.0},
            {"user_id": "u1", "item_id": "i2", "rating": 4.0},
            {"user_id": "u2", "item_id": "i1", "rating": 4.0},
            {"user_id": "u2", "item_id": "i3", "rating": 5.0},
        ]
    )
    artifacts = CFArtifacts(
        user_ids=["u1", "u2"],
        item_ids=["i1", "i2"],
        user_item_matrix=csr_matrix(np.array([[5.0, 4.0], [4.0, 0.0]])),
        user_similarity=np.eye(2),
        item_similarity=np.eye(2),
        svd_user_factors=np.array([[1.0], [1.0]]),
        svd_item_factors=np.array([[1.0, 1.0]]),
    )
    recs = item_based_cf_recommend("u1", ratings, cf_artifacts=artifacts, top_n=5)
    assert isinstance(recs, list)


def test_svd_predict_matrix_handles_tiny_matrix():
    ratings = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rating": 4.0}])
    recon, u_factors, v_factors = svd_predict_matrix(ratings)
    assert recon.shape == (1, 1)
    assert u_factors.shape[0] == 1
    assert v_factors.shape[1] == 1

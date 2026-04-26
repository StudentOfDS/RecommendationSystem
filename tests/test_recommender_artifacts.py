import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from artifacts import CFArtifacts
from recommender import user_based_cf_recommend


def test_user_cf_artifact_path_without_dense_rebuild():
    ratings = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rating": 5.0},
            {"user_id": "u2", "item_id": "i1", "rating": 4.0},
            {"user_id": "u2", "item_id": "i2", "rating": 5.0},
        ]
    )
    artifacts = CFArtifacts(
        user_ids=["u1", "u2"],
        item_ids=["i1", "i2"],
        user_item_matrix=csr_matrix(np.array([[5.0, 0.0], [4.0, 5.0]])),
        user_similarity=np.array([[1.0, 0.9], [0.9, 1.0]]),
        item_similarity=np.eye(2),
        svd_user_factors=np.array([[1.0], [1.0]]),
        svd_item_factors=np.array([[1.0, 1.0]]),
    )
    recs = user_based_cf_recommend("u1", ratings, cf_artifacts=artifacts, top_n=5)
    assert any(r.item_id == "i2" for r in recs)

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import ARTIFACT_DIR
from recommender import (
    content_based_recommend,
    hybrid_recommend,
    item_based_cf_recommend,
    svd_predict_matrix,
    user_based_cf_recommend,
)


def temporal_or_random_split(ratings: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    if "timestamp" in ratings.columns and ratings["timestamp"].notna().any():
        df = ratings.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")
        n_test = int(len(df) * test_size)
        test = df.tail(n_test)
        train = df.iloc[:-n_test]
        return train.reset_index(drop=True), test.reset_index(drop=True)
    train, test = train_test_split(ratings, test_size=test_size, random_state=random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def precision_recall_at_k(recommended: list[str], relevant: set[str], k: int) -> tuple[float, float]:
    top_k = recommended[:k]
    if k == 0:
        return 0.0, 0.0
    hits = sum(1 for i in top_k if i in relevant)
    precision = hits / k
    recall = hits / max(1, len(relevant))
    return precision, recall


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for idx, item in enumerate(recommended[:k], start=1):
        rel = 1.0 if item in relevant else 0.0
        dcg += rel / np.log2(idx + 1)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    errs = y_true - y_pred
    rmse = float(np.sqrt(np.mean(errs**2)))
    mae = float(np.mean(np.abs(errs)))
    return rmse, mae


def evaluate_ranking_for_user(recommended: list[str], test_user_df: pd.DataFrame, k: int = 10) -> dict:
    relevant = set(test_user_df[test_user_df["rating"] >= 4.0]["item_id"].astype(str))
    p, r = precision_recall_at_k(recommended, relevant, k)
    n = ndcg_at_k(recommended, relevant, k)
    return {f"precision@{k}": p, f"recall@{k}": r, f"ndcg@{k}": n}


def run_offline_benchmark(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_artifacts, k: int = 10) -> pd.DataFrame:
    users = sorted(set(test_df["user_id"].astype(str).tolist()))
    rows = []
    for algo in ["content", "user_cf", "item_cf", "hybrid"]:
        per_user = []
        for u in users:
            if algo == "content":
                recs = content_based_recommend(u, train_df, feature_artifacts, top_n=k)
            elif algo == "user_cf":
                recs = user_based_cf_recommend(u, train_df, top_n=k)
            elif algo == "item_cf":
                recs = item_based_cf_recommend(u, train_df, top_n=k)
            else:
                recs = hybrid_recommend(u, train_df, feature_artifacts, top_n=k)
            user_test = test_df[test_df["user_id"].astype(str) == u]
            if user_test.empty:
                continue
            metrics = evaluate_ranking_for_user([r.item_id for r in recs], user_test, k=k)
            per_user.append(metrics)
        if per_user:
            rows.append(
                {
                    "algorithm": algo,
                    f"precision@{k}": float(np.mean([m[f"precision@{k}"] for m in per_user])),
                    f"recall@{k}": float(np.mean([m[f"recall@{k}"] for m in per_user])),
                    f"ndcg@{k}": float(np.mean([m[f"ndcg@{k}"] for m in per_user])),
                }
            )
    return pd.DataFrame(rows)


def evaluate_svd_regression(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    recon, _, _ = svd_predict_matrix(train_df)
    pairs = test_df[["user_id", "item_id", "rating"]].copy()
    preds = []
    actual = []
    for _, row in pairs.iterrows():
        u = str(row["user_id"])
        i = str(row["item_id"])
        if u in recon.index and i in recon.columns:
            preds.append(float(recon.loc[u, i]))
            actual.append(float(row["rating"]))
    if not preds:
        return {"rmse": None, "mae": None}
    rmse, mae = rmse_mae(np.array(actual), np.array(preds))
    return {"rmse": rmse, "mae": mae}


def save_evaluation_report(report: dict, output_path: Path = ARTIFACT_DIR / "evaluation_report.json") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

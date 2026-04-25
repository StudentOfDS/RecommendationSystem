import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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

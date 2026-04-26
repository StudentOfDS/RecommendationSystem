from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import MAX_TFIDF_FEATURES, TEXT_FIELDS

POSITIVE_WORDS = {"great", "excellent", "amazing", "love", "good", "favorite", "best"}
NEGATIVE_WORDS = {"bad", "boring", "awful", "hate", "worst", "poor", "terrible"}


@dataclass
class FeatureArtifacts:
    item_ids: np.ndarray
    feature_matrix: sparse.csr_matrix
    transformer: Pipeline


def clean_ratings(
    ratings: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> pd.DataFrame:
    df = ratings.copy()
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="last")
    df = df.dropna(subset=["user_id", "item_id", "rating"])
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])

    user_counts = df.groupby("user_id")["item_id"].count()
    valid_users = user_counts[user_counts >= min_user_interactions].index
    df = df[df["user_id"].isin(valid_users)]

    item_counts = df.groupby("item_id")["user_id"].count()
    valid_items = item_counts[item_counts >= min_item_interactions].index
    return df[df["item_id"].isin(valid_items)].reset_index(drop=True)


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    return ratings.pivot_table(index="user_id", columns="item_id", values="rating", aggfunc="mean")


def merge_review_texts(items: pd.DataFrame, reviews: pd.DataFrame | None = None) -> pd.DataFrame:
    df = items.copy()
    if reviews is None or reviews.empty or "review_text" not in reviews.columns:
        df["review_corpus"] = ""
        return df
    temp = reviews.copy()
    temp["item_id"] = temp["item_id"].astype(str)
    grouped = temp.groupby("item_id")["review_text"].apply(lambda s: " ".join(s.astype(str).tolist())).rename("review_corpus")
    df["item_id"] = df["item_id"].astype(str)
    return df.merge(grouped, how="left", on="item_id").fillna({"review_corpus": ""})


def build_item_features(items: pd.DataFrame, reviews: pd.DataFrame | None = None) -> FeatureArtifacts:
    df = merge_review_texts(items, reviews)
    df["item_id"] = df["item_id"].astype(str)

    text_cols = [c for c in TEXT_FIELDS if c in df.columns]
    for c in text_cols:
        df[c] = df[c].fillna("")
    df["all_text"] = (df[text_cols].agg(" ".join, axis=1) if text_cols else "") + " " + df["review_corpus"].fillna("")

    if "year" not in df.columns:
        df["year"] = 0
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0)

    preprocess = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=MAX_TFIDF_FEATURES, ngram_range=(1, 2)), "all_text"),
            ("genres", OneHotEncoder(handle_unknown="ignore"), ["genres"] if "genres" in df.columns else ["all_text"]),
            ("num", StandardScaler(with_mean=False), ["year", "review_sentiment"]),
        ],
        sparse_threshold=0.7,
    )
    pipeline = Pipeline([("preprocess", preprocess)])
    matrix = pipeline.fit_transform(df)

    return FeatureArtifacts(item_ids=df["item_id"].to_numpy(), feature_matrix=matrix.tocsr(), transformer=pipeline)


def save_feature_artifacts(artifacts: FeatureArtifacts, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, output_path)


def load_feature_artifacts(path):
    return joblib.load(path)


def build_user_profile(
    user_id: str,
    ratings: pd.DataFrame,
    item_features: FeatureArtifacts,
    min_positive_rating: float = 3.5,
):
    user_rows = ratings[ratings["user_id"].astype(str) == str(user_id)]
    positives = user_rows[user_rows["rating"] >= min_positive_rating]
    if positives.empty:
        return None

    item_index = {iid: i for i, iid in enumerate(item_features.item_ids)}
    vec = None
    total_weight = 0.0
    for _, row in positives.iterrows():
        iid = str(row["item_id"])
        if iid not in item_index:
            continue
        weight = float(row["rating"])
        row_vec = item_features.feature_matrix[item_index[iid]]
        vec = row_vec.multiply(weight) if vec is None else vec + row_vec.multiply(weight)
        total_weight += weight
    if vec is None or total_weight == 0:
        return None
    return vec / total_weight

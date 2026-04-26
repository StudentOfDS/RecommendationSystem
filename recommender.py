import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from artifacts import CFArtifacts
from config import (
    ALPHA_HYBRID_DEFAULT,
    BANDIT_REWARD_MAX,
    BANDIT_REWARD_MIN,
    BANDIT_STATE_DIR,
    COLLAB_WEIGHTS,
    EPSILON,
    K_NEIGHBORS_DEFAULT,
    RATING_SCALE_MAX,
    RATING_SCALE_MIN,
    SIMILARITY_METRIC,
    SVD_COMPONENTS,
    SVD_RANDOM_STATE,
)
from features import FeatureArtifacts, build_user_item_matrix, build_user_profile


@dataclass
class Recommendation:
    item_id: str
    score: float
    reason: str


def _exclude_consumed(ratings: pd.DataFrame, user_id: str, candidate_scores: pd.Series) -> pd.Series:
    consumed = set(ratings[ratings["user_id"].astype(str) == str(user_id)]["item_id"].astype(str))
    return candidate_scores[~candidate_scores.index.astype(str).isin(consumed)]


def popularity_fallback(ratings: pd.DataFrame, top_n: int = 10) -> list[Recommendation]:
    agg = ratings.groupby("item_id").agg(mean_rating=("rating", "mean"), cnt=("rating", "count"))
    agg["score"] = agg["mean_rating"] * np.log1p(agg["cnt"])
    top = agg.sort_values("score", ascending=False).head(top_n)
    return [Recommendation(str(iid), float(row["score"]), "popular among all users") for iid, row in top.iterrows()]


def content_based_recommend(
    user_id: str,
    ratings: pd.DataFrame,
    item_features: FeatureArtifacts,
    top_n: int = 10,
) -> list[Recommendation]:
    profile = build_user_profile(user_id, ratings, item_features)
    if profile is None:
        return []
    sims = cosine_similarity(profile, item_features.feature_matrix).flatten()
    scores = pd.Series(sims, index=item_features.item_ids)
    scores = _exclude_consumed(ratings, user_id, scores)
    top = scores.sort_values(ascending=False).head(top_n)
    return [
        Recommendation(str(iid), float(score), "similar in content/genre/tags/reviews to your liked items")
        for iid, score in top.items()
    ]


def _normalize_item_vector(vec: pd.Series) -> pd.Series:
    mu = vec.mean()
    denom = max(1e-9, RATING_SCALE_MAX - RATING_SCALE_MIN)
    return (vec - mu) / denom


def _item_similarity(item_mat: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric == "pearson":
        return item_mat.T.corr(method="pearson").fillna(0.0)
    return pd.DataFrame(cosine_similarity(item_mat), index=item_mat.index, columns=item_mat.index)


def user_based_cf_recommend(
    user_id: str,
    ratings: pd.DataFrame,
    k_neighbors: int = K_NEIGHBORS_DEFAULT,
    top_n: int = 10,
    cf_artifacts: CFArtifacts | None = None,
):
    if cf_artifacts is not None:
        if str(user_id) not in cf_artifacts.user_ids:
            return []
        user_idx = cf_artifacts.user_ids.index(str(user_id))
        sim_vec = pd.Series(cf_artifacts.user_similarity[user_idx], index=cf_artifacts.user_ids)
        neighbors = sim_vec.drop(str(user_id)).sort_values(ascending=False).head(k_neighbors)
        uim = pd.DataFrame(cf_artifacts.user_item_matrix.toarray(), index=cf_artifacts.user_ids, columns=cf_artifacts.item_ids)
    else:
        uim = build_user_item_matrix(ratings)
        if str(user_id) not in uim.index:
            return []
        filled = uim.fillna(0)
        sim = pd.DataFrame(cosine_similarity(filled), index=filled.index, columns=filled.index)
        neighbors = sim.loc[str(user_id)].drop(str(user_id)).sort_values(ascending=False).head(k_neighbors)

    user_rated = set(ratings[ratings["user_id"].astype(str) == str(user_id)]["item_id"].astype(str))
    preds = {}
    for item in uim.columns:
        if str(item) in user_rated:
            continue
        numer = 0.0
        denom = 0.0
        for nb, s in neighbors.items():
            r = uim.loc[nb, item]
            if pd.notna(r):
                numer += s * float(r)
                denom += abs(float(s))
        if denom > 0:
            preds[str(item)] = numer / denom

    top = pd.Series(preds).sort_values(ascending=False).head(top_n)
    return [Recommendation(iid, float(score), "users with similar tastes liked this") for iid, score in top.items()]


def item_based_cf_recommend(
    user_id: str,
    ratings: pd.DataFrame,
    k_neighbors: int = K_NEIGHBORS_DEFAULT,
    top_n: int = 10,
    metric: str = SIMILARITY_METRIC,
    cf_artifacts: CFArtifacts | None = None,
):
    uim = build_user_item_matrix(ratings)
    if str(user_id) not in uim.index:
        return []

    centered = uim.apply(_normalize_item_vector, axis=0)
    sparse_mat = csr_matrix(centered.fillna(0).values)

    if cf_artifacts is not None:
        sim = pd.DataFrame(cf_artifacts.item_similarity, index=cf_artifacts.item_ids, columns=cf_artifacts.item_ids)
    else:
        item_mat = pd.DataFrame(sparse_mat.T.toarray(), index=centered.columns, columns=centered.index)
        sim = _item_similarity(item_mat, metric)

    user_row = centered.loc[str(user_id)]
    rated = user_row.dropna()
    preds = {}
    for candidate in centered.columns:
        candidate_str = str(candidate)
        if pd.notna(user_row[candidate]):
            continue
        if artifact_items is not None and candidate_str not in artifact_items:
            continue

        numer = 0.0
        denom = 0.0
        nbrs = sim.loc[candidate, rated.index].sort_values(ascending=False).head(k_neighbors)
        for item, s in nbrs.items():
            numer += float(s) * float(rated[item])
            denom += abs(float(s))
        if denom > 0:
            preds[str(candidate)] = float(ratings["rating"].mean()) + (numer / denom)

    top = pd.Series(preds).sort_values(ascending=False).head(top_n)
    return [Recommendation(iid, float(score), f"similar items using {metric} similarity") for iid, score in top.items()]


def svd_predict_matrix(ratings: pd.DataFrame, n_components: int = SVD_COMPONENTS):
    uim = build_user_item_matrix(ratings)
    mat = uim.fillna(0)
    min_dim = min(mat.shape)
    if min_dim < 2:
        recon = mat.values.astype(float)
        u_factors = mat.values.astype(float)
        v_factors = np.eye(mat.shape[1], dtype=float)
        return pd.DataFrame(recon, index=mat.index, columns=mat.columns), u_factors, v_factors

    n_comp = max(1, min(n_components, min_dim - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=SVD_RANDOM_STATE)
    U = svd.fit_transform(mat)
    V = svd.components_
    recon = np.dot(U, V)
    return pd.DataFrame(recon, index=mat.index, columns=mat.columns), U, V


def svd_recommend(
    user_id: str,
    ratings: pd.DataFrame,
    top_n: int = 10,
    n_components: int = SVD_COMPONENTS,
    cf_artifacts: CFArtifacts | None = None,
):
    uim = build_user_item_matrix(ratings)
    if str(user_id) not in uim.index:
        return []

    if cf_artifacts is not None and str(user_id) in cf_artifacts.user_ids:
        uidx = cf_artifacts.user_ids.index(str(user_id))
        recon_row = np.dot(cf_artifacts.svd_user_factors[uidx], cf_artifacts.svd_item_factors)
        scores = pd.Series(recon_row, index=cf_artifacts.item_ids)
    else:
        recon_df, _, _ = svd_predict_matrix(ratings, n_components=n_components)
        scores = recon_df.loc[str(user_id)]

    scores = _exclude_consumed(ratings, user_id, scores)
    top = scores.sort_values(ascending=False).head(top_n)
    return [Recommendation(str(iid), float(score), "latent factor match from matrix factorization") for iid, score in top.items()]


def _blend_recs(rec_lists: dict[str, list[Recommendation]], top_n: int) -> list[Recommendation]:
    combined: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}
    for name, recs in rec_lists.items():
        w = COLLAB_WEIGHTS.get(name, 0.0)
        for rec in recs:
            combined[rec.item_id] = combined.get(rec.item_id, 0.0) + (w * rec.score)
            reasons.setdefault(rec.item_id, []).append(name)
    sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [
        Recommendation(iid, float(score), f"combined from {', '.join(reasons.get(iid, []))}")
        for iid, score in sorted_items
    ]


def hybrid_recommend(
    user_id: str,
    ratings: pd.DataFrame,
    item_features: FeatureArtifacts,
    top_n: int = 10,
    alpha: float = ALPHA_HYBRID_DEFAULT,
    cf_artifacts: CFArtifacts | None = None,
):
    user_hist_len = int((ratings["user_id"].astype(str) == str(user_id)).sum())

    cb = content_based_recommend(user_id, ratings, item_features, top_n=top_n * 3)
    ub = user_based_cf_recommend(user_id, ratings, top_n=top_n * 3, cf_artifacts=cf_artifacts)
    ib = item_based_cf_recommend(user_id, ratings, top_n=top_n * 3, cf_artifacts=cf_artifacts)
    svd = svd_recommend(user_id, ratings, top_n=top_n * 3, cf_artifacts=cf_artifacts)

    if user_hist_len < 3:
        alpha = min(alpha, 0.3)

    collab = _blend_recs({"user_cf": ub, "item_cf": ib, "svd": svd}, top_n=top_n * 4)
    collab_map = {r.item_id: r.score for r in collab}
    cb_map = {r.item_id: r.score for r in cb}

    all_items = set(collab_map) | set(cb_map)
    scored = []
    for iid in all_items:
        c_score = collab_map.get(iid, 0.0)
        b_score = cb_map.get(iid, 0.0)
        final = alpha * c_score + (1 - alpha) * b_score
        scored.append(Recommendation(iid, float(final), "hybrid of collaborative and content models"))

    if not scored:
        return popularity_fallback(ratings, top_n=top_n)

    return sorted(scored, key=lambda r: r.score, reverse=True)[:top_n]


class EpsilonGreedyBandit:
    def __init__(self, state_dir: Path = BANDIT_STATE_DIR, epsilon: float = EPSILON):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.epsilon = epsilon

    def _user_file(self, user_id: str) -> Path:
        return self.state_dir / f"{user_id}.json"

    def _load_state(self, user_id: str) -> dict:
        file = self._user_file(user_id)
        if not file.exists():
            return {"counts": {}, "means": {}, "impressions": {}}
        return json.loads(file.read_text())

    def _save_state(self, user_id: str, state: dict):
        self._user_file(user_id).write_text(json.dumps(state, indent=2))

    def rating_to_reward(self, rating: float, min_rating: float = 0.5, max_rating: float = 5.0) -> float:
        reward = (rating - min_rating) / (max_rating - min_rating)
        return float(max(BANDIT_REWARD_MIN, min(BANDIT_REWARD_MAX, reward)))

    def log_impression(self, user_id: str, item_id: str) -> None:
        state = self._load_state(user_id)
        state.setdefault("impressions", {})[item_id] = int(state.get("impressions", {}).get(item_id, 0)) + 1
        self._save_state(user_id, state)

    def update(self, user_id: str, item_id: str, reward: float) -> None:
        state = self._load_state(user_id)
        counts = state["counts"]
        means = state["means"]

        c = int(counts.get(item_id, 0))
        m = float(means.get(item_id, 0.0))
        new_c = c + 1
        new_m = m + (reward - m) / new_c

        counts[item_id] = new_c
        means[item_id] = new_m
        self._save_state(user_id, state)

    def rerank(self, user_id: str, candidates: list[Recommendation]) -> list[Recommendation]:
        if not candidates:
            return []
        state = self._load_state(user_id)
        means = state.get("means", {})

        for rec in candidates:
            self.log_impression(user_id, rec.item_id)

        if random.random() < self.epsilon:
            shuffled = candidates[:]
            random.shuffle(shuffled)
            return shuffled

        def score(rec: Recommendation):
            return 0.7 * rec.score + 0.3 * float(means.get(rec.item_id, 0.0))

        return sorted(candidates, key=score, reverse=True)

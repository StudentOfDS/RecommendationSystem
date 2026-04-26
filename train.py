import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from artifacts import CFArtifacts, save_cf_artifacts, save_metadata
from config import ARTIFACT_DIR, SVD_COMPONENTS, SVD_RANDOM_STATE
from database import load_core_data, read_table
from features import build_item_features, build_user_item_matrix, save_feature_artifacts


def build_cf_artifacts(ratings_df: pd.DataFrame) -> CFArtifacts:
    uim = build_user_item_matrix(ratings_df)
    filled = uim.fillna(0)
    sparse_uim = csr_matrix(filled.values)

    user_similarity = cosine_similarity(sparse_uim)
    item_similarity = cosine_similarity(sparse_uim.T)

    n_comp = max(2, min(SVD_COMPONENTS, min(filled.shape) - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=SVD_RANDOM_STATE)
    U = svd.fit_transform(sparse_uim)
    V = svd.components_

    return CFArtifacts(
        user_ids=filled.index.astype(str).tolist(),
        item_ids=filled.columns.astype(str).tolist(),
        user_item_matrix=sparse_uim,
        user_similarity=user_similarity,
        item_similarity=item_similarity,
        svd_user_factors=U,
        svd_item_factors=V,
    )


def main(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    movies_df, ratings_df, reviews_df = load_core_data()
    scraped_df = read_table("scraped_reviews")

    reviews_all = pd.concat([
        reviews_df[["item_id", "review_text"]] if not reviews_df.empty else pd.DataFrame(columns=["item_id", "review_text"]),
        scraped_df[["item_id", "review_text"]] if not scraped_df.empty else pd.DataFrame(columns=["item_id", "review_text"]),
    ], ignore_index=True)

    feature_artifacts = build_item_features(movies_df, reviews=reviews_all)
    save_feature_artifacts(feature_artifacts, output_dir / "feature_artifacts.joblib")

    cf_artifacts = build_cf_artifacts(ratings_df)
    save_cf_artifacts(cf_artifacts, output_dir / "cf_artifacts.joblib")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_movies": int(len(movies_df)),
        "n_ratings": int(len(ratings_df)),
        "n_reviews": int(len(reviews_all)),
        "svd_components": int(cf_artifacts.svd_item_factors.shape[0]),
        "artifacts": ["feature_artifacts.joblib", "cf_artifacts.joblib"],
    }
    save_metadata(metadata, output_dir / "metadata.json")
    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline training pipeline for recommendation artifacts.")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()
    main(args.output_dir)

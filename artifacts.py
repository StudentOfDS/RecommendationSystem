import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse


@dataclass
class CFArtifacts:
    user_ids: list[str]
    item_ids: list[str]
    user_item_matrix: sparse.csr_matrix
    user_similarity: np.ndarray
    item_similarity: np.ndarray
    svd_user_factors: np.ndarray
    svd_item_factors: np.ndarray


@dataclass
class PersistedArtifacts:
    feature_artifacts_path: Path
    cf_artifacts_path: Path
    metadata_path: Path


def save_cf_artifacts(artifacts: CFArtifacts, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "user_ids": artifacts.user_ids,
        "item_ids": artifacts.item_ids,
        "user_item_matrix": artifacts.user_item_matrix,
        "user_similarity": artifacts.user_similarity,
        "item_similarity": artifacts.item_similarity,
        "svd_user_factors": artifacts.svd_user_factors,
        "svd_item_factors": artifacts.svd_item_factors,
    }
    joblib.dump(payload, output_path)


def load_cf_artifacts(path: Path) -> CFArtifacts:
    payload = joblib.load(path)
    return CFArtifacts(
        user_ids=payload["user_ids"],
        item_ids=payload["item_ids"],
        user_item_matrix=payload["user_item_matrix"],
        user_similarity=payload["user_similarity"],
        item_similarity=payload["item_similarity"],
        svd_user_factors=payload["svd_user_factors"],
        svd_item_factors=payload["svd_item_factors"],
    )


def save_metadata(metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2))


def load_metadata(path: Path) -> dict:
    return json.loads(path.read_text())

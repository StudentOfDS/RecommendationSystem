from pathlib import Path

import pandas as pd

from features import _simple_sentiment, merge_review_texts
from train import _artifact_version


def test_app_imports_logging_utils_symbols():
    app_src = Path("app.py").read_text()
    assert "from logging_utils import get_logger, log_event, timed_block" in app_src
    assert "def _artifact_freshness_warning" in app_src


def test_artifact_version_changes_with_data_shape():
    movies = pd.DataFrame({"item_id": ["i1"]})
    ratings_a = pd.DataFrame({"user_id": ["u1"], "item_id": ["i1"], "rating": [4.0]})
    ratings_b = pd.DataFrame({"user_id": ["u1", "u2"], "item_id": ["i1", "i1"], "rating": [4.0, 5.0]})
    reviews = pd.DataFrame({"item_id": ["i1"], "review_text": ["great movie"]})
    assert _artifact_version(movies, ratings_a, reviews) != _artifact_version(movies, ratings_b, reviews)


def test_sentiment_feature_present_in_merged_reviews():
    items = pd.DataFrame({"item_id": ["i1", "i2"]})
    reviews = pd.DataFrame(
        {
            "item_id": ["i1", "i2"],
            "review_text": ["great amazing good", "awful bad boring"],
        }
    )
    merged = merge_review_texts(items, reviews)
    assert "review_sentiment" in merged.columns
    assert merged.loc[merged["item_id"] == "i1", "review_sentiment"].iloc[0] > 0
    assert merged.loc[merged["item_id"] == "i2", "review_sentiment"].iloc[0] < 0


def test_simple_sentiment_neutral_empty():
    assert _simple_sentiment("") == 0.0

import numpy as np
import pandas as pd
import pytest

from database import validate_movies


def test_validate_movies_requires_item_id():
    with pytest.raises(ValueError):
        validate_movies(pd.DataFrame({"title": ["x"]}))


def test_validate_movies_rejects_missing_or_blank_ids():
    bad_frames = [
        pd.DataFrame({"item_id": [None]}),
        pd.DataFrame({"item_id": [np.nan]}),
        pd.DataFrame({"item_id": [""]}),
        pd.DataFrame({"item_id": ["   "]}),
        pd.DataFrame({"item_id": ["None"]}),
        pd.DataFrame({"item_id": ["nan"]}),
        pd.DataFrame({"item_id": ["null"]}),
    ]
    for frame in bad_frames:
        with pytest.raises(ValueError):
            validate_movies(frame)

import pandas as pd
import pytest

from database import validate_movies


def test_validate_movies_requires_item_id():
    with pytest.raises(ValueError):
        validate_movies(pd.DataFrame({"title": ["x"]}))

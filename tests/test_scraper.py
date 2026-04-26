import pytest
from bs4 import BeautifulSoup

from scraper import _extract_rating, _parse_generic, validate_url


def test_validate_url_rejects_non_http():
    with pytest.raises(ValueError):
        validate_url("ftp://example.com")


def test_extract_rating_patterns():
    assert _extract_rating("Score 8/10 overall") == 4.0
    assert _extract_rating("Score 3.5/5 overall") == 3.5


def test_parse_generic_review_selector():
    html = '<article data-author="a1">This movie was great and amazing with good acting and story depth.</article>'
    soup = BeautifulSoup(html, "html.parser")
    rows = _parse_generic(soup)
    assert len(rows) == 1

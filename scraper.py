import hashlib
import time
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import SCRAPER_MAX_RETRIES, SCRAPER_TIMEOUT_SECONDS, SCRAPER_USER_AGENT


@dataclass
class ScrapeResult:
    item_id: str
    source_url: str
    reviewer: str
    review_text: str
    rating: float | None


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _hash_review(item_id: str, reviewer: str, review_text: str) -> str:
    raw = f"{item_id}|{reviewer}|{review_text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def scrape_generic_reviews(item_id: str, url: str, delay_seconds: float = 1.0) -> pd.DataFrame:
    headers = {"User-Agent": SCRAPER_USER_AGENT}
    last_err = None
    for attempt in range(SCRAPER_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=SCRAPER_TIMEOUT_SECONDS)
            resp.raise_for_status()
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt >= SCRAPER_MAX_RETRIES:
                raise RuntimeError(f"Failed scraping {url}: {exc}") from exc
            time.sleep(delay_seconds)
    else:
        raise RuntimeError(f"Failed scraping {url}: {last_err}")

    soup = BeautifulSoup(resp.text, "html.parser")

    candidates: Iterable = soup.select(".review, .user-review, .comment, article")
    rows = []
    for c in candidates:
        txt = _normalize_text(c.get_text(" ", strip=True))
        if len(txt) < 40:
            continue
        reviewer = c.get("data-author") or "anonymous"
        rows.append(
            {
                "item_id": str(item_id),
                "source_url": url,
                "reviewer": reviewer,
                "review_text": txt,
                "rating": None,
                "review_hash": _hash_review(str(item_id), reviewer, txt),
            }
        )

    if delay_seconds > 0:
        time.sleep(delay_seconds)

    if not rows:
        return pd.DataFrame(columns=["item_id", "source_url", "reviewer", "review_text", "rating", "review_hash"])

    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["review_hash"]).reset_index(drop=True)

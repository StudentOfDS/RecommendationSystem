import hashlib
import re
import time
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import SCRAPER_MAX_RETRIES, SCRAPER_TIMEOUT_SECONDS, SCRAPER_USER_AGENT

ALLOWED_SCHEMES = {"http", "https"}


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


def validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES or not parsed.netloc:
        raise ValueError("Invalid URL. Use fully-qualified http(s) URLs.")


def check_robots_allowed(url: str, user_agent: str = SCRAPER_USER_AGENT) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception:
        return False
    return rp.can_fetch(user_agent, url)


def _extract_rating(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", text)
    if m:
        return float(m.group(1)) / 2.0
    m2 = re.search(r"(\d(?:\.\d+)?)\s*/\s*5", text)
    if m2:
        return float(m2.group(1))
    return None


def _parse_imdb(soup: BeautifulSoup) -> list[dict]:
    rows = []
    for c in soup.select('[data-testid="review-content"]'):
        txt = _normalize_text(c.get_text(" ", strip=True))
        if len(txt) < 40:
            continue
        rows.append({"review_text": txt, "reviewer": "imdb_user", "rating": _extract_rating(txt)})
    return rows


def _parse_rotten_tomatoes(soup: BeautifulSoup) -> list[dict]:
    rows = []
    for c in soup.select(".review_text, .audience-reviews__review-wrap"):
        txt = _normalize_text(c.get_text(" ", strip=True))
        if len(txt) < 40:
            continue
        rows.append({"review_text": txt, "reviewer": "rt_user", "rating": _extract_rating(txt)})
    return rows


def _parse_generic(soup: BeautifulSoup) -> list[dict]:
    candidates: Iterable = soup.select(".review, .user-review, .comment, article")
    rows = []
    for c in candidates:
        txt = _normalize_text(c.get_text(" ", strip=True))
        if len(txt) < 40:
            continue
        reviewer = c.get("data-author") or "anonymous"
        rows.append({"review_text": txt, "reviewer": reviewer, "rating": _extract_rating(txt)})
    return rows


def scrape_generic_reviews(item_id: str, url: str, delay_seconds: float = 1.0) -> pd.DataFrame:
    validate_url(url)
    if not check_robots_allowed(url):
        raise PermissionError("robots.txt does not allow scraping this URL with configured user-agent")

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
    host = urlparse(url).netloc.lower()
    if "imdb.com" in host:
        parsed_rows = _parse_imdb(soup)
    elif "rottentomatoes.com" in host:
        parsed_rows = _parse_rotten_tomatoes(soup)
    else:
        parsed_rows = _parse_generic(soup)

    rows = []
    for row in parsed_rows:
        txt = row["review_text"]
        reviewer = row["reviewer"]
        rows.append(
            {
                "item_id": str(item_id),
                "source_url": url,
                "reviewer": reviewer,
                "review_text": txt,
                "rating": row["rating"],
                "review_hash": _hash_review(str(item_id), reviewer, txt),
            }
        )

    if delay_seconds > 0:
        time.sleep(delay_seconds)

    if not rows:
        return pd.DataFrame(columns=["item_id", "source_url", "reviewer", "review_text", "rating", "review_hash"])

    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["review_hash"]).reset_index(drop=True)

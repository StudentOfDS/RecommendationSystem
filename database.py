import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from config import DB_PATH


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextmanager
def db_cursor(db_path: Path = DB_PATH):
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    with db_cursor(db_path) as cur:
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS movies (
                item_id TEXT PRIMARY KEY,
                title TEXT,
                genres TEXT,
                description TEXT,
                tags TEXT,
                year INTEGER
            );

            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                rating REAL NOT NULL,
                timestamp TEXT,
                UNIQUE(user_id, item_id),
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                FOREIGN KEY(item_id) REFERENCES movies(item_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                item_id TEXT,
                review_text TEXT,
                source TEXT DEFAULT 'manual',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE SET NULL,
                FOREIGN KEY(item_id) REFERENCES movies(item_id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS scraped_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT,
                source_url TEXT,
                reviewer TEXT,
                review_text TEXT,
                rating REAL,
                scraped_at TEXT DEFAULT CURRENT_TIMESTAMP,
                review_hash TEXT UNIQUE,
                FOREIGN KEY(item_id) REFERENCES movies(item_id) ON DELETE SET NULL
            );
            """
        )


def upsert_users(user_ids: Iterable[str]) -> None:
    with db_cursor() as cur:
        cur.executemany(
            "INSERT OR IGNORE INTO users(user_id) VALUES (?)",
            [(str(u),) for u in user_ids],
        )


def upsert_movies(df_movies: pd.DataFrame) -> None:
    cols = ["item_id", "title", "genres", "description", "tags", "year"]
    safe = df_movies.copy()
    for col in cols:
        if col not in safe.columns:
            safe[col] = None
    records = list(safe[cols].itertuples(index=False, name=None))
    with db_cursor() as cur:
        cur.executemany(
            """
            INSERT INTO movies(item_id, title, genres, description, tags, year)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id)
            DO UPDATE SET
              title=excluded.title,
              genres=excluded.genres,
              description=excluded.description,
              tags=excluded.tags,
              year=excluded.year
            """,
            records,
        )


def upsert_ratings(df_ratings: pd.DataFrame) -> None:
    req = ["user_id", "item_id", "rating"]
    missing = [c for c in req if c not in df_ratings.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    safe = df_ratings.copy()
    if "timestamp" not in safe.columns:
        safe["timestamp"] = None
    upsert_users(safe["user_id"].astype(str).unique())

    records = list(
        safe[["user_id", "item_id", "rating", "timestamp"]].itertuples(index=False, name=None)
    )
    with db_cursor() as cur:
        cur.executemany(
            """
            INSERT INTO ratings(user_id, item_id, rating, timestamp)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, item_id)
            DO UPDATE SET rating=excluded.rating, timestamp=excluded.timestamp
            """,
            records,
        )


def add_manual_review(user_id: Optional[str], item_id: str, review_text: str) -> None:
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO reviews(user_id, item_id, review_text, source) VALUES (?, ?, ?, 'manual')",
            (user_id, item_id, review_text),
        )


def add_scraped_reviews(df_scraped: pd.DataFrame) -> None:
    cols = ["item_id", "source_url", "reviewer", "review_text", "rating", "review_hash"]
    safe = df_scraped.copy()
    for col in cols:
        if col not in safe.columns:
            safe[col] = None
    records = list(safe[cols].itertuples(index=False, name=None))

    with db_cursor() as cur:
        cur.executemany(
            """
            INSERT OR IGNORE INTO scraped_reviews(item_id, source_url, reviewer, review_text, rating, review_hash)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            records,
        )


def read_table(table_name: str) -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()


def load_core_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return read_table("movies"), read_table("ratings"), read_table("reviews")

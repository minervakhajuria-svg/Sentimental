"""DuckDB access: connection, schema setup and idempotent post upserts."""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path
from typing import Iterable

import duckdb

from collectors.base import Post

log = logging.getLogger(__name__)

POST_COLUMNS = [
    "id", "source", "community", "author_id", "author_age_days", "author_karma",
    "created_at", "collected_at", "title", "body", "url", "engagement", "content_hash",
]


def connect(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) the database and make sure the schema exists."""
    db_path = Path(db_path)
    if str(db_path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    init_schema(con)
    return con


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    sql = resources.files("storage").joinpath("schema.sql").read_text(encoding="utf-8")
    con.execute(sql)


# On conflict, refresh only the fields that legitimately change over time.
# Title/body are left alone so a later "[removed]" or edit can't wipe the text
# we originally scored; created_at/collected_at keep their first-seen values.
_UPSERT_SQL = f"""
INSERT INTO posts ({", ".join(POST_COLUMNS)})
VALUES ({", ".join("?" for _ in POST_COLUMNS)})
ON CONFLICT (id) DO UPDATE SET
    engagement      = excluded.engagement,
    author_age_days = COALESCE(excluded.author_age_days, posts.author_age_days),
    author_karma    = COALESCE(excluded.author_karma, posts.author_karma)
"""


def upsert_posts(con: duckdb.DuckDBPyConnection, posts: Iterable[Post]) -> tuple[int, int]:
    """Insert new posts and refresh existing ones, atomically.

    Returns (inserted, updated). Runs in one transaction so a crash mid-batch
    leaves the table exactly as it was.
    """
    # Duplicates within a batch (e.g. a post seen twice in one listing) would
    # make DuckDB's ON CONFLICT error out, so keep the last copy of each id.
    batch = {p.id: p for p in posts}
    if not batch:
        return 0, 0
    ids = list(batch)
    con.execute("BEGIN TRANSACTION")
    try:
        existing = {
            r[0]
            for r in con.execute(
                "SELECT id FROM posts WHERE id IN (SELECT unnest(?))", [ids]
            ).fetchall()
        }
        con.executemany(
            _UPSERT_SQL,
            [[p.as_row()[c] for c in POST_COLUMNS] for p in batch.values()],
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    updated = len(existing)
    return len(batch) - updated, updated


def posts_without_tickers(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str, str]]:
    """(id, title, body) of posts with no post_tickers rows yet.

    Posts that matched nothing get re-checked on later runs. That's cheap
    (regex only) and means a universe refresh picks up newly listed tickers.
    """
    return con.execute(
        """
        SELECT id, title, body FROM posts p
        WHERE NOT EXISTS (SELECT 1 FROM post_tickers t WHERE t.post_id = p.id)
        """
    ).fetchall()


def all_posts_text(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str, str]]:
    return con.execute("SELECT id, title, body FROM posts").fetchall()


def replace_post_tickers(
    con: duckdb.DuckDBPyConnection, matches: dict[str, list]
) -> int:
    """Set the ticker matches for each given post, replacing any old ones.

    `matches` maps post_id -> list of Match (ticker, match_type, confidence).
    Replace rather than append, so re-extracting after an alias or blocklist
    change can also remove matches. Returns the number of rows written.
    """
    if not matches:
        return 0
    rows = [
        [post_id, m.ticker, m.match_type, m.confidence]
        for post_id, ms in matches.items()
        for m in ms
    ]
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute("DELETE FROM post_tickers WHERE post_id IN (SELECT unnest(?))", [list(matches)])
        if rows:
            con.executemany(
                "INSERT INTO post_tickers (post_id, ticker, match_type, confidence) VALUES (?, ?, ?, ?)",
                rows,
            )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(rows)



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

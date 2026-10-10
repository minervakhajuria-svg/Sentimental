"""DuckDB access: connection, schema setup and idempotent post upserts."""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

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
    """(id, title, body) of posts the extractor hasn't tagged yet.

    Rows the source itself supplied (match_type 'source') don't count: the text
    may mention other tickers too. Posts that matched nothing get re-checked on later runs. That's cheap
    (regex only) and means a universe refresh picks up newly listed tickers.
    """
    return con.execute(
        """
        SELECT id, title, body FROM posts p
        WHERE NOT EXISTS (SELECT 1 FROM post_tickers t
                          WHERE t.post_id = p.id AND t.match_type <> 'source')
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
    change can also remove matches. Source-supplied rows are kept, and win
    over an extractor match for the same ticker. Returns rows written.
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
        con.execute("DELETE FROM post_tickers WHERE post_id IN (SELECT unnest(?)) "
                    "AND match_type <> 'source'", [list(matches)])
        if rows:
            con.executemany(
                "INSERT INTO post_tickers (post_id, ticker, match_type, confidence) VALUES (?, ?, ?, ?) "
                "ON CONFLICT DO NOTHING",
                rows,
            )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(rows)


def pairs_to_score(con: duckdb.DuckDBPyConnection, model: str) -> list[tuple]:
    """(post_id, ticker, title, body, n_tickers_in_post) with no score from `model` yet.

    post_scores is the cache: anything already scored is skipped, so rerunning
    only pays for new posts.
    """
    return con.execute(
        """
        SELECT pt.post_id, pt.ticker, p.title, p.body,
               count(*) OVER (PARTITION BY pt.post_id) AS n_tickers
        FROM post_tickers pt
        JOIN posts p ON p.id = pt.post_id
        WHERE NOT EXISTS (
            SELECT 1 FROM post_scores s
            WHERE s.post_id = pt.post_id AND s.ticker = pt.ticker AND s.model = ?
        )
        ORDER BY p.created_at, pt.post_id, pt.ticker
        """,
        [model],
    ).fetchall()


def insert_scores(con: duckdb.DuckDBPyConnection, rows: list[tuple]) -> int:
    """Insert (post_id, ticker, model, label, score, scored_at) rows.

    Existing scores are never overwritten (CLAUDE.md §8): a later model gets
    its own `model` value instead.
    """
    if not rows:
        return 0
    con.execute("BEGIN TRANSACTION")
    try:
        con.executemany(
            """INSERT INTO post_scores (post_id, ticker, model, label, score, scored_at)
               VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING""",
            rows,
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(rows)


PRICE_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]


def upsert_prices(con: duckdb.DuckDBPyConnection, bars) -> int:
    """Insert or overwrite daily bars (a DataFrame with PRICE_COLUMNS).

    Overwrite, not skip: adjusted prices change after splits and dividends, and
    the latest download is the consistent one.
    """
    if bars is None or len(bars) == 0:
        return 0
    # One set-based statement: DuckDB runs executemany as a statement per row,
    # which took 45+ minutes for ~50,000 bars.
    df = bars[PRICE_COLUMNS].drop_duplicates(["ticker", "date"], keep="last").copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["volume"] = df["volume"].round().astype("Int64")
    con.register("_bars", df)
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            f"""INSERT INTO prices_daily ({", ".join(PRICE_COLUMNS)})
                SELECT {", ".join(PRICE_COLUMNS)} FROM _bars
                ON CONFLICT (ticker, date) DO UPDATE SET
                    open = excluded.open, high = excluded.high, low = excluded.low,
                    close = excluded.close, volume = excluded.volume""")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister("_bars")
    return len(df)


def load_prices(con: duckdb.DuckDBPyConnection, tickers: list[str], start, end):
    """Bars for `tickers` with start <= date <= end, as a DataFrame."""
    return con.execute(
        f"""SELECT {", ".join(PRICE_COLUMNS)} FROM prices_daily
            WHERE ticker IN (SELECT unnest(?)) AND date >= ? AND date <= ?
            ORDER BY ticker, date""",
        [list(tickers), start, end],
    ).df().assign(date=lambda d: pd.to_datetime(d["date"]).dt.date)


def recently_mentioned(con: duckdb.DuckDBPyConnection, since) -> list[str]:
    return [r[0] for r in con.execute(
        """SELECT DISTINCT pt.ticker FROM post_tickers pt JOIN posts p ON p.id = pt.post_id
           WHERE p.created_at >= ? ORDER BY 1""",
        [since],
    ).fetchall()]


def add_source_tags(con: duckdb.DuckDBPyConnection, posts, confidence: float,
                    universe: set[str] | None = None) -> tuple[int, int]:
    """Store tickers and sentiment tags that the source itself supplied.

    Tickers become post_tickers rows with match_type 'source' (replacing any
    extractor match for the same ticker). Tags become post_scores rows under
    model '<source>_tag', never touching the FinBERT scores. Tickers outside
    `universe` (when given) are skipped. Returns (ticker rows, tag rows).
    """
    tick_rows, tag_rows = [], []
    for p in posts:
        tickers = [t for t in dict.fromkeys(p.source_tickers) if universe is None or t in universe]
        for t in tickers:
            tick_rows.append([p.id, t, "source", confidence])
            if p.source_sentiment is not None:
                label = "pos" if p.source_sentiment > 0 else "neg"
                tag_rows.append([p.id, t, f"{p.source}_tag", label, float(p.source_sentiment), p.collected_at])
    if not tick_rows:
        return 0, 0
    con.execute("BEGIN TRANSACTION")
    try:
        con.executemany(
            """INSERT INTO post_tickers (post_id, ticker, match_type, confidence) VALUES (?, ?, ?, ?)
               ON CONFLICT (post_id, ticker) DO UPDATE SET
                   match_type = excluded.match_type, confidence = excluded.confidence""",
            tick_rows,
        )
        if tag_rows:
            con.executemany(
                "INSERT INTO post_scores VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING", tag_rows)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(tick_rows), len(tag_rows)


def top_liquid_tickers(con: duckdb.DuckDBPyConnection, n: int, cfg: dict) -> list[str]:
    """The n most traded tickers that pass the universe floors."""
    return [r[0] for r in con.execute(
        """SELECT ticker FROM ticker_universe
           WHERE exchange IN (SELECT unnest(?)) AND market_cap >= ? AND avg_dollar_volume_30d >= ?
           ORDER BY avg_dollar_volume_30d DESC LIMIT ?""",
        [cfg["exchanges"], cfg["min_market_cap"], cfg["min_avg_dollar_volume"], n],
    ).fetchall()]


def most_mentioned(con: duckdb.DuckDBPyConnection, since, n: int) -> list[str]:
    return [r[0] for r in con.execute(
        """SELECT pt.ticker FROM post_tickers pt JOIN posts p ON p.id = pt.post_id
           WHERE p.created_at >= ? GROUP BY 1 ORDER BY count(*) DESC LIMIT ?""",
        [since, n],
    ).fetchall()]


# Per (post, ticker), the score from the first model in a priority list that has
# one, e.g. ["claude", "finbert"]: Claude's second-pass score where it exists,
# FinBERT otherwise. Binds ONE parameter (the list), so it drops into an existing
# query where a single `s.model = ?` used to be without reordering parameters.
EFFECTIVE_SCORES = """(
    SELECT ps.post_id, ps.ticker, ps.label, ps.score
    FROM post_scores ps, (SELECT ? AS models) m
    WHERE list_position(m.models, ps.model) > 0
    QUALIFY row_number() OVER (PARTITION BY ps.post_id, ps.ticker
                               ORDER BY list_position(m.models, ps.model)) = 1
)"""


def as_models(model) -> list[str]:
    """Accept one model name or a priority list."""
    return [model] if isinstance(model, str) else list(model)


def universe_by_liquidity(con: duckdb.DuckDBPyConnection, cfg: dict) -> list[tuple[str, str]]:
    """(ticker, company_name) passing the floors, most traded first."""
    return con.execute(
        """SELECT ticker, company_name FROM ticker_universe
           WHERE exchange IN (SELECT unnest(?)) AND market_cap >= ? AND avg_dollar_volume_30d >= ?
           ORDER BY avg_dollar_volume_30d DESC""",
        [cfg["exchanges"], cfg["min_market_cap"], cfg["min_avg_dollar_volume"]],
    ).fetchall()

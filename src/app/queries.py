"""Read-only queries behind the Streamlit app.

Kept separate from the UI so they can be unit-tested without Streamlit, and
so the app never writes to the database.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import duckdb
import pandas as pd

from signals.aggregate import Week
from signals.composite import ranked_lists
from storage import db

DISPLAY_COLUMNS = [
    "ticker", "company_name", "composite", "mentions", "attention_z", "sentiment",
    "momentum", "breadth", "ret_5d", "ret_30d", "rel_volume", "early_chatter_flag",
    "insider_buy_flag",
]


def connect(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path, read_only=True)


def available_weeks(con: duckdb.DuckDBPyConnection) -> list[date]:
    """Weeks with stored signals, newest first."""
    return [r[0] for r in con.execute(
        "SELECT DISTINCT week_start FROM weekly_signals ORDER BY week_start DESC"
    ).fetchall()]


def week_for(week_start: date) -> Week:
    return Week.ending(week_start + timedelta(days=4))


def ranked_tables(con: duckdb.DuckDBPyConnection, week_start: date,
                  top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two ranked lists for a week, with company names and context columns."""
    names = dict(con.execute("SELECT ticker, company_name FROM ticker_universe").fetchall())

    def tidy(df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(company_name=df["ticker"].map(names))
        df.insert(0, "rank", range(1, len(df) + 1))
        return df[["rank"] + DISPLAY_COLUMNS]

    bull, bear = ranked_lists(con, week_start, top_n)
    return tidy(bull), tidy(bear)


def eligible_count(con: duckdb.DuckDBPyConnection, week_start: date) -> int:
    return con.execute("SELECT count(*) FROM weekly_signals WHERE week_start = ?",
                       [week_start]).fetchone()[0]


def tracked_tickers(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Every ticker that has ever been mentioned, for the drill-down picker."""
    return [r[0] for r in con.execute(
        "SELECT ticker FROM post_tickers GROUP BY ticker ORDER BY count(*) DESC"
    ).fetchall()]


def company_name(con: duckdb.DuckDBPyConnection, ticker: str) -> str | None:
    row = con.execute("SELECT company_name FROM ticker_universe WHERE ticker = ?",
                      [ticker]).fetchone()
    return row[0] if row else None


def daily_trend(con: duckdb.DuckDBPyConnection, ticker: str, end: datetime,
                days: int, model: str) -> pd.DataFrame:
    """Daily mentions, mean sentiment and close price for one ticker.

    Every day in the range appears, so quiet days show as zero mentions rather
    than gaps. Close is NULL until price data is collected (phase 5).
    """
    start = end - timedelta(days=days)
    return con.execute(
        f"""
        WITH days AS (
            SELECT CAST(d AS DATE) AS day
            FROM generate_series(CAST(? AS DATE), CAST(? AS DATE) - INTERVAL 1 DAY,
                                 INTERVAL 1 DAY) t(d)
        ),
        m AS (
            SELECT CAST(p.created_at AS DATE) AS day, count(*) AS mentions,
                   avg(s.score) AS sentiment
            FROM post_tickers pt
            JOIN posts p ON p.id = pt.post_id
            LEFT JOIN {db.EFFECTIVE_SCORES} s
                   ON s.post_id = pt.post_id AND s.ticker = pt.ticker
            WHERE pt.ticker = ? AND p.created_at >= ? AND p.created_at < ?
            GROUP BY 1
        )
        SELECT days.day, coalesce(m.mentions, 0) AS mentions, m.sentiment, pr.close
        FROM days
        LEFT JOIN m ON m.day = days.day
        LEFT JOIN prices_daily pr ON pr.ticker = ? AND pr.date = days.day
        ORDER BY days.day
        """,
        [start, end, db.as_models(model), ticker, start, end, ticker],
    ).df()


def weekly_history(con: duckdb.DuckDBPyConnection, ticker: str) -> pd.DataFrame:
    """The ticker's stored weekly signals, oldest first (only weeks it was eligible)."""
    return con.execute(
        """
        SELECT week_start, mentions, attention_z, sentiment, momentum,
               composite_bull, composite_bear
        FROM weekly_signals WHERE ticker = ? ORDER BY week_start
        """,
        [ticker],
    ).df()


def top_posts(con: duckdb.DuckDBPyConnection, ticker: str, week: Week,
              model: str, limit: int = 20) -> pd.DataFrame:
    """Most-engaged posts about `ticker` in the week, so the user can read why it ranked.

    Reposts (same content hash) are shown once, matching the aggregation.
    """
    return con.execute(
        f"""
        SELECT p.created_at, p.source, p.community, coalesce(p.title, p.body) AS title,
               s.score AS sentiment, s.label,
               p.engagement, pt.match_type, p.url
        FROM post_tickers pt
        JOIN posts p ON p.id = pt.post_id
        LEFT JOIN {db.EFFECTIVE_SCORES} s
               ON s.post_id = pt.post_id AND s.ticker = pt.ticker
        WHERE pt.ticker = ? AND p.created_at >= ? AND p.created_at < ?
        QUALIFY row_number() OVER (PARTITION BY coalesce(p.content_hash, p.id)
                                   ORDER BY p.created_at) = 1
        ORDER BY p.engagement DESC NULLS LAST, p.created_at DESC
        LIMIT ?
        """,
        [db.as_models(model), ticker, week.start, week.end, limit],
    ).df()


def week_tape(con: duckdb.DuckDBPyConnection, week_start: date) -> pd.DataFrame:
    """Every eligible ticker that week with its sentiment, busiest first (for the ticker tape)."""
    return con.execute(
        """
        SELECT ticker, sentiment, momentum, mentions FROM weekly_signals
        WHERE week_start = ? ORDER BY mentions DESC
        """,
        [week_start],
    ).df()


def week_summary(con: duckdb.DuckDBPyConnection, ticker: str, week: Week, model: str) -> dict:
    """Plain (unweighted) post counts and label mix for one ticker and week.

    The ranked sentiment in weekly_signals is engagement-weighted and filtered;
    this is the simple tally, used for the mention-mix bar and as the gauge
    fallback when the ticker wasn't eligible that week.
    """
    row = con.execute(
        f"""
        SELECT count(*) AS posts,
               avg(s.score) AS mean_score,
               count(*) FILTER (WHERE s.label = 'pos') AS pos,
               count(*) FILTER (WHERE s.label = 'neu') AS neu,
               count(*) FILTER (WHERE s.label = 'neg') AS neg
        FROM post_tickers pt
        JOIN posts p ON p.id = pt.post_id
        LEFT JOIN {db.EFFECTIVE_SCORES} s
               ON s.post_id = pt.post_id AND s.ticker = pt.ticker
        WHERE pt.ticker = ? AND p.created_at >= ? AND p.created_at < ?
        """,
        [db.as_models(model), ticker, week.start, week.end],
    ).fetchone()
    summary = dict(zip(["posts", "mean_score", "pos", "neu", "neg"], row))
    cols = ["sentiment", "momentum", "ret_5d", "ret_30d", "rel_volume", "updown_vol_ratio",
            "early_chatter_flag", "insider_buy_flag"]
    signal = con.execute(
        f"SELECT {', '.join(cols)} FROM weekly_signals WHERE week_start = ? AND ticker = ?",
        [week.week_start, ticker],
    ).fetchone()
    values = dict(zip(cols, signal)) if signal else dict.fromkeys(cols)
    summary["ranked_sentiment"] = values.pop("sentiment")
    summary.update(values)
    return summary


def community_breakdown(con: duckdb.DuckDBPyConnection, ticker: str, week: Week,
                        model: str) -> pd.DataFrame:
    """Posts and mean sentiment per community for one ticker and week."""
    return con.execute(
        f"""
        SELECT p.source, p.community, count(*) AS posts, avg(s.score) AS sentiment
        FROM post_tickers pt
        JOIN posts p ON p.id = pt.post_id
        LEFT JOIN {db.EFFECTIVE_SCORES} s
               ON s.post_id = pt.post_id AND s.ticker = pt.ticker
        WHERE pt.ticker = ? AND p.created_at >= ? AND p.created_at < ?
        GROUP BY 1, 2 ORDER BY posts DESC
        """,
        [db.as_models(model), ticker, week.start, week.end],
    ).df()


def daily_prices(con: duckdb.DuckDBPyConnection, ticker: str, end: datetime,
                 days: int) -> pd.DataFrame:
    """Close and volume for the drill-down price chart (sessions only)."""
    return con.execute(
        """SELECT date, close, volume FROM prices_daily
           WHERE ticker = ? AND date >= CAST(? AS DATE) AND date < CAST(? AS DATE)
           ORDER BY date""",
        [ticker, end - timedelta(days=days), end],
    ).df()

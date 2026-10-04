"""Per-ticker weekly components: attention, sentiment, momentum, breadth (CLAUDE.md §10).

Week definition: a "week" is the 7 days from Saturday 00:00 UTC through
Friday 23:59 UTC. That captures weekend chatter ahead of Monday's open and
ends after Friday's US close. It is labelled by its Monday (`week_start`).
Only posts created before the window end are read, so nothing from after
ranking time can leak in.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import duckdb
import numpy as np
import pandas as pd

from storage import db
from signals.filters import apply_noise_filters, cap_per_author


@dataclass(frozen=True)
class Week:
    friday: date

    @property
    def start(self) -> datetime:  # Saturday 00:00 UTC
        return datetime.combine(self.friday - timedelta(days=6), datetime.min.time())

    @property
    def end(self) -> datetime:  # following Saturday 00:00 UTC (exclusive)
        return datetime.combine(self.friday + timedelta(days=1), datetime.min.time())

    @property
    def week_start(self) -> date:  # the Monday, used as the table key
        return self.friday - timedelta(days=4)

    @classmethod
    def latest_completed(cls, now: datetime) -> "Week":
        """The most recent week whose window has fully closed by `now`."""
        d = now.date() - timedelta(days=1)  # a week ending today hasn't closed yet
        return cls(d - timedelta(days=(d.weekday() - 4) % 7))

    @classmethod
    def ending(cls, friday: date) -> "Week":
        if friday.weekday() != 4:
            raise ValueError(f"{friday} is not a Friday")
        return cls(friday)


def load_mentions(con: duckdb.DuckDBPyConnection, start: datetime, end: datetime,
                  model: str, min_confidence: float) -> pd.DataFrame:
    """One row per (post, ticker) in [start, end), with its sentiment if scored.

    `model` is one model name or a priority list (first available score wins).
    """
    return con.execute(
        f"""
        SELECT p.id AS post_id, pt.ticker, p.source, p.community, p.author_id,
               p.author_age_days, p.author_karma, p.created_at, p.engagement,
               p.content_hash, p.title, p.body, pt.confidence, s.score
        FROM posts p
        JOIN post_tickers pt ON pt.post_id = p.id
        LEFT JOIN {db.EFFECTIVE_SCORES} s
               ON s.post_id = pt.post_id AND s.ticker = pt.ticker
        WHERE p.created_at >= ? AND p.created_at < ? AND pt.confidence >= ?
        """,
        [db.as_models(model), start, end, min_confidence],
    ).df()


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna()
    v, w = values[mask], weights[mask]
    if v.empty:
        return np.nan
    # If every post has zero engagement, fall back to a plain mean.
    return float(np.average(v, weights=w)) if w.sum() > 0 else float(v.mean())


def compute_components(con: duckdb.DuckDBPyConnection, week: Week, cfg: dict) -> pd.DataFrame:
    """Per-ticker components for `week`, before eligibility filtering.

    Columns: mentions, n_authors, attention_z, sentiment, sentiment_prev,
    momentum, breadth.
    """
    baseline_days = cfg["baseline_days"]
    start = week.start - timedelta(days=baseline_days)
    models = cfg.get("sentiment_models") or cfg["sentiment_model"]
    df = load_mentions(con, start, week.end, models, cfg["min_match_confidence"])
    if df.empty:
        return pd.DataFrame()

    age = week.end - pd.to_datetime(df["created_at"])
    df["day"] = (age // pd.Timedelta(days=1)).astype(int)  # 0..6 = this week
    df["block"] = df["day"] // 7  # 7-day blocks counting back: 0 = this week, 1 = last week
    df = apply_noise_filters(df, cfg)
    df = cap_per_author(df, cfg["author_cap_per_ticker_week"], period_col="block")

    tickers = sorted(df.loc[df["block"] == 0, "ticker"].unique())
    if not tickers:
        return pd.DataFrame()

    # Attention: daily weighted mentions, zero-filled so quiet days count.
    n_days = 7 + baseline_days
    daily = (
        df.groupby(["ticker", "day"])["weight"].sum()
        .unstack(fill_value=0.0)
        .reindex(index=tickers, columns=range(n_days), fill_value=0.0)
    )
    this_mean = daily[list(range(7))].mean(axis=1)
    base = daily[list(range(7, n_days))]
    base_std = base.std(axis=1, ddof=0).clip(lower=cfg["attention_std_floor"])
    clip = cfg["attention_clip"]
    attention = ((this_mean - base.mean(axis=1)) / base_std).clip(-clip, clip)

    this_week = df[df["block"] == 0]
    last_week = df[df["block"] == 1]
    rows = []
    for t, g in this_week.groupby("ticker"):
        prev = last_week[last_week["ticker"] == t]
        s = _weighted_mean(g["score"], g["weight"])
        s_prev = _weighted_mean(prev["score"], prev["weight"]) if not prev.empty else np.nan
        n_authors = g["author_id"].nunique()
        rows.append({
            "ticker": t,
            "mentions": len(g),
            "n_authors": n_authors,
            "attention_z": float(attention[t]),
            "sentiment": s,
            "sentiment_prev": s_prev,
            "momentum": s - s_prev if not np.isnan(s_prev) else np.nan,
            # log-scaled so 1 -> 10 authors matters more than 100 -> 110
            "breadth": float(np.log1p(n_authors) + np.log1p(g["community"].nunique())
                             + np.log1p(g["source"].nunique())),
        })
    return pd.DataFrame(rows).set_index("ticker")

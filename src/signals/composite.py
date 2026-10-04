"""Eligibility, cross-sectional composite and the two ranked lists (CLAUDE.md §10).

Only sentiment and attention components go into the composite. Price, volume
and insider data must never be added here (they're context columns only).
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from signals.aggregate import Week

SIGNAL_COLUMNS = [
    "week_start", "ticker", "mentions", "attention_z", "sentiment", "sentiment_prev",
    "momentum", "breadth", "composite_bull", "composite_bear", "ret_5d", "ret_30d",
    "rel_volume", "updown_vol_ratio", "insider_buy_flag", "early_chatter_flag",
]


def apply_eligibility(components: pd.DataFrame, universe: set[str], cfg: dict) -> pd.DataFrame:
    """Keep tickers in the floor-passing universe with enough mentions and authors."""
    if components.empty:
        return components
    keep = (
        components.index.isin(list(universe))
        & (components["mentions"] >= cfg["min_weekly_mentions"])
        & (components["n_authors"] >= cfg["min_distinct_authors"])
        & components["sentiment"].notna()
    )
    return components[keep]


def zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score. Missing or constant inputs contribute 0."""
    std = s.std(ddof=0)
    if s.notna().sum() < 2 or not std or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / std).fillna(0.0)


def add_composites(eligible: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    za, zs = zscore(eligible["attention_z"]), zscore(eligible["sentiment"])
    zm, zb = zscore(eligible["momentum"]), zscore(eligible["breadth"])
    w = weights
    return eligible.assign(
        composite_bull=w["attention"] * za + w["sentiment"] * zs + w["momentum"] * zm + w["breadth"] * zb,
        composite_bear=w["attention"] * za - w["sentiment"] * zs - w["momentum"] * zm + w["breadth"] * zb,
    )


def write_signals(con: duckdb.DuckDBPyConnection, week: Week, signals: pd.DataFrame) -> int:
    """Replace this week's rows, so re-running a week is idempotent."""
    rows = []
    for ticker, r in signals.iterrows():
        rec = {c: None for c in SIGNAL_COLUMNS}
        rec.update(week_start=week.week_start, ticker=ticker)
        for c in ("mentions", "attention_z", "sentiment", "sentiment_prev", "momentum",
                  "breadth", "composite_bull", "composite_bear", "ret_5d", "ret_30d",
                  "rel_volume", "updown_vol_ratio"):
            v = r.get(c)
            rec[c] = None if v is None or pd.isna(v) else (int(v) if c == "mentions" else float(v))
        for c in ("insider_buy_flag", "early_chatter_flag"):
            v = r.get(c)
            rec[c] = None if v is None or pd.isna(v) else bool(v)
        rows.append([rec[c] for c in SIGNAL_COLUMNS])
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute("DELETE FROM weekly_signals WHERE week_start = ?", [week.week_start])
        if rows:
            con.executemany(
                f"INSERT INTO weekly_signals ({', '.join(SIGNAL_COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(SIGNAL_COLUMNS))})",
                rows,
            )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(rows)


def ranked_lists(con: duckdb.DuckDBPyConnection, week_start, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(bullish, bearish) lists for a stored week.

    Bullish: S > 0 and A > 0, by composite_bull. Bearish: S < 0 and A > 0, by
    composite_bear. "Heating up" means attention is above the ticker's own baseline.
    """
    q = """
        SELECT ticker, composite_{side} AS composite, mentions, attention_z, sentiment,
               momentum, breadth, ret_5d, ret_30d, rel_volume, early_chatter_flag, insider_buy_flag
        FROM weekly_signals
        WHERE week_start = ? AND attention_z > 0 AND sentiment {cmp} 0
        ORDER BY composite_{side} DESC
        LIMIT ?
    """
    bull = con.execute(q.format(side="bull", cmp=">"), [week_start, top_n]).df()
    bear = con.execute(q.format(side="bear", cmp="<"), [week_start, top_n]).df()
    return bull, bear

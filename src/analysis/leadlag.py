"""Does chatter lead price, or follow it? (CLAUDE.md §12)

Builds a weekly panel: each ranked (week, ticker) with its signals and three
returns measured from Friday closes:

  ret_prior  the week before the signal week  (did price move first?)
  ret_same   the signal week itself
  ret_next   the week after                    (does the signal predict?)

Then, for every signal and every return, the cross-sectional Spearman rank
correlation is computed week by week and averaged. If sentiment mostly
follows price, its correlation with ret_prior will beat ret_next.

No lookahead: signals in weekly_signals were computed from data up to each
week's Friday; ret_next is the target being predicted, never an input.
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

SIGNALS = ["sentiment", "momentum", "attention_z", "mentions", "composite_bull"]
HORIZONS = {"ret_prior": "Prior week", "ret_same": "Same week", "ret_next": "Next week"}
NAMES = {"sentiment": "Sentiment", "momentum": "Momentum", "attention_z": "Attention",
         "mentions": "Mentions", "composite_bull": "Bull composite"}


def friday_closes(prices: pd.DataFrame, fridays: list[date]) -> pd.DataFrame:
    """Close on the last session on or before each Friday, per ticker (holidays handled)."""
    if prices.empty:
        return pd.DataFrame(columns=["ticker", "friday", "close"])
    p = prices[["ticker", "date", "close"]].copy()
    # Same datetime precision on both sides, or merge_asof refuses to join.
    p["date"] = pd.to_datetime(p["date"]).astype("datetime64[ns]")
    p = p.sort_values("date")
    f = pd.DataFrame({"friday": pd.to_datetime(sorted(fridays)).astype("datetime64[ns]")})
    out = []
    for ticker, g in p.groupby("ticker"):
        m = pd.merge_asof(f, g[["date", "close"]], left_on="friday", right_on="date", direction="backward",
                          tolerance=pd.Timedelta(days=6))
        out.append(m.assign(ticker=ticker)[["ticker", "friday", "close"]])
    res = pd.concat(out, ignore_index=True)
    res["friday"] = res["friday"].dt.date
    return res


def build_panel(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """weekly_signals joined with prior/same/next weekly returns."""
    sig = con.execute(
        """SELECT week_start, ticker, mentions, attention_z, sentiment, momentum, breadth,
                  composite_bull, composite_bear, rel_volume
           FROM weekly_signals ORDER BY week_start, ticker"""
    ).df()
    if sig.empty:
        return sig
    sig["week_start"] = pd.to_datetime(sig["week_start"]).dt.date
    sig["friday"] = sig["week_start"].map(lambda d: d + timedelta(days=4))
    # Each signal week needs closes from two Fridays before to one Friday after.
    fridays = sorted({w + timedelta(weeks=k) for w in sig["friday"].unique() for k in (-2, -1, 0, 1)})
    prices = con.execute(
        """SELECT ticker, date, close FROM prices_daily
           WHERE ticker IN (SELECT DISTINCT ticker FROM weekly_signals)
             AND date >= ? AND date <= ?""",
        [fridays[0] - timedelta(days=7), fridays[-1]],
    ).df()
    closes = friday_closes(prices, fridays).set_index(["ticker", "friday"])["close"]

    def close(t, d):
        return closes.get((t, d), np.nan)

    t, f = sig["ticker"], sig["friday"]
    c0 = [close(a, b) for a, b in zip(t, f)]
    c_prev = [close(a, b - timedelta(weeks=1)) for a, b in zip(t, f)]
    c_prev2 = [close(a, b - timedelta(weeks=2)) for a, b in zip(t, f)]
    c_next = [close(a, b + timedelta(weeks=1)) for a, b in zip(t, f)]
    sig["ret_same"] = np.array(c0) / np.array(c_prev) - 1
    sig["ret_prior"] = np.array(c_prev) / np.array(c_prev2) - 1
    sig["ret_next"] = np.array(c_next) / np.array(c0) - 1
    return sig


def spearman(x: pd.Series, y: pd.Series) -> float:
    """Rank correlation without scipy: Pearson on ranks."""
    m = x.notna() & y.notna()
    if m.sum() < 3:
        return np.nan
    rx, ry = x[m].rank(), y[m].rank()
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def weekly_ic(panel: pd.DataFrame, signal: str, target: str, min_tickers: int) -> pd.Series:
    """Per-week cross-sectional Spearman correlation, indexed by week_start."""
    out = {}
    for week, g in panel.groupby("week_start"):
        if g[[signal, target]].dropna().shape[0] >= min_tickers:
            out[week] = spearman(g[signal], g[target])
    return pd.Series(out, dtype=float).dropna()


def summarise(ics: pd.Series) -> dict:
    n = len(ics)
    mean = float(ics.mean()) if n else np.nan
    sd = float(ics.std(ddof=1)) if n > 1 else np.nan
    t = mean / (sd / np.sqrt(n)) if n > 1 and sd and sd > 0 else np.nan
    return {"mean_ic": mean, "t_stat": t, "weeks": n,
            "positive_share": float((ics > 0).mean()) if n else np.nan}


def lead_lag_table(panel: pd.DataFrame, min_tickers: int, signals=SIGNALS) -> pd.DataFrame:
    """Rows: signal x horizon, with mean IC, t-stat, weeks."""
    rows = []
    for s in signals:
        if s not in panel:
            continue
        for h, label in HORIZONS.items():
            rows.append({"signal": s, "horizon": label, **summarise(weekly_ic(panel, s, h, min_tickers))})
    return pd.DataFrame(rows)


def verdict(table: pd.DataFrame, signal: str = "sentiment", min_t: float = 2.0) -> str:
    """Plain-language reading of the lead-lag table for one signal."""
    name = NAMES.get(signal, signal)
    t = table[table["signal"] == signal].set_index("horizon")
    if t.empty or t["weeks"].max() == 0:
        return "Not enough data yet."
    prior, nxt = t.loc["Prior week"], t.loc["Next week"]
    if np.isnan(prior["mean_ic"]) or np.isnan(nxt["mean_ic"]):
        return "Not enough data yet."
    sig_next = abs(nxt["t_stat"]) >= min_t if not np.isnan(nxt["t_stat"]) else False
    sig_prior = abs(prior["t_stat"]) >= min_t if not np.isnan(prior["t_stat"]) else False
    if sig_next and nxt["mean_ic"] > 0 and abs(nxt["mean_ic"]) >= abs(prior["mean_ic"]):
        return f"{name} leads price: it ranks next week's returns better than last week's."
    if sig_next and nxt["mean_ic"] < 0:
        return (f"{name} predicts next week's returns in reverse: it may work better "
                "as a contrarian or crowding indicator.")
    if sig_prior and abs(prior["mean_ic"]) > abs(nxt["mean_ic"]):
        return f"{name} mostly follows price: it lines up with last week's returns, not next week's."
    return f"No reliable relationship yet between {name.lower()} and returns."

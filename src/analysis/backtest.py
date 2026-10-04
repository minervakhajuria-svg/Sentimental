"""Backtest the weekly lists against next-week returns (CLAUDE.md §12).

Metrics, all on the panel from analysis.leadlag.build_panel:
  * IC: mean weekly Spearman correlation of composite vs next-week return
  * Quintile spread: mean next-week return of the top composite quintile
    minus the bottom quintile, each week (needs >= 5 tickers that week)
  * Hit rate: share of list picks whose next-week return beats that week's
    cross-sectional median (bull) or trails it (bear), vs a naive baseline
    that picks the most-mentioned tickers instead
  * rel_volume experiment: does composite + w * z(rel_volume) rank next-week
    returns better than the composite alone? If yes, rel_volume earns a place
    in the composite; if not, it stays a context column.

Results are descriptive. With few weeks they're noise; the report says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.leadlag import summarise, weekly_ic


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0


def quintile_spread(panel: pd.DataFrame, signal: str, min_tickers: int = 5) -> pd.Series:
    """Per-week (top quintile - bottom quintile) mean next-week return."""
    out = {}
    for week, g in panel.dropna(subset=[signal, "ret_next"]).groupby("week_start"):
        if len(g) < max(min_tickers, 5):
            continue
        q = pd.qcut(g[signal].rank(method="first"), 5, labels=False)
        out[week] = g.loc[q == 4, "ret_next"].mean() - g.loc[q == 0, "ret_next"].mean()
    return pd.Series(out, dtype=float)


def hit_rate(panel: pd.DataFrame, pick: str, n: int, side: str, require=None) -> dict:
    """Share of the top-n picks by `pick` that beat (bull) or trail (bear) the weekly median."""
    hits = total = 0
    weeks = 0
    for _, g in panel.dropna(subset=["ret_next"]).groupby("week_start"):
        median = g["ret_next"].median()
        cand = g if require is None else g[require(g)]
        picks = cand.nlargest(n, pick)
        if picks.empty:
            continue
        weeks += 1
        good = picks["ret_next"] > median if side == "bull" else picks["ret_next"] < median
        hits += int(good.sum())
        total += len(picks)
    return {"hit_rate": hits / total if total else np.nan, "picks": total, "weeks": weeks}


def rel_volume_experiment(panel: pd.DataFrame, weight: float, min_tickers: int) -> dict:
    """IC of composite_bull vs composite_bull + weight * z(rel_volume), same weeks only."""
    p = panel.dropna(subset=["composite_bull", "rel_volume", "ret_next"]).copy()
    if p.empty:
        return {"base": summarise(pd.Series(dtype=float)), "with_rel_volume": summarise(pd.Series(dtype=float)),
                "improves": None}
    p["composite_plus"] = p["composite_bull"] + weight * p.groupby("week_start")["rel_volume"].transform(_z)
    base = summarise(weekly_ic(p, "composite_bull", "ret_next", min_tickers))
    plus = summarise(weekly_ic(p, "composite_plus", "ret_next", min_tickers))
    improves = (None if np.isnan(base["mean_ic"]) or np.isnan(plus["mean_ic"])
                else bool(plus["mean_ic"] > base["mean_ic"]))
    return {"base": base, "with_rel_volume": plus, "improves": improves}


def run_backtest(panel: pd.DataFrame, cfg: dict) -> dict:
    """Everything the report and the app's validation page need."""
    n, min_t = cfg["top_n"], cfg["min_tickers_per_week"]
    bull_ok = lambda g: (g["sentiment"] > 0) & (g["attention_z"] > 0)  # noqa: E731
    bear_ok = lambda g: (g["sentiment"] < 0) & (g["attention_z"] > 0)  # noqa: E731
    ic_bull = weekly_ic(panel, "composite_bull", "ret_next", min_t)
    ic_bear = weekly_ic(panel, "composite_bear", "ret_next", min_t)
    spread = quintile_spread(panel, "composite_bull", min_t)
    return {
        "weeks_with_returns": int(panel.dropna(subset=["ret_next"])["week_start"].nunique()) if len(panel) else 0,
        "ic_bull": summarise(ic_bull),
        "ic_bear": summarise(ic_bear),  # expected negative if the bear list works
        "ic_bull_series": ic_bull,
        "quintile_spread": {"mean": float(spread.mean()) if len(spread) else np.nan,
                            "weeks": len(spread)},
        "quintile_spread_series": spread,
        "hit_rates": {
            "bull_list": hit_rate(panel, "composite_bull", n, "bull", bull_ok),
            "bull_baseline_mentions": hit_rate(panel, "mentions", n, "bull"),
            "bear_list": hit_rate(panel, "composite_bear", n, "bear", bear_ok),
            "bear_baseline_mentions": hit_rate(panel, "mentions", n, "bear"),
        },
        "rel_volume_experiment": rel_volume_experiment(panel, cfg["rel_volume_weight"], min_t),
    }

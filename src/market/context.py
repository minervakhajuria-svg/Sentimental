"""Price/volume context columns for a ranked week (CLAUDE.md §11).

Display and validation only: none of this feeds the composite. Everything is
computed from bars dated on or before the week's Friday, so re-ranking an old
week can't see later prices.

Definitions:
  ret_5d            close on the last session <= Friday vs 5 sessions earlier
  ret_30d           same, vs 21 sessions earlier (about 30 calendar days)
  rel_volume        mean volume of this week's sessions / mean volume of the
                    sessions in the 30 calendar days before the week
  updown_vol_ratio  this week's volume on up-days / volume on down-days
                    (a rough buying-vs-selling proxy; true buy volume isn't
                    observable with free data). NULL if no down-days.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from signals.aggregate import Week

CONTEXT_COLUMNS = ["ret_5d", "ret_30d", "rel_volume", "updown_vol_ratio"]


def _ret(closes: pd.Series, sessions: int) -> float:
    if len(closes) <= sessions:
        return np.nan
    return float(closes.iloc[-1] / closes.iloc[-1 - sessions] - 1)


def ticker_context(bars: pd.DataFrame, week: Week, cfg: dict) -> dict:
    """Context for one ticker. `bars` has date, close, volume (any order, may be empty)."""
    out = dict.fromkeys(CONTEXT_COLUMNS, np.nan)
    if bars.empty:
        return out
    b = bars[bars["date"] <= week.friday].sort_values("date")
    if b.empty:
        return out
    closes = b["close"].reset_index(drop=True)
    out["ret_5d"] = _ret(closes, cfg["ret_short_sessions"])
    out["ret_30d"] = _ret(closes, cfg["ret_long_sessions"])

    week_start = week.start.date()
    this_week = b[b["date"] >= week_start]
    baseline = b[(b["date"] < week_start)
                 & (b["date"] >= week_start - timedelta(days=cfg["rel_volume_baseline_days"]))]
    if not this_week.empty and not baseline.empty and baseline["volume"].mean() > 0:
        out["rel_volume"] = float(this_week["volume"].mean() / baseline["volume"].mean())

    # Up/down needs the previous session's close, including the one before the week.
    change = b["close"].diff()
    in_week = b["date"] >= week_start
    up = b.loc[in_week & (change > 0), "volume"].sum()
    down = b.loc[in_week & (change < 0), "volume"].sum()
    if down > 0:
        out["updown_vol_ratio"] = float(up / down)
    return out


def add_context(signals: pd.DataFrame, prices: pd.DataFrame, week: Week, cfg: dict) -> pd.DataFrame:
    """Attach context columns and early_chatter_flag to the week's signals.

    early_chatter_flag encodes the hypothesis that chatter precedes price: the
    composite (bull or bear) is in the top decile, price hasn't moved much yet,
    and volume is picking up.
    """
    if signals.empty:
        return signals.assign(**{c: np.nan for c in CONTEXT_COLUMNS}, early_chatter_flag=False)
    by_ticker = {t: g for t, g in prices.groupby("ticker")} if not prices.empty else {}
    empty = pd.DataFrame(columns=["date", "close", "volume"])
    ctx = pd.DataFrame(
        [ticker_context(by_ticker.get(t, empty), week, cfg) for t in signals.index],
        index=signals.index,
    )
    out = signals.join(ctx)

    ec = cfg["early_chatter"]
    # Top decile by rank, not by quantile value: with ties (e.g. all equal) a
    # quantile threshold would put every ticker in the "top" decile.
    n_top = max(1, int(np.ceil(len(out) * (1 - ec["top_quantile"]))))
    top = ((out["composite_bull"].rank(ascending=False, method="first") <= n_top)
           | (out["composite_bear"].rank(ascending=False, method="first") <= n_top))
    quiet_price = out["ret_5d"].abs() < ec["max_abs_ret_5d"]
    busy_volume = out["rel_volume"] > ec["min_rel_volume"]
    # Missing price data compares as False, so the flag needs real data to fire.
    out["early_chatter_flag"] = (top & quiet_price & busy_volume).astype(bool)
    return out


def divergence_warning(row, cfg: dict) -> bool:
    """Rising volume + falling price + bullish chatter: crowd may be buying a falling knife."""
    w = cfg["divergence_warning"]
    return bool(
        pd.notna(row.get("rel_volume")) and pd.notna(row.get("ret_5d"))
        and row["rel_volume"] > w["min_rel_volume"]
        and row["ret_5d"] < -w["min_price_drop"]
        and row.get("sentiment", 0) > 0
    )

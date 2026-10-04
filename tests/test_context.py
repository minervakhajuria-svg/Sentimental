from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from market.context import add_context, divergence_warning, ticker_context
from signals.aggregate import Week
from storage import db

WEEK = Week.ending(date(2026, 10, 2))  # Sat 26 Sep .. Fri 2 Oct
CFG = {
    "ret_short_sessions": 5, "ret_long_sessions": 21, "rel_volume_baseline_days": 30,
    "early_chatter": {"top_quantile": 0.9, "max_abs_ret_5d": 0.03, "min_rel_volume": 1.2},
    "divergence_warning": {"min_rel_volume": 1.2, "min_price_drop": 0.02},
}


def bars(closes, volumes, end=WEEK.friday):
    """Weekday bars ending on `end`, oldest first."""
    days, d = [], end
    while len(days) < len(closes):
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return pd.DataFrame({"date": days[::-1], "close": closes, "volume": volumes})


def test_returns_use_session_counts():
    closes = list(range(100, 130))  # 30 sessions, +1 per day
    ctx = ticker_context(bars(closes, [1] * 30), WEEK, CFG)
    assert ctx["ret_5d"] == pytest.approx(129 / 124 - 1)
    assert ctx["ret_30d"] == pytest.approx(129 / 108 - 1)


def test_rel_volume_and_updown():
    # 25 baseline sessions at volume 100, then this week's 5 sessions at 200.
    closes = [10] * 25 + [11, 10, 12, 13, 12]  # this week: up, down, up, up, down
    vols = [100] * 25 + [200, 300, 200, 200, 100]
    ctx = ticker_context(bars(closes, vols), WEEK, CFG)
    assert ctx["rel_volume"] == pytest.approx(200.0 / 100)
    assert ctx["updown_vol_ratio"] == pytest.approx((200 + 200 + 200) / (300 + 100))


def test_no_down_days_gives_null_ratio():
    ctx = ticker_context(bars(list(range(1, 31)), [1] * 30), WEEK, CFG)
    assert np.isnan(ctx["updown_vol_ratio"])


def test_bars_after_friday_are_ignored():
    b = bars([10] * 30, [1] * 30, end=WEEK.friday + timedelta(days=3))  # through next Monday
    b.loc[b.index[-1], "close"] = 1000  # a huge move after ranking time
    assert ticker_context(b, WEEK, CFG)["ret_5d"] == pytest.approx(0.0)


def test_short_history_gives_nulls():
    ctx = ticker_context(bars([10, 11], [1, 1]), WEEK, CFG)
    assert np.isnan(ctx["ret_5d"]) and np.isnan(ctx["ret_30d"])
    assert ticker_context(pd.DataFrame(columns=["date", "close", "volume"]), WEEK, CFG) == {
        "ret_5d": pytest.approx(np.nan, nan_ok=True), "ret_30d": pytest.approx(np.nan, nan_ok=True),
        "rel_volume": pytest.approx(np.nan, nan_ok=True), "updown_vol_ratio": pytest.approx(np.nan, nan_ok=True)}


def _signals(**cols):
    n = len(next(iter(cols.values())))
    base = {"composite_bull": [0.0] * n, "composite_bear": [0.0] * n, "sentiment": [0.1] * n}
    return pd.DataFrame({**base, **cols}, index=[f"T{i}" for i in range(n)])


def test_early_chatter_flag():
    # T0: top bull composite, flat price, busy volume -> flag.
    # T1: top bear composite, busy volume, but price already fell 10% -> no flag.
    sig = _signals(composite_bull=[3.0] + [0.0] * 9, composite_bear=[-3.0, 2.0] + [-1.0] * 8)
    flat = bars([10] * 25 + [10.0, 10.1, 10.0, 10.1, 10.1], [100] * 25 + [200] * 5)
    moved = bars([10] * 25 + [9.8, 9.6, 9.4, 9.2, 9.0], [100] * 25 + [200] * 5)
    prices = pd.concat([(flat if t == "T0" else moved).assign(ticker=t) for t in sig.index])
    out = add_context(sig, prices, WEEK, CFG)
    assert out["early_chatter_flag"].tolist() == [True] + [False] * 9
    assert out.loc["T1", "ret_5d"] == pytest.approx(9.0 / 10 - 1)


def test_early_chatter_needs_price_data():
    sig = _signals(composite_bull=[3.0, 0.0])
    out = add_context(sig, pd.DataFrame(columns=["ticker", "date", "close", "volume"]), WEEK, CFG)
    assert not out["early_chatter_flag"].any()
    assert out["ret_5d"].isna().all()


def test_divergence_warning():
    row = {"rel_volume": 1.5, "ret_5d": -0.04, "sentiment": 0.4}
    assert divergence_warning(row, CFG)
    assert not divergence_warning({**row, "sentiment": -0.2}, CFG)
    assert not divergence_warning({**row, "ret_5d": 0.02}, CFG)
    assert not divergence_warning({**row, "rel_volume": None}, CFG)


def test_price_upsert_overwrites(tmp_path):
    con = db.connect(tmp_path / "p.duckdb")
    row = {"ticker": "X", "date": date(2026, 10, 1), "open": 1.0, "high": 1.0, "low": 1.0,
           "close": 1.0, "volume": 10.0}
    db.upsert_prices(con, pd.DataFrame([row]))
    db.upsert_prices(con, pd.DataFrame([{**row, "close": 2.0}]))  # re-adjusted after a split
    got = db.load_prices(con, ["X"], date(2026, 9, 1), date(2026, 10, 2))
    assert got["close"].tolist() == [2.0]
    assert got["date"].tolist() == [date(2026, 10, 1)]
    con.close()


def test_top_decile_is_by_rank_even_with_ties():
    # All composites equal: exactly one ticker per side can be "top decile", not all ten.
    sig = _signals(composite_bull=[1.0] * 10, composite_bear=[1.0] * 10)
    flat = bars([10] * 25 + [10.0] * 5, [100] * 25 + [200] * 5)
    prices = pd.concat([flat.assign(ticker=t) for t in sig.index])
    assert add_context(sig, prices, WEEK, CFG)["early_chatter_flag"].sum() == 1


def test_divergence_ignores_small_moves():
    assert not divergence_warning({"rel_volume": 1.9, "ret_5d": -0.002, "sentiment": 0.5}, CFG)

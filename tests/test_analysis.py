"""Lead-lag and backtest on synthetic panels with a known answer."""

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from analysis.backtest import hit_rate, quintile_spread, rel_volume_experiment, run_backtest
from analysis.leadlag import build_panel, friday_closes, lead_lag_table, spearman, verdict
from jobs.run_analysis import analyse, format_report
from storage import db

ACFG = {"min_tickers_per_week": 5, "top_n": 5, "rel_volume_weight": 0.5}
FIRST_MONDAY = date(2026, 1, 5)


def synthetic_panel(kind: str, weeks: int = 20, tickers: int = 30, seed: int = 1) -> pd.DataFrame:
    """kind: 'leads' (sentiment predicts next week), 'follows' (sentiment echoes last week),
    'none' (unrelated), 'volume' (next week driven by rel_volume)."""
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(weeks):
        ws = FIRST_MONDAY + timedelta(weeks=w)
        for i in range(tickers):
            sentiment = rng.normal()
            ret_prior = rng.normal(0, 0.03)
            rel_volume = rng.uniform(0.5, 2.0)
            noise = rng.normal(0, 0.03)
            ret_next = {
                "leads": 0.02 * sentiment + noise,
                "follows": noise,
                "none": noise,
                "volume": 0.03 * (rel_volume - 1.25) + 0.002 * sentiment + noise,
            }[kind]
            if kind == "follows":
                sentiment = ret_prior * 30 + rng.normal(0, 0.3)
            rows.append({
                "week_start": ws, "ticker": f"T{i}", "sentiment": sentiment, "momentum": rng.normal(),
                "attention_z": abs(rng.normal()) + 0.1, "mentions": int(rng.integers(20, 200)),
                "composite_bull": sentiment, "composite_bear": -sentiment, "rel_volume": rel_volume,
                "ret_prior": ret_prior, "ret_same": rng.normal(0, 0.03), "ret_next": ret_next,
            })
    return pd.DataFrame(rows)


def test_spearman_basics():
    a = pd.Series([1, 2, 3, 4, 5.0])
    assert spearman(a, a ** 3) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)
    assert np.isnan(spearman(a, pd.Series([1.0] * 5)))  # constant
    assert np.isnan(spearman(pd.Series([1.0, 2.0]), pd.Series([2.0, 1.0])))  # too few


def test_leading_signal_is_detected():
    panel = synthetic_panel("leads")
    table = lead_lag_table(panel, 5, signals=["sentiment"])
    nxt = table.set_index("horizon").loc["Next week"]
    assert nxt["mean_ic"] > 0.2 and nxt["t_stat"] > 3
    assert "leads price" in verdict(table, "sentiment")


def test_following_signal_is_detected():
    table = lead_lag_table(synthetic_panel("follows"), 5, signals=["sentiment"])
    prior = table.set_index("horizon").loc["Prior week"]
    assert prior["mean_ic"] > 0.5
    assert "follows price" in verdict(table, "sentiment")


def test_no_relationship():
    table = lead_lag_table(synthetic_panel("none"), 5, signals=["sentiment"])
    assert "No reliable relationship" in verdict(table, "sentiment")


def test_contrarian_signal():
    panel = synthetic_panel("leads")
    panel["sentiment"] = -panel["sentiment"]
    assert "contrarian" in verdict(lead_lag_table(panel, 5, signals=["sentiment"]), "sentiment")


def test_quintile_spread_and_hit_rates():
    panel = synthetic_panel("leads")
    spread = quintile_spread(panel, "composite_bull")
    assert len(spread) == 20 and spread.mean() > 0.02
    bull = hit_rate(panel, "composite_bull", 5, "bull")
    base = hit_rate(panel, "mentions", 5, "bull")
    assert bull["hit_rate"] > 0.7 and abs(base["hit_rate"] - 0.5) < 0.15
    assert bull["picks"] == 100 and bull["weeks"] == 20


def test_rel_volume_experiment():
    helps = rel_volume_experiment(synthetic_panel("volume"), weight=1.0, min_tickers=5)
    assert helps["improves"] is True
    no_help = rel_volume_experiment(synthetic_panel("leads"), weight=1.0, min_tickers=5)
    assert no_help["improves"] is False


def test_small_weeks_are_skipped():
    panel = synthetic_panel("leads", tickers=3)
    assert lead_lag_table(panel, 5, signals=["sentiment"])["weeks"].max() == 0
    assert verdict(lead_lag_table(panel, 5, signals=["sentiment"]), "sentiment") == "Not enough data yet."


def test_run_backtest_shape():
    bt = run_backtest(synthetic_panel("leads"), ACFG)
    assert bt["weeks_with_returns"] == 20
    assert bt["ic_bull"]["mean_ic"] > 0 and bt["ic_bear"]["mean_ic"] < 0
    assert set(bt["hit_rates"]) == {"bull_list", "bull_baseline_mentions", "bear_list", "bear_baseline_mentions"}


# --- returns from stored prices --------------------------------------------------

def test_friday_close_falls_back_to_thursday_on_holidays():
    prices = pd.DataFrame({"ticker": ["X", "X"], "date": [date(2026, 4, 2), date(2026, 4, 9)],
                           "close": [10.0, 11.0]})  # 3 Apr 2026 (Good Friday) has no session
    out = friday_closes(prices, [date(2026, 4, 3), date(2026, 4, 10)])
    assert out["close"].tolist() == [10.0, 11.0]


def test_build_panel_returns(tmp_path):
    con = db.connect(tmp_path / "a.duckdb")
    # Fridays 18, 25 Sep, 2, 9 Oct 2026 with closes 100, 110, 99, 120.
    for d, c in ((date(2026, 9, 18), 100.0), (date(2026, 9, 25), 110.0),
                 (date(2026, 10, 2), 99.0), (date(2026, 10, 9), 120.0)):
        con.execute("INSERT INTO prices_daily VALUES ('X', ?, ?, ?, ?, ?, 1000)", [d, c, c, c, c])
    con.execute("INSERT INTO weekly_signals (week_start, ticker, mentions, sentiment, composite_bull) "
                "VALUES ('2026-09-28', 'X', 30, 0.4, 1.0)")
    p = build_panel(con).iloc[0]
    assert p["ret_prior"] == pytest.approx(110 / 100 - 1)
    assert p["ret_same"] == pytest.approx(99 / 110 - 1)
    assert p["ret_next"] == pytest.approx(120 / 99 - 1)
    con.close()


def test_analyse_and_report(tmp_path):
    import demo_data
    path = tmp_path / "demo.duckdb"
    demo_data.build(path)
    con = db.connect(path)
    cfg = {"analysis": {**ACFG, "min_weeks": 12}, "signals": {"top_n": 5}}
    r = analyse(con, cfg)
    con.close()
    assert r["weeks"] == 8 and r["enough_data"] is False
    text = format_report(r)
    assert "WARNING: fewer than 12 weeks" in text
    assert "not investment advice" in text
    assert "nan" not in text.lower().replace("n/a", "")


def test_report_with_no_weeks(tmp_path):
    con = db.connect(tmp_path / "e.duckdb")
    r = analyse(con, {"analysis": {**ACFG, "min_weeks": 12}, "signals": {"top_n": 5}})
    con.close()
    assert format_report(r) == "No ranked weeks yet."

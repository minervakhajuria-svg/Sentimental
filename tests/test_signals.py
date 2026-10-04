"""Weekly signals on synthetic data with known answers."""

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from collectors.base import Post
from jobs.rank_weekly import rank
from signals.aggregate import Week, compute_components
from signals.composite import add_composites, apply_eligibility, zscore
from signals.filters import add_post_weight, cap_per_author, drop_reposts
from storage import db
from tickers.extractor import Match
from tickers.universe import save_universe

WEEK = Week.ending(date(2026, 10, 2))  # Sat 26 Sep .. Fri 2 Oct

CONTEXT_CFG = {
    "ret_short_sessions": 5, "ret_long_sessions": 21, "rel_volume_baseline_days": 30,
    "early_chatter": {"top_quantile": 0.9, "max_abs_ret_5d": 0.03, "min_rel_volume": 1.2},
    "divergence_warning": {"min_rel_volume": 1.2, "min_price_drop": 0.02},
}

SCFG = {
    "sentiment_model": "fake",
    "min_match_confidence": 0.5,
    "source_weights": {"reddit": 1.0},
    "author_cap_per_ticker_week": 5,
    "baseline_days": 30,
    "attention_std_floor": 1.0,
    "attention_clip": 10,
    "min_weekly_mentions": 3,
    "min_distinct_authors": 2,
    "composite_weights": {"attention": 0.35, "sentiment": 0.35, "momentum": 0.15, "breadth": 0.15},
    "top_n": 10,
}


# --- Week ----------------------------------------------------------------

def test_week_window():
    assert WEEK.start == datetime(2026, 9, 26)
    assert WEEK.end == datetime(2026, 10, 3)
    assert WEEK.week_start == date(2026, 9, 28)  # Monday


@pytest.mark.parametrize("now,friday", [
    (datetime(2026, 10, 3, 9), date(2026, 10, 2)),   # Saturday: last week just closed
    (datetime(2026, 10, 4, 9), date(2026, 10, 2)),   # Sunday
    (datetime(2026, 10, 2, 23), date(2026, 9, 25)),  # Friday: current week still open
    (datetime(2026, 9, 30, 9), date(2026, 9, 25)),   # Wednesday
])
def test_latest_completed_week(now, friday):
    assert Week.latest_completed(now).friday == friday


def test_week_ending_must_be_friday():
    with pytest.raises(ValueError):
        Week.ending(date(2026, 10, 3))


# --- filters ---------------------------------------------------------------

def _frame(rows):
    base = {"source": "reddit", "community": "stocks", "engagement": 10,
            "created_at": datetime(2026, 10, 1), "content_hash": None, "block": 0}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_drop_reposts_keeps_earliest():
    df = _frame([
        {"post_id": "a", "ticker": "X", "content_hash": "h", "created_at": datetime(2026, 10, 2)},
        {"post_id": "b", "ticker": "X", "content_hash": "h", "created_at": datetime(2026, 10, 1)},
        {"post_id": "c", "ticker": "X", "content_hash": None},
    ])
    assert set(drop_reposts(df)["post_id"]) == {"b", "c"}


def test_post_weight_is_log_engagement():
    df = add_post_weight(_frame([{"engagement": 0}, {"engagement": np.e - 1}]),
                         {"source_weights": {"reddit": 2.0}})
    assert df["weight"].tolist() == pytest.approx([0.0, 2.0])


def test_cap_per_author_keeps_top_k_by_weight():
    df = _frame([{"post_id": str(i), "ticker": "X", "author_id": "spammer", "weight": float(i)}
                 for i in range(10)]
                + [{"post_id": "anon", "ticker": "X", "author_id": None, "weight": 1.0}])
    capped = cap_per_author(df, k=3, period_col="block")
    assert set(capped["post_id"]) == {"9", "8", "7", "anon"}


# --- components and composite on stored data ------------------------------------

class Builder:
    """Writes synthetic posts with tickers and scores straight into the DB."""

    def __init__(self, con):
        self.con, self.n = con, 0

    def add(self, ticker, when, score, author=None, engagement=10, community="stocks"):
        self.n += 1
        pid = f"reddit:{self.n}"
        db.upsert_posts(self.con, [Post(
            id=pid, source="reddit", community=community, author_id=author or f"u{self.n}",
            author_age_days=None, author_karma=None, created_at=when, collected_at=when,
            title=f"post {self.n}", body=None, url=None, engagement=engagement,
            content_hash=f"h{self.n}")])
        db.replace_post_tickers(self.con, {pid: [Match(ticker, "cashtag", 0.95)]})
        self.con.execute("INSERT INTO post_scores VALUES (?, ?, 'fake', 'neu', ?, now())",
                         [pid, ticker, score])


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "sig.duckdb")
    yield c
    c.close()


def _day(d):
    """Datetime `d` days before the week's end (0 = Friday)."""
    return WEEK.end - timedelta(days=d, hours=12)


def test_components_known_values(con):
    b = Builder(con)
    # HOT: quiet baseline (1 post every 3rd baseline day), busy this week.
    for d in range(7, 37, 3):
        b.add("HOT", _day(d), -0.2)
    for d in range(7):
        b.add("HOT", _day(d), 0.6, community="wallstreetbets")
        b.add("HOT", _day(d), 0.6)
    # last week's HOT sentiment is -0.2 (days 7..13 posts above)

    comp = compute_components(con, WEEK, SCFG)
    hot = comp.loc["HOT"]
    assert hot["mentions"] == 14
    assert hot["sentiment"] == pytest.approx(0.6)
    assert hot["sentiment_prev"] == pytest.approx(-0.2)
    assert hot["momentum"] == pytest.approx(0.8)
    assert hot["attention_z"] > 0
    w = np.log1p(10)
    this_mean, base_mean = 2 * w, 10 * w / 30
    base_std = max(np.std([w if d in range(0, 30, 3) else 0 for d in range(30)]), 1.0)
    assert hot["attention_z"] == pytest.approx((this_mean - base_mean) / base_std)
    assert hot["breadth"] == pytest.approx(np.log1p(14) + np.log1p(2) + np.log1p(1))


def test_posts_after_week_end_are_ignored(con):
    b = Builder(con)
    b.add("X", _day(1), 0.5)
    b.add("X", WEEK.end + timedelta(hours=1), -1.0)  # Saturday after: lookahead
    comp = compute_components(con, WEEK, SCFG)
    assert comp.loc["X", "mentions"] == 1
    assert comp.loc["X", "sentiment"] == pytest.approx(0.5)


def test_author_cap_limits_one_account(con):
    b = Builder(con)
    for i in range(20):
        b.add("SPAM", _day(i % 7), 1.0, author="bot")
    assert compute_components(con, WEEK, SCFG).loc["SPAM", "mentions"] == 5


def test_attention_is_clipped(con):
    b = Builder(con)
    for _ in range(200):
        b.add("VIRAL", _day(0), 0.5, engagement=10_000)
    assert compute_components(con, WEEK, SCFG).loc["VIRAL", "attention_z"] == 10


def test_empty_week(con):
    assert compute_components(con, WEEK, SCFG).empty


def test_zscore_handles_degenerate_input():
    assert zscore(pd.Series([1.0])).tolist() == [0.0]
    assert zscore(pd.Series([2.0, 2.0])).tolist() == [0.0, 0.0]
    assert zscore(pd.Series([1.0, np.nan, 3.0])).tolist() == [-1.0, 0.0, 1.0]


def test_eligibility_filters():
    comp = pd.DataFrame({
        "mentions": [10, 1, 10, 10], "n_authors": [5, 5, 1, 5],
        "sentiment": [0.1, 0.1, 0.1, 0.1],
    }, index=["OK", "FEW", "ONEAUTHOR", "NOTINUNIVERSE"])
    out = apply_eligibility(comp, {"OK", "FEW", "ONEAUTHOR"}, SCFG)
    assert list(out.index) == ["OK"]


def test_composites_are_mirror_images_on_sentiment():
    df = pd.DataFrame({"attention_z": [1.0, 2.0, 3.0], "sentiment": [0.5, -0.5, 0.0],
                       "momentum": [0.1, 0.2, 0.3], "breadth": [1.0, 1.0, 1.0]},
                      index=["A", "B", "C"])
    out = add_composites(df, SCFG["composite_weights"])
    za = zscore(df["attention_z"])
    zs = zscore(df["sentiment"])
    zm = zscore(df["momentum"])
    assert out["composite_bull"].tolist() == pytest.approx((0.35 * za + 0.35 * zs + 0.15 * zm).tolist())
    assert out["composite_bear"].tolist() == pytest.approx((0.35 * za - 0.35 * zs - 0.15 * zm).tolist())


# --- end-to-end rank ---------------------------------------------------------

def test_rank_produces_both_lists(con):
    save_universe(con, [
        {"ticker": t, "company_name": t, "exchange": "NASDAQ", "market_cap": 1e9,
         "avg_dollar_volume_30d": 1e7, "updated_at": datetime(2026, 10, 4)}
        for t in ("BULL", "BEAR", "QUIET", "COOL")
    ], min_coverage=0.5)
    b = Builder(con)
    for d in range(7, 37, 5):  # light baseline for everyone
        for t in ("BULL", "BEAR", "QUIET", "COOL"):
            b.add(t, _day(d), 0.0)
    for d in range(7):
        b.add("BULL", _day(d), 0.7)
        b.add("BEAR", _day(d), -0.7)
    for d in range(0, 7, 2):
        b.add("QUIET", _day(d), 0.3)
    for d in range(7, 37):  # COOL was busy before, silent now except a few posts
        b.add("COOL", _day(d), 0.5)
    for d in range(3):
        b.add("COOL", _day(d), 0.5)

    cfg = {"signals": SCFG, "extraction": {}, "scoring": {},
           "universe": {"exchanges": ["NASDAQ"], "min_market_cap": 1, "min_avg_dollar_volume": 1},
           "prices": {"context_days": 60}, "context": CONTEXT_CFG}
    bull, bear = rank(con, cfg, WEEK, scorer=None)

    assert bull["ticker"].tolist()[0] == "BULL"
    assert "BEAR" not in bull["ticker"].tolist()
    assert bear["ticker"].tolist() == ["BEAR"]
    assert "COOL" not in bull["ticker"].tolist()  # attention fell, so not "heating up"

    # Re-running the week replaces rows instead of duplicating them.
    rank(con, cfg, WEEK, scorer=None)
    n = con.execute("SELECT count(*) FROM weekly_signals WHERE week_start = ?",
                    [WEEK.week_start]).fetchone()[0]
    assert n == 4

"""News classification and its effect on aggregation."""

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import yaml

from signals.filters import apply_noise_filters, cap_per_author, drop_syndicated
from signals.news_quality import classify, is_about, name_keys

CASES = yaml.safe_load((Path(__file__).resolve().parent.parent / "fixtures" / "news_headlines.yaml")
                       .read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[c["title"][:50] for c in CASES])
def test_headline_kind(case):
    assert classify(case["title"], None, case.get("outlet")) == case["kind"]


def test_name_keys():
    assert name_keys("Costco Wholesale Corporation") == ["costco wholesale", "costco"]
    assert name_keys("Meta Platforms, Inc. Class A", ["Facebook"]) == ["facebook", "meta platforms", "meta"]
    assert name_keys("AT&T Inc.") == ["at&t"]
    assert name_keys(None, []) == []


def test_is_about():
    keys = name_keys("Costco Wholesale Corporation")
    assert is_about("COST", "Costco raises membership fee", None, keys)
    assert is_about("COST", "Retail roundup", "Shares of $COST and WMT moved", keys)
    assert not is_about("COST", "A long boat ride for groceries", "A trip to a big-box store", keys)


# --- aggregation effects ---------------------------------------------------------------

NOISE = {
    "news": {"sources": ["news"], "off_topic_weight": 0.3,
             "kind_weights": {"price_recap": 0.0, "promo": 0.3, "press_release": 0.5, "news": 1.0}},
}
CFG = {"source_weights": {"news": 1.0, "reddit": 1.0}, "noise": {**NOISE, "engagement_floor": {"news": 10}}}
KEYS = {"ACME": ["acme robotics", "acme"]}


def frame(rows):
    base = {"ticker": "ACME", "source": "news", "community": "Yahoo", "engagement": None,
            "created_at": datetime(2026, 10, 1), "content_hash": None, "block": 0, "author_id": "yahoo",
            "author_age_days": None, "author_karma": None, "body": None}
    return pd.DataFrame([{**base, "post_id": f"n{i}", **r} for i, r in enumerate(rows)])


def test_recaps_dropped_promos_and_off_topic_downweighted():
    df = frame([
        {"title": "Acme Robotics Wins Defense Contract"},
        {"title": "Why Acme Robotics Stock Jumped 15% Today"},
        {"title": "Is Acme Robotics Stock a Buy?"},
        {"title": "Big-box retail weekend roundup"},   # tagged ACME but never names it
    ])
    out = apply_noise_filters(df, CFG, KEYS).set_index("post_id")
    assert "n1" not in out.index                       # recap: gone, doesn't even count as a mention
    assert out.loc["n2", "news_kind"] == "promo"
    w = out["weight"]
    assert w["n2"] == pytest.approx(w["n0"] * 0.3)
    assert w["n3"] == pytest.approx(w["n0"] * 0.3)      # off-topic


def test_syndicated_headlines_collapse_to_earliest():
    df = frame([
        {"title": "Acme Robotics Wins Defense Contract", "created_at": datetime(2026, 10, 2)},
        {"title": "ACME ROBOTICS wins defense contract!", "created_at": datetime(2026, 10, 1)},
        {"title": "Acme Robotics Wins Defense Contract", "source": "reddit"},  # social: untouched
    ])
    assert set(drop_syndicated(df, ["news"])["post_id"]) == {"n1", "n2"}


def test_author_cap_skips_news_outlets():
    df = frame([{"title": f"Story {i}", "weight": 1.0} for i in range(12)])
    assert len(cap_per_author(df, 5, "block", sources=["reddit", "stocktwits"])) == 12
    assert len(cap_per_author(df, 5, "block")) == 5  # old behaviour when no source list is given


def test_failure_tolerance(tmp_path, monkeypatch):
    """A few failed tickers out of many is a normal day, not a partial run."""
    from jobs import collect_daily

    class Collector:
        source = "news"
        communities = [f"T{i}" for i in range(100)]
        failed_communities = ["T1", "T2"]
        failure_tolerance = 0.05
        def fetch(self, since):
            return []

    cfg = {"storage": {"db_path": str(tmp_path / "t.duckdb")}, "collect": {"lookback_hours": 36}}
    assert collect_daily.run(cfg, [Collector()], extract_tickers=False, collect_prices=False) == 0
    Collector.failed_communities = [f"T{i}" for i in range(10)]   # 10% > 5%
    assert collect_daily.run(cfg, [Collector()], extract_tickers=False, collect_prices=False) == 2

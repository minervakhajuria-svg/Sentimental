"""Noise and bot filtering."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from signals.filters import (add_post_weight, apply_noise_filters, author_weights, drop_near_duplicates,
                             drop_spam, near_duplicate_ids, simhash)

NOISE = {
    "engagement_floor": {"news": 10},
    "near_duplicates": {"max_hamming": 3, "min_words": 12},
    "spam_patterns": [r"join (my|our) (discord|telegram)", r"t\.me/", r"\bdm me\b"],
    "authors": {"reddit": {"min_age_days": 30, "new_account_weight": 0.25,
                           "min_karma": 100, "low_karma_weight": 0.5}},
    "bots": {"sources": ["reddit"], "max_posts_per_week": 3, "weight": 0.1},
}
CFG = {"source_weights": {"reddit": 1.0, "news": 1.0}, "noise": NOISE}
LONG = ("I have been looking at the quarterly numbers and the data center segment keeps "
        "growing faster than anyone expected which makes me think the guidance is conservative")


def frame(rows):
    base = {"ticker": "X", "source": "reddit", "community": "stocks", "engagement": 10,
            "created_at": datetime(2026, 10, 1), "content_hash": None, "block": 0,
            "author_id": None, "author_age_days": None, "author_karma": None,
            "title": "", "body": None}
    out = [{**base, "post_id": f"p{i}", **r} for i, r in enumerate(rows)]
    return pd.DataFrame(out)


def test_simhash_is_stable_and_close_for_small_edits():
    a, b = simhash(LONG), simhash(LONG + " honestly")
    assert a == simhash(LONG)
    assert bin(a ^ b).count("1") <= 6
    assert bin(a ^ simhash("completely different words about oil prices and shipping rates today")).count("1") > 10
    assert simhash("too short") is None


def test_near_duplicates_keep_earliest():
    df = frame([
        {"title": LONG, "created_at": datetime(2026, 10, 2)},
        {"title": LONG.replace("conservative", "conservative!!"), "created_at": datetime(2026, 10, 1)},
        {"title": "Totally unrelated thoughts on oil majors and refining margins this quarter going forward now"},
    ])
    assert near_duplicate_ids(df, 3, 12) == {"p0"}
    assert set(drop_near_duplicates(df, 3, 12)["post_id"]) == {"p1", "p2"}


def test_short_posts_are_not_fuzzy_matched():
    df = frame([{"title": "NVDA calls printing"}, {"title": "AMD calls printing"}])
    assert near_duplicate_ids(df, 3, 12) == set()


def test_spam_patterns():
    df = frame([{"title": "Join our Discord for picks"}, {"body": "free picks t.me/pumpgroup"},
                {"title": "DM me for the play"}, {"title": "Honest DD on margins"}])
    assert list(drop_spam(df, NOISE["spam_patterns"])["post_id"]) == ["p3"]


def test_author_weights_by_age_and_karma():
    df = frame([
        {"author_id": "a", "author_age_days": 5, "author_karma": 5000},     # new
        {"author_id": "b", "author_age_days": 400, "author_karma": 10},     # low karma
        {"author_id": "c", "author_age_days": 5, "author_karma": 10},       # both
        {"author_id": "d", "author_age_days": None, "author_karma": None},  # unknown: no penalty
        {"author_id": "e", "author_age_days": 5, "source": "news"},         # rules are per source
    ])
    assert author_weights(df, NOISE).tolist() == pytest.approx([0.25, 0.5, 0.125, 1.0, 1.0])


def test_hyperactive_authors_are_downweighted():
    rows = [{"author_id": "bot", "ticker": f"T{i}"} for i in range(5)]  # 5 posts > 3 per week
    rows += [{"author_id": "human"}]
    w = author_weights(frame(rows), NOISE)
    assert w.tolist() == pytest.approx([0.1] * 5 + [1.0])


def test_news_gets_engagement_floor():
    df = add_post_weight(frame([{"source": "news", "engagement": None},
                                {"source": "reddit", "engagement": 0}]), CFG)
    assert df["weight"].tolist() == pytest.approx([np.log1p(10), 0.0])


def test_zero_engagement_posts_still_count_but_zero_weight_authors_drop():
    cfg = {**CFG, "noise": {**NOISE, "authors": {"reddit": {
        "min_age_days": 30, "new_account_weight": 0.0, "min_karma": 0, "low_karma_weight": 1}}}}
    df = frame([{"engagement": 0, "author_id": "old", "author_age_days": 900},
                {"engagement": 50, "author_id": "new", "author_age_days": 2}])
    out = apply_noise_filters(df, cfg)
    assert list(out["post_id"]) == ["p0"]  # zero-engagement post kept, brand-new account dropped

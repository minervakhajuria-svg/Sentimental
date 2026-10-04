from datetime import datetime

import pytest

from collectors.base import Post
from jobs.score_posts import score_pending
from scoring.context import ContextBuilder, split_sentences
from scoring.finbert import Score, Scorer, probs_to_scores
from storage import db
from tickers.extractor import Match

# ProsusAI/finbert's real label order.
FINBERT_LABELS = {0: "positive", 1: "negative", 2: "neutral"}


class KeywordScorer(Scorer):
    """Deterministic stand-in for FinBERT."""

    name = "fake"

    def __init__(self):
        self.calls: list[list[str]] = []

    def score(self, texts):
        self.calls.append(list(texts))
        out = []
        for t in texts:
            t = t.lower()
            if "great" in t or "moon" in t:
                out.append(Score("pos", 0.8))
            elif "dead" in t or "crash" in t:
                out.append(Score("neg", -0.8))
            else:
                out.append(Score("neu", 0.0))
        return out


def test_probs_to_scores_uses_p_pos_minus_p_neg():
    [a, b, c] = probs_to_scores(
        [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.1, 0.7]], FINBERT_LABELS
    )
    assert a == Score("pos", 0.5)
    assert b == Score("neg", -0.5)
    assert c.label == "neu" and c.score == pytest.approx(0.1)


def test_probs_to_scores_reads_label_order_from_config():
    other_order = {0: "neutral", 1: "positive", 2: "negative"}
    [s] = probs_to_scores([[0.1, 0.9, 0.0]], other_order)
    assert s == Score("pos", 0.9)


def test_split_sentences():
    assert split_sentences("NVDA up. AMD down!\nTSLA flat?") == ["NVDA up.", "AMD down!", "TSLA flat?"]


def test_context_single_ticker_uses_whole_post():
    ctx = ContextBuilder([])
    assert ctx.build("Title", "Body text.", "NVDA", 1) == "Title\nBody text."


def test_context_multi_ticker_uses_mentioning_sentences():
    ctx = ContextBuilder([("Nvidia", "NVDA", False)])
    title, body = "Earnings week", "Nvidia crushed it. $AMD is dead. Market was flat."
    assert ctx.build(title, body, "NVDA", 2) == "Nvidia crushed it."
    assert ctx.build(title, body, "AMD", 2) == "$AMD is dead."


def test_context_falls_back_to_full_text():
    ctx = ContextBuilder([])
    assert ctx.build("No mention", "here", "NVDA", 2) == "No mention\nhere"


def test_context_is_truncated():
    ctx = ContextBuilder([], max_chars=10)
    assert len(ctx.build("x" * 50, None, "NVDA", 1)) == 10


def _post(pid, title, body=None):
    return Post(id=pid, source="reddit", community="stocks", author_id="a",
                author_age_days=None, author_karma=None,
                created_at=datetime(2026, 10, 1), collected_at=datetime(2026, 10, 1),
                title=title, body=body, url=None, engagement=5, content_hash=pid)


SCFG = {"max_chars": 2000, "commit_every": 2}


def test_score_pending_scores_per_ticker_and_caches(tmp_path):
    con = db.connect(tmp_path / "s.duckdb")
    db.upsert_posts(con, [_post("p1", "Earnings", "NVDA to the moon. AMD is dead."),
                          _post("p2", "TSLA great quarter")])
    db.replace_post_tickers(con, {
        "p1": [Match("NVDA", "bare", 0.5), Match("AMD", "bare", 0.5)],
        "p2": [Match("TSLA", "bare", 0.5)],
    })
    scorer = KeywordScorer()
    assert score_pending(con, scorer, SCFG) == 3
    scores = dict(((p, t), s) for p, t, s in con.execute(
        "SELECT post_id, ticker, score FROM post_scores").fetchall())
    assert scores == {("p1", "NVDA"): 0.8, ("p1", "AMD"): -0.8, ("p2", "TSLA"): 0.8}

    # Second run scores nothing: post_scores is the cache.
    assert score_pending(con, scorer, SCFG) == 0
    con.close()


def test_scores_are_never_overwritten(tmp_path):
    con = db.connect(tmp_path / "s.duckdb")
    db.upsert_posts(con, [_post("p1", "TSLA great")])
    db.replace_post_tickers(con, {"p1": [Match("TSLA", "bare", 0.5)]})
    con.execute("INSERT INTO post_scores VALUES ('p1', 'TSLA', 'fake', 'neg', -1.0, now())")
    assert score_pending(con, KeywordScorer(), SCFG) == 0
    assert con.execute("SELECT score FROM post_scores").fetchone()[0] == -1.0
    con.close()


def test_identical_texts_are_scored_once(tmp_path):
    con = db.connect(tmp_path / "s.duckdb")
    db.upsert_posts(con, [_post("p1", "TSLA great"), _post("p2", "TSLA great")])
    db.replace_post_tickers(con, {"p1": [Match("TSLA", "bare", 0.5)],
                                  "p2": [Match("TSLA", "bare", 0.5)]})
    scorer = KeywordScorer()
    score_pending(con, scorer, SCFG)
    assert scorer.calls == [["TSLA great"]]
    con.close()


def test_scorer_agreement_with_source_tags(tmp_path):
    from analysis.scorer_check import agreement
    con = db.connect(tmp_path / "a.duckdb")
    rows = [  # (post, tag, finbert)
        ("a", 1.0, 0.7), ("b", -1.0, -0.4), ("c", 1.0, 0.02), ("d", -1.0, 0.6),
    ]
    for pid, tag, pred in rows:
        con.execute("INSERT INTO post_scores VALUES (?, 'X', 'stocktwits_tag', 'pos', ?, now())", [pid, tag])
        con.execute("INSERT INTO post_scores VALUES (?, 'X', 'finbert', 'neu', ?, now())", [pid, pred])
    r = agreement(con, "finbert", "stocktwits_tag")
    assert (r["pairs"], r["agree"], r["neutral"], r["opposite"]) == (4, 2, 1, 1)
    assert r["accuracy"] == 0.5 and r["accuracy_when_decisive"] == pytest.approx(2 / 3)
    con.close()

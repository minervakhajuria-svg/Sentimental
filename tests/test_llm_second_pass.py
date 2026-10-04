"""Claude second pass, against fake clients (no API calls)."""

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from collectors.base import Post
from scoring.finbert import Score
from scoring.llm_second_pass import ClaudeScorer, parse_reply, second_pass, select_pairs
from signals.aggregate import load_mentions
from storage import db
from tickers.extractor import Match

CFG = {"ambiguity_threshold": 0.3, "top_engagement_per_ticker_week": 1, "lookback_days": 14,
       "max_items_per_run": 100}
NOW = datetime(2026, 10, 3, 18)


def reply(label, score, stop="end_turn"):
    return SimpleNamespace(stop_reason=stop,
                           content=[SimpleNamespace(type="text", text=json.dumps({"label": label, "score": score}))])


def judge(text: str):
    """Pretend model: reads options slang the way the prompt asks."""
    t = text.lower()
    if "puts" in t:
        return reply("bearish", -0.8)
    if "calls" in t:
        return reply("bullish", 0.7)
    if "refuse" in t:
        return reply("neutral", 0, stop="refusal")
    return reply("neutral", 0.0)


class FakeMessages:
    def __init__(self):
        self.calls = []
        self.batches = FakeBatches()

    def create(self, **params):
        self.calls.append(params)
        return judge(params["messages"][0]["content"])


class FakeBatches:
    def __init__(self):
        self.requests, self.polls = [], 0

    def create(self, requests):
        self.requests = requests
        return SimpleNamespace(id="batch_1", processing_status="in_progress")

    def retrieve(self, batch_id):
        self.polls += 1
        return SimpleNamespace(id=batch_id, processing_status="ended" if self.polls >= 2 else "in_progress")

    def results(self, batch_id):
        out = []
        for r in reversed(self.requests):  # results arrive in any order
            if "error" in r["params"]["messages"][0]["content"]:
                out.append(SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(type="errored")))
            else:
                msg = judge(r["params"]["messages"][0]["content"])
                out.append(SimpleNamespace(custom_id=r["custom_id"],
                                           result=SimpleNamespace(type="succeeded", message=msg)))
        return out


def client():
    return SimpleNamespace(messages=FakeMessages())


def test_parse_reply():
    assert parse_reply('{"label": "bullish", "score": 0.6}') == Score("pos", 0.6)
    assert parse_reply('{"label": "bearish", "score": -3}') == Score("neg", -1.0)  # clamped
    assert parse_reply("not json") is None
    assert parse_reply('{"label": "maybe", "score": 0}') is None


def test_sync_scoring_and_request_shape():
    c = client()
    s = ClaudeScorer(c, "claude-haiku-4-5", use_batch=False)
    out = s.score_items([("TSLA puts are free money", "TSLA"), ("loaded calls", "NVDA"),
                         ("please refuse", "X")])
    assert out == [Score("neg", -0.8), Score("pos", 0.7), None]  # refusal -> retried later
    p = c.messages.calls[0]
    assert p["model"] == "claude-haiku-4-5" and p["max_tokens"] == 256
    assert p["output_config"]["format"]["type"] == "json_schema"
    assert p["messages"][0]["content"].startswith("Ticker: TSLA")


def test_batch_scoring_maps_results_by_custom_id():
    c = client()
    s = ClaudeScorer(c, "claude-haiku-4-5", use_batch=True, sleep=lambda x: None)
    out = s.score_items([("calls", "A"), ("this errors", "B"), ("puts", "C")])
    assert out == [Score("pos", 0.7), None, Score("neg", -0.8)]
    assert [r["custom_id"] for r in c.messages.batches.requests] == ["i0", "i1", "i2"]


def test_batch_timeout_fails_loudly():
    c = client()
    c.messages.batches.retrieve = lambda bid: SimpleNamespace(id=bid, processing_status="in_progress")
    s = ClaudeScorer(c, "m", use_batch=True, max_wait_minutes=0, sleep=lambda x: None)
    with pytest.raises(TimeoutError):
        s.score_items([("calls", "A")])


# --- selection and storage --------------------------------------------------------

@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "l.duckdb")
    rows = [  # (id, title, engagement, finbert score)
        ("a", "TSLA puts are free money", 5, 0.02),    # ambiguous -> selected
        ("b", "TSLA is a great company, strong quarter", 3, 0.9),  # confident, low engagement -> skipped
        ("c", "TSLA calls printing, huge week", 500, 0.8),  # confident but top engagement -> selected
    ]
    for pid, title, eng, base in rows:
        db.upsert_posts(c, [Post(id=pid, source="reddit", community="wsb", author_id=pid, author_age_days=None,
                                 author_karma=None, created_at=datetime(2026, 10, 2), collected_at=NOW,
                                 title=title, body=None, url=None, engagement=eng, content_hash=pid)])
        db.replace_post_tickers(c, {pid: [Match("TSLA", "cashtag", 0.95)]})
        c.execute("INSERT INTO post_scores VALUES (?, 'TSLA', 'finbert', 'neu', ?, now())", [pid, base])
    yield c
    c.close()


def test_select_pairs(con):
    picked = {r[0] for r in select_pairs(con, CFG, "finbert", "claude", now=NOW)}
    assert picked == {"a", "c"}
    assert select_pairs(con, {**CFG, "max_items_per_run": 1}, "finbert", "claude", now=NOW)[0][0] == "c"


def test_second_pass_stores_separately_and_takes_priority(con):
    s = ClaudeScorer(client(), "m", use_batch=False)
    assert second_pass(con, s, CFG) == 2
    # FinBERT rows untouched.
    assert con.execute("SELECT count(*) FROM post_scores WHERE model = 'finbert'").fetchone()[0] == 3
    # Already-scored pairs aren't sent again.
    assert select_pairs(con, CFG, "finbert", "claude", now=NOW) == []
    # Aggregation prefers Claude where present, FinBERT otherwise.
    df = load_mentions(con, datetime(2026, 9, 1), NOW, ["claude", "finbert"], 0.5).set_index("post_id")
    assert df.loc["a", "score"] == -0.8   # sarcasm fixed by pass 2
    assert df.loc["b", "score"] == 0.9    # FinBERT fallback
    assert load_mentions(con, datetime(2026, 9, 1), NOW, "finbert", 0.5).set_index("post_id").loc["a", "score"] == 0.02

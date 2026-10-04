"""News and StockTwits collectors, the HTTP helper, and source-supplied tags."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from collectors.http import HttpError, get_json
from collectors.news import FINNHUB_NEWS_URL, FinnhubNewsCollector
from collectors.stocktwits import StockTwitsCollector
from storage import db
from tickers.extractor import Match

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
NEWS = json.loads((FIXTURES / "finnhub_company_news.json").read_text(encoding="utf-8"))
STREAM = json.loads((FIXTURES / "stocktwits_stream.json").read_text(encoding="utf-8"))
NOW = datetime(2026, 10, 3, 18)
SINCE = datetime(2026, 10, 2, 6)


# --- HTTP helper -----------------------------------------------------------

def test_get_json_retries_rate_limits_then_succeeds():
    calls, sleeps = [], []

    def opener(url, headers, timeout):
        calls.append(url)
        if len(calls) < 3:
            raise HttpError(429, url)
        return {"ok": True}

    out = get_json("https://x/api", params={"a": 1}, headers={"K": "secret"},
                   backoff_seconds=1, sleep=sleeps.append, opener=opener)
    assert out == {"ok": True}
    assert sleeps == [1, 2]
    assert calls[0] == "https://x/api?a=1"


def test_get_json_does_not_retry_auth_errors():
    def opener(url, headers, timeout):
        raise HttpError(401, url)
    with pytest.raises(HttpError):
        get_json("https://x/api", sleep=lambda s: pytest.fail("should not retry"), opener=opener)


def test_http_error_message_has_no_query_string():
    assert "token" not in str(HttpError(401, "https://x/api"))


# --- Finnhub news ------------------------------------------------------------

def fake_finnhub(calls):
    def get(url, params=None, headers=None, sleep=None):
        assert url == FINNHUB_NEWS_URL
        assert headers == {"X-Finnhub-Token": "KEY"}  # key in a header, never the URL
        calls.append(params)
        if params["symbol"] == "FAIL":
            raise HttpError(500, url)
        return NEWS.get(params["symbol"], [])
    return get


def test_news_collector_normalises_and_merges():
    calls = []
    c = FinnhubNewsCollector("KEY", lambda: ["NVDA", "AMD"], get=fake_finnhub(calls),
                             sleep=lambda s: None, now=lambda: NOW)
    posts = {p.id: p for p in c.fetch(SINCE)}
    assert set(posts) == {"news:finnhub:9001", "news:finnhub:9002"}  # old, empty, malformed skipped
    p = posts["news:finnhub:9001"]
    assert p.source == "news" and p.community == "Newswire"
    assert p.title.startswith("Chipmaker") and p.body == "The company raised full-year guidance."
    assert p.engagement is None
    assert p.source_tickers == ("NVDA",)
    # Same article returned for both tickers: one post, both tags.
    assert posts["news:finnhub:9002"].source_tickers == ("AMD", "NVDA")
    assert posts["news:finnhub:9002"].body is None
    assert calls[0] == {"symbol": "NVDA", "from": "2026-10-02", "to": "2026-10-03"}


def test_news_collector_skips_failing_ticker():
    c = FinnhubNewsCollector("KEY", lambda: ["FAIL", "NVDA"], get=fake_finnhub([]),
                             sleep=lambda s: None, now=lambda: NOW)
    posts = list(c.fetch(SINCE))
    assert posts and c.failed_communities == ["FAIL"]


def test_news_from_config_requires_key(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FINNHUB_API_KEY"):
        FinnhubNewsCollector.from_config({}, lambda: [])


# --- StockTwits ----------------------------------------------------------------

def fake_stocktwits(url, params=None, headers=None, sleep=None):
    assert headers == {"Authorization": "OAuth TOKEN"}
    return STREAM["TSLA_page2"] if params and params.get("max") else STREAM["TSLA"]


def test_stocktwits_collector():
    c = StockTwitsCollector("TOKEN", lambda: ["TSLA"], get=fake_stocktwits,
                            sleep=lambda s: None, now=lambda: NOW)
    posts = {p.id: p for p in c.fetch(SINCE)}
    assert set(posts) == {"stocktwits:7003", "stocktwits:7002", "stocktwits:7001"}  # page 2 is too old
    bull, bear, plain = posts["stocktwits:7003"], posts["stocktwits:7002"], posts["stocktwits:7001"]
    assert bull.source_sentiment == 1 and bear.source_sentiment == -1 and plain.source_sentiment is None
    assert bull.engagement == 15 and bear.engagement == 0
    assert bear.source_tickers == ("RIVN", "TSLA")
    assert bear.author_age_days == 3 and bear.author_karma == 1
    assert bull.author_id != "trader_one"
    assert bull.url == "https://stocktwits.com/message/7003"


def test_stocktwits_from_config_requires_token(monkeypatch):
    monkeypatch.delenv("STOCKTWITS_ACCESS_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="STOCKTWITS_ACCESS_TOKEN"):
        StockTwitsCollector.from_config({}, lambda: [])


# --- source tags in storage ---------------------------------------------------

@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "s.duckdb")
    yield c
    c.close()


def _stocktwits_posts():
    c = StockTwitsCollector("TOKEN", lambda: ["TSLA"], get=fake_stocktwits,
                            sleep=lambda s: None, now=lambda: NOW)
    return list(c.fetch(SINCE))


def test_source_tags_are_stored_and_filtered_by_universe(con):
    posts = _stocktwits_posts()
    db.upsert_posts(con, posts)
    tickers, tags = db.add_source_tags(con, posts, 0.9, universe={"TSLA"})
    assert tickers == 3 and tags == 2  # RIVN not in universe; one message untagged
    rows = con.execute("SELECT post_id, ticker, match_type FROM post_tickers ORDER BY 1").fetchall()
    assert all(t == "TSLA" and m == "source" for _, t, m in rows)
    scores = dict(con.execute("SELECT post_id, score FROM post_scores WHERE model = 'stocktwits_tag'").fetchall())
    assert scores == {"stocktwits:7003": 1.0, "stocktwits:7002": -1.0}


def test_extractor_rebuild_keeps_source_tags(con):
    posts = _stocktwits_posts()
    db.upsert_posts(con, posts)
    db.add_source_tags(con, posts, 0.9, universe={"TSLA"})
    # A rebuild writes extractor matches: the source row for TSLA must survive and win.
    db.replace_post_tickers(con, {"stocktwits:7003": [Match("TSLA", "cashtag", 0.95), Match("AMD", "bare", 0.5)]})
    rows = dict(con.execute("SELECT ticker, match_type FROM post_tickers WHERE post_id = 'stocktwits:7003'").fetchall())
    assert rows == {"TSLA": "source", "AMD": "bare"}


def test_source_tagged_posts_still_get_extracted(con):
    posts = _stocktwits_posts()
    db.upsert_posts(con, posts)
    db.add_source_tags(con, posts, 0.9, universe={"TSLA"})
    pending = {r[0] for r in db.posts_without_tickers(con)}
    assert pending == {p.id for p in posts}

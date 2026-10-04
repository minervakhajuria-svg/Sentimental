from datetime import datetime

import pytest

from collectors.base import Post
from jobs.extract_tickers import extract
from storage import db
from tickers.universe import save_universe

CFG_UNIVERSE = {"exchanges": ["NASDAQ"], "min_market_cap": 1, "min_avg_dollar_volume": 1}
CFG_EXTRACTION = {
    "confidence": {"cashtag": 0.95, "alias": 0.75, "bare": 0.5},
    "bare_min_length": 2,
    "shouting_min_words": 3,
    "finance_context_words": ["shares"],
}


def post(pid, title, body=None):
    return Post(id=f"reddit:{pid}", source="reddit", community="stocks", author_id="a",
                author_age_days=None, author_karma=None,
                created_at=datetime(2026, 10, 3), collected_at=datetime(2026, 10, 3),
                title=title, body=body, url=None, engagement=1, content_hash=pid)


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.duckdb")
    save_universe(c, [
        {"ticker": t, "company_name": t, "exchange": "NASDAQ", "market_cap": 1e9,
         "avg_dollar_volume_30d": 1e7, "updated_at": datetime(2026, 10, 4)}
        for t in ("NVDA", "AMD", "TSLA")
    ], min_coverage=0.5)
    yield c
    c.close()


def cfg():
    return {"universe": CFG_UNIVERSE, "extraction": CFG_EXTRACTION}


def tags(con):
    return con.execute(
        "SELECT post_id, ticker, match_type FROM post_tickers ORDER BY post_id, ticker"
    ).fetchall()


def test_tags_pending_posts_and_is_idempotent(con):
    db.upsert_posts(con, [post("1", "$NVDA and Tesla"), post("2", "nothing here")])
    assert extract(con, cfg()) == (2, 2)
    assert tags(con) == [("reddit:1", "NVDA", "cashtag"), ("reddit:1", "TSLA", "alias")]
    # Second run: post 1 is done; post 2 (no match) is re-checked but writes nothing.
    assert extract(con, cfg()) == (1, 0)
    assert len(tags(con)) == 2


def test_rebuild_replaces_old_matches(con):
    db.upsert_posts(con, [post("1", "$NVDA")])
    extract(con, cfg())
    con.execute("INSERT INTO post_tickers VALUES ('reddit:1', 'AMD', 'bare', 0.5)")  # stale
    extract(con, cfg(), rebuild=True)
    assert tags(con) == [("reddit:1", "NVDA", "cashtag")]


def test_empty_universe_fails_loudly(tmp_path):
    c = db.connect(tmp_path / "empty.duckdb")
    db.upsert_posts(c, [post("1", "$NVDA")])
    with pytest.raises(RuntimeError, match="universe is empty"):
        extract(c, cfg())
    c.close()

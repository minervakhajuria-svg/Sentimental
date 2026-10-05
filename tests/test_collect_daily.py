"""End-to-end daily job against fixtures: two runs must store posts once."""

from datetime import datetime
from types import SimpleNamespace

import duckdb
import prawcore
import pytest

from collectors.reddit import RedditCollector
from conftest import FakeReddit
from jobs import collect_daily


@pytest.fixture
def cfg(tmp_path):
    return {
        "storage": {"db_path": str(tmp_path / "sentiment.duckdb")},
        "collect": {"lookback_hours": 36},
    }


@pytest.fixture(autouse=True)
def freeze_job_clock(monkeypatch, reddit_fixture):
    # The job computes `since` from the wall clock; pin it to the fixture's day.
    now = datetime.fromisoformat(reddit_fixture["collected_at"])
    monkeypatch.setattr(collect_daily, "utc_now", lambda: now)


def collector(reddit_fixture, fixed_now, errors=None):
    return RedditCollector(
        FakeReddit(reddit_fixture, errors),
        communities=["wallstreetbets", "stocks"],
        sleep=lambda s: None,
        now=fixed_now,
    )


def count_posts(cfg):
    with duckdb.connect(cfg["storage"]["db_path"]) as con:
        return con.execute("SELECT count(*) FROM posts").fetchone()[0]


def test_daily_run_is_idempotent(cfg, reddit_fixture, fixed_now):
    assert collect_daily.run(cfg, [collector(reddit_fixture, fixed_now)], extract_tickers=False, collect_prices=False, collect_filings=False) == 0
    assert count_posts(cfg) == 5
    assert collect_daily.run(cfg, [collector(reddit_fixture, fixed_now)], extract_tickers=False, collect_prices=False, collect_filings=False) == 0
    assert count_posts(cfg) == 5


def test_partial_failure_saves_what_it_got(cfg, reddit_fixture, fixed_now):
    err = prawcore.exceptions.Forbidden(SimpleNamespace(status_code=403))
    code = collect_daily.run(cfg, [collector(reddit_fixture, fixed_now, {"stocks": [err]})], extract_tickers=False, collect_prices=False, collect_filings=False)
    assert code == collect_daily.EXIT_PARTIAL
    assert count_posts(cfg) == 3


def test_total_failure_exits_nonzero(cfg, reddit_fixture, fixed_now):
    def forbidden():
        return prawcore.exceptions.Forbidden(SimpleNamespace(status_code=403))
    errors = {"wallstreetbets": [forbidden()], "stocks": [forbidden()]}
    code = collect_daily.run(cfg, [collector(reddit_fixture, fixed_now, errors)], extract_tickers=False, collect_prices=False, collect_filings=False)
    assert code == collect_daily.EXIT_FAILED
    assert count_posts(cfg) == 0


def test_crashing_collector_does_not_touch_existing_data(cfg, reddit_fixture, fixed_now):
    collect_daily.run(cfg, [collector(reddit_fixture, fixed_now)], extract_tickers=False, collect_prices=False, collect_filings=False)

    class Boom:
        source = "boom"
        def fetch(self, since):
            yield from []
            raise RuntimeError("network down")

    assert collect_daily.run(cfg, [Boom()], extract_tickers=False, collect_prices=False, collect_filings=False) == collect_daily.EXIT_FAILED
    assert count_posts(cfg) == 5


def test_daily_run_tags_tickers(cfg, reddit_fixture, fixed_now):
    from storage import db
    from tickers.universe import save_universe

    cfg["universe"] = {"exchanges": ["NASDAQ"], "min_market_cap": 1, "min_avg_dollar_volume": 1}
    cfg["extraction"] = {
        "confidence": {"cashtag": 0.95, "alias": 0.75, "bare": 0.5},
        "bare_min_length": 2, "shouting_min_words": 3, "finance_context_words": ["shares"],
    }
    con = db.connect(cfg["storage"]["db_path"])
    save_universe(con, [
        {"ticker": t, "company_name": t, "exchange": "NASDAQ", "market_cap": 1e9,
         "avg_dollar_volume_30d": 1e7, "updated_at": datetime(2026, 10, 4)}
        for t in ("NVDA", "MSFT", "TSLA", "AMD")
    ], min_coverage=0.5)
    con.close()

    assert collect_daily.run(cfg, [collector(reddit_fixture, fixed_now)],
                             collect_prices=False, collect_filings=False) == 0
    with duckdb.connect(cfg["storage"]["db_path"]) as con:
        tagged = set(con.execute("SELECT post_id, ticker FROM post_tickers").fetchall())
    assert ("reddit:1aaa01", "NVDA") in tagged
    assert ("reddit:1bbb01", "MSFT") in tagged


def test_extraction_failure_is_partial_but_keeps_posts(cfg, reddit_fixture, fixed_now):
    cfg["universe"] = {"exchanges": ["NASDAQ"], "min_market_cap": 1, "min_avg_dollar_volume": 1}
    cfg["extraction"] = {}
    # No universe saved -> extraction raises; collection must still be kept.
    assert collect_daily.run(cfg, [collector(reddit_fixture, fixed_now)],
                             collect_prices=False, collect_filings=False) == collect_daily.EXIT_PARTIAL
    assert count_posts(cfg) == 5


def test_one_source_missing_credentials_does_not_stop_others(cfg, monkeypatch):
    """Reddit still awaiting approval: news keeps collecting, exit is partial."""
    import json
    from pathlib import Path
    from collectors import news as news_mod

    for k in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FINNHUB_API_KEY", "KEY")
    fixture = json.loads((Path(__file__).resolve().parent.parent / "fixtures" /
                          "finnhub_company_news.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(news_mod, "get_json",
                        lambda url, params=None, headers=None, sleep=None: fixture.get(params["symbol"], []))
    monkeypatch.setattr(news_mod.time, "sleep", lambda s: None)

    from storage import db
    from tickers.universe import save_universe
    cfg.update({
        "reddit": {"communities": ["stocks"]},
        "news": {"enabled": True, "max_tickers": 5, "mention_lookback_days": 7, "requests_per_minute": 6000},
        "stocktwits": {"enabled": False},
        "universe": {"exchanges": ["NASDAQ"], "min_market_cap": 1, "min_avg_dollar_volume": 1},
        "extraction": {"confidence": {"cashtag": 0.95, "alias": 0.75, "bare": 0.5, "source": 0.9},
                       "bare_min_length": 2, "shouting_min_words": 3, "finance_context_words": []},
    })
    con = db.connect(cfg["storage"]["db_path"])
    save_universe(con, [
        {"ticker": t, "company_name": t, "exchange": "NASDAQ", "market_cap": 1e9,
         "avg_dollar_volume_30d": v, "updated_at": datetime(2026, 10, 4)}
        for t, v in (("NVDA", 2e9), ("AMD", 1e9))
    ], min_coverage=0.5)
    con.close()

    code = collect_daily.run(cfg, collect_prices=False, collect_filings=False)
    assert code == collect_daily.EXIT_PARTIAL  # reddit couldn't start
    with duckdb.connect(cfg["storage"]["db_path"]) as con:
        rows = con.execute("SELECT post_id, ticker, match_type FROM post_tickers "
                           "WHERE match_type = 'source' ORDER BY 1, 2").fetchall()
    assert ("news:finnhub:9002", "AMD", "source") in rows
    assert ("news:finnhub:9002", "NVDA", "source") in rows

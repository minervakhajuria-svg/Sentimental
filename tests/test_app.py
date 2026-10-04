"""App queries against the synthetic demo DB, plus a headless run of the Streamlit script."""

from datetime import timedelta
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import demo_data
from app import queries

APP = Path(__file__).resolve().parent.parent / "src" / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    path = tmp_path_factory.mktemp("demo") / "demo.duckdb"
    weeks = demo_data.build(path)
    return path, weeks


@pytest.fixture
def con(demo):
    c = queries.connect(str(demo[0]))
    yield c
    c.close()


def test_weeks_newest_first(con, demo):
    weeks = queries.available_weeks(con)
    assert weeks[0] == demo[1][-1].week_start
    assert weeks == sorted(weeks, reverse=True)


def test_ranked_tables_match_the_storylines(con, demo):
    bull, bear = queries.ranked_tables(con, demo[1][-1].week_start, top_n=20)
    assert bull.iloc[0]["ticker"] == "NVDA"
    assert bull.iloc[0]["company_name"] == "NVIDIA Corporation"
    assert bear.iloc[0]["ticker"] == "INTC"
    assert list(bull["rank"]) == list(range(1, len(bull) + 1))
    assert list(bull.columns) == ["rank"] + queries.DISPLAY_COLUMNS
    assert (bull["sentiment"] > 0).all() and (bear["sentiment"] < 0).all()


def test_daily_trend_has_every_day(con, demo):
    week = demo[1][-1]
    trend = queries.daily_trend(con, "NVDA", week.end, 30, "finbert")
    assert len(trend) == 30
    assert trend["day"].iloc[-1].date() == (week.end - timedelta(days=1)).date()
    assert trend["mentions"].min() >= 0
    assert trend["close"].isna().all()  # no price data until phase 5


def test_top_posts_are_from_the_week_and_sorted(con, demo):
    week = demo[1][-1]
    posts = queries.top_posts(con, "NVDA", week, "finbert", limit=10)
    assert len(posts) == 10
    assert posts["created_at"].between(week.start, week.end).all()
    assert posts["engagement"].is_monotonic_decreasing
    assert posts["url"].str.startswith("https://www.reddit.com/").all()


def test_tracked_tickers_and_history(con):
    assert set(queries.tracked_tickers(con)) == set(demo_data.COMPANIES)
    assert len(queries.weekly_history(con, "NVDA")) >= 1


def _drilldown_script():
    from app import views
    views.drilldown()


def test_rankings_page_renders(demo, monkeypatch):
    monkeypatch.setenv("SENTIMENT_DB_PATH", str(demo[0]))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert [s.value for s in at.subheader] == ["Heating up, bullish", "Heating up, bearish"]
    assert at.dataframe[0].value.iloc[0]["ticker"] == "NVDA"
    assert any("not investment advice" in c.value for c in at.caption)


def test_drilldown_page_renders(demo, monkeypatch):
    monkeypatch.setenv("SENTIMENT_DB_PATH", str(demo[0]))
    at = AppTest.from_function(_drilldown_script, default_timeout=60)
    at.session_state["ticker"] = "INTC"
    at.run()
    assert not at.exception
    assert at.header[0].value.startswith("INTC")
    titles = [s.value for s in at.subheader]
    assert "Mentions and sentiment" in titles
    assert any(t.startswith("Top posts") for t in titles)


def test_missing_database_shows_message(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTIMENT_DB_PATH", str(tmp_path / "nope.duckdb"))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert "No database yet" in at.info[0].value

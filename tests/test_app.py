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
    weekday = trend["day"].dt.weekday < 5
    assert trend.loc[weekday, "close"].notna().all()   # sessions have a close
    assert trend.loc[~weekday, "close"].isna().all()   # weekends don't


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


def _markdown(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_rankings_page_renders(demo, monkeypatch):
    monkeypatch.setenv("SENTIMENT_DB_PATH", str(demo[0]))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    md = _markdown(at)
    assert "Sentimental<span>.</span>" in md
    assert "Heating up · bullish" in md and "Heating up · bearish" in md
    assert at.dataframe[0].value.iloc[0]["ticker"] == "NVDA"
    assert "not investment advice" in md


def test_drilldown_page_renders(demo, monkeypatch):
    monkeypatch.setenv("SENTIMENT_DB_PATH", str(demo[0]))
    at = AppTest.from_function(_drilldown_script, default_timeout=60)
    at.session_state["ticker"] = "INTC"
    at.run()
    assert not at.exception
    md = _markdown(at)
    assert "Intel Corporation" in md and ">INTC<" in md
    for section in ("Weekly score", "Mention mix", "By source", "Top posts"):
        assert section in md
    assert "Bearish" in md  # INTC's storyline


def test_missing_database_shows_message(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTIMENT_DB_PATH", str(tmp_path / "nope.duckdb"))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert "No database yet" in at.info[0].value


def test_week_summary_and_breakdown(con, demo):
    week = demo[1][-1]
    summary = queries.week_summary(con, "INTC", week, "finbert")
    assert summary["posts"] == summary["pos"] + summary["neu"] + summary["neg"]
    assert summary["ranked_sentiment"] < 0
    breakdown = queries.community_breakdown(con, "INTC", week, "finbert")
    assert breakdown["posts"].sum() == summary["posts"]
    tape = queries.week_tape(con, week.week_start)
    assert tape["mentions"].is_monotonic_decreasing


def test_post_text_is_escaped():
    import pandas as pd
    from app import theme
    df = pd.DataFrame([{"created_at": pd.Timestamp("2026-10-01"), "source": "news", "community": "<b>x</b>",
                        "title": "<script>alert(1)</script>", "sentiment": 0.5, "label": "pos",
                        "engagement": 3, "match_type": "cashtag",
                        "url": 'https://example.com/"onmouseover="x'}])
    out = theme.post_list(df)
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert '"onmouseover' not in out
    assert "<b>x</b>" not in out


def test_gauge_maps_sentiment_to_0_100():
    from app import theme
    assert ">75<" in theme.gauge(0.5, None, True)
    assert ">0<" in theme.gauge(-1.0, None, True)
    assert "Bearish" in theme.gauge(-0.5, -0.2, True)
    assert "–" in theme.gauge(None, None, False)


def test_demo_storylines_set_context_flags(con, demo):
    week = demo[1][-1]
    rows = con.execute("SELECT ticker, ret_5d, rel_volume, early_chatter_flag FROM weekly_signals "
                       "WHERE week_start = ?", [week.week_start]).df().set_index("ticker")
    assert rows.loc["NVDA", "early_chatter_flag"]  # flat price, rising volume, top composite
    assert rows.loc["INTC", "ret_5d"] < 0
    assert rows["rel_volume"].notna().all()


def test_source_labels():
    from app import theme
    assert theme.source_label("reddit", "stocks") == "r/stocks"
    assert theme.source_label("news", "Newswire") == "Newswire"
    assert theme.source_label("stocktwits", "stocktwits") == "StockTwits"

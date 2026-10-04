"""Page functions for the Streamlit app (navigation lives in streamlit_app.py).

The app only reads the database. It opens a short-lived read-only connection
per page render rather than holding one open, so the daily and weekly jobs
can still write to the file while the app is running.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import altair as alt
import duckdb
import streamlit as st

from app import queries
from settings import load_config

DISCLAIMER = "Screening aid only, not investment advice."

cfg = load_config(os.environ.get("SENTIMENT_CONFIG"))


def db_path() -> str:
    # Read per render (not at import) so the env override always applies.
    return os.environ.get("SENTIMENT_DB_PATH") or cfg["storage"]["db_path"]

MODEL = cfg["signals"]["sentiment_model"]
TOP_N = cfg["signals"]["top_n"]

# Set by streamlit_app.py once the page objects exist, so a click in the
# rankings table can jump to the drill-down.
drilldown_page = None


@contextmanager
def db():
    path = db_path()
    if not Path(path).exists():
        st.info(f"No database yet at `{path}`. Run the daily collection job first.")
        st.stop()
    try:
        con = queries.connect(path)
    except duckdb.IOException:
        st.warning("The database is busy (a collection or ranking job is writing). "
                   "Try again in a minute.")
        st.stop()
    try:
        yield con
    finally:
        con.close()


def week_label(d) -> str:
    return f"Week of {d:%d %b %Y}"


def pick_week(weeks):
    """Week selector shared by both pages (kept in session state)."""
    if st.session_state.get("week") not in weeks:
        st.session_state["week"] = weeks[0]
    return st.sidebar.selectbox("Week", weeks, format_func=week_label, key="week")


RANK_COLUMNS = {
    "rank": st.column_config.NumberColumn("#", width="small"),
    "ticker": st.column_config.TextColumn("Ticker"),
    "company_name": st.column_config.TextColumn("Company"),
    "composite": st.column_config.NumberColumn("Composite", format="%.2f",
        help="Weighted cross-sectional z-score of attention, sentiment, momentum and breadth"),
    "mentions": st.column_config.NumberColumn("Mentions"),
    "attention_z": st.column_config.NumberColumn("Attention z", format="%.2f",
        help="This week's weighted mentions vs the ticker's own 30-day baseline"),
    "sentiment": st.column_config.NumberColumn("Sentiment", format="%+.2f", help="-1 to +1"),
    "momentum": st.column_config.NumberColumn("Momentum", format="%+.2f",
        help="Sentiment change vs last week"),
    "breadth": st.column_config.NumberColumn("Breadth", format="%.2f",
        help="Distinct authors, communities and sources (log-scaled)"),
    "ret_5d": st.column_config.NumberColumn("5d return", format="percent"),
    "ret_30d": st.column_config.NumberColumn("30d return", format="percent"),
    "rel_volume": st.column_config.NumberColumn("Rel. volume", format="%.2f"),
    "early_chatter_flag": st.column_config.CheckboxColumn("Early chatter",
        help="Top-decile composite while price hasn't moved yet and volume is rising"),
}


def ranked_table(title: str, df, key: str) -> None:
    st.subheader(title)
    if df.empty:
        st.caption("No tickers qualified this week.")
        return
    # Tall enough to show every row without an inner scrollbar.
    event = st.dataframe(df, column_config=RANK_COLUMNS, hide_index=True,
                         height=35 * (len(df) + 1) + 3,
                         on_select="rerun", selection_mode="single-row",
                         width="stretch", key=key)
    rows = event.selection.rows
    if rows:
        st.session_state["ticker"] = df.iloc[rows[0]]["ticker"]
        if drilldown_page is not None:
            st.switch_page(drilldown_page)


def rankings() -> None:
    st.title("Weekly rankings")
    with db() as con:
        weeks = queries.available_weeks(con)
        if not weeks:
            st.info("No weekly rankings yet. Run `python -m jobs.rank_weekly` after a week of collection.")
            return
        week_start = pick_week(weeks)
        bull, bear = queries.ranked_tables(con, week_start, TOP_N)
        n = queries.eligible_count(con, week_start)

    week = queries.week_for(week_start)
    st.caption(f"Posts from {week.start:%a %d %b} to {week.friday:%a %d %b %Y} (UTC). "
               f"{n} tickers passed the eligibility filters. Click a row to see the posts behind it.")
    ranked_table("Heating up, bullish", bull, "bull")
    ranked_table("Heating up, bearish", bear, "bear")
    if bull["ret_5d"].isna().all() and bear["ret_5d"].isna().all():
        st.caption("Price and volume columns fill in once price data is collected (phase 5).")


def trend_chart(trend):
    base = alt.Chart(trend).encode(x=alt.X("day:T", title=None))
    bars = base.mark_bar(opacity=0.45).encode(
        y=alt.Y("mentions:Q", title="Mentions / day"),
        tooltip=["day:T", "mentions:Q", alt.Tooltip("sentiment:Q", format="+.2f")],
    )
    line = base.mark_line(point=True, color="#d62728").encode(
        y=alt.Y("sentiment:Q", title="Mean sentiment", scale=alt.Scale(domain=[-1, 1])),
    ).transform_filter("isValid(datum.sentiment)")
    return alt.layer(bars, line).resolve_scale(y="independent").properties(height=280)


def drilldown() -> None:
    st.title("Ticker drill-down")
    with db() as con:
        tickers = queries.tracked_tickers(con)
        if not tickers:
            st.info("No tickers have been mentioned yet.")
            return
        weeks = queries.available_weeks(con)
        week_start = pick_week(weeks) if weeks else None
        if st.session_state.get("ticker") not in tickers:
            st.session_state["ticker"] = tickers[0]
        ticker = st.sidebar.selectbox("Ticker", tickers, key="ticker")
        days = st.sidebar.select_slider("Trend window (days)", [30, 60, 90, 180], value=90)

        week = queries.week_for(week_start) if week_start else None
        end = week.end if week else datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        trend = queries.daily_trend(con, ticker, end, days, MODEL)
        history = queries.weekly_history(con, ticker)
        posts = queries.top_posts(con, ticker, week, MODEL) if week else None
        name = queries.company_name(con, ticker)

    st.header(f"{ticker}" + (f" · {name}" if name else ""))

    st.subheader("Mentions and sentiment")
    st.altair_chart(trend_chart(trend), width="stretch")
    if trend["close"].notna().any():
        st.subheader("Price")
        st.line_chart(trend.set_index("day")["close"], height=200)
    else:
        st.caption("Price chart appears once price data is collected (phase 5).")

    st.subheader("Weekly signals")
    if history.empty:
        st.caption("This ticker hasn't passed the eligibility filters in any ranked week yet.")
    else:
        st.dataframe(history, hide_index=True, width="stretch", column_config={
            "week_start": st.column_config.DateColumn("Week of"),
            **{c: st.column_config.NumberColumn(format="%.2f")
               for c in ("attention_z", "sentiment", "momentum", "composite_bull", "composite_bear")},
        })

    if week is not None:
        st.subheader(f"Top posts, week of {week_start:%d %b %Y}")
        if posts is None or posts.empty:
            st.caption("No posts about this ticker that week.")
        else:
            # Link right after the title so it stays visible on narrow screens.
            order = ["created_at", "community", "title", "url", "sentiment", "label",
                     "engagement", "match_type"]
            st.dataframe(posts[order], hide_index=True, width="stretch", column_config={
                "created_at": st.column_config.DatetimeColumn("Posted (UTC)", format="D MMM, HH:mm"),
                "community": st.column_config.TextColumn("Community"),
                "title": st.column_config.TextColumn("Title", width="medium"),
                "sentiment": st.column_config.NumberColumn("Sentiment", format="%+.2f"),
                "label": st.column_config.TextColumn("Label", width="small"),
                "engagement": st.column_config.NumberColumn("Engagement"),
                "match_type": st.column_config.TextColumn("Matched by", width="small"),
                "url": st.column_config.LinkColumn("Link", display_text="open"),
            })


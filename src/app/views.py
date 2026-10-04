"""Page functions for Sentimental (navigation lives in streamlit_app.py).

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

from app import queries, theme
from market.context import divergence_warning
from settings import load_config

APP_NAME = "Sentimental"
DISCLAIMER = "Screening aid only, not investment advice."

cfg = load_config(os.environ.get("SENTIMENT_CONFIG"))
MODEL = cfg["signals"]["sentiment_model"]
TOP_N = cfg["signals"]["top_n"]
CONTEXT_CFG = cfg["context"]
ROW_PX = 40

# Set by streamlit_app.py once the page objects exist, so a click in the
# rankings table can jump to the drill-down.
drilldown_page = None


def db_path() -> str:
    # Read per render (not at import) so the env override always applies.
    return os.environ.get("SENTIMENT_DB_PATH") or cfg["storage"]["db_path"]


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


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def week_label(d) -> str:
    return f"Week of {d:%d %b %Y}"


def pick_week(weeks):
    """Week selector shared by both pages (kept in session state)."""
    if st.session_state.get("week") not in weeks:
        st.session_state["week"] = weeks[0]
    return st.sidebar.selectbox("Week", weeks, format_func=week_label, key="week")


RANK_COLUMNS = {
    "rank": st.column_config.NumberColumn("#", width="small"),
    "ticker": st.column_config.TextColumn("Ticker", width="small"),
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


def ranked_table(title: str, dot: str, df, key: str) -> None:
    with st.container(key=f"card-{key}"):
        html(theme.label(title, dot=dot))
        if df.empty:
            html('<div class="sm-muted">No tickers qualified this week.</div>')
            return
        # Fixed row height, and a table tall enough to show every row without scrolling.
        event = st.dataframe(theme.style_table(df), column_config=RANK_COLUMNS, hide_index=True,
                             row_height=ROW_PX, height=ROW_PX * (len(df) + 1) + 4, on_select="rerun",
                             selection_mode="single-row", width="stretch", key=key)
        rows = event.selection.rows
        if rows:
            st.session_state["ticker"] = df.iloc[rows[0]]["ticker"]
            if drilldown_page is not None:
                st.switch_page(drilldown_page)


def rankings() -> None:
    with db() as con:
        weeks = queries.available_weeks(con)
        if not weeks:
            html(theme.label("Weekly rankings"))
            html(theme.title("No rankings yet"))
            st.info("Run `python -m jobs.rank_weekly` after a week of collection.")
            return
        week_start = pick_week(weeks)
        bull, bear = queries.ranked_tables(con, week_start, TOP_N)
        tape = queries.week_tape(con, week_start)

    week = queries.week_for(week_start)
    html(theme.label(f"{week.start:%a %d %b} – {week.friday:%a %d %b %Y} · UTC"))
    html(theme.title("Weekly rankings"))
    html('<p class="sm-sub">Where attention and sentiment are shifting. '
         'Click a row to read the posts behind it.</p>')
    if not tape.empty:
        html(theme.tape(tape))

    c1, c2, c3, c4 = st.columns(4)
    with c1, st.container(key="card-stat-eligible"):
        html(theme.stat("Eligible tickers", str(len(tape)), "passed the filters"))
    with c2, st.container(key="card-stat-mentions"):
        html(theme.stat("Mentions", f"{int(tape['mentions'].sum()):,}" if not tape.empty else "0",
                        "across eligible tickers"))
    with c3, st.container(key="card-glow-stat-bull"):
        top = bull.iloc[0] if not bull.empty else None
        html(theme.stat("Top bullish", top["ticker"] if top is not None else "–",
                        f"composite {top['composite']:.2f}" if top is not None else "none qualified",
                        tone="green"))
    with c4, st.container(key="card-stat-bear"):
        top = bear.iloc[0] if not bear.empty else None
        html(theme.stat("Top bearish", top["ticker"] if top is not None else "–",
                        f"composite {top['composite']:.2f}" if top is not None else "none qualified",
                        tone="red"))

    ranked_table("Heating up · bullish", theme.GREEN, bull, "bull")

    warnings = bull[bull.apply(lambda r: divergence_warning(r, CONTEXT_CFG), axis=1)] if not bull.empty else bull
    if not warnings.empty:
        with st.container(key="card-warnings"):
            html(theme.label("Divergence · volume up, price down, chatter bullish", dot=theme.RED))
            html(theme.divergence_list(warnings))

    ranked_table("Heating up · bearish", theme.RED, bear, "bear")
    if bull["ret_5d"].isna().all() and bear["ret_5d"].isna().all():
        html('<div class="sm-muted">No price data for this week yet: run '
             '<code>python -m jobs.collect_prices</code> or re-run the ranking.</div>')
    else:
        html('<div class="sm-muted">Price and volume are context only and never affect the ranking. '
             'Early chatter = top-decile composite, price moved less than '
             f'{CONTEXT_CFG["early_chatter"]["max_abs_ret_5d"]:.0%} and volume above '
             f'{CONTEXT_CFG["early_chatter"]["min_rel_volume"]}× normal.</div>')


def trend_chart(trend):
    base = alt.Chart(trend).encode(x=alt.X("day:T", title=None, axis=alt.Axis(format="%d %b")))
    bars = base.mark_bar(color=theme.NEUTRAL_BAR, opacity=0.85, cornerRadiusTopLeft=2,
                         cornerRadiusTopRight=2).encode(
        y=alt.Y("mentions:Q", title="MENTIONS / DAY"),
        tooltip=[alt.Tooltip("day:T", format="%d %b %Y"), "mentions:Q",
                 alt.Tooltip("sentiment:Q", format="+.2f")],
    )
    line = base.mark_line(color=theme.GREEN, strokeWidth=2.5,
                          point=alt.OverlayMarkDef(color=theme.GREEN, size=18)).encode(
        y=alt.Y("sentiment:Q", title="MEAN SENTIMENT", scale=alt.Scale(domain=[-1, 1])),
    ).transform_filter("isValid(datum.sentiment)")
    chart = alt.layer(bars, line).resolve_scale(y="independent").properties(height=280)
    return theme.chart_config(chart)


def price_chart(prices):
    """Close line over volume bars, same layered style as the mentions chart."""
    base = alt.Chart(prices).encode(x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b")))
    vol = base.mark_bar(color=theme.NEUTRAL_BAR, opacity=0.85).encode(
        y=alt.Y("volume:Q", title="VOLUME", axis=alt.Axis(format="~s")),
        tooltip=[alt.Tooltip("date:T", format="%d %b %Y"), alt.Tooltip("close:Q", format=",.2f"),
                 alt.Tooltip("volume:Q", format=",.0f")],
    )
    line = base.mark_line(color=theme.GREEN, strokeWidth=2.5).encode(
        y=alt.Y("close:Q", title="CLOSE", scale=alt.Scale(zero=False)),
    )
    chart = alt.layer(vol, line).resolve_scale(y="independent").properties(height=260)
    return theme.chart_config(chart)


def drilldown() -> None:
    with db() as con:
        tickers = queries.tracked_tickers(con)
        if not tickers:
            html(theme.label("Ticker drill-down"))
            html(theme.title("No tickers yet"))
            st.info("No tickers have been mentioned yet.")
            return
        weeks = queries.available_weeks(con)
        week_start = pick_week(weeks) if weeks else None
        if st.session_state.get("ticker") not in tickers:
            st.session_state["ticker"] = tickers[0]
        ticker = st.sidebar.selectbox("Ticker", tickers, key="ticker")
        days = st.sidebar.select_slider("Trend window (days)", [30, 60, 90, 180], value=90)

        week = queries.week_for(week_start) if week_start else None
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end = week.end if week else today + timedelta(days=1)
        trend = queries.daily_trend(con, ticker, end, days, MODEL)
        prices = queries.daily_prices(con, ticker, end, days)
        history = queries.weekly_history(con, ticker)
        name = queries.company_name(con, ticker)
        if week:
            summary = queries.week_summary(con, ticker, week, MODEL)
            communities = queries.community_breakdown(con, ticker, week, MODEL)
            posts = queries.top_posts(con, ticker, week, MODEL)

    html(theme.label(f"Sentiment · {week_label(week_start)}" if week else "Sentiment"))
    html(theme.title(name or ticker, ticker if name else None))
    if week:
        badges = []
        if summary.get("early_chatter_flag"):
            badges.append(theme.badge("EARLY CHATTER"))
        if divergence_warning({**summary, "sentiment": summary["ranked_sentiment"] or 0}, CONTEXT_CFG):
            badges.append(theme.badge("DIVERGENCE: VOLUME UP, PRICE DOWN", theme.RED))
        if badges:
            html('<div style="display:flex;gap:10px;flex-wrap:wrap;margin:4px 0 8px">'
                 + "".join(badges) + "</div>")

    if week:
        left, right = st.columns([1, 2])
        with left, st.container(key="card-glow-gauge"):
            html(theme.label("Weekly score"))
            ranked = summary["ranked_sentiment"] is not None
            score = summary["ranked_sentiment"] if ranked else summary["mean_score"]
            html(theme.gauge(score, summary["momentum"], ranked))
        with right, st.container(key="card-mix"):
            html(theme.label("Mention mix"))
            html(theme.mix_bar(summary["pos"], summary["neu"], summary["neg"]))
            html('<div style="height:18px"></div>' + theme.label("By source"))
            if communities.empty:
                html('<div class="sm-muted">No posts this week.</div>')
            else:
                html(theme.community_bars(communities))

    with st.container(key="card-trend"):
        html(theme.label(f"Mentions and sentiment · last {days} days"))
        st.altair_chart(trend_chart(trend), width="stretch")

    with st.container(key="card-price"):
        html(theme.label(f"Price and volume · last {days} days"))
        if prices.empty:
            html('<div class="sm-muted">No price data for this ticker yet.</div>')
        else:
            st.altair_chart(price_chart(prices), width="stretch")
        if week:
            html(theme.context_kv(summary))

    with st.container(key="card-history"):
        html(theme.label("Weekly signals"))
        if history.empty:
            html('<div class="sm-muted">This ticker hasn\'t passed the eligibility filters '
                 'in any ranked week yet.</div>')
        else:
            st.dataframe(theme.style_table(history), hide_index=True, width="stretch", column_config={
                "week_start": st.column_config.DateColumn("Week of", format="D MMM YYYY"),
                "mentions": st.column_config.NumberColumn("Mentions"),
                "attention_z": st.column_config.NumberColumn("Attention z", format="%.2f"),
                "sentiment": st.column_config.NumberColumn("Sentiment", format="%+.2f"),
                "momentum": st.column_config.NumberColumn("Momentum", format="%+.2f"),
                "composite_bull": st.column_config.NumberColumn("Bull composite", format="%.2f"),
                "composite_bear": st.column_config.NumberColumn("Bear composite", format="%.2f"),
            })

    if week:
        with st.container(key="card-posts"):
            html(theme.label(f"Top posts · {week_label(week_start)}"))
            if posts.empty:
                html('<div class="sm-muted">No posts about this ticker that week.</div>')
            else:
                html(theme.post_list(posts))

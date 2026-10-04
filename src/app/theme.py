"""Sentimental's look: fluorescent green on black.

Base colours and fonts live in .streamlit/config.toml. This module adds what
the theme can't express: the logo, mono section labels, card styling, glows,
and small HTML pieces (ticker tape, gauge, mix bar, post list).

Everything that renders post text goes through `html.escape`, because titles
come from the internet.
"""

from __future__ import annotations

import math
from html import escape

import altair as alt
import pandas as pd

GREEN = "#39FF14"
RED = "#FF4D4D"
MUTED = "#8A8F98"
TEXT = "#E6E6E6"
BG = "#0A0A0A"
CARD = "#111314"
BORDER = "#1F2421"
ROW_BORDER = "#1A1E1C"
GRID = "#161A18"
NEUTRAL_BAR = "#4A524E"
MONO = "'JetBrains Mono', ui-monospace, monospace"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;700&display=swap');

h1, h2, h3 {{ letter-spacing: -0.5px; }}
h1 {{ letter-spacing: -1px; }}

/* Cards: any container whose key starts with "card" */
[class*="st-key-card"] {{
  background: {CARD};
  border: 1px solid {BORDER};
  border-radius: 16px;
  padding: 24px;
}}
[class*="st-key-card-glow"] {{ box-shadow: 0 0 60px rgba(57,255,20,0.07); }}

.sm-logo {{ font-size: 24px; font-weight: 700; letter-spacing: -0.5px; color: {TEXT};
  padding: 4px 4px 18px; }}
.sm-logo span {{ color: {GREEN}; text-shadow: 0 0 12px rgba(57,255,20,0.6); }}
.sm-label {{ font-family: {MONO}; font-size: 12px; letter-spacing: 1.5px;
  text-transform: uppercase; color: {MUTED}; margin: 0 0 4px; }}
.sm-label .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 4px;
  margin-right: 8px; vertical-align: 1px; }}
.sm-title {{ font-size: 36px; font-weight: 700; letter-spacing: -1px; line-height: 1.15;
  margin: 2px 0 6px; color: {TEXT}; }}
.sm-title .sym {{ font-family: {MONO}; font-size: 20px; font-weight: 500; color: {MUTED};
  margin-left: 10px; letter-spacing: 0; }}
.sm-sub {{ color: {MUTED}; font-size: 14px; margin: 0; }}
.sm-muted {{ color: {MUTED}; font-size: 13px; }}
.sm-footer {{ font-family: {MONO}; font-size: 11px; letter-spacing: 1px; color: {MUTED};
  text-transform: uppercase; }}

.sm-tape {{ display: flex; gap: 36px; overflow-x: auto; white-space: nowrap;
  padding: 10px 2px; border-top: 1px solid {BORDER}; border-bottom: 1px solid {BORDER};
  font-family: {MONO}; font-size: 13px; scrollbar-width: none; }}
.sm-tape .sym {{ color: {MUTED}; margin-right: 8px; }}

.sm-stat .v {{ font-family: {MONO}; font-size: 30px; font-weight: 700; color: {TEXT};
  line-height: 1.1; margin-top: 6px; }}
.sm-stat .v.green {{ color: {GREEN}; text-shadow: 0 0 18px rgba(57,255,20,0.35); }}
.sm-stat .v.red {{ color: {RED}; }}
.sm-stat .n {{ font-family: {MONO}; font-size: 12px; color: {MUTED}; margin-top: 4px; }}

.sm-gauge {{ position: relative; width: 100%; max-width: 300px; margin: 18px auto 0; }}
.sm-gauge svg {{ width: 100%; display: block; }}
.sm-gauge .read {{ position: absolute; left: 0; right: 0; bottom: 2px; text-align: center; }}
.sm-gauge .num {{ font-family: {MONO}; font-size: 52px; font-weight: 700; line-height: 1; color: {TEXT}; }}
.sm-gauge .word {{ font-size: 15px; margin-top: 6px; }}
.sm-scale {{ max-width: 300px; margin: 10px auto 0; display: flex; justify-content: space-between;
  font-family: {MONO}; font-size: 11px; }}

.sm-mix {{ display: flex; height: 14px; border-radius: 7px; overflow: hidden; gap: 3px; margin-top: 16px; }}
.sm-legend {{ display: flex; flex-wrap: wrap; gap: 24px; margin-top: 14px; font-family: {MONO}; font-size: 13px; }}
.sm-bars {{ display: flex; flex-direction: column; gap: 14px; margin-top: 14px; }}
.sm-bar {{ display: flex; align-items: center; gap: 14px; }}
.sm-bar .name {{ flex: 0 0 130px; font-size: 14px; color: {MUTED}; overflow: hidden; text-overflow: ellipsis; }}
.sm-bar .track {{ flex: 1; height: 8px; border-radius: 4px; background: {BORDER}; }}
.sm-bar .fill {{ height: 8px; border-radius: 4px; }}
.sm-bar .val {{ flex: 0 0 52px; text-align: right; font-family: {MONO}; font-size: 14px; }}

.sm-badge {{ display: inline-block; padding: 3px 8px; border-radius: 6px; font-family: {MONO};
  font-size: 11px; letter-spacing: 1px; border: 1px solid currentColor; }}
.sm-warn {{ display: flex; gap: 14px; align-items: baseline; padding: 12px 0;
  border-bottom: 1px solid {ROW_BORDER}; }}
.sm-warn:last-child {{ border-bottom: none; }}
.sm-warn .sym {{ font-family: {MONO}; font-weight: 700; flex: 0 0 64px; color: {TEXT}; }}
.sm-warn .why {{ color: {MUTED}; font-size: 14px; line-height: 1.45; }}
.sm-kv {{ display: flex; flex-wrap: wrap; gap: 28px; margin-top: 14px; font-family: {MONO};
  font-size: 12px; color: {MUTED}; }}
.sm-kv b {{ color: {TEXT}; font-weight: 500; margin-left: 6px; }}

.sm-posts {{ overflow-x: auto; }}
.sm-post {{ display: flex; align-items: center; gap: 16px; padding: 14px 0; border-bottom: 1px solid {ROW_BORDER}; }}
.sm-post .src {{ flex: 0 0 112px; min-width: 0; overflow-wrap: anywhere; font-family: {MONO}; font-size: 11px; color: {MUTED}; line-height: 1.5; }}
.sm-post .txt {{ flex: 1; min-width: 0; font-size: 15px; line-height: 1.4; overflow-wrap: anywhere; }}
.sm-post .txt a {{ color: {TEXT}; text-decoration: none; }}
.sm-post .txt a:hover {{ color: {GREEN}; }}
.sm-post .meta {{ font-family: {MONO}; font-size: 11px; color: {MUTED}; margin-top: 4px; }}
.sm-post .tone {{ flex: 0 0 76px; text-align: right; font-family: {MONO}; font-size: 12px; line-height: 1.5; }}
@media (max-width: 640px) {{ .sm-post .src {{ flex-basis: 84px; }} }}
</style>
"""


def tone_color(score: float | None, band: float = 0.1) -> str:
    if score is None or pd.isna(score):
        return MUTED
    return GREEN if score > band else RED if score < -band else MUTED


def tone_word(score: float | None, band: float = 0.1) -> str:
    if score is None or pd.isna(score):
        return "No data"
    return "Bullish" if score > band else "Bearish" if score < -band else "Neutral"


def arrow(score: float | None, band: float = 0.0) -> str:
    if score is None or pd.isna(score):
        return "■"
    return "▲" if score > band else "▼" if score < -band else "■"


def logo() -> str:
    return '<div class="sm-logo">Sentimental<span>.</span></div>'


def active_nav_css(key: str) -> str:
    return (f"<style>.st-key-{key} a {{ background: rgba(57,255,20,0.1); }}"
            f".st-key-{key} a span {{ color: {GREEN} !important; font-weight: 500; }}</style>")


def label(text: str, dot: str | None = None) -> str:
    d = f'<span class="dot" style="background:{dot};box-shadow:0 0 8px {dot}"></span>' if dot else ""
    return f'<div class="sm-label">{d}{escape(text)}</div>'


def title(text: str, symbol: str | None = None) -> str:
    sym = f'<span class="sym">{escape(symbol)}</span>' if symbol else ""
    return f'<div class="sm-title">{escape(text)}{sym}</div>'


def tape(df: pd.DataFrame) -> str:
    items = "".join(
        f'<span><span class="sym">{escape(r.ticker)}</span>'
        f'<span style="color:{tone_color(r.sentiment)}">{arrow(r.sentiment)} {r.sentiment:+.2f}</span></span>'
        for r in df.itertuples()
    )
    return f'<div class="sm-tape" aria-label="This week\'s eligible tickers and sentiment">{items}</div>'


def stat(name: str, value: str, note: str = "", tone: str = "") -> str:
    return (f'<div class="sm-stat">{label(name)}<div class="v {tone}">{escape(value)}</div>'
            f'<div class="n">{escape(note)}</div></div>')


def gauge(score: float | None, momentum: float | None, ranked: bool) -> str:
    """Semicircle gauge: sentiment in [-1, 1] shown as 0-100."""
    color = tone_color(score)
    if score is None or pd.isna(score):
        value_txt, arc = "–", ""
    else:
        v = max(0.0, min(100.0, (score + 1) * 50))
        a = math.pi * v / 100
        x, y = 100 - 80 * math.cos(a), 100 - 80 * math.sin(a)
        large = 0  # the arc never exceeds 180 degrees
        arc = (f'<path d="M20,100 A80,80 0 {large} 1 {x:.1f},{y:.1f}" fill="none" stroke="{color}" '
               f'stroke-width="14" stroke-linecap="round" '
               f'style="filter: drop-shadow(0 0 8px {color}cc)"></path>') if v > 0.5 else ""
        value_txt = f"{v:.0f}"
    if momentum is None or pd.isna(momentum):
        delta = "No previous week to compare" if ranked else "Not ranked this week: plain average of posts"
    else:
        delta = f"{arrow(momentum)} {momentum * 50:+.0f} vs last week"
    return f"""
<div class="sm-gauge">
  <svg viewBox="0 0 200 110" role="img" aria-label="Sentiment {value_txt} out of 100, {tone_word(score)}">
    <path d="M20,100 A80,80 0 0 1 180,100" fill="none" stroke="{BORDER}" stroke-width="14" stroke-linecap="round"></path>
    {arc}
  </svg>
  <div class="read"><div class="num">{value_txt}</div>
    <div class="word" style="color:{color}">{tone_word(score)}</div></div>
</div>
<div class="sm-scale"><span style="color:{RED}">0 BEARISH</span><span style="color:{GREEN}">100 BULLISH</span></div>
<div class="sm-muted" style="margin-top:18px">{escape(delta)}</div>
"""


def mix_bar(pos: int, neu: int, neg: int) -> str:
    total = pos + neu + neg
    if not total:
        return '<div class="sm-muted" style="margin-top:12px">No scored posts this week.</div>'
    pct = lambda n: round(100 * n / total)  # noqa: E731
    segs = "".join(
        f'<div style="flex:{n};background:{c};{glow}"></div>'
        for n, c, glow in ((pos, GREEN, "box-shadow:0 0 12px rgba(57,255,20,0.5)"),
                           (neu, NEUTRAL_BAR, ""), (neg, RED, "")) if n
    )
    return (f'<div class="sm-mix" role="img" aria-label="Positive {pct(pos)}%, neutral {pct(neu)}%, '
            f'negative {pct(neg)}%">{segs}</div>'
            f'<div class="sm-legend"><span><span style="color:{GREEN}">■</span> Positive {pct(pos)}%</span>'
            f'<span><span style="color:{MUTED}">■</span> Neutral {pct(neu)}%</span>'
            f'<span><span style="color:{RED}">■</span> Negative {pct(neg)}%</span></div>')


def community_bars(df: pd.DataFrame) -> str:
    rows = []
    for r in df.itertuples():
        s = r.sentiment
        width = 0 if pd.isna(s) else (s + 1) * 50
        val = "–" if pd.isna(s) else f"{(s + 1) * 50:.0f}"
        rows.append(
            f'<div class="sm-bar"><div class="name">r/{escape(str(r.community))}</div>'
            f'<div class="track"><div class="fill" style="width:{width:.0f}%;background:{tone_color(s)}"></div></div>'
            f'<div class="val">{val}</div></div>'
        )
    return f'<div class="sm-bars">{"".join(rows)}</div>'


def post_list(df: pd.DataFrame) -> str:
    rows = []
    for r in df.itertuples():
        color = tone_color(r.sentiment)
        tone = ("UNSCORED" if pd.isna(r.sentiment)
                else f"{arrow(r.sentiment, 0.1)} {r.sentiment:+.2f}<br>{tone_word(r.sentiment).upper()}")
        when = pd.Timestamp(r.created_at).strftime("%d %b %H:%M")
        link = escape(r.url or "#", quote=True)
        rows.append(
            f'<div class="sm-post"><div class="src">R/{escape(str(r.community)).upper()}<br>{when}</div>'
            f'<div class="txt"><a href="{link}" target="_blank" rel="noopener noreferrer">{escape(r.title or "(no title)")}</a>'
            f'<div class="meta">{int(r.engagement or 0)} engagement · matched by {escape(str(r.match_type))}</div></div>'
            f'<div class="tone" style="color:{color}">{tone}</div></div>'
        )
    return f'<div class="sm-posts">{"".join(rows)}</div>'


def style_table(df: pd.DataFrame, signed=("sentiment", "momentum")):
    """Green/red text for signed columns; formatting stays in column_config."""
    def color(v):
        if pd.isna(v):
            return ""
        return f"color: {GREEN}" if v > 0 else f"color: {RED}" if v < 0 else ""
    return df.style.map(color, subset=[c for c in signed if c in df.columns])


def chart_config(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure(background="transparent", font="Space Grotesk")
        .configure_view(stroke=None)
        .configure_axis(labelFont="JetBrains Mono", labelColor=MUTED, titleColor=MUTED,
                        titleFont="JetBrains Mono", titleFontWeight=400, titleFontSize=11,
                        gridColor=GRID, domainColor=BORDER, tickColor=BORDER, labelFontSize=11)
    )


def badge(text: str, color: str = GREEN) -> str:
    return f'<span class="sm-badge" style="color:{color}">{escape(text)}</span>'


def fmt_pct(v) -> str:
    return "–" if v is None or pd.isna(v) else f"{v:+.1%}"


def fmt_x(v) -> str:
    return "–" if v is None or pd.isna(v) else f"{v:.2f}×"


def divergence_list(df: pd.DataFrame) -> str:
    """Rows: ticker, ret_5d, rel_volume, sentiment."""
    rows = "".join(
        f'<div class="sm-warn"><div class="sym">{escape(r.ticker)}</div>'
        f'<div class="why">Bullish chatter (sentiment {r.sentiment:+.2f}) while the price fell '
        f'{abs(r.ret_5d):.1%} this week on {r.rel_volume:.1f}× normal volume. '
        f'The crowd may be buying into selling.</div></div>'
        for r in df.itertuples()
    )
    return f'<div>{rows}</div>'


def context_kv(summary: dict) -> str:
    items = [("5D RETURN", fmt_pct(summary.get("ret_5d"))), ("30D RETURN", fmt_pct(summary.get("ret_30d"))),
             ("REL. VOLUME", fmt_x(summary.get("rel_volume"))),
             ("UP/DOWN VOLUME", fmt_x(summary.get("updown_vol_ratio")))]
    return '<div class="sm-kv">' + "".join(f"<span>{k}<b>{v}</b></span>" for k, v in items) + "</div>"

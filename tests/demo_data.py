"""Synthetic multi-week database for app tests and local previews.

All posts are invented. Three storylines make the rankings predictable:
  NVDA  attention and bullish sentiment ramp up in the final week
  INTC  turns sharply bearish in the final week
  others steady background chatter

Preview the app on it (writes data/demo.duckdb):
    python tests/demo_data.py
    SENTIMENT_DB_PATH=data/demo.duckdb streamlit run src/app/streamlit_app.py
"""

from __future__ import annotations

import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from collectors.base import Post, content_hash, hash_author  # noqa: E402
from jobs.rank_weekly import rank  # noqa: E402
from signals.aggregate import Week  # noqa: E402
from storage import db  # noqa: E402
from tickers.extractor import Match  # noqa: E402
from tickers.universe import save_universe  # noqa: E402

LAST_FRIDAY = date(2026, 10, 2)
N_WEEKS = 8
COMPANIES = {
    "NVDA": "NVIDIA Corporation", "INTC": "Intel Corporation", "TSLA": "Tesla, Inc.",
    "AAPL": "Apple Inc.", "AMD": "Advanced Micro Devices, Inc.", "PLTR": "Palantir Technologies Inc.",
    "MSFT": "Microsoft Corporation", "GME": "GameStop Corp.",
}
COMMUNITIES = ["wallstreetbets", "stocks", "investing", "options", "StockMarket"]

CFG = {
    "universe": {"exchanges": ["NASDAQ"], "min_market_cap": 1, "min_avg_dollar_volume": 1},
    "extraction": {},
    "scoring": {},
    "signals": {
        "sentiment_model": "finbert", "min_match_confidence": 0.5,
        "source_weights": {"reddit": 1.0}, "author_cap_per_ticker_week": 5,
        "baseline_days": 30, "attention_std_floor": 1.0, "attention_clip": 10,
        "min_weekly_mentions": 10, "min_distinct_authors": 5,
        "composite_weights": {"attention": 0.35, "sentiment": 0.35, "momentum": 0.15, "breadth": 0.15},
        "top_n": 20,
    },
    "prices": {"context_days": 60},
    "context": {
        "ret_short_sessions": 5, "ret_long_sessions": 21, "rel_volume_baseline_days": 30,
        "early_chatter": {"top_quantile": 0.9, "max_abs_ret_5d": 0.03, "min_rel_volume": 1.2},
        "divergence_warning": {"min_rel_volume": 1.2, "min_price_drop": 0.02},
    },
}

# Final-week price story per ticker: (daily drift, volume multiplier).
#   NVDA  price flat, volume up        -> early chatter (talk before price)
#   GME   price falling on heavy volume while chatter is bullish -> divergence warning
#   INTC  price falling
PRICE_STORY = {"NVDA": (0.0005, 1.8), "GME": (-0.02, 1.7), "INTC": (-0.015, 1.3)}


def _plan(ticker: str, weeks_ago: int) -> tuple[float, float]:
    """(posts per day, mean sentiment) for a ticker in a given week."""
    if ticker == "NVDA":
        return (12.0, 0.55) if weeks_ago == 0 else (3.0, 0.1)
    if ticker == "INTC":
        return (9.0, -0.6) if weeks_ago == 0 else (2.5, 0.05)
    if ticker == "GME":
        return (5.0, 0.35) if weeks_ago == 0 else (2.0, 0.3)
    return 3.0, {"TSLA": -0.15, "AAPL": 0.2, "AMD": 0.1, "PLTR": 0.25, "MSFT": 0.05}[ticker]


def _prices(rng: random.Random, last: Week) -> pd.DataFrame:
    """Invented weekday bars from well before the first ranked week to the last Friday."""
    rows = []
    start = last.friday - timedelta(weeks=N_WEEKS + 10)
    for ticker in COMPANIES:
        price, base_vol = rng.uniform(40, 400), rng.uniform(5e6, 6e7)
        d = start
        while d <= last.friday:
            if d.weekday() < 5:
                final_week = d >= last.start.date()
                drift, vol_mult = PRICE_STORY.get(ticker, (0.0, 1.0)) if final_week else (0.0, 1.0)
                price *= 1 + drift + rng.gauss(0, 0.004 if final_week else 0.012)
                vol = base_vol * vol_mult * rng.uniform(0.85, 1.15)
                rows.append({"ticker": ticker, "date": d, "open": price * 0.995,
                             "high": price * 1.01, "low": price * 0.99, "close": price, "volume": vol})
            d += timedelta(days=1)
    return pd.DataFrame(rows)


def build(path: Path, seed: int = 7) -> list[Week]:
    """Create the demo DB at `path` and rank every week. Returns the weeks, oldest first."""
    rng = random.Random(seed)
    path.unlink(missing_ok=True)
    con = db.connect(path)
    save_universe(con, [
        {"ticker": t, "company_name": n, "exchange": "NASDAQ", "market_cap": 1e11,
         "avg_dollar_volume_30d": 1e9, "updated_at": datetime(2026, 10, 3)}
        for t, n in COMPANIES.items()
    ], min_coverage=0.5)

    last = Week.ending(LAST_FRIDAY)
    first_day = last.start - timedelta(weeks=N_WEEKS - 1, days=30)
    n = 0
    posts, tags, scores = [], {}, []
    for day in range((last.end - first_day).days):
        when = first_day + timedelta(days=day)
        weeks_ago = max(0, (last.end - when - timedelta(seconds=1)).days // 7)
        for ticker in COMPANIES:
            rate, mood = _plan(ticker, weeks_ago)
            for _ in range(rng.choices([int(rate), int(rate) + 1], [1 - rate % 1, rate % 1])[0]):
                n += 1
                score = max(-1.0, min(1.0, rng.gauss(mood, 0.25)))
                title = f"{ticker} thoughts #{n}: " + ("bullish" if score > 0.2 else
                                                       "bearish" if score < -0.2 else "watching")
                pid = f"reddit:demo{n}"
                posts.append(Post(
                    id=pid, source="reddit", community=rng.choice(COMMUNITIES),
                    author_id=hash_author(f"user{rng.randint(1, 400)}"),
                    author_age_days=rng.randint(30, 3000), author_karma=rng.randint(10, 50_000),
                    created_at=when + timedelta(minutes=rng.randint(0, 1439)),
                    collected_at=when + timedelta(days=1), title=title,
                    body=f"Demo post about ${ticker}.",
                    url=f"https://www.reddit.com/r/demo/comments/demo{n}/",
                    engagement=int(rng.lognormvariate(3, 1.2)),
                    content_hash=content_hash(title, None),
                ))
                tags[pid] = [Match(ticker, "cashtag", 0.95)]
                label = "pos" if score > 0.2 else "neg" if score < -0.2 else "neu"
                scores.append((pid, ticker, "finbert", label, round(score, 3), datetime(2026, 10, 3)))
    db.upsert_prices(con, _prices(rng, last))
    # Invented 8-Ks in the final week: NVDA reports earnings; INTC files a
    # restructuring (a red-flag item).
    for acc, ticker, items, day in (("0000000099-26-000001", "NVDA", "2.02,9.01", 3),
                                    ("0000000099-26-000002", "INTC", "2.05,9.01", 4)):
        when = last.start + timedelta(days=day, hours=20, minutes=30)
        con.execute("INSERT INTO sec_filings VALUES (?, ?, 99, '8-K', ?, ?, ?, ?, ?)",
                    [acc, ticker, when.date(), when, items,
                     f"https://www.sec.gov/Archives/edgar/data/99/{acc.replace('-', '')}/{acc}-index.htm", when])
    db.upsert_posts(con, posts)
    db.replace_post_tickers(con, tags)
    db.insert_scores(con, scores)

    weeks = [Week.ending(LAST_FRIDAY - timedelta(weeks=k)) for k in reversed(range(N_WEEKS))]
    for week in weeks:
        rank(con, CFG, week, scorer=None)
    con.close()
    return weeks


if __name__ == "__main__":
    out = ROOT / "data" / "demo.duckdb"
    out.parent.mkdir(exist_ok=True)
    build(out)
    print(f"wrote {out}")

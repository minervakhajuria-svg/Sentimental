"""Fetch daily bars for recently mentioned tickers into prices_daily.

Runs at the end of collect_daily (short refresh) and inside rank_weekly
(longer refresh for the week's eligible tickers). By hand, e.g. to backfill:
    python -m jobs.collect_prices [--days 120] [--config config.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone

import duckdb

from market.prices import MarketDataProvider, YFinanceProvider
from settings import load_config, setup_logging
from storage import db

log = logging.getLogger("jobs.collect_prices")


def update_prices(con: duckdb.DuckDBPyConnection, provider: MarketDataProvider,
                  tickers: list[str], start: date, end: date) -> int:
    """Download bars for sessions in [start, end) and upsert them. Returns rows written."""
    if not tickers:
        return 0
    bars = provider.daily_bars(sorted(set(tickers)), start, end)
    written = db.upsert_prices(con, bars)
    got = bars["ticker"].nunique() if len(bars) else 0
    log.info("prices: %d rows for %d/%d tickers (%s to %s)", written, got, len(set(tickers)),
             start, end - timedelta(days=1))
    if got < len(set(tickers)):
        log.warning("prices: no data for %d tickers", len(set(tickers)) - got)
    return written


def refresh_recent(con: duckdb.DuckDBPyConnection, provider: MarketDataProvider, cfg: dict,
                   days: int | None = None, today: date | None = None) -> int:
    """Refresh the last `days` of bars for tickers mentioned in the last few weeks."""
    pcfg = cfg["prices"]
    today = today or datetime.now(timezone.utc).date()
    since = datetime.combine(today - timedelta(days=pcfg["recent_mention_days"]), datetime.min.time())
    tickers = db.recently_mentioned(con, since)
    days = days or pcfg["daily_refresh_days"]
    return update_prices(con, provider, tickers, today - timedelta(days=days), today + timedelta(days=1))


def provider_from_config(cfg: dict) -> MarketDataProvider:
    return YFinanceProvider(**cfg["universe"]["yfinance"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--days", type=int, default=None, help="how many days back to fetch")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg, "collect_prices")
    con = db.connect(cfg["storage"]["db_path"])
    try:
        refresh_recent(con, provider_from_config(cfg), cfg, days=args.days)
        return 0
    except Exception:
        log.exception("collect_prices failed")
        return 1
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())

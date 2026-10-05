"""Fetch recent SEC filings (8-K events, Form 4 insider trades) for tracked tickers.

Runs as a step of collect_daily; by hand:
    python -m jobs.collect_filings [--days 30] [--config config.yaml]

One request per company for its filing list (shared by both form types), plus
one per new Form 4 to read the trades. Paced under the SEC's 10 requests/second.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone

import duckdb

from market.insiders import EdgarClient, update_insider_cache
from market.sec_filings import store_filings
from settings import load_config, setup_logging
from storage import db

log = logging.getLogger("jobs.collect_filings")


def collect_filings(con: duckdb.DuckDBPyConnection, edgar: EdgarClient, tickers: list[str],
                    since: date, until: date) -> tuple[int, int]:
    """Store 8-Ks and parse new Form 4s. Returns (8-K rows offered, Form 4s added)."""
    eight_k = 0
    for ticker in tickers:
        try:
            cik = edgar.cik(ticker)
            if cik is not None:
                eight_k += store_filings(con, ticker, cik, edgar.filings(cik, {"8-K", "8-K/A"}, since, until))
        except Exception as e:  # one company's bad response shouldn't stop the rest
            log.warning("8-K filings for %s: %s", ticker, e)
    # Reuses the filing lists fetched above, so this only costs the Form 4 documents.
    form4 = update_insider_cache(con, edgar, tickers, since, until)
    log.info("sec: %d 8-Ks seen, %d new Form 4s for %d tickers", eight_k, form4, len(tickers))
    return eight_k, form4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--days", type=int, default=None, help="how many days back to fetch")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg, "collect_filings")
    con = db.connect(cfg["storage"]["db_path"])
    try:
        from jobs.collect_daily import ticker_list
        scfg = cfg["sec"]
        today = datetime.now(timezone.utc).date()
        days = args.days or scfg["lookback_days"]
        collect_filings(con, EdgarClient.from_env(scfg), ticker_list(con, cfg, scfg["max_tickers"])(),
                        today - timedelta(days=days), today)
        return 0
    except Exception:
        log.exception("collect_filings failed")
        return 1
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())

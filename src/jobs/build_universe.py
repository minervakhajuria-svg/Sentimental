"""Refresh the ticker universe: listings plus market cap and liquidity.

Slow (one market-cap lookup per liquid ticker, ~20-30 min), so run it weekly
before rank_weekly, not daily:
    python -m jobs.build_universe [--config config.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys

from market.prices import YFinanceProvider
from settings import load_config, setup_logging
from storage import db
from tickers.universe import build_universe, fetch_listings, load_universe, save_universe

log = logging.getLogger("jobs.build_universe")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg, "build_universe")
    ucfg = cfg["universe"]
    try:
        listings = fetch_listings()
        log.info("fetched %d equity listings", len(listings))
        provider = YFinanceProvider(**ucfg["yfinance"])
        rows = build_universe(listings, provider, ucfg)
        con = db.connect(cfg["storage"]["db_path"])
        try:
            save_universe(con, rows, ucfg["min_data_coverage"])
            log.info("universe saved: %d listings, %d pass the floors",
                     len(rows), len(load_universe(con, ucfg)))
        finally:
            con.close()
        return 0
    except Exception:
        log.exception("build_universe failed; existing universe left unchanged")
        return 1


if __name__ == "__main__":
    sys.exit(main())

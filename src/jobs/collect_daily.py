"""Daily collection job: fetch recent posts from every source, upsert them,
tag new posts with tickers, then refresh recent daily prices.

Run once a day (cron/Task Scheduler):
    python -m jobs.collect_daily [--config config.yaml]

Exit codes: 0 = all sources ok, 2 = partial failure (data that was fetched is
saved), 1 = nothing could be collected. Non-zero codes make failures visible
to the scheduler instead of silently leaving a gap in the history.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta

from collectors.base import Collector
from collectors.reddit import RedditCollector, utc_now
from jobs import collect_prices as collect_prices_job
from jobs import extract_tickers as extract_tickers_job
from market.prices import MarketDataProvider
from settings import load_config, setup_logging
from storage import db

log = logging.getLogger("jobs.collect_daily")

EXIT_OK, EXIT_FAILED, EXIT_PARTIAL = 0, 1, 2


def build_collectors(cfg: dict) -> list[Collector]:
    # Later sources (StockTwits, news, X) get added here.
    return [RedditCollector.from_config(cfg["reddit"])]


def run(cfg: dict, collectors: list[Collector] | None = None,
        extract_tickers: bool = True, price_provider: MarketDataProvider | None = None,
        collect_prices: bool = True) -> int:
    since = utc_now() - timedelta(hours=cfg["collect"]["lookback_hours"])
    collectors = collectors if collectors is not None else build_collectors(cfg)
    con = db.connect(cfg["storage"]["db_path"])
    failures = 0  # collectors with any failure
    dead = 0      # collectors that produced nothing because everything failed
    followup_failed = False
    try:
        for collector in collectors:
            try:
                # Materialise before writing so each source is one atomic upsert.
                posts = list(collector.fetch(since))
                inserted, updated = db.upsert_posts(con, posts)
                log.info("%s: %d fetched, %d new, %d updated",
                         collector.source, len(posts), inserted, updated)
                failed = getattr(collector, "failed_communities", [])
                if failed:
                    failures += 1
                    log.error("%s: failed communities: %s", collector.source, ", ".join(failed))
                    if len(failed) == len(getattr(collector, "communities", failed)):
                        dead += 1
            except Exception:
                failures += 1
                dead += 1
                log.exception("%s: collection failed", collector.source)
        total = con.execute("SELECT count(*) FROM posts").fetchone()[0]
        log.info("posts table now holds %d rows", total)
        if extract_tickers:
            try:
                extract_tickers_job.extract(con, cfg)
            except Exception:
                # Collected posts are already saved; tagging can be rerun later.
                log.exception("ticker extraction failed")
                followup_failed = True
        if collect_prices:
            try:
                provider = price_provider or collect_prices_job.provider_from_config(cfg)
                collect_prices_job.refresh_recent(con, provider, cfg)
            except Exception:
                # Prices are context only; a gap is refilled by the next run.
                log.exception("price refresh failed")
                followup_failed = True
    finally:
        con.close()

    if dead == len(collectors):
        return EXIT_FAILED
    if failures or followup_failed:
        return EXIT_PARTIAL
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg, "collect_daily")
    try:
        code = run(cfg)
    except Exception:
        log.exception("collect_daily aborted")
        code = EXIT_FAILED
    log.info("collect_daily finished with exit code %d", code)
    return code


if __name__ == "__main__":
    sys.exit(main())

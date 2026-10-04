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
from typing import Callable

import duckdb

from collectors.base import Collector
from collectors.news import FinnhubNewsCollector
from collectors.reddit import RedditCollector, utc_now
from collectors.stocktwits import StockTwitsCollector
from jobs import collect_prices as collect_prices_job
from jobs import extract_tickers as extract_tickers_job
from market.prices import MarketDataProvider
from settings import load_config, setup_logging
from storage import db
from tickers.universe import load_universe

log = logging.getLogger("jobs.collect_daily")

EXIT_OK, EXIT_FAILED, EXIT_PARTIAL = 0, 1, 2


def ticker_list(con: duckdb.DuckDBPyConnection, cfg: dict, n: int) -> Callable[[], list[str]]:
    """Tickers for per-ticker APIs: the most mentioned lately, topped up with the
    most liquid names so news collection works even before any posts exist."""
    def pick() -> list[str]:
        since = utc_now() - timedelta(days=cfg["news"]["mention_lookback_days"])
        chosen = db.most_mentioned(con, since, n)
        for t in db.top_liquid_tickers(con, n, cfg["universe"]):
            if len(chosen) >= n:
                break
            if t not in chosen:
                chosen.append(t)
        return chosen
    return pick


def build_collectors(cfg: dict, con: duckdb.DuckDBPyConnection) -> tuple[list[Collector], int]:
    """Collectors for every enabled source, plus how many failed to start.

    Each source is built on its own, so missing credentials for one (e.g.
    Reddit still awaiting approval) don't stop the others from collecting.
    """
    builders = {
        "reddit": lambda: RedditCollector.from_config(cfg["reddit"]),
        "news": lambda: FinnhubNewsCollector.from_config(
            cfg["news"], ticker_list(con, cfg, cfg["news"]["max_tickers"])),
        "stocktwits": lambda: StockTwitsCollector.from_config(
            cfg["stocktwits"], ticker_list(con, cfg, cfg["stocktwits"]["max_tickers"])),
    }
    collectors, failed = [], 0
    for name, build in builders.items():
        if not cfg.get(name, {}).get("enabled", True):
            continue
        try:
            collectors.append(build())
        except Exception as e:
            log.error("%s: not collecting (%s)", name, e)
            failed += 1
    return collectors, failed


def run(cfg: dict, collectors: list[Collector] | None = None,
        extract_tickers: bool = True, price_provider: MarketDataProvider | None = None,
        collect_prices: bool = True) -> int:
    since = utc_now() - timedelta(hours=cfg["collect"]["lookback_hours"])
    con = db.connect(cfg["storage"]["db_path"])
    not_started = 0
    if collectors is None:
        collectors, not_started = build_collectors(cfg, con)
    failures = not_started  # sources with any failure
    dead = not_started      # sources that produced nothing because everything failed
    universe = set(load_universe(con, cfg["universe"])) if "universe" in cfg else set()
    source_conf = cfg.get("extraction", {}).get("confidence", {}).get("source", 0.9)
    followup_failed = False
    try:
        for collector in collectors:
            try:
                # Materialise before writing so each source is one atomic upsert.
                posts = list(collector.fetch(since))
                inserted, updated = db.upsert_posts(con, posts)
                tagged, tags = db.add_source_tags(con, posts, source_conf, universe or None)
                log.info("%s: %d fetched, %d new, %d updated, %d source tickers, %d source tags",
                         collector.source, len(posts), inserted, updated, tagged, tags)
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

    if dead >= len(collectors) + not_started:
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

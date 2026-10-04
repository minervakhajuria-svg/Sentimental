"""Tag posts with the tickers they mention (fills post_tickers).

Runs automatically at the end of collect_daily. Run it by hand with
--rebuild after editing aliases.csv or blocklist.txt, to re-tag every post:
    python -m jobs.extract_tickers [--rebuild] [--config config.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys

import duckdb

from settings import load_config, setup_logging
from storage import db
from tickers.extractor import TickerExtractor
from tickers.universe import load_universe

log = logging.getLogger("jobs.extract_tickers")


def extract(con: duckdb.DuckDBPyConnection, cfg: dict, rebuild: bool = False,
            extractor: TickerExtractor | None = None) -> tuple[int, int]:
    """Tag pending posts (or all of them if `rebuild`). Returns (posts, matches)."""
    posts = db.all_posts_text(con) if rebuild else db.posts_without_tickers(con)
    if not posts:
        return 0, 0
    if extractor is None:
        universe = load_universe(con, cfg["universe"])
        if not universe:
            raise RuntimeError("ticker universe is empty; run `python -m jobs.build_universe` first")
        extractor = TickerExtractor.from_config(set(universe), cfg["extraction"])

    results = {pid: extractor.extract(title, body) for pid, title, body in posts}
    if rebuild:
        written = db.replace_post_tickers(con, results)
    else:
        # Pending posts have no rows yet, so only posts with matches need writing.
        written = db.replace_post_tickers(con, {k: v for k, v in results.items() if v})
    tagged = sum(1 for v in results.values() if v)
    log.info("checked %d posts: %d mention a ticker, %d matches written",
             len(posts), tagged, written)
    return len(posts), written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--rebuild", action="store_true", help="re-tag every post")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg, "extract_tickers")
    con = db.connect(cfg["storage"]["db_path"])
    try:
        extract(con, cfg, rebuild=args.rebuild)
        return 0
    except Exception:
        log.exception("extract_tickers failed")
        return 1
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())

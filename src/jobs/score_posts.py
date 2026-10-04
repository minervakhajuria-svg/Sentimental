"""Score every (post, ticker) pair that has no pass-1 sentiment yet.

Runs as the first step of rank_weekly; can also be run on its own:
    python -m jobs.score_posts [--config config.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import duckdb

from scoring.context import ContextBuilder
from scoring.finbert import FinBERTScorer, Scorer
from settings import load_config, setup_logging
from storage import db
from tickers.extractor import load_aliases

log = logging.getLogger("jobs.score_posts")


def score_pending(con: duckdb.DuckDBPyConnection, scorer: Scorer, cfg: dict) -> int:
    """Score unscored pairs in chunks. Returns the number of scores written.

    Each chunk is committed separately, so an interrupted run keeps its
    progress and the next run carries on from there.
    """
    pairs = db.pairs_to_score(con, scorer.name)
    if not pairs:
        log.info("nothing to score")
        return 0
    log.info("%d (post, ticker) pairs to score with %s", len(pairs), scorer.name)
    ctx = ContextBuilder(load_aliases(), max_chars=cfg["max_chars"])
    chunk = cfg["commit_every"]
    written = 0
    for i in range(0, len(pairs), chunk):
        part = pairs[i : i + chunk]
        texts = [ctx.build(title, body, ticker, n) for _, ticker, title, body, n in part]
        # Reposts and single-ticker posts share text: score each distinct text once.
        unique = list(dict.fromkeys(texts))
        by_text = dict(zip(unique, scorer.score(unique)))
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        rows = [
            (post_id, ticker, scorer.name, by_text[t].label, by_text[t].score, now)
            for (post_id, ticker, *_), t in zip(part, texts)
        ]
        written += db.insert_scores(con, rows)
        log.info("scored %d/%d", min(i + chunk, len(pairs)), len(pairs))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg, "score_posts")
    con = db.connect(cfg["storage"]["db_path"])
    try:
        scfg = cfg["scoring"]
        score_pending(con, FinBERTScorer(scfg["model_name"], scfg["batch_size"]), scfg)
        return 0
    except Exception:
        log.exception("score_posts failed")
        return 1
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())

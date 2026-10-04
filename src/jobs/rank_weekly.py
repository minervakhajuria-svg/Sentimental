"""Weekly ranking: tag, score, aggregate and write the two ranked lists.

Run on the weekend after Friday's US close:
    python -m jobs.rank_weekly [--week-ending YYYY-MM-DD] [--no-score]

Screening aid only, not investment advice.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timezone

import duckdb
import pandas as pd

from jobs import extract_tickers as extract_tickers_job
from jobs.score_posts import score_pending
from scoring.finbert import Scorer
from settings import load_config, setup_logging
from signals.aggregate import Week, compute_components
from signals.composite import add_composites, apply_eligibility, ranked_lists, write_signals
from storage import db
from tickers.universe import load_universe

log = logging.getLogger("jobs.rank_weekly")

DISCLAIMER = "Screening aid only, not investment advice."


def rank(con: duckdb.DuckDBPyConnection, cfg: dict, week: Week,
         scorer: Scorer | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute and store one week's signals; return (bullish, bearish) lists.

    Pass `scorer=None` to skip scoring (e.g. when scores are already up to date).
    """
    extract_tickers_job.extract(con, cfg)
    if scorer is not None:
        score_pending(con, scorer, cfg["scoring"])

    scfg = cfg["signals"]
    components = compute_components(con, week, scfg)
    universe = set(load_universe(con, cfg["universe"]))
    eligible = apply_eligibility(components, universe, scfg)
    log.info("week %s: %d tickers mentioned, %d eligible",
             week.week_start, len(components), len(eligible))
    signals = add_composites(eligible, scfg["composite_weights"]) if not eligible.empty else eligible
    write_signals(con, week, signals)
    return ranked_lists(con, week.week_start, scfg["top_n"])


def format_list(title: str, df: pd.DataFrame) -> str:
    if df.empty:
        return f"{title}\n  (no tickers qualified)"
    cols = ["ticker", "composite", "mentions", "attention_z", "sentiment", "momentum", "breadth"]
    table = df[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}")
    return f"{title}\n{table}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--week-ending", type=date.fromisoformat,
                        help="Friday the week ends on (default: latest completed week)")
    parser.add_argument("--no-score", action="store_true", help="skip FinBERT scoring")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg, "rank_weekly")

    try:
        week = (Week.ending(args.week_ending) if args.week_ending
                else Week.latest_completed(datetime.now(timezone.utc)))
        scorer = None
        if not args.no_score:
            from scoring.finbert import FinBERTScorer
            s = cfg["scoring"]
            scorer = FinBERTScorer(s["model_name"], s["batch_size"])
        con = db.connect(cfg["storage"]["db_path"])
        try:
            bull, bear = rank(con, cfg, week, scorer)
        finally:
            con.close()
    except Exception:
        log.exception("rank_weekly failed")
        return 1

    print(f"\nWeek of {week.week_start} ({week.start:%a %d %b} - {week.friday:%a %d %b})\n")
    print(format_list("Heating up, bullish", bull))
    print()
    print(format_list("Heating up, bearish", bear))
    print(f"\n{DISCLAIMER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

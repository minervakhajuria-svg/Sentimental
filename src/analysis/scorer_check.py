"""How well does FinBERT read retail chatter? Check it against authors' own tags.

StockTwits authors can tag a message Bullish or Bearish (stored as model
`stocktwits_tag`). Comparing FinBERT's score sign on the same (post, ticker)
pairs gives a cheap, honest accuracy estimate. Neutral FinBERT calls are
reported separately: on tagged messages they're misses, not wrong-way calls.

    python -m analysis.scorer_check [--model finbert] [--tag-model stocktwits_tag]
"""

from __future__ import annotations

import argparse
import sys

import duckdb

from settings import load_config
from storage import db


def agreement(con: duckdb.DuckDBPyConnection, model: str, tag_model: str,
              neutral_band: float = 0.1) -> dict:
    rows = con.execute(
        """SELECT t.score AS tag, m.score AS pred FROM post_scores t
           JOIN post_scores m ON m.post_id = t.post_id AND m.ticker = t.ticker AND m.model = ?
           WHERE t.model = ?""",
        [model, tag_model],
    ).fetchall()
    n = len(rows)
    right = sum(1 for tag, pred in rows if pred > neutral_band and tag > 0 or pred < -neutral_band and tag < 0)
    neutral = sum(1 for _, pred in rows if abs(pred) <= neutral_band)
    wrong = n - right - neutral
    return {
        "pairs": n,
        "agree": right,
        "neutral": neutral,
        "opposite": wrong,
        "accuracy": right / n if n else None,
        "accuracy_when_decisive": right / (right + wrong) if right + wrong else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default="finbert")
    parser.add_argument("--tag-model", default="stocktwits_tag")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    con = db.connect(cfg["storage"]["db_path"])
    try:
        r = agreement(con, args.model, args.tag_model)
    finally:
        con.close()
    if not r["pairs"]:
        print("No tagged messages scored yet (needs StockTwits data plus a scoring run).")
        return 0
    print(f"{r['pairs']} tagged pairs: {r['agree']} agree, {r['neutral']} neutral, {r['opposite']} opposite")
    print(f"accuracy {r['accuracy']:.1%}; when FinBERT takes a side: {r['accuracy_when_decisive']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

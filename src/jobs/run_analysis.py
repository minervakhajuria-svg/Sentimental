"""Lead-lag analysis and backtest over all ranked weeks.

    python -m jobs.run_analysis [--refresh-prices] [--config config.yaml]

--refresh-prices re-downloads the full adjusted price history for every
ranked ticker first. Do this before trusting results: stored bars from
different dates can disagree across a stock split.

Prints a report and saves it to data/reports/analysis_<date>.json.
Screening aid only, not investment advice.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.backtest import run_backtest
from analysis.leadlag import build_panel, lead_lag_table, verdict
from jobs.collect_prices import provider_from_config, update_prices
from settings import load_config, setup_logging
from storage import db

log = logging.getLogger("jobs.run_analysis")


def analyse(con, cfg: dict) -> dict:
    acfg = {**cfg["analysis"], "top_n": cfg["signals"]["top_n"]}
    panel = build_panel(con)
    if panel.empty:
        return {"weeks": 0, "enough_data": False, "panel": panel}
    table = lead_lag_table(panel, acfg["min_tickers_per_week"])
    bt = run_backtest(panel, acfg)
    weeks = int(panel["week_start"].nunique())
    return {
        "weeks": weeks,
        "enough_data": bt["weeks_with_returns"] >= acfg["min_weeks"],
        "min_weeks": acfg["min_weeks"],
        "lead_lag": table,
        "verdicts": {s: verdict(table, s) for s in ("sentiment", "attention_z", "composite_bull")},
        "backtest": bt,
        "panel": panel,
    }


def refresh_history(con, cfg: dict) -> None:
    span = con.execute("SELECT min(week_start), max(week_start) FROM weekly_signals").fetchone()
    if not span[0]:
        return
    tickers = [r[0] for r in con.execute("SELECT DISTINCT ticker FROM weekly_signals").fetchall()]
    start = span[0] - timedelta(weeks=3)
    end = min(span[1] + timedelta(weeks=2), datetime.now(timezone.utc).date() + timedelta(days=1))
    update_prices(con, provider_from_config(cfg), tickers, start, end)


def _jsonable(o):
    if isinstance(o, pd.DataFrame):
        return o.replace({np.nan: None}).to_dict(orient="records")
    if isinstance(o, pd.Series):
        return {str(k): (None if pd.isna(v) else float(v)) for k, v in o.items()}
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    return str(o)


def _num(v, fmt: str) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, fmt)


def format_report(r: dict) -> str:
    if not r["weeks"]:
        return "No ranked weeks yet."
    bt = r["backtest"]
    lines = [f"Ranked weeks: {r['weeks']} (with next-week returns: {bt['weeks_with_returns']})"]
    if not r["enough_data"]:
        lines.append(f"WARNING: fewer than {r['min_weeks']} weeks. Treat everything below as noise.")
    lines += ["", "Lead-lag (mean weekly Spearman IC):"]
    t = r["lead_lag"].pivot(index="signal", columns="horizon", values="mean_ic")
    lines.append(t.to_string(float_format=lambda v: _num(v, "+.3f"), na_rep="n/a"))
    lines += ["", *(f"- {v}" for v in r["verdicts"].values()), ""]
    ic = bt["ic_bull"]
    lines.append(f"Bull composite IC vs next week: {_num(ic['mean_ic'], '+.3f')} "
                 f"(t={_num(ic['t_stat'], '.2f')}, {ic['weeks']} weeks)")
    lines.append(f"Top-minus-bottom quintile next-week return: {_num(bt['quintile_spread']['mean'], '+.2%')} "
                 f"over {bt['quintile_spread']['weeks']} weeks")
    hr = bt["hit_rates"]
    for side in ("bull", "bear"):
        lines.append(f"Hit rate, {side} list: {_num(hr[f'{side}_list']['hit_rate'], '.1%')} vs mentions-only "
                     f"{_num(hr[f'{side}_baseline_mentions']['hit_rate'], '.1%')}")
    rv = bt["rel_volume_experiment"]
    outcome = {True: "improves", False: "no improvement", None: "n/a"}[rv["improves"]]
    lines.append(f"rel_volume experiment: IC {_num(rv['base']['mean_ic'], '+.3f')} -> "
                 f"{_num(rv['with_rel_volume']['mean_ic'], '+.3f')} ({outcome})")
    lines += ["", "Screening aid only, not investment advice."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--refresh-prices", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg, "run_analysis")
    con = db.connect(cfg["storage"]["db_path"])
    try:
        if args.refresh_prices:
            refresh_history(con, cfg)
        report = analyse(con, cfg)
    except Exception:
        log.exception("run_analysis failed")
        return 1
    finally:
        con.close()
    print(format_report(report))
    out_dir = Path(cfg["storage"]["db_path"]).parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"analysis_{datetime.now(timezone.utc):%Y-%m-%d}.json"
    saved = {k: v for k, v in report.items() if k != "panel"}
    path.write_text(json.dumps(saved, default=_jsonable, indent=2), encoding="utf-8")
    print(f"\nSaved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

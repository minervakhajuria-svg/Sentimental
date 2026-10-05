"""8-K event context, against invented EDGAR fixtures."""

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from jobs.collect_filings import collect_filings
from jobs.rank_weekly import add_events
from market.insiders import EdgarClient
from market.sec_filings import categories, describe, week_events
from signals.aggregate import Week
from storage import db

EDGAR = Path(__file__).resolve().parent.parent / "fixtures" / "edgar"
WEEK = Week.ending(date(2026, 10, 2))   # Sat 26 Sep 00:00 UTC .. Sat 3 Oct 00:00 UTC


def fake_edgar(calls=None):
    calls = [] if calls is None else calls

    def get_json(url, headers=None):
        calls.append(url)
        if url.endswith("company_tickers.json"):
            return json.loads((EDGAR / "company_tickers.json").read_text())
        if "CIK0000000002" in url:
            return json.loads((EDGAR / "submissions_othr.json").read_text())
        if "CIK0000000001" in url:
            return json.loads((EDGAR / "submissions_exmp.json").read_text())
        return {"filings": {"recent": {}}}

    def get_text(url, headers=None):
        calls.append(url)
        return (EDGAR / url.rsplit("/", 1)[1]).read_text()

    return EdgarClient("Test Person test@example.com", get_json, get_text, pause_seconds=0, sleep=lambda s: None)


def test_item_descriptions_and_categories():
    assert describe("2.02,9.01") == ["Earnings results"]       # exhibits item hidden
    assert describe("4.02") == ["Prior financials can't be relied on"]
    assert categories(["2.02,9.01", "5.02", "4.02"]) == ["Earnings", "Leadership", "Red flag"]
    assert categories(["9.01"]) == []


def test_filings_parse_items_acceptance_and_links():
    e = fake_edgar()
    f = {x["accession"]: x for x in e.filings(2, {"8-K", "8-K/A"}, date(2026, 9, 1), date(2026, 10, 5))}
    assert set(f) == {"0000000002-26-000010", "0000000002-26-000008", "0000000002-26-000007",
                      "0000000002-26-000006"}                                   # July one outside window
    r = f["0000000002-26-000008"]
    assert r["items"] == "4.02,9.01" and r["accepted_at"] == datetime(2026, 10, 2, 21, 5)
    assert r["index_url"] == ("https://www.sec.gov/Archives/edgar/data/2/000000000226000008/"
                              "0000000002-26-000008-index.htm")


def test_one_filing_list_request_per_company(tmp_path):
    con = db.connect(tmp_path / "s.duckdb")
    calls = []
    eight_k, form4 = collect_filings(con, fake_edgar(calls), ["OTHR"], date(2026, 9, 1), date(2026, 10, 5))
    assert eight_k == 4 and form4 == 1
    assert sum("submissions" in u for u in calls) == 1     # shared by 8-K and Form 4
    # Idempotent: a second run stores nothing new.
    collect_filings(con, fake_edgar(), ["OTHR"], date(2026, 9, 1), date(2026, 10, 5))
    assert con.execute("SELECT count(*) FROM sec_filings").fetchone()[0] == 4
    con.close()


def test_week_events_respect_acceptance_cutoff(tmp_path):
    con = db.connect(tmp_path / "w.duckdb")
    collect_filings(con, fake_edgar(), ["OTHR"], date(2026, 9, 1), date(2026, 10, 5))
    events = week_events(con, ["OTHR", "NONE"], WEEK.start, WEEK.end)
    # Includes Mon officer change (8-K/A), Tue earnings and the Fri 21:05 UTC restatement;
    # the 8.01 accepted Sat 01:15 UTC is after the cutoff even though it's dated Friday.
    assert events == {"OTHR": ("Earnings, Leadership; Red flag: Prior financials can't be relied on", True)}
    con.close()


def test_rank_adds_event_columns(tmp_path):
    con = db.connect(tmp_path / "r.duckdb")
    signals = pd.DataFrame({"composite_bull": [1.0, 0.5]}, index=["OTHR", "QUIET"])
    assert "events" not in add_events(con, signals, WEEK)   # no SEC data at all: stay NULL
    collect_filings(con, fake_edgar(), ["OTHR"], date(2026, 9, 1), date(2026, 10, 5))
    out = add_events(con, signals, WEEK)
    assert out.loc["OTHR", "events"].endswith("Red flag: Prior financials can't be relied on")
    assert out.loc["OTHR", "red_flag"]
    assert pd.isna(out.loc["QUIET", "events"]) and not out.loc["QUIET", "red_flag"]
    con.close()


def test_migration_is_repeatable(tmp_path):
    path = tmp_path / "m.duckdb"
    for _ in range(2):  # ADD COLUMN IF NOT EXISTS must be safe on every connect
        c = db.connect(path)
        cols = {r[0] for r in c.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'weekly_signals'").fetchall()}
        c.close()
    assert {"events", "red_flag"} <= cols


def test_filings_list_escapes_and_flags():
    from app import theme
    df = pd.DataFrame([{"accepted_at": pd.Timestamp("2026-10-02 21:05"), "form": "8-K",
                        "items": "4.02,9.01", "url": 'https://x/"><script>'}])
    out = theme.filings_list(df)
    assert "Prior financials can" in out and "RED FLAG" in out
    assert "<script>" not in out

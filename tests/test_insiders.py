"""SEC Form 4 insider-buying flag, against invented EDGAR fixtures."""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from market.insiders import EdgarClient, insider_buy_flags, parse_form4, update_insider_cache
from storage import db

EDGAR = Path(__file__).resolve().parent.parent / "fixtures" / "edgar"
FRIDAY = date(2026, 10, 2)


def fake_edgar(calls):
    def get_json(url, headers=None):
        calls.append((url, headers))
        if url.endswith("company_tickers.json"):
            return json.loads((EDGAR / "company_tickers.json").read_text())
        if "CIK0000000001" in url:
            return json.loads((EDGAR / "submissions_exmp.json").read_text())
        return {"filings": {"recent": {}}}

    def get_text(url, headers=None):
        calls.append((url, headers))
        return (EDGAR / url.rsplit("/", 1)[1]).read_text()

    return EdgarClient("Test Person test@example.com", get_json, get_text, pause_seconds=0, sleep=lambda s: None)


def test_parse_form4_counts_only_open_market_purchases():
    count, value = parse_form4((EDGAR / "form4_purchase.xml").read_text())
    assert count == 1 and value == pytest.approx(50250.0)
    assert parse_form4((EDGAR / "form4_option_exercise.xml").read_text()) == (0, 0.0)


def test_cik_lookup_handles_share_classes():
    e = fake_edgar([])
    assert e.cik("EXMP") == 1 and e.cik("BRK.B") == 3 and e.cik("NOPE") is None


def test_filings_in_window_and_document_url():
    e = fake_edgar([])
    filings = e.form4_filings(1, date(2026, 9, 2), FRIDAY)
    assert [f["accession"] for f in filings] == ["0000000001-26-000003", "0000000001-26-000001"]
    assert filings[0]["url"].endswith("/1/000000000126000003/form4_purchase.xml")  # XSL prefix stripped


def test_cache_and_flags(tmp_path):
    con = db.connect(tmp_path / "i.duckdb")
    calls = []
    e = fake_edgar(calls)
    assert update_insider_cache(con, e, ["EXMP", "OTHR"], date(2026, 9, 2), FRIDAY) == 2
    assert all(h == {"User-Agent": "Test Person test@example.com"} for _, h in calls)
    n_calls = len(calls)
    # Second run: filings are cached, so no XML is fetched again.
    assert update_insider_cache(con, e, ["EXMP"], date(2026, 9, 2), FRIDAY) == 0
    assert not any(u.endswith(".xml") for u, _ in calls[n_calls:])

    assert insider_buy_flags(con, ["EXMP", "OTHR"], FRIDAY, 30) == {"EXMP": True, "OTHR": False}
    # Lookahead guard: ranking an earlier week can't see the 30 Sep purchase.
    assert insider_buy_flags(con, ["EXMP"], date(2026, 9, 25), 30) == {"EXMP": False}
    con.close()


def test_from_env_requires_user_agent(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        EdgarClient.from_env({})


def test_rank_sets_insider_flag(tmp_path):
    from jobs.rank_weekly import add_insider_flags
    from signals.aggregate import Week
    con = db.connect(tmp_path / "r.duckdb")
    signals = pd.DataFrame({"composite_bull": [1.0, 0.5]}, index=["EXMP", "OTHR"])
    week = Week.ending(FRIDAY)
    out = add_insider_flags(con, signals, week, {"enabled": True, "lookback_days": 30}, fake_edgar([]))
    assert out["insider_buy_flag"].tolist() == [True, False]
    # Not configured and nothing cached: the column stays absent (NULL in the table).
    empty = db.connect(tmp_path / "e.duckdb")
    assert "insider_buy_flag" not in add_insider_flags(empty, signals, week, {"enabled": True, "lookback_days": 30}, None)
    con.close(); empty.close()

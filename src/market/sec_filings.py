"""SEC 8-K filings as event context (EDGAR, free).

An 8-K is a company's official notice that something material happened. Each
lists item codes (e.g. "2.02,9.01"), so the event type is known without
reading the document. Two uses, both context only, never ranking inputs:

  * events     what happened this week ("Earnings", "Leadership change", ...).
               Attention that spikes in an earnings week is scheduled, not news.
  * red_flag   items that usually signal trouble: restatement, delisting
               notice, auditor change, bankruptcy, impairment, debt default.

Weekly columns use the SEC's acceptance timestamp, so a filing accepted after
the ranking cutoff (Saturday 00:00 UTC) can't leak into that week.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import duckdb

log = logging.getLogger(__name__)

# Form 8-K item codes (SEC General Instructions).
ITEM_NAMES = {
    "1.01": "Material agreement", "1.02": "Agreement terminated", "1.03": "Bankruptcy or receivership",
    "1.04": "Mine safety", "1.05": "Cybersecurity incident",
    "2.01": "Acquisition or disposition completed", "2.02": "Earnings results",
    "2.03": "New debt or obligation", "2.04": "Debt acceleration or default trigger",
    "2.05": "Restructuring or exit costs", "2.06": "Material impairment",
    "3.01": "Delisting notice", "3.02": "Unregistered share sale", "3.03": "Change to shareholder rights",
    "4.01": "Auditor change", "4.02": "Prior financials can't be relied on",
    "5.01": "Change in control", "5.02": "Director or officer change", "5.03": "Bylaw or charter change",
    "5.04": "Benefit plan trading suspension", "5.05": "Code of ethics change", "5.07": "Shareholder vote",
    "5.08": "Director nominations", "7.01": "Reg FD disclosure", "8.01": "Other event",
    "9.01": "Financial statements and exhibits",
}

# Short labels shown in the rankings, by category. 9.01 (exhibits) is a
# companion item with no meaning of its own, so it's not shown.
CATEGORIES = {
    "Earnings": {"2.02"},
    "Deal": {"1.01", "1.02", "2.01", "5.01"},
    "Leadership": {"5.02"},
    "Financing": {"2.03", "3.02", "3.03"},
    "Red flag": {"1.03", "1.05", "2.04", "2.05", "2.06", "3.01", "4.01", "4.02"},
    "Other": {"1.04", "5.03", "5.04", "5.05", "5.07", "5.08", "7.01", "8.01"},
}
RED_FLAG_ITEMS = CATEGORIES["Red flag"]
_ORDER = list(CATEGORIES)


def split_items(items: str | None) -> list[str]:
    return [i.strip() for i in (items or "").split(",") if i.strip()]


def describe(items: str | None) -> list[str]:
    """Human names for an 8-K's items, skipping the exhibits companion item."""
    return [ITEM_NAMES.get(i, f"Item {i}") for i in split_items(items) if i != "9.01"]


def categories(items_list: list[str]) -> list[str]:
    found = {cat for items in items_list for i in split_items(items)
             for cat, codes in CATEGORIES.items() if i in codes}
    return [c for c in _ORDER if c in found]


def store_filings(con: duckdb.DuckDBPyConnection, ticker: str, cik: int, filings: list[dict]) -> int:
    """Insert 8-K rows (idempotent on accession). Returns rows offered."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = [[f["accession"], ticker, cik, f["form"], f["filing_date"], f["accepted_at"], f["items"],
             f["index_url"], now] for f in filings]
    if rows:
        con.executemany(
            "INSERT INTO sec_filings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING", rows)
    return len(rows)


def week_events(con: duckdb.DuckDBPyConnection, tickers: list[str], start: datetime,
                end: datetime) -> dict[str, tuple[str, bool]]:
    """Per ticker with 8-Ks accepted in [start, end):
    ("Earnings, Leadership; Red flag: Auditor change", red_flag)."""
    rows = con.execute(
        """SELECT ticker, list(items) FROM sec_filings
           WHERE ticker IN (SELECT unnest(?)) AND form IN ('8-K', '8-K/A')
             AND coalesce(accepted_at, CAST(filing_date AS TIMESTAMP)) >= ?
             AND coalesce(accepted_at, CAST(filing_date AS TIMESTAMP)) < ?
           GROUP BY ticker""",
        [list(tickers), start, end],
    ).fetchall()
    out = {}
    for ticker, items_list in rows:
        cats = categories(items_list)
        red = sorted({ITEM_NAMES[i] for items in items_list for i in split_items(items) if i in RED_FLAG_ITEMS})
        # Name the red-flag items themselves: "Red flag" alone says nothing useful.
        label = ", ".join(c for c in cats if c != "Red flag")
        if red:
            label = (label + "; " if label else "") + "Red flag: " + ", ".join(red)
        out[ticker] = (label, bool(red))
    return out


def filings_between(con: duckdb.DuckDBPyConnection, ticker: str, start: datetime, end: datetime):
    """8-Ks for one ticker in [start, end), newest first (for the drill-down)."""
    return con.execute(
        """SELECT coalesce(accepted_at, CAST(filing_date AS TIMESTAMP)) AS accepted_at, form, items, url
           FROM sec_filings WHERE ticker = ? AND form IN ('8-K', '8-K/A')
             AND coalesce(accepted_at, CAST(filing_date AS TIMESTAMP)) >= ?
             AND coalesce(accepted_at, CAST(filing_date AS TIMESTAMP)) < ?
           ORDER BY 1 DESC""",
        [ticker, start, end],
    ).df()

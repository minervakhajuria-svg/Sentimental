"""Insider open-market purchases from SEC Form 4 filings (EDGAR, free).

insider_buy_flag = an insider filed an open-market *purchase* (transaction
code "P", shares acquired) within the lookback window ending on the week's
Friday, by filing date, so no filing from after ranking time is used.
Context only: it never feeds the composite.

SEC fair-access rules: identify yourself in the User-Agent (name + email, set
SEC_USER_AGENT in .env) and stay under 10 requests/second. Each filing is
parsed once and cached in `insider_filings`.
"""

from __future__ import annotations

import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from typing import Callable

import duckdb

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
_XSL_PREFIX = re.compile(r"^xslF345X\d+/")


def parse_form4(xml_text: str) -> tuple[int, float]:
    """(number of open-market purchases, total $ value) in one Form 4."""
    root = ET.fromstring(xml_text)
    count, value = 0, 0.0
    for tx in root.iter("nonDerivativeTransaction"):
        code = tx.findtext("transactionCoding/transactionCode", "").strip()
        acq = tx.findtext("transactionAmounts/transactionAcquiredDisposedCode/value", "").strip()
        if code == "P" and acq == "A":
            count += 1
            try:
                shares = float(tx.findtext("transactionAmounts/transactionShares/value", "0"))
                price = float(tx.findtext("transactionAmounts/transactionPricePerShare/value", "0") or 0)
                value += shares * price
            except ValueError:
                pass
    return count, value


class EdgarClient:
    def __init__(self, user_agent: str, get_json: Callable, get_text: Callable,
                 pause_seconds: float = 0.15, sleep: Callable[[float], None] = time.sleep):
        self.headers = {"User-Agent": user_agent}
        self._get_json, self._get_text = get_json, get_text
        self.pause_seconds = pause_seconds
        self._sleep = sleep
        self._ciks: dict[str, int] | None = None

    @classmethod
    def from_env(cls, cfg: dict) -> "EdgarClient":
        ua = os.environ.get("SEC_USER_AGENT")
        if not ua:
            raise RuntimeError("Missing SEC_USER_AGENT in .env (SEC asks for 'Your Name your@email')")
        from collectors.http import get_json, get_text
        return cls(ua, get_json, get_text, pause_seconds=cfg.get("pause_seconds", 0.15))

    def _json(self, url):
        self._sleep(self.pause_seconds)
        return self._get_json(url, headers=self.headers)

    def cik(self, ticker: str) -> int | None:
        if self._ciks is None:
            data = self._json(TICKERS_URL)
            # SEC writes share classes with a dash (BRK-B); we use a dot.
            self._ciks = {row["ticker"].upper().replace("-", "."): int(row["cik_str"]) for row in data.values()}
        return self._ciks.get(ticker.upper())

    def form4_filings(self, cik: int, since: date, until: date) -> list[dict]:
        recent = self._json(SUBMISSIONS_URL.format(cik=cik)).get("filings", {}).get("recent", {})
        out = []
        for form, acc, filed, doc in zip(recent.get("form", []), recent.get("accessionNumber", []),
                                         recent.get("filingDate", []), recent.get("primaryDocument", [])):
            filed_on = date.fromisoformat(filed)
            if form == "4" and since <= filed_on <= until:
                out.append({"accession": acc, "filing_date": filed_on,
                            "url": ARCHIVE_URL.format(cik=cik, acc=acc.replace("-", ""),
                                                      doc=_XSL_PREFIX.sub("", doc))})
        return out

    def form4_xml(self, url: str) -> str:
        self._sleep(self.pause_seconds)
        return self._get_text(url, headers=self.headers)


def update_insider_cache(con: duckdb.DuckDBPyConnection, edgar: EdgarClient, tickers: list[str],
                         since: date, until: date) -> int:
    """Fetch and parse Form 4s not yet cached. Returns filings added."""
    known = {r[0] for r in con.execute("SELECT accession FROM insider_filings").fetchall()}
    added = 0
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for ticker in tickers:
        try:
            cik = edgar.cik(ticker)
            if cik is None:
                continue
            for f in edgar.form4_filings(cik, since, until):
                if f["accession"] in known:
                    continue
                count, value = parse_form4(edgar.form4_xml(f["url"]))
                con.execute(
                    "INSERT INTO insider_filings VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                    [f["accession"], ticker, f["filing_date"], count, value, now],
                )
                known.add(f["accession"])
                added += 1
        except Exception as e:  # one company's bad filing shouldn't stop the rest
            log.warning("insider filings for %s: %s", ticker, e)
    log.info("insiders: %d new Form 4 filings cached for %d tickers", added, len(tickers))
    return added


def insider_buy_flags(con: duckdb.DuckDBPyConnection, tickers: list[str], friday: date,
                      lookback_days: int) -> dict[str, bool]:
    """True where a purchase was filed in (friday - lookback, friday]."""
    rows = con.execute(
        """SELECT ticker, bool_or(purchase_count > 0) FROM insider_filings
           WHERE ticker IN (SELECT unnest(?)) AND filing_date > ? AND filing_date <= ?
           GROUP BY ticker""",
        [list(tickers), friday - timedelta(days=lookback_days), friday],
    ).fetchall()
    found = dict(rows)
    return {t: bool(found.get(t, False)) for t in tickers}

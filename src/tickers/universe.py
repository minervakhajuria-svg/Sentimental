"""Ticker universe: which US-listed common stocks we track.

Listings come from Nasdaq Trader's public symbol directory, which covers every
Nasdaq- and NYSE-listed security and flags ETFs and test issues. Market cap and
liquidity come from a MarketDataProvider (yfinance for now).

All listings are stored with their market data, and floors are applied when the
universe is *loaded*. That way, changing a floor in config doesn't require a
slow rebuild.
"""

from __future__ import annotations

import logging
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

import duckdb

from market.prices import MarketDataProvider

log = logging.getLogger(__name__)

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# otherlisted.txt exchange codes. Arca (P), Cboe (Z) and IEX (V) are mostly
# ETFs and secondary listings, so they're out of scope.
OTHER_EXCHANGES = {"N": "NYSE", "A": "NYSE American"}

# Securities that aren't common equity. LP "Common Units" (e.g. ET) are
# equity, so units are only excluded when they're SPAC-style units.
_NON_EQUITY = re.compile(
    r"warrant|\brights?\b|preferred|\bnotes?\b|debenture|subordinated|\d+(\.\d+)?%",
    re.IGNORECASE,
)
_SPAC_UNITS = re.compile(r"\bunits?\b", re.IGNORECASE)
_LP_UNITS = re.compile(r"common units|limited partner", re.IGNORECASE)

_NAME_SUFFIX = re.compile(
    r"\s*-?\s*(?:New\s+)?(?:Class [A-Z] )?(?:Common Stock|Capital Stock|Ordinary Shares?|"
    r"Common Shares?|Common Units.*|American Depositary Shares.*)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Listing:
    ticker: str
    company_name: str
    exchange: str


def clean_name(raw: str) -> str:
    name = raw.split(" - ")[0] if " - " in raw else raw
    return _NAME_SUFFIX.sub("", name).strip()


def _is_equity(symbol: str, name: str) -> bool:
    if "$" in symbol:  # otherlisted marks preferred series with $ (e.g. ABR$D)
        return False
    if _NON_EQUITY.search(name):
        return False
    if _SPAC_UNITS.search(name) and not _LP_UNITS.search(name):
        return False
    return True


def _rows(text: str) -> list[dict]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = lines[0].split("|")
    return [
        dict(zip(header, ln.split("|")))
        for ln in lines[1:]
        if not ln.startswith("File Creation Time")
    ]


def parse_nasdaq_listed(text: str) -> list[Listing]:
    out = []
    for r in _rows(text):
        if r["Test Issue"] == "Y" or r["ETF"] == "Y" or r.get("NextShares") == "Y":
            continue
        if _is_equity(r["Symbol"], r["Security Name"]):
            out.append(Listing(r["Symbol"], clean_name(r["Security Name"]), "NASDAQ"))
    return out


def parse_other_listed(text: str) -> list[Listing]:
    out = []
    for r in _rows(text):
        exchange = OTHER_EXCHANGES.get(r["Exchange"])
        if exchange is None or r["Test Issue"] == "Y" or r["ETF"] == "Y":
            continue
        # ACT symbols use a dot for share classes (BRK.B), our canonical form.
        if _is_equity(r["ACT Symbol"], r["Security Name"]):
            out.append(Listing(r["ACT Symbol"], clean_name(r["Security Name"]), exchange))
    return out


def _http_get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "sentiment-analyser/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_listings(get=_http_get) -> list[Listing]:
    listings = parse_nasdaq_listed(get(NASDAQ_LISTED_URL)) + parse_other_listed(get(OTHER_LISTED_URL))
    # A symbol can't legitimately appear on both lists; keep the first if it does.
    seen: set[str] = set()
    return [l for l in listings if not (l.ticker in seen or seen.add(l.ticker))]


def build_universe(
    listings: list[Listing],
    provider: MarketDataProvider,
    cfg: dict,
    now: datetime | None = None,
) -> list[dict]:
    """Attach liquidity and market cap to each listing.

    Liquidity comes first because it's a cheap bulk download. Market cap costs
    one request per ticker, so it's only fetched for tickers that pass the
    dollar-volume floor; the rest could never be eligible anyway.
    """
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    tickers = [l.ticker for l in listings]
    dollar_vol = provider.avg_dollar_volume(tickers, days=30)
    liquid = [t for t in tickers if (dollar_vol.get(t) or 0) >= cfg["min_avg_dollar_volume"]]
    log.info("%d listings, %d with dollar-volume data, %d above liquidity floor",
             len(tickers), len(dollar_vol), len(liquid))
    caps = provider.market_caps(liquid)
    return [
        {
            "ticker": l.ticker,
            "company_name": l.company_name,
            "exchange": l.exchange,
            "market_cap": caps.get(l.ticker),
            "avg_dollar_volume_30d": dollar_vol.get(l.ticker),
            "updated_at": now,
        }
        for l in listings
    ]


def save_universe(con: duckdb.DuckDBPyConnection, rows: list[dict], min_coverage: float) -> None:
    """Replace the universe atomically, refusing obviously broken refreshes.

    If the market-data source is down or rate-limiting, most rows come back
    without data. Overwriting a good universe with that would silently empty
    the weekly rankings, so fail loudly instead.
    """
    if not rows:
        raise ValueError("refusing to save an empty universe")
    covered = sum(1 for r in rows if r["avg_dollar_volume_30d"] is not None) / len(rows)
    if covered < min_coverage:
        raise ValueError(
            f"only {covered:.0%} of listings have market data (need {min_coverage:.0%}); "
            "keeping the existing universe"
        )
    cols = ["ticker", "company_name", "exchange", "market_cap", "avg_dollar_volume_30d", "updated_at"]
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute("DELETE FROM ticker_universe")
        con.executemany(
            f"INSERT INTO ticker_universe ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            [[r[c] for c in cols] for r in rows],
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def load_universe(con: duckdb.DuckDBPyConnection, cfg: dict) -> dict[str, str]:
    """Tickers passing the floors, mapped to company name.

    Rows missing market data are excluded: we can't show they pass the floors.
    """
    rows = con.execute(
        """
        SELECT ticker, company_name FROM ticker_universe
        WHERE exchange IN (SELECT unnest(?))
          AND market_cap >= ?
          AND avg_dollar_volume_30d >= ?
        """,
        [cfg["exchanges"], cfg["min_market_cap"], cfg["min_avg_dollar_volume"]],
    ).fetchall()
    return dict(rows)

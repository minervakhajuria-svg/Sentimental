from datetime import datetime
from pathlib import Path

import pytest

from market.prices import MarketDataProvider, to_yahoo
from storage import db
from tickers.universe import (
    Listing, build_universe, clean_name, fetch_listings, load_universe,
    parse_nasdaq_listed, parse_other_listed, save_universe,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
NASDAQ = (FIXTURES / "nasdaqlisted_sample.txt").read_text(encoding="utf-8")
OTHER = (FIXTURES / "otherlisted_sample.txt").read_text(encoding="utf-8")

CFG = {
    "exchanges": ["NASDAQ", "NYSE", "NYSE American"],
    "min_market_cap": 300e6,
    "min_avg_dollar_volume": 5e6,
}


class FakeProvider(MarketDataProvider):
    def __init__(self, dollar_vol, caps):
        self.dollar_vol, self.caps = dollar_vol, caps
        self.cap_requests: list[str] = []

    def avg_dollar_volume(self, tickers, days=30):
        return {t: v for t, v in self.dollar_vol.items() if t in tickers}

    def market_caps(self, tickers):
        self.cap_requests = list(tickers)
        return {t: v for t, v in self.caps.items() if t in tickers}


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "u.duckdb")
    yield c
    c.close()


def test_nasdaq_parsing_keeps_only_common_equity():
    tickers = {l.ticker for l in parse_nasdaq_listed(NASDAQ)}
    assert {"NVDA", "AAPL", "TSLA", "ARM", "GOOG", "GOOGL"} <= tickers
    assert "AAAP" not in tickers   # ETF
    assert "AACIU" not in tickers  # SPAC units
    assert "AACIW" not in tickers  # warrant
    assert "ZXZZT" not in tickers  # test issue


def test_other_listed_parsing():
    listings = {l.ticker: l for l in parse_other_listed(OTHER)}
    assert {"A", "BRK.B", "BF.B", "ALL", "ET", "F", "T"} <= set(listings)
    assert "ABR$D" not in listings  # preferred
    assert "AAC.W" not in listings  # warrant
    assert "SPY" not in listings    # ETF on Arca
    assert listings["ET"].company_name == "Energy Transfer LP"  # LP units are equity
    assert listings["BRK.B"].exchange == "NYSE"


def test_fetch_listings_combines_both_files():
    urls = []
    def fake_get(url):
        urls.append(url)
        return NASDAQ if "nasdaqlisted" in url else OTHER
    tickers = [l.ticker for l in fetch_listings(get=fake_get)]
    assert len(urls) == 2
    assert "NVDA" in tickers and "BRK.B" in tickers
    assert len(tickers) == len(set(tickers))


@pytest.mark.parametrize("raw,clean", [
    ("NVIDIA Corporation - Common Stock", "NVIDIA Corporation"),
    ("Coinbase Global, Inc. - Class A Common Stock", "Coinbase Global, Inc."),
    ("Agilent Technologies, Inc. Common Stock", "Agilent Technologies, Inc."),
    ("Berkshire Hathaway Inc. New Common Stock", "Berkshire Hathaway Inc."),
    ("Arm Holdings plc - American Depositary Shares", "Arm Holdings plc"),
    ("AT&T Inc.", "AT&T Inc."),
])
def test_clean_name(raw, clean):
    assert clean_name(raw) == clean


def test_to_yahoo_symbol():
    assert to_yahoo("BRK.B") == "BRK-B"
    assert to_yahoo("NVDA") == "NVDA"


def test_build_only_fetches_caps_for_liquid_tickers():
    listings = [Listing("BIG", "Big Co", "NYSE"), Listing("TINY", "Tiny Co", "NASDAQ")]
    provider = FakeProvider({"BIG": 50e6, "TINY": 10e3}, {"BIG": 10e9, "TINY": 50e6})
    rows = {r["ticker"]: r for r in build_universe(listings, provider, CFG, now=datetime(2026, 10, 4))}
    assert provider.cap_requests == ["BIG"]
    assert rows["BIG"]["market_cap"] == 10e9
    assert rows["TINY"]["market_cap"] is None
    assert rows["TINY"]["avg_dollar_volume_30d"] == 10e3  # still stored


def _row(t, cap, vol, exchange="NYSE"):
    return {"ticker": t, "company_name": t, "exchange": exchange, "market_cap": cap,
            "avg_dollar_volume_30d": vol, "updated_at": datetime(2026, 10, 4)}


def test_load_applies_floors_and_exchanges(con):
    save_universe(con, [
        _row("OK", 1e9, 10e6),
        _row("SMALLCAP", 100e6, 10e6),
        _row("ILLIQUID", 1e9, 1e6),
        _row("NODATA", None, None),
        _row("OTHEREX", 1e9, 10e6, exchange="Cboe"),
    ], min_coverage=0.5)
    assert load_universe(con, CFG) == {"OK": "OK"}


def test_floors_are_read_from_config(con):
    save_universe(con, [_row("SMALLCAP", 100e6, 10e6)], min_coverage=0.5)
    assert load_universe(con, {**CFG, "min_market_cap": 50e6}) == {"SMALLCAP": "SMALLCAP"}


def test_save_replaces_previous_universe(con):
    save_universe(con, [_row("OLD", 1e9, 10e6)], min_coverage=0.5)
    save_universe(con, [_row("NEW", 1e9, 10e6)], min_coverage=0.5)
    assert set(load_universe(con, CFG)) == {"NEW"}


def test_save_refuses_when_market_data_missing(con):
    save_universe(con, [_row("GOOD", 1e9, 10e6)], min_coverage=0.5)
    broken = [_row("A", None, None), _row("B", None, None), _row("C", 1e9, 10e6)]
    with pytest.raises(ValueError, match="market data"):
        save_universe(con, broken, min_coverage=0.5)
    assert set(load_universe(con, CFG)) == {"GOOD"}  # untouched


def test_save_refuses_empty(con):
    with pytest.raises(ValueError):
        save_universe(con, [], min_coverage=0.5)

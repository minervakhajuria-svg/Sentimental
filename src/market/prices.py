"""Market data behind an interface, so yfinance can be swapped for a paid feed.

Used for the universe filters (average dollar volume, market cap) and for the
daily OHLCV behind the context columns.

Market data is never an input to the sentiment ranking (see CLAUDE.md §10).
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

log = logging.getLogger(__name__)


class MarketDataProvider(ABC):
    @abstractmethod
    def avg_dollar_volume(self, tickers: list[str], days: int = 30) -> dict[str, float]:
        """Mean daily close*volume over the last `days` sessions. Missing tickers are omitted."""

    @abstractmethod
    def market_caps(self, tickers: list[str]) -> dict[str, float]:
        """Current market cap in USD. Missing tickers are omitted."""

    def daily_bars(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """Split- and dividend-adjusted daily bars for sessions in [start, end).

        Columns: ticker, date, open, high, low, close, volume. Missing tickers
        are omitted.
        """
        raise NotImplementedError


def to_yahoo(ticker: str) -> str:
    # Yahoo writes share classes with a dash: BRK.B -> BRK-B.
    return ticker.replace(".", "-")


class YFinanceProvider(MarketDataProvider):
    """Free, unofficial Yahoo data. Fine for a personal screen; expect gaps.

    Requests are chunked and paced to stay polite and avoid Yahoo's rate limiter.
    """

    def __init__(self, chunk_size: int = 200, pause_seconds: float = 1.0):
        self.chunk_size = chunk_size
        self.pause_seconds = pause_seconds

    def avg_dollar_volume(self, tickers: list[str], days: int = 30) -> dict[str, float]:
        import yfinance as yf  # imported lazily: slow import, and tests never need it

        out: dict[str, float] = {}
        for i in range(0, len(tickers), self.chunk_size):
            chunk = tickers[i : i + self.chunk_size]
            yahoo = {to_yahoo(t): t for t in chunk}
            try:
                df = yf.download(list(yahoo), period="3mo", progress=False,
                                 auto_adjust=False, threads=True, group_by="column",
                                 multi_level_index=True)
            except Exception:
                log.exception("price download failed for chunk starting %s", chunk[0])
                continue
            if df.empty:
                continue
            dollar = (df["Close"] * df["Volume"]).tail(days).mean(skipna=True)
            for ysym, value in dollar.items():
                if value == value and value > 0:  # drop NaN
                    out[yahoo[ysym]] = float(value)
            log.info("dollar volume: %d/%d tickers done", min(i + self.chunk_size, len(tickers)),
                     len(tickers))
            time.sleep(self.pause_seconds)
        return out

    def market_caps(self, tickers: list[str]) -> dict[str, float]:
        import yfinance as yf

        out: dict[str, float] = {}
        for n, t in enumerate(tickers, 1):
            try:
                cap = yf.Ticker(to_yahoo(t)).fast_info["marketCap"]
                if cap:
                    out[t] = float(cap)
            except Exception as e:
                log.debug("no market cap for %s: %s", t, type(e).__name__)
            if n % 100 == 0:
                log.info("market caps: %d/%d tickers done", n, len(tickers))
                time.sleep(self.pause_seconds)
        return out

    def daily_bars(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        import yfinance as yf

        frames = []
        for i in range(0, len(tickers), self.chunk_size):
            chunk = tickers[i : i + self.chunk_size]
            yahoo = {to_yahoo(t): t for t in chunk}
            try:
                # auto_adjust: returns across splits and dividends stay correct.
                df = yf.download(list(yahoo), start=start, end=end, progress=False,
                                 auto_adjust=True, threads=True, group_by="column",
                                 multi_level_index=True)
            except Exception:
                log.exception("bar download failed for chunk starting %s", chunk[0])
                continue
            if df.empty:
                continue
            long = df.stack(level="Ticker", future_stack=True).reset_index()
            long.columns = [str(c).lower() for c in long.columns]
            long = long.rename(columns={"ticker": "ysym"}).dropna(subset=["close"])
            long["ticker"] = long["ysym"].map(yahoo)
            long["date"] = pd.to_datetime(long["date"]).dt.date
            frames.append(long[["ticker", "date", "open", "high", "low", "close", "volume"]])
            time.sleep(self.pause_seconds)
        if not frames:
            return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])
        return pd.concat(frames, ignore_index=True)

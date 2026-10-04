"""StockTwits symbol streams via the official API.

StockTwits isn't accepting new developer registrations at the moment (checked
2026-10-04), and its terms prohibit scraping, so this collector only runs with
an access token from an approved app (STOCKTWITS_ACCESS_TOKEN). It's off by
default in config.yaml.

Why bother: authors can tag a message Bullish or Bearish. Those tags are stored
as `stocktwits_tag` scores, a free labelled set for checking how well FinBERT
reads retail chatter.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Callable, Iterable

from .base import Collector, Post, content_hash, hash_author
from .http import get_json

log = logging.getLogger(__name__)

STREAM_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
TAGS = {"Bullish": 1, "Bearish": -1}


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).replace(tzinfo=None)


class StockTwitsCollector(Collector):
    source = "stocktwits"

    def __init__(
        self,
        access_token: str,
        tickers: Callable[[], list[str]],
        max_pages: int = 5,
        pause_seconds: float = 2.0,
        get: Callable | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    ):
        self._token = access_token
        self._tickers = tickers
        self.max_pages = max_pages
        self.pause_seconds = pause_seconds
        # Looked up at construction (not as a default arg) so tests can swap it.
        self._get = get if get is not None else get_json
        self._sleep = sleep
        self._now = now
        self.communities: list[str] = []
        self.failed_communities: list[str] = []

    @classmethod
    def from_config(cls, cfg: dict, tickers: Callable[[], list[str]]) -> "StockTwitsCollector":
        token = os.environ.get("STOCKTWITS_ACCESS_TOKEN")
        if not token:
            raise RuntimeError("Missing STOCKTWITS_ACCESS_TOKEN in .env (needs an approved StockTwits app)")
        return cls(token, tickers, max_pages=cfg.get("max_pages", 5),
                   pause_seconds=cfg.get("pause_seconds", 2.0))

    def fetch(self, since: datetime) -> Iterable[Post]:
        tickers = self._tickers()
        self.communities, self.failed_communities = list(tickers), []
        out: dict[str, Post] = {}
        for ticker in tickers:
            try:
                for post in self._symbol_stream(ticker, since):
                    out.setdefault(post.id, post)
            except Exception as e:
                log.error("stocktwits %s: %s", ticker, e)
                self.failed_communities.append(ticker)
        log.info("stocktwits: %d messages for %d tickers", len(out), len(tickers))
        return list(out.values())

    def _symbol_stream(self, ticker: str, since: datetime) -> Iterable[Post]:
        max_id = None
        for _ in range(self.max_pages):
            params = {"max": max_id} if max_id else None
            data = self._get(STREAM_URL.format(symbol=ticker), params=params,
                             headers={"Authorization": f"OAuth {self._token}"}, sleep=self._sleep)
            messages = data.get("messages") or []
            if not messages:
                return
            for m in messages:
                post = self._to_post(m, ticker)
                if post is None:
                    continue
                if post.created_at < since:
                    return  # newest first: everything after this is older
                yield post
            max_id = messages[-1]["id"] - 1
            self._sleep(self.pause_seconds)

    def _to_post(self, m: dict, ticker: str) -> Post | None:
        try:
            user = m.get("user") or {}
            created = _parse_ts(m["created_at"])
            joined = user.get("join_date")
            age = (self._now().date() - datetime.fromisoformat(joined).date()).days if joined else None
            symbols = {s["symbol"].upper() for s in m.get("symbols") or [] if s.get("symbol")}
            tag = ((m.get("entities") or {}).get("sentiment") or {}).get("basic")
            body = m.get("body") or ""
            likes = (m.get("likes") or {}).get("total", 0) or 0
            replies = (m.get("conversation") or {}).get("replies", 0) or 0
            return Post(
                id=f"stocktwits:{m['id']}",
                source=self.source,
                community="stocktwits",
                author_id=hash_author(f"stocktwits:{user.get('username')}") if user.get("username") else None,
                author_age_days=age,
                author_karma=user.get("followers"),  # closest StockTwits analogue to karma
                created_at=created,
                collected_at=self._now(),
                title=None,
                body=body,
                url=f"https://stocktwits.com/message/{m['id']}",
                engagement=int(likes) + int(replies),
                content_hash=content_hash(None, body),
                source_tickers=tuple(sorted(symbols | {ticker})),
                source_sentiment=TAGS.get(tag),
            )
        except (KeyError, TypeError, ValueError) as e:
            log.warning("stocktwits %s: skipping malformed message (%s)", ticker, type(e).__name__)
            return None

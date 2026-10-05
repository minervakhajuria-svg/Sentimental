"""Company news from Finnhub's free API (personal, non-commercial use).

One request per ticker per run, paced to stay under the free tier's 60 calls/minute.
Finnhub says which ticker each article is about (its `related` field), so
those become source-supplied ticker tags.

News has no upvotes, so `engagement` is NULL; aggregation gives the source a
configured engagement floor instead of letting the weight collapse to zero.
Each outlet is treated as one "author" so breadth counts distinct outlets.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Iterable

from .base import Collector, Post, content_hash, hash_author
from .http import get_json

log = logging.getLogger(__name__)

FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"


def _utc(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)


class FinnhubNewsCollector(Collector):
    source = "news"

    def __init__(
        self,
        api_key: str,
        tickers: Callable[[], list[str]],
        requests_per_minute: float = 55,
        get: Callable | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        clock: Callable[[], float] = time.monotonic,
    ):
        self._key = api_key
        self._tickers = tickers  # called at fetch time, so it sees the latest DB state
        # Pace request *starts* to the rate limit. A fixed pause after each call
        # would add the request's own latency on top and roughly halve throughput.
        self.interval = 60.0 / requests_per_minute
        self._clock = clock
        # Looked up at construction (not as a default arg) so tests can swap it.
        self._get = get if get is not None else get_json
        self._sleep = sleep
        self._now = now
        self.failed_communities: list[str] = []
        self.communities: list[str] = []
        self.failure_tolerance = 0.0  # share of tickers allowed to fail before the run is "partial"

    @classmethod
    def from_config(cls, cfg: dict, tickers: Callable[[], list[str]]) -> "FinnhubNewsCollector":
        key = os.environ.get("FINNHUB_API_KEY")
        if not key:
            raise RuntimeError("Missing FINNHUB_API_KEY in .env (free key at finnhub.io)")
        c = cls(key, tickers, requests_per_minute=cfg.get("requests_per_minute", 55))
        c.failure_tolerance = cfg.get("max_failure_share", 0.0)
        return c

    def fetch(self, since: datetime) -> Iterable[Post]:
        tickers = self._tickers()
        # Failures are tracked per ticker, reusing the "communities" bookkeeping
        # the daily job already reports on.
        self.communities, self.failed_communities = list(tickers), []
        start: date = since.date()
        end: date = self._now().date()
        seen: dict[str, Post] = {}
        last_start = None
        for ticker in tickers:
            if last_start is not None:
                wait = self.interval - (self._clock() - last_start)
                if wait > 0:
                    self._sleep(wait)
            last_start = self._clock()
            try:
                items = self._get(
                    FINNHUB_NEWS_URL,
                    params={"symbol": ticker, "from": start.isoformat(), "to": end.isoformat()},
                    headers={"X-Finnhub-Token": self._key},
                    sleep=self._sleep,
                )
            except Exception as e:
                log.error("news %s: %s", ticker, e)
                self.failed_communities.append(ticker)
                continue
            for item in items or []:
                post = self._to_post(item, ticker, since)
                if post is None:
                    continue
                if post.id in seen:  # same article under several tickers: merge tags
                    old = seen[post.id]
                    merged = tuple(dict.fromkeys(old.source_tickers + post.source_tickers))
                    post = Post(**{**old.as_row(), "source_tickers": merged})
                seen[post.id] = post
        log.info("news: %d articles for %d tickers", len(seen), len(tickers))
        return list(seen.values())

    def _to_post(self, item: dict, ticker: str, since: datetime) -> Post | None:
        try:
            created = _utc(item["datetime"])
            if created < since or not item.get("headline"):
                return None
            related = {t.strip().upper() for t in (item.get("related") or "").split(",") if t.strip()}
            tickers = tuple(sorted(related | {ticker}))
            outlet = (item.get("source") or "unknown").strip()
            title, body = item["headline"].strip(), (item.get("summary") or "").strip() or None
            return Post(
                id=f"news:finnhub:{item['id']}",
                source=self.source,
                community=outlet,
                author_id=hash_author(f"news:{outlet}"),
                author_age_days=None,
                author_karma=None,
                created_at=created,
                collected_at=self._now(),
                title=title,
                body=body,
                url=item.get("url"),
                engagement=None,
                content_hash=content_hash(title, body),
                source_tickers=tickers,
            )
        except (KeyError, TypeError, ValueError) as e:
            log.warning("news %s: skipping malformed item (%s)", ticker, type(e).__name__)
            return None

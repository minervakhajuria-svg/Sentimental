"""Common collector interface and the normalised Post record.

Every source (Reddit now; StockTwits, news, X later) yields `Post` objects
shaped like the `posts` table, so storage and everything downstream never
need to know which source an item came from.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable


@dataclass(frozen=True)
class Post:
    id: str  # "{source}:{native_id}" so ids never collide across sources
    source: str
    community: str | None
    author_id: str | None
    author_age_days: int | None
    author_karma: int | None
    created_at: datetime  # UTC, naive (DuckDB TIMESTAMP has no zone)
    collected_at: datetime  # UTC, naive
    title: str | None
    body: str | None
    url: str | None
    engagement: int | None
    content_hash: str | None

    def as_row(self) -> dict:
        return asdict(self)


class Collector(ABC):
    """A source of posts. Implementations must be safe to re-run over the same window."""

    source: str

    @abstractmethod
    def fetch(self, since: datetime) -> Iterable[Post]:
        """Yield posts created at or after `since` (UTC).

        Implementations should log and skip failures for one community/feed
        rather than abort the whole fetch, so a single bad subreddit can't
        cost a day of history.
        """


_WS = re.compile(r"\s+")


def content_hash(title: str | None, body: str | None) -> str:
    """Hash of case- and whitespace-normalised text.

    Used to spot exact reposts across communities. Normalising first means
    trivial differences (trailing spaces, capitalisation) don't defeat it;
    fuzzy near-duplicate matching is a later phase.
    """
    text = f"{title or ''}\n{body or ''}".lower()
    text = _WS.sub(" ", text).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_author(username: str | None) -> str | None:
    """One-way hash of a username.

    We only need to count distinct authors and cap per-author influence, so
    there's no reason to store real usernames.
    """
    if not username:
        return None
    return hashlib.sha256(username.lower().encode("utf-8")).hexdigest()[:16]

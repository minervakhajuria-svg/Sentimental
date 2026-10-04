"""Reddit collector built on PRAW (read-only, script app).

Walks each configured subreddit's "new" listing back to `since`. PRAW
already throttles to Reddit's published rate limit, so the only extra
handling here is retry with backoff on transient errors.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Callable, Iterable

import praw
import prawcore

from .base import Collector, Post, content_hash, hash_author

log = logging.getLogger(__name__)

REDDIT_BASE = "https://www.reddit.com"

# Errors worth retrying. Auth/permission errors (403, 404, bad credentials) are
# not: retrying won't fix them, so they fall through to "log and skip".
TRANSIENT_ERRORS = (
    prawcore.exceptions.ServerError,
    prawcore.exceptions.RequestException,
    prawcore.exceptions.TooManyRequests,
)


def utc_naive(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class RedditCollector(Collector):
    source = "reddit"

    def __init__(
        self,
        reddit,
        communities: list[str],
        max_posts_per_community: int = 1000,
        fetch_author_details: bool = True,
        max_retries: int = 3,
        backoff_seconds: float = 5,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = utc_now,
    ):
        # `reddit` is a praw.Reddit, or any object with the same `subreddit()`
        # shape; tests pass a fake built from fixtures so no live calls happen.
        self.reddit = reddit
        self.communities = communities
        self.max_posts = max_posts_per_community
        self.fetch_author_details = fetch_author_details
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._now = now
        # Per-run cache: prolific authors post many times a day, and each
        # profile lookup is a separate API call.
        self._author_cache: dict[str, tuple[int | None, int | None]] = {}
        self.failed_communities: list[str] = []

    @classmethod
    def from_config(cls, cfg: dict) -> "RedditCollector":
        """Build from config.yaml's `reddit` section and credentials in the environment."""
        missing = [
            k
            for k in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT")
            if not os.environ.get(k)
        ]
        if missing:
            raise RuntimeError(f"Missing Reddit credentials in .env: {', '.join(missing)}")
        reddit = praw.Reddit(
            client_id=os.environ["REDDIT_CLIENT_ID"],
            client_secret=os.environ["REDDIT_CLIENT_SECRET"],
            user_agent=os.environ["REDDIT_USER_AGENT"],
        )
        reddit.read_only = True
        return cls(
            reddit,
            communities=cfg["communities"],
            max_posts_per_community=cfg.get("max_posts_per_community", 1000),
            fetch_author_details=cfg.get("fetch_author_details", True),
            max_retries=cfg.get("max_retries", 3),
            backoff_seconds=cfg.get("backoff_seconds", 5),
        )

    def fetch(self, since: datetime) -> Iterable[Post]:
        self.failed_communities = []
        for community in self.communities:
            try:
                submissions = self._with_retries(
                    lambda: self._list_since(community, since), community
                )
            except Exception:
                log.exception("r/%s: giving up, skipping this community", community)
                self.failed_communities.append(community)
                continue
            log.info("r/%s: %d posts since %s", community, len(submissions), since)
            for sub in submissions:
                try:
                    yield self._to_post(sub, community)
                except Exception:
                    log.exception("r/%s: could not normalise post %s", community,
                                  getattr(sub, "id", "?"))

    def _list_since(self, community: str, since: datetime) -> list:
        """Materialise the listing so a mid-listing error retries cleanly."""
        out = []
        for sub in self.reddit.subreddit(community).new(limit=self.max_posts):
            if utc_naive(sub.created_utc) < since:
                break  # "new" is newest-first, so everything after is older
            out.append(sub)
        return out

    def _with_retries(self, fn, community: str):
        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except TRANSIENT_ERRORS as e:
                if attempt == self.max_retries:
                    raise
                wait = self.backoff_seconds * (2**attempt)
                log.warning("r/%s: %s (attempt %d), retrying in %ss",
                            community, type(e).__name__, attempt + 1, wait)
                self._sleep(wait)

    def _to_post(self, sub, community: str) -> Post:
        author = getattr(sub, "author", None)  # None when the account is deleted
        username = getattr(author, "name", None) if author is not None else None
        age_days, karma = self._author_details(author, username)
        title = sub.title
        body = sub.selftext or None
        return Post(
            id=f"reddit:{sub.id}",
            source=self.source,
            community=community,
            author_id=hash_author(username),
            author_age_days=age_days,
            author_karma=karma,
            created_at=utc_naive(sub.created_utc),
            collected_at=self._now(),
            title=title,
            body=body,
            url=REDDIT_BASE + sub.permalink,
            engagement=int(sub.score or 0) + int(sub.num_comments or 0),
            content_hash=content_hash(title, body),
        )

    def _author_details(self, author, username: str | None) -> tuple[int | None, int | None]:
        """Account age (days) and total karma, or (None, None) if unavailable.

        Suspended or shadow-banned accounts raise or lack these fields; that's
        a useful noise signal in itself, so we store NULL rather than fail.
        """
        if not self.fetch_author_details or author is None or not username:
            return None, None
        if username in self._author_cache:
            return self._author_cache[username]
        try:
            created = utc_naive(author.created_utc)
            age = (self._now() - created).days
            karma = int(getattr(author, "link_karma", 0) or 0) + int(
                getattr(author, "comment_karma", 0) or 0
            )
            result = (age, karma)
        except Exception as e:  # suspended/deleted accounts, transient errors
            log.debug("author details unavailable for a user: %s", type(e).__name__)
            result = (None, None)
        self._author_cache[username] = result
        return result

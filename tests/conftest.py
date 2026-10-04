"""Shared test helpers: a fake PRAW client fed from saved fixtures.

The fake mirrors only the bits of PRAW the collector touches
(`reddit.subreddit(name).new(limit=...)`, submission and Redditor attributes),
so tests never hit the network.
"""

from __future__ import annotations

import json
import socket
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


class FakeRedditor:
    def __init__(self, name: str, info: dict):
        self.name = name
        self._info = info

    def __getattr__(self, attr):
        # Mimic PRAW: suspended accounts blow up when profile fields are read.
        if self._info.get("suspended"):
            raise AttributeError(f"{self.name} is suspended")
        try:
            return self._info[attr]
        except KeyError:
            raise AttributeError(attr) from None


class FakeSubreddit:
    def __init__(self, items, error_queue):
        self._items = items
        self._errors = error_queue

    def new(self, limit=None):
        if self._errors:
            raise self._errors.pop(0)
        return iter(self._items[:limit])


class FakeReddit:
    def __init__(self, fixture: dict, errors: dict[str, list[Exception]] | None = None):
        self._authors = fixture["authors"]
        self._subs = fixture["subreddits"]
        self.errors = errors or {}

    def subreddit(self, name):
        items = [self._submission(d) for d in self._subs.get(name, [])]
        return FakeSubreddit(items, self.errors.setdefault(name, []))

    def _submission(self, d):
        author = FakeRedditor(d["author"], self._authors[d["author"]]) if d["author"] else None
        return SimpleNamespace(**{**d, "author": author})


@pytest.fixture
def reddit_fixture() -> dict:
    return json.loads((FIXTURES / "reddit_submissions.json").read_text(encoding="utf-8"))


@pytest.fixture
def fixed_now(reddit_fixture):
    now = datetime.fromisoformat(reddit_fixture["collected_at"])
    return lambda: now


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any test that tries to reach the internet (CLAUDE.md: never hit live APIs).

    Local connections stay allowed, since some tools talk to themselves over loopback.
    """
    real_connect = socket.socket.connect

    def guarded(sock, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise RuntimeError(f"test tried to open a network connection to {host!r}")
        return real_connect(sock, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded)

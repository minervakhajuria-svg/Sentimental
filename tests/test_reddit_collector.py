from datetime import datetime
from types import SimpleNamespace

import prawcore

from collectors.base import content_hash, hash_author
from collectors.reddit import RedditCollector
from conftest import FakeReddit

SINCE = datetime(2026, 10, 2, 6, 0)  # 36h before the fixture's collected_at


def make_collector(fixture, now, errors=None, **kw):
    sleeps = []
    c = RedditCollector(
        FakeReddit(fixture, errors),
        communities=["wallstreetbets", "stocks"],
        sleep=sleeps.append,
        now=now,
        **kw,
    )
    return c, sleeps


def http_error(cls, status):
    return cls(SimpleNamespace(status_code=status, headers={}, text=""))


def test_fetch_normalises_posts(reddit_fixture, fixed_now):
    c, _ = make_collector(reddit_fixture, fixed_now)
    posts = {p.id: p for p in c.fetch(SINCE)}

    p = posts["reddit:1aaa01"]
    assert p.source == "reddit"
    assert p.community == "wallstreetbets"
    assert p.created_at == datetime(2026, 10, 3, 15, 0)
    assert p.collected_at == datetime(2026, 10, 3, 18, 0)
    assert p.engagement == 420 + 88
    assert p.url == "https://www.reddit.com/r/wallstreetbets/comments/1aaa01/nvda_calls/"
    assert p.author_id == hash_author("ape_trader_01")
    assert p.author_id != "ape_trader_01"  # usernames are never stored
    assert p.author_karma == 10_000
    assert p.author_age_days == (datetime(2026, 10, 3, 18) - datetime(2021, 1, 27)).days
    assert p.content_hash == content_hash(p.title, p.body)


def test_since_cutoff_excludes_older_posts(reddit_fixture, fixed_now):
    c, _ = make_collector(reddit_fixture, fixed_now)
    ids = {p.id for p in c.fetch(SINCE)}
    assert "reddit:1aaa04" not in ids
    assert ids == {"reddit:1aaa01", "reddit:1aaa02", "reddit:1aaa03",
                   "reddit:1bbb01", "reddit:1bbb02"}


def test_deleted_and_suspended_authors_become_null(reddit_fixture, fixed_now):
    c, _ = make_collector(reddit_fixture, fixed_now)
    posts = {p.id: p for p in c.fetch(SINCE)}
    deleted = posts["reddit:1aaa03"]
    assert deleted.author_id is None and deleted.author_karma is None
    suspended = posts["reddit:1bbb02"]
    assert suspended.author_id is not None  # still countable as a distinct author
    assert suspended.author_age_days is None and suspended.author_karma is None


def test_link_post_with_empty_body_stores_null(reddit_fixture, fixed_now):
    c, _ = make_collector(reddit_fixture, fixed_now)
    posts = {p.id: p for p in c.fetch(SINCE)}
    assert posts["reddit:1aaa02"].body is None


def test_author_details_can_be_disabled(reddit_fixture, fixed_now):
    c, _ = make_collector(reddit_fixture, fixed_now, fetch_author_details=False)
    for p in c.fetch(SINCE):
        assert p.author_age_days is None and p.author_karma is None


def test_reposts_share_a_content_hash(reddit_fixture, fixed_now):
    c, _ = make_collector(reddit_fixture, fixed_now)
    posts = {p.id: p for p in c.fetch(SINCE)}
    # Same text with different spacing/case across two subreddits.
    assert posts["reddit:1aaa01"].content_hash == posts["reddit:1bbb02"].content_hash


def test_transient_error_is_retried_with_backoff(reddit_fixture, fixed_now):
    errors = {"wallstreetbets": [http_error(prawcore.exceptions.ServerError, 503),
                                 http_error(prawcore.exceptions.TooManyRequests, 429)]}
    c, sleeps = make_collector(reddit_fixture, fixed_now, errors=errors, backoff_seconds=2)
    ids = {p.id for p in c.fetch(SINCE)}
    assert "reddit:1aaa01" in ids
    assert sleeps == [2, 4]
    assert c.failed_communities == []


def test_persistent_failure_skips_community_and_continues(reddit_fixture, fixed_now):
    errors = {"wallstreetbets": [http_error(prawcore.exceptions.ServerError, 500)] * 10}
    c, _ = make_collector(reddit_fixture, fixed_now, errors=errors, max_retries=2)
    ids = {p.id for p in c.fetch(SINCE)}
    assert ids == {"reddit:1bbb01", "reddit:1bbb02"}
    assert c.failed_communities == ["wallstreetbets"]


def test_non_transient_error_is_not_retried(reddit_fixture, fixed_now):
    errors = {"stocks": [http_error(prawcore.exceptions.Forbidden, 403)]}
    c, sleeps = make_collector(reddit_fixture, fixed_now, errors=errors)
    list(c.fetch(SINCE))
    assert sleeps == []
    assert c.failed_communities == ["stocks"]


def test_from_config_requires_credentials(monkeypatch):
    for k in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"):
        monkeypatch.delenv(k, raising=False)
    try:
        RedditCollector.from_config({"communities": ["stocks"]})
    except RuntimeError as e:
        assert "REDDIT_CLIENT_ID" in str(e)
    else:
        raise AssertionError("expected RuntimeError")

from dataclasses import replace
from datetime import datetime

import pytest

from collectors.base import Post
from storage import db


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "test.duckdb")
    yield c
    c.close()


def make_post(native_id="abc", **kw):
    base = dict(
        id=f"reddit:{native_id}", source="reddit", community="stocks",
        author_id="h1", author_age_days=100, author_karma=500,
        created_at=datetime(2026, 10, 3, 12), collected_at=datetime(2026, 10, 3, 18),
        title="title", body="body", url="https://example/x", engagement=10,
        content_hash="hash",
    )
    base.update(kw)
    return Post(**base)


def test_schema_creates_all_spec_tables(con):
    tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    assert {"posts", "post_tickers", "post_scores", "prices_daily",
            "ticker_universe", "weekly_signals"} <= tables


def test_connect_is_safe_to_repeat(tmp_path):
    path = tmp_path / "x.duckdb"
    c = db.connect(path)
    db.upsert_posts(c, [make_post()])
    c.close()
    c = db.connect(path)  # re-running schema must not drop data
    assert c.execute("SELECT count(*) FROM posts").fetchone()[0] == 1
    c.close()


def test_upsert_is_idempotent(con):
    posts = [make_post("a"), make_post("b")]
    assert db.upsert_posts(con, posts) == (2, 0)
    assert db.upsert_posts(con, posts) == (0, 2)
    assert con.execute("SELECT count(*) FROM posts").fetchone()[0] == 2


def test_upsert_refreshes_engagement_but_keeps_original_text(con):
    original = make_post("a", engagement=10, body="real text")
    db.upsert_posts(con, [original])
    later = replace(original, engagement=99, body="[removed]",
                    collected_at=datetime(2026, 10, 4, 18))
    db.upsert_posts(con, [later])
    body, engagement, collected = con.execute(
        "SELECT body, engagement, collected_at FROM posts WHERE id = 'reddit:a'"
    ).fetchone()
    assert body == "real text"
    assert engagement == 99
    assert collected == datetime(2026, 10, 3, 18)  # first-seen time kept


def test_upsert_does_not_null_out_known_author_details(con):
    db.upsert_posts(con, [make_post("a", author_karma=500)])
    db.upsert_posts(con, [make_post("a", author_karma=None, author_age_days=None)])
    assert con.execute("SELECT author_karma, author_age_days FROM posts").fetchone() == (500, 100)


def test_duplicate_ids_within_a_batch_are_collapsed(con):
    assert db.upsert_posts(con, [make_post("a", engagement=1), make_post("a", engagement=2)]) == (1, 0)
    assert con.execute("SELECT engagement FROM posts").fetchone()[0] == 2


def test_empty_batch_is_a_noop(con):
    assert db.upsert_posts(con, []) == (0, 0)


def test_failed_batch_rolls_back_entirely(con):
    db.upsert_posts(con, [make_post("a")])
    bad = make_post("b", created_at=None)  # violates NOT NULL
    with pytest.raises(Exception):
        db.upsert_posts(con, [make_post("c"), bad])
    ids = [r[0] for r in con.execute("SELECT id FROM posts ORDER BY id").fetchall()]
    assert ids == ["reddit:a"]

"""Noise filters and post weighting applied before aggregation (CLAUDE.md §9).

Order matters and is applied in signals.aggregate:
  1. drop_reposts         exact duplicates (same content hash), keep earliest
  2. drop_near_duplicates fuzzy reposts (SimHash), keep earliest
  3. drop_spam            configured spam patterns (Discord/Telegram pumps, "DM me")
  4. add_post_weight      log(1 + engagement) * author_weight * source_weight
  5. cap_per_author       at most K posts per author per ticker per week

All thresholds live in config.yaml under signals.noise.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np
import pandas as pd

_WORD = re.compile(r"[a-z0-9$']+")


def drop_reposts(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the earliest post for each content hash.

    The same text cross-posted to several subreddits is one opinion, not several.
    Rows without a hash are kept.
    """
    has_hash = df["content_hash"].notna()
    first = (
        df[has_hash].sort_values("created_at")
        .drop_duplicates("content_hash")["post_id"]
    )
    return df[~has_hash | df["post_id"].isin(set(first))]


def simhash(text: str, shingle: int = 3) -> int | None:
    """64-bit SimHash over word shingles; None if the text is too short to judge."""
    words = _WORD.findall(text.lower())
    if len(words) < shingle:
        return None
    v = [0] * 64
    for i in range(len(words) - shingle + 1):
        h = int.from_bytes(hashlib.blake2b(" ".join(words[i:i + shingle]).encode(),
                                           digest_size=8).digest(), "big")
        for bit in range(64):
            v[bit] += 1 if h >> bit & 1 else -1
    return sum(1 << bit for bit in range(64) if v[bit] > 0)


def near_duplicate_ids(posts: pd.DataFrame, max_hamming: int, min_words: int) -> set[str]:
    """post_ids that are near-copies of an earlier post (by SimHash distance).

    Banding: the 64-bit hash is split into (max_hamming + 1) bands, so any two
    hashes within `max_hamming` bits share at least one identical band. Only
    posts sharing a band are compared, which keeps this fast on large weeks.
    Short posts are skipped: "NVDA calls" vs "AMD calls" would look alike.
    """
    bands = max_hamming + 1
    width = 64 // bands
    mask = (1 << width) - 1
    buckets: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)
    dupes: set[str] = set()
    for row in posts.sort_values("created_at").itertuples():
        text = f"{row.title or ''} {row.body or ''}"
        if len(_WORD.findall(text.lower())) < min_words:
            continue
        h = simhash(text)
        if h is None:
            continue
        keys = [(b, (h >> (b * width)) & mask) for b in range(bands)]
        if any(bin(h ^ other).count("1") <= max_hamming
               for k in keys for _, other in buckets.get(k, ())):
            dupes.add(row.post_id)
            continue
        for k in keys:
            buckets[k].append((row.post_id, h))
    return dupes


def drop_near_duplicates(df: pd.DataFrame, max_hamming: int, min_words: int) -> pd.DataFrame:
    posts = df.drop_duplicates("post_id")[["post_id", "created_at", "title", "body"]]
    dupes = near_duplicate_ids(posts, max_hamming, min_words)
    return df[~df["post_id"].isin(dupes)]


def drop_spam(df: pd.DataFrame, patterns: list[str]) -> pd.DataFrame:
    if not patterns:
        return df
    rx = re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)
    text = df["title"].fillna("") + " " + df["body"].fillna("")
    return df[~text.map(lambda t: bool(rx.search(t)))]


def author_weights(df: pd.DataFrame, noise: dict) -> pd.Series:
    """Per-row author weight in [0, 1].

    New accounts, low-karma accounts and hyperactive accounts are down-weighted
    (weight 0 drops them). Thresholds are per source because "karma" means
    different things (Reddit karma, StockTwits followers). Missing data is
    not penalised: we only act on evidence.
    """
    w = pd.Series(1.0, index=df.index)
    for source, rules in noise.get("authors", {}).items():
        mine = df["source"] == source
        age, karma = df["author_age_days"], df["author_karma"]
        w[mine & age.notna() & (age < rules["min_age_days"])] *= rules["new_account_weight"]
        w[mine & karma.notna() & (karma < rules["min_karma"])] *= rules["low_karma_weight"]

    bots = noise.get("bots", {})
    if bots:
        watched = df["source"].isin(bots["sources"]) & df["author_id"].notna()
        per_author = (df[watched].drop_duplicates("post_id")
                      .groupby(["block", "author_id"]).size())
        busy = per_author[per_author > bots["max_posts_per_week"]].reset_index()[["block", "author_id"]]
        if not busy.empty:
            hit = df.merge(busy.assign(_busy=True), on=["block", "author_id"], how="left")["_busy"]
            w[hit.fillna(False).to_numpy(dtype=bool) & watched.to_numpy()] *= bots["weight"]
    return w


def add_post_weight(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """weight = log(1 + engagement) * author_weight * source_weight.

    Sources without engagement counts (news) get a floor from config, so their
    weight isn't zero just because nobody can upvote a headline.
    """
    noise = cfg.get("noise", {})
    floors = df["source"].map(noise.get("engagement_floor", {})).fillna(0)
    engagement = np.maximum(df["engagement"].astype(float).fillna(0).clip(lower=0), floors)
    source_weight = df["source"].map(cfg["source_weights"]).fillna(1.0)
    author_w = author_weights(df, noise)
    return df.assign(author_weight=author_w,
                     weight=np.log1p(engagement) * author_w * source_weight)


def cap_per_author(df: pd.DataFrame, k: int, period_col: str) -> pd.DataFrame:
    """Keep at most `k` posts per (period, ticker, author), highest weight first.

    Stops one prolific account from manufacturing attention. Posts by deleted
    accounts (no author_id) can't be grouped, so they aren't capped.
    """
    known = df["author_id"].notna()
    ranked = (
        df[known].sort_values("weight", ascending=False)
        .groupby([period_col, "ticker", "author_id"], sort=False)
        .head(k)
    )
    return pd.concat([ranked, df[~known]]).sort_index()


def apply_noise_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Steps 1-4 (the cap is applied by the caller, which knows the period)."""
    noise = cfg.get("noise", {})
    df = drop_reposts(df)
    nd = noise.get("near_duplicates")
    if nd:
        df = drop_near_duplicates(df, nd["max_hamming"], nd["min_words"])
    df = drop_spam(df, noise.get("spam_patterns", []))
    df = add_post_weight(df, cfg)
    # An author weight of 0 means "drop" (e.g. brand-new accounts, if so configured).
    # Zero *engagement* is different: the post still counts as a mention.
    return df[df["author_weight"] > 0]

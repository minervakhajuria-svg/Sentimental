"""Noise filters and post weighting applied before aggregation (CLAUDE.md §9).

Phase 3 covers exact-repost removal, the per-author cap and engagement
weighting. Bot detection and author-quality weights come in phase 6; until
then every author has weight 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def add_post_weight(df: pd.DataFrame, source_weights: dict[str, float]) -> pd.DataFrame:
    """weight = log(1 + engagement) * author_weight * source_weight."""
    author_weight = 1.0  # phase 6: account age, karma, bot score
    source_weight = df["source"].map(source_weights).fillna(1.0)
    engagement = df["engagement"].fillna(0).clip(lower=0)
    return df.assign(weight=np.log1p(engagement) * author_weight * source_weight)


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

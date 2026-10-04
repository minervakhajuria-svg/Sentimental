"""Pass-2 sentiment with a small Claude model (CLAUDE.md §8).

FinBERT was trained on financial news, so it misreads retail chatter:
"TSLA puts are free money" comes out neutral. This pass re-scores only the
posts where that matters most:
  * ambiguous FinBERT scores (|score| < threshold), and
  * each ticker's most-engaged posts per week (the ones that move the average).

Results are stored as their own model ("claude") in post_scores and never
overwrite FinBERT. Aggregation picks scores by a priority list in config.

Cost control: off by default; capped items per run; the Message Batches API
(half price, usually done within an hour) unless `use_batch` is false.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Callable

import duckdb

from scoring.context import ContextBuilder
from scoring.finbert import Score, Scorer
from storage import db
from tickers.extractor import load_aliases

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You rate the sentiment of a social-media or news post toward ONE specific stock ticker.

Read it the way an experienced retail trader would:
- Options slang: buying calls or "calls printing" is bullish; buying puts or "puts printing" is bearish.
- "To the moon", "tendies", "loading up", "buying the dip" are bullish. "Bagholding", "rug pull",
  "dead cat bounce", "it's over" are usually bearish.
- Sarcasm and irony are common. Judge what the author actually believes, not the literal words.
- Loss porn or gain porn on its own says little about the outlook; judge the author's view going forward.
- Only the stance toward the named ticker counts, even if other tickers are mentioned.
- Questions, news recaps with no opinion, and unclear posts are neutral.

Return label "bullish", "bearish" or "neutral", and score from -1.0 (very bearish) to 1.0 (very bullish),
near 0 for neutral."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
        "score": {"type": "number"},
    },
    "required": ["label", "score"],
    "additionalProperties": False,
}
LABELS = {"bullish": "pos", "bearish": "neg", "neutral": "neu"}


def parse_reply(text: str) -> Score | None:
    """Turn the model's JSON into a Score; None if it's unusable."""
    try:
        data = json.loads(text)
        label = LABELS[data["label"]]
        score = max(-1.0, min(1.0, float(data["score"])))
    except (ValueError, KeyError, TypeError):
        return None
    return Score(label, round(score, 4))


class ClaudeScorer(Scorer):
    """Scores (text, ticker) items with Claude. `client` is an anthropic.Anthropic
    (tests pass a fake with the same methods)."""

    def __init__(self, client, model: str, name: str = "claude", use_batch: bool = True,
                 max_tokens: int = 256, poll_seconds: float = 30, max_wait_minutes: float = 120,
                 sleep: Callable[[float], None] = time.sleep):
        self.client = client
        self.model = model
        self.name = name
        self.use_batch = use_batch
        self.max_tokens = max_tokens
        self.poll_seconds = poll_seconds
        self.max_wait_minutes = max_wait_minutes
        self._sleep = sleep

    @classmethod
    def from_config(cls, cfg: dict) -> "ClaudeScorer":
        import anthropic  # imported lazily: only needed when pass 2 is enabled

        # Credentials come from the environment (ANTHROPIC_API_KEY in .env).
        return cls(anthropic.Anthropic(), cfg["model"], use_batch=cfg.get("use_batch", True),
                   max_wait_minutes=cfg.get("max_wait_minutes", 120))

    def _params(self, text: str, ticker: str) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": f"Ticker: {ticker}\n\nPost:\n{text}"}],
            "output_config": {"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        }

    def score(self, texts: list[str]) -> list[Score]:  # Scorer interface; ticker-less use
        return [s or Score("neu", 0.0) for s in self.score_items([(t, "") for t in texts])]

    def score_items(self, items: list[tuple[str, str]]) -> list[Score | None]:
        """Score (text, ticker) pairs. None marks an item that failed (it'll be retried next run)."""
        if not items:
            return []
        return self._score_batch(items) if self.use_batch else self._score_sync(items)

    def _score_sync(self, items):
        out = []
        for text, ticker in items:
            try:
                msg = self.client.messages.create(**self._params(text, ticker))
                out.append(self._from_message(msg))
            except Exception as e:  # one bad item shouldn't sink the run
                log.warning("claude scoring failed for one item: %s", type(e).__name__)
                out.append(None)
        return out

    def _score_batch(self, items):
        # custom_id must be [a-zA-Z0-9_-]{1,64}; post ids contain ':' so use the index.
        requests = [{"custom_id": f"i{i}", "params": self._params(t, k)} for i, (t, k) in enumerate(items)]
        batch = self.client.messages.batches.create(requests=requests)
        log.info("claude batch %s submitted with %d items", batch.id, len(items))
        deadline = time.monotonic() + self.max_wait_minutes * 60
        while True:
            batch = self.client.messages.batches.retrieve(batch.id)
            if batch.processing_status == "ended":
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"claude batch {batch.id} not finished after "
                                   f"{self.max_wait_minutes} minutes; scores will be retried next run")
            self._sleep(self.poll_seconds)
        out: list[Score | None] = [None] * len(items)
        for r in self.client.messages.batches.results(batch.id):
            if r.result.type == "succeeded":  # results arrive in any order: key by custom_id
                out[int(r.custom_id[1:])] = self._from_message(r.result.message)
            else:
                log.warning("claude batch item %s: %s", r.custom_id, r.result.type)
        return out

    @staticmethod
    def _from_message(msg) -> Score | None:
        if getattr(msg, "stop_reason", None) == "refusal":
            return None
        text = next((b.text for b in msg.content if b.type == "text"), "")
        return parse_reply(text)


def select_pairs(con: duckdb.DuckDBPyConnection, cfg: dict, base_model: str,
                 model: str, now: datetime | None = None) -> list[tuple]:
    """(post_id, ticker, title, body, n_tickers) worth a second look, most important first.

    Only recent posts (the coming ranking can use them), only pairs FinBERT has
    already scored, and never pairs this model already scored.
    """
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    since = now - timedelta(days=cfg["lookback_days"])
    return con.execute(
        """
        WITH cand AS (
            SELECT pt.post_id, pt.ticker, p.title, p.body, p.engagement, b.score AS base,
                   count(*) OVER (PARTITION BY pt.post_id) AS n_tickers,
                   row_number() OVER (PARTITION BY pt.ticker, date_trunc('week', p.created_at)
                                      ORDER BY p.engagement DESC NULLS LAST) AS eng_rank
            FROM post_tickers pt
            JOIN posts p ON p.id = pt.post_id
            JOIN post_scores b ON b.post_id = pt.post_id AND b.ticker = pt.ticker AND b.model = ?
            WHERE p.created_at >= ?
        )
        -- Engagement rank is computed over all posts above, *then* already-scored
        -- pairs are removed; otherwise each run would promote the next post into
        -- the "top" slot and slowly re-score everything.
        SELECT post_id, ticker, title, body, n_tickers FROM cand
        WHERE NOT EXISTS (SELECT 1 FROM post_scores s WHERE s.post_id = cand.post_id
                          AND s.ticker = cand.ticker AND s.model = ?)
          AND (abs(base) < ? OR eng_rank <= ?)
        ORDER BY (eng_rank <= ?) DESC, engagement DESC NULLS LAST
        LIMIT ?
        """,
        [base_model, since, model, cfg["ambiguity_threshold"], cfg["top_engagement_per_ticker_week"],
         cfg["top_engagement_per_ticker_week"], cfg["max_items_per_run"]],
    ).fetchall()


def second_pass(con: duckdb.DuckDBPyConnection, scorer: ClaudeScorer, cfg: dict,
                base_model: str = "finbert", max_chars: int = 2000) -> int:
    """Score selected pairs with Claude and store them. Returns rows written."""
    pairs = select_pairs(con, cfg, base_model, scorer.name)
    if not pairs:
        log.info("second pass: nothing to score")
        return 0
    log.info("second pass: %d pairs (cap %d)", len(pairs), cfg["max_items_per_run"])
    ctx = ContextBuilder(load_aliases(), max_chars=max_chars)
    items = [(ctx.build(title, body, ticker, n), ticker) for _, ticker, title, body, n in pairs]
    scores = scorer.score_items(items)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = [(pid, ticker, scorer.name, s.label, s.score, now)
            for (pid, ticker, *_), s in zip(pairs, scores) if s is not None]
    written = db.insert_scores(con, rows)
    if written < len(pairs):
        log.warning("second pass: %d of %d items failed and will be retried", len(pairs) - written, len(pairs))
    return written

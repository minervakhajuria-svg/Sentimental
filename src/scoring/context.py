"""Choose the text to score for each (post, ticker) pair.

When a post mentions several tickers, scoring the whole post gives every
ticker the same score ("NVDA crushed it, AMD is dead" would score the same
for both). So for multi-ticker posts we score only the sentences that mention
the ticker. Single-ticker posts, and posts where no mentioning sentence is
found, are scored as a whole.
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


class ContextBuilder:
    def __init__(self, aliases: list[tuple[str, str, bool]], max_chars: int = 2000):
        self.max_chars = max_chars
        self._alias_by_ticker: dict[str, list[tuple[str, bool]]] = {}
        for alias, ticker, case_sensitive in aliases:
            self._alias_by_ticker.setdefault(ticker, []).append((alias, case_sensitive))
        self._patterns: dict[str, list[re.Pattern]] = {}

    def _mention_patterns(self, ticker: str) -> list[re.Pattern]:
        if ticker not in self._patterns:
            sym = re.escape(ticker)
            pats = [re.compile(rf"(?<![\w$])\$?{sym}(?![\w])", re.IGNORECASE)]
            for alias, cs in self._alias_by_ticker.get(ticker, []):
                pats.append(re.compile(rf"(?<![\w&-]){re.escape(alias)}(?![\w&-])",
                                       0 if cs else re.IGNORECASE))
            self._patterns[ticker] = pats
        return self._patterns[ticker]

    def build(self, title: str | None, body: str | None, ticker: str,
              n_tickers_in_post: int) -> str:
        full = "\n".join(p for p in (title, body) if p)
        if n_tickers_in_post <= 1:
            return full[: self.max_chars]
        pats = self._mention_patterns(ticker)
        hits = [s for s in split_sentences(full) if any(p.search(s) for p in pats)]
        return (" ".join(hits) if hits else full)[: self.max_chars]

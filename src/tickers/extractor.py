"""Find which tickers a post is about.

Three routes, in falling order of confidence (CLAUDE.md §7):
  cashtag  "$TSLA"           explicit, so trusted
  alias    "Nvidia"          curated names in aliases.csv
  bare     "TSLA"            plain uppercase; the noisiest route, so it is gated

Bare matches are where false positives come from ("I think IT is going UP"),
so they need all of:
  * the token is in the universe and not on the blocklist,
  * the post has finance context (keywords, a $ amount, a % or another match),
  * the line isn't shouting. In an all-caps line ("THIS IS GOING TO THE MOON")
    every word looks like a ticker, so bare matches are skipped there.

Only one match is kept per (post, ticker): the one with the highest confidence.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

_URL = re.compile(r"https?://\S+|www\.\S+")
_REDDIT_REF = re.compile(r"(?<![\w/])/?[ru]/\w+")  # r/AMD, u/someone
_CASHTAG = re.compile(r"(?<![\w$])\$([A-Za-z]{1,5}(?:\.[A-Za-z])?)(?![\w])")
_BARE = re.compile(r"(?<![\w$.])([A-Z]{1,5}(?:\.[A-Z])?)(?![\w])")
_MONEY_OR_PCT = re.compile(r"\$\s?\d|\d\s?%")
_CAPS_WORD = re.compile(r"(?<![\w$])[A-Z]{2,}(?![\w])")


@dataclass(frozen=True)
class Match:
    ticker: str
    match_type: str  # cashtag | alias | bare
    confidence: float


def load_blocklist(path: Path = DATA_DIR / "blocklist.txt") -> set[str]:
    words: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0]
        words.update(w.upper() for w in line.split())
    return words


def load_aliases(path: Path = DATA_DIR / "aliases.csv") -> list[tuple[str, str, bool]]:
    with open(path, encoding="utf-8", newline="") as f:
        return [
            (r["alias"].strip(), r["ticker"].strip().upper(), r["case_sensitive"].strip() == "1")
            for r in csv.DictReader(f)
        ]


def _alias_pattern(aliases: list[str], flags: int) -> re.Pattern | None:
    if not aliases:
        return None
    # Longest first, so "Meta Platforms" wins over a shorter overlapping alias.
    alternatives = "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))
    return re.compile(rf"(?<![\w&-])({alternatives})(?![\w&-])", flags)


class TickerExtractor:
    def __init__(
        self,
        universe: set[str],
        aliases: list[tuple[str, str, bool]],
        blocklist: set[str],
        cfg: dict,
    ):
        self.universe = {t.upper() for t in universe}
        self.blocklist = blocklist
        self.conf = cfg["confidence"]
        self.bare_min_length = cfg["bare_min_length"]
        self.shouting_min_words = cfg["shouting_min_words"]
        self.context_words = {w.lower() for w in cfg["finance_context_words"]}

        # Aliases for tickers outside the universe are dropped: a name match is
        # only useful if we track that ticker.
        live = [(a, t, cs) for a, t, cs in aliases if t in self.universe]
        self._alias_ticker = {a.lower(): t for a, t, _ in live}
        self._alias_cs = _alias_pattern([a for a, _, cs in live if cs], 0)
        self._alias_ci = _alias_pattern([a for a, _, cs in live if not cs], re.IGNORECASE)

    @classmethod
    def from_config(cls, universe: set[str], cfg: dict) -> "TickerExtractor":
        return cls(universe, load_aliases(), load_blocklist(), cfg)

    def extract(self, title: str | None, body: str | None) -> list[Match]:
        text = f"{title or ''}\n{body or ''}"
        # Links and r/ or u/ references often contain tickers ("r/AMD_Stock",
        # ".../quote/TSLA") that aren't a mention by the author.
        text = _REDDIT_REF.sub(" ", _URL.sub(" ", text))

        best: dict[str, Match] = {}

        def add(ticker: str, match_type: str) -> None:
            m = Match(ticker, match_type, self.conf[match_type])
            if ticker not in best or m.confidence > best[ticker].confidence:
                best[ticker] = m

        for sym in _CASHTAG.findall(text):
            if sym.upper() in self.universe:
                add(sym.upper(), "cashtag")

        for pattern in (self._alias_cs, self._alias_ci):
            if pattern is not None:
                for name in pattern.findall(text):
                    add(self._alias_ticker[name.lower()], "alias")

        if best or self._has_finance_context(text):
            for line in text.splitlines():
                if self._is_shouting(line):
                    continue
                for sym in _BARE.findall(line):
                    if (
                        len(sym.replace(".", "")) >= self.bare_min_length
                        and sym in self.universe
                        and sym not in self.blocklist
                    ):
                        add(sym, "bare")

        return sorted(best.values(), key=lambda m: (-m.confidence, m.ticker))

    def _has_finance_context(self, text: str) -> bool:
        if _MONEY_OR_PCT.search(text):
            return True
        words = set(re.findall(r"[a-z]+", text.lower()))
        return not words.isdisjoint(self.context_words)

    def _is_shouting(self, line: str) -> bool:
        """True if a line has several all-caps words that aren't tickers.

        Counting only non-ticker caps words means a plain list of tickers
        ("NVDA AMD TSLA") isn't mistaken for shouting.
        """
        caps = _CAPS_WORD.findall(line)
        non_tickers = [w for w in caps if w not in self.universe or w in self.blocklist]
        return len(non_tickers) >= self.shouting_min_words

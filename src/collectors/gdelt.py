"""News coverage from The GDELT Project's bulk GKG files (free; cite GDELT).

GDELT's search API throttles some connections heavily, so this reads the raw
15-minute "Global Knowledge Graph" files instead, which GDELT recommends for
regular users. Each row is one news article with its real publisher domain,
the organizations it mentions, a headline and a publication time.

An article is matched to a ticker only when BOTH hold: GDELT lists the company
among the organizations it mentions, and the headline names it (ticker, company
name or alias). GDELT lists every organization in an article, so on its own
the org match is mostly incidental: in a first live sample only 14% of org
matches had the company in the headline ("shared on Facebook", "the Nasdaq
closed higher"). Organization names that are rarely about the company itself
(e.g. "nasdaq", the index) can be excluded in config.

Files appear with a lag, and occasionally late. Each processed file is
recorded in `gdelt_files`; missing ones are retried on later runs until they
age out.

Terms: free for any use, with citation of and a link to https://www.gdeltproject.org/.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import logging
import re
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

import duckdb

from signals.news_quality import is_about, name_keys

from .base import Collector, Post, content_hash, hash_author
from .http import USER_AGENT

log = logging.getLogger(__name__)

FILE_URL = "http://data.gdeltproject.org/gdeltv2/{stamp}.gkg.csv.zip"
STAMP = "%Y%m%d%H%M%S"
csv.field_size_limit(2**31 - 1)

# GKG 2.1 columns used here.
C_DATE, C_DOMAIN, C_URL, C_ORGS, C_EXTRAS = 1, 3, 4, 13, 26
_TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)
_PUBTIME = re.compile(r"<PAGE_PRECISEPUBTIMESTAMP>(\d{14})</PAGE_PRECISEPUBTIMESTAMP>")


def fetch_file(stamp: str, timeout: float = 60) -> bytes | None:
    """The zipped GKG file for a 15-minute stamp, or None if it isn't published (yet)."""
    req = urllib.request.Request(FILE_URL.format(stamp=stamp), headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def parse_gkg(raw: bytes) -> Iterable[list[str]]:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        text = z.read(z.namelist()[0]).decode("utf-8", errors="replace")
    for row in csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(row) > C_EXTRAS:
            yield row


def expected_stamps(start: datetime, end: datetime) -> list[str]:
    """15-minute stamps in [start, end), aligned to the quarter hour."""
    t = start.replace(minute=start.minute - start.minute % 15, second=0, microsecond=0)
    if t < start:
        t += timedelta(minutes=15)
    out = []
    while t < end:
        out.append(t.strftime(STAMP))
        t += timedelta(minutes=15)
    return out


class GdeltCollector(Collector):
    source = "news"

    def __init__(self, con: duckdb.DuckDBPyConnection, matcher: Callable[[], "Matcher"],
                 files_per_hour: int = 4, give_up_after_hours: float = 48, min_title_chars: int = 20,
                 pause_seconds: float = 1.0, fetch: Callable[[str], bytes | None] = fetch_file,
                 sleep: Callable[[float], None] = time.sleep,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc).replace(tzinfo=None)):
        self.con = con
        self._matcher = matcher  # built at fetch time from the current universe
        self.files_per_hour = max(1, min(4, files_per_hour))
        self.give_up_after = timedelta(hours=give_up_after_hours)
        self.min_title_chars = min_title_chars
        self.pause_seconds = pause_seconds
        self._fetch, self._sleep, self._now = fetch, sleep, now
        self.communities: list[str] = []
        self.failed_communities: list[str] = []
        self.failure_tolerance = 0.25  # a few late files is normal for GDELT

    @classmethod
    def from_config(cls, cfg: dict, con: duckdb.DuckDBPyConnection,
                    matcher: Callable[[], "Matcher"]) -> "GdeltCollector":
        return cls(con, matcher, files_per_hour=cfg.get("files_per_hour", 4),
                   give_up_after_hours=cfg.get("give_up_after_hours", 48),
                   min_title_chars=cfg.get("min_title_chars", 20),
                   pause_seconds=cfg.get("pause_seconds", 1.0))

    def _pending(self, since: datetime) -> list[str]:
        now = self._now()
        # Also revisit files that were missing earlier, as long as they're recent
        # enough. The very first run just takes the normal window.
        first_run = not self.con.execute("SELECT count(*) FROM gdelt_files").fetchone()[0]
        start = since if first_run else min(since, now - self.give_up_after)
        stamps = expected_stamps(start, now - timedelta(minutes=15))
        stamps = [s for s in stamps if int(s[10:12]) // 15 < self.files_per_hour]  # optional sampling
        done = {r[0] for r in self.con.execute(
            "SELECT stamp FROM gdelt_files WHERE status = 'done'").fetchall()}
        return [s for s in stamps if s not in done]

    def fetch(self, since: datetime) -> Iterable[Post]:
        matcher = self._matcher()
        pending = self._pending(since)
        self.communities, self.failed_communities = pending, []
        posts: dict[str, Post] = {}
        for i, stamp in enumerate(pending):
            if i:
                self._sleep(self.pause_seconds)
            status, n_rows, n_matched = "done", 0, 0
            try:
                raw = self._fetch(stamp)
                if raw is None:
                    status = "missing"
                    age = self._now() - datetime.strptime(stamp, STAMP)
                    if age < self.give_up_after:
                        self.failed_communities.append(stamp)
                else:
                    for row in parse_gkg(raw):
                        n_rows += 1
                        post = self._to_post(row, matcher)
                        if post is not None:
                            n_matched += 1
                            posts.setdefault(post.id, post)
            except Exception as e:
                status = "error"
                self.failed_communities.append(stamp)
                log.warning("gdelt %s: %s", stamp, e)
            self.con.execute(
                "INSERT OR REPLACE INTO gdelt_files VALUES (?, ?, ?, ?, ?)",
                [stamp, status, n_rows, n_matched, self._now()],
            )
        log.info("gdelt: %d files checked, %d articles matched to tickers (%d files not published yet)",
                 len(pending), len(posts), len(self.failed_communities))
        return list(posts.values())

    def _to_post(self, row: list[str], matcher: "Matcher") -> Post | None:
        candidates = matcher.tickers_for(row[C_ORGS].split(";"))
        if not candidates:
            return None
        extras = row[C_EXTRAS]
        m = _TITLE.search(extras)
        title = html.unescape(m.group(1)).strip() if m else ""
        if len(title) < self.min_title_chars:
            return None  # no headline to score or classify
        tickers = [t for t in candidates if matcher.named_in(t, title)]
        if not tickers:
            return None  # mentioned somewhere in the article, but not what it's about
        pub = _PUBTIME.search(extras)
        try:
            created = datetime.strptime(pub.group(1) if pub else row[C_DATE], STAMP)
        except ValueError:
            created = datetime.strptime(row[C_DATE], STAMP)
        url, domain = row[C_URL], row[C_DOMAIN].lower()
        return Post(
            id=f"news:gdelt:{hashlib.sha1(url.encode()).hexdigest()[:20]}",
            source=self.source,
            community=domain,                     # the real publisher
            author_id=hash_author(f"news:{domain}"),
            author_age_days=None, author_karma=None,
            created_at=created, collected_at=self._now(),
            title=title, body=None, url=url, engagement=None,
            content_hash=content_hash(title, None),
            source_tickers=tuple(tickers),
        )


_CLEAN = re.compile(r"\b(inc|incorporated|corp|corporation|co|company|holdings?|group|plc|ltd|limited|"
                    r"n\.?v|s\.?a|ag|the|class [a-z]|common stock)\b\.?", re.IGNORECASE)


class Matcher:
    """Organization name -> tickers, plus the name forms to confirm in a headline."""

    def __init__(self, org_map: dict[str, list[str]], title_keys: dict[str, list[str]],
                 exclude_orgs: set[str] = frozenset()):
        self.org_map, self.title_keys = org_map, title_keys
        self.exclude = {o.lower() for o in exclude_orgs}

    def tickers_for(self, orgs: list[str]) -> list[str]:
        names = {o.strip() for o in orgs if o.strip()} - self.exclude
        return sorted({t for o in names for t in self.org_map.get(o, ())})

    def named_in(self, ticker: str, title: str) -> bool:
        return is_about(ticker, title, None, self.title_keys.get(ticker, []))


def build_matcher(universe: list[tuple[str, str]], aliases: list[tuple[str, str, bool]],
                  exclude_orgs: list[str] = ()) -> Matcher:
    alias_by_ticker: dict[str, list[str]] = {}
    for alias, ticker, _ in aliases:
        alias_by_ticker.setdefault(ticker, []).append(alias)
    keys = {t: name_keys(n, alias_by_ticker.get(t, [])) for t, n in universe}
    return Matcher(build_org_map(universe, aliases), keys, set(exclude_orgs))


def build_org_map(universe: list[tuple[str, str]], aliases: list[tuple[str, str, bool]]) -> dict[str, list[str]]:
    """Organization name (as GDELT writes it: lower case) -> tickers.

    `universe` is (ticker, company_name), most liquid first: when two listings
    share a name (GOOG/GOOGL) the more liquid one wins. Only full names and
    curated aliases are used; first-word shortcuts ("general", "american")
    would match far too much.
    """
    out: dict[str, list[str]] = {}
    known = {t for t, _ in universe}
    for alias, ticker, _ in aliases:
        if ticker in known:
            out.setdefault(alias.lower(), [ticker])
    for ticker, name in universe:
        key = " ".join(re.sub(r"[^\w\s&'-]", " ", _CLEAN.sub(" ", name or "")).split()).lower()
        if len(key) >= 4:
            out.setdefault(key, [ticker])
    return out

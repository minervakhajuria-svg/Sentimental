"""GDELT bulk-file collector, against an invented GKG sample (no network)."""

import io
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

from collectors.gdelt import GdeltCollector, build_matcher, build_org_map, expected_stamps, parse_gkg
from storage import db

SAMPLE = (Path(__file__).resolve().parent.parent / "fixtures" / "gdelt" / "sample.gkg.tsv").read_bytes()
UNIVERSE = [("NVDA", "NVIDIA Corporation"), ("AAPL", "Apple Inc."), ("MSFT", "Microsoft Corporation"),
            ("GM", "General Motors Company"), ("GOOGL", "Alphabet Inc."), ("GOOG", "Alphabet Inc.")]
ALIASES = [("Apple", "AAPL", True), ("Google", "GOOGL", False), ("Nvidia", "NVDA", False)]
NOW = datetime(2026, 10, 1, 13, 0)


def zipped(data: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("20261001120000.gkg.csv", data)
    return buf.getvalue()


def test_org_map_uses_full_names_and_aliases_only():
    m = build_org_map(UNIVERSE, ALIASES)
    assert m["nvidia"] == ["NVDA"] and m["apple"] == ["AAPL"] and m["microsoft"] == ["MSFT"]
    assert m["general motors"] == ["GM"]
    assert "general" not in m                      # no first-word shortcuts
    assert m["alphabet"] == ["GOOGL"]              # first listing (most liquid) wins
    assert m["google"] == ["GOOGL"]


def test_expected_stamps_align_to_quarter_hours():
    assert expected_stamps(datetime(2026, 10, 1, 11, 7), datetime(2026, 10, 1, 12, 1)) == [
        "20261001111500", "20261001113000", "20261001114500", "20261001120000"]


def collector(con, files, **kw):
    fetched = []

    def fetch(stamp):
        fetched.append(stamp)
        return files.get(stamp)

    c = GdeltCollector(con, lambda: build_matcher(UNIVERSE, ALIASES, ["nasdaq"]), fetch=fetch,
                       sleep=lambda s: None, now=lambda: NOW, **kw)
    return c, fetched


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "g.duckdb")
    yield c
    c.close()


def test_matches_articles_to_tickers(con):
    c, _ = collector(con, {"20261001120000": zipped(SAMPLE)})
    posts = {p.url: p for p in c.fetch(datetime(2026, 10, 1, 11, 50))}
    # The two radio stories list Nvidia among organizations but the headline
    # never names it: incidental, so not matched.
    assert set(posts) == {"https://example-news.com/a1", "https://tech-site.com/c"}
    a = posts["https://example-news.com/a1"]
    assert a.source == "news" and a.community == "example-news.com"   # the real publisher
    assert a.source_tickers == ("NVDA",)
    assert a.created_at == datetime(2026, 10, 1, 11, 45)                # precise publication time
    t = posts["https://tech-site.com/c"]
    assert t.title == "Apple & Microsoft Face New Café Rules"           # HTML entities decoded
    assert t.source_tickers == ("AAPL", "MSFT")


def test_files_are_fetched_once_and_late_ones_retried(con):
    files = {}
    c, fetched = collector(con, files)
    list(c.fetch(datetime(2026, 10, 1, 11, 50)))
    assert fetched == ["20261001120000", "20261001121500", "20261001123000"]
    assert c.failed_communities == fetched          # none published yet: recorded as missing

    files["20261001120000"] = zipped(SAMPLE)        # the late file appears
    c2, fetched2 = collector(con, files)
    posts = list(c2.fetch(datetime(2026, 10, 1, 11, 50)))
    assert "20261001120000" in fetched2 and len(posts) == 2
    c3, fetched3 = collector(con, files)
    list(c3.fetch(datetime(2026, 10, 1, 11, 50)))
    assert "20261001120000" not in fetched3         # done: never downloaded again


def test_sampling_reduces_files(con):
    c, fetched = collector(con, {}, files_per_hour=2)
    list(c.fetch(datetime(2026, 10, 1, 11, 0)))
    assert all(s[10:12] in ("00", "15") for s in fetched)


def test_parse_skips_short_rows():
    assert len(list(parse_gkg(zipped(SAMPLE + b"broken\trow\n")))) == 6


def test_excluded_orgs_and_headline_confirmation():
    m = build_matcher(UNIVERSE + [("NDAQ", "Nasdaq, Inc.")], ALIASES, ["nasdaq"])
    assert m.tickers_for(["nasdaq", "nvidia"]) == ["NVDA"]          # the index name is ignored
    assert m.named_in("NVDA", "Nvidia shares climb on new order")
    assert not m.named_in("NVDA", "A startup plans a huge data center")

from pathlib import Path

import pytest
import yaml

from settings import PROJECT_ROOT
from tickers.extractor import TickerExtractor, load_aliases, load_blocklist

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
CASES = yaml.safe_load((FIXTURES / "extractor_cases.yaml").read_text(encoding="utf-8"))

# Includes real tickers that are also common words (IT, NOW, ALL, DD, ON, AI, UP)
# so the blocklist is actually exercised.
UNIVERSE = {
    "NVDA", "TSLA", "AMD", "AAPL", "MSFT", "GOOGL", "PLTR", "SOFI", "HOOD", "GME",
    "AMC", "BRK.B", "T", "F", "IT", "NOW", "ALL", "DD", "ON", "AI", "UP",
}


@pytest.fixture(scope="module")
def cfg():
    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["extraction"]


@pytest.fixture(scope="module")
def extractor(cfg):
    return TickerExtractor(UNIVERSE, load_aliases(), load_blocklist(), cfg)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_fixture_case(extractor, case):
    got = {m.ticker: m.match_type for m in extractor.extract(case.get("title"), case.get("body"))}
    assert got == (case["expect"] or {})


def test_overall_precision_and_recall(extractor):
    """Aggregate score over all fixture cases, as a regression guard on accuracy."""
    tp = fp = fn = 0
    for case in CASES:
        got = {m.ticker for m in extractor.extract(case.get("title"), case.get("body"))}
        want = set(case["expect"] or {})
        tp += len(got & want)
        fp += len(got - want)
        fn += len(want - got)
    assert tp / (tp + fp) >= 0.95
    assert tp / (tp + fn) >= 0.9


def test_confidence_comes_from_config(extractor, cfg):
    [m] = extractor.extract("$NVDA", None)
    assert m.confidence == cfg["confidence"]["cashtag"]


def test_aliases_for_untracked_tickers_are_dropped(cfg):
    ex = TickerExtractor({"NVDA"}, [("Nvidia", "NVDA", False), ("Tesla", "TSLA", False)],
                         set(), cfg)
    assert [m.ticker for m in ex.extract("Nvidia and Tesla", None)] == ["NVDA"]


def test_empty_post(extractor):
    assert extractor.extract(None, None) == []


def test_shipped_aliases_and_blocklist_are_well_formed():
    aliases = load_aliases()
    assert len(aliases) > 50
    assert all(a and t and t == t.upper() for a, t, _ in aliases)
    assert len({a.lower() for a, _, _ in aliases}) == len(aliases), "duplicate alias"
    block = load_blocklist()
    assert {"IT", "NOW", "ALL", "ARE", "FOR", "A", "ON", "CAN"} <= block

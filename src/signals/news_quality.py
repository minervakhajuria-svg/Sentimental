"""What kind of news item is this, and is it really about the ticker?

News feeds mix genuine reporting with three kinds of noise, each handled
differently by aggregation (weights in config, signals.noise.news):

  price_recap    "Why NVDA stock jumped today", "Visa enters October 6% below its
                 high". These narrate a price move that already happened. Counting
                 them makes news tone an echo of past returns and would fool the
                 lead-lag test, so they are dropped from scoring entirely.
  promo          Listicles and buy/sell opinion pieces ("2 stocks to buy and hold
                 forever", "Is X a buy right now?"). Mostly evergreen content
                 marketing; down-weighted.
  press_release  The company's own announcements. Useful facts, but the tone is
                 always upbeat; down-weighted.
  news           Everything else.

Finnhub reports the aggregator ("Yahoo") rather than the publisher, so this is
judged from the headline and summary, not the outlet.
"""

from __future__ import annotations

import re

_FLAGS = re.IGNORECASE

# Past-tense or present price moves. Deliberately excludes bare "up"/"down",
# which also appear in phrases like "yielding up to 13%".
_MOVE = (r"(soar|jump|plunge|plummet|tumble|surge|sink|sank|slump|fall|fell|rise|rose|rall(y|ie)|"
         r"drop|climb|gain|slide|slid|spike|crash|pop|rocket|skyrocket|dip|rebound|tripl|doubl|halv)\w*")
PRICE_RECAP = [re.compile(p, _FLAGS) for p in (
    rf"\bwhy\b.{{0,80}}\b(stock|shares|stocks)\b.{{0,60}}\b{_MOVE}",           # "Why X Stock Plummeted 56%"
    rf"\b{_MOVE}\b.{{0,40}}\b\d+(\.\d+)?\s?%",                              # "jumps 12%", "fell 6.5% after"
    r"\b(is|are|was|were)\s+(up|down)\s+\d+(\.\d+)?\s?%",                    # "Is Up 33% in 90 Days"
    r"\b\d+(\.\d+)?\s?%\s+(below|above|off)\s+(its|their)\s+(high|low|peak)",
    r"\b52-week (high|low)s?\b",
    r"\bwhat'?s going on with\b",
    r"\b(stock|shares) (is|are) (moving|trading (higher|lower))\b",
    r"\bhow (they'?re|it'?s) performing\b",
    r"\b(biggest|top) (gainers|losers|movers)\b",
    r"\b(climbed|jumped|dropped|fell|surged|rose|soared|plunged|slid|tumbled|sank|rallied)\b,?\s+(what|why|here'?s)\b",
)]
# Forecasts aren't recaps: "could surge 250%" looks forward, not back.
FORECAST = re.compile(r"\b(could|may|might|will|predicts?|forecasts?|before it|set to|poised to|expects?)\b",
                      _FLAGS)

PROMO = [re.compile(p, _FLAGS) for p in (
    r"^\s*\d+\s+\S+.{0,40}\b(stocks?|etfs?|reasons)\b",          # "2 Super Semiconductor Stocks..."
    r"\b(stocks?|etfs?) to (buy|own|hold|sell|avoid|watch)\b",
    r"\bis (it )?.{0,50}\ba (buy|sell|smart buy|good buy)\b",   # "Is Netflix (NFLX) Stock a Buy?"
    r"\bshould you (buy|sell)\b",
    r"\bbetter\b.{0,40}\b(buy|bet|stock|pick)\b",               # "Better High-Yield Dividend Stock: ..."
    r"\bbetter than\b",
    r"\bbuy\b.{0,30}\binstead\b",
    r"\b(buy|own|hold)\b.{0,15}\b(forever|for the next|through the next|for decades|never sell)\b",
    r"\bmillionaire\b",
    r"\bmake you rich\b",
    r"\bset you up for life\b",
    r"\bi'?d\b",                                               # first-person headlines are opinion
    r"\bi'?m (still )?(buying|selling|adding)\b",
    r"\bmy (top|favorite|favourite|best)\b",
    r"\bdon'?t (sell|buy)\b",
    r"\bshould own\b",
    r"\baway from this stock\b",
    r"\bjim cramer\b",
    r"\bhistory says\b",
    r"\btoo late to buy\b",
    r"\b(with|invest(ing)?|invested|split) \$[\d,]+\b",          # "$1,000 Invested in ...", "with $10,000"
    r"\btime to (buy|sell)\b",
    r"\byou'?d need\b",                                        # "How much you'd need to invest..."
    r"\$[\d,]+ investment\b",
    r"\b(no-brainer|screaming buy|unstoppable|magnificent seven stocks?)\b",
    r"\b(here'?s|the) (1 |one )?reason\b",
    r"\b(trade|opportunity) of the (decade|century)\b",
    r"\bignore them\b",
)]

PRESS_RELEASE = [re.compile(p, _FLAGS) for p in (
    r"^(?!.*\bamong\b).{0,60}\bannounces?\b",                 # "Acme Announces ...", not "... among 16 to announce"
    r"\bdeclares?\b.{0,40}\bdividend\b",
    r"\bto (report|release|announce)\b.{0,60}\b(results|earnings)\b",
    r"\b(conference call|earnings call|webcast)\b",
    r"\bto (participate|present) (in|at)\b",
    r"\bprices?\b.{0,30}\b(offering|notes)\b",
    r"\bcompletes? (acquisition|merger|offering)\b",
    r"\bappoints?\b",
)]
PRESS_RELEASE_OUTLETS = {"business wire", "businesswire", "pr newswire", "prnewswire", "globenewswire",
                         "accesswire", "newsfile", "accesswire.com", "globe newswire"}


def classify(title: str | None, body: str | None = None, outlet: str | None = None) -> str:
    """One of price_recap, promo, press_release, news. Judged from the headline.

    Order matters: buy/sell pitches often quote a past move ("down 41%... buy
    now?"), and those are promotion first.
    """
    head = title or ""
    if (outlet or "").strip().lower() in PRESS_RELEASE_OUTLETS or any(p.search(head) for p in PRESS_RELEASE):
        return "press_release"
    if any(p.search(head) for p in PROMO):
        return "promo"
    if not FORECAST.search(head) and any(p.search(head) for p in PRICE_RECAP):
        return "price_recap"
    return "news"


_SUFFIX = re.compile(r"\b(inc|incorporated|corp|corporation|co|company|holdings?|group|plc|ltd|limited|"
                     r"n\.?v|s\.?a|ag|the|class [a-z]|common stock)\b\.?", _FLAGS)


def name_keys(company_name: str | None, aliases: list[str] = ()) -> list[str]:
    """Short forms of a company's name to look for in text: 'Costco Wholesale Corporation' -> ['costco wholesale', 'costco']."""
    keys = [a.lower() for a in aliases]
    if company_name:
        clean = re.sub(r"[^\w\s&'-]", " ", _SUFFIX.sub(" ", company_name)).split()
        if clean:
            keys.append(" ".join(clean).lower())
            if len(clean[0]) >= 4:
                keys.append(clean[0].lower())
    return list(dict.fromkeys(k for k in keys if k))


def is_about(ticker: str, title: str | None, body: str | None, keys: list[str]) -> bool:
    """Does the text actually mention the ticker or company? Feeds sometimes tag
    loosely related stories (a lifestyle piece that mentions a store, say)."""
    text = f"{title or ''} {body or ''}"
    if re.search(rf"(?<!\w)\$?{re.escape(ticker)}(?!\w)", text):  # "COST" or "$COST"
        return True
    low = text.lower()
    return any(re.search(rf"(?<!\w){re.escape(k)}(?!\w)", low) for k in keys)

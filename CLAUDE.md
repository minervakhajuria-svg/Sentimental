# Sentimental: Stock Sentiment Analyser Project Spec

## 1. Purpose
A personal research tool that scans online discussion about US-listed stocks and produces a **weekly ranked list** of tickers where attention and sentiment are shifting, ideally *before* price reacts.

It is a **screening aid**, not a trading signal. It must never output buy/sell advice. The ranking is **sentiment-driven**; price and volume are context columns only (until the backtest proves otherwise).

## 2. Constraints and decisions (already made)
- **Market:** US equities (NYSE/Nasdaq) only.
- **Cadence:** collect **daily**, score and rank **weekly** (run on the weekend after Friday's US close).
- **User time budget:** one weekly review. No intraday logic, no real-time streaming.
- **Ranking input:** sentiment and attention only. Price and volume are never inputs to the composite score in v1.
- **X/Twitter:** out of scope for v1 (cost). Design the collector interface so it can be added later.
- **Personal use.** Respect each platform's terms and rate limits. Do not scrape sites that prohibit it.

## 3. Tech stack
- Python 3.11+, `uv` or `venv`
- Storage: **DuckDB** (single file `data/sentiment.duckdb`); keep schema portable to Postgres
- Reddit: `praw`
- NLP: `transformers` + `torch` (FinBERT: `ProsusAI/finbert`)
- Market data: `yfinance` (MVP), abstracted behind an interface
- UI: **Streamlit**
- Scheduling: cron or APScheduler
- Config: `config.yaml` (non-secret) + `.env` (secrets, never committed)
- Tests: `pytest`, using saved fixture files (never hit live APIs in tests)

## 4. Repository layout
```
sentiment-analyser/
├── CLAUDE.md
├── config.yaml
├── .env.example
├── pyproject.toml
├── data/                  # gitignored: duckdb file, caches
├── fixtures/              # small saved API responses for tests
├── src/
│   ├── collectors/        # base.py, reddit.py, (later) stocktwits.py, news.py
│   ├── storage/           # db.py, schema.sql
│   ├── tickers/           # universe.py, extractor.py, aliases.csv, blocklist.txt
│   ├── scoring/           # finbert.py, llm_second_pass.py (later)
│   ├── signals/           # aggregate.py, composite.py, filters.py
│   ├── market/            # prices.py (context data), insiders.py (Form 4, later)
│   ├── analysis/          # leadlag.py, backtest.py
│   ├── jobs/              # collect_daily.py, rank_weekly.py
│   └── app/               # streamlit_app.py
└── tests/
```

## 5. Data model (DuckDB)

**posts**: one row per collected item
| column | type | notes |
|---|---|---|
| id | TEXT PK | `{source}:{native_id}` |
| source | TEXT | reddit, stocktwits, news |
| community | TEXT | subreddit / feed name |
| author_id | TEXT | hashed or native username |
| author_age_days | INT | nullable |
| author_karma | INT | nullable |
| created_at | TIMESTAMP | UTC |
| collected_at | TIMESTAMP | UTC |
| title | TEXT | |
| body | TEXT | keep raw text so it can be re-scored later |
| url | TEXT | |
| engagement | INT | upvotes + comments (source-specific, normalised later) |
| content_hash | TEXT | for de-duplication |

**post_tickers**: one row per (post, ticker) match
`post_id, ticker, match_type (cashtag|alias|bare|source), confidence`

**post_scores**: one row per (post, ticker, model)
`post_id, ticker, model, label (pos|neg|neu), score (-1..1), scored_at`

**prices_daily**: `ticker, date, open, high, low, close, volume`

**ticker_universe**: `ticker, company_name, exchange, market_cap, avg_dollar_volume_30d, updated_at`

**weekly_signals**: one row per (week_start, ticker)
`mentions, attention_z, sentiment, sentiment_prev, momentum, breadth, composite_bull, composite_bear, ret_5d, ret_30d, rel_volume, updown_vol_ratio, insider_buy_flag, early_chatter_flag`

## 6. Collectors
- Common interface: `fetch(since: datetime) -> Iterable[Post]`, normalised to the `posts` schema.
- Idempotent: re-running a day never creates duplicates (upsert on `id`).
- Respect rate limits; back off on errors; log failures and continue.
- **Reddit v1 communities:** r/wallstreetbets, r/stocks, r/investing, r/options, r/StockMarket, r/SecurityAnalysis. List lives in `config.yaml`.
- Credentials from `.env` (`REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`). Check Reddit's current API access rules before building.
- **Start collecting in phase 1.** The accumulated history is the project's most valuable asset.

## 7. Ticker extraction
1. Build the universe from a listed-US-equities source; filter by market cap and liquidity floors (config).
2. Match **cashtags** (`$TSLA`) with high confidence.
3. Match **company names/aliases** (`aliases.csv`, e.g. "Nvidia" → NVDA) with medium confidence.
4. Match **bare uppercase tickers** only if not on `blocklist.txt` (ambiguous words such as ALL, NOW, IT, ARE, FOR, A, ON, CAN) and the post has finance context.
5. Store match type and confidence; downstream aggregation can weight or drop low-confidence matches.
6. Unit-test the extractor heavily with fixture sentences, including false-positive cases.

## 8. Sentiment scoring
- **Pass 1 (all posts):** FinBERT locally, mapped to a score in [-1, 1] (`p_pos - p_neg`). Batch inference; cache results.
- **Pass 2 (later phase):** a small, cheap Claude model for ambiguous posts (|score| < threshold) and top-engagement posts per ticker. Handles sarcasm and memes. Store as a separate `model` value in `post_scores`; never overwrite pass 1.
- Score at the (post, ticker) level. If a post mentions several tickers, score the sentences near each mention where feasible.

## 9. Noise filtering and weighting
- Drop exact and near-duplicate content (`content_hash`, plus fuzzy match for reposts).
- Down-weight or drop: accounts under N days old, very low karma, obvious bots, spam patterns.
- Cap each author's influence per ticker per week (e.g. at most K posts counted) so one person can't dominate.
- **Post weight** = `log(1 + engagement) * author_weight * source_weight`. Weights live in config.

## 10. Weekly signals (the core)
Computed per ticker over the week ending Friday, for tickers passing the filters below.

| Component | Definition |
|---|---|
| **Attention (A)** | Mean daily weighted mentions this week vs the ticker's trailing 30-day daily baseline, as a z-score. Floor the baseline std to avoid blow-ups on quiet tickers; clip to a sane range |
| **Sentiment (S)** | Post-weight-averaged net sentiment this week, in [-1, 1] |
| **Momentum (M)** | `S(this week) - S(last week)` |
| **Breadth (B)** | Combination of distinct authors, distinct communities and distinct sources (log-scaled) |

**Eligibility filters (config):** minimum weekly mentions (default 20), minimum distinct authors, market-cap floor, average dollar volume floor.

**Composite (cross-sectional z-scores across eligible tickers that week; weights in config):**
```
composite_bull = 0.35*z(A) + 0.35*z(S)  + 0.15*z(M)  + 0.15*z(B)
composite_bear = 0.35*z(A) - 0.35*z(S)  - 0.15*z(M)  + 0.15*z(B)
```
**Outputs:** two ranked lists.
1. **Heating up, bullish:** top N by `composite_bull` where S > 0 and A > 0
2. **Heating up, bearish:** top N by `composite_bear` where S < 0 and A > 0

Price, volume and insider data **must not** feed into these formulas.

## 11. Context columns (display and validation only)
Computed from `prices_daily` and filings, shown next to each ranked ticker:
- `ret_5d`, `ret_30d`
- `rel_volume`: this week's average volume / trailing 30-day average volume
- `updown_vol_ratio`: volume on up-days / volume on down-days (rough buying-vs-selling pressure proxy; note that true buy-side volume is not observable with free data)
- `insider_buy_flag`: recent open-market purchase in SEC EDGAR Form 4 filings (later phase)
- **`early_chatter_flag`** = composite in the top decile **and** `abs(ret_5d)` below a threshold (default 3%) **and** `rel_volume` above a threshold (default 1.2). This encodes the hypothesis that chatter precedes price.

Display warnings in the UI for: rising volume + falling price + bullish chatter.

## 12. Validation: lead-lag and backtest
The assumption "chatter leads price" is **unproven**; test it.
- For each week and ticker, compare sentiment/attention change with returns in the **prior**, **same** and **next** week.
- Metrics: Spearman rank correlation of composite vs next-week return; return spread between top and bottom quintile; hit rate vs a naive baseline (e.g. mention volume alone).
- Experiment: does adding `rel_volume` to the composite improve next-week rank correlation? If yes, promote it from context to input; if no, leave it as context.
- Require several months of accumulated data before drawing conclusions. Beware lookahead bias: only use data available at ranking time.
- If sentiment mostly follows price, report it, and consider repurposing it as a crowding or contrarian indicator.

## 13. Streamlit app
- Page 1: the two ranked tables with context columns, `early_chatter_flag`, and a week selector.
- Page 2: ticker drill-down showing trend of mentions and sentiment vs price, plus the **top source posts** behind the score (with links) so the user can read *why* a ticker ranked.
- Page 3 (later): backtest and lead-lag results.
- Persistent footer: "Screening aid only, not investment advice."

## 14. Build phases (do one at a time; commit after each)
1. **Collector and storage.** DuckDB schema, Reddit collector, daily job, fixtures and tests. *Done when:* a daily run stores posts idempotently.
2. **Ticker extraction.** Universe, aliases, blocklist, extractor and tests. *Done when:* fixture accuracy is acceptable and false positives are low.
3. **Scoring and weekly ranking.** FinBERT, aggregation, filters, composite, two ranked lists written to `weekly_signals`. *Done when:* `rank_weekly.py` produces both lists from stored data.
4. **Streamlit output.** Ranked tables and drill-down to source posts.
5. **Price and volume context.** yfinance, return and volume columns, `early_chatter_flag`.
6. **More sources.** StockTwits (its bullish/bearish tags are useful for validating the scorer), news via Finnhub/RSS; then noise and bot filtering.
7. **Lead-lag analysis and backtest.**
8. **LLM second-pass scoring**, Form 4 insider flag, scheduling and deployment.

## 15. Working conventions for Claude Code
- Read this file first each session; keep it updated when decisions change.
- Build one module at a time, with tests against fixtures. Never call live APIs in tests.
- Keep all thresholds and weights in `config.yaml`, not hardcoded.
- Secrets only in `.env`; never log or commit them.
- Log to file; jobs must fail loudly but never corrupt existing data.
- Prefer simple, readable code. Add docstrings that explain *why*, not just *what*.
- Don't add features outside the current phase without asking.

## 16. Status and implementation notes
- **Phase 1 (collector and storage): done 2026-10-04.** 22 tests passing against fixtures.
- **Reddit API access:** since Nov 2025, new apps need manual approval under Reddit's Responsible Builder Policy (free non-commercial tier ~100 QPM). Credentials must be obtained before live collection can start.
- Setup: `py -3 -m venv .venv` then `.venv/Scripts/python -m pip install -e ".[dev]"`. Packages under `src/` are top-level (`collectors`, `storage`, `jobs`, `settings`).
- Run daily: `.venv/Scripts/python -m jobs.collect_daily`. Exit 0 = ok, 2 = partial (fetched data saved), 1 = nothing collected. Logs in `logs/collect_daily.log`.
- Phase 1 collects Reddit **submissions only** (not comments). Usernames are stored as a 16-char SHA-256 hash; `url` is the Reddit permalink; `engagement` = score + num_comments.
- Upsert on `id` refreshes `engagement` (and fills author details if newly available) but never overwrites title/body/created_at/collected_at, so later "[removed]" edits can't erase text.
- Daily runs look back `collect.lookback_hours` (36h) so consecutive runs overlap; upserts make this harmless.
- **Phase 2 (ticker extraction): done 2026-10-04.**
  - Universe = Nasdaq Trader symbol directory (`nasdaqlisted.txt`, `otherlisted.txt`): NASDAQ, NYSE, NYSE American common equity; ETFs, test issues, warrants, rights, units, preferreds and notes excluded (LP common units kept). Share classes use a dot (`BRK.B`); converted to Yahoo's dash form only inside `market/prices.py`.
  - Market cap and 30-day dollar volume via yfinance (`YFinanceProvider` behind `MarketDataProvider`). Liquidity is fetched in bulk first; market cap only for tickers above the liquidity floor (one request each). All listings are stored; floors apply at load time, so they can be changed in config without a rebuild. A refresh where fewer than `min_data_coverage` of listings have data is refused, and the old universe is kept.
  - Refresh weekly: `python -m jobs.build_universe` (~20-30 min).
  - Match types: `cashtag` (0.95), `alias` (0.75, curated `aliases.csv` with a case-sensitivity flag for common words like Apple/Visa/Ford), `bare` (0.5). Bare matches need finance context and length >= 2, must not be on the blocklist, and are skipped on "shouting" lines (>= 3 all-caps non-ticker words). URLs and r/ or u/ references are stripped first.
  - `collect_daily` tags new posts after collecting (exit 2 if tagging fails). Run `python -m jobs.extract_tickers --rebuild` after editing aliases or the blocklist.
- **Phase 3 (scoring and weekly ranking): done 2026-10-04.**
  - Week = Saturday 00:00 UTC through Friday 23:59 UTC (weekend chatter counts toward the following trading week), keyed by its Monday in `weekly_signals.week_start`.
  - Multi-ticker posts are scored on the sentences that mention each ticker; single-ticker posts on the whole text. `post_scores` doubles as the scoring cache; scores are inserted, never overwritten.
  - Filters before aggregation: exact reposts (same `content_hash`, keep earliest), per-author cap per ticker per 7-day block, matches below `min_match_confidence`. Author weight is fixed at 1 until phase 6.
  - Attention baseline = 30 days before the week, zero-filled daily weighted mentions, std floored and z clipped (config). Missing last-week sentiment gives NULL momentum, which contributes 0 to the composite.
  - `weekly_signals` stores every eligible ticker; the two lists are queries over it (`signals.composite.ranked_lists`). Context columns stay NULL until phase 5.
  - Run: `python -m jobs.rank_weekly [--week-ending YYYY-MM-DD] [--no-score]`. The first run downloads FinBERT (~440 MB) from Hugging Face.
- **Phase 4 (Streamlit output): done 2026-10-04.**
  - Run: `streamlit run src/app/streamlit_app.py`. `streamlit_app.py` holds only navigation and the footer; pages live in `app/views.py`, read-only queries in `app/queries.py`.
  - Page 1: week selector, both ranked lists with context columns (blank until phase 5) and `early_chatter_flag`. Clicking a row opens the drill-down. Page 2: daily mentions and mean sentiment (and price once collected), weekly signal history, and the week's top posts by engagement (reposts collapsed) with links.
  - The app opens a short-lived read-only DuckDB connection per render, so it never blocks the collection/ranking jobs; if a job holds the write lock it shows a "busy" message.
  - The disclaimer shows in the sidebar and at the bottom of every page.
  - Demo data: `python tests/demo_data.py` writes `data/demo.duckdb` (invented posts); point the app at it with `SENTIMENT_DB_PATH=data/demo.duckdb`. `.claude/launch.json` has a `streamlit-demo` config for this.
- **App name and look (2026-10-04):** the app is called **Sentimental**. Fluorescent green on black, from the user's "Neon green + black finance UI" design canvas: background `#0A0A0A`, cards `#111314` with `#1F2421` borders and 16px radius, accent `#39FF14` with a soft glow, bearish `#FF4D4D`, muted text `#8A8F98`; Space Grotesk for UI text, JetBrains Mono for numbers and uppercase section labels. Base theme is in `.streamlit/config.toml`; the rest (logo, cards, gauge, mix bar, post list, chart styling) is in `src/app/theme.py`. Containers keyed `card-*` get card styling; `card-glow-*` adds the glow. All post text rendered as HTML goes through `html.escape`.
- **Phase 5 (price and volume context): done 2026-10-04.**
  - Bars are split- and dividend-adjusted (yfinance `auto_adjust=True`) and upserted with overwrite. `collect_daily` refreshes the last 10 days for tickers mentioned in the last 30 days. `rank_weekly` re-fetches 60 days for that week's eligible tickers before computing context, so adjustments are consistent across the window. Backfill by hand: `python -m jobs.collect_prices --days 120`. Phase 7's backtest should re-download full adjusted history rather than trust older stored rows across a split.
  - Context (`market/context.py`) uses only bars dated on or before the week's Friday: `ret_5d` = 5 sessions, `ret_30d` = 21 sessions, `rel_volume` = this week's mean volume / mean of the 30 calendar days before the week, `updown_vol_ratio` = up-day / down-day volume (NULL with no down-days).
  - `early_chatter_flag`: top decile **by rank** of `composite_bull` or `composite_bear` (a quantile threshold breaks on ties), `|ret_5d|` < 3%, `rel_volume` > 1.2. Without price data the flag is false.
  - Divergence warning (UI only): bullish sentiment, `rel_volume` > 1.2 and `ret_5d` below -2%. Shown as a card on the rankings page and a badge on the drill-down.
  - Context is attached after the composite is computed and never feeds it.
- **Phase 6 (more sources, noise and bot filtering): done 2026-10-04.**
  - **News (Finnhub):** free personal-use key (`FINNHUB_API_KEY`, 60 calls/min). One request per ticker per run, for the most-mentioned tickers in the last 7 days topped up with the most liquid universe names (max 50), so news collects even before Reddit approval. Each outlet counts as one author; news has no engagement, so it gets an engagement floor (config).
  - **StockTwits:** new developer registrations were closed on 2026-10-04 and its terms prohibit scraping, so the collector uses the official API with `STOCKTWITS_ACCESS_TOKEN` and is **disabled** in config until access exists. Author Bullish/Bearish tags are stored as `post_scores` model `stocktwits_tag`; `python -m analysis.scorer_check` measures FinBERT's agreement with them. `author_karma` holds follower count for StockTwits.
  - **Source tickers:** collectors can name the tickers an item is about (`Post.source_tickers`); these become `post_tickers` rows with `match_type = 'source'` (confidence 0.9). They survive extractor rebuilds and win over extractor matches for the same ticker; the extractor still runs on those posts to find other tickers.
  - **Daily job:** each source starts independently, so missing credentials for one only fail that source (exit 2), not the run.
  - **Noise filtering** (`signals/filters.py`, config `signals.noise`): exact reposts, near-duplicates (64-bit SimHash over word shingles with banding; posts under 12 words only exact-matched), spam regexes, per-source author down-weighting for new or low-karma accounts (missing data isn't penalised; weight 0 drops), hyperactive-author down-weighting (> 40 posts/week), and the per-author cap. Reddit bots (AutoModerator, VisualMod) are dropped at collection.
  - **Tests** block all non-local network connections (autouse fixture in `tests/conftest.py`).
- **Phase 7 (lead-lag analysis and backtest): done 2026-10-04.** Built and tested on synthetic panels with known answers; needs real weeks before it means anything.
  - Panel (`analysis/leadlag.py`): every `weekly_signals` row plus `ret_prior`, `ret_same` and `ret_next`, from the close on the last session on or before each Friday (holiday-safe).
  - Lead-lag: per-week cross-sectional Spearman IC (rank correlation; no scipy needed) for each signal against each horizon, averaged with a t-stat. `verdict()` turns it into plain language: leads price, follows price, contrarian, or no reliable relationship (|t| >= 2 required).
  - Backtest (`analysis/backtest.py`): bull/bear composite IC vs next week, top-minus-bottom quintile spread (>= 5 tickers/week), hit rates of both lists vs a mentions-only baseline (hit = beat / trail that week's median), and the rel_volume experiment (composite + 0.15 * z(rel_volume); promote rel_volume into the composite only if this wins on real data).
  - `python -m jobs.run_analysis [--refresh-prices]` prints the report and saves `data/reports/analysis_<date>.json`. Use `--refresh-prices` to re-download full adjusted history before trusting results.
  - App page 3, "Validation": data-sufficiency bar (`analysis.min_weeks`, default 12), plain-language readings, IC heatmap, weekly IC bars, cumulative quintile spread, and hit-rate and experiment cards.
  - Known caveat: re-ranking an old week uses engagement counts refreshed up to 36h after posting, a small lookahead in post weights only.
- **Phase 8 (LLM second pass, insider flag, scheduling): done 2026-10-04.**
  - **Claude second pass** (`scoring/llm_second_pass.py`): `claude-haiku-4-5` (the spec's "small, cheap" model) with structured JSON output (`output_config.format`). It re-scores FinBERT calls with |score| < 0.3 plus each ticker's top-5 most-engaged posts per week (engagement rank computed before excluding already-scored pairs), over the last 14 days, capped at 2,000 items per run. Uses the Message Batches API by default (half price). Stored as model `claude`; never overwrites FinBERT. **Off by default** (`llm_second_pass.enabled`), needs `ANTHROPIC_API_KEY`. A failure is logged and ranking continues on FinBERT.
  - **Score priority:** `signals.sentiment_models: [claude, finbert]`. Aggregation and the app use the first available score per (post, ticker) via `db.EFFECTIVE_SCORES`. Mixing scorers means scores aren't on exactly the same scale; compare both against StockTwits tags with `analysis.scorer_check --model claude`.
  - **Insider flag** (`market/insiders.py`): SEC EDGAR company tickers → submissions → Form 4 XML; counts open-market purchases (code P, acquired). Flag = purchase filed in the 30 days up to the week's Friday (by filing date, no lookahead). Filings are cached in `insider_filings`. Needs `SEC_USER_AGENT` (name + email, SEC fair-access rule); otherwise the flag stays NULL. Shown as a column and a drill-down badge; never feeds the composite.
  - **Scheduling:** `scripts/run_daily.ps1`, `scripts/run_weekly.ps1` (universe → rank → analysis), and `scripts/register_tasks.ps1` (Task Scheduler: daily 07:00, Saturday 10:00 local, run while logged in, catch up if missed; `-Remove` to undo).
  - Deployment is local: scheduled jobs plus `streamlit run` on demand. README.md covers setup.
- **Direction change (2026-10-05): news-first, free or minimal cost.** Reddit rejected the Data API request (and is ending RSS on 2026-11-13 and its public API by March 2027), so `reddit.enabled` is false. X's API is pay-per-use ($0.005 per post read, roughly $150-750+/month for useful coverage), so it's out unless free sources show a real signal. Sentimental becomes a news-flow and tone screen: attention = coverage vs the ticker's baseline, sentiment = news tone, breadth = distinct outlets. Planned next: wider Finnhub coverage, news cleanup (syndicated copies, company press releases, "why the stock moved" price-recap articles that only echo past returns, content-farm listicles, outlet weights), then SEC 8-K filings and GDELT (check terms first), then optional event tagging. Bluesky and StockTwits stay optional social add-ons. The Reddit collector code stays in place.
- **News cleanup and wider coverage (2026-10-05):**
  - Finnhub now covers up to 1,000 tickers a day (most mentioned, then most liquid). Requests are paced to 55/min (start-to-start, not a fixed pause after each call, which nearly doubled run time to ~32 min in the first live run); expect ~18 min. Free company news is concentrated on large caps: the first 1,000-ticker run returned 558 articles. Up to 5% of tickers may fail without the run counting as partial (`news.max_failure_share`).
  - Finnhub attributes about 90% of articles to the aggregator "Yahoo" and gives only finnhub.io redirect URLs, so the real publisher is unknown. Outlet weights aren't possible, and breadth (distinct outlets) is weak for news.
  - `signals/news_quality.py` classifies each headline: `price_recap` (dropped entirely: it narrates past moves and would make news tone echo returns), `promo` (listicles and buy/sell pitches, x0.3), `press_release` (x0.5), `news`. Forecasts ("could surge 200%") aren't recaps; pitches that quote a past move count as promo. Tested on invented headlines in `fixtures/news_headlines.yaml`. On the first 307 real articles: about 4% recaps, 24% promo.
  - Off-topic tags (the story never names the ticker or company, by symbol, cashtag, company name or alias) get x0.3. About half of Finnhub's ticker tags were off-topic on day one.
  - Identical headlines across outlets collapse to the earliest copy. The per-author cap and the distinct-authors minimum now apply only to social sources (`author_cap_sources`); eligibility is 5 mentions/week and 1 author for news-first.
  - The drill-down post list labels each news item's kind (recap shown as excluded).
- **SEC 8-K events (2026-10-05):** `market/sec_filings.py`, `jobs/collect_filings.py`. Event context, **not** a ranking input (8-Ks are company self-disclosure, and changing the composite is the user's call).
  - The EDGAR submissions JSON lists each 8-K's item codes and exact acceptance time, so there's no need to download documents: one request per company, shared with Form 4. It runs daily in `collect_daily` (config `sec`: 1,000 tickers, 10-day lookback); without `SEC_USER_AGENT` it's skipped with a warning.
  - Categories: Earnings (2.02), Deal, Leadership (5.02), Financing, Other, and **Red flag** (1.03 bankruptcy, 1.05 cyber incident, 2.04 debt trigger, 2.05 restructuring, 2.06 impairment, 3.01 delisting notice, 4.01 auditor change, 4.02 non-reliance/restatement). `weekly_signals.events` names the red-flag items; `red_flag` is a boolean. Weekly windows use SEC acceptance time (a Friday filing accepted after Saturday 00:00 UTC belongs to the next week).
  - New columns are added with `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` in schema.sql, so existing databases migrate on the next job run.
  - App: "8-K events" and "Red flag" columns, a red-flag card on the rankings page, "EARNINGS WEEK" and "RED-FLAG 8-K" badges, and a filings list linking to EDGAR on the drill-down.
  - First live backfill (10 days, 1,000 tickers): 192 8-Ks, 1,210 Form 4s, about 30 minutes; daily runs only fetch new filings. Possible refinements: count only officer/director purchases for the insider flag (10%-owner funds also file code P), and treat 2.05 restructuring announced with earnings as less severe.
- **GDELT news (2026-10-05):** `collectors/gdelt.py`. GDELT's DOC search API returned HTTP 429 to this connection even after minutes idle (per-IP throttling), so Sentimental reads the bulk 15-minute GKG files from data.gdeltproject.org instead (free for any use, **must cite GDELT**: done in the app footer and README).
  - Each file is about 2-6 MB zipped (roughly 300-500 MB/day). Processed files are tracked in `gdelt_files`; files not yet published (GDELT can lag hours) are retried for 48 hours, and the first run only takes the normal 36-hour window. `files_per_hour` (1-4) samples to save bandwidth.
  - Matching requires **both** GDELT's organization list naming the company (full cleaned name or alias, never first-word shortcuts) **and** the headline naming it. Org-only matching was 86% incidental in a live sample ("Facebook", "the Nasdaq", "The New York Times" cited as a source). `gdelt.exclude_orgs` drops index or exchange names. Posts use source `news` (so news cleanup applies) with the publisher domain as `community`, which makes breadth meaningful.
  - First live sample (6 hours): 172 confirmed articles from 116 outlets across 50 tickers (about 700/day).

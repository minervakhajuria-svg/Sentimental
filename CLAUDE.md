# Stock Sentiment Analyser: Project Spec

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
`post_id, ticker, match_type (cashtag|alias|bare), confidence`

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

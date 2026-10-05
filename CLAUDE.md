# Sentimental: Project Spec

Read this first each session. It describes the system as built and the decisions behind it; keep it current when decisions change.

## 1. Purpose
A personal research tool that tracks **news coverage and tone** for US-listed stocks and produces a **weekly ranked list** of tickers where attention and sentiment are shifting, ideally *before* price reacts.

It is a **screening aid**, not a trading signal, and must never output buy/sell advice. The ranking uses **attention and sentiment only**. Price, volume, insider trades and SEC filings are context, never ranking inputs, unless validation proves otherwise and the user decides to change the formula.

## 2. Constraints and decisions
- **Market:** US equities (NYSE, NYSE American, Nasdaq).
- **Cadence:** collect **daily**; score and rank **weekly**, on the weekend after Friday's US close. The user does one weekly review; no intraday logic.
- **Cost: free or minimal.** Everything runs at $0. The only paid piece (Claude second pass) is optional and off by default.
- **News-first (since 2026-10-05).** The original Reddit-first design was dropped:
  - Reddit rejected the Data API request; Reddit is also ending RSS on 2026-11-13 and its public API by March 2027.
  - X's API is pay-per-use ($0.005 per post read, roughly $150-750+/month for useful coverage): out unless the free version shows a real signal.
  - StockTwits isn't accepting new developers, and its terms forbid scraping.
  - Bluesky (open API) is an optional social add-on, not built.
- **Respect terms and rate limits; never scrape sites that forbid it.**
- **Data licences:** Finnhub's free tier and Yahoo prices (yfinance) are **personal use**: never display them publicly. GDELT is free for any use **with citation**; SEC data is public. A public deployment would need GDELT and SEC only, plus a check of local rules on publishing stock lists (e.g. SEBI in India).
- **Secrets** live only in `.env` (git-ignored); never log, print or commit them.

## 3. Tech stack
Python 3.14 (works on 3.11+) in `.venv`; DuckDB (`data/sentiment.duckdb`, schema kept Postgres-portable); FinBERT (`ProsusAI/finbert`) via transformers/torch on CPU; yfinance behind a `MarketDataProvider` interface; Streamlit + Altair UI; optional `anthropic` SDK (Claude Haiku 4.5); Windows Task Scheduler; `config.yaml` (all thresholds) + `.env` (secrets); pytest with saved fixtures.

## 4. Repository layout
```
config.yaml  .env.example  pyproject.toml  README.md  CLAUDE.md
.streamlit/config.toml     theme (green on black)
.claude/launch.json        preview configs: sentimental (real data), streamlit-demo (demo DB)
scripts/                   run_daily.ps1, run_weekly.ps1, register_tasks.ps1
fixtures/                  invented samples for tests (edgar/, gdelt/, headlines, API responses)
data/  logs/               git-ignored
src/
  collectors/   base.py (Post, Collector), http.py, news.py (Finnhub), gdelt.py,
                reddit.py, stocktwits.py (both disabled)
  storage/      db.py, schema.sql
  tickers/      universe.py, extractor.py, aliases.csv, blocklist.txt
  scoring/      finbert.py, context.py, llm_second_pass.py
  signals/      aggregate.py, filters.py, news_quality.py, composite.py
  market/       prices.py, context.py, insiders.py (EDGAR client + Form 4), sec_filings.py (8-K)
  analysis/     leadlag.py, backtest.py, scorer_check.py
  jobs/         collect_daily, collect_prices, collect_filings, extract_tickers, score_posts,
                build_universe, rank_weekly, run_analysis
  app/          streamlit_app.py (nav + footer), views.py (pages), queries.py (read-only), theme.py
tests/          one file per area; demo_data.py builds the synthetic demo DB
```
Packages under `src/` are top-level imports (`collectors`, `signals`, ...); the project is installed editable (`pip install -e ".[dev]"`).

## 5. Data model (DuckDB)
- **posts**: one row per article or post. `id` = `{source}:{native_id}` (`news:finnhub:<id>`, `news:gdelt:<sha1(url)>`, `reddit:...`). Fields: source, community (outlet domain for GDELT, "Yahoo"/etc. for Finnhub), author_id (hashed; for news the outlet), author_age_days, author_karma, created_at/collected_at (UTC), title, body (raw, kept for re-scoring), url, engagement (NULL for news), content_hash. Upserts refresh engagement and author details only; text and first-seen times are never overwritten.
- **post_tickers**: (post_id, ticker) → match_type `cashtag | alias | bare | source`, confidence.
- **post_scores**: (post_id, ticker, model) → label pos/neg/neu, score -1..1. Models: `finbert`, `claude` (optional), `stocktwits_tag`. Inserted, never overwritten; also the scoring cache.
- **prices_daily**: adjusted OHLCV per (ticker, date), upserted with overwrite.
- **ticker_universe**: all listings with market cap and 30-day dollar volume; floors applied at load time.
- **weekly_signals**: (week_start = Monday, ticker) → mentions, attention_z, sentiment, sentiment_prev, momentum, breadth, composite_bull, composite_bear, ret_5d, ret_30d, rel_volume, updown_vol_ratio, insider_buy_flag, early_chatter_flag, events, red_flag.
- **insider_filings** (parsed Form 4s), **sec_filings** (8-Ks with item codes and acceptance time), **gdelt_files** (processed 15-minute files).
- Schema changes after release use `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` in schema.sql, so existing databases migrate on the next job run.

## 6. Sources
| Source | Status | Notes |
|---|---|---|
| **Finnhub company news** | On (`FINNHUB_API_KEY`) | Up to 1,000 tickers/day (most mentioned, then most liquid), paced to 55 req/min start-to-start (~18 min). ~90% of articles are attributed to aggregator "Yahoo" with finnhub.io redirect URLs, so the publisher is unknown. Finnhub's `related` field becomes `source` ticker tags. Up to 5% failed tickers is tolerated. |
| **GDELT GKG bulk files** | On (no key) | The DOC search API returned 429 to this connection, so it reads the free 15-minute GKG files (~300-500 MB/day). A match needs **both** GDELT's org list naming the company (full cleaned name or alias, never first-word shortcuts) **and** the headline naming it (org-only matches were 86% incidental). `exclude_orgs` drops index/exchange names. Real publisher domain → meaningful breadth. Processed files tracked; late files retried 48h. ~700 confirmed articles/day. **Cite GDELT** (app footer, README). |
| **SEC EDGAR** | On (`SEC_USER_AGENT` = name + email) | One submissions request per company (cached per run), shared by 8-K and Form 4. Under 10 req/s. Skipped with a warning if the user agent is missing. |
| **Prices (yfinance)** | On | Split/dividend-adjusted. Daily: last 10 days for tickers mentioned in 30 days; weekly: 60 days for eligible tickers. |
| Reddit, StockTwits | Built, disabled | Kept for if access ever opens. |

The daily job starts each source independently: missing credentials fail only that source (exit 2 = partial), never the run.

## 7. Ticker matching
- **Universe:** Nasdaq Trader symbol directory (common equity only; ETFs, warrants, units, rights, preferreds excluded; LP common units kept). Share classes use a dot (`BRK.B`; Yahoo dash form only inside `market/prices.py`). Floors: market cap ≥ $300M, 30-day dollar volume ≥ $5M (~2,960 tickers). Refresh weekly; a refresh with < 50% market-data coverage is refused.
- **Extractor** (all posts): cashtag 0.95, alias 0.75 (curated `aliases.csv`, case-sensitive flag for common words), bare 0.5 (needs finance context, not blocklisted, not on "shouting" lines). URLs and r/ / u/ references stripped first.
- **Source tags** (0.9): tickers named by the source (Finnhub `related`, GDELT matches). They survive extractor rebuilds and win over extractor matches; the extractor still runs on those posts.

## 8. Scoring
- **FinBERT** on every (post, ticker) pair: score = p_pos − p_neg. Multi-ticker posts use the sentences mentioning each ticker. Runs at the start of `rank_weekly`.
- **Claude second pass** (optional, `llm_second_pass.enabled`, `ANTHROPIC_API_KEY`): `claude-haiku-4-5`, structured JSON output, Message Batches API (half price); re-scores |FinBERT| < 0.3 plus each ticker's top-5 posts per week; ≤ 2,000 items/run. Failure → ranking continues on FinBERT.
- **Score priority:** `signals.sentiment_models: [claude, finbert]` (first available per pair, via `db.EFFECTIVE_SCORES`).

## 9. Filtering and weighting (before aggregation)
1. Exact reposts (content hash) and near-duplicates (SimHash; posts < 12 words exact-only) → keep earliest.
2. Spam regexes (config).
3. **News quality** (`signals/news_quality.py`, headline-based): `price_recap` → **dropped** (narrates past moves; would make tone echo returns); `promo` (listicles, buy/sell pitches) ×0.3; `press_release` ×0.5; forecasts aren't recaps; pitches quoting a past move are promo. Identical headlines across outlets collapse. Off-topic tags (story never names the ticker/company/alias/cashtag) ×0.3.
4. Post weight = log(1 + engagement) × author_weight × quality × source_weight; news gets an engagement floor of 10.
5. Social-only rules (`author_cap_sources: [reddit, stocktwits]`): ≤ 5 posts per author per ticker per week; new/low-karma and hyperactive accounts down-weighted. News "authors" are outlets and are not capped.

## 10. Weekly signals
- **Week** = Saturday 00:00 UTC → next Saturday 00:00 UTC, keyed by its Monday. Only data before the cutoff is used.
- **Attention (A):** mean daily weighted mentions vs the ticker's prior 30-day baseline, as a z-score (std floored at 1.0, clipped ±10).
- **Sentiment (S):** weighted mean score; **Momentum (M):** S − last week's S (NULL → contributes 0); **Breadth (B):** log-scaled distinct authors (outlets), communities and sources.
- **Eligibility (news-first):** ≥ 5 mentions/week, ≥ 1 author, inside the universe floors.
- **Composite:** cross-sectional z-scores; bull = 0.35A + 0.35S + 0.15M + 0.15B; bear = 0.35A − 0.35S − 0.15M + 0.15B (weights in config).
- **Lists:** bullish = top N by bull with S > 0 and A > 0; bearish = top N by bear with S < 0 and A > 0. Stored in `weekly_signals`; the lists are queries over it.

## 11. Context columns (display only, attached after the composite)
- `ret_5d` (5 sessions), `ret_30d` (21 sessions), `rel_volume` (week mean / prior 30 days), `updown_vol_ratio`, from bars dated ≤ the week's Friday.
- `early_chatter_flag`: top decile **by rank** of bull or bear composite, |ret_5d| < 3%, rel_volume > 1.2.
- **Divergence warning** (UI): bullish sentiment, rel_volume > 1.2, ret_5d < −2%.
- `insider_buy_flag`: an open-market purchase (Form 4 code P) filed in the 30 days to Friday. Note: 10%-owner funds also file code P.
- `events` / `red_flag` from 8-Ks accepted during the week (by SEC acceptance time): Earnings (2.02), Deal, Leadership (5.02), Financing, Other; **red flag** items (bankruptcy, cyber incident, debt trigger, restructuring, impairment, delisting notice, auditor change, non-reliance/restatement) are named in `events`. "Earnings week" explains scheduled attention spikes.

## 12. Validation (`analysis/`, app page "Validation")
Panel of weekly signals with prior/same/next-week returns from Friday closes (holiday-safe). Per-week Spearman IC with t-stats per signal × horizon; plain-language verdict (leads, follows, contrarian, none; |t| ≥ 2). Backtest: composite IC, top-minus-bottom quintile spread, list hit rates vs a mentions-only baseline, and the rel_volume experiment (promote into the composite only if it wins on real data). Needs ≥ 12 weeks (`analysis.min_weeks`). Run `python -m jobs.run_analysis --refresh-prices` (re-downloads full adjusted history first). Caveat: re-ranking old weeks uses engagement refreshed up to 36h after posting.

## 13. App (`streamlit run src/app/streamlit_app.py`, from the project folder)
- Pages: **Weekly rankings** (ticker tape, stat cards, both lists with context columns, divergence and red-flag cards; click a row to drill down), **Ticker drill-down** (badges: early chatter, insider buy, earnings week, red-flag 8-K; weekly score gauge; mention mix; by-source bars; mentions/sentiment chart; price/volume chart; weekly history; SEC filings list; top posts labelled by news kind), **Validation**.
- Read-only, short-lived DuckDB connections per render (shows "busy" if a job holds the write lock). `SENTIMENT_DB_PATH` overrides the database (e.g. `data/demo.duckdb`).
- Look: name **Sentimental**, fluorescent green on black from the user's design canvas (bg `#0A0A0A`, cards `#111314`/`#1F2421`, accent `#39FF14` with glow, bearish `#FF4D4D`, muted `#8A8F98`; Space Grotesk + JetBrains Mono). Theme in `.streamlit/config.toml`, the rest in `app/theme.py`; containers keyed `card-*` get card styling. All external text in HTML goes through `html.escape`.
- Footer: "Screening aid only, not investment advice" + GDELT/Finnhub/SEC attribution.
- Demo: `python tests/demo_data.py` writes `data/demo.duckdb` (invented posts, prices, 8-Ks).

## 14. Operations
- **Keys in `.env`:** `FINNHUB_API_KEY` (set), `SEC_USER_AGENT` (set), optional `ANTHROPIC_API_KEY`; Reddit/StockTwits blank.
- **Scheduled tasks** (registered 2026-10-04, run while logged in, catch up if missed): `Sentimental-Daily` 07:00 (`run_daily.ps1` → collect_daily: Finnhub, GDELT, ticker tagging, SEC filings, prices; ~30 min) and `Sentimental-Weekly` Saturday 10:00 (`run_weekly.ps1` → build_universe, rank_weekly incl. FinBERT, run_analysis). Remove with `register_tasks.ps1 -Remove`. On 2026-10-05 the 07:00 run was missed (PC likely off); consider an evening time.
- **Exit codes:** 0 ok, 2 partial (data kept), 1 failed. Logs in `logs/<job>.log`.
- Manual: `python -m jobs.collect_filings [--days N]`, `jobs.collect_prices [--days N]`, `jobs.extract_tickers --rebuild` (after editing aliases/blocklist), `jobs.rank_weekly [--week-ending YYYY-MM-DD] [--no-score]`, `analysis.scorer_check`.
- Run commands in **PowerShell** opened in the project folder (not the Python `>>>` prompt).

## 15. Status (2026-10-05) and next steps
- All 8 original phases built, then pivoted to news-first with news cleanup, wider Finnhub coverage, SEC 8-K events and GDELT. 239 tests pass. Everything is pushed to GitHub (public repo `minervakhajuria-svg/Sentimental`).
- Real data so far: ~560 Finnhub articles/day, ~700 GDELT/day, 8-Ks and Form 4s for 1,000 tickers. **First weekly ranking: Saturday 2026-10-10.** The first month is noisy (empty attention baseline); validation needs ~12 weeks.
- **Deferred by the user:** event tagging (keyword rules or the Claude pass); a public live demo on Streamlit Community Cloud with the invented demo data (free, licence-safe); remote access (Tailscale now, or a private Streamlit Cloud app fed by a daily snapshot).
- **Ideas:** count only officer/director purchases for the insider flag; treat 2.05 restructuring announced with earnings as a softer red flag; tune thresholds and weights after a few real weeks; Bluesky as a social source; auto-start the app at login.

## 16. Working conventions
- Build one area at a time with fixture tests. **Tests never touch the network** (autouse guard in `tests/conftest.py`); fixtures are **invented** content in real formats (don't commit real publishers' headlines).
- Check each new source's current terms and rate limits (web search) before building, and inspect real responses before designing parsers or rules.
- Keep thresholds and weights in `config.yaml`. Secrets only in `.env`.
- Jobs log to file, fail loudly (non-zero exit), never corrupt data (transactions, idempotent upserts).
- Docstrings explain *why*. Simple, readable code.
- Commit after each piece of work; **push only when the user asks** (they usually say "push and continue").
- Don't add features beyond the agreed step without asking; changes to the ranking formula are the user's decision.
- Tooling tip: on this Windows machine, multi-line edit scripts with apostrophes or backslashes break in bash heredocs; write them to a file in the scratchpad and run them, or use the Edit tool.

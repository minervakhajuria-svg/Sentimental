# Sentimental

A personal research tool that scans online discussion about US-listed stocks and produces a **weekly ranked list** of tickers where attention and sentiment are shifting.

**Screening aid only, not investment advice.** The ranking uses sentiment and attention only; price, volume and insider data are shown as context and never feed the score.

## How it works

1. **Collect daily:** Reddit posts (six investing subreddits), company news (Finnhub), and StockTwits when access is available. Each post is tagged with the tickers it mentions.
2. **Score:** FinBERT runs locally on every (post, ticker) pair. An optional Claude second pass re-reads ambiguous and high-engagement posts, which helps with sarcasm and options slang.
3. **Rank weekly:** attention vs each ticker's own baseline, sentiment, momentum and breadth, after removing reposts, spam and bot-like accounts. The output is two lists: *heating up, bullish* and *heating up, bearish*.
4. **Context:** 5- and 30-day returns, relative volume, an early-chatter flag (talk rising before price moves), and recent insider purchases from SEC Form 4 filings.
5. **Validate:** a lead-lag analysis and backtest check whether chatter actually leads price. It needs about 12 weeks of data before results mean anything.

## Setup (Windows)

```powershell
py -3 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
copy .env.example .env   # then fill in the keys you have
```

| Key in `.env` | Needed for | Notes |
|---|---|---|
| `REDDIT_CLIENT_ID` / `_SECRET` / `_USER_AGENT` | Reddit | New apps need approval under Reddit's Responsible Builder Policy |
| `FINNHUB_API_KEY` | News | Free personal-use key at finnhub.io |
| `SEC_USER_AGENT` | Insider flag | Your name and email; the SEC asks every client to identify itself |
| `ANTHROPIC_API_KEY` | Claude second pass (optional, paid) | Also set `llm_second_pass.enabled: true` in `config.yaml` |
| `STOCKTWITS_ACCESS_TOKEN` | StockTwits | Registrations currently closed; disabled in config |

Every source is optional on its own: missing credentials skip that source, not the run.

## Running

```powershell
.venv\Scripts\python -m jobs.build_universe      # weekly: which tickers to track (~25 min)
.venv\Scripts\python -m jobs.collect_daily       # daily: collect, tag, prices
.venv\Scripts\python -m jobs.rank_weekly         # weekend: score and rank the week
.venv\Scripts\python -m jobs.run_analysis --refresh-prices   # lead-lag and backtest report
.venv\Scripts\python -m streamlit run src/app/streamlit_app.py
```

To schedule the daily and weekly jobs with Windows Task Scheduler: `.\scripts\register_tasks.ps1` (remove them with `-Remove`).

To preview the app without collected data: `python tests/demo_data.py`, then set `SENTIMENT_DB_PATH=data/demo.duckdb` before starting Streamlit. All demo posts are invented.

## Development

`pytest` runs the suite against saved fixtures; tests block all network access. Settings and thresholds live in `config.yaml`. The full spec and the design decisions are in `CLAUDE.md`.

-- DuckDB schema. Kept to plain SQL types so it ports to Postgres unchanged.
-- All timestamps are UTC.

CREATE TABLE IF NOT EXISTS posts (
    id              TEXT PRIMARY KEY,      -- {source}:{native_id}
    source          TEXT NOT NULL,         -- reddit, stocktwits, news
    community       TEXT,                  -- subreddit / feed name
    author_id       TEXT,                  -- hashed username
    author_age_days INTEGER,
    author_karma    INTEGER,
    created_at      TIMESTAMP NOT NULL,
    collected_at    TIMESTAMP NOT NULL,
    title           TEXT,
    body            TEXT,                  -- raw text, so it can be re-scored later
    url             TEXT,
    engagement      INTEGER,               -- source-specific; normalised later
    content_hash    TEXT
);

CREATE TABLE IF NOT EXISTS post_tickers (
    post_id    TEXT NOT NULL,
    ticker     TEXT NOT NULL,
    match_type TEXT NOT NULL,              -- cashtag | alias | bare | source
    confidence DOUBLE,
    PRIMARY KEY (post_id, ticker)
);

CREATE TABLE IF NOT EXISTS post_scores (
    post_id   TEXT NOT NULL,
    ticker    TEXT NOT NULL,
    model     TEXT NOT NULL,
    label     TEXT,                        -- pos | neg | neu
    score     DOUBLE,                      -- -1..1
    scored_at TIMESTAMP,
    PRIMARY KEY (post_id, ticker, model)
);

CREATE TABLE IF NOT EXISTS prices_daily (
    ticker TEXT NOT NULL,
    date   DATE NOT NULL,
    open   DOUBLE,
    high   DOUBLE,
    low    DOUBLE,
    close  DOUBLE,
    volume BIGINT,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS ticker_universe (
    ticker                TEXT PRIMARY KEY,
    company_name          TEXT,
    exchange              TEXT,
    market_cap            DOUBLE,
    avg_dollar_volume_30d DOUBLE,
    updated_at            TIMESTAMP
);

CREATE TABLE IF NOT EXISTS weekly_signals (
    week_start         DATE NOT NULL,
    ticker             TEXT NOT NULL,
    mentions           INTEGER,
    attention_z        DOUBLE,
    sentiment          DOUBLE,
    sentiment_prev     DOUBLE,
    momentum           DOUBLE,
    breadth            DOUBLE,
    composite_bull     DOUBLE,
    composite_bear     DOUBLE,
    ret_5d             DOUBLE,
    ret_30d            DOUBLE,
    rel_volume         DOUBLE,
    updown_vol_ratio   DOUBLE,
    insider_buy_flag   BOOLEAN,
    early_chatter_flag BOOLEAN,
    PRIMARY KEY (week_start, ticker)
);

-- Parsed SEC Form 4 filings (cache, so each filing is fetched once).
CREATE TABLE IF NOT EXISTS insider_filings (
    accession      TEXT PRIMARY KEY,
    ticker         TEXT NOT NULL,
    filing_date    DATE NOT NULL,
    purchase_count INTEGER,          -- open-market purchases (code P, acquired)
    purchase_value DOUBLE,           -- shares * price, USD
    fetched_at     TIMESTAMP
);

-- SEC 8-K filings (event context; never a ranking input).
CREATE TABLE IF NOT EXISTS sec_filings (
    accession   TEXT PRIMARY KEY,
    ticker      TEXT NOT NULL,
    cik         INTEGER,
    form        TEXT NOT NULL,           -- 8-K or 8-K/A
    filing_date DATE NOT NULL,
    accepted_at TIMESTAMP,               -- SEC acceptance time, UTC
    items       TEXT,                    -- e.g. "2.02,9.01"
    url         TEXT,                    -- EDGAR filing index page
    fetched_at  TIMESTAMP
);

-- Columns added after the first release. ADD COLUMN IF NOT EXISTS keeps
-- existing databases working without a manual migration.
ALTER TABLE weekly_signals ADD COLUMN IF NOT EXISTS events TEXT;      -- 8-K categories that week
ALTER TABLE weekly_signals ADD COLUMN IF NOT EXISTS red_flag BOOLEAN; -- a red-flag 8-K that week

-- GDELT 15-minute GKG files already processed (or found missing), so each is
-- downloaded once and late files are retried.
CREATE TABLE IF NOT EXISTS gdelt_files (
    stamp      TEXT PRIMARY KEY,         -- YYYYMMDDHHMMSS (UTC)
    status     TEXT NOT NULL,            -- done | missing | error
    rows       INTEGER,
    matched    INTEGER,
    checked_at TIMESTAMP
);

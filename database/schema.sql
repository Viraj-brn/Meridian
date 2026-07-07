-- ─────────────────────────────────────────────
-- Meridian — Database Schema
-- SQLite table definitions for the Index Tracker
-- ─────────────────────────────────────────────

-- Securities master table
CREATE TABLE IF NOT EXISTS securities (
    ticker TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    sector TEXT NOT NULL,
    market_cap REAL DEFAULT 0,
    shares_outstanding REAL DEFAULT 0,
    free_float_factor REAL DEFAULT 1.0
);

-- Daily OHLCV price data
CREATE TABLE IF NOT EXISTS daily_prices (
    ticker TEXT NOT NULL,
    trade_date DATE NOT NULL,
    open_price REAL,
    high_price REAL,
    low_price REAL,
    close_price REAL,
    adj_close REAL,
    volume INTEGER,
    PRIMARY KEY (ticker, trade_date),
    FOREIGN KEY (ticker) REFERENCES securities(ticker)
);

-- Index constituent membership snapshots
CREATE TABLE IF NOT EXISTS index_constituents (
    index_name TEXT NOT NULL,
    ticker TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    weight REAL NOT NULL,
    is_active BOOLEAN DEFAULT 1,
    PRIMARY KEY (index_name, ticker, as_of_date),
    FOREIGN KEY (ticker) REFERENCES securities(ticker)
);

-- Computed daily index levels
CREATE TABLE IF NOT EXISTS index_levels (
    index_name TEXT NOT NULL,
    trade_date DATE NOT NULL,
    index_level REAL NOT NULL,
    daily_return REAL DEFAULT 0,
    PRIMARY KEY (index_name, trade_date)
);

-- Rebalancing audit log
CREATE TABLE IF NOT EXISTS rebalance_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    index_name TEXT NOT NULL,
    rebalance_date DATE NOT NULL,
    ticker TEXT NOT NULL,
    action TEXT CHECK(action IN ('ADD', 'REMOVE', 'HOLD', 'ADJUST')),
    old_weight REAL,
    new_weight REAL
);

-- ─────────────────────────────────────────────
-- Performance indexes for query optimization
-- ─────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date
    ON daily_prices(ticker, trade_date);

CREATE INDEX IF NOT EXISTS idx_prices_date
    ON daily_prices(trade_date);

CREATE INDEX IF NOT EXISTS idx_constituents_index_date
    ON index_constituents(index_name, as_of_date);

CREATE INDEX IF NOT EXISTS idx_levels_index_date
    ON index_levels(index_name, trade_date);

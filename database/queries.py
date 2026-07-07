"""
Named SQL Queries — MSCI-style analytics queries.

Each query demonstrates a specific SQL pattern relevant to MSCI work:
window functions, CTEs, gap detection, anomaly detection, and ranking.
"""

# ─────────────────────────────────────────────
# Index Analytics Queries
# ─────────────────────────────────────────────

TOP_N_BY_WEIGHT = """
-- Top N constituents by current index weight
SELECT
    ic.ticker,
    s.name,
    s.sector,
    ic.weight,
    s.market_cap
FROM index_constituents ic
JOIN securities s ON ic.ticker = s.ticker
WHERE ic.index_name = ?
  AND ic.as_of_date = (
      SELECT MAX(as_of_date) FROM index_constituents WHERE index_name = ?
  )
ORDER BY ic.weight DESC
LIMIT ?;
"""

SECTOR_BREAKDOWN = """
-- Aggregate weights by sector
SELECT
    s.sector,
    COUNT(*) AS n_stocks,
    ROUND(SUM(ic.weight) * 100, 2) AS weight_pct,
    ROUND(AVG(s.market_cap), 0) AS avg_mcap
FROM index_constituents ic
JOIN securities s ON ic.ticker = s.ticker
WHERE ic.index_name = ?
  AND ic.as_of_date = (
      SELECT MAX(as_of_date) FROM index_constituents WHERE index_name = ?
  )
GROUP BY s.sector
ORDER BY weight_pct DESC;
"""

# ─────────────────────────────────────────────
# Time-Series Analytics
# ─────────────────────────────────────────────

HIGHEST_VOLATILITY = """
-- Stocks ranked by trailing 20-day volatility
-- Uses window function to compute rolling std of log returns
WITH daily_rets AS (
    SELECT
        ticker,
        trade_date,
        LN(adj_close / LAG(adj_close) OVER (
            PARTITION BY ticker ORDER BY trade_date
        )) AS log_return
    FROM daily_prices
    WHERE trade_date >= date(?, '-30 days')
      AND ticker IN (SELECT ticker FROM securities)
),
recent_vol AS (
    SELECT
        ticker,
        COUNT(*) AS n_days,
        ROUND(
            AVG(log_return) * 252, 4
        ) AS annualized_mean_return,
        -- SQLite doesn't have STDDEV in window, so we compute it
        -- For display, we use the Python-side computation
        ROUND(
            SQRT(AVG(log_return * log_return) - AVG(log_return) * AVG(log_return))
            * SQRT(252), 4
        ) AS annualized_vol
    FROM daily_rets
    WHERE log_return IS NOT NULL
    GROUP BY ticker
)
SELECT
    rv.ticker,
    s.name,
    s.sector,
    rv.annualized_vol,
    rv.annualized_mean_return,
    rv.n_days
FROM recent_vol rv
JOIN securities s ON rv.ticker = s.ticker
ORDER BY rv.annualized_vol DESC;
"""

ROLLING_RETURN_RANK = """
-- Rolling 30-day return ranking per stock
WITH prices_window AS (
    SELECT
        ticker,
        trade_date,
        adj_close,
        LAG(adj_close, 30) OVER (
            PARTITION BY ticker ORDER BY trade_date
        ) AS price_30d_ago
    FROM daily_prices
    WHERE trade_date >= date(?, '-60 days')
)
SELECT
    ticker,
    trade_date,
    ROUND((adj_close - price_30d_ago) / price_30d_ago * 100, 2) AS return_30d_pct,
    RANK() OVER (
        PARTITION BY trade_date
        ORDER BY (adj_close - price_30d_ago) / price_30d_ago DESC
    ) AS rank_by_return
FROM prices_window
WHERE price_30d_ago IS NOT NULL
  AND trade_date = (SELECT MAX(trade_date) FROM daily_prices)
ORDER BY rank_by_return;
"""

# ─────────────────────────────────────────────
# Data Quality Queries
# ─────────────────────────────────────────────

DATA_QUALITY_GAPS = """
-- Find gaps in daily price data (missing trading days)
WITH date_gaps AS (
    SELECT
        ticker,
        trade_date,
        LAG(trade_date) OVER (
            PARTITION BY ticker ORDER BY trade_date
        ) AS prev_date,
        JULIANDAY(trade_date) - JULIANDAY(
            LAG(trade_date) OVER (
                PARTITION BY ticker ORDER BY trade_date
            )
        ) AS gap_days
    FROM daily_prices
)
SELECT
    ticker,
    prev_date AS gap_start,
    trade_date AS gap_end,
    CAST(gap_days AS INTEGER) AS gap_days
FROM date_gaps
WHERE gap_days > 4  -- More than a long weekend
ORDER BY gap_days DESC
LIMIT 50;
"""

PRICE_ANOMALIES = """
-- Detect potential corporate actions: sudden price jumps > 20%
WITH price_changes AS (
    SELECT
        ticker,
        trade_date,
        adj_close AS close_price,
        LAG(adj_close) OVER (
            PARTITION BY ticker ORDER BY trade_date
        ) AS prev_close
    FROM daily_prices
)
SELECT
    pc.ticker,
    s.name,
    pc.trade_date,
    pc.close_price,
    pc.prev_close,
    ROUND(ABS(pc.close_price - pc.prev_close) / pc.prev_close * 100, 2)
        AS pct_change
FROM price_changes pc
JOIN securities s ON pc.ticker = s.ticker
WHERE pc.prev_close IS NOT NULL
  AND ABS(pc.close_price - pc.prev_close) / pc.prev_close > 0.20
ORDER BY pc.trade_date DESC;
"""

REBALANCE_CANDIDATES = """
-- Find stocks that might be added or removed at next rebalance
-- Based on market cap percentile rank
WITH ranked AS (
    SELECT
        s.ticker,
        s.name,
        s.sector,
        s.market_cap,
        PERCENT_RANK() OVER (ORDER BY s.market_cap DESC) AS mcap_percentile,
        CASE
            WHEN ic.ticker IS NOT NULL THEN 1 ELSE 0
        END AS currently_in_index
    FROM securities s
    LEFT JOIN (
        SELECT DISTINCT ticker
        FROM index_constituents
        WHERE index_name = ?
          AND as_of_date = (
              SELECT MAX(as_of_date) FROM index_constituents WHERE index_name = ?
          )
    ) ic ON s.ticker = ic.ticker
)
SELECT
    ticker, name, sector,
    ROUND(market_cap, 0) AS market_cap,
    ROUND(mcap_percentile, 4) AS mcap_pctl,
    currently_in_index,
    CASE
        WHEN currently_in_index = 0 AND mcap_percentile <= 0.85 THEN 'CANDIDATE_ADD'
        WHEN currently_in_index = 1 AND mcap_percentile > 0.90 THEN 'CANDIDATE_REMOVE'
        ELSE 'NO_ACTION'
    END AS rebalance_signal
FROM ranked
ORDER BY market_cap DESC;
"""

# ─────────────────────────────────────────────
# Summary Statistics
# ─────────────────────────────────────────────

DATABASE_SUMMARY = """
-- Quick summary of what's in the database
SELECT
    'Securities' AS entity,
    COUNT(*) AS count
FROM securities
UNION ALL
SELECT
    'Price Records',
    COUNT(*)
FROM daily_prices
UNION ALL
SELECT
    'Unique Trade Dates',
    COUNT(DISTINCT trade_date)
FROM daily_prices
UNION ALL
SELECT
    'Index Level Records',
    COUNT(*)
FROM index_levels;
"""

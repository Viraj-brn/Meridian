"""
Meridian — Configuration & Constants

Central configuration for the Market-Cap Weighted Index Tracker.
All tunable parameters live here so nothing is hardcoded in business logic.
"""

from datetime import date
from pathlib import Path

# ─────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_ROOT / "data" / "meridian.db"
CHARTS_DIR = PROJECT_ROOT / "web" / "static" / "charts"

# ─────────────────────────────────────────────
# Date Range
# ─────────────────────────────────────────────
START_DATE = date(2020, 1, 1)
END_DATE = date(2025, 6, 1)

# ─────────────────────────────────────────────
# Index Parameters
# ─────────────────────────────────────────────
INDEX_NAME = "MERIDIAN_50"
INDEX_BASE_LEVEL = 1000.0
REBALANCE_FREQUENCY_DAYS = 63  # ~quarterly (252 / 4)

# Buffer rules for rebalancing (hysteresis)
INCLUSION_PERCENTILE = 0.85   # Must be in top 85% by mcap to be added
EXCLUSION_PERCENTILE = 0.90   # Must fall below top 90% to be removed

# ─────────────────────────────────────────────
# Risk & Analytics Parameters
# ─────────────────────────────────────────────
RISK_FREE_RATE = 0.04          # 4% annualized (approx. current T-bill rate)
TRADING_DAYS_PER_YEAR = 252
ROLLING_WINDOW_SHORT = 20     # ~1 month
ROLLING_WINDOW_MEDIUM = 60    # ~3 months
ROLLING_WINDOW_LONG = 252     # ~1 year
VAR_CONFIDENCE = 0.95         # 95% VaR
EWMA_SPAN = 20               # Span for exponentially weighted calculations
N_PCA_FACTORS = 5             # Number of principal components to extract

# ─────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────
BENCHMARK_TICKER = "SPY"      # S&P 500 ETF as benchmark

# ─────────────────────────────────────────────
# Universe — 50 Stocks Across All 11 GICS Sectors
# ─────────────────────────────────────────────
UNIVERSE = {
    # ── Information Technology (10 stocks) ──
    "AAPL":  {"name": "Apple Inc.",              "sector": "Information Technology"},
    "MSFT":  {"name": "Microsoft Corp.",         "sector": "Information Technology"},
    "NVDA":  {"name": "NVIDIA Corp.",            "sector": "Information Technology"},
    "AVGO":  {"name": "Broadcom Inc.",           "sector": "Information Technology"},
    "ORCL":  {"name": "Oracle Corp.",            "sector": "Information Technology"},
    "CRM":   {"name": "Salesforce Inc.",         "sector": "Information Technology"},
    "AMD":   {"name": "Advanced Micro Devices",  "sector": "Information Technology"},
    "ADBE":  {"name": "Adobe Inc.",              "sector": "Information Technology"},
    "INTC":  {"name": "Intel Corp.",             "sector": "Information Technology"},
    "CSCO":  {"name": "Cisco Systems",           "sector": "Information Technology"},

    # ── Financials (6 stocks) ──
    "JPM":   {"name": "JPMorgan Chase & Co.",    "sector": "Financials"},
    "V":     {"name": "Visa Inc.",               "sector": "Financials"},
    "MA":    {"name": "Mastercard Inc.",          "sector": "Financials"},
    "BAC":   {"name": "Bank of America Corp.",   "sector": "Financials"},
    "GS":    {"name": "Goldman Sachs Group",     "sector": "Financials"},
    "BLK":   {"name": "BlackRock Inc.",          "sector": "Financials"},

    # ── Health Care (6 stocks) ──
    "UNH":   {"name": "UnitedHealth Group",      "sector": "Health Care"},
    "JNJ":   {"name": "Johnson & Johnson",       "sector": "Health Care"},
    "LLY":   {"name": "Eli Lilly & Co.",         "sector": "Health Care"},
    "PFE":   {"name": "Pfizer Inc.",             "sector": "Health Care"},
    "ABT":   {"name": "Abbott Laboratories",     "sector": "Health Care"},
    "MRK":   {"name": "Merck & Co.",             "sector": "Health Care"},

    # ── Consumer Discretionary (5 stocks) ──
    "AMZN":  {"name": "Amazon.com Inc.",         "sector": "Consumer Discretionary"},
    "TSLA":  {"name": "Tesla Inc.",              "sector": "Consumer Discretionary"},
    "HD":    {"name": "Home Depot Inc.",          "sector": "Consumer Discretionary"},
    "NKE":   {"name": "Nike Inc.",               "sector": "Consumer Discretionary"},
    "MCD":   {"name": "McDonald's Corp.",        "sector": "Consumer Discretionary"},

    # ── Communication Services (4 stocks) ──
    "META":  {"name": "Meta Platforms Inc.",      "sector": "Communication Services"},
    "GOOGL": {"name": "Alphabet Inc.",           "sector": "Communication Services"},
    "NFLX":  {"name": "Netflix Inc.",            "sector": "Communication Services"},
    "DIS":   {"name": "Walt Disney Co.",         "sector": "Communication Services"},

    # ── Consumer Staples (4 stocks) ──
    "PG":    {"name": "Procter & Gamble Co.",    "sector": "Consumer Staples"},
    "KO":    {"name": "Coca-Cola Co.",           "sector": "Consumer Staples"},
    "PEP":   {"name": "PepsiCo Inc.",            "sector": "Consumer Staples"},
    "COST":  {"name": "Costco Wholesale",        "sector": "Consumer Staples"},

    # ── Industrials (4 stocks) ──
    "CAT":   {"name": "Caterpillar Inc.",        "sector": "Industrials"},
    "BA":    {"name": "Boeing Co.",              "sector": "Industrials"},
    "UPS":   {"name": "United Parcel Service",   "sector": "Industrials"},
    "HON":   {"name": "Honeywell International", "sector": "Industrials"},

    # ── Energy (3 stocks) ──
    "XOM":   {"name": "Exxon Mobil Corp.",       "sector": "Energy"},
    "CVX":   {"name": "Chevron Corp.",           "sector": "Energy"},
    "COP":   {"name": "ConocoPhillips",          "sector": "Energy"},

    # ── Utilities (3 stocks) ──
    "NEE":   {"name": "NextEra Energy Inc.",     "sector": "Utilities"},
    "DUK":   {"name": "Duke Energy Corp.",       "sector": "Utilities"},
    "SO":    {"name": "Southern Co.",            "sector": "Utilities"},

    # ── Real Estate (3 stocks) ──
    "PLD":   {"name": "Prologis Inc.",           "sector": "Real Estate"},
    "AMT":   {"name": "American Tower Corp.",    "sector": "Real Estate"},
    "EQIX":  {"name": "Equinix Inc.",            "sector": "Real Estate"},

    # ── Materials (2 stocks) ──
    "LIN":   {"name": "Linde plc",               "sector": "Materials"},
    "APD":   {"name": "Air Products & Chemicals", "sector": "Materials"},
}

# Derived
TICKERS = list(UNIVERSE.keys())
N_STOCKS = len(TICKERS)

# ─────────────────────────────────────────────
# Flask Configuration
# ─────────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

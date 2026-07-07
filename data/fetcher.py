"""
Data Fetcher — Downloads market data from Yahoo Finance.

Handles:
  - OHLCV price data for all universe constituents
  - Market cap and shares outstanding
  - Sector classification
  - Caching: skips download if data already in SQLite
  - Rate limiting and error handling
"""

import logging
import random
import time
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from config import UNIVERSE, START_DATE, END_DATE, DATABASE_PATH
from database.connection import DatabaseManager
from models.security import Security

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches and stores market data from Yahoo Finance.

    Implements caching: if data is already in the database for a ticker,
    it will not re-download unless force_refresh is True.
    """

    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = db_path

    def fetch_all(
        self,
        tickers: Optional[List[str]] = None,
        start: date = START_DATE,
        end: date = END_DATE,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for all tickers and store in database.

        Args:
            tickers: List of tickers to fetch. Defaults to full universe.
            start: Start date for historical data.
            end: End date for historical data.
            force_refresh: If True, re-download even if cached.

        Returns:
            Pivot DataFrame: index=date, columns=tickers, values=adj_close.
        """
        if tickers is None:
            tickers = list(UNIVERSE.keys())

        logger.info(
            f"Fetching data for {len(tickers)} tickers "
            f"from {start} to {end}"
        )

        with DatabaseManager(self.db_path) as db:
            db.initialize_schema()

            # Check what we already have
            if not force_refresh:
                existing_count = db.get_price_count()
                if existing_count > 0:
                    logger.info(
                        f"Database has {existing_count} price records. "
                        f"Use force_refresh=True to re-download."
                    )
                    prices = db.get_prices(tickers, str(start), str(end))
                    if not prices.empty and len(prices.columns) >= len(tickers) * 0.8:
                        logger.info("Using cached data from database")
                        return prices

            # Download from yfinance
            all_prices = self._download_prices(tickers, start, end)

            # Fetch security info and store
            securities = self._fetch_security_info(tickers)
            for sec in securities:
                db.upsert_security(
                    ticker=sec.ticker,
                    name=sec.name,
                    sector=sec.sector,
                    market_cap=sec.market_cap,
                    shares_outstanding=sec.shares_outstanding,
                    free_float_factor=sec.free_float_factor,
                )

            # Store prices
            if not all_prices.empty:
                db.insert_prices(all_prices)
                logger.info(f"Stored {len(all_prices)} price records in database")

            # Return pivot table
            return db.get_prices(tickers, str(start), str(end))

    def _download_prices(
        self, tickers: List[str], start: date, end: date,
    ) -> pd.DataFrame:
        """Download OHLCV data from Yahoo Finance.

        Downloads all tickers at once for efficiency, then reshapes
        into the format expected by the database.
        """
        logger.info(f"Downloading prices from Yahoo Finance...")

        try:
            # Download all at once — yfinance handles batching
            data = yf.download(
                tickers=tickers,
                start=str(start),
                end=str(end),
                auto_adjust=False,
                progress=True,
                threads=True,
            )
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            return pd.DataFrame()

        if data.empty:
            logger.warning("No data returned from Yahoo Finance")
            return pd.DataFrame()

        # Reshape multi-level columns into rows
        rows = []

        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    ticker_data = data
                else:
                    # Multi-ticker download has multi-level columns
                    ticker_data = pd.DataFrame()
                    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                        if (col, ticker) in data.columns:
                            ticker_data[col] = data[(col, ticker)]
                        elif col in data.columns:
                            ticker_data[col] = data[col]

                if ticker_data.empty:
                    logger.warning(f"No data for {ticker}")
                    continue

                for dt, row in ticker_data.iterrows():
                    rows.append({
                        "ticker": ticker,
                        "trade_date": dt.strftime("%Y-%m-%d"),
                        "open_price": row.get("Open"),
                        "high_price": row.get("High"),
                        "low_price": row.get("Low"),
                        "close_price": row.get("Close"),
                        "adj_close": row.get("Adj Close"),
                        "volume": int(row.get("Volume", 0)) if pd.notna(row.get("Volume")) else 0,
                    })
            except Exception as e:
                logger.warning(f"Error processing {ticker}: {e}")
                continue

        result = pd.DataFrame(rows)
        logger.info(f"Downloaded {len(result)} price records for {len(tickers)} tickers")
        return result

    def _fetch_security_info(self, tickers: List[str]) -> List[Security]:
        """Fetch market cap, shares outstanding, and sector for each ticker.

        Uses yfinance's info endpoint. Falls back to config for sector
        if yfinance doesn't return it.
        """
        securities = []

        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info

                market_cap = info.get("marketCap", 0) or 0
                shares = info.get("sharesOutstanding", 0) or 0

                # Market cap in millions, shares in millions
                market_cap_m = market_cap / 1e6
                shares_m = shares / 1e6

                # Sector from yfinance or fallback to config
                sector = info.get("sector", "")
                if not sector and ticker in UNIVERSE:
                    sector = UNIVERSE[ticker]["sector"]

                name = info.get("shortName", "") or info.get("longName", "")
                if not name and ticker in UNIVERSE:
                    name = UNIVERSE[ticker]["name"]

                # Simulate free float factor (in production, this comes from
                # MSCI's proprietary data — we approximate with a realistic range)
                free_float = self._estimate_free_float(ticker)

                securities.append(Security(
                    ticker=ticker,
                    name=name,
                    sector=sector,
                    market_cap=market_cap_m,
                    shares_outstanding=shares_m,
                    free_float_factor=free_float,
                ))

                logger.debug(
                    f"{ticker}: mcap=${market_cap_m:,.0f}M, "
                    f"shares={shares_m:,.0f}M, ff={free_float:.2f}"
                )

                # Small delay to avoid rate limiting
                time.sleep(0.1)

            except Exception as e:
                logger.warning(f"Failed to get info for {ticker}: {e}")
                # Fallback to config data
                if ticker in UNIVERSE:
                    securities.append(Security(
                        ticker=ticker,
                        name=UNIVERSE[ticker]["name"],
                        sector=UNIVERSE[ticker]["sector"],
                        market_cap=0,
                        shares_outstanding=0,
                        free_float_factor=0.85,
                    ))

        logger.info(f"Fetched info for {len(securities)} securities")
        return securities

    @staticmethod
    def _estimate_free_float(ticker: str) -> float:
        """Estimate free-float factor for a ticker.

        In production, MSCI calculates this from detailed ownership data.
        We simulate with realistic values — most large US stocks have
        high free floats (0.80–0.98).

        Deterministic seed from ticker so results are reproducible.
        """
        # Seed based on ticker for reproducibility
        seed = sum(ord(c) for c in ticker)
        rng = random.Random(seed)

        # Most S&P 500 stocks have free floats between 0.80 and 0.98
        return round(rng.uniform(0.80, 0.98), 2)

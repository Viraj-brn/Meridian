"""
Corporate Action Handler — Detection and adjustment for stock splits.

Corporate actions (splits, dividends, mergers) are MSCI's daily
operational reality. This module detects and adjusts for them
using the adjustment factor derived from raw vs adjusted prices.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CorporateActionHandler:
    """Detects and adjusts for corporate actions in price data.

    In production, MSCI receives corporate action feeds from exchanges
    and data vendors. Here we detect them from price patterns and use
    yfinance's adjustment factors.

    Detected actions:
      - Stock splits: price drops >40% with volume spike
      - Reverse splits: price jumps >100%
      - These are already handled by yfinance's 'Adj Close', but we
        demonstrate the detection logic as an interview talking point.
    """

    def __init__(self, split_threshold: float = 0.40):
        """
        Args:
            split_threshold: Minimum daily drop to flag as potential split.
        """
        self.split_threshold = split_threshold

    def detect_splits(
        self, prices_df: pd.DataFrame,
    ) -> List[Dict]:
        """Detect potential stock splits from price patterns.

        A split is characterized by:
          - Sudden large price drop (>40%) — e.g., 4-for-1 split = 75% drop
          - Market cap should remain roughly unchanged
          - Volume often spikes on split day

        Args:
            prices_df: DataFrame with columns = tickers, values = raw prices.

        Returns:
            List of dicts with: ticker, date, price_change, likely_ratio.
        """
        detected = []
        returns = prices_df.pct_change()

        for ticker in returns.columns:
            series = returns[ticker].dropna()

            # Large drops (potential forward splits)
            drops = series[series < -self.split_threshold]
            for dt, ret in drops.items():
                ratio = self._estimate_split_ratio(ret)
                detected.append({
                    "ticker": ticker,
                    "date": dt,
                    "price_change": ret,
                    "likely_ratio": ratio,
                    "type": "forward_split",
                })
                logger.info(
                    f"Detected potential split: {ticker} on {dt}, "
                    f"price change {ret:.1%}, likely ratio {ratio}"
                )

            # Large jumps (potential reverse splits)
            jumps = series[series > 1.0]  # >100% jump
            for dt, ret in jumps.items():
                ratio = self._estimate_reverse_ratio(ret)
                detected.append({
                    "ticker": ticker,
                    "date": dt,
                    "price_change": ret,
                    "likely_ratio": ratio,
                    "type": "reverse_split",
                })

        return detected

    def compute_adjustment_factors(
        self,
        raw_close: pd.DataFrame,
        adj_close: pd.DataFrame,
    ) -> pd.DataFrame:
        """Derive cumulative adjustment factors from raw vs adjusted prices.

        adjustment_factor = adj_close / raw_close

        When a 2-for-1 split occurs:
          - Raw close halves on split day
          - Adj close remains smooth (all historical prices pre-adjusted)
          - Therefore: factor changes from ~1.0 to ~0.5 on split day

        Args:
            raw_close: DataFrame of raw (unadjusted) close prices.
            adj_close: DataFrame of adjusted close prices.

        Returns:
            DataFrame of adjustment factors per ticker per date.
        """
        # Avoid division by zero
        safe_raw = raw_close.replace(0, np.nan)
        factors = adj_close / safe_raw

        return factors

    @staticmethod
    def _estimate_split_ratio(daily_return: float) -> str:
        """Estimate the split ratio from the daily return.

        Common splits and their price changes:
          2-for-1:  -50%
          3-for-1:  -67%
          4-for-1:  -75%
          5-for-1:  -80%
          10-for-1: -90%
        """
        if daily_return > -0.45:
            return "unknown"

        remaining = 1 + daily_return  # e.g., -0.75 → 0.25
        estimated_ratio = round(1 / remaining)
        return f"{estimated_ratio}-for-1"

    @staticmethod
    def _estimate_reverse_ratio(daily_return: float) -> str:
        """Estimate reverse split ratio from price jump."""
        multiple = 1 + daily_return  # e.g., +300% → 4.0
        estimated_ratio = round(multiple)
        return f"1-for-{estimated_ratio}"

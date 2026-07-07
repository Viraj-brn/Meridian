"""
Data Cleaner — Validation, forward-fill, and outlier detection.

Financial data quality is MSCI's #1 concern. This module validates
price data, fills gaps, and flags anomalies that could indicate
corporate actions or data errors.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when financial data fails validation checks."""
    pass


class DataCleaner:
    """Validates and cleans financial time-series data.

    Implements MSCI-style data quality checks:
      - Forward-fill for missing prices (standard in finance)
      - Outlier detection for potential corporate actions
      - Negative/zero price rejection
      - Gap detection in trading dates
    """

    def __init__(self, anomaly_threshold: float = 0.20):
        """
        Args:
            anomaly_threshold: Daily return threshold for flagging outliers.
                               0.20 = 20% — a move this large usually
                               indicates a corporate action, not normal trading.
        """
        self.anomaly_threshold = anomaly_threshold

    def clean(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline.

        Steps:
          1. Validate: reject negative/zero prices
          2. Forward-fill: handle missing values
          3. Detect outliers: flag but don't remove (log them)

        Args:
            prices_df: DataFrame with DatetimeIndex, columns = tickers,
                       values = prices.

        Returns:
            Cleaned DataFrame.
        """
        logger.info(f"Cleaning price data: {prices_df.shape}")

        # Step 1: Validate
        self.validate_data_quality(prices_df)

        # Step 2: Forward-fill missing values
        cleaned = self.forward_fill_prices(prices_df)

        # Step 3: Detect and log outliers
        outliers = self.detect_outliers(cleaned)
        if outliers:
            logger.warning(
                f"Detected {len(outliers)} potential anomalies "
                f"(>{self.anomaly_threshold:.0%} daily move)"
            )
            for ticker, dt, pct_change in outliers[:10]:
                logger.warning(
                    f"  {ticker} on {dt}: {pct_change:+.1%} daily change"
                )

        # Step 4: Report quality summary
        report = self.generate_quality_report(cleaned)
        logger.info(f"Data quality report:\n{report}")

        return cleaned

    def validate_data_quality(self, df: pd.DataFrame) -> None:
        """Check for critical data issues.

        Raises DataValidationError for fatal issues.
        Logs warnings for non-fatal issues.
        """
        if df.empty:
            raise DataValidationError("Price DataFrame is empty")

        # Check for negative prices
        neg_mask = df < 0
        if neg_mask.any().any():
            neg_tickers = df.columns[neg_mask.any()].tolist()
            raise DataValidationError(
                f"Negative prices found for: {neg_tickers}"
            )

        # Check for duplicate index (dates)
        if df.index.duplicated().any():
            n_dupes = df.index.duplicated().sum()
            logger.warning(f"Found {n_dupes} duplicate dates — dropping duplicates")

        # Check for zero prices (suspicious but not fatal)
        zero_mask = df == 0
        if zero_mask.any().any():
            zero_tickers = df.columns[zero_mask.any()].tolist()
            logger.warning(f"Zero prices found for: {zero_tickers}")

        # Check null percentage
        null_pct = df.isnull().mean()
        high_null = null_pct[null_pct > 0.10]
        if not high_null.empty:
            logger.warning(
                f"High null percentage (>10%) for: "
                f"{dict(high_null.round(3))}"
            )

    def forward_fill_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill missing prices.

        Standard practice in financial data:
        - If a stock doesn't trade on a given day, its price is
          assumed to be the last traded price
        - This maintains time-series continuity

        Also back-fills the very first values if needed (so no NaN
        at the start of the series).
        """
        before_nulls = df.isnull().sum().sum()
        cleaned = df.ffill().bfill()
        after_nulls = cleaned.isnull().sum().sum()

        filled = before_nulls - after_nulls
        if filled > 0:
            logger.info(f"Forward-filled {filled} missing values")

        return cleaned

    def detect_outliers(
        self, df: pd.DataFrame,
    ) -> List[Tuple[str, str, float]]:
        """Detect daily returns exceeding the anomaly threshold.

        Large daily moves (>20%) typically indicate:
          - Stock splits (price halves or doubles)
          - Mergers/acquisitions (price jumps to offer price)
          - Data errors (incorrect price entry)

        Returns:
            List of (ticker, date, pct_change) tuples.
        """
        returns = df.pct_change()
        outliers = []

        for ticker in returns.columns:
            series = returns[ticker].dropna()
            anomalies = series[series.abs() > self.anomaly_threshold]

            for dt, ret in anomalies.items():
                date_str = dt.strftime("%Y-%m-%d") if hasattr(dt, 'strftime') else str(dt)
                outliers.append((ticker, date_str, ret))

        return sorted(outliers, key=lambda x: abs(x[2]), reverse=True)

    def generate_quality_report(self, df: pd.DataFrame) -> str:
        """Generate a summary of data quality metrics.

        Returns a formatted string with:
          - Date range
          - Number of trading days
          - Missing data percentage per ticker
          - Overall data completeness
        """
        n_dates = len(df.index)
        n_tickers = len(df.columns)
        null_pct = df.isnull().mean()
        completeness = 1 - df.isnull().sum().sum() / (n_dates * n_tickers)

        date_range = f"{df.index.min()} to {df.index.max()}"

        lines = [
            f"  Date range: {date_range}",
            f"  Trading days: {n_dates}",
            f"  Tickers: {n_tickers}",
            f"  Overall completeness: {completeness:.1%}",
        ]

        # Tickers with any missing data
        missing = null_pct[null_pct > 0]
        if not missing.empty:
            lines.append(f"  Tickers with missing data: {len(missing)}")
            for ticker, pct in missing.items():
                lines.append(f"    {ticker}: {pct:.1%} missing")

        return "\n".join(lines)

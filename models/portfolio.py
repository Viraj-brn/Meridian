"""
Portfolio — Weighted collection of securities with performance tracking.

A Portfolio represents a specific allocation of capital across securities.
It can compute its return series given price data and compare itself
against a benchmark.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from models.security import Security


@dataclass
class Portfolio:
    """A portfolio of securities with specified weights.

    Attributes:
        name: Portfolio identifier.
        holdings: Dict mapping ticker → weight (should sum to 1.0).
    """
    name: str
    holdings: Dict[str, float] = field(default_factory=dict)

    @property
    def tickers(self) -> List[str]:
        """List of tickers in the portfolio."""
        return list(self.holdings.keys())

    @property
    def n_holdings(self) -> int:
        """Number of holdings."""
        return len(self.holdings)

    def weight_sum(self) -> float:
        """Sum of all weights (should be ~1.0)."""
        return sum(self.holdings.values())

    def is_valid(self, tolerance: float = 1e-6) -> bool:
        """Check if weights sum to 1.0 within tolerance."""
        return abs(self.weight_sum() - 1.0) < tolerance

    def top_holdings(self, n: int = 10) -> Dict[str, float]:
        """Return the top N holdings by weight."""
        sorted_holdings = sorted(
            self.holdings.items(), key=lambda x: x[1], reverse=True
        )
        return dict(sorted_holdings[:n])

    def sector_weights(self, securities: List[Security]) -> Dict[str, float]:
        """Aggregate weights by sector.

        Args:
            securities: List of Security objects to look up sector info.

        Returns:
            Dict mapping sector → total weight.
        """
        sector_map = {s.ticker: s.sector for s in securities}
        sector_wts: Dict[str, float] = {}

        for ticker, weight in self.holdings.items():
            sector = sector_map.get(ticker, "Unknown")
            sector_wts[sector] = sector_wts.get(sector, 0.0) + weight

        return dict(sorted(sector_wts.items(), key=lambda x: x[1], reverse=True))

    def compute_returns(
        self,
        prices_df: pd.DataFrame,
        log_returns: bool = True,
    ) -> pd.Series:
        """Compute the portfolio return series from constituent prices.

        Args:
            prices_df: DataFrame with DatetimeIndex, columns = tickers.
            log_returns: If True, use log returns; otherwise simple returns.

        Returns:
            pd.Series of daily portfolio returns.
        """
        # Compute individual stock returns
        if log_returns:
            stock_returns = np.log(prices_df / prices_df.shift(1))
        else:
            stock_returns = prices_df.pct_change()

        # Filter to only held tickers that exist in the data
        held_tickers = [t for t in self.tickers if t in stock_returns.columns]
        weights = np.array([self.holdings[t] for t in held_tickers])

        # Weighted sum of returns
        portfolio_returns = (stock_returns[held_tickers] * weights).sum(axis=1)

        return portfolio_returns.dropna()

    def compute_cumulative_returns(
        self,
        prices_df: pd.DataFrame,
    ) -> pd.Series:
        """Compute cumulative return series.

        Uses log returns internally for additivity, then converts back
        to simple cumulative returns for display.
        """
        log_rets = self.compute_returns(prices_df, log_returns=True)
        cumulative = np.exp(log_rets.cumsum()) - 1
        return cumulative

    def active_returns(
        self,
        prices_df: pd.DataFrame,
        benchmark_prices: pd.Series,
    ) -> pd.Series:
        """Compute active return series (portfolio - benchmark).

        Args:
            prices_df: Constituent price data.
            benchmark_prices: Benchmark price series (e.g., SPY).

        Returns:
            pd.Series of daily active returns.
        """
        port_returns = self.compute_returns(prices_df, log_returns=True)
        bench_returns = np.log(benchmark_prices / benchmark_prices.shift(1)).dropna()

        # Align dates
        common_idx = port_returns.index.intersection(bench_returns.index)
        return port_returns.loc[common_idx] - bench_returns.loc[common_idx]

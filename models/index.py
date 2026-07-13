"""
Equity Index — Core index construction with the divisor method.

Implements MSCI-style index computation:
  1. Constituents are weighted by free-float adjusted market cap
  2. Index level is computed via the divisor method for continuity
  3. Quarterly rebalancing with buffer rules to reduce turnover
  4. Full rebalance history tracking

The Divisor Method:
  Index(t) = Σ(P_i(t) × Q_i × FFF_i) / Divisor
  
  The divisor is set at inception so that Index(t=0) = base_level.
  At each rebalance, the divisor is adjusted so the index level
  is continuous (no jump) despite constituent changes.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.security import Security
from models.weighting import WeightingStrategy, MarketCapWeighting

logger = logging.getLogger(__name__)


@dataclass
class RebalanceRecord:
    """Record of a single rebalance event."""
    rebalance_date: date
    additions: List[str] = field(default_factory=list)
    removals: List[str] = field(default_factory=list)
    weights_before: Dict[str, float] = field(default_factory=dict)
    weights_after: Dict[str, float] = field(default_factory=dict)


class EquityIndex:
    """A market-cap weighted equity index using the divisor method.

    This class handles:
      - Index level computation from constituent prices
      - Divisor management for continuity across rebalances
      - Quarterly rebalancing with full history

    Args:
        name: Index identifier (e.g., 'MERIDIAN_50').
        constituents: Initial list of Security objects.
        weighting_strategy: How to compute weights (default: market-cap).
        base_level: Starting index level (default: 1000).
    """

    def __init__(
        self,
        name: str,
        constituents: List[Security],
        weighting_strategy: Optional[WeightingStrategy] = None,
        base_level: float = 1000.0,
    ):
        self.name = name
        self.constituents = list(constituents)
        self.strategy = weighting_strategy or MarketCapWeighting()
        self.base_level = base_level
        self.divisor: Optional[float] = None
        self.rebalance_history: List[RebalanceRecord] = []

    @property
    def weights(self) -> Dict[str, float]:
        """Current constituent weights."""
        return self.strategy.compute_weights(self.constituents)

    @property
    def tickers(self) -> List[str]:
        """List of current constituent tickers."""
        return [s.ticker for s in self.constituents]

    def total_free_float_mcap(self) -> float:
        """Sum of all constituents' free-float adjusted market caps."""
        return sum(s.free_float_market_cap for s in self.constituents)

    def _compute_numerator(
        self,
        prices: Dict[str, float],
        securities: List[Security],
    ) -> float:
        """Compute the raw index numerator: Σ(P_i × Q_i × FFF_i).

        Uses shares_outstanding and free_float_factor from each Security,
        multiplied by the current price.
        """
        numerator = 0.0
        for sec in securities:
            price = prices.get(sec.ticker)
            if price is not None and price > 0:
                # shares_outstanding is in millions, price is per share
                numerator += price * sec.shares_outstanding * sec.free_float_factor
        return numerator

    def initialize_divisor(self, prices: Dict[str, float]) -> float:
        """Set the initial divisor so that Index(t=0) = base_level.

        Divisor = Σ(P_i(0) × Q_i × FFF_i) / base_level

        Must be called once before computing index levels.
        """
        numerator = self._compute_numerator(prices, self.constituents)
        if numerator <= 0:
            raise ValueError("Cannot initialize divisor: numerator is zero")

        self.divisor = numerator / self.base_level
        logger.info(
            f"Index '{self.name}' initialized: divisor={self.divisor:.4f}, "
            f"base_level={self.base_level}"
        )
        return self.divisor

    def compute_level(self, prices: Dict[str, float]) -> float:
        """Compute the index level for a given set of prices.

        Index(t) = Σ(P_i(t) × Q_i × FFF_i) / Divisor

        Args:
            prices: Dict mapping ticker → price for the current date.

        Returns:
            The computed index level.

        Raises:
            RuntimeError: If the divisor hasn't been initialized.
        """
        if self.divisor is None:
            raise RuntimeError("Divisor not initialized. Call initialize_divisor first.")

        numerator = self._compute_numerator(prices, self.constituents)
        return numerator / self.divisor

    def rebalance(
        self,
        new_constituents: List[Security],
        prices: Dict[str, float],
        rebalance_date: date,
    ) -> RebalanceRecord:
        """Rebalance the index with new constituents.

        The divisor is adjusted so the index level is continuous:
          new_divisor = old_divisor × (new_numerator / old_numerator)

        Both numerators are evaluated using prices on the rebalance date,
        but with old vs new constituent lists.

        Args:
            new_constituents: Updated list of Securities.
            prices: Prices on the rebalance date.
            rebalance_date: The date of rebalancing.

        Returns:
            RebalanceRecord documenting what changed.
        """
        if self.divisor is None:
            raise RuntimeError("Divisor not initialized. Call initialize_divisor first.")

        # Record state before rebalance
        old_tickers = set(self.tickers)
        new_tickers = {s.ticker for s in new_constituents}
        weights_before = self.weights

        # Compute old numerator (old constituents, today's prices)
        old_numerator = self._compute_numerator(prices, self.constituents)

        # Update constituents
        self.constituents = list(new_constituents)

        # Compute new numerator (new constituents, same prices)
        new_numerator = self._compute_numerator(prices, self.constituents)

        # Adjust divisor for continuity: index level should not change
        if old_numerator > 0:
            self.divisor = self.divisor * (new_numerator / old_numerator)

        weights_after = self.weights

        # Build record
        record = RebalanceRecord(
            rebalance_date=rebalance_date,
            additions=sorted(new_tickers - old_tickers),
            removals=sorted(old_tickers - new_tickers),
            weights_before=weights_before,
            weights_after=weights_after,
        )
        self.rebalance_history.append(record)

        logger.info(
            f"Rebalanced '{self.name}' on {rebalance_date}: "
            f"+{len(record.additions)} / -{len(record.removals)}, "
            f"new divisor={self.divisor:.4f}"
        )

        return record

    def compute_index_series(
        self,
        prices_df: pd.DataFrame,
        rebalance_freq: int = 63,
    ) -> pd.DataFrame:
        """Compute the full historical index level series.

        Iterates through each trading day, computing the index level.
        Every `rebalance_freq` days, rebalances using updated market caps.

        Args:
            prices_df: DataFrame with DatetimeIndex, columns = tickers,
                       values = adjusted close prices.
            rebalance_freq: Number of trading days between rebalances.

        Returns:
            DataFrame with columns: ['date', 'index_level', 'daily_return'].
        """
        dates = prices_df.index.tolist()
        if not dates:
            return pd.DataFrame(columns=["date", "index_level", "daily_return"])

        # Initialize with first day's prices
        first_prices = prices_df.iloc[0].to_dict()
        self.initialize_divisor(first_prices)

        levels = []
        prev_level = self.base_level

        for i, dt in enumerate(dates):
            day_prices = prices_df.iloc[i].to_dict()

            # Rebalance check (every rebalance_freq trading days, skip day 0)
            if i > 0 and i % rebalance_freq == 0:
                # Update market caps based on current prices
                updated = []
                for sec in self.constituents:
                    price = day_prices.get(sec.ticker, 0)
                    if price > 0 and sec.shares_outstanding > 0:
                        new_mcap = price * sec.shares_outstanding
                        updated.append(Security(
                            ticker=sec.ticker,
                            name=sec.name,
                            sector=sec.sector,
                            market_cap=new_mcap,
                            shares_outstanding=sec.shares_outstanding,
                            free_float_factor=sec.free_float_factor,
                        ))
                    else:
                        updated.append(sec)

                trade_date = dt.date() if hasattr(dt, 'date') else dt
                self.rebalance(updated, day_prices, trade_date)

            level = self.compute_level(day_prices)
            daily_ret = (level - prev_level) / prev_level if prev_level > 0 else 0.0

            levels.append({
                "date": dt,
                "index_level": level,
                "daily_return": daily_ret,
            })
            prev_level = level

        result = pd.DataFrame(levels)
        if not result.empty:
            result.set_index("date", inplace=True)

        return result

"""
Weighting Strategies — Strategy Pattern for index weight computation.

Different indexes use different weighting schemes. The Strategy pattern
lets us swap weighting logic without modifying the Index class itself.

Supported strategies:
  - MarketCapWeighting:  w_i = FF_mcap_i / Σ FF_mcap_j   (MSCI World style)
  - EqualWeighting:      w_i = 1/N                        (S&P 500 EW style)
  - CappedWeighting:     Market-cap with a maximum weight per constituent
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from models.security import Security


class WeightingStrategy(ABC):
    """Abstract interface for index weighting schemes."""

    @abstractmethod
    def compute_weights(self, securities: List[Security]) -> Dict[str, float]:
        """Compute weight for each security. Weights must sum to 1.0.

        Args:
            securities: List of Security objects with valid market cap data.

        Returns:
            Dict mapping ticker → weight (0.0 to 1.0).

        Raises:
            ValueError: If the securities list is empty or all mcaps are zero.
        """
        pass


class MarketCapWeighting(WeightingStrategy):
    """Free-float market-cap weighted — the industry standard.

    Each stock's weight is proportional to its free-float adjusted
    market capitalization. Larger companies get larger weights.
    This is self-rebalancing: as prices move, weights adjust naturally.
    """

    def compute_weights(self, securities: List[Security]) -> Dict[str, float]:
        if not securities:
            raise ValueError("Cannot compute weights for empty securities list")

        total_ff_mcap = sum(s.free_float_market_cap for s in securities)

        if total_ff_mcap <= 0:
            raise ValueError(
                "Total free-float market cap is zero — cannot compute weights"
            )

        return {
            s.ticker: s.free_float_market_cap / total_ff_mcap
            for s in securities
        }


class EqualWeighting(WeightingStrategy):
    """Equal-weighted — each stock gets 1/N weight regardless of size.

    Tilts the index toward smaller companies relative to a cap-weighted
    index. Requires frequent rebalancing since weights drift with prices.
    """

    def compute_weights(self, securities: List[Security]) -> Dict[str, float]:
        if not securities:
            raise ValueError("Cannot compute weights for empty securities list")

        n = len(securities)
        return {s.ticker: 1.0 / n for s in securities}


class CappedWeighting(WeightingStrategy):
    """Market-cap weighted with a maximum weight cap per constituent.

    Prevents single-stock domination. After capping, excess weight is
    redistributed proportionally among uncapped constituents.

    Common in regulatory-compliant indexes (e.g., UCITS requires ≤10%
    per stock, ≤40% for all stocks above 5%).

    Args:
        max_weight: Maximum allowed weight for any single constituent
                    (e.g., 0.10 for 10% cap).
    """

    def __init__(self, max_weight: float = 0.10):
        if not 0 < max_weight <= 1.0:
            raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")
        self.max_weight = max_weight

    def compute_weights(self, securities: List[Security]) -> Dict[str, float]:
        if not securities:
            raise ValueError("Cannot compute weights for empty securities list")

        # Start with market-cap weights
        total_ff_mcap = sum(s.free_float_market_cap for s in securities)
        if total_ff_mcap <= 0:
            raise ValueError("Total free-float market cap is zero")

        raw_weights = {
            s.ticker: s.free_float_market_cap / total_ff_mcap
            for s in securities
        }

        # Iteratively cap and redistribute until no stock exceeds the cap
        # This converges in a few iterations for typical universes
        for _ in range(50):  # Safety limit
            capped = {}
            excess = 0.0
            uncapped_total = 0.0

            for ticker, w in raw_weights.items():
                if w > self.max_weight:
                    capped[ticker] = self.max_weight
                    excess += w - self.max_weight
                else:
                    uncapped_total += w

            if excess == 0:
                break

            # Redistribute excess proportionally among uncapped stocks
            for ticker, w in raw_weights.items():
                if ticker in capped:
                    raw_weights[ticker] = self.max_weight
                else:
                    if uncapped_total > 0:
                        raw_weights[ticker] = w + excess * (w / uncapped_total)

        return raw_weights

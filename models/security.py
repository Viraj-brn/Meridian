"""
Security — Core financial instrument representation.

A Security is the atomic unit of an index: a single stock with its
identifying information, market capitalization, and free-float factor.
MSCI uses free-float adjusted market cap for all index weighting.
"""

from dataclasses import dataclass


@dataclass
class Security:
    """Represents a single equity security in the index universe.

    Attributes:
        ticker: Exchange ticker symbol (e.g., 'AAPL').
        name: Full company name.
        sector: GICS sector classification.
        market_cap: Total market capitalization in USD millions.
        shares_outstanding: Total shares outstanding in millions.
        free_float_factor: Fraction of shares available for public trading
                           (0.0 to 1.0). Excludes insider, government,
                           and strategic holdings.
    """
    ticker: str
    name: str
    sector: str
    market_cap: float = 0.0
    shares_outstanding: float = 0.0
    free_float_factor: float = 1.0

    @property
    def free_float_market_cap(self) -> float:
        """Free-float adjusted market cap = market_cap × free_float_factor.

        This is the value MSCI uses to determine index weights.
        Example: Saudi Aramco has ~$2T total mcap but only ~2% free float,
        so its free-float mcap is ~$40B — dramatically different weight.
        """
        return self.market_cap * self.free_float_factor

    def __repr__(self) -> str:
        return (
            f"Security({self.ticker}, "
            f"mcap=${self.market_cap:,.0f}M, "
            f"ff={self.free_float_factor:.2f})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Security):
            return NotImplemented
        return self.ticker == other.ticker

    def __hash__(self) -> int:
        return hash(self.ticker)

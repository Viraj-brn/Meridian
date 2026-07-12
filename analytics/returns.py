"""
Returns — Financial return calculations.

Implements both simple and log (continuously compounded) returns.
MSCI uses log returns because they're additive over time and
approximately normally distributed.

Key formulas:
  Simple return:  R_t = (P_t - P_{t-1}) / P_{t-1}
  Log return:     r_t = ln(P_t / P_{t-1})

  Relationship:   r_t = ln(1 + R_t)   (nearly equal for small returns)
  Additivity:     r(t₁→t₃) = r(t₁→t₂) + r(t₂→t₃)
"""

import numpy as np
import pandas as pd

from config import TRADING_DAYS_PER_YEAR


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute simple (arithmetic) returns.

    R_t = (P_t - P_{t-1}) / P_{t-1}

    Simple returns are intuitive but NOT additive over time.
    A +10% followed by -10% does NOT net to 0%.
    """
    return prices.pct_change()


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute log (continuously compounded) returns.

    r_t = ln(P_t / P_{t-1})

    Preferred in quantitative finance because:
    1. Additive: r(t₁→t₃) = r(t₁→t₂) + r(t₂→t₃)
    2. Symmetric: +5% and -5% are truly equal magnitudes
    3. More normally distributed than simple returns
    4. Convenient for continuous-time models
    """
    return np.log(prices / prices.shift(1))


def cumulative_returns(
    returns_series: pd.Series,
    log: bool = True,
) -> pd.Series:
    """Compute cumulative return series from daily returns.

    For log returns: cumulative = exp(Σ r_t) - 1
    For simple returns: cumulative = Π(1 + R_t) - 1
    """
    if log:
        return np.exp(returns_series.cumsum()) - 1
    else:
        return (1 + returns_series).cumprod() - 1


def annualize_return(
    total_return: float,
    n_periods: int,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Convert a total return over N periods to an annualized return.

    Annualized = (1 + total_return)^(periods_per_year / n_periods) - 1

    Example: 50% total return over 3 years → ~14.5% annualized
    """
    if n_periods <= 0:
        return 0.0
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1


def total_return(prices: pd.Series) -> float:
    """Compute total return from a price series.

    total_return = (P_final - P_initial) / P_initial
    """
    prices_clean = prices.dropna()
    if len(prices_clean) < 2:
        return 0.0
    return (prices_clean.iloc[-1] / prices_clean.iloc[0]) - 1


def active_return(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.Series:
    """Compute active return (alpha) = portfolio - benchmark.

    This is what fund managers are hired to generate.
    Positive active return → outperformance.
    """
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    return portfolio_returns.loc[common] - benchmark_returns.loc[common]

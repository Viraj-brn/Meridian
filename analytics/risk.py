"""
Risk Analytics — Volatility, VaR, drawdown, and covariance.

Implements the core risk metrics that Barra/MSCI computes at scale:
  - Rolling and EWMA volatility
  - Value at Risk (Historical and Parametric)
  - Conditional VaR (Expected Shortfall)
  - Maximum drawdown and drawdown series
  - Covariance and correlation matrices
"""

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

from config import TRADING_DAYS_PER_YEAR, ROLLING_WINDOW_SHORT, VAR_CONFIDENCE


def rolling_volatility(
    returns: pd.Series,
    window: int = ROLLING_WINDOW_SHORT,
    annualize: bool = True,
) -> pd.Series:
    """Rolling standard deviation of returns.

    Vol = std(returns) × √(252)   [annualized]

    Args:
        returns: Daily return series (log or simple).
        window: Rolling window in trading days (default: 20 ≈ 1 month).
        annualize: If True, multiply by √252 for annualized volatility.
    """
    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol


def ewma_volatility(
    returns: pd.Series,
    span: int = 20,
    annualize: bool = True,
) -> pd.Series:
    """Exponentially weighted moving average volatility.

    Gives more weight to recent observations. Used by RiskMetrics
    and preferred when volatility is time-varying (which it always is).

    λ ≈ 1 - 2/(span+1)  for equivalence with span parameter.
    """
    vol = returns.ewm(span=span).std()
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol


def max_drawdown(
    prices_or_cumret: pd.Series,
) -> Tuple[float, object, object]:
    """Compute maximum drawdown and its peak/trough dates.

    Drawdown = (Peak - Trough) / Peak

    Returns:
        Tuple of (max_drawdown_value, peak_date, trough_date).
        max_drawdown_value is negative (e.g., -0.35 for a 35% drawdown).
    """
    # Running maximum
    running_max = prices_or_cumret.cummax()

    # Drawdown at each point
    drawdowns = (prices_or_cumret - running_max) / running_max

    # Maximum drawdown (most negative value)
    max_dd = drawdowns.min()

    # Find trough date
    trough_date = drawdowns.idxmin()

    # Find peak date (last peak before the trough)
    peak_date = prices_or_cumret.loc[:trough_date].idxmax()

    return max_dd, peak_date, trough_date


def drawdown_series(prices_or_cumret: pd.Series) -> pd.Series:
    """Compute the running drawdown at each point in time.

    Returns a series of negative values showing how far below
    the running peak the value is at each point.
    """
    running_max = prices_or_cumret.cummax()
    return (prices_or_cumret - running_max) / running_max


def value_at_risk(
    returns: pd.Series,
    confidence: float = VAR_CONFIDENCE,
    method: str = "historical",
) -> float:
    """Compute Value at Risk.

    VaR answers: "What is the maximum loss at the X% confidence level?"
    If 1-day 95% VaR = 2%, then on 95% of days, you lose less than 2%.

    Methods:
      - historical: Sort returns, take the (1-confidence) percentile
      - parametric: Assume normal distribution, use z-score

    Returns:
        VaR as a positive number (i.e., the loss magnitude).
    """
    returns_clean = returns.dropna()

    if method == "historical":
        var = np.percentile(returns_clean, (1 - confidence) * 100)
    elif method == "parametric":
        mu = returns_clean.mean()
        sigma = returns_clean.std()
        z = stats.norm.ppf(1 - confidence)
        var = mu + z * sigma
    else:
        raise ValueError(f"Unknown VaR method: {method}")

    return abs(var)


def conditional_var(
    returns: pd.Series,
    confidence: float = VAR_CONFIDENCE,
) -> float:
    """Conditional VaR (Expected Shortfall / CVaR).

    CVaR = expected loss GIVEN that the loss exceeds VaR.
    It answers: "If we're in the worst 5% of days, how bad is it on average?"

    CVaR is considered superior to VaR because:
      1. It captures tail risk (VaR just gives a threshold)
      2. It is a coherent risk measure (satisfies subadditivity)
    """
    returns_clean = returns.dropna()
    var_threshold = np.percentile(returns_clean, (1 - confidence) * 100)
    tail_losses = returns_clean[returns_clean <= var_threshold]

    if len(tail_losses) == 0:
        return abs(var_threshold)

    return abs(tail_losses.mean())


def covariance_matrix(
    returns_df: pd.DataFrame,
    annualize: bool = True,
) -> pd.DataFrame:
    """Compute the covariance matrix of asset returns.

    The covariance matrix Σ is the heart of risk models:
      Portfolio variance = w^T Σ w

    For N stocks, Σ is N×N, symmetric, and positive semi-definite.
    """
    cov = returns_df.cov()
    if annualize:
        cov = cov * TRADING_DAYS_PER_YEAR
    return cov


def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the correlation matrix of asset returns.

    Correlation normalizes covariance to [-1, +1]:
      corr(i,j) = cov(i,j) / (σ_i × σ_j)
    """
    return returns_df.corr()


def portfolio_variance(
    weights: np.ndarray,
    cov_mat: np.ndarray,
) -> float:
    """Compute portfolio variance from weights and covariance matrix.

    σ²_p = w^T · Σ · w

    This is the fundamental equation of Modern Portfolio Theory.
    """
    return float(weights @ cov_mat @ weights)


def portfolio_volatility(
    weights: np.ndarray,
    cov_mat: np.ndarray,
    annualize: bool = False,
) -> float:
    """Compute portfolio volatility (standard deviation of returns).

    σ_p = √(w^T · Σ · w)

    Note: if cov_mat is already annualized, set annualize=False to
    avoid double-annualizing. Our pipeline pre-annualizes the
    covariance matrix, so the default is False.
    """
    var = portfolio_variance(weights, cov_mat)
    vol = np.sqrt(var)
    if annualize:
        vol *= np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol


def diversification_ratio(
    weights: np.ndarray,
    cov_mat: np.ndarray,
) -> float:
    """Compute the diversification ratio.

    DR = (w^T · σ) / σ_p

    Where σ is the vector of individual volatilities.
    DR > 1 indicates diversification benefit.
    DR = 1 means no diversification (perfectly correlated or single asset).
    """
    # Individual volatilities
    individual_vols = np.sqrt(np.diag(cov_mat))
    weighted_avg_vol = weights @ individual_vols

    port_vol = portfolio_volatility(weights, cov_mat, annualize=False)

    if port_vol == 0:
        return 1.0

    return weighted_avg_vol / port_vol

"""
Performance Metrics — Sharpe, tracking error, information ratio, and more.

These are the standard metrics used to evaluate portfolio and index
performance. Every interview at MSCI or any asset manager will ask
about at least one of these.
"""

import numpy as np
import pandas as pd

from config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """Sharpe Ratio — risk-adjusted return.

    Sharpe = (annualized_return - risk_free_rate) / annualized_volatility

    Interpretation:
      < 0.5: Poor
      0.5-1.0: Adequate
      1.0-2.0: Good
      > 2.0: Excellent (rare for long-only equity)

    Named after William Sharpe (Nobel Prize 1990).
    """
    if len(returns.dropna()) < 2:
        return 0.0

    annual_ret = returns.mean() * TRADING_DAYS_PER_YEAR
    annual_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    if annual_vol == 0:
        return 0.0

    return (annual_ret - risk_free_rate) / annual_vol


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """Sortino Ratio — like Sharpe but only penalizes downside volatility.

    Sortino = (annualized_return - rf) / downside_deviation

    Why it's better than Sharpe for some purposes:
      - Sharpe penalizes upside volatility equally to downside
      - Investors don't mind upside surprises — only downside hurts
      - Sortino focuses on what actually matters: losing money
    """
    if len(returns.dropna()) < 2:
        return 0.0

    annual_ret = returns.mean() * TRADING_DAYS_PER_YEAR
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR

    # Downside returns only (below risk-free rate)
    downside = returns[returns < daily_rf] - daily_rf
    downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR)

    if downside_dev == 0:
        return 0.0

    return (annual_ret - risk_free_rate) / downside_dev


def tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Tracking Error — how closely a portfolio follows its benchmark.

    TE = std(R_p - R_b) × √252

    Interpretation:
      ~0%: Index fund (tracks perfectly)
      1-3%: Enhanced index / low-active strategy
      3-8%: Active management
      >8%: Very active / different mandate

    MSCI's index clients target TE near 0%.
    """
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common) < 2:
        return 0.0

    active = portfolio_returns.loc[common] - benchmark_returns.loc[common]
    return active.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Information Ratio — active return per unit of active risk.

    IR = mean(R_p - R_b) × 252 / TE

    The IR tells you whether the active risk you're taking is
    being compensated with active return.

    IR > 0.5: Good active manager
    IR > 1.0: Excellent (top quartile)
    """
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common) < 2:
        return 0.0

    active = portfolio_returns.loc[common] - benchmark_returns.loc[common]
    te = active.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    if te == 0:
        return 0.0

    active_return_annualized = active.mean() * TRADING_DAYS_PER_YEAR
    return active_return_annualized / te


def calmar_ratio(
    returns: pd.Series,
    max_dd: float,
) -> float:
    """Calmar Ratio — return relative to maximum drawdown.

    Calmar = annualized_return / |max_drawdown|

    Measures how much return you get per unit of worst-case loss.
    """
    if max_dd == 0:
        return 0.0

    annual_ret = returns.mean() * TRADING_DAYS_PER_YEAR
    return annual_ret / abs(max_dd)


def performance_summary(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """Comprehensive performance summary — all metrics in one call.

    Returns a dict with all computed metrics for easy display.
    """
    from analytics.risk import (
        max_drawdown as compute_max_dd,
        value_at_risk,
        conditional_var,
    )
    from analytics.returns import cumulative_returns, total_return

    returns_clean = returns.dropna()
    n_days = len(returns_clean)

    # Cumulative prices for drawdown
    cum_ret = cumulative_returns(returns_clean, log=True) + 1

    # Max drawdown
    max_dd, peak_date, trough_date = compute_max_dd(cum_ret)

    summary = {
        "total_return": float(cum_ret.iloc[-1] - 1) if len(cum_ret) > 0 else 0,
        "annualized_return": float(returns_clean.mean() * TRADING_DAYS_PER_YEAR),
        "annualized_volatility": float(returns_clean.std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "sharpe_ratio": sharpe_ratio(returns_clean, risk_free_rate),
        "sortino_ratio": sortino_ratio(returns_clean, risk_free_rate),
        "max_drawdown": float(max_dd),
        "max_dd_peak": str(peak_date),
        "max_dd_trough": str(trough_date),
        "calmar_ratio": calmar_ratio(returns_clean, max_dd),
        "var_95": float(value_at_risk(returns_clean, 0.95)),
        "cvar_95": float(conditional_var(returns_clean, 0.95)),
        "skewness": float(returns_clean.skew()),
        "kurtosis": float(returns_clean.kurtosis()),
        "n_trading_days": n_days,
        "positive_days_pct": float((returns_clean > 0).mean()),
    }

    # Benchmark-relative metrics
    if benchmark_returns is not None:
        summary["tracking_error"] = tracking_error(returns_clean, benchmark_returns)
        summary["information_ratio"] = information_ratio(returns_clean, benchmark_returns)

        common = returns_clean.index.intersection(benchmark_returns.index)
        active = returns_clean.loc[common] - benchmark_returns.loc[common]
        summary["active_return_annualized"] = float(active.mean() * TRADING_DAYS_PER_YEAR)

    return summary

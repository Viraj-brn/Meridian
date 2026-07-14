"""
Factor Analysis — PCA-based risk decomposition.

This implements a simplified version of what Barra does:
  1. Compute the covariance matrix of stock returns
  2. Eigendecompose it to extract principal risk factors
  3. Determine how much of total risk each factor explains
  4. Compute factor exposures (betas) for each stock

The key insight: instead of tracking N×N correlations (millions of
parameters for large universes), we can explain most of the risk
with K << N factors.

Factor Model: R = B·F + ε
  R = N×1 stock returns
  B = N×K factor loading matrix
  F = K×1 factor returns
  ε = N×1 stock-specific (idiosyncratic) returns
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from config import N_PCA_FACTORS, TRADING_DAYS_PER_YEAR

logger = logging.getLogger(__name__)


@dataclass
class FactorDecomposition:
    """Results of a factor decomposition analysis.

    Attributes:
        eigenvalues: Variance explained by each factor.
        eigenvectors: Factor loadings (N_stocks × N_factors).
        variance_explained: Fraction of total variance per factor.
        cumulative_explained: Cumulative fraction of variance.
        factor_returns: Time series of factor returns.
        factor_names: Human-readable names for each factor.
    """
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    variance_explained: np.ndarray
    cumulative_explained: np.ndarray
    factor_returns: Optional[pd.DataFrame] = None
    factor_names: Optional[List[str]] = None


class FactorAnalyzer:
    """PCA-based factor analysis for equity risk decomposition.

    Performs eigendecomposition of the covariance matrix to extract
    the principal risk factors driving portfolio returns.

    In MSCI's Barra model, these factors would be identified and
    named (Market, Size, Value, Momentum, etc.). Our PCA extracts
    statistical factors — the first one is typically "the market."
    """

    def __init__(self, n_factors: int = N_PCA_FACTORS):
        self.n_factors = n_factors

    def decompose(
        self, returns_df: pd.DataFrame,
    ) -> FactorDecomposition:
        """Perform eigendecomposition of the return covariance matrix.

        Steps:
        1. Compute covariance matrix
        2. Extract eigenvalues and eigenvectors
        3. Sort by descending eigenvalue
        4. Compute variance explained

        Args:
            returns_df: DataFrame of daily returns (dates × stocks).

        Returns:
            FactorDecomposition with all decomposition results.
        """
        # Drop rows with any NaN to avoid numerical issues
        clean = returns_df.dropna()

        if len(clean) < self.n_factors:
            logger.warning(
                f"Only {len(clean)} clean observations — "
                f"need at least {self.n_factors} for PCA"
            )

        # Compute covariance matrix
        cov_matrix = clean.cov().values

        # Eigendecomposition — use eigh for symmetric matrices (guaranteed real eigenvalues)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by descending eigenvalue
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Variance explained
        total_var = eigenvalues.sum()
        variance_explained = eigenvalues / total_var
        cumulative_explained = np.cumsum(variance_explained)

        # Extract factor returns using PCA
        pca = PCA(n_components=min(self.n_factors, len(clean.columns)))
        factor_returns_arr = pca.fit_transform(clean.values)

        factor_names = [f"Factor {i+1}" for i in range(pca.n_components_)]
        factor_returns_df = pd.DataFrame(
            factor_returns_arr,
            index=clean.index,
            columns=factor_names,
        )

        logger.info(
            f"PCA decomposition: top {self.n_factors} factors explain "
            f"{cumulative_explained[min(self.n_factors, len(cumulative_explained))-1]:.1%} of variance"
        )

        return FactorDecomposition(
            eigenvalues=eigenvalues[:self.n_factors],
            eigenvectors=eigenvectors[:, :self.n_factors],
            variance_explained=variance_explained[:self.n_factors],
            cumulative_explained=cumulative_explained[:self.n_factors],
            factor_returns=factor_returns_df,
            factor_names=factor_names,
        )

    def factor_exposures(
        self,
        stock_returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute a stock's exposure (beta) to each factor via OLS regression.

        R_stock = α + β₁·F₁ + β₂·F₂ + ... + βₖ·Fₖ + ε

        Args:
            stock_returns: Return series for a single stock.
            factor_returns: DataFrame of factor return series.

        Returns:
            Dict mapping factor name → beta coefficient.
        """
        # Align dates
        common = stock_returns.index.intersection(factor_returns.index)
        y = stock_returns.loc[common].values
        X = factor_returns.loc[common].values

        # Add intercept (alpha)
        X_with_intercept = np.column_stack([np.ones(len(X)), X])

        # OLS: β = (X^T X)^(-1) X^T y
        try:
            betas = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            logger.warning("OLS failed — returning zero exposures")
            return {name: 0.0 for name in factor_returns.columns}

        result = {"alpha": betas[0]}
        for i, name in enumerate(factor_returns.columns):
            result[name] = betas[i + 1]

        return result

    def all_exposures(
        self,
        returns_df: pd.DataFrame,
        factor_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute factor exposures for all stocks.

        Returns:
            DataFrame with stocks as rows and factors as columns.
        """
        exposures = {}
        for ticker in returns_df.columns:
            exposures[ticker] = self.factor_exposures(
                returns_df[ticker], factor_returns
            )

        return pd.DataFrame(exposures).T

    def variance_attribution(
        self,
        weights: np.ndarray,
        decomposition: FactorDecomposition,
        returns_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Decompose portfolio risk into factor contributions.

        Total risk = Factor risk + Specific (idiosyncratic) risk

        This is the core of Barra's value proposition:
        "Your portfolio lost money because of the Momentum factor reversal,
         not because of bad stock picks."
        """
        clean = returns_df.dropna()
        cov = clean.cov().values

        # Total portfolio variance
        total_var = float(weights @ cov @ weights)

        if total_var == 0:
            return {"total_variance": 0, "factor_risk_pct": 0, "specific_risk_pct": 0}

        # Factor-explained variance
        n_factors = len(decomposition.eigenvalues)
        factor_var = sum(
            decomposition.eigenvalues[i]
            * (weights @ decomposition.eigenvectors[:, i]) ** 2
            for i in range(n_factors)
        )

        specific_var = total_var - factor_var

        attribution = {
            "total_variance": total_var * TRADING_DAYS_PER_YEAR,
            "total_volatility": np.sqrt(total_var * TRADING_DAYS_PER_YEAR),
            "factor_risk_pct": factor_var / total_var * 100,
            "specific_risk_pct": max(0, specific_var / total_var * 100),
        }

        # Per-factor attribution
        for i in range(n_factors):
            factor_contribution = (
                decomposition.eigenvalues[i]
                * (weights @ decomposition.eigenvectors[:, i]) ** 2
            )
            name = f"Factor {i+1}"
            attribution[f"{name}_pct"] = factor_contribution / total_var * 100

        return attribution

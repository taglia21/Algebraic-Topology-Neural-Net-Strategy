"""
Portfolio Optimizer — Mean-Variance, Risk Parity, Max Sharpe
=============================================================

Provides portfolio-level weight optimization using:
  1. Mean-Variance Optimization (Markowitz)
  2. Risk Parity (equal risk contribution)
  3. Maximum Sharpe Ratio portfolio
  4. Minimum Variance portfolio

All methods work with numpy only — no external optimization libraries required.
Uses analytical solutions and iterative methods where appropriate.

Usage:
    optimizer = PortfolioOptimizer()
    weights = optimizer.optimize_weights(
        symbols=["AAPL", "MSFT", "GOOG"],
        returns_matrix=np.array(...),  # shape (T, N)
    )
    frontier = optimizer.efficient_frontier(returns_matrix, n_points=50)

Author: Tier 1 Implementation — Feb 2026
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class PortfolioResult:
    """Result of portfolio optimization."""
    weights: Dict[str, float]         # symbol -> weight (0-1, sum=1)
    expected_return: float             # Annualized expected return
    volatility: float                  # Annualized portfolio volatility
    sharpe_ratio: float                # Sharpe ratio (assumes rf=0)
    method: str                        # "max_sharpe", "risk_parity", "min_variance"


@dataclass
class EfficientFrontierPoint:
    """Single point on the efficient frontier."""
    target_return: float
    volatility: float
    sharpe_ratio: float
    weights: Dict[str, float]


@dataclass
class OptimizerConfig:
    """Configuration for portfolio optimizer."""
    risk_free_rate: float = 0.05       # Annual risk-free rate (5% T-bills)
    min_weight: float = 0.0            # Minimum weight per asset (0 = allow zero)
    max_weight: float = 0.40           # Maximum weight per asset (40%)
    annualization_factor: int = 252    # Trading days per year
    regularization: float = 1e-6       # Covariance matrix regularization
    max_iterations: int = 1000         # Max iterations for iterative solvers
    tolerance: float = 1e-8            # Convergence tolerance
    n_frontier_points: int = 50        # Points on efficient frontier


# ============================================================================
# PORTFOLIO OPTIMIZER
# ============================================================================

class PortfolioOptimizer:
    """
    Portfolio weight optimizer supporting multiple methods.

    All methods accept a returns matrix of shape (T, N) where:
      T = number of time periods (daily returns)
      N = number of assets

    Returns are expected to be simple returns (not log returns).
    """

    def __init__(self, config: Optional[OptimizerConfig] = None):
        self.cfg = config or OptimizerConfig()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def optimize_weights(
        self,
        symbols: List[str],
        returns_matrix: np.ndarray,
        method: str = "max_sharpe",
    ) -> PortfolioResult:
        """
        Optimize portfolio weights.

        Args:
            symbols: List of N asset symbols
            returns_matrix: Array of shape (T, N) of daily returns
            method: "max_sharpe", "risk_parity", "min_variance", "equal_weight"

        Returns:
            PortfolioResult with optimal weights
        """
        if returns_matrix.ndim != 2:
            raise ValueError(f"returns_matrix must be 2D, got {returns_matrix.ndim}D")

        n_assets = returns_matrix.shape[1]
        if n_assets != len(symbols):
            raise ValueError(f"symbols ({len(symbols)}) != returns columns ({n_assets})")

        if n_assets < 2:
            # Single asset: 100% weight
            w = {symbols[0]: 1.0}
            mu = float(np.mean(returns_matrix[:, 0])) * self.cfg.annualization_factor
            vol = float(np.std(returns_matrix[:, 0])) * np.sqrt(self.cfg.annualization_factor)
            sr = (mu - self.cfg.risk_free_rate) / vol if vol > 0 else 0.0
            return PortfolioResult(
                weights=w, expected_return=mu, volatility=vol,
                sharpe_ratio=sr, method=method,
            )

        # Compute expected returns and covariance
        mu = np.mean(returns_matrix, axis=0) * self.cfg.annualization_factor
        cov = np.cov(returns_matrix, rowvar=False) * self.cfg.annualization_factor
        # Regularize covariance for numerical stability
        cov += np.eye(n_assets) * self.cfg.regularization

        if method == "max_sharpe":
            w = self._max_sharpe(mu, cov)
        elif method == "risk_parity":
            w = self._risk_parity(cov)
        elif method == "min_variance":
            w = self._min_variance(cov)
        elif method == "equal_weight":
            w = np.ones(n_assets) / n_assets
        else:
            raise ValueError(f"Unknown method: {method}")

        # Apply weight bounds
        w = self._apply_bounds(w)

        # Compute portfolio metrics
        port_ret = float(w @ mu)
        port_vol = float(np.sqrt(w @ cov @ w))
        port_sr = (port_ret - self.cfg.risk_free_rate) / port_vol if port_vol > 0 else 0.0

        weights_dict = {symbols[i]: round(float(w[i]), 6) for i in range(n_assets)}

        result = PortfolioResult(
            weights=weights_dict,
            expected_return=round(port_ret, 6),
            volatility=round(port_vol, 6),
            sharpe_ratio=round(port_sr, 4),
            method=method,
        )

        logger.info(
            f"Portfolio optimized ({method}): "
            f"E[R]={port_ret:.2%} σ={port_vol:.2%} SR={port_sr:.2f}"
        )
        return result

    def efficient_frontier(
        self,
        symbols: List[str],
        returns_matrix: np.ndarray,
        n_points: Optional[int] = None,
    ) -> List[EfficientFrontierPoint]:
        """
        Calculate the efficient frontier.

        Returns:
            List of EfficientFrontierPoint from min-variance to max-return
        """
        n_points = n_points or self.cfg.n_frontier_points
        n_assets = returns_matrix.shape[1]

        mu = np.mean(returns_matrix, axis=0) * self.cfg.annualization_factor
        cov = np.cov(returns_matrix, rowvar=False) * self.cfg.annualization_factor
        cov += np.eye(n_assets) * self.cfg.regularization

        # Get min and max return bounds
        w_min_var = self._min_variance(cov)
        w_min_var = self._apply_bounds(w_min_var)
        min_ret = float(w_min_var @ mu)
        max_ret = float(np.max(mu))

        # Ensure we have a valid range
        if max_ret <= min_ret:
            max_ret = min_ret + 0.01

        frontier = []
        target_returns = np.linspace(min_ret, max_ret * 0.95, n_points)

        for target in target_returns:
            w = self._min_variance_target_return(mu, cov, target)
            w = self._apply_bounds(w)
            port_ret = float(w @ mu)
            port_vol = float(np.sqrt(w @ cov @ w))
            sr = (port_ret - self.cfg.risk_free_rate) / port_vol if port_vol > 0 else 0.0
            weights_dict = {symbols[i]: round(float(w[i]), 6) for i in range(n_assets)}
            frontier.append(EfficientFrontierPoint(
                target_return=round(port_ret, 6),
                volatility=round(port_vol, 6),
                sharpe_ratio=round(sr, 4),
                weights=weights_dict,
            ))

        return frontier

    def risk_contributions(
        self,
        weights: np.ndarray,
        cov: np.ndarray,
    ) -> np.ndarray:
        """
        Compute each asset's risk contribution to portfolio volatility.

        Returns:
            Array of fractional risk contributions (sum = 1.0)
        """
        port_vol = np.sqrt(weights @ cov @ weights)
        if port_vol < 1e-12:
            return np.ones(len(weights)) / len(weights)

        # Marginal risk contribution
        marginal = cov @ weights / port_vol
        # Component risk contribution
        component = weights * marginal
        # Fractional
        return component / np.sum(component)

    # ------------------------------------------------------------------ #
    # Optimization methods (analytical / iterative)
    # ------------------------------------------------------------------ #

    def _max_sharpe(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Maximum Sharpe ratio portfolio (analytical solution).

        For unconstrained case: w* ∝ Σ^{-1} (μ - rf)
        """
        n = len(mu)
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        excess = mu - self.cfg.risk_free_rate
        w = cov_inv @ excess

        # Handle case where all excess returns are negative
        if np.sum(w) <= 0:
            return np.ones(n) / n

        # Normalize to sum to 1
        w = w / np.sum(w)

        # Project negative weights to zero and renormalize
        w = np.maximum(w, 0)
        w_sum = np.sum(w)
        if w_sum > 0:
            w /= w_sum
        else:
            w = np.ones(n) / n

        return w

    def _min_variance(self, cov: np.ndarray) -> np.ndarray:
        """
        Minimum variance portfolio (analytical solution).

        w* = Σ^{-1} 1 / (1' Σ^{-1} 1)
        """
        n = cov.shape[0]
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        ones = np.ones(n)
        w = cov_inv @ ones
        w /= np.sum(w)

        # Project to non-negative
        w = np.maximum(w, 0)
        w_sum = np.sum(w)
        if w_sum > 0:
            w /= w_sum
        else:
            w = np.ones(n) / n

        return w

    def _min_variance_target_return(
        self, mu: np.ndarray, cov: np.ndarray, target: float
    ) -> np.ndarray:
        """
        Minimum variance portfolio subject to target return.
        Uses Lagrangian analytical solution.
        """
        n = len(mu)
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        ones = np.ones(n)

        # Lagrangian parameters
        a = float(ones @ cov_inv @ ones)
        b = float(ones @ cov_inv @ mu)
        c = float(mu @ cov_inv @ mu)
        det = a * c - b * b

        if abs(det) < 1e-12:
            return np.ones(n) / n

        # Optimal weights: w = g + h * target
        g = (c * (cov_inv @ ones) - b * (cov_inv @ mu)) / det
        h = (a * (cov_inv @ mu) - b * (cov_inv @ ones)) / det
        w = g + h * target

        # Project to non-negative
        w = np.maximum(w, 0)
        w_sum = np.sum(w)
        if w_sum > 0:
            w /= w_sum
        else:
            w = np.ones(n) / n

        return w

    def _risk_parity(self, cov: np.ndarray) -> np.ndarray:
        """
        Risk parity portfolio (iterative Spinu 2013 method).

        Target: each asset contributes equally to portfolio risk.
        Uses the Newton-Raphson iterative approach.
        """
        n = cov.shape[0]
        target_risk = np.ones(n) / n  # Equal risk contribution

        # Initialize with inverse-volatility weights
        vols = np.sqrt(np.diag(cov))
        vols = np.maximum(vols, 1e-8)
        w = (1.0 / vols)
        w /= np.sum(w)

        for iteration in range(self.cfg.max_iterations):
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-12:
                break

            # Risk contributions
            marginal = cov @ w / port_vol
            rc = w * marginal
            rc_sum = np.sum(rc)
            if rc_sum < 1e-12:
                break
            rc_frac = rc / rc_sum

            # Update: adjust weights to equalize risk contributions
            # Simple gradient step
            diff = rc_frac - target_risk
            step = 0.1 / (1 + iteration * 0.01)
            adjustment = -step * diff * w

            w_new = w + adjustment
            w_new = np.maximum(w_new, 1e-8)
            w_new /= np.sum(w_new)

            # Check convergence
            if np.max(np.abs(w_new - w)) < self.cfg.tolerance:
                w = w_new
                break
            w = w_new

        return w

    # ------------------------------------------------------------------ #
    # Weight bounds enforcement
    # ------------------------------------------------------------------ #

    def _apply_bounds(self, w: np.ndarray) -> np.ndarray:
        """Apply min/max weight bounds and renormalize."""
        w = np.maximum(w, self.cfg.min_weight)
        w = np.minimum(w, self.cfg.max_weight)

        # Renormalize
        w_sum = np.sum(w)
        if w_sum > 0:
            w /= w_sum
        else:
            w = np.ones(len(w)) / len(w)

        # Iteratively enforce max_weight (may need multiple passes)
        for _ in range(10):
            excess = w > self.cfg.max_weight
            if not excess.any():
                break
            excess_total = np.sum(w[excess] - self.cfg.max_weight)
            w[excess] = self.cfg.max_weight
            # Redistribute excess proportionally
            not_excess = ~excess
            if not_excess.any():
                redistribute = excess_total * (w[not_excess] / np.sum(w[not_excess]))
                w[not_excess] += redistribute

        # Final normalization
        w_sum = np.sum(w)
        if w_sum > 0:
            w /= w_sum

        return w

    # ------------------------------------------------------------------ #
    # Utility methods
    # ------------------------------------------------------------------ #

    def portfolio_metrics(
        self,
        weights: Dict[str, float],
        symbols: List[str],
        returns_matrix: np.ndarray,
    ) -> Dict[str, float]:
        """Compute portfolio metrics for given weights."""
        n = len(symbols)
        w = np.array([weights.get(s, 0.0) for s in symbols])

        mu = np.mean(returns_matrix, axis=0) * self.cfg.annualization_factor
        cov = np.cov(returns_matrix, rowvar=False) * self.cfg.annualization_factor
        cov += np.eye(n) * self.cfg.regularization

        port_ret = float(w @ mu)
        port_vol = float(np.sqrt(w @ cov @ w))
        sr = (port_ret - self.cfg.risk_free_rate) / port_vol if port_vol > 0 else 0.0

        rc = self.risk_contributions(w, cov)

        return {
            "expected_return": round(port_ret, 6),
            "volatility": round(port_vol, 6),
            "sharpe_ratio": round(sr, 4),
            "max_weight": round(float(np.max(w)), 4),
            "min_weight": round(float(np.min(w[w > 0])) if np.any(w > 0) else 0.0, 4),
            "n_assets": int(np.sum(w > 0.001)),
            "concentration_hhi": round(float(np.sum(w ** 2)), 4),
            "max_risk_contribution": round(float(np.max(rc)), 4),
        }

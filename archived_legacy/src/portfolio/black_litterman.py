"""
Phase O — Black-Litterman Portfolio Optimizer.

Item 7: BlackLittermanOptimizer — Bayesian, Pi = delta * Sigma * w_eq, posterior returns.
Item 8: RiskParityAllocator — equal risk contribution, use when HMM regime=choppy.
Item 9: MeanVarianceOptimizer — Ledoit-Wolf shrinkage, max weight 0.20, turnover penalty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Item 7 — BlackLittermanOptimizer
# ---------------------------------------------------------------------------

@dataclass
class BLResult:
    """Black-Litterman optimization result."""
    posterior_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    posterior_cov: np.ndarray = field(default_factory=lambda: np.array([]))
    optimal_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    equilibrium_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    n_assets: int = 0
    symbols: List[str] = field(default_factory=list)


class BlackLittermanOptimizer:
    """Black-Litterman portfolio optimizer.

    Steps:
      1. Compute equilibrium returns: Pi = delta * Sigma * w_eq
      2. Incorporate views: P @ mu = Q + eps
      3. Posterior: mu_BL = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 *
                            [(tau*Sigma)^-1*Pi + P'*Omega^-1*Q]
      4. Optimize weights using posterior returns.

    Args:
        delta: Risk aversion coefficient (market-implied).
        tau: Uncertainty scalar on equilibrium (typically 0.025-0.05).
    """

    def __init__(self, delta: float = 2.5, tau: float = 0.05):
        self.delta = delta
        self.tau = tau

    def equilibrium_returns(
        self,
        cov_matrix: np.ndarray,
        market_weights: np.ndarray,
    ) -> np.ndarray:
        """Compute implied equilibrium returns: Pi = delta * Sigma * w_eq."""
        return self.delta * cov_matrix @ market_weights

    def posterior(
        self,
        cov_matrix: np.ndarray,
        market_weights: np.ndarray,
        P: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        omega: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Black-Litterman posterior returns and covariance.

        Args:
            cov_matrix: N x N covariance matrix.
            market_weights: N-vector of equilibrium weights.
            P: K x N pick matrix (which assets each view references).
            Q: K-vector of view returns.
            omega: K x K view uncertainty matrix.

        Returns:
            (posterior_returns, posterior_covariance)
        """
        Pi = self.equilibrium_returns(cov_matrix, market_weights)
        tau_sigma = self.tau * cov_matrix
        tau_sigma_inv = np.linalg.inv(tau_sigma)

        if P is None or Q is None:
            # No views: posterior = equilibrium
            return Pi, cov_matrix + tau_sigma

        P = np.atleast_2d(P)
        Q = np.atleast_1d(Q)

        if omega is None:
            # Proportional to projection of prior uncertainty
            omega = np.diag(np.diag(P @ tau_sigma @ P.T))

        omega_inv = np.linalg.inv(omega)

        # Posterior precision = tau_sigma_inv + P' * omega_inv * P
        post_precision = tau_sigma_inv + P.T @ omega_inv @ P
        post_cov = np.linalg.inv(post_precision)

        # Posterior mean
        post_mean = post_cov @ (tau_sigma_inv @ Pi + P.T @ omega_inv @ Q)

        return post_mean, post_cov + cov_matrix

    def optimize(
        self,
        cov_matrix: np.ndarray,
        market_weights: np.ndarray,
        symbols: Optional[List[str]] = None,
        P: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        omega: Optional[np.ndarray] = None,
        max_weight: float = 0.30,
    ) -> BLResult:
        """Full Black-Litterman optimization.

        Args:
            cov_matrix: Covariance matrix.
            market_weights: Equilibrium market-cap weights.
            symbols: Asset symbols.
            P, Q, omega: Optional views.
            max_weight: Maximum position weight.

        Returns:
            BLResult with posterior returns and optimal weights.
        """
        n = cov_matrix.shape[0]
        post_ret, post_cov = self.posterior(cov_matrix, market_weights, P, Q, omega)
        Pi = self.equilibrium_returns(cov_matrix, market_weights)

        # Mean-variance optimization with posterior
        def neg_utility(w):
            ret = w @ post_ret
            risk = w @ post_cov @ w
            return -(ret - 0.5 * self.delta * risk)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, max_weight)] * n
        w0 = market_weights.copy()

        result = minimize(neg_utility, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)

        weights = result.x if result.success else market_weights.copy()
        weights = np.maximum(weights, 0)
        weights /= max(weights.sum(), 1e-12)

        return BLResult(
            posterior_returns=post_ret,
            posterior_cov=post_cov,
            optimal_weights=weights,
            equilibrium_returns=Pi,
            n_assets=n,
            symbols=symbols or [f"asset_{i}" for i in range(n)],
        )


# ---------------------------------------------------------------------------
# Item 8 — RiskParityAllocator
# ---------------------------------------------------------------------------

@dataclass
class RiskParityResult:
    """Risk parity allocation result."""
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    risk_contributions: np.ndarray = field(default_factory=lambda: np.array([]))
    portfolio_vol: float = 0.0
    max_rc_deviation: float = 0.0  # max deviation from equal RC
    n_assets: int = 0


class RiskParityAllocator:
    """Equal-risk-contribution portfolio allocator.

    Each asset contributes equally to total portfolio variance.
    Use this allocation when HMM regime = choppy (risk-off).

    RC_i = w_i * (Sigma @ w)_i / (w' @ Sigma @ w)
    Target: RC_i = 1/N for all i.
    """

    def __init__(self, max_iter: int = 1000, tol: float = 1e-8):
        self.max_iter = max_iter
        self.tol = tol

    def allocate(self, cov_matrix: np.ndarray) -> RiskParityResult:
        """Compute risk parity weights.

        Uses iterative optimization to equalize risk contributions.

        Args:
            cov_matrix: N x N covariance matrix.

        Returns:
            RiskParityResult with weights and risk contributions.
        """
        n = cov_matrix.shape[0]
        target_rc = 1.0 / n

        def risk_contribution(w):
            port_var = w @ cov_matrix @ w
            marginal = cov_matrix @ w
            rc = w * marginal / max(np.sqrt(port_var), 1e-12)
            return rc

        def objective(w):
            rc = risk_contribution(w)
            rc_normalized = rc / max(rc.sum(), 1e-12)
            return np.sum((rc_normalized - target_rc) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.01, 1.0)] * n
        w0 = np.ones(n) / n

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": self.max_iter, "ftol": self.tol})

        weights = result.x if result.success else w0
        weights = np.maximum(weights, 0)
        weights /= max(weights.sum(), 1e-12)

        rc = risk_contribution(weights)
        rc_norm = rc / max(rc.sum(), 1e-12)
        port_vol = float(np.sqrt(weights @ cov_matrix @ weights))

        return RiskParityResult(
            weights=weights,
            risk_contributions=rc_norm,
            portfolio_vol=port_vol,
            max_rc_deviation=float(np.max(np.abs(rc_norm - target_rc))),
            n_assets=n,
        )


# ---------------------------------------------------------------------------
# Item 9 — MeanVarianceOptimizer
# ---------------------------------------------------------------------------

@dataclass
class MVOResult:
    """Mean-variance optimization result."""
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    expected_return: float = 0.0
    portfolio_vol: float = 0.0
    sharpe_ratio: float = 0.0
    turnover: float = 0.0
    n_assets: int = 0


class MeanVarianceOptimizer:
    """Mean-variance optimizer with Ledoit-Wolf shrinkage.

    Features:
      - Ledoit-Wolf shrinkage for covariance estimation.
      - Maximum weight constraint (default 0.20).
      - Turnover penalty for rebalancing cost control.
    """

    def __init__(
        self,
        max_weight: float = 0.20,
        risk_aversion: float = 2.5,
        turnover_penalty: float = 0.005,
        risk_free_rate: float = 0.05,
    ):
        self.max_weight = max_weight
        self.risk_aversion = risk_aversion
        self.turnover_penalty = turnover_penalty
        self.risk_free_rate = risk_free_rate

    @staticmethod
    def ledoit_wolf_shrinkage(returns: np.ndarray) -> np.ndarray:
        """Apply Ledoit-Wolf shrinkage to sample covariance.

        Target: scaled identity matrix (F = mu * I).
        Shrinkage: Sigma_LW = alpha * F + (1 - alpha) * S.
        """
        T, N = returns.shape
        sample_cov = np.cov(returns, rowvar=False)

        # Target: average variance * identity
        mu = np.trace(sample_cov) / N
        F = mu * np.eye(N)

        # Compute optimal shrinkage intensity
        delta = sample_cov - F
        d2 = np.sum(delta ** 2) / N

        # Estimate phi
        # Simplified: use Ledoit-Wolf formula
        x2 = returns - np.mean(returns, axis=0)
        sum_pi = 0.0
        for t in range(T):
            outer = np.outer(x2[t], x2[t]) - sample_cov
            sum_pi += np.sum(outer ** 2)
        phi = sum_pi / (T * N)

        # Shrinkage intensity
        kappa = (phi - d2) / max(d2, 1e-12)
        alpha = max(0.0, min(1.0, kappa / T))

        shrunk = alpha * F + (1 - alpha) * sample_cov
        return shrunk

    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
    ) -> MVOResult:
        """Optimize portfolio weights with turnover penalty.

        Args:
            expected_returns: N-vector of expected returns.
            cov_matrix: N x N covariance matrix (should be Ledoit-Wolf shrunk).
            current_weights: Current portfolio weights for turnover penalty.

        Returns:
            MVOResult with optimal weights and statistics.
        """
        n = len(expected_returns)
        if current_weights is None:
            current_weights = np.ones(n) / n

        def objective(w):
            ret = w @ expected_returns
            risk = 0.5 * self.risk_aversion * w @ cov_matrix @ w
            turnover = self.turnover_penalty * np.sum(np.abs(w - current_weights))
            return -(ret - risk - turnover)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, self.max_weight)] * n

        result = minimize(
            objective,
            current_weights.copy(),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        weights = result.x if result.success else current_weights.copy()
        weights = np.maximum(weights, 0)
        weights /= max(weights.sum(), 1e-12)

        port_ret = float(weights @ expected_returns)
        port_vol = float(np.sqrt(weights @ cov_matrix @ weights))
        turnover = float(np.sum(np.abs(weights - current_weights)))
        sharpe = (port_ret - self.risk_free_rate / 252) / max(port_vol, 1e-12)

        return MVOResult(
            weights=weights,
            expected_return=port_ret,
            portfolio_vol=port_vol,
            sharpe_ratio=sharpe,
            turnover=turnover,
            n_assets=n,
        )

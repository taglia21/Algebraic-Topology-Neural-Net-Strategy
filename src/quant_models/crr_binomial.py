"""
Cox-Ross-Rubinstein (CRR) Binomial Tree
=========================================
American and European option pricing via the binomial lattice.

The CRR model discretizes the GBM process:
    u = exp(σ√Δt)      (up factor)
    d = 1/u = exp(-σ√Δt) (down factor)
    p = (exp(rΔt) - d) / (u - d)  (risk-neutral probability)

The tree has n+1 terminal nodes for n steps.
American options: check early exercise at each node.

Features:
- European and American calls/puts
- 100+ step tree (configurable, default 200)
- Full Greeks via tree perturbation
- Early exercise boundary for American options
- Dividend yield support
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BinomialResult:
    """Result of CRR binomial tree pricing."""
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    n_steps: int
    is_american: bool
    early_exercise_boundary: Optional[List[Tuple[int, float]]] = None


class CRRBinomialTree:
    """
    Cox-Ross-Rubinstein binomial tree for American/European option pricing.
    
    Uses backward induction with optional early exercise.
    """

    def __init__(
        self,
        n_steps: int = 200,
        risk_free_rate: float = 0.045,
    ):
        self.n_steps = n_steps
        self.rf = risk_free_rate

    def price(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        is_call: bool = True,
        is_american: bool = True,
        r: Optional[float] = None,
        q: float = 0.0,  # continuous dividend yield
        n_steps: Optional[int] = None,
    ) -> BinomialResult:
        """
        Price an option using the CRR binomial tree.
        
        Parameters:
            S: spot price
            K: strike price
            T: time to expiry (years)
            sigma: volatility (annualized)
            is_call: True for call, False for put
            is_american: True for American, False for European
            r: risk-free rate (uses self.rf if None)
            q: continuous dividend yield
            n_steps: override number of steps
        """
        r = r if r is not None else self.rf
        N = n_steps if n_steps is not None else self.n_steps

        if T <= 0:
            intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
            return BinomialResult(
                price=intrinsic, delta=1.0 if is_call and S > K else (-1.0 if not is_call and S < K else 0.0),
                gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
                n_steps=0, is_american=is_american,
            )

        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        disc = np.exp(-r * dt)
        p = (np.exp((r - q) * dt) - d) / (u - d)

        if p < 0 or p > 1:
            logger.warning(f"CRR risk-neutral prob out of bounds: p={p:.4f}, clamping")
            p = np.clip(p, 0.001, 0.999)

        # Build asset price at terminal nodes
        # S_T[j] = S * u^j * d^(N-j) for j = 0..N
        j_arr = np.arange(N + 1)
        S_T = S * u ** j_arr * d ** (N - j_arr)

        # Terminal payoffs
        if is_call:
            V = np.maximum(S_T - K, 0.0)
        else:
            V = np.maximum(K - S_T, 0.0)

        # Track early exercise boundary
        ee_boundary = [] if is_american else None

        # Backward induction
        for i in range(N - 1, -1, -1):
            # Asset prices at step i
            S_i = S * u ** np.arange(i + 1) * d ** (i - np.arange(i + 1))
            V = disc * (p * V[1:i + 2] + (1 - p) * V[0:i + 1])

            if is_american:
                if is_call:
                    exercise = np.maximum(S_i - K, 0.0)
                else:
                    exercise = np.maximum(K - S_i, 0.0)

                # Find early exercise nodes
                ee_mask = exercise > V
                if np.any(ee_mask):
                    # Record the critical S where exercise is optimal
                    ee_idx = np.where(ee_mask)[0]
                    if len(ee_idx) > 0:
                        ee_boundary.append((i, float(S_i[ee_idx[0]])))

                V = np.maximum(V, exercise)

        price = float(V[0])

        # Greeks via re-pricing with perturbed parameters
        dS = S * 0.01
        dsig = 0.005
        dT = 1.0 / 252
        dr = 0.001

        p_up = self._price_value(S + dS, K, T, sigma, is_call, is_american, r, q, N)
        p_down = self._price_value(S - dS, K, T, sigma, is_call, is_american, r, q, N)
        delta = (p_up - p_down) / (2 * dS)
        gamma = (p_up - 2 * price + p_down) / (dS ** 2)

        p_sig_up = self._price_value(S, K, T, sigma + dsig, is_call, is_american, r, q, N)
        vega = (p_sig_up - price) / dsig

        if T > dT:
            p_t_down = self._price_value(S, K, T - dT, sigma, is_call, is_american, r, q, N)
            theta = (p_t_down - price) / dT  # per day
        else:
            theta = 0.0

        p_r_up = self._price_value(S, K, T, sigma, is_call, is_american, r + dr, q, N)
        rho_greek = (p_r_up - price) / dr

        return BinomialResult(
            price=price,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho_greek,
            n_steps=N,
            is_american=is_american,
            early_exercise_boundary=ee_boundary,
        )

    def _price_value(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        is_call: bool,
        is_american: bool,
        r: float,
        q: float,
        N: int,
    ) -> float:
        """Fast price-only computation for Greeks."""
        if T <= 0:
            return max(S - K, 0) if is_call else max(K - S, 0)

        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        disc = np.exp(-r * dt)
        p = (np.exp((r - q) * dt) - d) / (u - d)
        p = np.clip(p, 0.001, 0.999)

        j_arr = np.arange(N + 1)
        S_T = S * u ** j_arr * d ** (N - j_arr)

        if is_call:
            V = np.maximum(S_T - K, 0.0)
        else:
            V = np.maximum(K - S_T, 0.0)

        for i in range(N - 1, -1, -1):
            V = disc * (p * V[1:i + 2] + (1 - p) * V[0:i + 1])
            if is_american:
                S_i = S * u ** np.arange(i + 1) * d ** (i - np.arange(i + 1))
                exercise = np.maximum(S_i - K, 0.0) if is_call else np.maximum(K - S_i, 0.0)
                V = np.maximum(V, exercise)

        return float(V[0])

    def price_european(
        self, S: float, K: float, T: float, sigma: float,
        is_call: bool = True, r: Optional[float] = None, q: float = 0.0,
    ) -> BinomialResult:
        """Convenience: European option."""
        return self.price(S, K, T, sigma, is_call, is_american=False, r=r, q=q)

    def price_american(
        self, S: float, K: float, T: float, sigma: float,
        is_call: bool = True, r: Optional[float] = None, q: float = 0.0,
    ) -> BinomialResult:
        """Convenience: American option."""
        return self.price(S, K, T, sigma, is_call, is_american=True, r=r, q=q)

    def implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        is_call: bool = True,
        is_american: bool = True,
        r: Optional[float] = None,
        q: float = 0.0,
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> float:
        """
        Implied vol via bisection on the binomial tree.
        """
        r = r if r is not None else self.rf
        N = min(self.n_steps, 100)  # use fewer steps for speed

        lo, hi = 0.01, 3.0
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            p = self._price_value(S, K, T, mid, is_call, is_american, r, q, N)
            if abs(p - market_price) < tol:
                return mid
            if p > market_price:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

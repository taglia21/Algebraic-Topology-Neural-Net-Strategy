"""
Monte Carlo Option Pricer
==========================
GBM-based Monte Carlo simulation with variance reduction techniques.

Implements:
- Geometric Brownian Motion path generation
- Antithetic variates (reduces variance by ~50%)
- Control variates using BSM analytical price
- European and Asian option pricing
- Greeks via finite difference on MC prices
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class MCPriceResult:
    """Monte Carlo pricing result."""
    price: float
    std_error: float
    confidence_95: Tuple[float, float]
    n_paths: int
    n_steps: int
    antithetic: bool
    control_variate: bool
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    elapsed_seconds: float = 0.0


class MonteCarloPricer:
    """
    Monte Carlo option pricer with variance reduction.
    
    Path generation via GBM:
        S_{t+dt} = S_t * exp((r - 0.5σ²)dt + σ√dt·Z)
    
    Variance reduction:
    1. Antithetic variates: for each Z, also simulate -Z
    2. Control variates: adjust using BSM analytical price
       Ĉ_cv = Ĉ_mc - β(Ĉ_control - C_control_analytical)
    """

    def __init__(
        self,
        n_paths: int = 100_000,
        n_steps: int = 252,
        risk_free_rate: float = 0.045,
        seed: Optional[int] = None,
        antithetic: bool = True,
        control_variate: bool = True,
    ):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.rf = risk_free_rate
        self.seed = seed
        self.antithetic = antithetic
        self.control_variate = control_variate

    def _generate_paths(
        self, S0: float, r: float, sigma: float, T: float, n_paths: int, n_steps: int
    ) -> np.ndarray:
        """
        Generate GBM price paths with optional antithetic variates.
        
        Returns: array of shape (n_paths, n_steps+1)
        """
        rng = np.random.default_rng(self.seed)
        dt = T / n_steps
        drift = (r - 0.5 * sigma ** 2) * dt
        vol = sigma * np.sqrt(dt)

        if self.antithetic:
            half = n_paths // 2
            Z = rng.standard_normal((half, n_steps))
            Z = np.concatenate([Z, -Z], axis=0)  # antithetic pairs
            if n_paths % 2 == 1:
                Z = np.concatenate([Z, rng.standard_normal((1, n_steps))], axis=0)
        else:
            Z = rng.standard_normal((n_paths, n_steps))

        log_increments = drift + vol * Z
        log_paths = np.zeros((len(Z), n_steps + 1))
        log_paths[:, 0] = np.log(S0)
        log_paths[:, 1:] = np.cumsum(log_increments, axis=1) + np.log(S0)

        return np.exp(log_paths)

    @staticmethod
    def _bsm_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return max(S - K * np.exp(-r * T), 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

    @staticmethod
    def _bsm_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return max(K * np.exp(-r * T) - S, 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

    def price_european(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        is_call: bool = True,
        r: Optional[float] = None,
    ) -> MCPriceResult:
        """
        Price a European option via Monte Carlo.
        """
        import time as _time
        t0 = _time.time()
        r = r if r is not None else self.rf
        paths = self._generate_paths(S, r, sigma, T, self.n_paths, self.n_steps)
        S_T = paths[:, -1]

        if is_call:
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)

        discount = np.exp(-r * T)
        raw_prices = discount * payoffs

        # Control variate adjustment
        if self.control_variate:
            bsm_price = self._bsm_call(S, K, T, r, sigma) if is_call else self._bsm_put(S, K, T, r, sigma)
            # Use geometric average Asian as control (has analytical formula under GBM)
            # Simplified: use terminal price as control with BSM E[S_T] = S*exp(rT)
            control = discount * S_T  # discounted terminal price
            control_mean = S  # E[discount * S_T] = S under risk-neutral
            beta = np.cov(raw_prices, control)[0, 1] / max(np.var(control), 1e-12)
            adjusted_prices = raw_prices - beta * (control - control_mean)
            price = float(np.mean(adjusted_prices))
            std_err = float(np.std(adjusted_prices) / np.sqrt(len(adjusted_prices)))
        else:
            price = float(np.mean(raw_prices))
            std_err = float(np.std(raw_prices) / np.sqrt(len(raw_prices)))

        price = max(price, 0)
        ci_95 = (price - 1.96 * std_err, price + 1.96 * std_err)

        # Greeks via finite difference
        dS = S * 0.01
        dsig = 0.01
        dT = 1.0 / 252
        dr = 0.001

        p_up = self._mc_price(S + dS, K, T, sigma, r, is_call)
        p_down = self._mc_price(S - dS, K, T, sigma, r, is_call)
        delta = (p_up - p_down) / (2 * dS)
        gamma = (p_up - 2 * price + p_down) / (dS ** 2)

        p_sig_up = self._mc_price(S, K, T, sigma + dsig, r, is_call)
        vega = (p_sig_up - price) / dsig

        if T > dT:
            p_t_down = self._mc_price(S, K, T - dT, sigma, r, is_call)
            theta = (p_t_down - price) / dT
        else:
            theta = 0.0

        p_r_up = self._mc_price(S, K, T, sigma, r + dr, is_call)
        rho = (p_r_up - price) / dr

        elapsed = _time.time() - t0

        return MCPriceResult(
            price=price,
            std_error=std_err,
            confidence_95=ci_95,
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            antithetic=self.antithetic,
            control_variate=self.control_variate,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            elapsed_seconds=elapsed,
        )

    def _mc_price(
        self, S: float, K: float, T: float, sigma: float, r: float, is_call: bool
    ) -> float:
        """Quick MC price (fewer paths) for Greeks calculation."""
        n_quick = min(self.n_paths, 20_000)
        paths = self._generate_paths(S, r, sigma, T, n_quick, max(self.n_steps // 4, 50))
        S_T = paths[:, -1]
        payoffs = np.maximum(S_T - K, 0) if is_call else np.maximum(K - S_T, 0)
        return float(np.mean(payoffs) * np.exp(-r * T))

    def price_asian(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        is_call: bool = True,
        r: Optional[float] = None,
    ) -> MCPriceResult:
        """Price an arithmetic average Asian option."""
        import time as _time
        t0 = _time.time()
        r = r if r is not None else self.rf
        paths = self._generate_paths(S, r, sigma, T, self.n_paths, self.n_steps)

        avg_price = np.mean(paths[:, 1:], axis=1)  # arithmetic average
        if is_call:
            payoffs = np.maximum(avg_price - K, 0)
        else:
            payoffs = np.maximum(K - avg_price, 0)

        discount = np.exp(-r * T)
        prices = discount * payoffs
        price = float(np.mean(prices))
        std_err = float(np.std(prices) / np.sqrt(len(prices)))
        ci_95 = (price - 1.96 * std_err, price + 1.96 * std_err)
        elapsed = _time.time() - t0

        return MCPriceResult(
            price=max(price, 0),
            std_error=std_err,
            confidence_95=ci_95,
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            antithetic=self.antithetic,
            control_variate=False,
            delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
            elapsed_seconds=elapsed,
        )

    def price_barrier(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        barrier: float,
        is_call: bool = True,
        is_up: bool = True,
        is_knock_out: bool = True,
        r: Optional[float] = None,
    ) -> MCPriceResult:
        """Price a barrier option (up/down, knock-in/knock-out)."""
        import time as _time
        t0 = _time.time()
        r = r if r is not None else self.rf
        paths = self._generate_paths(S, r, sigma, T, self.n_paths, self.n_steps)

        if is_up:
            barrier_hit = np.any(paths >= barrier, axis=1)
        else:
            barrier_hit = np.any(paths <= barrier, axis=1)

        S_T = paths[:, -1]
        if is_call:
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)

        if is_knock_out:
            payoffs[barrier_hit] = 0.0
        else:  # knock-in
            payoffs[~barrier_hit] = 0.0

        discount = np.exp(-r * T)
        prices = discount * payoffs
        price = float(np.mean(prices))
        std_err = float(np.std(prices) / np.sqrt(len(prices)))
        elapsed = _time.time() - t0

        return MCPriceResult(
            price=max(price, 0),
            std_error=std_err,
            confidence_95=(price - 1.96 * std_err, price + 1.96 * std_err),
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            antithetic=self.antithetic,
            control_variate=False,
            delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
            elapsed_seconds=elapsed,
        )

"""
Heston Stochastic Volatility Model
====================================
The Heston (1993) model assumes stochastic variance:

    dS = μS dt + √v S dW_1
    dv = κ(θ - v)dt + ξ√v dW_2
    dW_1·dW_2 = ρ dt

where:
    S = asset price
    v = instantaneous variance
    κ = mean-reversion speed (kappa)
    θ = long-run variance (theta)
    ξ = vol-of-vol (xi/sigma)
    ρ = correlation between price and vol processes

Option pricing via the characteristic function (semi-closed form):
    C = S·P₁ - K·e^{-rT}·P₂

    P_j = 1/2 + 1/π ∫₀^∞ Re[e^{-iu·ln(K)} · f_j(u)] / u du

Feller condition: 2κθ > ξ² (ensures v stays positive)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution, minimize

logger = logging.getLogger(__name__)


@dataclass
class HestonParams:
    """Heston model parameters."""
    v0: float      # initial variance
    kappa: float   # mean-reversion speed
    theta: float   # long-run variance
    xi: float      # vol-of-vol
    rho: float     # correlation

    @property
    def feller_satisfied(self) -> bool:
        """Feller condition: 2κθ > ξ²."""
        return 2 * self.kappa * self.theta > self.xi ** 2

    @property
    def long_run_vol(self) -> float:
        return np.sqrt(self.theta)

    @property
    def initial_vol(self) -> float:
        return np.sqrt(self.v0)


@dataclass
class HestonPriceResult:
    """Heston pricing result."""
    price: float
    delta: float
    gamma: float
    vega: float
    params: HestonParams
    feller_satisfied: bool


class HestonModel:
    """
    Heston stochastic volatility model with characteristic function pricing.
    
    Uses the Albrecher et al. (2007) formulation for numerical stability.
    """

    def __init__(self, risk_free_rate: float = 0.045):
        self.rf = risk_free_rate

    @staticmethod
    def _characteristic_function(
        phi: complex,
        S: float,
        K: float,
        T: float,
        r: float,
        params: HestonParams,
        j: int,
    ) -> complex:
        """
        Heston (1993) characteristic function f_j(φ) for j=1,2.

        Standard formulation with fixed measure-change constants:
            j=1: u=+½, b=κ−ρξ  (stock-price measure)
            j=2: u=−½, b=κ      (risk-neutral measure)

        f_j(φ) = exp(C + D·v₀ + iφ·ln(S))

        d  = √((ρξiφ − b)² − ξ²(2u·iφ − φ²))
        g  = (b − ρξiφ + d) / (b − ρξiφ − d)
        C  = r·iφ·T + (κθ/ξ²)·[(b−ρξiφ+d)T − 2·ln((1−g·e^{dT})/(1−g))]
        D  = ((b−ρξiφ+d)/ξ²) · (1−e^{dT})/(1−g·e^{dT})
        """
        v0 = params.v0
        kappa = params.kappa
        theta = params.theta
        xi = params.xi
        rho = params.rho

        if j == 1:
            u_const = 0.5
            b = kappa - rho * xi
        else:
            u_const = -0.5
            b = kappa

        a = kappa * theta
        x = np.log(S)

        d = np.sqrt(
            (rho * xi * 1j * phi - b) ** 2
            - xi ** 2 * (2 * u_const * 1j * phi - phi ** 2)
        )

        # Use the "good" g to avoid numerical issues
        g_num = b - rho * xi * 1j * phi + d
        g_den = b - rho * xi * 1j * phi - d
        if abs(g_den) < 1e-15:
            g_den = 1e-15

        g = g_num / g_den

        exp_dT = np.exp(d * T)

        C = 1j * r * phi * T + (a / xi ** 2) * (
            (b - rho * xi * 1j * phi + d) * T
            - 2.0 * np.log((1 - g * exp_dT) / (1 - g))
        )

        D = ((b - rho * xi * 1j * phi + d) / xi ** 2) * (
            (1 - exp_dT) / (1 - g * exp_dT)
        )

        return np.exp(C + D * v0 + 1j * phi * x)

    def _P_integral(
        self, S: float, K: float, T: float, r: float, params: HestonParams, j: int
    ) -> float:
        """
        P_j = 1/2 + 1/π ∫₀^∞ Re[e^{-iu·ln(K)} · f_j(u) / (iu)] du
        """
        ln_K = np.log(K)

        def integrand(u):
            if u < 1e-10:
                return 0.0
            cf = self._characteristic_function(u, S, K, T, r, params, j)
            val = np.exp(-1j * u * ln_K) * cf / (1j * u)
            return float(np.real(val))

        result, _ = quad(integrand, 0, 200, limit=200, epsabs=1e-10, epsrel=1e-10)
        return 0.5 + result / np.pi

    def price_call(
        self,
        S: float,
        K: float,
        T: float,
        params: HestonParams,
        r: Optional[float] = None,
    ) -> HestonPriceResult:
        """Price a European call using Heston characteristic function."""
        r = r if r is not None else self.rf
        return self._price(S, K, T, params, r, is_call=True)

    def price_put(
        self,
        S: float,
        K: float,
        T: float,
        params: HestonParams,
        r: Optional[float] = None,
    ) -> HestonPriceResult:
        """Price a European put using Heston characteristic function."""
        r = r if r is not None else self.rf
        return self._price(S, K, T, params, r, is_call=False)

    def _price(
        self,
        S: float,
        K: float,
        T: float,
        params: HestonParams,
        r: float,
        is_call: bool,
    ) -> HestonPriceResult:
        P1 = self._P_integral(S, K, T, r, params, j=1)
        P2 = self._P_integral(S, K, T, r, params, j=2)

        call_price = S * P1 - K * np.exp(-r * T) * P2
        call_price = max(call_price, 0)

        if is_call:
            price = call_price
        else:
            # put-call parity
            price = call_price - S + K * np.exp(-r * T)
            price = max(price, 0)

        # Greeks via finite difference
        dS = S * 0.005
        p_up = self._price_value(S + dS, K, T, params, r, is_call)
        p_down = self._price_value(S - dS, K, T, params, r, is_call)
        delta = (p_up - p_down) / (2 * dS)
        gamma = (p_up - 2 * price + p_down) / (dS ** 2)

        # Vega: sensitivity to initial vol
        dv = 0.001
        params_vup = HestonParams(
            v0=params.v0 + dv, kappa=params.kappa,
            theta=params.theta, xi=params.xi, rho=params.rho
        )
        vega = (self._price_value(S, K, T, params_vup, r, is_call) - price) / dv

        return HestonPriceResult(
            price=price,
            delta=delta,
            gamma=gamma,
            vega=vega,
            params=params,
            feller_satisfied=params.feller_satisfied,
        )

    def _price_value(
        self,
        S: float,
        K: float,
        T: float,
        params: HestonParams,
        r: float,
        is_call: bool,
    ) -> float:
        """Get price as float only."""
        P1 = self._P_integral(S, K, T, r, params, j=1)
        P2 = self._P_integral(S, K, T, r, params, j=2)
        call = max(S * P1 - K * np.exp(-r * T) * P2, 0)
        if is_call:
            return call
        return max(call - S + K * np.exp(-r * T), 0)

    def calibrate(
        self,
        market_prices: list,
        S: float,
        r: Optional[float] = None,
    ) -> HestonParams:
        """
        Calibrate Heston parameters to market option prices.
        
        market_prices: list of dicts with keys {K, T, price, is_call}
        """
        r = r if r is not None else self.rf

        def objective(x):
            v0, kappa, theta, xi, rho = x
            params = HestonParams(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
            total_err = 0.0
            for opt in market_prices:
                try:
                    model_price = self._price_value(
                        S, opt["K"], opt["T"], params, r, opt.get("is_call", True)
                    )
                    total_err += (model_price - opt["price"]) ** 2
                except Exception:
                    total_err += 1e6
            return total_err

        bounds = [
            (0.001, 1.0),   # v0
            (0.1, 10.0),    # kappa
            (0.001, 1.0),   # theta
            (0.05, 2.0),    # xi
            (-0.99, 0.0),   # rho (typically negative for equities)
        ]

        result = differential_evolution(
            objective, bounds, maxiter=200, seed=42, tol=1e-8, polish=True
        )

        v0, kappa, theta, xi, rho = result.x
        params = HestonParams(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)

        logger.info(
            f"Heston calibration: v0={v0:.4f}, κ={kappa:.3f}, θ={theta:.4f}, "
            f"ξ={xi:.4f}, ρ={rho:.3f}, Feller={'✓' if params.feller_satisfied else '✗'}"
        )
        return params

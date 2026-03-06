"""
Merton Jump-Diffusion Model
=============================
Extension of Black-Scholes with Poisson-distributed jumps.

dS/S = (μ - λk)dt + σ dW + J dN

where:
- dW = Brownian motion
- dN = Poisson process with intensity λ
- J = jump size ~ LogNormal(μ_J, σ_J)
- k = E[e^J - 1] = exp(μ_J + σ_J²/2) - 1

Option pricing via the infinite series expansion:
    C_Merton = Σ_{n=0}^{∞} (e^{-λ'T} (λ'T)^n / n!) * BSM(S, K, T, r_n, σ_n)

where:
    λ' = λ(1 + k)
    r_n = r - λk + n·log(1+k)/T
    σ_n² = σ² + n·σ_J²/T
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from math import factorial
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, poisson

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
except ImportError:
    yf = None


@dataclass
class MertonParams:
    """Merton Jump-Diffusion parameters."""
    sigma: float      # diffusion volatility
    lam: float         # jump intensity (jumps per year)
    mu_j: float        # mean log-jump size
    sigma_j: float     # std of log-jump size

    @property
    def k(self) -> float:
        """E[e^J - 1]: expected relative jump size."""
        return np.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1.0

    @property
    def total_variance(self) -> float:
        """Total variance per unit time (diffusion + jump)."""
        return self.sigma ** 2 + self.lam * (self.mu_j ** 2 + self.sigma_j ** 2)

    @property
    def excess_kurtosis(self) -> float:
        """Excess kurtosis from jump component."""
        if self.total_variance < 1e-12:
            return 0.0
        m4_jump = self.lam * (self.mu_j ** 4 + 6 * self.mu_j ** 2 * self.sigma_j ** 2 + 3 * self.sigma_j ** 4)
        return m4_jump / self.total_variance ** 2


@dataclass
class MertonPriceResult:
    """Result of Merton Jump-Diffusion option pricing."""
    price: float
    bsm_price: float
    jump_premium: float
    delta: float
    gamma: float
    vega: float
    n_terms: int
    params: MertonParams


class MertonJumpDiffusion:
    """
    Merton (1976) Jump-Diffusion model.
    
    Calibrates from return kurtosis and prices options using the 
    Merton series expansion (sum of BSM prices weighted by Poisson probs).
    """

    def __init__(self, risk_free_rate: float = 0.045, n_terms: int = 50):
        self.rf = risk_free_rate
        self.n_terms = n_terms

    @staticmethod
    def _bsm_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Standard Black-Scholes-Merton call price."""
        if T <= 0 or sigma <= 0:
            return max(S - K * np.exp(-r * T), 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def _bsm_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Standard BSM put price."""
        if T <= 0 or sigma <= 0:
            return max(K * np.exp(-r * T) - S, 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def calibrate_from_returns(self, returns: np.ndarray) -> MertonParams:
        """
        Calibrate Merton model parameters from historical returns.
        
        Uses method of moments matching:
        - Variance → σ² + λ(μ_J² + σ_J²)
        - Kurtosis → 3 + λ(μ_J⁴ + 6μ_J²σ_J² + 3σ_J⁴) / total_var²
        """
        daily_returns = returns[np.isfinite(returns)]
        n = len(daily_returns)
        if n < 60:
            raise ValueError(f"Need at least 60 observations, got {n}")

        # Annualize daily stats
        daily_var = np.var(daily_returns, ddof=1)
        annual_var = daily_var * 252
        daily_kurt = float(np.mean((daily_returns - np.mean(daily_returns)) ** 4) / daily_var ** 2 - 3.0)
        excess_kurt = max(daily_kurt, 0.01)  # floor at small positive

        # Heuristic decomposition:
        # If excess kurtosis > 0, attribute to jumps
        # Start with sigma = sqrt(0.7 * annual_var), rest is jump component
        sigma_init = np.sqrt(0.7 * annual_var)
        jump_var = annual_var - sigma_init ** 2

        # Estimate jump parameters
        # λ ~ 2-10 jumps/year for equities
        lam_init = max(min(excess_kurt * 2, 20), 1)
        sigma_j_init = np.sqrt(max(jump_var / lam_init, 0.001))
        mu_j_init = -0.5 * sigma_j_init ** 2  # slight negative mean jump

        def objective(x):
            sigma, lam, mu_j, sigma_j = x
            total_var = sigma ** 2 + lam * (mu_j ** 2 + sigma_j ** 2)
            m4_jump = lam * (mu_j ** 4 + 6 * mu_j ** 2 * sigma_j ** 2 + 3 * sigma_j ** 4)
            model_kurt = m4_jump / max(total_var ** 2, 1e-12)

            err_var = (total_var - annual_var) ** 2 / max(annual_var ** 2, 1e-12)
            err_kurt = (model_kurt - excess_kurt) ** 2 / max(excess_kurt ** 2, 1e-4)
            return err_var + err_kurt

        result = minimize(
            objective,
            [sigma_init, lam_init, mu_j_init, sigma_j_init],
            method="Nelder-Mead",
            bounds=[(0.01, 2.0), (0.1, 50), (-0.5, 0.1), (0.01, 1.0)],
            options={"maxiter": 5000, "xatol": 1e-8},
        )

        sigma, lam, mu_j, sigma_j = result.x
        sigma = max(abs(sigma), 0.01)
        lam = max(lam, 0.1)
        sigma_j = max(abs(sigma_j), 0.01)

        params = MertonParams(sigma=sigma, lam=lam, mu_j=mu_j, sigma_j=sigma_j)
        logger.info(
            f"Merton calibration: σ={sigma:.4f}, λ={lam:.2f}, "
            f"μ_J={mu_j:.4f}, σ_J={sigma_j:.4f}, k={params.k:.4f}"
        )
        return params

    def price_call(
        self, S: float, K: float, T: float, params: MertonParams, r: Optional[float] = None
    ) -> MertonPriceResult:
        """
        Price a European call using Merton's series expansion.
        
        C = Σ_{n=0}^{N} P(N=n) * BSM(S, K, T, r_n, σ_n)
        """
        r = r if r is not None else self.rf
        return self._price_option(S, K, T, params, r, is_call=True)

    def price_put(
        self, S: float, K: float, T: float, params: MertonParams, r: Optional[float] = None
    ) -> MertonPriceResult:
        """Price a European put using Merton's series expansion."""
        r = r if r is not None else self.rf
        return self._price_option(S, K, T, params, r, is_call=False)

    def _price_option(
        self, S: float, K: float, T: float, params: MertonParams, r: float, is_call: bool
    ) -> MertonPriceResult:
        sigma = params.sigma
        lam = params.lam
        mu_j = params.mu_j
        sigma_j = params.sigma_j
        k = params.k

        lam_prime = lam * (1 + k)
        price = 0.0
        bsm_fn = self._bsm_call if is_call else self._bsm_put

        for n in range(self.n_terms):
            # Poisson weight
            poisson_weight = np.exp(-lam_prime * T) * (lam_prime * T) ** n / factorial(n)
            if poisson_weight < 1e-15 and n > 5:
                break

            # Adjusted parameters for n-th term
            sigma_n = np.sqrt(sigma ** 2 + n * sigma_j ** 2 / T) if T > 0 else sigma
            r_n = r - lam * k + n * np.log(1 + k) / T if T > 0 else r

            bsm_price_n = bsm_fn(S, K, T, r_n, sigma_n)
            price += poisson_weight * bsm_price_n

        # BSM baseline (no jumps)
        bsm_base = bsm_fn(S, K, T, r, sigma)

        # Numerical Greeks
        dS = S * 0.001
        price_up = self._price_option_value(S + dS, K, T, params, r, is_call)
        price_down = self._price_option_value(S - dS, K, T, params, r, is_call)
        delta = (price_up - price_down) / (2 * dS)
        gamma = (price_up - 2 * price + price_down) / (dS ** 2)

        dsig = 0.001
        params_up = MertonParams(sigma + dsig, lam, mu_j, sigma_j)
        vega = (self._price_option_value(S, K, T, params_up, r, is_call) - price) / dsig

        return MertonPriceResult(
            price=price,
            bsm_price=bsm_base,
            jump_premium=price - bsm_base,
            delta=delta,
            gamma=gamma,
            vega=vega,
            n_terms=min(self.n_terms, n + 1),
            params=params,
        )

    def _price_option_value(
        self, S: float, K: float, T: float, params: MertonParams, r: float, is_call: bool
    ) -> float:
        """Helper: compute option price as a float only (no Greeks)."""
        sigma = params.sigma
        lam = params.lam
        mu_j = params.mu_j
        sigma_j = params.sigma_j
        k = params.k
        lam_prime = lam * (1 + k)
        bsm_fn = self._bsm_call if is_call else self._bsm_put
        price = 0.0
        for n in range(self.n_terms):
            pw = np.exp(-lam_prime * T) * (lam_prime * T) ** n / factorial(n)
            if pw < 1e-15 and n > 5:
                break
            sigma_n = np.sqrt(sigma ** 2 + n * sigma_j ** 2 / T) if T > 0 else sigma
            r_n = r - lam * k + n * np.log(1 + k) / T if T > 0 else r
            price += pw * bsm_fn(S, K, T, r_n, sigma_n)
        return price

    def calibrate_from_symbol(self, symbol: str, lookback_days: int = 504) -> MertonParams:
        """Convenience: fetch returns and calibrate."""
        if yf is None:
            raise ImportError("yfinance required")
        from datetime import timedelta
        end = datetime.now()
        start = end - timedelta(days=int(lookback_days * 1.5))
        data = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if data.empty:
            raise ValueError(f"No data for {symbol}")
        prices = data["Close"].dropna()
        log_ret = np.log(prices / prices.shift(1)).dropna().values
        return self.calibrate_from_returns(log_ret[-lookback_days:])

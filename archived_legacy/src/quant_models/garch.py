"""
GARCH(1,1) Model
=================
Generalized Autoregressive Conditional Heteroskedasticity.

Implements:
- Full GARCH(1,1) with Maximum Likelihood Estimation (MLE)
- Multi-step ahead volatility forecasting
- Volatility term structure
- Conditional VaR and ES
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
except ImportError:
    yf = None


@dataclass
class GARCHParams:
    """GARCH(1,1) parameters: sigma²_t = omega + alpha * eps²_{t-1} + beta * sigma²_{t-1}"""
    omega: float  # long-run variance weight
    alpha: float  # ARCH coefficient (reaction to shocks)
    beta: float   # GARCH coefficient (persistence)

    @property
    def persistence(self) -> float:
        """alpha + beta: persistence of volatility shocks."""
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        """Unconditional variance = omega / (1 - alpha - beta)."""
        denom = 1.0 - self.alpha - self.beta
        if denom <= 0:
            return self.omega / 0.001  # near unit root
        return self.omega / denom

    @property
    def long_run_vol(self) -> float:
        """Annualized long-run volatility."""
        return np.sqrt(self.long_run_variance * 252)

    @property
    def half_life(self) -> float:
        """Half-life of volatility shocks (in days)."""
        p = self.persistence
        if p <= 0 or p >= 1:
            return np.inf
        return np.log(2) / (-np.log(p))


@dataclass
class GARCHForecast:
    """Multi-step GARCH volatility forecast."""
    symbol: str
    params: GARCHParams
    current_vol: float  # annualized current conditional vol
    forecast_vols: List[float]  # annualized vol for each horizon step
    horizon: int
    var_95: float  # 1-day 95% VaR
    var_99: float  # 1-day 99% VaR
    es_95: float   # 1-day 95% Expected Shortfall
    log_likelihood: float
    n_observations: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def signal(self) -> float:
        """
        Vol signal in [-1, 1].
        High vol relative to long-run → negative (risk-off).
        Low vol relative to long-run → positive (risk-on).
        """
        lr_vol = self.params.long_run_vol
        if lr_vol < 1e-8:
            return 0.0
        ratio = self.current_vol / lr_vol
        # ratio < 1 → low vol → positive signal; ratio > 1 → high vol → negative
        signal = np.clip(1.0 - ratio, -1, 1)
        return float(signal)

    @property
    def confidence(self) -> float:
        """Confidence based on model fit and parameter quality."""
        # Good persistence (0.9-0.99) → higher confidence
        p = self.params.persistence
        if p < 0.5 or p >= 1.0:
            return 0.3
        return float(np.clip(p, 0.5, 0.99))


class GARCHModel:
    """
    GARCH(1,1) with MLE estimation.
    
    The conditional variance evolves as:
        σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    
    where ε_t = r_t - μ (demeaned returns).
    
    Log-likelihood (Gaussian):
        L = -0.5 * Σ [log(2π) + log(σ²_t) + ε²_t / σ²_t]
    """

    def __init__(self, lookback_days: int = 504):
        self.lookback_days = lookback_days
        self._cache: Dict[str, GARCHForecast] = {}

    def _fetch_returns(self, symbol: str) -> np.ndarray:
        """Fetch daily log returns."""
        if yf is None:
            raise ImportError("yfinance required")
        end = datetime.now()
        start = end - timedelta(days=int(self.lookback_days * 1.5))
        data = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if data.empty:
            raise ValueError(f"No data for {symbol}")
        prices = data["Close"].dropna()
        log_ret = np.log(prices / prices.shift(1)).dropna().values
        return log_ret[-self.lookback_days:]

    @staticmethod
    def _garch_log_likelihood(
        params: np.ndarray, returns: np.ndarray
    ) -> float:
        """
        Negative log-likelihood for GARCH(1,1) with Gaussian innovations.
        params = [omega, alpha, beta]
        """
        omega, alpha, beta = params
        T = len(returns)
        eps = returns - np.mean(returns)  # demean

        # Initialize variance at unconditional level
        persistence = alpha + beta
        if persistence >= 1.0:
            sigma2_0 = np.var(eps)
        else:
            sigma2_0 = omega / (1.0 - persistence)

        sigma2 = np.empty(T)
        sigma2[0] = max(sigma2_0, 1e-10)

        for t in range(1, T):
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
            sigma2[t] = max(sigma2[t], 1e-10)  # positivity

        # Gaussian log-likelihood
        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps ** 2 / sigma2)

        if not np.isfinite(ll):
            return 1e10
        return -ll  # minimize negative LL

    def fit(self, returns: np.ndarray) -> Tuple[GARCHParams, float, np.ndarray]:
        """
        Fit GARCH(1,1) via MLE.
        
        Returns: (params, log_likelihood, conditional_variances)
        """
        var_ret = np.var(returns)
        if var_ret < 1e-12:
            raise ValueError("Zero variance in returns")

        # Initial guess: omega=10% of var, alpha=0.08, beta=0.88
        x0 = np.array([var_ret * 0.05, 0.08, 0.88])

        # Constraints: omega>0, alpha>0, beta>0, alpha+beta<1
        bounds = [
            (1e-10, var_ret * 10),  # omega
            (1e-6, 0.5),            # alpha
            (0.5, 0.9999),          # beta
        ]
        constraints = [
            {"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]},  # alpha+beta < 1
        ]

        result = minimize(
            self._garch_log_likelihood,
            x0,
            args=(returns,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not result.success:
            logger.warning(f"GARCH optimization did not converge: {result.message}")

        omega, alpha, beta = result.x
        params = GARCHParams(omega=omega, alpha=alpha, beta=beta)

        # Reconstruct conditional variances
        eps = returns - np.mean(returns)
        T = len(eps)
        persistence = alpha + beta
        sigma2_0 = omega / (1.0 - persistence) if persistence < 1 else np.var(eps)

        sigma2 = np.empty(T)
        sigma2[0] = sigma2_0
        for t in range(1, T):
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
            sigma2[t] = max(sigma2[t], 1e-10)

        log_ll = -result.fun  # convert back from negative

        return params, log_ll, sigma2

    def forecast(
        self, params: GARCHParams, last_eps: float, last_sigma2: float, horizon: int = 10
    ) -> List[float]:
        """
        Multi-step ahead variance forecast.
        
        σ²_{t+k} = ω·Σ_{j=0}^{k-2} (α+β)^j + (α+β)^{k-1} · σ²_{t+1}
        
        where σ²_{t+1} = ω + α·ε²_t + β·σ²_t
        
        Returns annualized vol for each step.
        """
        omega = params.omega
        alpha = params.alpha
        beta = params.beta
        ab = alpha + beta

        # One-step ahead
        sigma2_1 = omega + alpha * last_eps ** 2 + beta * last_sigma2
        long_run_var = params.long_run_variance

        forecasts = []
        for k in range(1, horizon + 1):
            if k == 1:
                var_k = sigma2_1
            else:
                # Mean-reverting forecast
                var_k = long_run_var + (ab ** (k - 1)) * (sigma2_1 - long_run_var)
            vol_annual = np.sqrt(max(var_k, 0) * 252)
            forecasts.append(float(vol_annual))

        return forecasts

    def fit_and_forecast(self, symbol: str, horizon: int = 10) -> GARCHForecast:
        """Full pipeline: fetch data, fit GARCH(1,1), forecast volatility."""
        returns = self._fetch_returns(symbol)
        params, log_ll, sigma2 = self.fit(returns)

        eps = returns - np.mean(returns)
        last_eps = eps[-1]
        last_sigma2 = sigma2[-1]

        current_vol = np.sqrt(last_sigma2 * 252)
        forecast_vols = self.forecast(params, last_eps, last_sigma2, horizon)

        # VaR and ES
        daily_vol = np.sqrt(last_sigma2)
        mu = np.mean(returns)
        var_95 = -(mu + daily_vol * norm.ppf(0.05))
        var_99 = -(mu + daily_vol * norm.ppf(0.01))
        # ES = E[loss | loss > VaR] = mu + sigma * phi(z_alpha) / alpha
        es_95 = -(mu - daily_vol * norm.pdf(norm.ppf(0.05)) / 0.05)

        result = GARCHForecast(
            symbol=symbol,
            params=params,
            current_vol=float(current_vol),
            forecast_vols=forecast_vols,
            horizon=horizon,
            var_95=float(var_95),
            var_99=float(var_99),
            es_95=float(es_95),
            log_likelihood=log_ll,
            n_observations=len(returns),
        )
        self._cache[symbol] = result

        logger.info(
            f"GARCH(1,1) {symbol}: ω={params.omega:.2e}, α={params.alpha:.4f}, "
            f"β={params.beta:.4f}, persistence={params.persistence:.4f}, "
            f"current_vol={current_vol:.1%}, half_life={params.half_life:.1f}d"
        )
        return result

    def get_vol_forecast(self, symbol: str, horizon: int = 1) -> float:
        """Get annualized vol forecast (cached or fresh)."""
        if symbol in self._cache:
            fc = self._cache[symbol]
            if horizon <= len(fc.forecast_vols):
                return fc.forecast_vols[horizon - 1]
        result = self.fit_and_forecast(symbol, max(horizon, 5))
        return result.forecast_vols[min(horizon - 1, len(result.forecast_vols) - 1)]

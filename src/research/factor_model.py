"""
Phase M — Factor Model & Alpha Research Engine.

Item 1: FamaFrenchFactorModel — FF5 factors, 60-day rolling OLS, alpha_tstat > 2.0 gate.
Item 2: AlphaDecayTracker — exponential decay half-life, Discord alert < 3 days.
Item 3: CrossSectionalMomentum — 12-1 month momentum, top/bottom decile, monthly rebalance.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Item 1 — FamaFrenchFactorModel
# ---------------------------------------------------------------------------

@dataclass
class FactorExposure:
    """Result of a factor regression."""
    alpha: float = 0.0
    alpha_tstat: float = 0.0
    beta_mkt: float = 0.0
    beta_smb: float = 0.0
    beta_hml: float = 0.0
    beta_rmw: float = 0.0
    beta_cma: float = 0.0
    r_squared: float = 0.0
    residual_vol: float = 0.0
    n_obs: int = 0


class FamaFrenchFactorModel:
    """Fama-French 5-factor model with rolling OLS regression.

    Factors: Mkt-RF, SMB, HML, RMW, CMA (+ alpha intercept).
    Uses 60-day rolling window by default.
    Gate: alpha_tstat() > 2.0 required to trade.
    """

    FACTOR_NAMES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

    def __init__(self, window: int = 60, min_observations: int = 30):
        self.window = window
        self.min_observations = min_observations
        self._last_exposure: Optional[FactorExposure] = None

    def fit(
        self,
        returns: np.ndarray,
        factor_returns: np.ndarray,
    ) -> FactorExposure:
        """Run OLS regression: R_i - Rf = alpha + sum(beta_k * F_k) + eps.

        Args:
            returns: Asset excess returns (T,).
            factor_returns: Factor returns (T, 5) in order Mkt-RF, SMB, HML, RMW, CMA.

        Returns:
            FactorExposure with coefficients and statistics.
        """
        returns = np.asarray(returns, dtype=np.float64)
        factor_returns = np.asarray(factor_returns, dtype=np.float64)

        if factor_returns.ndim == 1:
            factor_returns = factor_returns.reshape(-1, 1)

        # Use last `window` observations
        n = len(returns)
        if n > self.window:
            returns = returns[-self.window:]
            factor_returns = factor_returns[-self.window:]
            n = self.window

        if n < self.min_observations:
            logger.warning("Insufficient observations: %d < %d", n, self.min_observations)
            self._last_exposure = FactorExposure(n_obs=n)
            return self._last_exposure

        # OLS: y = X @ beta, X includes intercept
        X = np.column_stack([np.ones(n), factor_returns])
        y = returns

        # Normal equations: beta = (X'X)^-1 X'y
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X.T @ X)

        beta = XtX_inv @ X.T @ y
        residuals = y - X @ beta
        residual_var = np.sum(residuals ** 2) / max(n - X.shape[1], 1)
        se = np.sqrt(np.diag(XtX_inv) * residual_var)

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_sq = 1.0 - ss_res / max(ss_tot, 1e-12)

        n_factors = factor_returns.shape[1]
        exposure = FactorExposure(
            alpha=float(beta[0]),
            alpha_tstat=float(beta[0] / max(se[0], 1e-12)),
            beta_mkt=float(beta[1]) if n_factors >= 1 else 0.0,
            beta_smb=float(beta[2]) if n_factors >= 2 else 0.0,
            beta_hml=float(beta[3]) if n_factors >= 3 else 0.0,
            beta_rmw=float(beta[4]) if n_factors >= 4 else 0.0,
            beta_cma=float(beta[5]) if n_factors >= 5 else 0.0,
            r_squared=float(max(r_sq, 0.0)),
            residual_vol=float(np.sqrt(residual_var)) if residual_var > 0 else 0.0,
            n_obs=n,
        )
        self._last_exposure = exposure
        logger.info(
            "FF5 fit: alpha=%.4f (t=%.2f), R²=%.3f, n=%d",
            exposure.alpha, exposure.alpha_tstat, exposure.r_squared, n,
        )
        return exposure

    def alpha_tstat(self) -> float:
        """Return t-statistic of alpha from last fit."""
        if self._last_exposure is None:
            return 0.0
        return self._last_exposure.alpha_tstat

    def should_trade(self, threshold: float = 2.0) -> bool:
        """Gate: only trade if |alpha_tstat| > threshold."""
        return abs(self.alpha_tstat()) > threshold

    def get_factor_betas(self) -> Dict[str, float]:
        """Return dictionary of factor betas from last fit."""
        if self._last_exposure is None:
            return {name: 0.0 for name in self.FACTOR_NAMES}
        exp = self._last_exposure
        return {
            "Mkt-RF": exp.beta_mkt,
            "SMB": exp.beta_smb,
            "HML": exp.beta_hml,
            "RMW": exp.beta_rmw,
            "CMA": exp.beta_cma,
        }

    @property
    def last_exposure(self) -> Optional[FactorExposure]:
        return self._last_exposure


# ---------------------------------------------------------------------------
# Item 2 — AlphaDecayTracker
# ---------------------------------------------------------------------------

@dataclass
class DecayEstimate:
    """Alpha decay estimation result."""
    half_life_days: float = 0.0
    decay_rate: float = 0.0  # lambda in exp(-lambda * t)
    current_alpha: float = 0.0
    peak_alpha: float = 0.0
    days_since_peak: int = 0
    is_critical: bool = False  # half_life < 3 days
    timestamp: str = ""


class AlphaDecayTracker:
    """Track alpha signal decay and alert when half-life < 3 days.

    Fits exponential decay: alpha(t) = A * exp(-lambda * t)
    Half-life = ln(2) / lambda.
    Persists history to logs/alpha_decay.json.
    """

    def __init__(
        self,
        critical_half_life: float = 3.0,
        log_path: str = "logs/alpha_decay.json",
    ):
        self.critical_half_life = critical_half_life
        self.log_path = log_path
        self._history: List[Dict[str, Any]] = []
        self._alphas: List[float] = []
        self._timestamps: List[float] = []

    def record_alpha(self, alpha: float, timestamp: Optional[float] = None) -> None:
        """Record an alpha observation."""
        ts = timestamp if timestamp is not None else time.time()
        self._alphas.append(alpha)
        self._timestamps.append(ts)

    def estimate_decay(self) -> DecayEstimate:
        """Estimate exponential decay from recorded alphas.

        Uses log-linear regression: ln|alpha| = ln(A) - lambda * t.
        """
        if len(self._alphas) < 3:
            return DecayEstimate(
                half_life_days=float("inf"),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        alphas = np.array(self._alphas)
        times = np.array(self._timestamps)

        # Convert timestamps to days from first observation
        t_days = (times - times[0]) / 86400.0

        # Use absolute values for log-linear fit
        abs_alphas = np.abs(alphas)
        mask = abs_alphas > 1e-10
        if mask.sum() < 3:
            return DecayEstimate(
                half_life_days=float("inf"),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        log_alphas = np.log(abs_alphas[mask])
        t_valid = t_days[mask]

        # Linear regression: log_alpha = a - lambda * t
        n = len(t_valid)
        if n < 2:
            return DecayEstimate(
                half_life_days=float("inf"),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        x_mean = np.mean(t_valid)
        y_mean = np.mean(log_alphas)
        slope = np.sum((t_valid - x_mean) * (log_alphas - y_mean)) / max(
            np.sum((t_valid - x_mean) ** 2), 1e-12
        )

        decay_rate = max(-slope, 1e-12)  # lambda > 0 for decay
        half_life = np.log(2) / decay_rate

        peak_alpha = float(np.max(np.abs(alphas)))
        peak_idx = int(np.argmax(np.abs(alphas)))
        days_since_peak = (t_days[-1] - t_days[peak_idx]) if len(t_days) > 0 else 0

        estimate = DecayEstimate(
            half_life_days=float(half_life),
            decay_rate=float(decay_rate),
            current_alpha=float(alphas[-1]),
            peak_alpha=peak_alpha,
            days_since_peak=int(days_since_peak),
            is_critical=half_life < self.critical_half_life,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self._history.append({
            "half_life_days": float(estimate.half_life_days),
            "decay_rate": float(estimate.decay_rate),
            "current_alpha": float(estimate.current_alpha),
            "is_critical": bool(estimate.is_critical),
            "timestamp": str(estimate.timestamp),
        })

        if estimate.is_critical:
            logger.warning(
                "ALPHA DECAY CRITICAL: half-life=%.1f days (< %.1f threshold)",
                estimate.half_life_days, self.critical_half_life,
            )

        return estimate

    def save_log(self) -> None:
        """Persist decay history to JSON file."""
        path = Path(self.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"history": self._history}, f, indent=2)
        logger.info("Alpha decay log saved to %s", self.log_path)

    def load_log(self) -> List[Dict[str, Any]]:
        """Load decay history from JSON file."""
        path = Path(self.log_path)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            self._history = data.get("history", [])
        return self._history

    @property
    def history(self) -> List[Dict[str, Any]]:
        return self._history


# ---------------------------------------------------------------------------
# Item 3 — CrossSectionalMomentum
# ---------------------------------------------------------------------------

@dataclass
class MomentumSignal:
    """Cross-sectional momentum signal for a single asset."""
    symbol: str
    momentum_score: float  # 12-1 month return
    decile: int  # 1 (bottom) to 10 (top)
    signal: str  # 'long', 'short', 'neutral'
    rank: int = 0


@dataclass
class MomentumPortfolio:
    """Monthly cross-sectional momentum portfolio."""
    long_symbols: List[str] = field(default_factory=list)
    short_symbols: List[str] = field(default_factory=list)
    signals: List[MomentumSignal] = field(default_factory=list)
    rebalance_date: str = ""
    n_assets: int = 0


class CrossSectionalMomentum:
    """Cross-sectional momentum strategy: 12-1 month momentum.

    - Compute 12-month return skipping the most recent month (momentum effect).
    - Rank assets, go long top decile, short bottom decile.
    - Rebalance monthly.
    """

    def __init__(
        self,
        lookback_months: int = 12,
        skip_months: int = 1,
        top_percentile: float = 0.1,
        bottom_percentile: float = 0.1,
        trading_days_per_month: int = 21,
    ):
        self.lookback_months = lookback_months
        self.skip_months = skip_months
        self.top_percentile = top_percentile
        self.bottom_percentile = bottom_percentile
        self.trading_days_per_month = trading_days_per_month
        self._last_portfolio: Optional[MomentumPortfolio] = None

    def compute_momentum(
        self,
        returns: Dict[str, np.ndarray],
    ) -> List[MomentumSignal]:
        """Compute 12-1 month momentum for each asset.

        Args:
            returns: Dict of symbol -> daily returns array.

        Returns:
            Sorted list of MomentumSignal (highest momentum first).
        """
        lookback_days = self.lookback_months * self.trading_days_per_month
        skip_days = self.skip_months * self.trading_days_per_month

        signals = []
        for symbol, ret in returns.items():
            ret = np.asarray(ret, dtype=np.float64)
            min_required = lookback_days + skip_days
            if len(ret) < min_required:
                continue

            # 12-1 month return: cumulative return from t-252 to t-21
            momentum_returns = ret[-(lookback_days + skip_days):-skip_days] if skip_days > 0 else ret[-lookback_days:]
            cum_return = float(np.prod(1 + momentum_returns) - 1)
            signals.append(MomentumSignal(
                symbol=symbol,
                momentum_score=cum_return,
                decile=0,
                signal="neutral",
            ))

        # Rank and assign deciles
        signals.sort(key=lambda s: s.momentum_score, reverse=True)
        n = len(signals)
        if n == 0:
            return signals

        for rank, sig in enumerate(signals):
            sig.rank = rank + 1
            sig.decile = min(int(rank / max(n, 1) * 10) + 1, 10)

        # Assign signals
        top_n = max(int(n * self.top_percentile), 1)
        bottom_n = max(int(n * self.bottom_percentile), 1)

        for sig in signals[:top_n]:
            sig.signal = "long"
        for sig in signals[-bottom_n:]:
            sig.signal = "short"

        return signals

    def generate_portfolio(
        self,
        returns: Dict[str, np.ndarray],
    ) -> MomentumPortfolio:
        """Generate long/short momentum portfolio.

        Args:
            returns: Dict of symbol -> daily returns array.

        Returns:
            MomentumPortfolio with long and short lists.
        """
        signals = self.compute_momentum(returns)

        portfolio = MomentumPortfolio(
            long_symbols=[s.symbol for s in signals if s.signal == "long"],
            short_symbols=[s.symbol for s in signals if s.signal == "short"],
            signals=signals,
            rebalance_date=datetime.now(timezone.utc).isoformat(),
            n_assets=len(signals),
        )
        self._last_portfolio = portfolio
        logger.info(
            "Momentum portfolio: %d long, %d short out of %d assets",
            len(portfolio.long_symbols), len(portfolio.short_symbols), portfolio.n_assets,
        )
        return portfolio

    @property
    def last_portfolio(self) -> Optional[MomentumPortfolio]:
        return self._last_portfolio

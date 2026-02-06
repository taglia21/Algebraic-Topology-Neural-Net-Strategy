"""
Capital Asset Pricing Model (CAPM)
===================================
Beta regression vs SPY benchmark.
Expected return = risk_free + beta * (market_return - risk_free)

Provides:
- Rolling beta estimation via OLS regression
- Expected return calculation
- Jensen's alpha
- Treynor ratio
- Stock screening by expected return
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
except ImportError:
    yf = None


@dataclass
class CAPMResult:
    """Result of CAPM analysis for a single asset."""
    symbol: str
    beta: float
    alpha: float  # Jensen's alpha (annualized)
    expected_return: float  # annualized
    residual_std: float  # idiosyncratic risk (annualized)
    r_squared: float
    treynor_ratio: float
    sharpe_ratio: float
    market_return: float
    risk_free_rate: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def signal(self) -> float:
        """Signal in [-1, 1] based on alpha and expected return rank."""
        # Positive alpha → bullish, negative → bearish
        alpha_signal = np.clip(self.alpha * 10, -1, 1)
        return float(alpha_signal)

    @property
    def confidence(self) -> float:
        """Confidence in [0, 1] based on R² and data quality."""
        return float(np.clip(self.r_squared, 0, 1))


class CAPMModel:
    """
    CAPM implementation with OLS beta regression.
    
    Uses daily log returns regressed against SPY returns.
    """

    def __init__(
        self,
        benchmark: str = "SPY",
        risk_free_rate: Optional[float] = None,
        lookback_days: int = 252,
        min_observations: int = 60,
    ):
        self.benchmark = benchmark
        self._rf_override = risk_free_rate
        self.lookback_days = lookback_days
        self.min_observations = min_observations
        self._cache: Dict[str, CAPMResult] = {}

    @property
    def risk_free_rate(self) -> float:
        """Annual risk-free rate. Uses 10Y Treasury yield or override."""
        if self._rf_override is not None:
            return self._rf_override
        try:
            tnx = yf.Ticker("^TNX")
            hist = tnx.history(period="5d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1]) / 100.0
        except Exception:
            pass
        return 0.045  # fallback

    def _fetch_returns(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch daily log returns for symbols + benchmark."""
        all_syms = list(set(symbols + [self.benchmark]))
        end = datetime.now()
        start = end - timedelta(days=int(self.lookback_days * 1.5))

        if yf is None:
            raise ImportError("yfinance required for CAPM")

        data = yf.download(all_syms, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data[["Close"]].copy()
            prices.columns = all_syms[:1]

        prices = prices.dropna(how="all").ffill()
        log_returns = np.log(prices / prices.shift(1)).dropna()
        return log_returns.tail(self.lookback_days)

    def _ols_regression(
        self, asset_returns: np.ndarray, market_returns: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        OLS: r_asset = alpha + beta * r_market + epsilon

        Returns: (beta, alpha_daily, residual_std_daily, r_squared)
        """
        n = len(asset_returns)
        X = np.column_stack([np.ones(n), market_returns])
        # Normal equations: (X'X)^{-1} X'y
        XtX = X.T @ X
        Xty = X.T @ asset_returns
        try:
            coeffs = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.lstsq(X, asset_returns, rcond=None)[0]

        alpha_daily = coeffs[0]
        beta = coeffs[1]

        predicted = X @ coeffs
        residuals = asset_returns - predicted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((asset_returns - np.mean(asset_returns)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        residual_std = np.std(residuals, ddof=2)

        return float(beta), float(alpha_daily), float(residual_std), float(r_squared)

    def analyze(self, symbol: str) -> CAPMResult:
        """Run CAPM analysis for a single symbol."""
        returns_df = self._fetch_returns([symbol])

        if symbol not in returns_df.columns or self.benchmark not in returns_df.columns:
            raise ValueError(f"Missing data for {symbol} or {self.benchmark}")

        asset_ret = returns_df[symbol].values
        mkt_ret = returns_df[self.benchmark].values

        # Align & remove NaN
        mask = np.isfinite(asset_ret) & np.isfinite(mkt_ret)
        asset_ret = asset_ret[mask]
        mkt_ret = mkt_ret[mask]

        if len(asset_ret) < self.min_observations:
            raise ValueError(
                f"Insufficient data: {len(asset_ret)} < {self.min_observations}"
            )

        beta, alpha_daily, resid_std_daily, r2 = self._ols_regression(asset_ret, mkt_ret)

        rf = self.risk_free_rate
        rf_daily = rf / 252.0

        # Annualize
        alpha_annual = alpha_daily * 252
        resid_std_annual = resid_std_daily * np.sqrt(252)
        mkt_annual = float(np.mean(mkt_ret)) * 252

        # CAPM expected return
        expected_return = rf + beta * (mkt_annual - rf)

        # Ratios
        asset_annual_return = float(np.mean(asset_ret)) * 252
        asset_annual_std = float(np.std(asset_ret)) * np.sqrt(252)
        excess_return = asset_annual_return - rf

        treynor = excess_return / beta if abs(beta) > 1e-8 else 0.0
        sharpe = excess_return / asset_annual_std if asset_annual_std > 1e-8 else 0.0

        result = CAPMResult(
            symbol=symbol,
            beta=beta,
            alpha=alpha_annual,
            expected_return=expected_return,
            residual_std=resid_std_annual,
            r_squared=r2,
            treynor_ratio=treynor,
            sharpe_ratio=sharpe,
            market_return=mkt_annual,
            risk_free_rate=rf,
        )

        self._cache[symbol] = result
        logger.info(
            f"CAPM {symbol}: β={beta:.3f}, α={alpha_annual:.4f}, "
            f"E[r]={expected_return:.4f}, R²={r2:.3f}"
        )
        return result

    def screen_stocks(
        self, symbols: List[str], min_expected_return: float = 0.08
    ) -> List[CAPMResult]:
        """Screen stocks by CAPM expected return. Returns sorted list."""
        results = []
        for sym in symbols:
            try:
                result = self.analyze(sym)
                if result.expected_return >= min_expected_return:
                    results.append(result)
            except Exception as exc:
                logger.warning(f"CAPM screen failed for {sym}: {exc}")
        results.sort(key=lambda r: r.expected_return, reverse=True)
        return results

    def get_beta(self, symbol: str) -> float:
        """Get cached or freshly computed beta."""
        if symbol in self._cache:
            return self._cache[symbol].beta
        return self.analyze(symbol).beta

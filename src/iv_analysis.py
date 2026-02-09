"""
IV (Implied Volatility) Analysis Engine
=========================================

Production IV Rank/Percentile calculations for options strategy selection.

Uses multiple data sources:
1. IVDataManager SQLite cache (primary - from Alpaca options chain)
2. yfinance VIX as proxy for broad market IV
3. Historical volatility calculation as fallback

Key Metrics:
- IV Rank: (Current IV - 52wk Low) / (52wk High - 52wk Low) * 100
- IV Percentile: % of days in past year where IV was BELOW current level
- HV/IV Ratio: Historical vol vs implied vol (>1 means IV is cheap)

Reference: Tastytrade research shows selling premium at IV Rank > 50
has a 77% win rate on SPY iron condors at 45 DTE managed at 50% profit.

Author: System Overhaul - Feb 2026
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
except ImportError:
    yf = None

# Try to use existing IV data manager
try:
    from src.options.iv_data_manager import IVDataManager
    _IV_DATA_MANAGER_AVAILABLE = True
except ImportError:
    _IV_DATA_MANAGER_AVAILABLE = False


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class IVMetrics:
    """Complete IV analysis for a single underlying."""
    symbol: str
    iv_rank: float              # 0-100, (current - low) / (high - low)
    iv_percentile: float        # 0-100, % of days below current
    current_iv: float           # Current annualized IV (decimal, e.g. 0.20 = 20%)
    iv_52w_high: float          # 52-week high IV
    iv_52w_low: float           # 52-week low IV
    iv_52w_mean: float          # 52-week average IV
    hv_20d: float               # 20-day historical (realized) volatility
    hv_iv_ratio: float          # HV/IV ratio (>1 = IV cheap, <1 = IV expensive)
    data_quality: str           # "live", "cached", "proxy", "synthetic"
    timestamp: datetime


@dataclass
class MarketIVSnapshot:
    """Broad market IV snapshot using VIX."""
    vix_level: float            # Current VIX level
    vix_rank: float             # VIX rank (0-100) over past year
    vix_percentile: float       # VIX percentile over past year
    vix_term_slope: float       # VIX/VIX3M ratio (<1 = contango, >1 = backwardation)
    timestamp: datetime


# ============================================================================
# IV ANALYSIS ENGINE
# ============================================================================

class IVAnalysisEngine:
    """
    Production IV analysis engine for options trading.

    Combines multiple data sources to provide reliable IV metrics:
    1. IVDataManager (SQLite cache from Alpaca options chain data)
    2. VIX as market-wide IV proxy
    3. yfinance historical data for HV calculations

    Usage:
        engine = IVAnalysisEngine()
        metrics = engine.get_iv_metrics("SPY")
        if metrics.iv_rank > 50:
            # Sell premium
        elif metrics.iv_rank < 30:
            # Buy premium
    """

    def __init__(self):
        """Initialize IV analysis engine."""
        self.logger = logging.getLogger(__name__)

        # Primary: IVDataManager with SQLite cache
        self._iv_manager: Optional[IVDataManager] = None
        if _IV_DATA_MANAGER_AVAILABLE:
            try:
                self._iv_manager = IVDataManager()
                self.logger.info("IV Analysis: Using IVDataManager (SQLite cache)")
            except Exception as e:
                self.logger.warning(f"IVDataManager init failed: {e}")

        # Cache for computed metrics
        self._metrics_cache: Dict[str, Tuple[IVMetrics, datetime]] = {}
        self._market_cache: Optional[Tuple[MarketIVSnapshot, datetime]] = None
        self._cache_ttl = timedelta(minutes=5)

        # Price data cache for HV calculations
        self._price_cache: Dict[str, Tuple[np.ndarray, datetime]] = {}
        self._price_cache_ttl = timedelta(minutes=15)

    # ================================================================== #
    # PUBLIC API
    # ================================================================== #

    def get_iv_metrics(self, symbol: str) -> Optional[IVMetrics]:
        """
        Get complete IV metrics for a symbol.

        Tries sources in order:
        1. IVDataManager (real options chain IV, cached in SQLite)
        2. VIX proxy (for broad ETFs)
        3. Historical volatility estimate

        Args:
            symbol: Underlying ticker (e.g., "SPY")

        Returns:
            IVMetrics or None if all sources fail
        """
        # Check cache
        if symbol in self._metrics_cache:
            cached, ts = self._metrics_cache[symbol]
            if datetime.now() - ts < self._cache_ttl:
                return cached

        metrics = self._compute_iv_metrics(symbol)
        if metrics is not None:
            self._metrics_cache[symbol] = (metrics, datetime.now())

        return metrics

    def get_market_iv_snapshot(self) -> Optional[MarketIVSnapshot]:
        """
        Get broad market IV snapshot using VIX.

        Returns:
            MarketIVSnapshot or None
        """
        if self._market_cache is not None:
            cached, ts = self._market_cache
            if datetime.now() - ts < self._cache_ttl:
                return cached

        snapshot = self._compute_market_snapshot()
        if snapshot is not None:
            self._market_cache = (snapshot, datetime.now())
        return snapshot

    def get_iv_rank(self, symbol: str) -> Optional[float]:
        """Convenience: get just the IV rank (0-100) for a symbol."""
        metrics = self.get_iv_metrics(symbol)
        return metrics.iv_rank if metrics else None

    def is_high_iv(self, symbol: str, threshold: float = 50.0) -> bool:
        """Check if IV rank is above threshold (sell premium zone)."""
        rank = self.get_iv_rank(symbol)
        return rank is not None and rank >= threshold

    def is_low_iv(self, symbol: str, threshold: float = 30.0) -> bool:
        """Check if IV rank is below threshold (buy premium zone)."""
        rank = self.get_iv_rank(symbol)
        return rank is not None and rank <= threshold

    # ================================================================== #
    # PRIVATE: IV Metrics Computation
    # ================================================================== #

    def _compute_iv_metrics(self, symbol: str) -> Optional[IVMetrics]:
        """Compute IV metrics from best available source."""

        # Source 1: IVDataManager (real options chain IV)
        if self._iv_manager is not None:
            try:
                iv_rank = self._iv_manager.get_iv_rank(symbol, lookback_days=252)
                current_iv = self._iv_manager.get_current_iv(symbol)

                if iv_rank is not None and current_iv is not None:
                    # Get full history for percentile and range calculations
                    history = self._get_iv_history_from_manager(symbol)
                    if history is not None and len(history) >= 20:
                        iv_percentile = self._calculate_percentile(history, current_iv)
                        iv_high = float(np.max(history))
                        iv_low = float(np.min(history))
                        iv_mean = float(np.mean(history))
                    else:
                        iv_percentile = iv_rank  # Approximate
                        iv_high = current_iv * 1.5
                        iv_low = current_iv * 0.5
                        iv_mean = current_iv

                    # Calculate HV for comparison
                    hv_20d = self._calculate_hv(symbol, window=20)
                    if hv_20d is None:
                        hv_20d = current_iv * 0.85  # Estimate

                    hv_iv_ratio = hv_20d / current_iv if current_iv > 0 else 1.0

                    return IVMetrics(
                        symbol=symbol,
                        iv_rank=iv_rank,
                        iv_percentile=iv_percentile,
                        current_iv=current_iv,
                        iv_52w_high=iv_high,
                        iv_52w_low=iv_low,
                        iv_52w_mean=iv_mean,
                        hv_20d=hv_20d,
                        hv_iv_ratio=hv_iv_ratio,
                        data_quality="cached",
                        timestamp=datetime.now(),
                    )
            except Exception as e:
                self.logger.debug(f"IVDataManager lookup failed for {symbol}: {e}")

        # Source 2: VIX proxy for broad market ETFs
        if symbol in ("SPY", "QQQ", "IWM", "DIA"):
            try:
                return self._compute_from_vix_proxy(symbol)
            except Exception as e:
                self.logger.debug(f"VIX proxy failed for {symbol}: {e}")

        # Source 3: Historical volatility as IV estimate
        try:
            return self._compute_from_historical(symbol)
        except Exception as e:
            self.logger.debug(f"Historical fallback failed for {symbol}: {e}")

        self.logger.warning(f"All IV sources failed for {symbol}")
        return None

    def _compute_from_vix_proxy(self, symbol: str) -> Optional[IVMetrics]:
        """Compute IV metrics using VIX as a proxy."""
        if yf is None:
            return None

        try:
            # Download VIX history (1 year)
            vix_data = yf.download("^VIX", period="1y", interval="1d", progress=False)
            if vix_data.empty or len(vix_data) < 20:
                return None

            vix_values = vix_data["Close"].values.flatten()
            current_vix = float(vix_values[-1])
            current_iv = current_vix / 100.0  # Convert to decimal

            # VIX adjustments per symbol
            iv_multiplier = {
                "SPY": 1.0,
                "QQQ": 1.15,   # QQQ typically ~15% higher vol than SPY
                "IWM": 1.20,   # Small caps ~20% higher
                "DIA": 0.95,   # DJIA slightly lower
            }.get(symbol, 1.0)

            current_iv *= iv_multiplier
            vix_history = vix_values / 100.0 * iv_multiplier

            iv_rank = self._calculate_rank(vix_history, current_iv)
            iv_percentile = self._calculate_percentile(vix_history, current_iv)

            hv_20d = self._calculate_hv(symbol, window=20) or current_iv * 0.85
            hv_iv_ratio = hv_20d / current_iv if current_iv > 0 else 1.0

            return IVMetrics(
                symbol=symbol,
                iv_rank=iv_rank,
                iv_percentile=iv_percentile,
                current_iv=current_iv,
                iv_52w_high=float(np.max(vix_history)),
                iv_52w_low=float(np.min(vix_history)),
                iv_52w_mean=float(np.mean(vix_history)),
                hv_20d=hv_20d,
                hv_iv_ratio=hv_iv_ratio,
                data_quality="proxy",
                timestamp=datetime.now(),
            )
        except Exception as e:
            self.logger.warning(f"VIX proxy computation failed: {e}")
            return None

    def _compute_from_historical(self, symbol: str) -> Optional[IVMetrics]:
        """Fallback: estimate IV from historical volatility."""
        if yf is None:
            return None

        try:
            prices = self._get_price_data(symbol, period="1y")
            if prices is None or len(prices) < 60:
                return None

            # Calculate rolling HV at different windows
            log_returns = np.diff(np.log(prices))

            hv_20d = float(np.std(log_returns[-20:]) * np.sqrt(252))
            hv_60d = float(np.std(log_returns[-60:]) * np.sqrt(252))

            # IV is typically ~15-20% above HV (variance risk premium)
            estimated_iv = hv_20d * 1.15

            # Build IV history from rolling HV + premium
            rolling_ivs = []
            for i in range(60, len(log_returns)):
                window = log_returns[i-20:i]
                hv = float(np.std(window) * np.sqrt(252))
                rolling_ivs.append(hv * 1.15)

            if len(rolling_ivs) < 20:
                return None

            rolling_ivs = np.array(rolling_ivs)
            iv_rank = self._calculate_rank(rolling_ivs, estimated_iv)
            iv_percentile = self._calculate_percentile(rolling_ivs, estimated_iv)

            hv_iv_ratio = hv_20d / estimated_iv if estimated_iv > 0 else 1.0

            return IVMetrics(
                symbol=symbol,
                iv_rank=iv_rank,
                iv_percentile=iv_percentile,
                current_iv=estimated_iv,
                iv_52w_high=float(np.max(rolling_ivs)),
                iv_52w_low=float(np.min(rolling_ivs)),
                iv_52w_mean=float(np.mean(rolling_ivs)),
                hv_20d=hv_20d,
                hv_iv_ratio=hv_iv_ratio,
                data_quality="synthetic",
                timestamp=datetime.now(),
            )
        except Exception as e:
            self.logger.warning(f"Historical IV fallback failed for {symbol}: {e}")
            return None

    # ================================================================== #
    # PRIVATE: Market Snapshot
    # ================================================================== #

    def _compute_market_snapshot(self) -> Optional[MarketIVSnapshot]:
        """Compute broad market IV snapshot from VIX data."""
        if yf is None:
            return None

        try:
            # VIX (30-day)
            vix_data = yf.download("^VIX", period="1y", interval="1d", progress=False)
            if vix_data.empty:
                return None

            vix_values = vix_data["Close"].values.flatten()
            current_vix = float(vix_values[-1])

            vix_rank = self._calculate_rank(vix_values, current_vix)
            vix_percentile = self._calculate_percentile(vix_values, current_vix)

            # VIX term structure: VIX vs VIX3M
            vix_term_slope = 1.0  # Default: flat
            try:
                vix3m_data = yf.download("^VIX3M", period="5d", interval="1d", progress=False)
                if not vix3m_data.empty:
                    current_vix3m = float(vix3m_data["Close"].values.flatten()[-1])
                    if current_vix3m > 0:
                        vix_term_slope = current_vix / current_vix3m
                        # < 1.0 = contango (normal, low fear)
                        # > 1.0 = backwardation (fear, hedging demand)
            except Exception:
                pass

            return MarketIVSnapshot(
                vix_level=current_vix,
                vix_rank=vix_rank,
                vix_percentile=vix_percentile,
                vix_term_slope=vix_term_slope,
                timestamp=datetime.now(),
            )
        except Exception as e:
            self.logger.warning(f"Market snapshot failed: {e}")
            return None

    # ================================================================== #
    # PRIVATE: Helpers
    # ================================================================== #

    def _get_iv_history_from_manager(self, symbol: str) -> Optional[np.ndarray]:
        """Extract IV history array from IVDataManager."""
        if self._iv_manager is None:
            return None
        try:
            import sqlite3
            with sqlite3.connect(self._iv_manager.db_path) as conn:
                cursor = conn.cursor()
                lookback = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                cursor.execute(
                    "SELECT atm_iv FROM iv_history WHERE symbol = ? AND date >= ? ORDER BY date",
                    (symbol, lookback),
                )
                rows = cursor.fetchall()
                if rows and len(rows) >= 20:
                    return np.array([r[0] for r in rows])
        except Exception as e:
            self.logger.debug(f"IV history extraction failed: {e}")
        return None

    def _calculate_rank(self, history: np.ndarray, current: float) -> float:
        """
        Calculate IV Rank: (current - min) / (max - min) * 100.

        Returns value 0-100.
        """
        iv_min = float(np.min(history))
        iv_max = float(np.max(history))

        if iv_max == iv_min:
            return 50.0  # No info

        rank = ((current - iv_min) / (iv_max - iv_min)) * 100.0
        return round(max(0.0, min(100.0, rank)), 2)

    def _calculate_percentile(self, history: np.ndarray, current: float) -> float:
        """
        Calculate IV Percentile: % of historical observations BELOW current.

        Returns value 0-100.
        """
        below = np.sum(history < current)
        pct = (below / len(history)) * 100.0
        return round(max(0.0, min(100.0, pct)), 2)

    def _calculate_hv(self, symbol: str, window: int = 20) -> Optional[float]:
        """
        Calculate historical (realized) volatility.

        Args:
            symbol: Ticker
            window: Lookback days for HV calculation

        Returns:
            Annualized HV as decimal (e.g. 0.20 = 20%), or None
        """
        prices = self._get_price_data(symbol, period="3mo")
        if prices is None or len(prices) < window + 1:
            return None

        log_returns = np.diff(np.log(prices[-window - 1:]))
        hv = float(np.std(log_returns) * np.sqrt(252))
        return round(hv, 4)

    def _get_price_data(self, symbol: str, period: str = "1y") -> Optional[np.ndarray]:
        """Get historical closing prices from yfinance with caching."""
        if yf is None:
            return None

        # Check cache
        if symbol in self._price_cache:
            cached, ts = self._price_cache[symbol]
            if datetime.now() - ts < self._price_cache_ttl:
                return cached

        try:
            data = yf.download(symbol, period=period, interval="1d", progress=False)
            if data.empty:
                return None
            prices = data["Close"].values.flatten()
            self._price_cache[symbol] = (prices, datetime.now())
            return prices
        except Exception as e:
            self.logger.warning(f"Price data download failed for {symbol}: {e}")
            return None

"""
IV (Implied Volatility) Analysis Engine
=========================================

Production IV Rank/Percentile calculations for options strategy selection.

Fetches REAL per-symbol implied volatility from Alpaca option snapshots,
with a full fallback hierarchy:

  1. Primary   – Real-time IV from Alpaca option chain snapshots (ATM 30-45 DTE)
  2. Secondary – Newton-Raphson / Brent's method IV solve on Black-Scholes
  3. Tertiary  – Historical IV from SQLite (IVDataManager) if market closed
  4. Last resort – VIX proxy with data_quality="proxy" flag

Key Metrics:
- IV Rank: (Current IV - 52wk Low) / (52wk High - 52wk Low) * 100
- IV Percentile: % of days in past year where IV was BELOW current level
- HV/IV Ratio: Historical vol vs implied vol (>1 means IV cheap)

Reference: Tastytrade research shows selling premium at IV Rank > 50
has a 77% win rate on SPY iron condors at 45 DTE managed at 50% profit.

Author: System Overhaul - Feb 2026  |  Fix #1: Real option snapshot IV
"""

import logging
import os
import sqlite3
import math
import numpy as np
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from scipy.optimize import brentq
from scipy.stats import norm, percentileofscore

logger = logging.getLogger(__name__)

# --- Optional imports (graceful degradation) ---
try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionChainRequest, OptionSnapshotRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOptionContractsRequest
    from alpaca.trading.enums import ContractType
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

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
    data_quality: str           # "live", "cached", "brentq", "proxy", "synthetic"
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
# BLACK-SCHOLES IV SOLVER  (standalone, no external BS module needed)
# ============================================================================

def _bs_price(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "call") -> float:
    """Black-Scholes European option price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_volatility_brentq(
    market_price: float, S: float, K: float, T: float,
    r: float = 0.05, option_type: str = "call",
    lo: float = 0.001, hi: float = 5.0,
) -> Optional[float]:
    """
    Solve for IV using Brent's method (scipy.optimize.brentq).

    Returns annualized IV as a decimal (e.g. 0.25 = 25%) or None if no solution.
    """
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    try:
        iv = brentq(
            lambda sigma: _bs_price(S, K, T, r, sigma, option_type) - market_price,
            lo, hi, xtol=1e-6, maxiter=200,
        )
        return float(iv)
    except (ValueError, RuntimeError):
        return None


# ============================================================================
# IV ANALYSIS ENGINE
# ============================================================================

class IVAnalysisEngine:
    """
    Production IV analysis engine for options trading.

    Fetches REAL per-symbol IV from Alpaca option chain snapshots
    (ATM options, 30-45 DTE), averages put + call ATM IV, and stores
    in SQLite for historical IV rank calculation.

    Fallback hierarchy:
      1. Real-time Alpaca option snapshots  → data_quality="live"
      2. Brent IV solve on mid-price        → data_quality="brentq"
      3. IVDataManager SQLite cache         → data_quality="cached"
      4. VIX proxy (broad ETFs only)        → data_quality="proxy"
      5. Historical volatility estimate     → data_quality="synthetic"

    Usage:
        engine = IVAnalysisEngine()
        metrics = engine.get_iv_metrics("SPY")
        if metrics.iv_rank > 50:
            # Sell premium
        elif metrics.iv_rank < 30:
            # Buy premium
    """

    # Target DTE range for ATM option selection
    ATM_DTE_MIN = 25
    ATM_DTE_MAX = 50

    def __init__(self, api_key: str = None, api_secret: str = None):
        """Initialize IV analysis engine with Alpaca clients."""
        self.logger = logging.getLogger(__name__)

        # Alpaca credentials
        self._api_key = api_key or os.getenv("ALPACA_API_KEY")
        self._api_secret = api_secret or os.getenv("ALPACA_SECRET_KEY")

        # --- Alpaca option data client (PRIMARY source) ---
        self._option_data_client = None
        self._trading_client = None
        if _ALPACA_AVAILABLE and self._api_key and self._api_secret:
            try:
                self._option_data_client = OptionHistoricalDataClient(
                    api_key=self._api_key,
                    secret_key=self._api_secret,
                )
                self._trading_client = TradingClient(
                    api_key=self._api_key,
                    secret_key=self._api_secret,
                    paper=True,
                )
                self.logger.info("IV Analysis: Alpaca option data client initialised")
            except Exception as e:
                self.logger.warning(f"Alpaca option client init failed: {e}")

        # --- IVDataManager SQLite cache (tertiary) ---
        self._iv_manager: Optional['IVDataManager'] = None
        if _IV_DATA_MANAGER_AVAILABLE:
            try:
                self._iv_manager = IVDataManager()
                self.logger.info("IV Analysis: IVDataManager (SQLite) available")
            except Exception as e:
                self.logger.warning(f"IVDataManager init failed: {e}")

        # Caches
        self._metrics_cache: Dict[str, Tuple[IVMetrics, datetime]] = {}
        self._market_cache: Optional[Tuple[MarketIVSnapshot, datetime]] = None
        self._cache_ttl = timedelta(minutes=5)
        self._price_cache: Dict[str, Tuple[np.ndarray, datetime]] = {}
        self._price_cache_ttl = timedelta(minutes=15)

    # ================================================================== #
    # PUBLIC API
    # ================================================================== #

    def get_iv_metrics(self, symbol: str) -> Optional[IVMetrics]:
        """
        Get complete IV metrics for a symbol using the full fallback hierarchy.

        Returns:
            IVMetrics or None if every source fails
        """
        if symbol in self._metrics_cache:
            cached, ts = self._metrics_cache[symbol]
            if datetime.now() - ts < self._cache_ttl:
                return cached

        metrics = self._compute_iv_metrics(symbol)
        if metrics is not None:
            self._metrics_cache[symbol] = (metrics, datetime.now())
            # Persist to SQLite for future IV rank calculations
            self._persist_iv_snapshot(symbol, metrics.current_iv)
        return metrics

    def get_market_iv_snapshot(self) -> Optional[MarketIVSnapshot]:
        """Get broad market IV snapshot using VIX."""
        if self._market_cache is not None:
            cached, ts = self._market_cache
            if datetime.now() - ts < self._cache_ttl:
                return cached
        snapshot = self._compute_market_snapshot()
        if snapshot is not None:
            self._market_cache = (snapshot, datetime.now())
        return snapshot

    def get_iv_rank(self, symbol: str) -> Optional[float]:
        """Convenience: get just the IV rank (0-100)."""
        metrics = self.get_iv_metrics(symbol)
        return metrics.iv_rank if metrics else None

    def is_high_iv(self, symbol: str, threshold: float = 50.0) -> bool:
        rank = self.get_iv_rank(symbol)
        return rank is not None and rank >= threshold

    def is_low_iv(self, symbol: str, threshold: float = 30.0) -> bool:
        rank = self.get_iv_rank(symbol)
        return rank is not None and rank <= threshold

    # ================================================================== #
    # PRIVATE: Main computation pipeline
    # ================================================================== #

    def _compute_iv_metrics(self, symbol: str) -> Optional[IVMetrics]:
        """Walk the fallback hierarchy to compute IV metrics."""

        # --- Source 1: Alpaca real-time option snapshot IV ---
        result = self._compute_from_alpaca_snapshots(symbol)
        if result is not None:
            return result

        # --- Source 2: Brent IV solve on option chain mid-prices ---
        result = self._compute_from_brentq(symbol)
        if result is not None:
            return result

        # --- Source 3: IVDataManager SQLite cache ---
        result = self._compute_from_iv_manager(symbol)
        if result is not None:
            return result

        # --- Source 4: VIX proxy (broad ETFs only, last resort) ---
        if symbol in ("SPY", "QQQ", "IWM", "DIA"):
            result = self._compute_from_vix_proxy(symbol)
            if result is not None:
                return result

        # --- Source 5: Historical volatility estimate ---
        result = self._compute_from_historical(symbol)
        if result is not None:
            return result

        self.logger.warning(f"All IV sources failed for {symbol}")
        return None

    # ================================================================== #
    # Source 1: Alpaca option snapshots (REAL IV)
    # ================================================================== #

    def _compute_from_alpaca_snapshots(self, symbol: str) -> Optional[IVMetrics]:
        """
        Fetch real IV from Alpaca option chain snapshots.

        Process:
          1. Get ATM option contracts (30-45 DTE) via trading client
          2. Fetch latest snapshots which contain `implied_volatility` field
          3. Average ATM call + put IV → symbol current_iv
          4. Combine with historical IV from SQLite for rank/percentile
        """
        if self._trading_client is None or self._option_data_client is None:
            return None

        try:
            # Get underlying price
            spot_price = self._get_spot_price(symbol)
            if spot_price is None or spot_price <= 0:
                return None

            # Target expiration window
            today = date.today()
            exp_min = today + timedelta(days=self.ATM_DTE_MIN)
            exp_max = today + timedelta(days=self.ATM_DTE_MAX)

            # Discover ATM contracts via Alpaca
            contracts_req = GetOptionContractsRequest(
                underlying_symbols=[symbol],
                status="active",
                expiration_date_gte=exp_min.isoformat(),
                expiration_date_lte=exp_max.isoformat(),
                strike_price_gte=str(round(spot_price * 0.97, 2)),
                strike_price_lte=str(round(spot_price * 1.03, 2)),
            )
            resp = self._trading_client.get_option_contracts(contracts_req)
            contracts = resp.option_contracts if hasattr(resp, 'option_contracts') else resp

            if not contracts:
                self.logger.debug(f"No ATM contracts found for {symbol}")
                return None

            # Collect OCC symbols and group by type
            call_symbols: List[str] = []
            put_symbols: List[str] = []
            for c in contracts:
                occ = c.symbol
                ctype = getattr(c, 'type', None) or getattr(c, 'contract_type', None)
                ctype_str = str(ctype).lower() if ctype else ""
                if "call" in ctype_str:
                    call_symbols.append(occ)
                elif "put" in ctype_str:
                    put_symbols.append(occ)

            all_symbols = call_symbols + put_symbols
            if not all_symbols:
                return None

            # Fetch snapshots (contains implied_volatility)
            try:
                snap_req = OptionSnapshotRequest(symbol_or_symbols=all_symbols)
                snapshots = self._option_data_client.get_option_snapshot(snap_req)
            except Exception:
                # Older SDK: try positional
                snapshots = self._option_data_client.get_option_snapshot(all_symbols)

            if not snapshots:
                return None

            # Extract implied_volatility values
            call_ivs: List[float] = []
            put_ivs: List[float] = []
            for sym, snap in snapshots.items():
                iv_val = getattr(snap, 'implied_volatility', None)
                if iv_val is None:
                    greeks = getattr(snap, 'greeks', None)
                    if greeks:
                        iv_val = getattr(greeks, 'implied_volatility', None)
                if iv_val is not None and float(iv_val) > 0:
                    iv_float = float(iv_val)
                    if sym in call_symbols:
                        call_ivs.append(iv_float)
                    elif sym in put_symbols:
                        put_ivs.append(iv_float)

            if not call_ivs and not put_ivs:
                self.logger.debug(f"No IV values in snapshots for {symbol}")
                return None

            # Current IV = average of ATM call + put IV
            all_ivs = call_ivs + put_ivs
            current_iv = float(np.mean(all_ivs))

            self.logger.info(
                f"Alpaca snapshot IV for {symbol}: {current_iv:.4f} "
                f"({len(call_ivs)} calls, {len(put_ivs)} puts)"
            )

            return self._build_metrics(symbol, current_iv, data_quality="live")

        except Exception as e:
            self.logger.debug(f"Alpaca snapshot IV failed for {symbol}: {e}")
            return None

    # ================================================================== #
    # Source 2: Newton-Raphson / Brent IV solve
    # ================================================================== #

    def _compute_from_brentq(self, symbol: str) -> Optional[IVMetrics]:
        """
        Calculate IV from option chain mid-prices using Brent's method.

        Fetches ATM option quotes, computes mid-price, and solves
        BS inverse for IV using scipy.optimize.brentq.
        """
        if self._trading_client is None or self._option_data_client is None:
            return None

        try:
            spot = self._get_spot_price(symbol)
            if spot is None or spot <= 0:
                return None

            today = date.today()
            exp_min = today + timedelta(days=self.ATM_DTE_MIN)
            exp_max = today + timedelta(days=self.ATM_DTE_MAX)

            contracts_req = GetOptionContractsRequest(
                underlying_symbols=[symbol],
                status="active",
                expiration_date_gte=exp_min.isoformat(),
                expiration_date_lte=exp_max.isoformat(),
                strike_price_gte=str(round(spot * 0.98, 2)),
                strike_price_lte=str(round(spot * 1.02, 2)),
            )
            resp = self._trading_client.get_option_contracts(contracts_req)
            contracts = resp.option_contracts if hasattr(resp, 'option_contracts') else resp
            if not contracts:
                return None

            # Get quotes for mid-price
            occ_symbols = [c.symbol for c in contracts]
            from alpaca.data.requests import OptionLatestQuoteRequest
            quote_req = OptionLatestQuoteRequest(symbol_or_symbols=occ_symbols)
            quotes = self._option_data_client.get_option_latest_quote(quote_req)

            solved_ivs: List[float] = []
            r = 0.05  # Risk-free rate assumption

            for c in contracts:
                occ = c.symbol
                q = quotes.get(occ)
                if q is None:
                    continue
                bid = float(q.bid_price) if q.bid_price else 0.0
                ask = float(q.ask_price) if q.ask_price else 0.0
                mid = (bid + ask) / 2.0
                if mid <= 0.01:
                    continue

                strike = float(c.strike_price)
                exp_date = c.expiration_date
                if isinstance(exp_date, str):
                    exp_date = datetime.strptime(exp_date, "%Y-%m-%d").date()
                T = max((exp_date - today).days / 365.0, 1 / 365)

                ctype = getattr(c, 'type', None) or getattr(c, 'contract_type', None)
                opt_type = "call" if "call" in str(ctype).lower() else "put"

                iv = implied_volatility_brentq(mid, spot, strike, T, r, opt_type)
                if iv is not None and 0.01 < iv < 3.0:
                    solved_ivs.append(iv)

            if not solved_ivs:
                return None

            current_iv = float(np.mean(solved_ivs))
            self.logger.info(
                f"Brent IV solve for {symbol}: {current_iv:.4f} ({len(solved_ivs)} contracts)"
            )
            return self._build_metrics(symbol, current_iv, data_quality="brentq")

        except Exception as e:
            self.logger.debug(f"Brent IV solve failed for {symbol}: {e}")
            return None

    # ================================================================== #
    # Source 3: IVDataManager (SQLite cache)
    # ================================================================== #

    def _compute_from_iv_manager(self, symbol: str) -> Optional[IVMetrics]:
        """Retrieve IV from IVDataManager's SQLite cache."""
        if self._iv_manager is None:
            return None
        try:
            iv_rank = self._iv_manager.get_iv_rank(symbol, lookback_days=252)
            current_iv = self._iv_manager.get_current_iv(symbol)
            if iv_rank is not None and current_iv is not None:
                return self._build_metrics(symbol, current_iv, data_quality="cached")
        except Exception as e:
            self.logger.debug(f"IVDataManager lookup failed for {symbol}: {e}")
        return None

    # ================================================================== #
    # Source 4: VIX proxy (last resort for broad ETFs)
    # ================================================================== #

    def _compute_from_vix_proxy(self, symbol: str) -> Optional[IVMetrics]:
        """Compute IV using VIX as a proxy. data_quality='proxy'."""
        if yf is None:
            return None
        try:
            vix_data = yf.download("^VIX", period="1y", interval="1d", progress=False)
            if vix_data.empty or len(vix_data) < 20:
                return None
            vix_values = vix_data["Close"].values.flatten()
            current_vix = float(vix_values[-1])
            current_iv = current_vix / 100.0

            iv_multiplier = {"SPY": 1.0, "QQQ": 1.15, "IWM": 1.20, "DIA": 0.95}.get(symbol, 1.0)
            current_iv *= iv_multiplier

            return self._build_metrics(symbol, current_iv, data_quality="proxy")
        except Exception as e:
            self.logger.warning(f"VIX proxy failed: {e}")
            return None

    # ================================================================== #
    # Source 5: Historical volatility
    # ================================================================== #

    def _compute_from_historical(self, symbol: str) -> Optional[IVMetrics]:
        """Estimate IV from historical volatility + variance risk premium."""
        if yf is None:
            return None
        try:
            prices = self._get_price_data(symbol, period="1y")
            if prices is None or len(prices) < 60:
                return None
            log_returns = np.diff(np.log(prices))
            hv_20d = float(np.std(log_returns[-20:]) * np.sqrt(252))
            estimated_iv = hv_20d * 1.15  # ~15% variance risk premium
            return self._build_metrics(symbol, estimated_iv, data_quality="synthetic")
        except Exception as e:
            self.logger.warning(f"Historical IV fallback failed for {symbol}: {e}")
            return None

    # ================================================================== #
    # Builder: assemble IVMetrics from current_iv + history
    # ================================================================== #

    def _build_metrics(self, symbol: str, current_iv: float,
                       data_quality: str) -> IVMetrics:
        """
        Given a current_iv value, combine with historical data from SQLite
        to produce complete IVMetrics with rank, percentile, HV/IV ratio.
        """
        history = self._get_iv_history(symbol)

        if history is not None and len(history) >= 20:
            iv_high = float(np.max(history))
            iv_low = float(np.min(history))
            iv_mean = float(np.mean(history))
            iv_rank = self._calculate_rank(history, current_iv)
            iv_pct = float(percentileofscore(history, current_iv, kind='weak'))
        else:
            # Bootstrap from current value with synthetic range
            iv_high = current_iv * 1.4
            iv_low = current_iv * 0.6
            iv_mean = current_iv
            iv_rank = 50.0
            iv_pct = 50.0

        hv_20d = self._calculate_hv(symbol, window=20) or current_iv * 0.85
        hv_iv_ratio = hv_20d / current_iv if current_iv > 0 else 1.0

        return IVMetrics(
            symbol=symbol,
            iv_rank=round(iv_rank, 2),
            iv_percentile=round(iv_pct, 2),
            current_iv=round(current_iv, 6),
            iv_52w_high=round(iv_high, 6),
            iv_52w_low=round(iv_low, 6),
            iv_52w_mean=round(iv_mean, 6),
            hv_20d=round(hv_20d, 4),
            hv_iv_ratio=round(hv_iv_ratio, 4),
            data_quality=data_quality,
            timestamp=datetime.now(),
        )

    # ================================================================== #
    # Persistence: write today's IV to SQLite
    # ================================================================== #

    def _persist_iv_snapshot(self, symbol: str, current_iv: float) -> None:
        """Store today's IV in the iv_history table for future rank calculations."""
        if self._iv_manager is None:
            return
        try:
            today_str = datetime.now().strftime("%Y-%m-%d")
            with sqlite3.connect(self._iv_manager.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO iv_history "
                    "(symbol, date, atm_iv, skew_25delta, term_structure, call_iv, put_iv) "
                    "VALUES (?, ?, ?, 0, 0, ?, ?)",
                    (symbol, today_str, current_iv, current_iv, current_iv),
                )
                conn.commit()
        except Exception as e:
            self.logger.debug(f"Failed to persist IV snapshot: {e}")

    # ================================================================== #
    # Helpers
    # ================================================================== #

    def _get_spot_price(self, symbol: str) -> Optional[float]:
        """Get current spot price from yfinance (fast)."""
        if yf is None:
            return None
        try:
            data = yf.download(symbol, period="5d", interval="1d", progress=False)
            if data.empty:
                return None
            return float(data["Close"].values.flatten()[-1])
        except Exception:
            return None

    def _get_iv_history(self, symbol: str) -> Optional[np.ndarray]:
        """Get 52-week IV history from SQLite."""
        if self._iv_manager is None:
            return None
        try:
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
        """IV Rank = (current - min) / (max - min) * 100."""
        iv_min = float(np.min(history))
        iv_max = float(np.max(history))
        if iv_max == iv_min:
            return 50.0
        rank = ((current - iv_min) / (iv_max - iv_min)) * 100.0
        return max(0.0, min(100.0, rank))

    def _calculate_hv(self, symbol: str, window: int = 20) -> Optional[float]:
        """Annualised historical (realised) volatility."""
        prices = self._get_price_data(symbol, period="3mo")
        if prices is None or len(prices) < window + 1:
            return None
        log_returns = np.diff(np.log(prices[-window - 1:]))
        return round(float(np.std(log_returns) * np.sqrt(252)), 4)

    def _get_price_data(self, symbol: str, period: str = "1y") -> Optional[np.ndarray]:
        """Closing prices from yfinance with caching."""
        if yf is None:
            return None
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
            self.logger.warning(f"Price download failed for {symbol}: {e}")
            return None

    # ================================================================== #
    # Market snapshot (VIX)
    # ================================================================== #

    def _compute_market_snapshot(self) -> Optional[MarketIVSnapshot]:
        """Broad market IV snapshot from VIX."""
        if yf is None:
            return None
        try:
            vix_data = yf.download("^VIX", period="1y", interval="1d", progress=False)
            if vix_data.empty:
                return None
            vix_values = vix_data["Close"].values.flatten()
            current_vix = float(vix_values[-1])
            vix_rank = self._calculate_rank(vix_values, current_vix)
            vix_pct = float(percentileofscore(vix_values, current_vix, kind='weak'))

            vix_term_slope = 1.0
            try:
                vix3m = yf.download("^VIX3M", period="5d", interval="1d", progress=False)
                if not vix3m.empty:
                    v3m = float(vix3m["Close"].values.flatten()[-1])
                    if v3m > 0:
                        vix_term_slope = current_vix / v3m
            except Exception:
                pass

            return MarketIVSnapshot(
                vix_level=current_vix,
                vix_rank=round(vix_rank, 2),
                vix_percentile=round(vix_pct, 2),
                vix_term_slope=round(vix_term_slope, 4),
                timestamp=datetime.now(),
            )
        except Exception as e:
            self.logger.warning(f"Market snapshot failed: {e}")
            return None

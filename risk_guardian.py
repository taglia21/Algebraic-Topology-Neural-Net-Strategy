#!/usr/bin/env python3
"""
Risk Guardian — Independent Safety Layer
==========================================

Monitors ALL positions, enforces hard stops, tracks daily P&L,
and force-liquidates everything if drawdown exceeds threshold.

This runs as a standalone safety check that can be imported by
unified_trader.py or invoked independently as a watchdog.

Emergency Thresholds:
  - Max drawdown from peak: 15% → LIQUIDATE ALL
  - Daily loss limit: 3% → HALT new trades
  - Per-position hard stop: -8% → FORCE CLOSE
  - Consecutive losers: 3 → PAUSE entries
  - Time-based exit: 5 days without profit → CLOSE
  - ATR trailing stop: 2.5x ATR → CLOSE
  - Profit target: 3x ATR → TAKE PROFIT
  - Scaled exit: 50% at +10%, trail rest

Usage:
    from risk_guardian import RiskGuardian

    guardian = RiskGuardian(initial_equity=100000)
    guardian.update(current_equity=95000, positions=positions_dict)
    if guardian.should_liquidate_all():
        # emergency liquidate everything
    if guardian.should_halt_trading():
        # stop new entries for the day
"""

import os
import sys
import json
import logging
import time
import requests
from pathlib import Path
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np
from dotenv import load_dotenv

try:
    from src.factor_monitor import FactorMonitor
    HAS_FACTOR_MONITOR = True
except ImportError:
    HAS_FACTOR_MONITOR = False

try:
    from src.correlation_manager import CrossAssetCorrelationMonitor
    _CROSS_CORR_AVAILABLE = True
except ImportError:
    _CROSS_CORR_AVAILABLE = False

load_dotenv()

logger = logging.getLogger("risk_guardian")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | GUARDIAN | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)
    fh = logging.FileHandler("logs/risk_guardian.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    ))
    Path("logs").mkdir(exist_ok=True)
    logger.addHandler(fh)


# ============================================================================
# ALPACA HELPERS (standalone — doesn't depend on unified_trader imports)
# ============================================================================

ALPACA_KEY = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA = "https://data.alpaca.markets"

_HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
    "Content-Type": "application/json",
}


def _api_get(path: str, base: str = None, params: dict = None) -> Optional[Any]:
    url = f"{base or ALPACA_BASE}{path}"
    try:
        r = requests.get(url, headers=_HEADERS, params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def _api_delete(path: str) -> bool:
    url = f"{ALPACA_BASE}{path}"
    try:
        r = requests.delete(url, headers=_HEADERS, timeout=10)
        return r.status_code in (200, 204)
    except Exception:
        return False


def _api_post(path: str, data: dict) -> Optional[dict]:
    url = f"{ALPACA_BASE}{path}"
    try:
        r = requests.post(url, headers=_HEADERS, json=data, timeout=10)
        if r.status_code in (200, 201):
            return r.json()
        return None
    except Exception:
        return None


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class GuardianAction(Enum):
    NONE = "none"
    HALT_NEW_ENTRIES = "halt_new_entries"
    CLOSE_POSITION = "close_position"
    CLOSE_PARTIAL = "close_partial"         # Scaled exit (50%)
    LIQUIDATE_ALL = "liquidate_all"
    TIGHTEN_STOPS = "tighten_stops"
    REDUCE_SIZE = "reduce_size"


@dataclass
class PositionRisk:
    """Risk assessment for a single position."""
    symbol: str
    entry_price: float
    current_price: float
    qty: int
    pnl_pct: float
    pnl_dollars: float
    days_held: int
    atr: float
    stop_price: float
    trailing_stop: float
    target_price: float
    action: GuardianAction = GuardianAction.NONE
    reason: str = ""


@dataclass
class GuardianState:
    """Overall risk state of the portfolio."""
    timestamp: str
    equity: float
    peak_equity: float
    drawdown_pct: float
    daily_pnl: float
    daily_pnl_pct: float
    consecutive_losses: int
    total_positions: int
    positions_at_stop: int
    positions_at_target: int
    should_halt: bool
    should_liquidate: bool
    halt_reasons: List[str] = field(default_factory=list)
    actions: List[dict] = field(default_factory=list)


# ============================================================================
# VIX MONITOR
# ============================================================================

class VIXMonitor:
    """Track VIX level for position sizing and entry filtering."""

    def __init__(self):
        self._vix_level: float = 20.0  # default
        self._last_fetch: Optional[datetime] = None
        self._fetch_interval_sec: int = 300  # refresh every 5 min

    def get_vix(self) -> float:
        """Get current VIX level. Uses CBOE VIX via Alpaca bars on ^VIX or VXX proxy."""
        now = datetime.now()
        if self._last_fetch and (now - self._last_fetch).total_seconds() < self._fetch_interval_sec:
            return self._vix_level

        self._last_fetch = now

        # Try VIX ETF proxies as approximation
        for vix_proxy in ["VIXY", "VXX", "UVXY"]:
            try:
                result = _api_get(
                    f"/v2/stocks/{vix_proxy}/bars",
                    base=ALPACA_DATA,
                    params={"timeframe": "1Day", "limit": 20, "adjustment": "split"},
                )
                if result and "bars" in result and len(result["bars"]) >= 2:
                    bars = result["bars"]
                    # Use price relative to 20-day average as VIX proxy
                    closes = [float(b["c"]) for b in bars]
                    current = closes[-1]
                    avg = np.mean(closes)
                    # Scale: if ETF is above its avg, VIX is elevated
                    # Rough mapping: VXX avg price maps to ~VIX 20
                    ratio = current / avg if avg > 0 else 1.0
                    self._vix_level = 20.0 * ratio
                    logger.debug(f"VIX proxy via {vix_proxy}: ~{self._vix_level:.1f}")
                    return self._vix_level
            except Exception:
                continue

        # Fallback: estimate from SPY realized vol
        try:
            result = _api_get(
                "/v2/stocks/SPY/bars",
                base=ALPACA_DATA,
                params={"timeframe": "1Day", "limit": 30, "adjustment": "split"},
            )
            if result and "bars" in result and len(result["bars"]) >= 20:
                closes = np.array([float(b["c"]) for b in result["bars"]])
                returns = np.diff(np.log(closes))
                realized_vol = float(np.std(returns) * np.sqrt(252) * 100)
                self._vix_level = realized_vol
                logger.debug(f"VIX from SPY realized vol: ~{self._vix_level:.1f}")
                return self._vix_level
        except Exception:
            pass

        return self._vix_level

    def get_position_scale(self) -> float:
        """Get position size multiplier based on VIX level.
        VIX < 20: 1.0 (normal)
        VIX 20-25: 0.75
        VIX 25-35: 0.50
        VIX > 35: 0.0 (skip entries)
        """
        vix = self.get_vix()
        if vix > 35:
            return 0.0
        elif vix > 25:
            return 0.50
        elif vix > 20:
            return 0.75
        return 1.0

    def should_skip_entries(self) -> bool:
        """Returns True if VIX is too high for new entries."""
        return self.get_vix() > 35.0


# ============================================================================
# CORRELATION CHECKER
# ============================================================================

class CorrelationChecker:
    """Check correlation between a new position and existing holdings."""

    def __init__(self, max_correlation: float = 0.70):
        self.max_correlation = max_correlation
        self._returns_cache: Dict[str, np.ndarray] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_sec: int = 600

    def _get_returns(self, symbol: str, lookback: int = 60) -> Optional[np.ndarray]:
        """Get daily returns for a symbol from Alpaca bars."""
        now = datetime.now()
        if (self._cache_time and (now - self._cache_time).total_seconds() > self._cache_ttl_sec):
            self._returns_cache.clear()

        if symbol in self._returns_cache:
            return self._returns_cache[symbol]

        try:
            result = _api_get(
                f"/v2/stocks/{symbol}/bars",
                base=ALPACA_DATA,
                params={"timeframe": "1Day", "limit": lookback + 1, "adjustment": "split"},
            )
            if result and "bars" in result and len(result["bars"]) >= 20:
                closes = np.array([float(b["c"]) for b in result["bars"]])
                returns = np.diff(np.log(closes))
                self._returns_cache[symbol] = returns
                self._cache_time = now
                return returns
        except Exception:
            pass
        return None

    def check_correlation(self, new_symbol: str, existing_symbols: List[str]) -> Tuple[bool, float, str]:
        """
        Check if new_symbol is too correlated with existing holdings.

        Returns: (allowed, max_corr_found, reason)
        """
        if not existing_symbols:
            return True, 0.0, "ok"

        new_returns = self._get_returns(new_symbol)
        if new_returns is None:
            return True, 0.0, "ok (no data)"

        max_corr = 0.0
        max_corr_sym = ""

        for sym in existing_symbols:
            existing_returns = self._get_returns(sym)
            if existing_returns is None:
                continue

            # Align lengths
            min_len = min(len(new_returns), len(existing_returns))
            if min_len < 15:
                continue

            corr = float(np.corrcoef(
                new_returns[-min_len:],
                existing_returns[-min_len:]
            )[0, 1])

            if abs(corr) > abs(max_corr):
                max_corr = corr
                max_corr_sym = sym

        if abs(max_corr) > self.max_correlation:
            reason = f"Correlation with {max_corr_sym}: {max_corr:.2f} > {self.max_correlation}"
            return False, max_corr, reason

        return True, max_corr, "ok"


# ============================================================================
# KELLY + VOLATILITY TARGETING + DRAWDOWN DAMPER
# ============================================================================

@dataclass
class SizingResult:
    """Output of the unified position-sizing pipeline."""
    symbol: str
    raw_kelly_frac: float          # Full Kelly fraction
    half_kelly_frac: float         # Half-Kelly (what we use)
    vol_target_scale: float        # Volatility targeting multiplier
    drawdown_damper: float         # Drawdown-based reduction [0, 1]
    vix_scale: float               # VIX-based reduction [0, 1]
    final_size_pct: float          # Final position size as % of equity
    capped: bool                   # True if size was capped at floor/ceiling
    rejected: bool                 # True if sizing pipeline rejected the trade
    reject_reason: str = ""


class KellyVolSizer:
    """
    Unified position-sizing pipeline: Kelly → Volatility Target → Drawdown Damper.

    Every trade signal passes through this pipeline before execution:
      1. Kelly Criterion (half-Kelly default) → raw fraction of equity
      2. Volatility targeting → scale so portfolio vol ≈ target (12% ann.)
      3. Drawdown damper → linearly reduce sizing as DD increases
      4. Floor/ceiling enforcement → [0.5%, 3%] of equity

    Usage:
        sizer = KellyVolSizer()
        result = sizer.compute_position_size(
            symbol="AAPL", equity=100000, symbol_vol=0.25,
            portfolio_vol=0.10, win_rate=0.55, avg_win=0.03, avg_loss=0.02,
            current_drawdown=0.05,
        )
        if not result.rejected:
            shares = int(result.final_size_pct * equity / price)
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,
        target_vol: float = 0.12,
        max_drawdown: float = 0.15,
        dd_full_scale_at: float = 0.0,
        dd_min_scale_at: float = 0.10,
        dd_zero_scale_at: float = 0.15,
        dd_min_scale: float = 0.25,
        floor_pct: float = 0.005,
        ceiling_pct: float = 0.03,
        vol_lookback: int = 20,
    ):
        """
        Parameters
        ----------
        kelly_fraction : float
            Fraction of full Kelly to use (0.5 = half-Kelly).
        target_vol : float
            Target annualized portfolio volatility (default 12%).
        max_drawdown : float
            Drawdown at which ALL new entries are rejected.
        dd_full_scale_at : float
            Drawdown level with 100% sizing (default 0%).
        dd_min_scale_at : float
            Drawdown level with minimum sizing (default 10%).
        dd_min_scale : float
            Minimum scaling at dd_min_scale_at (default 25%).
        dd_zero_scale_at : float
            Drawdown level with 0% sizing — reject all (default 15%).
        floor_pct : float
            Minimum position size as fraction of equity.
        ceiling_pct : float
            Maximum position size as fraction of equity.
        vol_lookback : int
            Lookback for realized vol computation (days).
        """
        self.kelly_fraction = kelly_fraction
        self.target_vol = target_vol
        self.max_drawdown = max_drawdown
        self.dd_full_scale_at = dd_full_scale_at
        self.dd_min_scale_at = dd_min_scale_at
        self.dd_zero_scale_at = dd_zero_scale_at
        self.dd_min_scale = dd_min_scale
        self.floor_pct = floor_pct
        self.ceiling_pct = ceiling_pct
        self.vol_lookback = vol_lookback

        # Track historical trade results for Kelly estimation
        self._trade_results: List[float] = []  # list of trade P&L percentages

    def compute_kelly_fraction(
        self,
        win_rate: float = 0.55,
        avg_win: float = 0.03,
        avg_loss: float = 0.02,
    ) -> float:
        """
        Compute Kelly fraction: f* = (p * b - q) / b
        where p=win_rate, q=1-p, b=avg_win/avg_loss.

        Returns half-Kelly by default (multiplied by kelly_fraction).
        """
        if avg_loss <= 0 or avg_win <= 0:
            return self.floor_pct
        b = avg_win / avg_loss  # Win/loss ratio
        q = 1.0 - win_rate
        kelly = (win_rate * b - q) / b
        # Kelly can be negative (edge is negative → don't trade)
        kelly = max(0.0, kelly)
        return kelly * self.kelly_fraction

    def volatility_target_scale(
        self,
        symbol_vol: float,
        portfolio_vol: float,
        target_vol: Optional[float] = None,
    ) -> float:
        """
        Scale position so portfolio vol stays near target.

        If portfolio vol is already at target, scale = 1.0.
        If portfolio vol is low, we can size larger (scale > 1).
        If portfolio vol is high, reduce (scale < 1).

        Also penalizes very-high-vol symbols.

        Returns multiplier in [0.25, 2.0].
        """
        tv = target_vol if target_vol is not None else self.target_vol
        if portfolio_vol <= 0:
            portfolio_vol = 0.01  # assume 1% if unknown

        # How much vol budget remains?
        vol_ratio = tv / portfolio_vol if portfolio_vol > 0 else 1.0

        # If portfolio vol is already above target, reduce aggressively
        if portfolio_vol >= tv:
            scale = max(0.25, tv / portfolio_vol)
        else:
            # Some room — scale up moderately
            scale = min(2.0, vol_ratio)

        # Additional penalty for very volatile symbols
        if symbol_vol > 0:
            # Symbol contributing more than 2x portfolio vol gets penalized
            sym_penalty = min(1.0, (tv * 1.5) / symbol_vol) if symbol_vol > tv else 1.0
            scale *= sym_penalty

        return float(np.clip(scale, 0.25, 2.0))

    def drawdown_damper(
        self,
        current_drawdown: float,
        max_allowed: Optional[float] = None,
    ) -> float:
        """
        Linearly reduce sizing as drawdown increases.

        DD = 0%  → 100% sizing
        DD = 10% → 25% sizing
        DD ≥ 15% → 0% sizing (reject)

        Returns multiplier in [0.0, 1.0].
        """
        dd = abs(current_drawdown)  # Ensure positive

        if dd <= self.dd_full_scale_at:
            return 1.0
        if dd >= self.dd_zero_scale_at:
            return 0.0

        # Linear interpolation between full and min
        if dd <= self.dd_min_scale_at:
            # From full_scale_at to min_scale_at: 1.0 → dd_min_scale
            t = (dd - self.dd_full_scale_at) / (self.dd_min_scale_at - self.dd_full_scale_at)
            return 1.0 - t * (1.0 - self.dd_min_scale)
        else:
            # From min_scale_at to zero_scale_at: dd_min_scale → 0.0
            t = (dd - self.dd_min_scale_at) / (self.dd_zero_scale_at - self.dd_min_scale_at)
            return self.dd_min_scale * (1.0 - t)

    def compute_realized_vol(self, close_prices: np.ndarray) -> float:
        """Compute annualized realized vol from close prices (20-day default)."""
        if len(close_prices) < 3:
            return 0.20  # Default assumption
        returns = np.diff(np.log(close_prices[-self.vol_lookback - 1:]))
        if len(returns) < 2:
            return 0.20
        return float(np.std(returns) * np.sqrt(252))

    def record_trade_result(self, pnl_pct: float):
        """Record a trade result for adaptive Kelly estimation."""
        self._trade_results.append(pnl_pct)
        if len(self._trade_results) > 200:
            self._trade_results = self._trade_results[-200:]

    def get_adaptive_kelly_params(self) -> Tuple[float, float, float]:
        """
        Compute win_rate, avg_win, avg_loss from recent trade history.
        Returns (win_rate, avg_win, avg_loss).
        Falls back to conservative defaults if insufficient history.
        """
        if len(self._trade_results) < 10:
            return 0.50, 0.02, 0.02  # Conservative defaults

        results = np.array(self._trade_results[-100:])  # Last 100 trades
        wins = results[results > 0]
        losses = results[results <= 0]

        win_rate = len(wins) / len(results)
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.02
        avg_loss = float(np.mean(np.abs(losses))) if len(losses) > 0 else 0.02

        return win_rate, avg_win, avg_loss

    def compute_position_size(
        self,
        symbol: str,
        equity: float,
        symbol_vol: float = 0.25,
        portfolio_vol: float = 0.10,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        current_drawdown: float = 0.0,
        vix_scale: float = 1.0,
        signal_confidence: float = 0.5,
    ) -> SizingResult:
        """
        Full position-sizing pipeline: Kelly → VolTarget → DD Damper → Clamp.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        equity : float
            Current portfolio equity.
        symbol_vol : float
            Annualized realized vol of the symbol.
        portfolio_vol : float
            Current annualized portfolio vol.
        win_rate, avg_win, avg_loss : float, optional
            If None, uses adaptive estimates from trade history.
        current_drawdown : float
            Current drawdown from equity peak (0.0 = at peak, 0.10 = -10%).
        vix_scale : float
            VIX-based modifier (from VIXMonitor).
        signal_confidence : float
            Signal confidence [0, 1] to modulate sizing.

        Returns
        -------
        SizingResult
            Full sizing breakdown with final_size_pct.
        """
        # Get Kelly params
        if win_rate is None or avg_win is None or avg_loss is None:
            win_rate, avg_win, avg_loss = self.get_adaptive_kelly_params()

        # 1. Kelly fraction
        raw_kelly = self.compute_kelly_fraction(win_rate, avg_win, avg_loss)
        full_kelly = raw_kelly / self.kelly_fraction  # Undo half-kelly for reporting
        half_kelly = raw_kelly

        # 2. Volatility targeting
        vol_scale = self.volatility_target_scale(symbol_vol, portfolio_vol)

        # 3. Drawdown damper
        dd_scale = self.drawdown_damper(current_drawdown)

        # Reject if drawdown kills sizing
        if dd_scale <= 0:
            return SizingResult(
                symbol=symbol,
                raw_kelly_frac=full_kelly,
                half_kelly_frac=half_kelly,
                vol_target_scale=vol_scale,
                drawdown_damper=dd_scale,
                vix_scale=vix_scale,
                final_size_pct=0.0,
                capped=False,
                rejected=True,
                reject_reason=f"Drawdown {current_drawdown:.1%} exceeds limit — sizing=0",
            )

        # 4. Combine: kelly * vol_scale * dd_damper * vix_scale * confidence
        raw_size = half_kelly * vol_scale * dd_scale * vix_scale
        # Modulate by signal confidence (high confidence → full size, low → reduced)
        raw_size *= (0.5 + 0.5 * signal_confidence)

        # 5. Floor / ceiling
        capped = False
        if raw_size < self.floor_pct:
            raw_size = self.floor_pct
            capped = True
        elif raw_size > self.ceiling_pct:
            raw_size = self.ceiling_pct
            capped = True

        logger.debug(
            f"KellyVolSizer {symbol}: kelly={half_kelly:.3f} "
            f"vol_scale={vol_scale:.2f} dd={dd_scale:.2f} vix={vix_scale:.2f} "
            f"conf={signal_confidence:.2f} → {raw_size:.3f} "
            f"{'(capped)' if capped else ''}"
        )

        return SizingResult(
            symbol=symbol,
            raw_kelly_frac=full_kelly,
            half_kelly_frac=half_kelly,
            vol_target_scale=vol_scale,
            drawdown_damper=dd_scale,
            vix_scale=vix_scale,
            final_size_pct=round(raw_size, 5),
            capped=capped,
            rejected=False,
        )


# ============================================================================
# RISK GUARDIAN — Main Class
# ============================================================================

class RiskGuardian:
    """
    Independent safety layer that monitors all positions and enforces risk limits.

    Thresholds:
      - Max drawdown from peak: 15% → LIQUIDATE ALL
      - Daily loss limit: 3% → HALT new trades
      - Per-position hard stop: -8% → FORCE CLOSE
      - Consecutive losers: 3 → PAUSE entries
      - Time-based exit: 5 days without profit → CLOSE
      - ATR trailing stop: 2.5x ATR → CLOSE
      - Profit target: 3x ATR → TAKE PROFIT
      - Scaled exit: sell 50% at +10%, trail rest
      - Max total positions: 10
      - Max sector positions: 3
    """

    def __init__(
        self,
        initial_equity: float = 100000.0,
        max_drawdown_pct: float = 0.15,
        daily_loss_limit_pct: float = 0.03,
        hard_stop_pct: float = 0.08,
        consecutive_loss_limit: int = 3,
        time_exit_days: int = 5,
        atr_trailing_mult: float = 2.5,
        profit_target_atr_mult: float = 3.0,
        scaled_exit_pct: float = 0.10,
        scaled_exit_fraction: float = 0.50,
        max_positions: int = 10,
        max_sector_positions: int = 3,
        max_correlation: float = 0.70,
    ):
        # Config
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.hard_stop_pct = hard_stop_pct
        self.consecutive_loss_limit = consecutive_loss_limit
        self.time_exit_days = time_exit_days
        self.atr_trailing_mult = atr_trailing_mult
        self.profit_target_atr_mult = profit_target_atr_mult
        self.scaled_exit_pct = scaled_exit_pct
        self.scaled_exit_fraction = scaled_exit_fraction
        self.max_positions = max_positions
        self.max_sector_positions = max_sector_positions

        # State
        self.peak_equity = initial_equity
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.daily_start_equity = initial_equity
        self.daily_start_date: Optional[date] = None
        self.consecutive_losses = 0
        self._halted = False
        self._liquidate_all = False
        self._halt_reasons: List[str] = []
        self._trade_results: List[bool] = []  # Recent trade outcomes (True=win)
        self._partial_exits_done: set = set()  # symbols already scaled out 50%

        # Sub-modules
        self.vix_monitor = VIXMonitor()
        self.correlation_checker = CorrelationChecker(max_correlation=max_correlation)
        self.kelly_vol_sizer = KellyVolSizer(
            kelly_fraction=0.5,
            target_vol=0.12,
            max_drawdown=max_drawdown_pct,
            dd_zero_scale_at=max_drawdown_pct,
        )
        self.factor_monitor: Optional['FactorMonitor'] = None
        if HAS_FACTOR_MONITOR:
            self.factor_monitor = FactorMonitor(
                neutral_tolerance=0.25,
                mkt_tolerance=0.35,
                max_single_factor=0.45,
            )
            logger.info("🛡️ Factor exposure monitor enabled")

        self.cross_corr_monitor: Optional['CrossAssetCorrelationMonitor'] = None
        if _CROSS_CORR_AVAILABLE:
            self.cross_corr_monitor = CrossAssetCorrelationMonitor(
                short_window=21,
                medium_window=63,
                breakdown_threshold=0.70,
                risk_score_threshold=70.0,
            )
            logger.info("🛡️ Cross-asset correlation monitor enabled")

        # State persistence
        self._state_file = Path("state/risk_guardian_state.json")
        self._load_state()

        logger.info(
            f"🛡️ Risk Guardian initialized: "
            f"DD_limit={max_drawdown_pct:.0%}, "
            f"daily_loss={daily_loss_limit_pct:.0%}, "
            f"hard_stop={hard_stop_pct:.0%}, "
            f"max_consec_losses={consecutive_loss_limit}, "
            f"time_exit={time_exit_days}d"
        )

    # ── Core Update ─────────────────────────────────────────────────

    def update(self, current_equity: float) -> GuardianState:
        """
        Update guardian with current portfolio equity.
        Checks all risk thresholds and returns recommended actions.

        Should be called every scan cycle.
        """
        self.current_equity = current_equity
        self._halt_reasons = []

        # Reset daily tracking at start of new day
        today = date.today()
        if self.daily_start_date != today:
            self.daily_start_equity = current_equity
            self.daily_start_date = today
            self._halted = False
            self._halt_reasons = []
            logger.info(f"🛡️ Daily reset: start equity ${current_equity:,.2f}")

        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Calculate drawdown from peak
        drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0

        # Calculate daily P&L
        daily_pnl = current_equity - self.daily_start_equity
        daily_pnl_pct = daily_pnl / self.daily_start_equity if self.daily_start_equity > 0 else 0

        # ── Check 1: Max drawdown from peak ──
        if drawdown_pct >= self.max_drawdown_pct:
            self._liquidate_all = True
            reason = (
                f"DRAWDOWN {drawdown_pct:.1%} >= {self.max_drawdown_pct:.0%} "
                f"(peak ${self.peak_equity:,.0f} → ${current_equity:,.0f})"
            )
            self._halt_reasons.append(reason)
            logger.error(f"🚨 EMERGENCY: {reason}")

        # ── Check 2: Daily loss limit ──
        if daily_pnl_pct <= -self.daily_loss_limit_pct:
            self._halted = True
            reason = (
                f"DAILY LOSS {daily_pnl_pct:.2%} >= {self.daily_loss_limit_pct:.0%} "
                f"(${daily_pnl:+,.0f})"
            )
            self._halt_reasons.append(reason)
            logger.warning(f"🔴 {reason}")

        # ── Check 3: Consecutive losses ──
        if self.consecutive_losses >= self.consecutive_loss_limit:
            self._halted = True
            reason = f"CONSECUTIVE LOSSES: {self.consecutive_losses} >= {self.consecutive_loss_limit}"
            self._halt_reasons.append(reason)
            logger.warning(f"🔴 {reason}")

        state = GuardianState(
            timestamp=datetime.now().isoformat(),
            equity=current_equity,
            peak_equity=self.peak_equity,
            drawdown_pct=drawdown_pct,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            consecutive_losses=self.consecutive_losses,
            total_positions=0,  # will be set by caller
            positions_at_stop=0,
            positions_at_target=0,
            should_halt=self._halted,
            should_liquidate=self._liquidate_all,
            halt_reasons=self._halt_reasons.copy(),
        )

        self._save_state()
        return state

    # ── Position-Level Risk Checks ──────────────────────────────────

    def check_position(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        qty: int,
        entry_time: datetime,
        atr: float,
        trailing_stop: float,
        highest_price: float,
    ) -> PositionRisk:
        """
        Check a single position for all exit conditions.

        Returns a PositionRisk with recommended action.
        """
        pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
        pnl_dollars = (current_price - entry_price) * qty
        days_held = (datetime.now() - entry_time).days

        # ATR-based stops and targets
        atr_stop = entry_price - (atr * self.atr_trailing_mult)
        atr_target = entry_price + (atr * self.profit_target_atr_mult)

        # Dynamic trailing stop: 2.5x ATR below highest price
        dynamic_trailing = highest_price - (atr * self.atr_trailing_mult) if highest_price > 0 else atr_stop

        # Use the higher of static stop and trailing stop
        effective_stop = max(atr_stop, dynamic_trailing, trailing_stop)

        risk = PositionRisk(
            symbol=symbol,
            entry_price=entry_price,
            current_price=current_price,
            qty=qty,
            pnl_pct=pnl_pct,
            pnl_dollars=pnl_dollars,
            days_held=days_held,
            atr=atr,
            stop_price=effective_stop,
            trailing_stop=dynamic_trailing,
            target_price=atr_target,
        )

        # ── Check 1: Hard stop loss at -8% ──
        if pnl_pct <= -self.hard_stop_pct:
            risk.action = GuardianAction.CLOSE_POSITION
            risk.reason = f"HARD STOP: {pnl_pct:.1%} <= -{self.hard_stop_pct:.0%}"
            logger.warning(f"🛑 {symbol}: {risk.reason}")
            return risk

        # ── Check 2: ATR trailing stop hit ──
        if current_price <= effective_stop:
            risk.action = GuardianAction.CLOSE_POSITION
            risk.reason = f"ATR STOP: ${current_price:.2f} <= ${effective_stop:.2f}"
            logger.warning(f"🛑 {symbol}: {risk.reason}")
            return risk

        # ── Check 3: Time-based exit (no profit after N days) ──
        if days_held >= self.time_exit_days and pnl_pct <= 0.005:
            risk.action = GuardianAction.CLOSE_POSITION
            risk.reason = f"TIME EXIT: {days_held}d held, P&L={pnl_pct:.1%}"
            logger.info(f"⏱️ {symbol}: {risk.reason}")
            return risk

        # ── Check 4: Profit target at 3x ATR ──
        if current_price >= atr_target:
            risk.action = GuardianAction.CLOSE_POSITION
            risk.reason = f"ATR TARGET: ${current_price:.2f} >= ${atr_target:.2f} (3x ATR)"
            logger.info(f"🎯 {symbol}: {risk.reason}")
            return risk

        # ── Check 5: Scaled exit — sell 50% at +10% gain ──
        if pnl_pct >= self.scaled_exit_pct and symbol not in self._partial_exits_done:
            risk.action = GuardianAction.CLOSE_PARTIAL
            risk.reason = f"SCALED EXIT: +{pnl_pct:.1%} >= +{self.scaled_exit_pct:.0%}, sell 50%"
            logger.info(f"📊 {symbol}: {risk.reason}")
            return risk

        # ── Check 6: Trailing stop hit (after gains) ──
        if pnl_pct > 0.03 and current_price <= dynamic_trailing:
            risk.action = GuardianAction.CLOSE_POSITION
            risk.reason = f"TRAILING STOP: ${current_price:.2f} <= ${dynamic_trailing:.2f}"
            logger.info(f"🛑 {symbol}: {risk.reason}")
            return risk

        return risk

    # ── Trade Result Tracking ───────────────────────────────────────

    def record_trade_result(self, won: bool):
        """Record a trade outcome for consecutive loss tracking."""
        self._trade_results.append(won)
        if won:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

        if self.consecutive_losses >= self.consecutive_loss_limit:
            logger.warning(
                f"🔴 {self.consecutive_losses} consecutive losses — "
                f"halting new entries until a win"
            )
            self._halted = True

    def mark_partial_exit(self, symbol: str):
        """Mark a symbol as having completed its scaled exit (50% sold)."""
        self._partial_exits_done.add(symbol)

    # ── Entry Validation ────────────────────────────────────────────

    def validate_entry(
        self,
        symbol: str,
        existing_positions: List[str],
        sector_map: Dict[str, str],
        num_confirming_signals: int,
        min_confirming_signals: int = 3,
    ) -> Tuple[bool, str]:
        """
        Validate whether a new entry is allowed.

        Checks:
          1. Not halted / liquidation mode
          2. Max total positions (10)
          3. Max sector positions (3)
          4. Minimum confirming signals (3)
          5. VIX filter
          6. Correlation check
          7. Consecutive loss circuit breaker
        """
        if self._liquidate_all:
            return False, "LIQUIDATION MODE — no new entries"

        if self._halted:
            return False, f"HALTED: {'; '.join(self._halt_reasons)}"

        # Max positions
        if len(existing_positions) >= self.max_positions:
            return False, f"Max positions ({self.max_positions}) reached"

        # Sector check
        new_sector = sector_map.get(symbol, "unknown")
        sector_count = sum(1 for s in existing_positions if sector_map.get(s, "x") == new_sector)
        if sector_count >= self.max_sector_positions:
            return False, f"Sector '{new_sector}' at limit ({sector_count}/{self.max_sector_positions})"

        # Confirming signals
        if num_confirming_signals < min_confirming_signals:
            return False, (
                f"Insufficient signals: {num_confirming_signals}/{min_confirming_signals} "
                f"confirming"
            )

        # VIX filter
        if self.vix_monitor.should_skip_entries():
            vix = self.vix_monitor.get_vix()
            return False, f"VIX too high ({vix:.1f} > 35) — no entries"

        # Correlation check
        corr_ok, max_corr, corr_reason = self.correlation_checker.check_correlation(
            symbol, existing_positions
        )
        if not corr_ok:
            return False, f"Corr filter: {corr_reason}"

        # Cross-asset correlation risk check
        if self.cross_corr_monitor is not None and existing_positions:
            block, cross_reason = self.cross_corr_monitor.should_block_entry(
                existing_positions
            )
            if block:
                return False, f"Cross-corr: {cross_reason}"

        # Factor exposure check: warn but don't block (soft gate)
        if self.factor_monitor is not None and existing_positions:
            hypothetical = existing_positions + [symbol]
            exposure = self.factor_monitor.get_factor_exposures(hypothetical)
            if not exposure.is_neutral:
                violations_str = "; ".join(exposure.violations[:2])
                logger.warning(
                    f"⚠️ Factor tilt if adding {symbol}: {violations_str}"
                )
                # Hard block only if any single factor exceeds hard limit
                for f_name, beta in exposure.factor_betas.items():
                    if f_name != "MKT-RF" and abs(beta) > self.factor_monitor.max_single_factor:
                        return False, (
                            f"Factor exposure hard limit: {f_name}={beta:+.3f} "
                            f"exceeds ±{self.factor_monitor.max_single_factor}"
                        )

        return True, "ok"

    def get_vix_position_scale(self) -> float:
        """Get position size multiplier based on VIX."""
        return self.vix_monitor.get_position_scale()

    # ── Kelly Position Sizing ───────────────────────────────────────

    def size_position(
        self,
        symbol: str,
        equity: float,
        signal_confidence: float = 0.5,
        symbol_vol: float = 0.25,
        portfolio_vol: float = 0.10,
    ) -> SizingResult:
        """
        Size a position using the full Kelly+VolTarget+DD pipeline.

        Uses current equity high-water mark and VIX for drawdown
        and volatility adjustments automatically.
        """
        current_dd = (
            (self.peak_equity - self.current_equity) / self.peak_equity
            if self.peak_equity > 0 else 0.0
        )
        vix_scale = self.vix_monitor.get_position_scale()

        return self.kelly_vol_sizer.compute_position_size(
            symbol=symbol,
            equity=equity,
            symbol_vol=symbol_vol,
            portfolio_vol=portfolio_vol,
            current_drawdown=current_dd,
            vix_scale=vix_scale,
            signal_confidence=signal_confidence,
        )

    def record_trade_for_kelly(self, pnl_pct: float):
        """Record trade P&L for adaptive Kelly estimation."""
        self.kelly_vol_sizer.record_trade_result(pnl_pct)

    def get_current_drawdown(self) -> float:
        """Get current drawdown from equity high-water mark."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity

    # ── Factor Exposure Helpers ─────────────────────────────────────

    def get_factor_exposures(
        self, positions: List[str], weights: Optional[Dict[str, float]] = None
    ) -> Optional[dict]:
        """
        Get current factor exposures for the portfolio.
        Returns dict with factor_betas, is_neutral, violations, etc.
        Returns None if FactorMonitor is not available.
        """
        if self.factor_monitor is None:
            return None
        exposure = self.factor_monitor.get_factor_exposures(positions, weights)
        return exposure.to_dict()

    def check_factor_risk(self, positions: List[str]) -> Tuple[bool, str]:
        """
        Quick factor risk check. Returns (is_ok, message).
        is_ok=True if factor-neutral or monitor unavailable.
        """
        if self.factor_monitor is None:
            return True, "Factor monitor not available"
        exposure = self.factor_monitor.get_factor_exposures(positions)
        if exposure.is_neutral:
            return True, "Factor-neutral"
        return False, f"Factor tilts: {'; '.join(exposure.violations[:3])}"

    def print_factor_report(self, positions: List[str]):
        """Print a formatted factor exposure report to console."""
        if self.factor_monitor is not None:
            self.factor_monitor.get_factor_exposures(positions)
            self.factor_monitor.print_report()

    # ── Query Methods ───────────────────────────────────────────────

    def should_halt_trading(self) -> bool:
        """Returns True if new entries should be paused."""
        return self._halted or self._liquidate_all

    def should_liquidate_all(self) -> bool:
        """Returns True if ALL positions should be force-closed."""
        return self._liquidate_all

    def get_halt_reasons(self) -> List[str]:
        """Get reasons for current halt."""
        return self._halt_reasons.copy()

    def reset_consecutive_losses(self):
        """Manually reset the consecutive loss counter (e.g., after market close)."""
        self.consecutive_losses = 0
        if self._halted and not self._liquidate_all:
            # Only un-halt if the daily loss limit isn't also breached
            daily_pnl_pct = (
                (self.current_equity - self.daily_start_equity) / self.daily_start_equity
                if self.daily_start_equity > 0 else 0
            )
            if daily_pnl_pct > -self.daily_loss_limit_pct:
                self._halted = False
                self._halt_reasons = []
                logger.info("🛡️ Consecutive loss halt lifted after reset")

    # ── Force Liquidation ───────────────────────────────────────────

    def force_liquidate_all(self, dry_run: bool = False) -> List[dict]:
        """
        EMERGENCY: Close ALL positions via Alpaca API.

        Returns list of close results.
        """
        logger.error("🚨🚨🚨 FORCE LIQUIDATE ALL POSITIONS 🚨🚨🚨")

        results = []

        if dry_run:
            logger.info("[DRY RUN] Would liquidate all positions")
            return results

        if not ALPACA_KEY or not ALPACA_SECRET:
            logger.error("No API credentials — cannot liquidate")
            return results

        # Method 1: Use Alpaca's close-all-positions endpoint
        try:
            success = _api_delete("/v2/positions?cancel_orders=true")
            if success:
                logger.info("✅ All positions liquidated via Alpaca API")
                results.append({"action": "liquidate_all", "status": "success"})
            else:
                logger.error("Failed to liquidate via bulk endpoint — trying individual")
                # Fall back to individual position closes
                positions = _api_get("/v2/positions")
                if isinstance(positions, list):
                    for pos in positions:
                        sym = pos.get("symbol", "??")
                        ok = _api_delete(f"/v2/positions/{sym}")
                        status = "closed" if ok else "failed"
                        results.append({"symbol": sym, "status": status})
                        logger.info(f"  {sym}: {status}")
        except Exception as e:
            logger.error(f"Liquidation error: {e}")

        self._liquidate_all = False  # Reset after execution
        return results

    # ── Equity Guardrails for Position Sizing ───────────────────────

    def compute_safe_position_size(
        self,
        base_pct: float,
        atr_pct: float,
        confidence: float,
        regime_scale: float = 1.0,
    ) -> float:
        """
        Compute position size with all safety adjustments.

        Applies:
          1. Quarter-Kelly cap (input should already be quarter-Kelly)
          2. Max 5% per position
          3. VIX scaling
          4. Volatility inverse scaling (ATR-based)
          5. Confidence scaling
          6. Regime scaling (Phase 4)
        """
        # Start with base
        size_pct = base_pct

        # Cap at 5% absolute max
        size_pct = min(size_pct, 0.05)

        # VIX scaling
        vix_scale = self.vix_monitor.get_position_scale()
        size_pct *= vix_scale

        # ATR-based volatility adjustment: lower size for higher vol
        # Target 2% daily ATR; scale inversely for higher
        vol_target = 0.02
        vol_scale = min(1.0, vol_target / max(atr_pct, 0.005))
        size_pct *= vol_scale

        # Confidence scaling: lower confidence → smaller size
        conf_scale = max(0.5, min(confidence, 1.0))
        size_pct *= conf_scale

        # Phase 4: Regime scaling
        size_pct *= max(0.1, min(regime_scale, 1.5))

        # Floor at 0.5%
        size_pct = max(size_pct, 0.005)

        # Hard cap at 5%
        size_pct = min(size_pct, 0.05)

        return size_pct

    # ── Weight Deviation Check ──────────────────────────────────────

    def check_weight_deviation(
        self,
        positions: Dict[str, float],  # symbol → market value
        total_equity: float,
    ) -> List[Tuple[str, float, str]]:
        """
        Check if any position deviates more than 2x from equal-weight target.

        Returns list of (symbol, weight_pct, reason) for overweight positions.
        """
        if not positions or total_equity <= 0:
            return []

        n_positions = len(positions)
        equal_weight = 1.0 / n_positions if n_positions > 0 else 1.0
        max_allowed = equal_weight * 2.0

        overweight = []
        for sym, mkt_val in positions.items():
            weight = mkt_val / total_equity
            if weight > max_allowed and weight > 0.05:  # Only flag if > 5%
                overweight.append((
                    sym, weight,
                    f"Weight {weight:.1%} > 2x equal ({equal_weight:.1%} target)"
                ))

        return overweight

    # ── State Persistence ───────────────────────────────────────────

    def _save_state(self):
        """Persist guardian state to disk."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "timestamp": datetime.now().isoformat(),
                "peak_equity": self.peak_equity,
                "daily_start_equity": self.daily_start_equity,
                "daily_start_date": str(self.daily_start_date) if self.daily_start_date else None,
                "consecutive_losses": self.consecutive_losses,
                "halted": self._halted,
                "liquidate_all": self._liquidate_all,
                "halt_reasons": self._halt_reasons,
                "partial_exits": list(self._partial_exits_done),
                "trade_results": self._trade_results[-50:],
            }
            with open(self._state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save guardian state: {e}")

    def _load_state(self):
        """Load persisted guardian state."""
        try:
            if self._state_file.exists():
                with open(self._state_file) as f:
                    state = json.load(f)
                self.peak_equity = state.get("peak_equity", self.initial_equity)
                self.consecutive_losses = state.get("consecutive_losses", 0)
                self._partial_exits_done = set(state.get("partial_exits", []))
                self._trade_results = state.get("trade_results", [])
                saved_date = state.get("daily_start_date")
                if saved_date and saved_date == str(date.today()):
                    self.daily_start_equity = state.get("daily_start_equity", self.initial_equity)
                    self.daily_start_date = date.today()
                logger.info(f"🛡️ Loaded guardian state: peak=${self.peak_equity:,.0f}, consec_losses={self.consecutive_losses}")
        except Exception as e:
            logger.debug(f"Failed to load guardian state: {e}")


# ============================================================================
# STANDALONE WATCHDOG MODE
# ============================================================================

def run_watchdog(check_interval_sec: int = 60, dry_run: bool = True):
    """
    Run Risk Guardian as a standalone watchdog process.

    Periodically checks Alpaca account and positions, enforces risk limits
    independently of unified_trader.py.
    """
    logger.info("🛡️ Risk Guardian Watchdog Starting")
    logger.info(f"  Check interval: {check_interval_sec}s")
    logger.info(f"  Dry run: {dry_run}")

    # Get initial equity
    account = _api_get("/v2/account")
    if not account:
        logger.error("Cannot connect to Alpaca — exiting")
        return

    initial_equity = float(account.get("equity", 100000))
    guardian = RiskGuardian(initial_equity=initial_equity)

    import signal as sig_module
    running = True

    def handle_signal(signum, frame):
        nonlocal running
        running = False
        logger.info("Watchdog shutting down...")

    sig_module.signal(sig_module.SIGINT, handle_signal)
    sig_module.signal(sig_module.SIGTERM, handle_signal)

    while running:
        try:
            # Get current account
            account = _api_get("/v2/account")
            if not account:
                logger.warning("Cannot reach Alpaca — retrying")
                time.sleep(check_interval_sec)
                continue

            equity = float(account.get("equity", 0))

            # Update guardian
            state = guardian.update(equity)

            # Get positions
            positions = _api_get("/v2/positions")
            if not isinstance(positions, list):
                positions = []

            logger.info(
                f"🛡️ Equity=${equity:,.0f} | DD={state.drawdown_pct:.1%} | "
                f"Daily={state.daily_pnl_pct:+.2%} | "
                f"Positions={len(positions)} | "
                f"ConsecLoss={state.consecutive_losses}"
            )

            # Emergency liquidation check
            if state.should_liquidate:
                logger.error("🚨 LIQUIDATION TRIGGERED BY WATCHDOG")
                guardian.force_liquidate_all(dry_run=dry_run)
                if not dry_run:
                    logger.error("All positions liquidated. Watchdog halting.")
                    break

            # Check individual positions
            for pos in positions:
                sym = pos.get("symbol", "")
                entry_price = float(pos.get("avg_entry_price", 0))
                current_price = float(pos.get("current_price", 0))
                qty = int(float(pos.get("qty", 0)))
                pnl_pct = float(pos.get("unrealized_plpc", 0))

                if entry_price <= 0 or current_price <= 0:
                    continue

                # Hard stop check
                if pnl_pct <= -guardian.hard_stop_pct:
                    logger.warning(
                        f"🛑 WATCHDOG HARD STOP: {sym} at {pnl_pct:.1%} "
                        f"(limit -{guardian.hard_stop_pct:.0%})"
                    )
                    if not dry_run:
                        _api_delete(f"/v2/positions/{sym}")
                        logger.info(f"  Force-closed {sym}")

            # Check if halted
            if state.should_halt:
                logger.warning(f"🔴 Trading halted: {'; '.join(state.halt_reasons)}")

        except Exception as e:
            logger.error(f"Watchdog error: {e}")

        time.sleep(check_interval_sec)

    logger.info("🛡️ Risk Guardian Watchdog stopped")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Risk Guardian Watchdog")
    parser.add_argument("--interval", type=int, default=60, help="Check interval (seconds)")
    parser.add_argument("--live", action="store_true", help="Enable live liquidation (default: dry run)")
    args = parser.parse_args()

    run_watchdog(
        check_interval_sec=args.interval,
        dry_run=not args.live,
    )

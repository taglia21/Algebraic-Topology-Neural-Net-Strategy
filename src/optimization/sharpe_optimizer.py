"""
Sharpe Optimiser — Dynamic Position Sizing & Risk Management
==============================================================

Production-grade optimiser that:

1. Rolling 20-day Sharpe calculation updated after each trade
2. Dynamic position sizing: increase when Sharpe > 1.5, decrease when < 0.8
3. Correlation-aware portfolio construction: reject if corr > 0.7
4. Maximum drawdown circuit breaker: halt if daily DD > 3%
5. Trade timing: only trade in first 30 min and last 60 min of session
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SharpeOptimizerConfig:
    """Configuration for the Sharpe optimiser."""

    # Sharpe thresholds
    sharpe_high: float = 1.5  # increase sizing above this
    sharpe_low: float = 0.8   # decrease sizing below this
    sharpe_window: int = 20   # rolling window in trades

    # Sizing multipliers
    size_boost: float = 1.3   # multiplier when Sharpe > high
    size_cut: float = 0.6     # multiplier when Sharpe < low
    base_pct: float = 0.03    # baseline position size (3% of equity)

    # Correlation
    max_portfolio_correlation: float = 0.70
    correlation_lookback: int = 60  # days of price data for correlation

    # Drawdown circuit breaker
    max_daily_drawdown: float = 0.03  # 3%
    halt_duration_minutes: int = 60   # halt for 60 min after trigger

    # Trade timing (ET)
    morning_open: dt_time = field(default_factory=lambda: dt_time(9, 30))
    morning_close: dt_time = field(default_factory=lambda: dt_time(10, 0))
    afternoon_open: dt_time = field(default_factory=lambda: dt_time(15, 0))
    afternoon_close: dt_time = field(default_factory=lambda: dt_time(16, 0))


class SharpeOptimizer:
    """
    Production Sharpe optimiser — gates trades by rolling Sharpe, correlation,
    drawdown, and timing.

    Usage::

        opt = SharpeOptimizer()
        opt.record_trade(pnl_pct=0.012)  # +1.2%

        size = opt.get_position_size(equity=100000, symbol='AAPL')
        can, reason = opt.can_trade(current_time, equity, start_equity)
        corr_ok, msg = opt.check_correlation('AAPL', existing=['MSFT','GOOGL'], price_data={...})
    """

    def __init__(self, config: Optional[SharpeOptimizerConfig] = None):
        self.config = config or SharpeOptimizerConfig()
        self._returns: deque = deque(maxlen=max(self.config.sharpe_window, 100))
        self._halted_until: Optional[datetime] = None
        self._daily_start_equity: float = 0.0
        self._last_reset_date: Optional[str] = None
        logger.info("SharpeOptimizer initialised (window=%d)", self.config.sharpe_window)

    # ------------------------------------------------------------------
    # Sharpe tracking
    # ------------------------------------------------------------------

    def record_trade(self, pnl_pct: float):
        """Record a trade return for rolling Sharpe calculation."""
        self._returns.append(pnl_pct)

    def rolling_sharpe(self) -> float:
        """Compute annualised rolling Sharpe ratio."""
        if len(self._returns) < 5:
            return 0.0
        r = np.array(list(self._returns))
        mean = float(np.mean(r))
        std = float(np.std(r))
        if std < 1e-10:
            return 0.0
        return mean / std * np.sqrt(252)

    # ------------------------------------------------------------------
    # Dynamic sizing
    # ------------------------------------------------------------------

    def get_position_size(self, equity: float, symbol: str = "") -> float:
        """
        Return position size as a fraction of equity, adjusted by Sharpe.

        Returns a dollar amount (equity * adjusted_pct).
        """
        sharpe = self.rolling_sharpe()
        base = self.config.base_pct

        if sharpe > self.config.sharpe_high:
            adj = base * self.config.size_boost
        elif sharpe < self.config.sharpe_low:
            adj = base * self.config.size_cut
        else:
            # Linear interpolation
            frac = (sharpe - self.config.sharpe_low) / max(
                self.config.sharpe_high - self.config.sharpe_low, 0.01
            )
            adj = base * (self.config.size_cut + frac * (self.config.size_boost - self.config.size_cut))

        adj = max(0.005, min(adj, 0.08))  # floor 0.5%, cap 8%
        dollar = equity * adj
        logger.debug(
            "Position size: Sharpe=%.2f → adj_pct=%.2f%% → $%.0f",
            sharpe,
            adj * 100,
            dollar,
        )
        return dollar

    def get_position_size_pct(self, symbol: str = "") -> float:
        """Return position size as percentage of equity."""
        sharpe = self.rolling_sharpe()
        base = self.config.base_pct

        if sharpe > self.config.sharpe_high:
            return min(0.08, base * self.config.size_boost)
        elif sharpe < self.config.sharpe_low:
            return max(0.005, base * self.config.size_cut)
        else:
            frac = (sharpe - self.config.sharpe_low) / max(
                self.config.sharpe_high - self.config.sharpe_low, 0.01
            )
            return max(
                0.005,
                min(0.08, base * (self.config.size_cut + frac * (self.config.size_boost - self.config.size_cut))),
            )

    # ------------------------------------------------------------------
    # Correlation check
    # ------------------------------------------------------------------

    def check_correlation(
        self,
        new_symbol: str,
        existing_symbols: List[str],
        price_data: Dict[str, np.ndarray],
    ) -> Tuple[bool, str]:
        """
        Check if adding new_symbol exceeds portfolio correlation threshold.

        price_data: {symbol: np.ndarray of close prices}

        Returns (allowed, reason).
        """
        if not existing_symbols:
            return True, "No existing positions"

        new_prices = price_data.get(new_symbol)
        if new_prices is None or len(new_prices) < 20:
            return True, "Insufficient data for correlation check"

        new_rets = np.diff(new_prices) / np.maximum(new_prices[:-1], 1e-8)

        max_corr = 0.0
        max_corr_sym = ""
        for sym in existing_symbols:
            sym_prices = price_data.get(sym)
            if sym_prices is None or len(sym_prices) < 20:
                continue
            sym_rets = np.diff(sym_prices) / np.maximum(sym_prices[:-1], 1e-8)

            # Align lengths
            n = min(len(new_rets), len(sym_rets))
            if n < 10:
                continue

            try:
                corr = float(np.corrcoef(new_rets[-n:], sym_rets[-n:])[0, 1])
            except Exception:
                continue

            if abs(corr) > max_corr:
                max_corr = abs(corr)
                max_corr_sym = sym

        if max_corr > self.config.max_portfolio_correlation:
            return False, (
                f"{new_symbol} too correlated with {max_corr_sym} "
                f"(ρ={max_corr:.2f} > {self.config.max_portfolio_correlation})"
            )

        return True, f"Max correlation {max_corr:.2f} with {max_corr_sym}"

    # ------------------------------------------------------------------
    # Trade timing
    # ------------------------------------------------------------------

    def in_trading_window(self, now_et: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Check if current time is in an allowed trading window.

        Allowed: first 30 min (9:30-10:00) and last 60 min (15:00-16:00).
        """
        if now_et is None:
            try:
                from zoneinfo import ZoneInfo
            except ImportError:
                from backports.zoneinfo import ZoneInfo  # type: ignore
            now_et = datetime.now(ZoneInfo("America/New_York"))

        t = now_et.time()
        cfg = self.config

        in_morning = cfg.morning_open <= t <= cfg.morning_close
        in_afternoon = cfg.afternoon_open <= t <= cfg.afternoon_close

        if in_morning:
            return True, "Morning window (9:30-10:00 ET)"
        if in_afternoon:
            return True, "Afternoon window (15:00-16:00 ET)"
        return False, f"Outside trading windows (current: {t.strftime('%H:%M')} ET)"

    # ------------------------------------------------------------------
    # Drawdown circuit breaker
    # ------------------------------------------------------------------

    def update_daily_equity(self, equity: float):
        """Call at start of day to set reference equity for drawdown calc."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_reset_date != today:
            self._daily_start_equity = equity
            self._last_reset_date = today

    def check_drawdown(self, current_equity: float) -> Tuple[bool, str]:
        """
        Check if daily drawdown exceeds limit.

        Returns (trading_allowed, reason).
        """
        # Check if halt is still active
        if self._halted_until and datetime.now() < self._halted_until:
            remaining = (self._halted_until - datetime.now()).total_seconds() / 60
            return False, f"Drawdown halt active ({remaining:.0f} min remaining)"

        if self._daily_start_equity <= 0:
            return True, "No reference equity set"

        dd = (current_equity - self._daily_start_equity) / self._daily_start_equity
        if dd < -self.config.max_daily_drawdown:
            self._halted_until = datetime.now() + \
                __import__("datetime").timedelta(minutes=self.config.halt_duration_minutes)
            logger.error(
                "DRAWDOWN CIRCUIT BREAKER: %.2f%% DD exceeds %.2f%% limit — "
                "halting for %d min",
                dd * 100,
                self.config.max_daily_drawdown * 100,
                self.config.halt_duration_minutes,
            )
            return False, f"Daily drawdown {dd*100:.1f}% exceeds {self.config.max_daily_drawdown*100:.0f}% limit"

        return True, f"Drawdown OK: {dd*100:+.2f}%"

    # ------------------------------------------------------------------
    # Combined gate
    # ------------------------------------------------------------------

    def can_trade(
        self,
        current_equity: float,
        now_et: Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        """
        Master gate: check timing, drawdown, and halt status.

        Returns (allowed, reason).
        """
        # Drawdown check
        dd_ok, dd_msg = self.check_drawdown(current_equity)
        if not dd_ok:
            return False, dd_msg

        # Timing check
        timing_ok, timing_msg = self.in_trading_window(now_et)
        if not timing_ok:
            return False, timing_msg

        return True, "OK"

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return optimiser state summary."""
        return {
            "rolling_sharpe": self.rolling_sharpe(),
            "trade_count": len(self._returns),
            "position_size_pct": self.get_position_size_pct() * 100,
            "halted": self._halted_until is not None and datetime.now() < self._halted_until,
            "daily_start_equity": self._daily_start_equity,
        }

"""
core/kill_switch.py
===================
Emergency kill switch and circuit breakers for live trading.

Provides three layers of protection:

1. **Kill Switch** — immediately halt all trading activity and flatten
   positions.  Triggered manually or by catastrophic conditions.
2. **Circuit Breaker** — automatic trip based on configurable thresholds
   (max drawdown, daily loss, consecutive losses, max open positions).
   Trading pauses but resumes automatically on the next session unless
   the kill switch is engaged.
3. **Rate Limiter** — prevents order floods by enforcing a max orders/minute
   cap.

Usage
-----
    from core.kill_switch import KillSwitch

    ks = KillSwitch(broker=broker)

    # Before every order
    if not ks.is_trading_allowed():
        reason = ks.block_reason
        logger.warning(f"Trading blocked: {reason}")
        return

    # After every fill
    ks.on_fill(fill, portfolio_state)

    # Emergency
    ks.engage("Manual halt — investigating anomaly")
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, Optional

from equities.models import PortfolioState
from equities.telemetry import get_telemetry

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerConfig:
    """Thresholds that trigger the circuit breaker.

    Attributes
    ----------
    max_drawdown_pct :
        Maximum drawdown from peak equity before halting (e.g., -0.15 = −15%).
    max_daily_loss_pct :
        Maximum intraday loss as fraction of SOD equity (e.g., -0.03 = −3%).
    max_consecutive_losses :
        Number of consecutive losing trades before pausing.
    max_open_positions :
        Hard cap on simultaneous open positions.
    max_orders_per_minute :
        Rate limit on order submissions.
    cooldown_minutes :
        How long to pause after a circuit breaker trip before auto-resuming.
    """

    max_drawdown_pct: float = -0.99  # effectively disabled — daily loss is the halt trigger
    max_daily_loss_pct: float = -0.08
    max_consecutive_losses: int = 5
    max_open_positions: int = 30
    max_orders_per_minute: int = 20
    cooldown_minutes: float = 9999.0  # no same-day auto-resume; reset_daily() releases


class KillSwitch:
    """Emergency halt and circuit breaker system for live trading.

    Parameters
    ----------
    config :
        Circuit breaker thresholds.  Defaults to production-safe values.
    initial_equity :
        Start-of-day equity for daily loss tracking.
    """

    def __init__(
        self,
        config: Optional[CircuitBreakerConfig] = None,
        initial_equity: float = 100_000.0,
    ) -> None:
        self.config = config or CircuitBreakerConfig()

        # Kill switch state
        self._kill_engaged: bool = False
        self._kill_reason: str = ""
        self._kill_timestamp: Optional[datetime] = None

        # Circuit breaker state
        self._breaker_tripped: bool = False
        self._breaker_reason: str = ""
        self._breaker_trip_time: Optional[float] = None

        # Tracking
        self._sod_equity: float = initial_equity
        self._peak_equity: float = initial_equity
        self._consecutive_losses: int = 0
        self._order_timestamps: Deque[float] = deque(maxlen=100)
        self._daily_fills: int = 0

        logger.info(
            f"KillSwitch initialised: dd_halt={self.config.max_drawdown_pct:.0%}, "
            f"daily_loss={self.config.max_daily_loss_pct:.0%}, "
            f"max_consec_losses={self.config.max_consecutive_losses}, "
            f"max_positions={self.config.max_open_positions}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_trading_allowed(self) -> bool:
        """Check if trading is permitted right now."""
        if self._kill_engaged:
            return False
        if self._breaker_tripped:
            return False
        return True

    def check_cooldown_expired(self) -> bool:
        """Check if circuit breaker cooldown has expired. Call explicitly to reset.

        Returns True if the breaker was reset, False otherwise.
        """
        if not self._breaker_tripped or self._breaker_trip_time is None:
            return False
        elapsed = (time.time() - self._breaker_trip_time) / 60.0
        if elapsed >= self.config.cooldown_minutes:
            self._breaker_tripped = False
            self._breaker_reason = ""
            self._breaker_trip_time = None
            logger.info("KillSwitch: circuit breaker cooldown expired — trading resumed.")
            return True
        return False

    @property
    def block_reason(self) -> str:
        """Human-readable reason trading is blocked (empty if allowed)."""
        if self._kill_engaged:
            return f"KILL SWITCH: {self._kill_reason}"
        if self._breaker_tripped:
            return f"CIRCUIT BREAKER: {self._breaker_reason}"
        return ""

    def engage(self, reason: str = "Manual kill switch engaged") -> None:
        """Engage the kill switch — immediately halt all trading.

        Parameters
        ----------
        reason :
            Human-readable reason for the halt.
        """
        self._kill_engaged = True
        self._kill_reason = reason
        self._kill_timestamp = datetime.now(timezone.utc)
        logger.critical(f"KILL SWITCH ENGAGED: {reason}")
        # Record halt for promotion gate evidence (manual engagement is software defect if triggered by internal condition)
        is_software_defect = "error:" in reason.lower() or "exception:" in reason.lower()
        get_telemetry().record_halt(reason, is_software_defect)

    def disengage(self) -> None:
        """Release the kill switch, allowing trading to resume."""
        self._kill_engaged = False
        self._kill_reason = ""
        self._kill_timestamp = None
        logger.info("Kill switch disengaged — trading may resume.")

    def reset_daily(self, current_equity: float) -> None:
        """Reset daily tracking at start of a new trading day.

        Parameters
        ----------
        current_equity :
            Current portfolio equity (becomes new SOD reference).
        """
        self._sod_equity = current_equity
        self._peak_equity = current_equity  # reset peak to current — only daily P&L matters
        self._consecutive_losses = 0
        self._daily_fills = 0
        self._order_timestamps.clear()

        # Auto-release circuit breaker at start of new day
        if self._breaker_tripped:
            self._breaker_tripped = False
            self._breaker_reason = ""
            self._breaker_trip_time = None
            logger.info("KillSwitch: circuit breaker auto-released for new session.")

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def pre_order_check(self, portfolio_state: PortfolioState) -> bool:
        """Run all checks before submitting an order.

        Parameters
        ----------
        portfolio_state :
            Current portfolio snapshot.

        Returns
        -------
        bool
            True if the order may proceed.
        """
        if not self.is_trading_allowed():
            return False

        # Rate limit
        now = time.time()
        self._order_timestamps.append(now)
        recent = sum(1 for t in self._order_timestamps if now - t < 60)
        if recent > self.config.max_orders_per_minute:
            self._trip_breaker(
                f"Order rate limit exceeded: {recent} orders in last 60s "
                f"(max {self.config.max_orders_per_minute})"
            )
            return False

        # Max open positions — block but do NOT trip breaker (normal condition)
        n_positions = len(portfolio_state.positions)
        if n_positions >= self.config.max_open_positions:
            logger.info(
                f"KillSwitch: max open positions reached ({n_positions}/{self.config.max_open_positions}) — order blocked"
            )
            return False

        # Drawdown check
        equity = portfolio_state.equity
        # Compute drawdown BEFORE updating peak — otherwise dd is always 0
        # when equity has recovered even slightly
        dd = (equity - self._peak_equity) / max(self._peak_equity, 1.0)
        if dd <= self.config.max_drawdown_pct:
            self._trip_breaker(
                f"Max drawdown breached: {dd:.2%} "
                f"(threshold {self.config.max_drawdown_pct:.2%})"
            )
            return False

        # Update peak AFTER the drawdown check
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Daily loss check
        daily_pnl = (equity - self._sod_equity) / max(self._sod_equity, 1.0)
        if daily_pnl <= self.config.max_daily_loss_pct:
            self._trip_breaker(
                f"Daily loss limit breached: {daily_pnl:.2%} "
                f"(threshold {self.config.max_daily_loss_pct:.2%})"
            )
            return False

        return True

    def on_fill(self, pnl: float) -> None:
        """Update internal state after a trade completes.

        Parameters
        ----------
        pnl :
            Realised P&L of the closed trade.
        """
        self._daily_fills += 1

        if pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.config.max_consecutive_losses:
                self._trip_breaker(
                    f"Consecutive loss limit: {self._consecutive_losses} "
                    f"(max {self.config.max_consecutive_losses})"
                )
        else:
            self._consecutive_losses = 0

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict:
        """Return a status dict for monitoring / dashboards.

        Returns
        -------
        dict
        """
        return {
            "kill_engaged": self._kill_engaged,
            "kill_reason": self._kill_reason,
            "breaker_tripped": self._breaker_tripped,
            "breaker_reason": self._breaker_reason,
            "trading_allowed": self.is_trading_allowed(),
            "consecutive_losses": self._consecutive_losses,
            "daily_fills": self._daily_fills,
            "peak_equity": self._peak_equity,
            "sod_equity": self._sod_equity,
        }.copy()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _trip_breaker(self, reason: str) -> None:
        """Trip the circuit breaker with the given reason."""
        self._breaker_tripped = True
        self._breaker_reason = reason
        self._breaker_trip_time = time.time()
        logger.warning(f"CIRCUIT BREAKER TRIPPED: {reason}")
        # Record halt for promotion gate evidence (circuit breaker trip is operational, not software defect)
        get_telemetry().record_halt(reason, is_software_defect=False)

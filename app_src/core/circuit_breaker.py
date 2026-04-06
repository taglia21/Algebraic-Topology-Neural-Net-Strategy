"""
core/circuit_breaker.py
=======================
Daily loss limit enforcement and emergency flatten.

Research basis: Industry standard for small accounts ($5K-$25K):
  - Hard limit: 2% of equity per day
  - Soft limit: 1% of equity (reduce sizing to minimum)
  - After 6 consecutive losing days: halt and review

Implementation:
  1. Track daily P&L via account summary at start of day
  2. Monitor continuously during open positions
  3. If daily loss > soft limit: reduce max_contracts to 1
  4. If daily loss > hard limit: emergency flatten, halt new entries

Emergency flatten:
  Step 1: reqGlobalCancel() — cancel ALL open orders first
  Step 2: Iterate ib.positions(), place MarketOrder for each

References:
  - P&L Ledger (2025): Daily loss limits & weekly max drawdown rules
  - SBAI (2020): ARP Backtesting — Sharpe degradation study
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

HARD_LIMIT_PCT = 0.02   # 2% — emergency flatten
SOFT_LIMIT_PCT = 0.01   # 1% — reduce sizing

STATE_FILE = Path("/opt/atnn/data/circuit_breaker.json")


def load_cb_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {
        "date": str(date.today()),
        "nav_at_open": 0.0,
        "max_daily_loss_hit": False,
        "soft_limit_hit": False,
        "halted": False,
        "consecutive_losing_days": 0,
        "daily_pnl": 0.0,
    }


def save_cb_state(s: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=STATE_FILE.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(s, f, indent=2, default=str)
        os.replace(tmp, STATE_FILE)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


class CircuitBreaker:
    """
    Tracks daily P&L and enforces loss limits.
    
    Usage:
        cb = CircuitBreaker()
        await cb.initialize(ib)          # call at start of each cycle
        if cb.should_halt():
            return                        # stop trading for the day
        allowed = cb.max_contracts(2)     # get size-adjusted maximum
        await cb.check_and_act(ib, nav)  # check limits, flatten if needed
    """

    def __init__(self):
        self.state = load_cb_state()
        self._reset_if_new_day()

    def _reset_if_new_day(self):
        today = str(date.today())
        if self.state.get("date") != today:
            # Carry over consecutive_losing_days
            prev_loss = self.state.get("daily_pnl", 0.0)
            consec = self.state.get("consecutive_losing_days", 0)
            if prev_loss < 0:
                consec += 1
            else:
                consec = 0

            self.state = {
                "date": today,
                "nav_at_open": 0.0,
                "max_daily_loss_hit": False,
                "soft_limit_hit": False,
                "halted": False,
                "consecutive_losing_days": consec,
                "daily_pnl": 0.0,
            }
            save_cb_state(self.state)
            log.info("Circuit breaker: new day. Consecutive losing days: %d", consec)

    async def initialize(self, ib) -> float:
        """Record NAV at start of day. Call once per morning cycle."""
        try:
            acct = await ib.accountSummaryAsync()
            nav = float(next((s.value for s in acct if s.tag == "NetLiquidation"), 0))
            if self.state["nav_at_open"] == 0.0:
                self.state["nav_at_open"] = nav
                save_cb_state(self.state)
                log.info("Circuit breaker: NAV at open = $%.2f", nav)
            return nav
        except Exception as e:
            log.error("Circuit breaker init failed: %s", e)
            return 0.0

    def update_pnl(self, current_nav: float):
        """Update daily P&L and check limits."""
        if self.state["nav_at_open"] == 0:
            return

        self.state["daily_pnl"] = current_nav - self.state["nav_at_open"]
        daily_pnl_pct = self.state["daily_pnl"] / self.state["nav_at_open"]

        if daily_pnl_pct <= -HARD_LIMIT_PCT and not self.state["max_daily_loss_hit"]:
            self.state["max_daily_loss_hit"] = True
            self.state["halted"] = True
            log.error(
                "CIRCUIT BREAKER HARD LIMIT: daily P&L = $%.2f (%.2f%%). HALTING.",
                self.state["daily_pnl"], daily_pnl_pct * 100,
            )

        elif daily_pnl_pct <= -SOFT_LIMIT_PCT and not self.state["soft_limit_hit"]:
            self.state["soft_limit_hit"] = True
            log.warning(
                "Circuit breaker soft limit: daily P&L = $%.2f (%.2f%%). "
                "Reducing max contracts to 1.",
                self.state["daily_pnl"], daily_pnl_pct * 100,
            )

        save_cb_state(self.state)

    def should_halt(self) -> bool:
        """True if new entries should be blocked."""
        if self.state["halted"]:
            log.warning("Circuit breaker: HALTED. No new entries today.")
            return True
        if self.state.get("consecutive_losing_days", 0) >= 6:
            log.warning(
                "Circuit breaker: 6 consecutive losing days. Review before re-enabling."
            )
            return True
        return False

    def max_contracts(self, requested: int) -> int:
        """Return size-adjusted maximum contracts."""
        if self.state["halted"]:
            return 0
        if self.state["soft_limit_hit"]:
            return min(requested, 1)
        return requested

    async def emergency_flatten(self, ib):
        """
        Emergency flatten: cancel all orders, close all positions.
        Step 1: reqGlobalCancel() — MUST come first
        Step 2: Iterate positions, market-close each
        """
        log.error("EMERGENCY FLATTEN: cancelling all orders and closing all positions")
        try:
            # Step 1: cancel everything
            ib.reqGlobalCancel()
            await asyncio.sleep(1.0)

            # Step 2: close each position
            from ib_async import MarketOrder
            positions = ib.positions()
            if not positions:
                log.info("Emergency flatten: no open positions")
                return

            for pos in positions:
                qty = int(pos.position)
                if qty == 0:
                    continue
                action = "SELL" if qty > 0 else "BUY"
                order = MarketOrder(action, abs(qty))
                ib.placeOrder(pos.contract, order)
                log.error("  FLATTENING: %s %s × %d", action, pos.contract.symbol, abs(qty))

            await asyncio.sleep(3.0)
            log.info("Emergency flatten complete")
        except Exception as e:
            log.error("Emergency flatten failed: %s", e)

    async def check_and_act(self, ib, current_nav: float):
        """Check limits and act if needed."""
        self.update_pnl(current_nav)
        if self.state["max_daily_loss_hit"]:
            await self.emergency_flatten(ib)

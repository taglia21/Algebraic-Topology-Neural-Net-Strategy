"""
vrp/risk.py
===========
Portfolio-level risk management for the VRP strategy.

This module enforces hard limits on:
- Account drawdown (halt trading if exceeded)
- Daily P&L (halt if daily loss too large)
- Portfolio greeks (delta, vega exposure limits)
- Position count and risk concentration

The risk manager is the final gate before any trade is executed.
It can reject trades that would exceed limits, reduce position sizes,
or halt trading entirely.

Design: fail-safe defaults. If any data is missing or calculation
fails, the risk manager defaults to RESTRICTING trades, not allowing them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from vrp.config import RiskConfig
from vrp.strategy import SpreadPosition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Risk state
# ---------------------------------------------------------------------------

@dataclass
class RiskState:
    """Current risk metrics for the portfolio."""
    equity: float = 0.0
    high_water_mark: float = 0.0
    drawdown: float = 0.0           # current drawdown (negative)
    daily_pnl: float = 0.0         # today's P&L
    daily_pnl_pct: float = 0.0
    open_risk: float = 0.0          # total max risk across open positions
    open_risk_pct: float = 0.0      # as fraction of equity
    portfolio_delta: float = 0.0
    portfolio_vega: float = 0.0
    n_open_positions: int = 0
    is_trading_allowed: bool = True
    halt_reason: str = ""


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------

class RiskManager:
    """Portfolio risk management and trade gating.

    Acts as the final safety layer between strategy signals and execution.
    Tracks account state, enforces limits, and can halt trading.
    """

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self._high_water_mark: float = 0.0
        self._day_start_equity: float = 0.0
        self._current_date: Optional[date] = None
        self._halted: bool = False
        self._halt_reason: str = ""

    def update(
        self,
        equity: float,
        positions: List[SpreadPosition],
        portfolio_greeks: Dict[str, float],
        as_of: Optional[date] = None,
    ) -> RiskState:
        """Update risk state with current portfolio data.

        Should be called at least once per trading day before any
        trade decisions.

        Parameters
        ----------
        equity : Current account equity
        positions : List of open positions
        portfolio_greeks : Aggregate greeks dict (delta, gamma, theta, vega)
        as_of : Current date

        Returns
        -------
        Current risk state
        """
        today = as_of or date.today()

        # Track high water mark
        if equity > self._high_water_mark:
            self._high_water_mark = equity

        # Reset daily tracking on new day
        if today != self._current_date:
            self._day_start_equity = equity
            self._current_date = today

        # Calculate drawdown
        drawdown = 0.0
        if self._high_water_mark > 0:
            drawdown = (equity - self._high_water_mark) / self._high_water_mark

        # Daily P&L
        daily_pnl = equity - self._day_start_equity if self._day_start_equity > 0 else 0.0
        daily_pnl_pct = daily_pnl / self._day_start_equity if self._day_start_equity > 0 else 0.0

        # Open risk
        open_risk = sum(p.total_max_risk for p in positions if p.status == "open")
        open_risk_pct = open_risk / equity if equity > 0 else float('inf')

        # Build state
        state = RiskState(
            equity=equity,
            high_water_mark=self._high_water_mark,
            drawdown=drawdown,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            open_risk=open_risk,
            open_risk_pct=open_risk_pct,
            portfolio_delta=portfolio_greeks.get("delta", 0),
            portfolio_vega=portfolio_greeks.get("vega", 0),
            n_open_positions=len([p for p in positions if p.status == "open"]),
        )

        # Check limits
        self._check_limits(state)
        state.is_trading_allowed = not self._halted
        state.halt_reason = self._halt_reason

        return state

    def _check_limits(self, state: RiskState) -> None:
        """Check all risk limits and set halt flag if breached."""
        # Drawdown halt
        if state.drawdown < self.config.max_drawdown_halt:
            self._halted = True
            self._halt_reason = (
                f"Drawdown halt: {state.drawdown:.1%} < {self.config.max_drawdown_halt:.0%}"
            )
            logger.critical(self._halt_reason)
            return

        # Daily loss limit
        if state.daily_pnl_pct < self.config.daily_loss_limit:
            self._halted = True
            self._halt_reason = (
                f"Daily loss halt: {state.daily_pnl_pct:.1%} < {self.config.daily_loss_limit:.0%}"
            )
            logger.critical(self._halt_reason)
            return

        # Minimum equity
        if state.equity < self.config.min_equity:
            self._halted = True
            self._halt_reason = (
                f"Minimum equity: ${state.equity:,.0f} < ${self.config.min_equity:,.0f}"
            )
            logger.critical(self._halt_reason)
            return

        # If we were halted, check if we can resume on a new day
        # (don't keep halted forever — that's a death sentence for the account)
        if self._halted:
            # Resume if: new day AND daily loss is acceptable AND equity above min
            if (state.daily_pnl_pct >= self.config.daily_loss_limit * 0.5 and
                    state.equity >= self.config.min_equity):
                # Only stay halted if we're STILL in drawdown halt territory
                if state.drawdown >= self.config.max_drawdown_halt:
                    self._halted = False
                    self._halt_reason = ""
                    logger.info("Risk limits restored — trading resumed")

    def can_open_trade(
        self,
        state: RiskState,
        proposed_risk: float,
        proposed_delta: float = 0.0,
        proposed_vega: float = 0.0,
    ) -> Tuple[bool, str]:
        """Check if a proposed new trade is allowed.

        Parameters
        ----------
        state : Current risk state
        proposed_risk : Max loss of the proposed trade
        proposed_delta : Delta of the proposed trade
        proposed_vega : Vega of the proposed trade

        Returns
        -------
        (allowed, reason) tuple
        """
        if self._halted:
            return False, self._halt_reason

        # Check if adding this trade would breach delta limit
        new_delta = abs(state.portfolio_delta + proposed_delta)
        if new_delta > self.config.max_portfolio_delta:
            return False, (
                f"Delta limit: |{new_delta:.1f}| > {self.config.max_portfolio_delta}"
            )

        # Check vega limit
        new_vega = state.portfolio_vega + proposed_vega
        if new_vega < self.config.max_portfolio_vega:
            return False, (
                f"Vega limit: {new_vega:.0f} < {self.config.max_portfolio_vega}"
            )

        # Check drawdown-reduced sizing
        if state.drawdown < self.config.max_drawdown_reduce:
            return False, (
                f"Drawdown reduction zone: {state.drawdown:.1%} — no new trades"
            )

        return True, "OK"

    def position_size_adjustment(self, state: RiskState) -> float:
        """Return a sizing adjustment factor based on risk state.

        Returns a multiplier (0.0 to 1.0) to apply to base position size.
        Reduces size when approaching risk limits.
        """
        if self._halted:
            return 0.0

        adjustment = 1.0

        # Reduce size in drawdown zone
        if state.drawdown < self.config.max_drawdown_reduce * 0.5:
            # Linear reduction: full size at 0%, half at reduce threshold
            drawdown_ratio = abs(state.drawdown) / abs(self.config.max_drawdown_reduce)
            adjustment *= max(0.25, 1.0 - drawdown_ratio * 0.5)

        # Reduce if risk budget filling up
        if state.open_risk_pct > 0.06:  # >6% of account at risk
            risk_ratio = state.open_risk_pct / 0.10  # normalize to 10% limit
            adjustment *= max(0.5, 1.0 - (risk_ratio - 0.6))

        return min(1.0, max(0.0, adjustment))

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    def reset(self, initial_equity: float) -> None:
        """Reset risk manager state (for backtesting)."""
        self._high_water_mark = initial_equity
        self._day_start_equity = initial_equity
        self._current_date = None
        self._halted = False
        self._halt_reason = ""

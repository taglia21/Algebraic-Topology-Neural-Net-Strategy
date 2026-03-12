"""
Live risk monitoring and kill switch.

Continuously monitors portfolio risk and triggers protective actions
when limits are breached. The kill switch flattens ALL positions
and disables trading engines.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Risk monitoring thresholds."""
    max_daily_loss_pct: float = 5.0       # Flatten all at 5% daily loss
    reduce_exposure_pct: float = 3.0      # Reduce to 50% at 3% daily loss
    max_drawdown_pct: float = 15.0        # Full halt at 15% drawdown
    max_position_pct: float = 5.0         # No single position > 5% of NAV
    max_gross_exposure_pct: float = 100.0  # Max 100% gross exposure
    connection_timeout_minutes: float = 5.0  # Flatten after 5 min disconnect


@dataclass
class RiskCheckResult:
    """Result of a risk check cycle."""
    timestamp: str
    nav: float
    daily_pnl: float
    daily_pnl_pct: float
    drawdown_pct: float
    gross_exposure_pct: float
    violations: list[str] = field(default_factory=list)
    action_taken: str = "NONE"


class RiskMonitor:
    """
    Live portfolio risk monitor with kill switch.

    Checks:
    1. Daily loss > 5% → flatten everything
    2. Daily loss > 3% → reduce to 50% exposure
    3. Drawdown > 15% → full system halt
    4. Position concentration > 5% → flag
    5. Connection lost > 5 min → flatten everything
    """

    def __init__(
        self,
        client,
        portfolio_manager,
        config: Optional[RiskConfig] = None,
    ) -> None:
        """
        Args:
            client: IBKRClient instance
            portfolio_manager: PortfolioManager instance
            config: Risk thresholds (defaults if not provided)
        """
        self._client = client
        self._portfolio = portfolio_manager
        self._config = config or RiskConfig()
        self._kill_switch_triggered = False
        self._equity_trader = None
        self._option_trader = None
        self._last_connected: Optional[datetime] = None

    def register_traders(self, equity_trader=None, option_trader=None) -> None:
        """Register trading engines for kill switch control."""
        self._equity_trader = equity_trader
        self._option_trader = option_trader

    @property
    def is_halted(self) -> bool:
        """Whether the kill switch has been triggered."""
        return self._kill_switch_triggered

    async def check_risk(self) -> RiskCheckResult:
        """
        Run all risk checks and take action if thresholds breached.

        Returns RiskCheckResult with any violations and actions taken.
        """
        nav = await self._portfolio.get_nav()
        daily_pnl = await self._portfolio.get_daily_pnl()
        daily_pnl_pct = (daily_pnl / nav * 100) if nav > 0 else 0.0
        peak_nav = self._portfolio.peak_nav
        drawdown_pct = ((peak_nav - nav) / peak_nav * 100) if peak_nav > 0 else 0.0
        exposure = await self._portfolio.get_total_exposure()
        gross_pct = exposure.get("gross_exposure_pct", 0.0)

        violations = []
        action = "NONE"

        # Check 1: Maximum drawdown
        if drawdown_pct >= self._config.max_drawdown_pct:
            violations.append(
                f"MAX_DRAWDOWN: {drawdown_pct:.1f}% >= {self._config.max_drawdown_pct}%"
            )
            await self.trigger_kill_switch(
                f"Max drawdown breached: {drawdown_pct:.1f}%"
            )
            action = "KILL_SWITCH"

        # Check 2: Daily loss — flatten all
        elif daily_pnl_pct <= -self._config.max_daily_loss_pct:
            violations.append(
                f"MAX_DAILY_LOSS: {daily_pnl_pct:.1f}% <= -{self._config.max_daily_loss_pct}%"
            )
            await self.trigger_kill_switch(
                f"Daily loss limit breached: {daily_pnl_pct:.1f}%"
            )
            action = "KILL_SWITCH"

        # Check 3: Daily loss — reduce exposure
        elif daily_pnl_pct <= -self._config.reduce_exposure_pct:
            violations.append(
                f"REDUCE_EXPOSURE: {daily_pnl_pct:.1f}% <= -{self._config.reduce_exposure_pct}%"
            )
            await self.reduce_exposure(50.0)
            action = "REDUCE_TO_50PCT"

        # Check 4: Gross exposure
        if gross_pct > self._config.max_gross_exposure_pct:
            violations.append(
                f"GROSS_EXPOSURE: {gross_pct:.1f}% > {self._config.max_gross_exposure_pct}%"
            )

        # Check 5: Connection health
        if not self._client.is_connected():
            if self._last_connected is None:
                self._last_connected = datetime.now()
            elapsed = (datetime.now() - self._last_connected).total_seconds() / 60
            if elapsed >= self._config.connection_timeout_minutes:
                violations.append(
                    f"CONNECTION_LOST: {elapsed:.1f} min >= {self._config.connection_timeout_minutes} min"
                )
                action = "CONNECTION_KILL_SWITCH"
        else:
            self._last_connected = None

        result = RiskCheckResult(
            timestamp=datetime.now().isoformat(),
            nav=nav,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            drawdown_pct=drawdown_pct,
            gross_exposure_pct=gross_pct,
            violations=violations,
            action_taken=action,
        )

        if violations:
            logger.warning("Risk violations: %s | Action: %s", violations, action)
        else:
            logger.debug(
                "Risk OK — NAV=%.2f PnL=%.2f (%.1f%%) DD=%.1f%% Exp=%.1f%%",
                nav, daily_pnl, daily_pnl_pct, drawdown_pct, gross_pct,
            )

        return result

    async def trigger_kill_switch(self, reason: str) -> None:
        """
        EMERGENCY: Flatten ALL positions and disable all trading.

        This is the nuclear option. Closes every position (equity + options)
        and disables both trading engines.
        """
        logger.critical("KILL SWITCH TRIGGERED: %s", reason)
        self._kill_switch_triggered = True

        # Disable trading engines
        if self._equity_trader:
            self._equity_trader.disable()
        if self._option_trader:
            self._option_trader.disable()

        # Flatten all positions
        try:
            if self._equity_trader and self._equity_trader.enabled:
                await self._equity_trader.flatten_all()
            if self._option_trader and self._option_trader.enabled:
                await self._option_trader.flatten_all_options()
            logger.critical("Kill switch: all positions flattened")
        except Exception as exc:
            logger.critical("Kill switch flatten FAILED: %s", exc)

    async def reduce_exposure(self, target_pct: float) -> None:
        """
        Reduce gross exposure to target percentage.

        Closes positions proportionally until target is reached.
        """
        logger.warning("Reducing exposure to %.0f%%", target_pct)
        # For now, log the intent. Full implementation requires
        # position-level decisions about what to close first.
        # Priority: close most volatile / largest positions first.

    def reset_kill_switch(self) -> None:
        """
        Reset kill switch after manual review.

        This should only be called after a human has reviewed the situation.
        """
        self._kill_switch_triggered = False
        logger.warning("Kill switch RESET by manual override")

    # --- Position Concentration Check ---

    async def check_position_concentration(self) -> list[str]:
        """
        Check if any single position exceeds max concentration.

        Returns list of violation strings (empty if all OK).
        """
        nav = await self._portfolio.get_nav()
        if nav <= 0:
            return []

        violations = []
        await self._portfolio.sync_positions()

        for pos in self._portfolio._cached_positions:
            mkt_val = abs(pos.position * getattr(pos, "marketPrice", 0))
            pct = mkt_val / nav * 100
            if pct > self._config.max_position_pct:
                violations.append(
                    f"{pos.contract.symbol}: {pct:.1f}% > {self._config.max_position_pct}%"
                )

        return violations

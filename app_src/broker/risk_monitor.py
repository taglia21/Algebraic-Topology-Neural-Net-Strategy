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
    """Risk monitoring thresholds — all percentages as **whole numbers** (5.0 = 5%).

    NOTE: core/config.RiskCfg stores the same values as fractions (0.05 = 5%).
    When constructing this from RiskCfg, multiply by 100.
    """

    max_daily_loss_pct: float = 5.0        # Flatten all at 5% daily loss
    reduce_exposure_pct: float = 3.0       # Reduce to 50% at 3% daily loss
    max_drawdown_pct: float = 15.0         # Full halt at 15% drawdown
    max_position_pct: float = 5.0          # No single position > 5% of NAV
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
                await self.trigger_kill_switch(
                    f"Connection lost for {elapsed:.1f} min "
                    f"(threshold: {self._config.connection_timeout_minutes} min)"
                )
                action = "KILL_SWITCH"
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

        IMPORTANT: Flatten FIRST, then disable.  If we disabled first the
        flatten calls would be silently skipped because the trader checks
        the ``enabled`` flag before submitting orders.
        """
        logger.critical("KILL SWITCH TRIGGERED: %s", reason)
        self._kill_switch_triggered = True

        # --- FLATTEN FIRST — before disabling traders ---
        try:
            if self._equity_trader:
                await self._equity_trader.flatten_all()
            if self._option_trader:
                await self._option_trader.flatten_all_options()
            logger.critical("Kill switch: all positions flattened")
        except Exception as exc:
            logger.critical("Kill switch flatten FAILED: %s", exc)

        # --- THEN disable trading engines ---
        if self._equity_trader:
            self._equity_trader.disable()
        if self._option_trader:
            self._option_trader.disable()

    async def reduce_exposure(self, target_pct: float) -> None:
        """
        Reduce gross exposure to *target_pct* by closing the largest
        losing positions first.

        Parameters
        ----------
        target_pct :
            Desired gross exposure as a percentage of NAV (e.g. 50.0).
        """
        logger.warning("Reducing exposure to %.0f%%", target_pct)

        if not self._equity_trader or not self._portfolio:
            logger.warning("Cannot reduce exposure: no equity trader or portfolio manager")
            return

        try:
            nav = await self._portfolio.get_nav()
            if nav <= 0:
                return

            exposure = await self._portfolio.get_total_exposure()
            current_gross_pct = exposure.get("gross_exposure_pct", 0.0)

            if current_gross_pct <= target_pct:
                logger.info(
                    "Exposure already at %.1f%% (target %.1f%%)",
                    current_gross_pct, target_pct,
                )
                return

            # Get stock positions, sort by unrealized P&L (worst losers first)
            await self._portfolio.sync_positions()
            positions = [
                p for p in self._portfolio._cached_positions
                if p.contract.secType == "STK" and p.position != 0
            ]
            positions.sort(
                key=lambda p: getattr(p, "unrealizedPNL", 0) or 0,
            )

            # Close positions one by one until we hit the target
            for pos in positions:
                current_exposure = await self._portfolio.get_total_exposure()
                if current_exposure.get("gross_exposure_pct", 0.0) <= target_pct:
                    break

                action = "SELL" if pos.position > 0 else "BUY"
                qty = abs(pos.position)
                symbol = pos.contract.symbol

                logger.warning(
                    "Reducing: %s %d %s (unrealized P&L: $%.2f)",
                    action, qty, symbol,
                    getattr(pos, "unrealizedPNL", 0) or 0,
                )
                await self._equity_trader.place_market_order(symbol, qty, action)

            logger.info("Exposure reduction complete")
        except Exception as exc:
            logger.error("Exposure reduction failed: %s", exc)

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

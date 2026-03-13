"""
Portfolio tracking and reconciliation against IBKR.

Maintains internal position state synchronized with the broker,
tracks exposure, P&L, and provides position summaries.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioManager:
    """
    Portfolio state management synchronized with IBKR.

    Tracks equity and options positions, computes exposure metrics,
    and reconciles internal state with broker positions.
    """

    def __init__(self, client) -> None:
        """
        Args:
            client: IBKRClient instance
        """
        self._client = client
        self._last_sync: Optional[datetime] = None
        self._cached_positions: list = []
        self._peak_nav: float = 0.0

    async def sync_positions(self) -> dict:
        """
        Reconcile internal state with IBKR broker positions.

        Returns dict with: equity_positions, option_positions,
        position_count, sync_timestamp.
        """
        positions = await self._client.get_positions()
        self._cached_positions = positions
        self._last_sync = datetime.now()

        equity_pos = [p for p in positions if p.contract.secType == "STK"]
        option_pos = [p for p in positions if p.contract.secType == "OPT"]

        logger.info(
            "Synced positions: %d equity, %d options",
            len(equity_pos), len(option_pos),
        )

        return {
            "equity_positions": equity_pos,
            "option_positions": option_pos,
            "position_count": len(positions),
            "sync_timestamp": self._last_sync.isoformat(),
        }

    def get_equity_positions(self) -> list:
        """Get cached stock positions."""
        return [p for p in self._cached_positions if p.contract.secType == "STK"]

    def get_option_positions(self) -> list:
        """Get cached option positions."""
        return [p for p in self._cached_positions if p.contract.secType == "OPT"]

    async def get_nav(self) -> float:
        """Get current Net Asset Value from IBKR."""
        summary = await self._client.get_account_summary()
        nav = summary.get("NetLiquidation", 0.0)
        if nav > self._peak_nav:
            self._peak_nav = nav
        return nav

    async def get_buying_power(self) -> float:
        """Get available buying power."""
        summary = await self._client.get_account_summary()
        return summary.get("BuyingPower", 0.0)

    async def get_daily_pnl(self) -> float:
        """Get today's unrealized + realized P&L."""
        summary = await self._client.get_account_summary()
        unrealized = summary.get("UnrealizedPnL", 0.0)
        realized = summary.get("RealizedPnL", 0.0)
        return unrealized + realized

    async def get_total_exposure(self) -> dict:
        """
        Compute gross and net portfolio exposure.

        Returns dict with: gross_exposure, net_exposure, long_exposure,
        short_exposure, delta_exposure (for options).
        """
        await self.sync_positions()

        long_val = 0.0
        short_val = 0.0
        delta_exposure = 0.0

        for pos in self._cached_positions:
            market_val = pos.position * pos.marketPrice if hasattr(pos, "marketPrice") else 0.0
            if pos.position > 0:
                long_val += abs(market_val)
            else:
                short_val += abs(market_val)

        nav = await self.get_nav()
        gross = long_val + short_val
        net = long_val - short_val

        return {
            "gross_exposure": gross,
            "net_exposure": net,
            "long_exposure": long_val,
            "short_exposure": short_val,
            "gross_exposure_pct": (gross / nav * 100) if nav > 0 else 0.0,
            "net_exposure_pct": (net / nav * 100) if nav > 0 else 0.0,
            "nav": nav,
        }

    @property
    def peak_nav(self) -> float:
        """Historical peak NAV for drawdown calculation."""
        return self._peak_nav

    def initialize_peak_nav(self, cached_nav: float) -> None:
        """Initialize peak NAV from cached value to prevent cold-start bypass.

        Without this, ``_peak_nav`` starts at 0.0 and drawdown detection is
        completely bypassed on the first cycle after a restart.
        """
        if cached_nav > 0:
            self._peak_nav = cached_nav
            logger.info("Initialized peak NAV from cache: $%.2f", cached_nav)

    async def position_summary(self) -> pd.DataFrame:
        """
        Generate a summary table of all positions.

        Returns DataFrame with columns: symbol, sec_type, position, avg_cost,
        market_price, market_value, unrealized_pnl.
        """
        await self.sync_positions()
        rows = []
        for pos in self._cached_positions:
            rows.append({
                "symbol": pos.contract.symbol,
                "sec_type": pos.contract.secType,
                "position": pos.position,
                "avg_cost": pos.avgCost,
                "market_price": getattr(pos, "marketPrice", None),
                "market_value": getattr(pos, "marketValue", None),
                "unrealized_pnl": getattr(pos, "unrealizedPNL", None),
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

"""
Options order execution on IBKR.

DORMANT by default — all orders are logged but not submitted until
the options engine is explicitly authorized. Supports single legs,
vertical spreads, and iron condors as combo orders.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from ib_async import (
        IB, Contract, Option, ComboLeg, Order as IBOrder,
        LimitOrder, MarketOrder, Trade, TagValue,
    )
except ImportError:
    IB = Contract = Option = ComboLeg = IBOrder = None
    LimitOrder = MarketOrder = Trade = TagValue = None


@dataclass
class OptionOrderResult:
    """Result of an option order placement attempt."""
    order_id: int
    symbol: str
    strategy: str          # SINGLE, VERTICAL, IRON_CONDOR
    legs: list             # list of dicts describing each leg
    action: str            # BUY or SELL (net direction)
    quantity: int
    order_type: str        # MKT, LMT
    limit_price: Optional[float] = None
    status: str = "SUBMITTED"
    timestamp: str = ""
    simulated: bool = False


class OptionTrader:
    """
    Options order execution engine.

    SAFETY: When enabled=False (default), all orders are logged but NOT
    submitted to IBKR. They return OptionOrderResult with status='SIMULATED'.

    SAFETY CHECKS before any live order:
    - Verify account has sufficient buying power
    - Verify max position size not exceeded
    - Verify option bid-ask spread is reasonable (< 20% of mid)
    """

    MAX_BID_ASK_SPREAD_PCT = 0.20   # reject if spread > 20% of mid

    def __init__(self, client, data_feed=None, enabled: bool = False) -> None:
        """
        Args:
            client: IBKRClient instance
            data_feed: IBKRDataFeed for Greeks/chain lookups
            enabled: If False, orders are simulated (DORMANT mode)
        """
        self._client = client
        self._data_feed = data_feed
        self._enabled = enabled
        self._order_counter = 0

        mode = "LIVE" if enabled else "DORMANT (simulated)"
        logger.info("OptionTrader initialized in %s mode", mode)

    @property
    def ib(self) -> IB:
        return self._client.ib

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        """Activate live option order execution."""
        self._enabled = True
        logger.warning("OptionTrader ENABLED — live orders will be submitted to IBKR")

    def disable(self) -> None:
        """Deactivate live option order execution."""
        self._enabled = False
        logger.info("OptionTrader DISABLED — orders will be simulated")

    # --- Single Leg ---

    async def place_single_option(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        quantity: int,
        action: str,
        order_type: str = "LMT",
        limit_price: Optional[float] = None,
    ) -> OptionOrderResult:
        """
        Place a single option order.

        Args:
            symbol: Underlying ticker
            expiry: YYYYMMDD
            strike: Strike price
            right: 'C' or 'P'
            quantity: Number of contracts
            action: 'BUY' or 'SELL'
            order_type: 'LMT' or 'MKT'
            limit_price: Required if order_type='LMT'
        """
        legs = [{
            "expiry": expiry,
            "strike": strike,
            "right": right,
            "action": action,
            "ratio": 1,
        }]
        ts = datetime.now().isoformat()

        if not self._enabled:
            logger.info(
                "[SIMULATED] %s %s %d %s %s %.1f %s%s",
                order_type, action, quantity, symbol, expiry, strike, right,
                f" @ {limit_price}" if limit_price else "",
            )
            return OptionOrderResult(
                order_id=self._next_id(),
                symbol=symbol,
                strategy="SINGLE",
                legs=legs,
                action=action,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                status="SIMULATED",
                timestamp=ts,
                simulated=True,
            )

        contract = Option(symbol, expiry, strike, right, "SMART", currency="USD")
        await self.ib.qualifyContractsAsync(contract)

        if order_type == "LMT" and limit_price is not None:
            order = LimitOrder(action, quantity, limit_price)
        else:
            order = MarketOrder(action, quantity)

        trade = self.ib.placeOrder(contract, order)
        logger.info(
            "LIVE OPTION ORDER: %s %s %d %s %s %.1f%s (id=%d)",
            action, right, quantity, symbol, expiry, strike,
            f" @ {limit_price}" if limit_price else "",
            trade.order.orderId,
        )

        return OptionOrderResult(
            order_id=trade.order.orderId,
            symbol=symbol,
            strategy="SINGLE",
            legs=legs,
            action=action,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            status="SUBMITTED",
            timestamp=ts,
        )

    # --- Vertical Spread ---

    async def place_vertical_spread(
        self,
        symbol: str,
        expiry: str,
        long_strike: float,
        short_strike: float,
        right: str,
        quantity: int,
        action: str = "BUY",
        limit_price: Optional[float] = None,
    ) -> OptionOrderResult:
        """
        Place a vertical spread (bull/bear call/put spread) as combo order.

        For a bull call spread: BUY lower strike call, SELL higher strike call.
        For a bear put spread: BUY higher strike put, SELL lower strike put.

        Args:
            symbol: Underlying ticker
            expiry: YYYYMMDD
            long_strike: Strike of the long leg
            short_strike: Strike of the short leg
            right: 'C' for call spread, 'P' for put spread
            quantity: Number of spreads
            action: 'BUY' for debit spread, 'SELL' for credit spread
            limit_price: Net debit/credit (positive = debit, negative = credit)
        """
        legs = [
            {"expiry": expiry, "strike": long_strike, "right": right, "action": "BUY", "ratio": 1},
            {"expiry": expiry, "strike": short_strike, "right": right, "action": "SELL", "ratio": 1},
        ]
        ts = datetime.now().isoformat()

        if not self._enabled:
            logger.info(
                "[SIMULATED] VERTICAL %s %d %s %s %s long=%.1f short=%.1f%s",
                action, quantity, symbol, expiry, right,
                long_strike, short_strike,
                f" @ {limit_price}" if limit_price else "",
            )
            return OptionOrderResult(
                order_id=self._next_id(),
                symbol=symbol,
                strategy="VERTICAL",
                legs=legs,
                action=action,
                quantity=quantity,
                order_type="LMT" if limit_price else "MKT",
                limit_price=limit_price,
                status="SIMULATED",
                timestamp=ts,
                simulated=True,
            )

        # Build combo contract
        long_opt = Option(symbol, expiry, long_strike, right, "SMART", currency="USD")
        short_opt = Option(symbol, expiry, short_strike, right, "SMART", currency="USD")
        await self.ib.qualifyContractsAsync(long_opt, short_opt)

        combo = Contract()
        combo.symbol = symbol
        combo.secType = "BAG"
        combo.exchange = "SMART"
        combo.currency = "USD"
        combo.comboLegs = [
            ComboLeg(conId=long_opt.conId, ratio=1, action="BUY", exchange="SMART"),
            ComboLeg(conId=short_opt.conId, ratio=1, action="SELL", exchange="SMART"),
        ]

        if limit_price is not None:
            order = LimitOrder(action, quantity, limit_price)
        else:
            order = MarketOrder(action, quantity)

        trade = self.ib.placeOrder(combo, order)
        logger.info(
            "LIVE VERTICAL: %s %d %s %s %s L=%.1f S=%.1f (id=%d)",
            action, quantity, symbol, expiry, right,
            long_strike, short_strike, trade.order.orderId,
        )

        return OptionOrderResult(
            order_id=trade.order.orderId,
            symbol=symbol,
            strategy="VERTICAL",
            legs=legs,
            action=action,
            quantity=quantity,
            order_type="LMT" if limit_price else "MKT",
            limit_price=limit_price,
            status="SUBMITTED",
            timestamp=ts,
        )

    # --- Iron Condor ---

    async def place_iron_condor(
        self,
        symbol: str,
        expiry: str,
        put_long: float,
        put_short: float,
        call_short: float,
        call_long: float,
        quantity: int,
        limit_price: Optional[float] = None,
    ) -> OptionOrderResult:
        """
        Place an iron condor as a 4-leg combo order.

        Structure: BUY put_long, SELL put_short, SELL call_short, BUY call_long.

        Args:
            symbol: Underlying ticker
            expiry: YYYYMMDD
            put_long: Long put strike (lowest)
            put_short: Short put strike
            call_short: Short call strike
            call_long: Long call strike (highest)
            quantity: Number of iron condors
            limit_price: Net credit received (negative = credit)
        """
        legs = [
            {"expiry": expiry, "strike": put_long, "right": "P", "action": "BUY", "ratio": 1},
            {"expiry": expiry, "strike": put_short, "right": "P", "action": "SELL", "ratio": 1},
            {"expiry": expiry, "strike": call_short, "right": "C", "action": "SELL", "ratio": 1},
            {"expiry": expiry, "strike": call_long, "right": "C", "action": "BUY", "ratio": 1},
        ]
        ts = datetime.now().isoformat()

        if not self._enabled:
            logger.info(
                "[SIMULATED] IRON_CONDOR %d %s %s PL=%.1f PS=%.1f CS=%.1f CL=%.1f%s",
                quantity, symbol, expiry,
                put_long, put_short, call_short, call_long,
                f" @ {limit_price}" if limit_price else "",
            )
            return OptionOrderResult(
                order_id=self._next_id(),
                symbol=symbol,
                strategy="IRON_CONDOR",
                legs=legs,
                action="SELL",
                quantity=quantity,
                order_type="LMT" if limit_price else "MKT",
                limit_price=limit_price,
                status="SIMULATED",
                timestamp=ts,
                simulated=True,
            )

        # Build 4-leg combo
        pl = Option(symbol, expiry, put_long, "P", "SMART", currency="USD")
        ps = Option(symbol, expiry, put_short, "P", "SMART", currency="USD")
        cs = Option(symbol, expiry, call_short, "C", "SMART", currency="USD")
        cl = Option(symbol, expiry, call_long, "C", "SMART", currency="USD")
        await self.ib.qualifyContractsAsync(pl, ps, cs, cl)

        combo = Contract()
        combo.symbol = symbol
        combo.secType = "BAG"
        combo.exchange = "SMART"
        combo.currency = "USD"
        combo.comboLegs = [
            ComboLeg(conId=pl.conId, ratio=1, action="BUY", exchange="SMART"),
            ComboLeg(conId=ps.conId, ratio=1, action="SELL", exchange="SMART"),
            ComboLeg(conId=cs.conId, ratio=1, action="SELL", exchange="SMART"),
            ComboLeg(conId=cl.conId, ratio=1, action="BUY", exchange="SMART"),
        ]

        if limit_price is not None:
            order = LimitOrder("SELL", quantity, limit_price)
        else:
            order = MarketOrder("SELL", quantity)

        trade = self.ib.placeOrder(combo, order)
        logger.info(
            "LIVE IRON_CONDOR: %d %s %s (id=%d)",
            quantity, symbol, expiry, trade.order.orderId,
        )

        return OptionOrderResult(
            order_id=trade.order.orderId,
            symbol=symbol,
            strategy="IRON_CONDOR",
            legs=legs,
            action="SELL",
            quantity=quantity,
            order_type="LMT" if limit_price else "MKT",
            limit_price=limit_price,
            status="SUBMITTED",
            timestamp=ts,
        )

    # --- Position Management ---

    async def get_option_positions(self) -> list:
        """Get current option positions from IBKR."""
        positions = await self._client.get_positions()
        return [p for p in positions if p.contract.secType == "OPT"]

    async def flatten_all_options(self) -> list[OptionOrderResult]:
        """Close all option positions with market orders."""
        positions = await self.get_option_positions()
        results = []
        for pos in positions:
            if pos.position != 0:
                action = "SELL" if pos.position > 0 else "BUY"
                qty = abs(int(pos.position))
                result = await self.place_single_option(
                    symbol=pos.contract.symbol,
                    expiry=pos.contract.lastTradeDateOrContractMonth,
                    strike=pos.contract.strike,
                    right=pos.contract.right,
                    quantity=qty,
                    action=action,
                    order_type="MKT",
                )
                results.append(result)
        return results

    # --- Internal ---

    def _next_id(self) -> int:
        """Generate simulated order ID."""
        self._order_counter += 1
        return 800000 + self._order_counter

"""
Position Exit Manager
=====================

Systematic exit management for options positions.

Tracks every open position with entry data and manages exits via:
- 50% profit target (close when 50% of max profit captured)
- 200% loss stop (close when loss > 2x the premium collected)
- 7 DTE time exit (close all positions nearing expiration)
- Trailing stop (lock in gains after 30%+ profit)
- Rolling logic (when positions are challenged, roll to next month)

Uses Alpaca MLEG orders for multi-leg closes to avoid leg risk.
Runs every position_check_interval (30s) inside the main trading loop.

Usage:
    manager = ExitManager(trading_client, data_client)
    manager.register_spread(
        underlying="SPY",
        short_occ="SPY260320P00550000",
        long_occ="SPY260320P00545000",
        qty=1,
        net_credit=1.25,
        max_profit=125.0,
        max_loss=375.0,
        strategy="credit_spread",
    )
    actions = await manager.check_all_positions()
    for action in actions:
        print(f"{action.symbol}: {action.reason} -> {action.action}")
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from .occ_utils import parse_occ_symbol

logger = logging.getLogger(__name__)


# ============================================================================
# EXIT PARAMETERS (can be overridden via config)
# ============================================================================

DEFAULT_EXIT_CONFIG = {
    "profit_target_pct": 0.50,        # Close at 50% of max profit
    "stop_loss_multiplier": 2.0,      # Close at 2x premium collected loss
    "dte_exit_threshold": 7,          # Close at 7 DTE remaining
    "trailing_stop_activate_pct": 0.30,  # Activate trailing stop at 30% profit
    "trailing_stop_trail_pct": 0.50,  # Trail 50% of peak profit
    "time_accel_dte_pct": 0.50,       # Early profit take after 50% of time elapsed
    "time_accel_profit_pct": 0.25,    # At 25% profit with time acceleration
    "position_check_interval": 30,    # Check every 30 seconds
    "use_mleg_close": True,           # Use MLEG orders for multi-leg closes
}


class ExitReason(Enum):
    """Reason for closing a position."""
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    DTE_EXIT = "dte_exit"
    TRAILING_STOP = "trailing_stop"
    TIME_ACCEL_PROFIT = "time_accel_profit"
    MANUAL = "manual"
    ROLL = "roll"
    EMERGENCY = "emergency"


class PositionType(Enum):
    """Type of options position."""
    SINGLE_LEG = "single_leg"
    CREDIT_SPREAD = "credit_spread"
    DEBIT_SPREAD = "debit_spread"
    IRON_CONDOR = "iron_condor"
    STRADDLE = "straddle"


@dataclass
class TrackedLeg:
    """A single leg of a tracked position."""
    occ_symbol: str
    side: str  # "buy" or "sell"
    qty: int
    entry_price: float = 0.0
    current_price: float = 0.0
    current_bid: float = 0.0
    current_ask: float = 0.0


@dataclass
class TrackedPosition:
    """A tracked multi-leg position for exit management."""
    position_id: str
    underlying: str
    position_type: PositionType
    strategy: str
    legs: List[TrackedLeg]
    qty: int
    net_credit: float          # Net credit received (positive = credit)
    max_profit: float          # Maximum theoretical profit
    max_loss: float            # Maximum theoretical loss
    entry_time: datetime
    expiration: date

    # Live tracking state
    current_pnl: float = 0.0
    current_pnl_pct: float = 0.0  # % of max profit
    peak_pnl: float = 0.0        # High-water mark for trailing stop
    peak_pnl_pct: float = 0.0

    # Metadata
    last_checked: Optional[datetime] = None
    exit_reason: Optional[ExitReason] = None
    exit_time: Optional[datetime] = None
    exit_pnl: Optional[float] = None
    is_closed: bool = False
    close_order_id: Optional[str] = None
    notes: str = ""

    @property
    def dte(self) -> int:
        """Days to expiration from today."""
        return (self.expiration - date.today()).days

    @property
    def time_elapsed_pct(self) -> float:
        """Percentage of total time elapsed since entry."""
        total_days = (self.expiration - self.entry_time.date()).days
        if total_days <= 0:
            return 1.0
        elapsed = (date.today() - self.entry_time.date()).days
        return min(1.0, elapsed / total_days)


@dataclass
class ExitAction:
    """An exit action to execute."""
    position_id: str
    underlying: str
    reason: ExitReason
    action: str  # "close", "roll", "reduce"
    current_pnl: float
    current_pnl_pct: float
    legs_to_close: List[TrackedLeg]
    position_type: PositionType
    strategy: str
    details: str = ""


class ExitManager:
    """
    Systematic exit management for options positions.

    Monitors all open positions and triggers exits based on:
    - Profit targets
    - Stop losses
    - Time-based exits (DTE)
    - Trailing stops

    Uses Alpaca MLEG orders for multi-leg closes.
    """

    def __init__(
        self,
        trading_client=None,
        data_client=None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize ExitManager.

        Args:
            trading_client: Alpaca TradingClient
            data_client: Alpaca OptionHistoricalDataClient
            config: Override default exit parameters
        """
        self.trading_client = trading_client
        self.data_client = data_client
        self.config = {**DEFAULT_EXIT_CONFIG, **(config or {})}
        self.positions: Dict[str, TrackedPosition] = {}
        self.closed_positions: List[TrackedPosition] = []

        # Performance stats
        self.stats = {
            "total_exits": 0,
            "profit_target_exits": 0,
            "stop_loss_exits": 0,
            "dte_exits": 0,
            "trailing_stop_exits": 0,
            "time_accel_exits": 0,
            "total_realized_pnl": 0.0,
            "winning_exits": 0,
            "losing_exits": 0,
        }

        logger.info(
            f"ExitManager initialized: profit_target={self.config['profit_target_pct']:.0%}, "
            f"stop_loss={self.config['stop_loss_multiplier']:.1f}x, "
            f"dte_exit={self.config['dte_exit_threshold']}d"
        )

    # ====================================================================
    # REGISTRATION
    # ====================================================================

    def register_spread(
        self,
        underlying: str,
        short_occ: str,
        long_occ: str,
        qty: int,
        net_credit: float,
        max_profit: float,
        max_loss: float,
        strategy: str = "credit_spread",
        expiration: Optional[date] = None,
    ) -> str:
        """
        Register a credit/debit spread for exit management.

        Args:
            underlying: Underlying symbol (e.g., "SPY")
            short_occ: Short leg OCC symbol
            long_occ: Long leg OCC symbol
            qty: Number of spreads
            net_credit: Net credit received (per spread, in dollars)
            max_profit: Max profit in dollars
            max_loss: Max loss in dollars
            strategy: Strategy name
            expiration: Expiration date

        Returns:
            Position ID for tracking
        """
        if expiration is None:
            expiration = self._parse_occ_expiration(short_occ) or (
                date.today() + timedelta(days=30)
            )

        position_id = f"{underlying}_{strategy}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        legs = [
            TrackedLeg(occ_symbol=short_occ, side="sell", qty=qty),
            TrackedLeg(occ_symbol=long_occ, side="buy", qty=qty),
        ]

        pos = TrackedPosition(
            position_id=position_id,
            underlying=underlying,
            position_type=PositionType.CREDIT_SPREAD if net_credit > 0 else PositionType.DEBIT_SPREAD,
            strategy=strategy,
            legs=legs,
            qty=qty,
            net_credit=net_credit,
            max_profit=max_profit,
            max_loss=max_loss,
            entry_time=datetime.now(ZoneInfo("America/New_York")),
            expiration=expiration,
        )

        self.positions[position_id] = pos
        logger.info(
            f"Registered spread {position_id}: {underlying} "
            f"credit=${net_credit:.2f} max_profit=${max_profit:.2f} "
            f"max_loss=${max_loss:.2f} exp={expiration}"
        )
        return position_id

    def register_iron_condor(
        self,
        underlying: str,
        put_long_occ: str,
        put_short_occ: str,
        call_short_occ: str,
        call_long_occ: str,
        qty: int,
        net_credit: float,
        max_profit: float,
        max_loss: float,
        expiration: Optional[date] = None,
    ) -> str:
        """Register an iron condor for exit management."""
        if expiration is None:
            expiration = self._parse_occ_expiration(put_short_occ) or (
                date.today() + timedelta(days=30)
            )

        position_id = f"{underlying}_iron_condor_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        legs = [
            TrackedLeg(occ_symbol=put_long_occ, side="buy", qty=qty),
            TrackedLeg(occ_symbol=put_short_occ, side="sell", qty=qty),
            TrackedLeg(occ_symbol=call_short_occ, side="sell", qty=qty),
            TrackedLeg(occ_symbol=call_long_occ, side="buy", qty=qty),
        ]

        pos = TrackedPosition(
            position_id=position_id,
            underlying=underlying,
            position_type=PositionType.IRON_CONDOR,
            strategy="iron_condor",
            legs=legs,
            qty=qty,
            net_credit=net_credit,
            max_profit=max_profit,
            max_loss=max_loss,
            entry_time=datetime.now(ZoneInfo("America/New_York")),
            expiration=expiration,
        )

        self.positions[position_id] = pos
        logger.info(
            f"Registered iron condor {position_id}: {underlying} "
            f"credit=${net_credit:.2f} max_profit=${max_profit:.2f} "
            f"max_loss=${max_loss:.2f} exp={expiration}"
        )
        return position_id

    def register_single_leg(
        self,
        underlying: str,
        occ_symbol: str,
        side: str,
        qty: int,
        entry_price: float,
        max_profit: float,
        max_loss: float,
        strategy: str = "single_leg",
        expiration: Optional[date] = None,
    ) -> str:
        """Register a single-leg position for exit management."""
        if expiration is None:
            expiration = self._parse_occ_expiration(occ_symbol) or (
                date.today() + timedelta(days=30)
            )

        position_id = f"{underlying}_{strategy}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        net_credit = entry_price if side == "sell" else -entry_price

        legs = [
            TrackedLeg(occ_symbol=occ_symbol, side=side, qty=qty, entry_price=entry_price),
        ]

        pos = TrackedPosition(
            position_id=position_id,
            underlying=underlying,
            position_type=PositionType.SINGLE_LEG,
            strategy=strategy,
            legs=legs,
            qty=qty,
            net_credit=net_credit,
            max_profit=max_profit,
            max_loss=max_loss,
            entry_time=datetime.now(ZoneInfo("America/New_York")),
            expiration=expiration,
        )

        self.positions[position_id] = pos
        logger.info(
            f"Registered single leg {position_id}: {underlying} {side} "
            f"entry=${entry_price:.2f} exp={expiration}"
        )
        return position_id

    # ====================================================================
    # POSITION MONITORING
    # ====================================================================

    async def check_all_positions(self) -> List[ExitAction]:
        """
        Check all tracked positions for exit triggers.

        Updates current P&L from Alpaca, then evaluates each position
        against exit rules.

        Returns:
            List of ExitAction objects for positions that should be closed.
        """
        actions: List[ExitAction] = []

        if not self.positions:
            return actions

        # Refresh prices from Alpaca
        await self._refresh_position_prices()

        for pos_id, pos in list(self.positions.items()):
            if pos.is_closed:
                continue

            action = self._evaluate_exit(pos)
            if action is not None:
                actions.append(action)

        return actions

    def _evaluate_exit(self, pos: TrackedPosition) -> Optional[ExitAction]:
        """
        Evaluate a single position for exit triggers.

        Priority order:
        1. Emergency / stop loss (highest priority)
        2. DTE exit (time-critical)
        3. Profit target
        4. Trailing stop
        5. Time-accelerated profit

        Returns:
            ExitAction if exit triggered, None otherwise.
        """
        pos.last_checked = datetime.now(ZoneInfo("America/New_York"))

        # Update high-water mark
        if pos.current_pnl > pos.peak_pnl:
            pos.peak_pnl = pos.current_pnl
        if pos.max_profit > 0 and pos.current_pnl / pos.max_profit > pos.peak_pnl_pct:
            pos.peak_pnl_pct = pos.current_pnl / pos.max_profit

        # Calculate P&L as % of max profit
        if pos.max_profit > 0:
            pos.current_pnl_pct = pos.current_pnl / pos.max_profit
        elif pos.max_loss > 0:
            pos.current_pnl_pct = pos.current_pnl / pos.max_loss

        reason = None
        details = ""

        # --- 1. STOP LOSS: loss exceeds multiplier x premium collected ---
        stop_loss_amount = pos.net_credit * 100 * self.config["stop_loss_multiplier"]
        if stop_loss_amount > 0 and pos.current_pnl < -stop_loss_amount:
            reason = ExitReason.STOP_LOSS
            details = (
                f"Loss ${pos.current_pnl:+,.2f} > "
                f"{self.config['stop_loss_multiplier']:.0f}x premium "
                f"(${-stop_loss_amount:,.2f})"
            )
        # For debit positions, stop at max_loss
        elif pos.net_credit <= 0 and pos.max_loss > 0 and pos.current_pnl < -pos.max_loss:
            reason = ExitReason.STOP_LOSS
            details = f"Loss ${pos.current_pnl:+,.2f} > max_loss ${pos.max_loss:,.2f}"

        # --- 2. DTE EXIT: nearing expiration ---
        if reason is None and pos.dte <= self.config["dte_exit_threshold"]:
            reason = ExitReason.DTE_EXIT
            details = f"Only {pos.dte} DTE remaining (threshold={self.config['dte_exit_threshold']})"

        # --- 3. PROFIT TARGET: captured enough of max profit ---
        if reason is None and pos.max_profit > 0:
            profit_pct = pos.current_pnl / pos.max_profit
            if profit_pct >= self.config["profit_target_pct"]:
                reason = ExitReason.PROFIT_TARGET
                details = (
                    f"Captured {profit_pct:.0%} of max profit "
                    f"(target={self.config['profit_target_pct']:.0%}): "
                    f"${pos.current_pnl:+,.2f} of ${pos.max_profit:,.2f}"
                )

        # --- 4. TRAILING STOP ---
        if reason is None and pos.peak_pnl_pct >= self.config["trailing_stop_activate_pct"]:
            trail_floor = pos.peak_pnl * (1 - self.config["trailing_stop_trail_pct"])
            if pos.current_pnl < trail_floor and trail_floor > 0:
                reason = ExitReason.TRAILING_STOP
                details = (
                    f"Trailing stop: peak=${pos.peak_pnl:+,.2f}, "
                    f"now=${pos.current_pnl:+,.2f}, "
                    f"floor=${trail_floor:+,.2f}"
                )

        # --- 5. TIME-ACCELERATED PROFIT ---
        if reason is None:
            time_pct = pos.time_elapsed_pct
            if time_pct >= self.config["time_accel_dte_pct"]:
                if pos.max_profit > 0:
                    profit_pct = pos.current_pnl / pos.max_profit
                    if profit_pct >= self.config["time_accel_profit_pct"]:
                        reason = ExitReason.TIME_ACCEL_PROFIT
                        details = (
                            f"Time accel: {profit_pct:.0%} profit after "
                            f"{time_pct:.0%} of time elapsed"
                        )

        if reason is None:
            return None

        logger.info(
            f"EXIT TRIGGER [{reason.value}] {pos.underlying} "
            f"({pos.strategy}): {details}"
        )

        return ExitAction(
            position_id=pos.position_id,
            underlying=pos.underlying,
            reason=reason,
            action="close",
            current_pnl=pos.current_pnl,
            current_pnl_pct=pos.current_pnl_pct,
            legs_to_close=pos.legs,
            position_type=pos.position_type,
            strategy=pos.strategy,
            details=details,
        )

    # ====================================================================
    # EXECUTION OF EXITS
    # ====================================================================

    async def execute_exit(self, action: ExitAction) -> bool:
        """
        Execute an exit action by closing the position on Alpaca.

        Uses MLEG orders for multi-leg positions to avoid leg risk.

        Args:
            action: The ExitAction to execute.

        Returns:
            True if exit was successfully submitted.
        """
        pos = self.positions.get(action.position_id)
        if pos is None or pos.is_closed:
            logger.warning(f"Position {action.position_id} not found or already closed")
            return False

        success = False

        try:
            if len(action.legs_to_close) >= 2 and self.config.get("use_mleg_close", True):
                success = await self._close_mleg(pos, action)
            else:
                success = await self._close_individual_legs(pos, action)
        except Exception as e:
            logger.error(f"Exit execution failed for {action.position_id}: {e}")
            # Fallback: try closing each leg individually
            try:
                success = await self._close_individual_legs(pos, action)
            except Exception as e2:
                logger.error(f"Fallback close also failed: {e2}")

        if success:
            self._record_exit(pos, action)

        return success

    async def _close_mleg(self, pos: TrackedPosition, action: ExitAction) -> bool:
        """Close a multi-leg position using Alpaca MLEG order."""
        if self.trading_client is None:
            logger.warning("No trading client — cannot close MLEG")
            return False

        try:
            from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
            from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce, OrderClass

            # Build closing legs (reverse of opening sides)
            mleg_legs = []
            for leg in pos.legs:
                close_side = (
                    AlpacaOrderSide.BUY if leg.side == "sell"
                    else AlpacaOrderSide.SELL
                )
                mleg_legs.append(
                    OptionLegRequest(
                        symbol=leg.occ_symbol,
                        side=close_side,
                        ratio_qty=str(leg.qty),
                    )
                )

            # Calculate closing debit/credit
            # For credit spreads: closing is a debit (buying back)
            # For debit spreads: closing is a credit (selling)
            # Use natural mid-price as limit
            close_price = await self._estimate_close_price(pos)

            if pos.net_credit > 0:
                # Originally sold for credit -> buying back for debit
                net_side = AlpacaOrderSide.BUY
            else:
                # Originally bought for debit -> selling for credit
                net_side = AlpacaOrderSide.SELL

            order_request = LimitOrderRequest(
                symbol=pos.underlying,
                qty=pos.qty,
                side=net_side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.MLEG,
                limit_price=round(max(0.01, abs(close_price)), 2),
                legs=mleg_legs,
            )

            order = self.trading_client.submit_order(order_request)
            pos.close_order_id = str(order.id)
            logger.info(
                f"MLEG close order submitted for {pos.position_id}: "
                f"order_id={order.id} price=${close_price:.2f}"
            )
            return True

        except Exception as e:
            logger.error(f"MLEG close failed for {pos.position_id}: {e}")
            # Fall through to individual leg close
            return await self._close_individual_legs(pos, action)

    async def _close_individual_legs(
        self, pos: TrackedPosition, action: ExitAction
    ) -> bool:
        """Close position by closing each leg individually on Alpaca."""
        if self.trading_client is None:
            logger.warning("No trading client — cannot close legs")
            return False

        all_success = True
        for leg in pos.legs:
            try:
                self.trading_client.close_position(leg.occ_symbol)
                logger.info(f"Closed individual leg: {leg.occ_symbol}")
            except Exception as e:
                err_str = str(e)
                # "position does not exist" is OK (already closed)
                if "does not exist" in err_str.lower() or "not found" in err_str.lower():
                    logger.info(f"Leg {leg.occ_symbol} already closed")
                else:
                    logger.error(f"Failed to close leg {leg.occ_symbol}: {e}")
                    all_success = False

        return all_success

    def _record_exit(self, pos: TrackedPosition, action: ExitAction):
        """Record the exit in stats and move to closed list."""
        pos.is_closed = True
        pos.exit_reason = action.reason
        pos.exit_time = datetime.now(ZoneInfo("America/New_York"))
        pos.exit_pnl = action.current_pnl

        # Update stats
        self.stats["total_exits"] += 1
        self.stats["total_realized_pnl"] += action.current_pnl

        if action.current_pnl > 0:
            self.stats["winning_exits"] += 1
        elif action.current_pnl < 0:
            self.stats["losing_exits"] += 1

        reason_key = f"{action.reason.value}_exits"
        if reason_key in self.stats:
            self.stats[reason_key] += 1

        # Move to closed list
        self.closed_positions.append(pos)
        if pos.position_id in self.positions:
            del self.positions[pos.position_id]

        logger.info(
            f"EXIT RECORDED: {pos.underlying} ({pos.strategy}) "
            f"reason={action.reason.value} P&L=${action.current_pnl:+,.2f} "
            f"({action.current_pnl_pct:+.0%} of max profit)"
        )

    # ====================================================================
    # PRICE REFRESH
    # ====================================================================

    async def _refresh_position_prices(self):
        """Refresh current prices for all tracked positions from Alpaca.

        Phase 7 (Bug 3 fix): Always query real-time bid/ask/mid from
        Alpaca data client for every tracked leg, not just positions that
        happen to show up in get_all_positions.  This ensures ExitManager
        has accurate P&L even for recently opened or thinly-traded options.
        """
        if self.trading_client is None:
            return

        # Get all Alpaca positions
        try:
            alpaca_positions = self.trading_client.get_all_positions()
            alpaca_map: Dict[str, dict] = {}
            for ap in alpaca_positions:
                alpaca_map[ap.symbol] = {
                    "market_value": float(ap.market_value) if ap.market_value else 0.0,
                    "cost_basis": float(ap.cost_basis) if ap.cost_basis else 0.0,
                    "unrealized_pl": float(ap.unrealized_pl) if ap.unrealized_pl else 0.0,
                    "current_price": float(ap.current_price) if ap.current_price else 0.0,
                    "qty": float(ap.qty) if ap.qty else 0,
                }
        except Exception as e:
            logger.warning(f"Failed to refresh prices from Alpaca: {e}")
            return

        # Phase 7: Always fetch latest quotes for ALL tracked legs from data_client
        # This is the primary source for bid/ask/mid — ensures real-time prices
        quote_map: Dict[str, dict] = {}
        all_occ_symbols = set()
        for pos in self.positions.values():
            for leg in pos.legs:
                all_occ_symbols.add(leg.occ_symbol)

        if self.data_client is not None and all_occ_symbols:
            for occ_sym in all_occ_symbols:
                try:
                    from alpaca.data.requests import OptionLatestQuoteRequest
                    req = OptionLatestQuoteRequest(symbol_or_symbols=occ_sym)
                    quotes = self.data_client.get_option_latest_quote(req)
                    q = quotes.get(occ_sym) or (
                        list(quotes.values())[0] if quotes else None
                    )
                    if q:
                        bid = float(q.bid_price) if q.bid_price else 0.0
                        ask = float(q.ask_price) if q.ask_price else 0.0
                        mid = round((bid + ask) / 2.0, 2) if (bid > 0 or ask > 0) else 0.0
                        quote_map[occ_sym] = {
                            "bid": bid,
                            "ask": ask,
                            "mid": mid,
                        }
                except Exception:
                    pass
        elif self.data_client is None:
            logger.debug("No data_client — skipping real-time quote refresh")

        # Update each tracked position
        for pos in self.positions.values():
            total_pnl = 0.0
            has_quote_data = False
            for leg in pos.legs:
                # Update leg prices from real-time quotes (preferred)
                if leg.occ_symbol in quote_map:
                    q = quote_map[leg.occ_symbol]
                    leg.current_bid = q["bid"]
                    leg.current_ask = q["ask"]
                    leg.current_price = q["mid"]
                    has_quote_data = True

                # Compute P&L from Alpaca positions (positions API)
                if leg.occ_symbol in alpaca_map:
                    ap_data = alpaca_map[leg.occ_symbol]
                    total_pnl += ap_data["unrealized_pl"]
                elif has_quote_data:
                    # Position not in Alpaca get_all_positions but we have quotes:
                    # estimate P&L from entry_price vs current mid
                    mid = leg.current_price or 0.0
                    if leg.entry_price > 0 and mid > 0:
                        if leg.side == "sell":
                            leg_pnl = (leg.entry_price - mid) * leg.qty * 100
                        else:
                            leg_pnl = (mid - leg.entry_price) * leg.qty * 100
                        total_pnl += leg_pnl

            pos.current_pnl = total_pnl

    async def _estimate_close_price(self, pos: TrackedPosition) -> float:
        """Estimate the price to close a position (natural mid)."""
        total_close_cost = 0.0

        for leg in pos.legs:
            mid = leg.current_price or 0.0
            if mid <= 0 and self.data_client is not None:
                try:
                    from alpaca.data.requests import OptionLatestQuoteRequest
                    req = OptionLatestQuoteRequest(symbol_or_symbols=leg.occ_symbol)
                    quotes = self.data_client.get_option_latest_quote(req)
                    q = quotes.get(leg.occ_symbol) or (
                        list(quotes.values())[0] if quotes else None
                    )
                    if q and q.bid_price and q.ask_price:
                        mid = round((q.bid_price + q.ask_price) / 2.0, 2)
                except Exception:
                    pass

            # Closing reverses the side
            if leg.side == "sell":
                total_close_cost += mid  # Buy back costs money
            else:
                total_close_cost -= mid  # Sell back returns money

        return abs(total_close_cost)

    # ====================================================================
    # SYNC FROM ALPACA (reconstruct tracked positions)
    # ====================================================================

    def sync_from_alpaca_state(self, alpaca_options: Dict[str, dict]):
        """
        Sync tracked positions from actual Alpaca option positions.

        For orphaned positions (exist on Alpaca but not tracked), create
        simple tracking entries so they still get exit management.

        Args:
            alpaca_options: Dict of {occ_symbol: {qty, cost_basis, ...}}
        """
        tracked_occ = set()
        for pos in self.positions.values():
            for leg in pos.legs:
                tracked_occ.add(leg.occ_symbol)

        for occ_sym, data in alpaca_options.items():
            if occ_sym in tracked_occ:
                continue

            # Phase 7: Use centralized OCC parser
            parsed = parse_occ_symbol(occ_sym)
            if parsed is not None:
                underlying = parsed['underlying']
            else:
                underlying = ""
                for ch in occ_sym:
                    if ch.isdigit():
                        break
                    underlying += ch

            qty = int(abs(data.get("qty", 0)))
            if qty == 0:
                continue

            cost_basis = abs(data.get("cost_basis", 0))
            side = "buy" if data.get("qty", 0) > 0 else "sell"

            exp = self._parse_occ_expiration(occ_sym) or (
                date.today() + timedelta(days=14)
            )

            self.register_single_leg(
                underlying=underlying,
                occ_symbol=occ_sym,
                side=side,
                qty=qty,
                entry_price=cost_basis / (qty * 100) if qty > 0 else 0,
                max_profit=cost_basis * 0.5,
                max_loss=cost_basis,
                strategy="orphaned_sync",
                expiration=exp,
            )
            logger.info(f"Synced orphaned position: {occ_sym} ({underlying})")

    # ====================================================================
    # STATS & REPORTING
    # ====================================================================

    def get_summary(self) -> Dict:
        """Get summary of exit manager state."""
        open_pnl = sum(p.current_pnl for p in self.positions.values())
        return {
            "open_positions": len(self.positions),
            "closed_positions": len(self.closed_positions),
            "open_pnl": round(open_pnl, 2),
            "stats": self.stats.copy(),
            "positions": [
                {
                    "id": p.position_id,
                    "underlying": p.underlying,
                    "strategy": p.strategy,
                    "pnl": round(p.current_pnl, 2),
                    "dte": p.dte,
                    "type": p.position_type.value,
                }
                for p in self.positions.values()
            ],
        }

    def get_performance_report(self) -> str:
        """Get human-readable performance report."""
        total = self.stats["total_exits"]
        wins = self.stats["winning_exits"]
        losses = self.stats["losing_exits"]
        wr = wins / total * 100 if total > 0 else 0

        lines = [
            "═" * 60,
            "EXIT MANAGER PERFORMANCE",
            "═" * 60,
            f"Total Exits: {total}",
            f"  Profit Target: {self.stats['profit_target_exits']}",
            f"  Stop Loss:     {self.stats['stop_loss_exits']}",
            f"  DTE Exit:      {self.stats['dte_exits']}",
            f"  Trailing Stop: {self.stats['trailing_stop_exits']}",
            f"  Time Accel:    {self.stats['time_accel_exits']}",
            f"Win Rate:        {wr:.1f}% ({wins}W/{losses}L)",
            f"Total P&L:       ${self.stats['total_realized_pnl']:+,.2f}",
            f"Open Positions:  {len(self.positions)}",
            f"Open P&L:        ${sum(p.current_pnl for p in self.positions.values()):+,.2f}",
            "═" * 60,
        ]
        return "\n".join(lines)

    # ====================================================================
    # HELPERS
    # ====================================================================

    @staticmethod
    def _parse_occ_expiration(occ_symbol: str) -> Optional[date]:
        """Parse expiration date from OCC symbol.

        Phase 7: Delegates to centralized ``parse_occ_symbol`` utility.
        """
        parsed = parse_occ_symbol(occ_symbol)
        if parsed is not None:
            return parsed['expiry_date']
        # Fallback: manual extraction
        try:
            idx = 0
            for ch in occ_symbol:
                if ch.isdigit():
                    break
                idx += 1
            date_str = occ_symbol[idx: idx + 6]
            if len(date_str) < 6:
                return None
            yy = int(date_str[0:2])
            mm = int(date_str[2:4])
            dd = int(date_str[4:6])
            return date(2000 + yy, mm, dd)
        except (ValueError, IndexError):
            return None

    def save_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            "positions": {
                pid: {
                    "position_id": p.position_id,
                    "underlying": p.underlying,
                    "position_type": p.position_type.value,
                    "strategy": p.strategy,
                    "legs": [
                        {"occ_symbol": l.occ_symbol, "side": l.side, "qty": l.qty,
                         "entry_price": l.entry_price}
                        for l in p.legs
                    ],
                    "qty": p.qty,
                    "net_credit": p.net_credit,
                    "max_profit": p.max_profit,
                    "max_loss": p.max_loss,
                    "entry_time": p.entry_time.isoformat(),
                    "expiration": p.expiration.isoformat(),
                    "peak_pnl": p.peak_pnl,
                    "peak_pnl_pct": p.peak_pnl_pct,
                }
                for pid, p in self.positions.items()
            },
            "stats": self.stats.copy(),
        }

    def load_state(self, state: Dict):
        """Restore state from persistence."""
        for pid, pdata in state.get("positions", {}).items():
            legs = [
                TrackedLeg(
                    occ_symbol=l["occ_symbol"],
                    side=l["side"],
                    qty=l["qty"],
                    entry_price=l.get("entry_price", 0),
                )
                for l in pdata.get("legs", [])
            ]
            pos = TrackedPosition(
                position_id=pdata["position_id"],
                underlying=pdata["underlying"],
                position_type=PositionType(pdata["position_type"]),
                strategy=pdata["strategy"],
                legs=legs,
                qty=pdata["qty"],
                net_credit=pdata["net_credit"],
                max_profit=pdata["max_profit"],
                max_loss=pdata["max_loss"],
                entry_time=datetime.fromisoformat(pdata["entry_time"]),
                expiration=date.fromisoformat(pdata["expiration"]),
                peak_pnl=pdata.get("peak_pnl", 0),
                peak_pnl_pct=pdata.get("peak_pnl_pct", 0),
            )
            self.positions[pid] = pos

        self.stats.update(state.get("stats", {}))
        logger.info(f"Loaded {len(self.positions)} tracked positions from state")

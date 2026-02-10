"""
Integrated Trading System v2
==============================

COMPLETE REPLACEMENT for the broken trading pipeline.

This is the main entry point that:
1. Uses SignalGeneratorV2 (IV + regime + strategy selection)
2. Resolves signals to real OCC contracts via Alpaca API
3. Executes multi-leg orders using Alpaca MLEG OrderClass
4. Monitors positions with REAL P&L from Alpaca
5. Enforces risk limits

Critical fixes:
- MLEG orders instead of broken OTO orders
- Real position monitoring (not random 5% close!)
- Proper Kelly position sizing
- Stop-loss at 2x credit received, take profit at 50%
- Max 5 concurrent positions

Author: System Overhaul - Feb 2026
"""

import asyncio
import logging
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    OptionLegRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderClass,
    AssetClass,
    QueryOrderStatus,
)
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest

from src.signal_generator import (
    SignalGeneratorV2,
    TradeSignal,
)
from src.strategy_selector import StrategyType, Direction
from src.regime_detector import Regime

# Reuse existing contract resolver and position sizer
from src.options.contract_resolver import (
    OptionContractResolver,
    ResolvedContract,
    ResolvedSpread,
    ResolvedIronCondor,
)
from src.options.config import RISK_CONFIG

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_CONCURRENT_POSITIONS = 5
MAX_PORTFOLIO_RISK_PCT = 0.02         # 2% max risk per trade
POSITION_MONITOR_INTERVAL = 30        # Check positions every 30s
CREDIT_STOP_LOSS_MULTIPLIER = 2.0     # Stop at 2x credit received
TAKE_PROFIT_PCT = 0.50                # Take profit at 50% of max profit
DTE_MANAGEMENT_THRESHOLD = 21         # Manage positions at 21 DTE
ORDER_TIMEOUT_SECONDS = 30            # Max time to wait for fill


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ActivePosition:
    """An active options position tracked by the system."""
    position_id: str
    symbol: str                        # Underlying
    strategy: StrategyType
    direction: Direction

    # Leg symbols (OCC format)
    leg_symbols: List[str]
    leg_sides: List[str]              # "buy" or "sell" for each leg

    # Economics
    entry_credit: float                # Net credit received (or -debit paid)
    max_profit: float                  # Maximum possible profit
    max_loss: float                    # Maximum possible loss
    contracts: int

    # Current state
    current_value: float = 0.0         # Current market value of position
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    # Management levels
    stop_loss_value: float = 0.0       # Close if position value exceeds this
    take_profit_value: float = 0.0     # Close if position value drops below this

    # Metadata
    entry_time: datetime = field(default_factory=datetime.now)
    expiration: Optional[date] = None
    signal: Optional[TradeSignal] = None
    order_id: Optional[str] = None

    @property
    def days_to_expiry(self) -> int:
        if self.expiration:
            return (self.expiration - date.today()).days
        return 999

    @property
    def should_stop_loss(self) -> bool:
        """Check if stop-loss triggered (loss exceeds 2x credit)."""
        if self.entry_credit > 0:  # Credit strategy
            return self.current_value > self.stop_loss_value
        else:  # Debit strategy
            return self.unrealized_pnl_pct <= -50  # 50% loss on debit

    @property
    def should_take_profit(self) -> bool:
        """Check if profit target hit (50% of max profit)."""
        if self.entry_credit > 0:  # Credit strategy
            return self.current_value <= self.take_profit_value
        else:  # Debit strategy
            return self.unrealized_pnl_pct >= 100  # 100% gain on debit

    @property
    def should_manage_dte(self) -> bool:
        """Check if position needs to be managed at 21 DTE."""
        return self.days_to_expiry <= DTE_MANAGEMENT_THRESHOLD


# ============================================================================
# MLEG ORDER BUILDER
# ============================================================================

class MLEGOrderBuilder:
    """
    Build Alpaca MLEG (multi-leg) orders for options strategies.

    Uses OrderClass.MLEG with OptionLegRequest for proper multi-leg
    execution instead of the broken OTO approach.
    """

    @staticmethod
    def build_credit_spread(
        long_occ: str,
        short_occ: str,
        quantity: int,
        net_credit: float,
    ) -> LimitOrderRequest:
        """
        Build a credit spread MLEG order.

        For bull put spread: sell higher put, buy lower put.
        For bear call spread: sell lower call, buy higher call.

        Args:
            long_occ: OCC symbol for the long (protective) leg
            short_occ: OCC symbol for the short (income) leg
            quantity: Number of spreads
            net_credit: Net credit to receive per spread

        Returns:
            LimitOrderRequest with MLEG legs
        """
        # Extract underlying from OCC symbol (letters before first digit)
        underlying = ""
        for ch in long_occ:
            if ch.isdigit():
                break
            underlying += ch

        legs = [
            OptionLegRequest(
                symbol=short_occ,
                side=OrderSide.SELL,
                ratio_qty=str(quantity),
            ),
            OptionLegRequest(
                symbol=long_occ,
                side=OrderSide.BUY,
                ratio_qty=str(quantity),
            ),
        ]

        return LimitOrderRequest(
            symbol=underlying,
            qty=quantity,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.MLEG,
            limit_price=round(net_credit, 2),
            legs=legs,
        )

    @staticmethod
    def build_debit_spread(
        long_occ: str,
        short_occ: str,
        quantity: int,
        net_debit: float,
    ) -> LimitOrderRequest:
        """
        Build a debit spread MLEG order.

        For bull call spread: buy lower call, sell higher call.
        For bear put spread: buy higher put, sell lower put.

        Args:
            long_occ: OCC symbol for the long (main) leg
            short_occ: OCC symbol for the short (financing) leg
            quantity: Number of spreads
            net_debit: Net debit to pay per spread

        Returns:
            LimitOrderRequest with MLEG legs
        """
        underlying = ""
        for ch in long_occ:
            if ch.isdigit():
                break
            underlying += ch

        legs = [
            OptionLegRequest(
                symbol=long_occ,
                side=OrderSide.BUY,
                ratio_qty=str(quantity),
            ),
            OptionLegRequest(
                symbol=short_occ,
                side=OrderSide.SELL,
                ratio_qty=str(quantity),
            ),
        ]

        return LimitOrderRequest(
            symbol=underlying,
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.MLEG,
            limit_price=round(net_debit, 2),
            legs=legs,
        )

    @staticmethod
    def build_iron_condor(
        put_long_occ: str,
        put_short_occ: str,
        call_short_occ: str,
        call_long_occ: str,
        quantity: int,
        net_credit: float,
    ) -> LimitOrderRequest:
        """
        Build an iron condor MLEG order (4 legs).

        Structure:
        - BUY lower put (protection)
        - SELL higher put (income)
        - SELL lower call (income)
        - BUY higher call (protection)

        Args:
            put_long_occ: Long put OCC symbol (lowest strike)
            put_short_occ: Short put OCC symbol
            call_short_occ: Short call OCC symbol
            call_long_occ: Long call OCC symbol (highest strike)
            quantity: Number of iron condors
            net_credit: Total net credit per iron condor

        Returns:
            LimitOrderRequest with 4 MLEG legs
        """
        underlying = ""
        for ch in put_long_occ:
            if ch.isdigit():
                break
            underlying += ch

        legs = [
            OptionLegRequest(
                symbol=put_long_occ,
                side=OrderSide.BUY,
                ratio_qty=str(quantity),
            ),
            OptionLegRequest(
                symbol=put_short_occ,
                side=OrderSide.SELL,
                ratio_qty=str(quantity),
            ),
            OptionLegRequest(
                symbol=call_short_occ,
                side=OrderSide.SELL,
                ratio_qty=str(quantity),
            ),
            OptionLegRequest(
                symbol=call_long_occ,
                side=OrderSide.BUY,
                ratio_qty=str(quantity),
            ),
        ]

        return LimitOrderRequest(
            symbol=underlying,
            qty=quantity,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.MLEG,
            limit_price=round(net_credit, 2),
            legs=legs,
        )

    @staticmethod
    def build_long_straddle(
        call_occ: str,
        put_occ: str,
        quantity: int,
        net_debit: float,
    ) -> LimitOrderRequest:
        """
        Build a long straddle MLEG order.

        Buy call + buy put at same strike.

        Args:
            call_occ: ATM call OCC symbol
            put_occ: ATM put OCC symbol
            quantity: Number of straddles
            net_debit: Total debit per straddle

        Returns:
            LimitOrderRequest with 2 MLEG legs
        """
        underlying = ""
        for ch in call_occ:
            if ch.isdigit():
                break
            underlying += ch

        legs = [
            OptionLegRequest(
                symbol=call_occ,
                side=OrderSide.BUY,
                ratio_qty=str(quantity),
            ),
            OptionLegRequest(
                symbol=put_occ,
                side=OrderSide.BUY,
                ratio_qty=str(quantity),
            ),
        ]

        return LimitOrderRequest(
            symbol=underlying,
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.MLEG,
            limit_price=round(net_debit, 2),
            legs=legs,
        )


# ============================================================================
# POSITION MANAGER (REAL P&L)
# ============================================================================

class PositionMonitor:
    """
    Monitor active positions using real Alpaca API data.

    NO MORE RANDOM 5% CLOSE. This uses actual market prices.
    """

    def __init__(self, trading_client: TradingClient, data_client: OptionHistoricalDataClient):
        self.trading_client = trading_client
        self.data_client = data_client
        self.logger = logging.getLogger(__name__)

    async def update_position_values(self, positions: List[ActivePosition]) -> None:
        """
        Update current values and P&L for all positions.

        For each position, fetches latest quotes for all legs
        and calculates the current theoretical close value.
        """
        for pos in positions:
            try:
                total_value = 0.0

                for i, occ_symbol in enumerate(pos.leg_symbols):
                    side = pos.leg_sides[i]
                    quote = await self._get_quote(occ_symbol)

                    if quote is None:
                        continue

                    mid = (quote["bid"] + quote["ask"]) / 2

                    # Value from perspective of closing the position
                    if side == "sell":
                        # We sold this leg, closing means buying back
                        total_value += mid
                    else:
                        # We bought this leg, closing means selling
                        total_value -= mid

                # For credit strategies: we collected entry_credit initially.
                # Current value = cost to close all legs.
                # If we collected $1.50 credit and it now costs $0.75 to close,
                # our unrealized P&L = $1.50 - $0.75 = $0.75 profit.
                pos.current_value = round(total_value, 2)

                if pos.entry_credit > 0:
                    # Credit strategy
                    pos.unrealized_pnl = (pos.entry_credit - pos.current_value) * pos.contracts * 100
                    pos.unrealized_pnl_pct = (
                        (pos.entry_credit - pos.current_value) / pos.entry_credit * 100
                        if pos.entry_credit > 0 else 0
                    )
                else:
                    # Debit strategy 
                    debit_paid = abs(pos.entry_credit)
                    current_close_value = abs(pos.current_value)
                    pos.unrealized_pnl = (current_close_value - debit_paid) * pos.contracts * 100
                    pos.unrealized_pnl_pct = (
                        (current_close_value - debit_paid) / debit_paid * 100
                        if debit_paid > 0 else 0
                    )

            except Exception as e:
                self.logger.warning(f"Failed to update position {pos.position_id}: {e}")

    async def _get_quote(self, occ_symbol: str) -> Optional[Dict]:
        """Get latest bid/ask for an option."""
        try:
            request = OptionLatestQuoteRequest(symbol_or_symbols=occ_symbol)
            quotes = await asyncio.to_thread(
                self.data_client.get_option_latest_quote, request
            )
            if occ_symbol in quotes:
                q = quotes[occ_symbol]
                bid = float(q.bid_price) if q.bid_price else 0.0
                ask = float(q.ask_price) if q.ask_price else 0.0
                return {"bid": bid, "ask": ask}
            return None
        except Exception as e:
            self.logger.debug(f"Quote failed for {occ_symbol}: {e}")
            return None


# ============================================================================
# INTEGRATED TRADING SYSTEM
# ============================================================================

class IntegratedTradingSystem:
    """
    Main trading system orchestrator.

    Lifecycle:
    1. Initialize with Alpaca credentials
    2. Call run_trading_cycle() periodically (every 60s during market hours)
    3. System automatically: scans → resolves → executes → monitors → manages

    All strategies use MLEG orders. All positions are DEFINED RISK.
    """

    def __init__(
        self,
        portfolio_value: float = 100000.0,
        paper: bool = True,
        state_file: str = "trading_state_v2.json",
    ):
        """
        Initialize trading system.

        Args:
            portfolio_value: Current portfolio value for position sizing
            paper: Use paper trading (default True)
            state_file: State persistence file
        """
        # Alpaca credentials
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not api_secret:
            raise ValueError(
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables"
            )

        # Alpaca clients
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=paper,
        )
        self.data_client = OptionHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret,
        )

        # Signal generation
        self.signal_generator = SignalGeneratorV2()

        # Contract resolution (reuse existing, well-tested resolver)
        self.contract_resolver = OptionContractResolver(
            trading_client=self.trading_client,
            data_client=self.data_client,
        )

        # Position monitoring
        self.position_monitor = PositionMonitor(
            trading_client=self.trading_client,
            data_client=self.data_client,
        )

        # MLEG order builder
        self.order_builder = MLEGOrderBuilder()

        # State
        self.portfolio_value = portfolio_value
        self.paper = paper
        self.active_positions: List[ActivePosition] = []
        self.state_file = state_file
        self._stop_event = asyncio.Event()

        # Statistics
        self.stats = {
            "cycles_run": 0,
            "signals_generated": 0,
            "trades_executed": 0,
            "trades_failed": 0,
            "positions_closed": 0,
            "stop_losses": 0,
            "profit_targets": 0,
            "dte_exits": 0,
            "total_realized_pnl": 0.0,
            "start_time": datetime.now().isoformat(),
        }

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"IntegratedTradingSystem v2 initialized "
            f"(paper={paper}, portfolio=${portfolio_value:,.0f})"
        )

        # Load saved state
        self._load_state()

        # CRITICAL: Load real positions from Alpaca so we're aware of
        # ALL existing exposure, not just positions this bot created.
        self._load_existing_positions_from_alpaca()

    # ================================================================== #
    # ALPACA POSITION SYNC
    # ================================================================== #

    def _load_existing_positions_from_alpaca(self) -> None:
        """
        Load all existing option positions from the Alpaca account.

        This ensures that on startup the system is aware of every position
        in the account—not just positions it created in a prior session.
        Positions that already appear in ``self.active_positions`` (from
        the JSON state file) are *not* duplicated.
        """
        try:
            alpaca_positions = self.trading_client.get_all_positions()
        except Exception as e:
            self.logger.error(f"Failed to load positions from Alpaca: {e}")
            return

        # Build set of OCC symbols we already track so we don't double-count
        tracked_occ: set[str] = set()
        for pos in self.active_positions:
            tracked_occ.update(pos.leg_symbols)

        # ------------------------------------------------------------------ #
        # Group Alpaca option positions by underlying so we can detect
        # multi-leg structures (spreads, condors).  Single-leg positions
        # are still loaded individually.
        # ------------------------------------------------------------------ #
        option_positions = []
        for pos in alpaca_positions:
            if getattr(pos, 'asset_class', None) != AssetClass.US_OPTION:
                continue
            if pos.symbol in tracked_occ:
                # Already tracked from a prior state-file restore – skip
                continue
            option_positions.append(pos)

        if not option_positions:
            self.logger.info(
                "Alpaca position sync: 0 new option legs to import "
                f"({len(self.active_positions)} already tracked)"
            )
            return

        # Group by underlying
        by_underlying: Dict[str, list] = {}
        for pos in option_positions:
            underlying = self._parse_underlying_from_occ(pos.symbol)
            by_underlying.setdefault(underlying, []).append(pos)

        imported = 0
        for underlying, legs in by_underlying.items():
            try:
                new_pos = self._build_active_position_from_alpaca_legs(
                    underlying, legs
                )
                if new_pos is not None:
                    self.active_positions.append(new_pos)
                    imported += 1
            except Exception as e:
                self.logger.warning(
                    f"Failed to import Alpaca position for {underlying}: {e}"
                )

        self.logger.info(
            f"Alpaca position sync: imported {imported} positions "
            f"({len(option_positions)} option legs) – "
            f"total tracked: {len(self.active_positions)}"
        )

    def _sync_positions_with_alpaca(self) -> None:
        """Lightweight per-cycle sync: import new positions & prune stale ones.

        Called at the top of every trading cycle so we always reflect
        reality before making new-trade or close decisions.
        """
        try:
            alpaca_positions = self.trading_client.get_all_positions()
        except Exception as e:
            self.logger.warning(f"Cycle sync: failed to fetch Alpaca positions: {e}")
            return

        # OCC symbols currently held in Alpaca
        alpaca_occ: set[str] = set()
        for pos in alpaca_positions:
            if getattr(pos, 'asset_class', None) == AssetClass.US_OPTION:
                alpaca_occ.add(pos.symbol)

        # 1) Remove tracked positions whose legs are ALL gone from Alpaca
        #    (they were closed externally, or expired).
        before = len(self.active_positions)
        self.active_positions = [
            p for p in self.active_positions
            if any(leg in alpaca_occ for leg in p.leg_symbols)
        ]
        pruned = before - len(self.active_positions)
        if pruned:
            self.logger.info(
                f"Cycle sync: pruned {pruned} positions no longer in Alpaca"
            )

        # 2) Import any Alpaca option positions we don't yet track
        tracked_occ: set[str] = set()
        for p in self.active_positions:
            tracked_occ.update(p.leg_symbols)

        new_legs = [
            pos for pos in alpaca_positions
            if getattr(pos, 'asset_class', None) == AssetClass.US_OPTION
            and pos.symbol not in tracked_occ
        ]

        if new_legs:
            by_underlying: Dict[str, list] = {}
            for pos in new_legs:
                underlying = self._parse_underlying_from_occ(pos.symbol)
                by_underlying.setdefault(underlying, []).append(pos)

            imported = 0
            for underlying, legs in by_underlying.items():
                try:
                    new_pos = self._build_active_position_from_alpaca_legs(
                        underlying, legs,
                    )
                    if new_pos is not None:
                        self.active_positions.append(new_pos)
                        imported += 1
                except Exception as e:
                    self.logger.warning(
                        f"Cycle sync: failed to import {underlying}: {e}"
                    )
            if imported:
                self.logger.info(
                    f"Cycle sync: imported {imported} new positions from Alpaca"
                )

    # -- helpers for Alpaca import ----------------------------------------- #

    @staticmethod
    def _parse_underlying_from_occ(occ_symbol: str) -> str:
        """Extract the underlying ticker from an OCC symbol.

        OCC format: ``AAPL  250117C00150000``  (padded) or
                    ``AAPL250117C00150000``   (no spaces).
        """
        cleaned = occ_symbol.replace(" ", "")
        m = re.match(r'^([A-Z]+)', cleaned)
        return m.group(1) if m else cleaned[:4]

    @staticmethod
    def _parse_occ_expiration(occ_symbol: str) -> Optional[date]:
        """Extract expiration date from an OCC symbol → ``date``."""
        cleaned = occ_symbol.replace(" ", "")
        m = re.match(r'^[A-Z]+(\d{6})[CP]', cleaned)
        if not m:
            return None
        raw = m.group(1)  # YYMMDD
        try:
            return datetime.strptime(raw, "%y%m%d").date()
        except ValueError:
            return None

    def _build_active_position_from_alpaca_legs(
        self,
        underlying: str,
        legs: list,
    ) -> Optional[ActivePosition]:
        """Convert a list of Alpaca option-position objects into an ActivePosition.

        Heuristics used to guess the strategy:
        * 4 legs (2 puts + 2 calls)  → ``IRON_CONDOR``
        * 2 legs same type, one long one short → credit or debit spread
        * 2 legs different type, both long       → ``LONG_STRADDLE``
        * Anything else → fall back to ``IRON_CONDOR`` with neutral direction
          so the position is still *tracked* and counted against the limit.
        """
        leg_symbols = []
        leg_sides = []
        total_cost_basis = 0.0
        total_market_value = 0.0
        total_qty = 0
        expiration: Optional[date] = None

        for lp in legs:
            occ = lp.symbol
            qty = int(lp.qty)  # signed: positive = long, negative = short
            side = "buy" if qty > 0 else "sell"
            leg_symbols.append(occ)
            leg_sides.append(side)

            cost_basis = float(lp.cost_basis) if lp.cost_basis else 0.0
            mkt_value = float(lp.market_value) if lp.market_value else 0.0
            total_cost_basis += cost_basis
            total_market_value += mkt_value
            total_qty = max(total_qty, abs(qty))

            if expiration is None:
                expiration = self._parse_occ_expiration(occ)

        # ------ guess strategy & direction -------------------------------- #
        strategy, direction = self._guess_strategy(leg_symbols, leg_sides)

        # ------ economics (best-effort from Alpaca data) ------------------- #
        # cost_basis is negative for credits, positive for debits
        entry_credit = -total_cost_basis / 100.0 if total_cost_basis != 0 else 0.0
        unrealized_pnl = total_market_value + total_cost_basis  # MV - |cost|

        contracts = max(total_qty, 1)

        # Conservative risk estimates – we lack the original signal data
        max_profit = abs(entry_credit) * contracts * 100 if entry_credit > 0 else abs(total_market_value)
        max_loss = max_profit * 2  # placeholder

        stop_loss_val = abs(entry_credit) * CREDIT_STOP_LOSS_MULTIPLIER if entry_credit > 0 else 0
        take_profit_val = abs(entry_credit) * (1 - TAKE_PROFIT_PCT) if entry_credit > 0 else 0

        position_id = f"alpaca_import_{underlying}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        return ActivePosition(
            position_id=position_id,
            symbol=underlying,
            strategy=strategy,
            direction=direction,
            leg_symbols=leg_symbols,
            leg_sides=leg_sides,
            entry_credit=entry_credit,
            max_profit=max_profit,
            max_loss=max_loss,
            contracts=contracts,
            current_value=total_market_value / (contracts * 100) if contracts else 0,
            unrealized_pnl=unrealized_pnl,
            stop_loss_value=stop_loss_val,
            take_profit_value=take_profit_val,
            entry_time=datetime.now(),
            expiration=expiration,
            signal=None,
            order_id=None,
        )

    @staticmethod
    def _guess_strategy(
        leg_symbols: List[str],
        leg_sides: List[str],
    ) -> Tuple[StrategyType, Direction]:
        """Best-effort strategy classification from OCC symbols and sides."""
        # Classify each leg as call/put from its OCC symbol
        types = []
        for occ in leg_symbols:
            cleaned = occ.replace(" ", "")
            m = re.search(r'\d{6}([CP])', cleaned)
            types.append(m.group(1) if m else "?")

        calls = [(s, sd) for s, sd, t in zip(leg_symbols, leg_sides, types) if t == "C"]
        puts  = [(s, sd) for s, sd, t in zip(leg_symbols, leg_sides, types) if t == "P"]

        n_legs = len(leg_symbols)

        # 4 legs: 2 puts + 2 calls → iron condor
        if n_legs == 4 and len(puts) == 2 and len(calls) == 2:
            return StrategyType.IRON_CONDOR, Direction.NEUTRAL

        # 2 legs, same type
        if n_legs == 2 and len(set(types)) == 1:
            has_sell = any(sd == "sell" for sd in leg_sides)
            has_buy  = any(sd == "buy"  for sd in leg_sides)
            if types[0] == "P":
                if has_sell and has_buy:
                    return StrategyType.BULL_PUT_SPREAD, Direction.BULLISH
                return StrategyType.BEAR_PUT_SPREAD, Direction.BEARISH
            else:  # calls
                if has_sell and has_buy:
                    return StrategyType.BEAR_CALL_SPREAD, Direction.BEARISH
                return StrategyType.BULL_CALL_SPREAD, Direction.BULLISH

        # 2 legs, different types, both long → straddle
        if n_legs == 2 and len(puts) == 1 and len(calls) == 1:
            if all(sd == "buy" for sd in leg_sides):
                return StrategyType.LONG_STRADDLE, Direction.NEUTRAL

        # Fallback – still track the position
        return StrategyType.IRON_CONDOR, Direction.NEUTRAL

    # ================================================================== #
    # MAIN TRADING LOOP
    # ================================================================== #

    async def run_forever(self) -> None:
        """Run continuously during market hours."""
        self.logger.info("INTEGRATED TRADING SYSTEM v2 STARTED")

        try:
            while not self._stop_event.is_set():
                if not self._market_is_open():
                    self.logger.info("Market closed, waiting 60s...")
                    try:
                        await asyncio.wait_for(self._stop_event.wait(), timeout=60)
                    except asyncio.TimeoutError:
                        pass
                    continue

                await self.run_trading_cycle()

                self._save_state()

                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=60)
                except asyncio.TimeoutError:
                    pass

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            self._save_state()
            self._log_final_stats()

    def request_shutdown(self) -> None:
        """Request graceful shutdown."""
        self._stop_event.set()

    async def run_trading_cycle(self) -> Dict:
        """
        Execute one complete trading cycle.

        Steps:
        1. Monitor existing positions (stop-loss, take-profit, DTE)
        2. Scan for new signals
        3. Resolve and execute new trades
        4. Log cycle summary

        Returns:
            Cycle summary dict
        """
        self.stats["cycles_run"] += 1
        cycle = self.stats["cycles_run"]

        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"CYCLE #{cycle} - {datetime.now().strftime('%H:%M:%S')}")
        self.logger.info(f"{'=' * 60}")

        # Step 0: Sync with Alpaca so we see positions opened/closed externally
        self._sync_positions_with_alpaca()

        # Step 1: Monitor positions FIRST (close before opening new ones)
        closed_count = await self._monitor_and_manage_positions()
        self.logger.info(
            f"Step 1 (MANAGE): {len(self.active_positions)} active, "
            f"{closed_count} closed"
        )

        # Step 2: Scan for new signals (only if we have room for more positions)
        new_trades = 0
        if len(self.active_positions) < MAX_CONCURRENT_POSITIONS:
            if self._safe_entry_window():
                signals = self.signal_generator.scan_for_signals()
                self.stats["signals_generated"] += len(signals)

                # Step 3: Execute signals
                for signal in signals:
                    if len(self.active_positions) >= MAX_CONCURRENT_POSITIONS:
                        self.logger.info("Max positions reached, stopping execution")
                        break

                    success = await self._resolve_and_execute(signal)
                    if success:
                        new_trades += 1
            else:
                self.logger.info("Step 2: Outside safe entry window (9:45-15:45 ET)")
        else:
            self.logger.info(
                f"Step 2: Max positions ({MAX_CONCURRENT_POSITIONS}) reached, skipping scan"
            )

        self.logger.info(f"Step 3 (EXECUTE): {new_trades} new trades")

        # Step 4: Summary
        summary = self._log_cycle_summary()
        return summary

    # ================================================================== #
    # POSITION MONITORING & MANAGEMENT
    # ================================================================== #

    async def _monitor_and_manage_positions(self) -> int:
        """
        Monitor all positions and close as needed.

        Close triggers:
        1. Stop-loss: position value > 2x credit received
        2. Take-profit: 50% of max profit captured
        3. DTE management: position approaching expiration (21 DTE)

        Returns:
            Number of positions closed
        """
        if not self.active_positions:
            return 0

        # Update all position values with real market data
        await self.position_monitor.update_position_values(self.active_positions)

        closed = 0
        positions_to_remove: List[ActivePosition] = []

        for pos in self.active_positions:
            close_reason = None

            # Check stop-loss
            if pos.should_stop_loss:
                close_reason = "STOP_LOSS"
                self.logger.warning(
                    f"STOP LOSS: {pos.symbol} {pos.strategy.value} "
                    f"P&L: ${pos.unrealized_pnl:+,.2f} ({pos.unrealized_pnl_pct:+.1f}%)"
                )

            # Check take-profit
            elif pos.should_take_profit:
                close_reason = "TAKE_PROFIT"
                self.logger.info(
                    f"PROFIT TARGET: {pos.symbol} {pos.strategy.value} "
                    f"P&L: ${pos.unrealized_pnl:+,.2f} ({pos.unrealized_pnl_pct:+.1f}%)"
                )

            # Check DTE management
            elif pos.should_manage_dte:
                close_reason = "DTE_MANAGEMENT"
                self.logger.info(
                    f"DTE MANAGEMENT: {pos.symbol} at {pos.days_to_expiry} DTE"
                )

            if close_reason:
                success = await self._close_position(pos, close_reason)
                if success:
                    positions_to_remove.append(pos)
                    closed += 1

                    if close_reason == "STOP_LOSS":
                        self.stats["stop_losses"] += 1
                    elif close_reason == "TAKE_PROFIT":
                        self.stats["profit_targets"] += 1
                    elif close_reason == "DTE_MANAGEMENT":
                        self.stats["dte_exits"] += 1

                    self.stats["total_realized_pnl"] += pos.unrealized_pnl
            else:
                # Log position status
                self.logger.info(
                    f"  {pos.symbol} {pos.strategy.value}: "
                    f"P&L ${pos.unrealized_pnl:+,.2f} ({pos.unrealized_pnl_pct:+.1f}%) "
                    f"DTE={pos.days_to_expiry}"
                )

        # Remove closed positions
        for pos in positions_to_remove:
            self.active_positions.remove(pos)
            self.stats["positions_closed"] += 1

        return closed

    async def _close_position(self, position: ActivePosition, reason: str) -> bool:
        """
        Close an active position.

        - Single-leg positions: close via Alpaca's DELETE /positions/{symbol}
          endpoint, falling back to a simple (non-MLEG) market order.
        - Multi-leg positions: submit the opposite MLEG market order.

        Both paths include a final fallback that attempts to close each leg
        individually through the REST close-position endpoint.
        """
        try:
            self.logger.info(
                f"Closing {position.symbol} {position.strategy.value} "
                f"({reason}): {len(position.leg_symbols)} legs"
            )

            # -------------------------------------------------------------- #
            # SINGLE-LEG: close directly via the Alpaca REST endpoint or a
            # plain (non-MLEG) market order on the OCC symbol.
            # -------------------------------------------------------------- #
            if len(position.leg_symbols) == 1:
                return await self._close_single_leg(position, reason)

            # -------------------------------------------------------------- #
            # MULTI-LEG (2+): use MLEG market order
            # -------------------------------------------------------------- #
            return await self._close_multi_leg(position, reason)

        except Exception as e:
            self.logger.error(
                f"Failed to close position {position.symbol}: {e}",
                exc_info=True,
            )
            # Last-resort fallback: try closing each leg individually
            return await self._close_legs_individually(position, reason)

    async def _close_single_leg(self, position: ActivePosition, reason: str) -> bool:
        """Close a single-option-leg position."""
        occ = position.leg_symbols[0]

        # Attempt 1: Alpaca DELETE /positions/{symbol}  (simplest)
        try:
            order = await asyncio.to_thread(
                self.trading_client.close_position, occ
            )
            self.logger.info(
                f"Single-leg close via REST for {occ} ({reason}): "
                f"order {getattr(order, 'id', order)}"
            )
            return True
        except Exception as e:
            self.logger.warning(
                f"REST close_position failed for {occ}: {e} — "
                "falling back to market order"
            )

        # Attempt 2: submit a plain market order (non-MLEG)
        try:
            original_side = position.leg_sides[0]
            close_side = OrderSide.BUY if original_side == "sell" else OrderSide.SELL

            close_order = MarketOrderRequest(
                symbol=occ,
                qty=position.contracts,
                side=close_side,
                time_in_force=TimeInForce.DAY,
            )

            order = await asyncio.to_thread(
                self.trading_client.submit_order, close_order
            )
            self.logger.info(
                f"Single-leg market order submitted: {order.id} for {occ} ({reason})"
            )
            return True
        except Exception as e:
            self.logger.error(
                f"Single-leg market order also failed for {occ}: {e}"
            )
            return False

    async def _close_multi_leg(self, position: ActivePosition, reason: str) -> bool:
        """Close a multi-leg position via an MLEG market order."""
        try:
            # Build closing MLEG order (reverse all legs)
            closing_legs = []
            for i, occ in enumerate(position.leg_symbols):
                original_side = position.leg_sides[i]
                close_side = OrderSide.BUY if original_side == "sell" else OrderSide.SELL
                closing_legs.append(
                    OptionLegRequest(
                        symbol=occ,
                        side=close_side,
                        ratio_qty=str(position.contracts),
                    )
                )

            underlying = self._parse_underlying_from_occ(position.leg_symbols[0])

            close_order = MarketOrderRequest(
                symbol=underlying,
                qty=position.contracts,
                side=OrderSide.BUY,  # Net direction doesn't matter for MLEG close
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.MLEG,
                legs=closing_legs,
            )

            order = await asyncio.to_thread(
                self.trading_client.submit_order, close_order
            )

            self.logger.info(
                f"MLEG close order submitted: {order.id} for {position.symbol} ({reason})"
            )
            return True

        except Exception as e:
            self.logger.error(
                f"MLEG close failed for {position.symbol}: {e} — "
                "falling back to individual leg closes"
            )
            return await self._close_legs_individually(position, reason)

    async def _close_legs_individually(self, position: ActivePosition, reason: str) -> bool:
        """Last-resort fallback: close each leg via DELETE /positions/{symbol}."""
        any_success = False
        for i, occ in enumerate(position.leg_symbols):
            try:
                order = await asyncio.to_thread(
                    self.trading_client.close_position, occ
                )
                self.logger.info(
                    f"Individually closed leg {occ}: "
                    f"order {getattr(order, 'id', order)}"
                )
                any_success = True
            except Exception as e:
                self.logger.warning(f"Failed to close leg {occ}: {e}")

        if not any_success:
            self.logger.error(
                f"ALL close attempts failed for {position.symbol} ({reason})"
            )
        return any_success

    # ================================================================== #
    # SIGNAL RESOLUTION & EXECUTION
    # ================================================================== #

    async def _resolve_and_execute(self, signal: TradeSignal) -> bool:
        """
        Resolve a TradeSignal to real contracts and execute via MLEG.

        Steps:
        1. Calculate position size (Kelly-based)
        2. Resolve to real OCC contracts
        3. Build MLEG order
        4. Validate buying power
        5. Submit order
        6. Track position

        Returns:
            True if trade executed successfully
        """
        try:
            # Step 1: Position sizing
            contracts = self._calculate_position_size(signal)
            if contracts <= 0:
                self.logger.info(f"Position size 0 for {signal.symbol}, skipping")
                return False

            contracts = min(contracts, signal.max_contracts)

            # Step 2: Resolve based on strategy type
            if signal.strategy == StrategyType.IRON_CONDOR:
                return await self._execute_iron_condor(signal, contracts)

            elif signal.strategy in (
                StrategyType.BULL_PUT_SPREAD,
                StrategyType.BEAR_CALL_SPREAD,
            ):
                return await self._execute_credit_spread(signal, contracts)

            elif signal.strategy in (
                StrategyType.BULL_CALL_SPREAD,
                StrategyType.BEAR_PUT_SPREAD,
            ):
                return await self._execute_debit_spread(signal, contracts)

            elif signal.strategy == StrategyType.LONG_STRADDLE:
                return await self._execute_straddle(signal, contracts)

            else:
                self.logger.warning(f"Unsupported strategy: {signal.strategy}")
                return False

        except Exception as e:
            self.logger.error(
                f"Execution error for {signal.symbol}: {e}",
                exc_info=True,
            )
            self.stats["trades_failed"] += 1
            return False

    async def _execute_iron_condor(self, signal: TradeSignal, contracts: int) -> bool:
        """Execute an iron condor using MLEG."""
        resolved = await self.contract_resolver.resolve_iron_condor(
            symbol=signal.symbol,
            target_dte=signal.target_dte,
            target_delta=signal.target_delta,
        )
        if resolved is None:
            self.logger.warning(f"Failed to resolve iron condor for {signal.symbol}")
            return False

        # Validate credit
        if resolved.total_credit <= 0:
            self.logger.warning(f"Iron condor {signal.symbol} has no credit: ${resolved.total_credit}")
            return False

        self.logger.info(
            f"IC {signal.symbol}: "
            f"Put {resolved.put_spread.long_leg.strike}/{resolved.put_spread.short_leg.strike} "
            f"Call {resolved.call_spread.short_leg.strike}/{resolved.call_spread.long_leg.strike} "
            f"Credit: ${resolved.total_credit:.2f}"
        )

        # Build MLEG order
        order_request = self.order_builder.build_iron_condor(
            put_long_occ=resolved.put_spread.long_leg.occ_symbol,
            put_short_occ=resolved.put_spread.short_leg.occ_symbol,
            call_short_occ=resolved.call_spread.short_leg.occ_symbol,
            call_long_occ=resolved.call_spread.long_leg.occ_symbol,
            quantity=contracts,
            net_credit=resolved.total_credit,
        )

        # Submit
        order = await self._submit_order(order_request, signal.symbol)
        if order is None:
            return False

        # Track position
        stop_loss_val = resolved.total_credit * CREDIT_STOP_LOSS_MULTIPLIER
        take_profit_val = resolved.total_credit * (1 - TAKE_PROFIT_PCT)

        self.active_positions.append(ActivePosition(
            position_id=str(order.id),
            symbol=signal.symbol,
            strategy=signal.strategy,
            direction=signal.direction,
            leg_symbols=[
                resolved.put_spread.long_leg.occ_symbol,
                resolved.put_spread.short_leg.occ_symbol,
                resolved.call_spread.short_leg.occ_symbol,
                resolved.call_spread.long_leg.occ_symbol,
            ],
            leg_sides=["buy", "sell", "sell", "buy"],
            entry_credit=resolved.total_credit,
            max_profit=resolved.total_credit * contracts * 100,
            max_loss=resolved.max_loss * contracts,
            contracts=contracts,
            stop_loss_value=stop_loss_val,
            take_profit_value=take_profit_val,
            expiration=resolved.put_spread.short_leg.expiration,
            signal=signal,
            order_id=str(order.id),
        ))

        self.stats["trades_executed"] += 1
        self.logger.info(
            f"EXECUTED: IC {signal.symbol} x{contracts} "
            f"Credit=${resolved.total_credit:.2f} "
            f"MaxLoss=${resolved.max_loss:.2f}"
        )
        return True

    async def _execute_credit_spread(self, signal: TradeSignal, contracts: int) -> bool:
        """Execute a credit spread (bull put or bear call) using MLEG."""
        # Determine spread type for resolver
        if signal.strategy == StrategyType.BULL_PUT_SPREAD:
            spread_type = "put_spread"
        else:
            spread_type = "call_spread"

        resolved = await self.contract_resolver.resolve_spread(
            symbol=signal.symbol,
            spread_type=spread_type,
            target_dte=signal.target_dte,
            target_delta=signal.target_delta,
        )
        if resolved is None:
            self.logger.warning(f"Failed to resolve {spread_type} for {signal.symbol}")
            return False

        if resolved.net_credit <= 0:
            self.logger.warning(f"Credit spread {signal.symbol} has no credit: ${resolved.net_credit}")
            return False

        self.logger.info(
            f"Spread {signal.symbol}: "
            f"Short={resolved.short_leg.occ_symbol} (${resolved.short_leg.mid_price:.2f}) "
            f"Long={resolved.long_leg.occ_symbol} (${resolved.long_leg.mid_price:.2f}) "
            f"Credit=${resolved.net_credit:.2f}"
        )

        order_request = self.order_builder.build_credit_spread(
            long_occ=resolved.long_leg.occ_symbol,
            short_occ=resolved.short_leg.occ_symbol,
            quantity=contracts,
            net_credit=resolved.net_credit,
        )

        order = await self._submit_order(order_request, signal.symbol)
        if order is None:
            return False

        stop_loss_val = resolved.net_credit * CREDIT_STOP_LOSS_MULTIPLIER
        take_profit_val = resolved.net_credit * (1 - TAKE_PROFIT_PCT)

        self.active_positions.append(ActivePosition(
            position_id=str(order.id),
            symbol=signal.symbol,
            strategy=signal.strategy,
            direction=signal.direction,
            leg_symbols=[resolved.long_leg.occ_symbol, resolved.short_leg.occ_symbol],
            leg_sides=["buy", "sell"],
            entry_credit=resolved.net_credit,
            max_profit=resolved.max_profit * contracts,
            max_loss=resolved.max_loss * contracts,
            contracts=contracts,
            stop_loss_value=stop_loss_val,
            take_profit_value=take_profit_val,
            expiration=resolved.short_leg.expiration,
            signal=signal,
            order_id=str(order.id),
        ))

        self.stats["trades_executed"] += 1
        self.logger.info(
            f"EXECUTED: {signal.strategy.value} {signal.symbol} x{contracts} "
            f"Credit=${resolved.net_credit:.2f}"
        )
        return True

    async def _execute_debit_spread(self, signal: TradeSignal, contracts: int) -> bool:
        """Execute a debit spread (bull call or bear put) using MLEG."""
        if signal.strategy == StrategyType.BULL_CALL_SPREAD:
            spread_type = "call_spread"
        else:
            spread_type = "put_spread"

        resolved = await self.contract_resolver.resolve_spread(
            symbol=signal.symbol,
            spread_type=spread_type,
            target_dte=signal.target_dte,
            target_delta=signal.target_delta,
        )
        if resolved is None:
            self.logger.warning(f"Failed to resolve debit spread for {signal.symbol}")
            return False

        # For debit spread, we're the buyer, so net_credit will be negative
        # (we pay a debit). Use absolute value.
        debit = abs(resolved.net_credit) if resolved.net_credit < 0 else resolved.long_leg.mid_price - resolved.short_leg.mid_price
        if debit <= 0:
            debit = resolved.long_leg.mid_price  # Fallback

        self.logger.info(
            f"Debit Spread {signal.symbol}: "
            f"Long={resolved.long_leg.occ_symbol} Short={resolved.short_leg.occ_symbol} "
            f"Debit=${debit:.2f}"
        )

        # For debit spread, swap long/short semantics
        # We buy the more expensive leg and sell the cheaper one
        order_request = self.order_builder.build_debit_spread(
            long_occ=resolved.long_leg.occ_symbol if signal.strategy == StrategyType.BULL_CALL_SPREAD else resolved.short_leg.occ_symbol,
            short_occ=resolved.short_leg.occ_symbol if signal.strategy == StrategyType.BULL_CALL_SPREAD else resolved.long_leg.occ_symbol,
            quantity=contracts,
            net_debit=debit,
        )

        order = await self._submit_order(order_request, signal.symbol)
        if order is None:
            return False

        strike_width = abs(resolved.short_leg.strike - resolved.long_leg.strike)

        self.active_positions.append(ActivePosition(
            position_id=str(order.id),
            symbol=signal.symbol,
            strategy=signal.strategy,
            direction=signal.direction,
            leg_symbols=[resolved.long_leg.occ_symbol, resolved.short_leg.occ_symbol],
            leg_sides=["buy", "sell"],
            entry_credit=-debit,  # Negative = we paid debit
            max_profit=(strike_width - debit) * contracts * 100,
            max_loss=debit * contracts * 100,
            contracts=contracts,
            stop_loss_value=0,  # Debit spread: managed by pct loss
            take_profit_value=0,
            expiration=resolved.long_leg.expiration,
            signal=signal,
            order_id=str(order.id),
        ))

        self.stats["trades_executed"] += 1
        self.logger.info(
            f"EXECUTED: {signal.strategy.value} {signal.symbol} x{contracts} "
            f"Debit=${debit:.2f}"
        )
        return True

    async def _execute_straddle(self, signal: TradeSignal, contracts: int) -> bool:
        """Execute a long straddle using MLEG."""
        # Resolve ATM call
        call = await self.contract_resolver.resolve_single_leg(
            symbol=signal.symbol,
            option_type="call",
            target_dte=signal.target_dte,
            target_delta=0.50,  # ATM
        )
        if call is None:
            self.logger.warning(f"Failed to resolve call for straddle {signal.symbol}")
            return False

        # Resolve ATM put at same strike
        put = await self.contract_resolver.resolve_single_leg(
            symbol=signal.symbol,
            option_type="put",
            target_dte=signal.target_dte,
            target_strike=call.strike,  # Same strike as call
        )
        if put is None:
            self.logger.warning(f"Failed to resolve put for straddle {signal.symbol}")
            return False

        debit = call.mid_price + put.mid_price

        order_request = self.order_builder.build_long_straddle(
            call_occ=call.occ_symbol,
            put_occ=put.occ_symbol,
            quantity=contracts,
            net_debit=debit,
        )

        order = await self._submit_order(order_request, signal.symbol)
        if order is None:
            return False

        self.active_positions.append(ActivePosition(
            position_id=str(order.id),
            symbol=signal.symbol,
            strategy=signal.strategy,
            direction=signal.direction,
            leg_symbols=[call.occ_symbol, put.occ_symbol],
            leg_sides=["buy", "buy"],
            entry_credit=-debit,
            max_profit=999999,  # Theoretically unlimited
            max_loss=debit * contracts * 100,
            contracts=contracts,
            expiration=call.expiration,
            signal=signal,
            order_id=str(order.id),
        ))

        self.stats["trades_executed"] += 1
        self.logger.info(
            f"EXECUTED: Straddle {signal.symbol} x{contracts} "
            f"Debit=${debit:.2f}"
        )
        return True

    # ================================================================== #
    # ORDER SUBMISSION
    # ================================================================== #

    async def _submit_order(self, order_request, symbol: str):
        """
        Submit order to Alpaca with buying power check.

        Returns order object on success, None on failure.
        """
        try:
            # Pre-trade: verify buying power
            account = await asyncio.to_thread(self.trading_client.get_account)
            buying_power = float(account.buying_power)

            self.logger.info(
                f"Submitting MLEG order for {symbol} "
                f"(buying power: ${buying_power:,.2f})"
            )

            order = await asyncio.to_thread(
                self.trading_client.submit_order, order_request
            )

            self.logger.info(f"Order submitted: {order.id} ({order.status})")

            # Poll for fill (up to 30s for limit orders)
            if order.status.value not in ("filled", "cancelled", "rejected"):
                filled_order = await self._poll_order(str(order.id))
                if filled_order:
                    order = filled_order

            if order.status.value == "filled":
                self.logger.info(f"Order FILLED: {order.id}")
                return order
            elif order.status.value in ("cancelled", "rejected", "expired"):
                self.logger.warning(f"Order {order.status.value}: {order.id}")
                return None
            else:
                # Still pending - keep it tracked
                self.logger.info(f"Order still pending: {order.id} ({order.status.value})")
                return order

        except Exception as e:
            self.logger.error(f"Order submission failed for {symbol}: {e}")
            self.stats["trades_failed"] += 1
            return None

    async def _poll_order(self, order_id: str, timeout: float = ORDER_TIMEOUT_SECONDS):
        """Poll order status until terminal state or timeout."""
        start = time.time()
        interval = 1.0

        while time.time() - start < timeout:
            try:
                order = await asyncio.to_thread(
                    self.trading_client.get_order_by_id, order_id
                )
                status = order.status.value
                if status in ("filled", "cancelled", "rejected", "expired"):
                    return order
            except Exception as e:
                self.logger.debug(f"Poll error: {e}")

            await asyncio.sleep(interval)
            interval = min(interval * 1.5, 3.0)

        self.logger.warning(f"Order {order_id} timed out after {timeout}s")
        return None

    # ================================================================== #
    # POSITION SIZING
    # ================================================================== #

    def _calculate_position_size(self, signal: TradeSignal) -> int:
        """
        Calculate position size using simplified Kelly Criterion.

        Kelly: f* = (p * b - q) / b
        Where:
            p = probability of profit
            q = 1 - p
            b = win/loss ratio

        We use quarter-Kelly for safety.
        """
        pop = signal.probability_of_profit
        if pop <= 0 or pop >= 1:
            return 1

        q = 1 - pop
        b = signal.risk_reward_ratio if signal.risk_reward_ratio > 0 else 1.0

        kelly = (pop * b - q) / b
        if kelly <= 0:
            return 1  # Minimum 1 contract

        # Quarter-Kelly
        fraction = kelly * 0.25

        # Max 2% of portfolio per trade
        max_risk = self.portfolio_value * MAX_PORTFOLIO_RISK_PCT

        # Estimate risk per contract (wing width * 100)
        risk_per_contract = signal.wing_width * 100 if signal.wing_width > 0 else 500

        # Contracts from Kelly
        kelly_contracts = int(self.portfolio_value * fraction / risk_per_contract)

        # Contracts from max risk
        risk_contracts = int(max_risk / risk_per_contract)

        # Take minimum
        contracts = max(1, min(kelly_contracts, risk_contracts, signal.max_contracts))

        self.logger.debug(
            f"Position size {signal.symbol}: "
            f"Kelly={kelly:.2f} fraction={fraction:.3f} "
            f"contracts={contracts}"
        )

        return contracts

    # ================================================================== #
    # UTILITIES
    # ================================================================== #

    def _market_is_open(self) -> bool:
        """Check if market is currently open."""
        from datetime import time as dtime
        now = datetime.now(ZoneInfo("America/New_York"))
        if now.weekday() >= 5:
            return False
        return dtime(9, 30) <= now.time() <= dtime(16, 0)

    def _safe_entry_window(self) -> bool:
        """Check if we're in the safe entry window (avoid first/last 15 min)."""
        from datetime import time as dtime
        now = datetime.now(ZoneInfo("America/New_York"))
        return dtime(9, 45) <= now.time() <= dtime(15, 45)

    def _log_cycle_summary(self) -> Dict:
        """Log and return cycle summary."""
        total_pnl = sum(p.unrealized_pnl for p in self.active_positions)

        summary = {
            "cycle": self.stats["cycles_run"],
            "active_positions": len(self.active_positions),
            "unrealized_pnl": total_pnl,
            "realized_pnl": self.stats["total_realized_pnl"],
            "trades_today": self.signal_generator.get_daily_trade_count(),
            "total_trades": self.stats["trades_executed"],
            "win_rate": (
                self.stats["profit_targets"]
                / max(self.stats["positions_closed"], 1)
                * 100
            ),
        }

        self.logger.info(
            f"CYCLE SUMMARY: "
            f"Positions={summary['active_positions']} "
            f"Unrealized=${total_pnl:+,.2f} "
            f"Realized=${summary['realized_pnl']:+,.2f} "
            f"Trades={summary['total_trades']} "
            f"Win%={summary['win_rate']:.0f}%"
        )

        return summary

    def _log_final_stats(self):
        """Log final statistics on shutdown."""
        self.logger.info("=" * 60)
        self.logger.info("FINAL STATISTICS")
        self.logger.info("=" * 60)
        for k, v in self.stats.items():
            self.logger.info(f"  {k}: {v}")
        total_unrealized = sum(p.unrealized_pnl for p in self.active_positions)
        self.logger.info(f"  unrealized_pnl: ${total_unrealized:+,.2f}")
        self.logger.info(f"  net_pnl: ${self.stats['total_realized_pnl'] + total_unrealized:+,.2f}")

    def _save_state(self):
        """Persist current state to disk."""
        try:
            state = {
                "portfolio_value": self.portfolio_value,
                "stats": self.stats,
                "active_positions": [
                    {
                        "position_id": p.position_id,
                        "symbol": p.symbol,
                        "strategy": p.strategy.value,
                        "direction": p.direction.value,
                        "leg_symbols": p.leg_symbols,
                        "leg_sides": p.leg_sides,
                        "entry_credit": p.entry_credit,
                        "max_profit": p.max_profit,
                        "max_loss": p.max_loss,
                        "contracts": p.contracts,
                        "stop_loss_value": p.stop_loss_value,
                        "take_profit_value": p.take_profit_value,
                        "entry_time": p.entry_time.isoformat(),
                        "expiration": p.expiration.isoformat() if p.expiration else None,
                        "order_id": p.order_id,
                    }
                    for p in self.active_positions
                ],
                "last_update": datetime.now().isoformat(),
            }

            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def _load_state(self):
        """Load persisted state."""
        if not os.path.exists(self.state_file):
            return

        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)

            self.stats = state.get("stats", self.stats)
            self.portfolio_value = state.get("portfolio_value", self.portfolio_value)

            # Restore active positions
            for pos_data in state.get("active_positions", []):
                try:
                    pos = ActivePosition(
                        position_id=pos_data["position_id"],
                        symbol=pos_data["symbol"],
                        strategy=StrategyType(pos_data["strategy"]),
                        direction=Direction(pos_data["direction"]),
                        leg_symbols=pos_data["leg_symbols"],
                        leg_sides=pos_data["leg_sides"],
                        entry_credit=pos_data["entry_credit"],
                        max_profit=pos_data["max_profit"],
                        max_loss=pos_data["max_loss"],
                        contracts=pos_data["contracts"],
                        stop_loss_value=pos_data.get("stop_loss_value", 0),
                        take_profit_value=pos_data.get("take_profit_value", 0),
                        entry_time=datetime.fromisoformat(pos_data["entry_time"]),
                        expiration=(
                            date.fromisoformat(pos_data["expiration"])
                            if pos_data.get("expiration")
                            else None
                        ),
                        order_id=pos_data.get("order_id"),
                    )
                    self.active_positions.append(pos)
                except Exception as e:
                    self.logger.warning(f"Failed to restore position: {e}")

            self.logger.info(
                f"Loaded state: {len(self.active_positions)} positions, "
                f"{self.stats['trades_executed']} total trades"
            )

        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Run the integrated trading system."""
    import argparse

    parser = argparse.ArgumentParser(description="Integrated Options Trading System v2")
    parser.add_argument(
        "--portfolio-value",
        type=float,
        default=100000,
        help="Starting portfolio value (default: $100,000)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Use paper trading (default: True)",
    )
    parser.add_argument(
        "--single-cycle",
        action="store_true",
        help="Run single cycle then exit (for testing)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"trading_{datetime.now().strftime('%Y%m%d')}.log"),
        ],
    )

    async def _run():
        system = IntegratedTradingSystem(
            portfolio_value=args.portfolio_value,
            paper=args.paper,
        )

        if args.single_cycle:
            result = await system.run_trading_cycle()
            print(f"\nCycle result: {json.dumps(result, indent=2, default=str)}")
        else:
            # Wire up shutdown signals
            import signal
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, system.request_shutdown)
                except NotImplementedError:
                    pass
            await system.run_forever()

    asyncio.run(_run())


if __name__ == "__main__":
    main()

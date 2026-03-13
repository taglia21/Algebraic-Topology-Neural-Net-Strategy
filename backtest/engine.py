"""
backtest/engine.py
==================
Event-driven backtesting engine for the ATNN v2 system.

Processes events in chronological order with proper t+1 execution,
FIFO order queue, IBKR commission schedule, and no look-ahead bias.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRADING_DAYS_PER_YEAR: int = 252

# IBKR commission schedule
_EQUITY_COMMISSION_PER_SHARE: float = 0.005
_EQUITY_COMMISSION_MIN: float = 1.00
_EQUITY_COMMISSION_MAX_PCT: float = 0.01  # 1% of trade value
_OPTION_COMMISSION_PER_CONTRACT: float = 0.65
_OPTION_COMMISSION_MIN: float = 1.00

# Default slippage
_DEFAULT_EQUITY_SLIPPAGE: float = 0.001  # 0.1% of price
_DEFAULT_OPTION_SLIPPAGE_TICKS: float = 1.0

# Market hours (ET)
_MARKET_OPEN: time = time(9, 30)
_MARKET_CLOSE: time = time(16, 0)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(enum.Enum):
    """Backtester event types processed in chronological order."""
    MARKET_DATA = "MARKET_DATA"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    END_OF_DAY = "END_OF_DAY"


@dataclass
class Event:
    """A single backtester event."""
    timestamp: pd.Timestamp
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Order types
# ---------------------------------------------------------------------------

class OrderType(enum.Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderSide(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(enum.Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"


class AssetType(enum.Enum):
    EQUITY = "EQUITY"
    OPTION = "OPTION"


@dataclass
class Order:
    """Represents a single order in the backtest."""
    order_id: int
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    asset_type: AssetType = AssetType.EQUITY
    status: OrderStatus = OrderStatus.PENDING
    submitted_at: Optional[pd.Timestamp] = None
    filled_at: Optional[pd.Timestamp] = None
    fill_price: Optional[float] = None
    filled_qty: float = 0.0
    commission: float = 0.0
    slippage_cost: float = 0.0
    strategy: str = ""


@dataclass
class Position:
    """Tracks an open position."""
    symbol: str
    quantity: float  # positive = long, negative = short
    avg_cost: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_date: Optional[pd.Timestamp] = None
    asset_type: AssetType = AssetType.EQUITY


@dataclass
class Trade:
    """A completed round-trip trade."""
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    commission: float
    holding_days: int
    strategy: str = ""
    asset_type: str = "EQUITY"


@dataclass
class BacktestResult:
    """Structured output from a completed backtest run."""
    equity_curve: pd.Series
    trades: pd.DataFrame
    positions_history: List[Dict[str, Any]]
    metrics: dict
    orders: List[Order] = field(default_factory=list)
    initial_capital: float = 444.0


# ---------------------------------------------------------------------------
# Commission calculator
# ---------------------------------------------------------------------------

class CommissionCalculator:
    """Compute IBKR-style commissions."""

    @staticmethod
    def equity_commission(qty: float, price: float) -> float:
        """IBKR tiered equity commission: $0.005/share, min $1, max 1% of trade."""
        qty = abs(qty)
        raw = qty * _EQUITY_COMMISSION_PER_SHARE
        trade_value = qty * price
        cap = trade_value * _EQUITY_COMMISSION_MAX_PCT
        return max(_EQUITY_COMMISSION_MIN, min(raw, cap))

    @staticmethod
    def option_commission(contracts: float) -> float:
        """IBKR option commission: $0.65/contract, min $1."""
        return max(_OPTION_COMMISSION_MIN, abs(contracts) * _OPTION_COMMISSION_PER_CONTRACT)


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Event-driven backtesting engine.

    Processes market data, signals, and orders in strict chronological order.
    Market orders fill at the next bar's open price (t+1 execution).
    Limit orders fill if the bar's range touches the limit price.

    Parameters
    ----------
    equity_slippage : float
        Slippage as fraction of price for equities (default 0.1%).
    option_slippage_ticks : float
        Slippage in ticks for options (default 1 tick).
    respect_market_hours : bool
        If True, only process orders during market hours.
    """

    def __init__(
        self,
        equity_slippage: float = _DEFAULT_EQUITY_SLIPPAGE,
        option_slippage_ticks: float = _DEFAULT_OPTION_SLIPPAGE_TICKS,
        respect_market_hours: bool = True,
    ) -> None:
        self.equity_slippage = equity_slippage
        self.option_slippage_ticks = option_slippage_ticks
        self.respect_market_hours = respect_market_hours

        # State (reset on each run)
        self._cash: float = 0.0
        self._positions: Dict[str, Position] = {}
        self._order_queue: List[Order] = []  # FIFO
        self._next_order_id: int = 1
        self._trades: List[Trade] = []
        self._equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self._positions_history: List[Dict[str, Any]] = []
        self._all_orders: List[Order] = []

    def _reset(self, initial_capital: float) -> None:
        """Reset all state for a new backtest run."""
        self._cash = initial_capital
        self._positions = {}
        self._order_queue = []
        self._next_order_id = 1
        self._trades = []
        self._equity_curve = []
        self._positions_history = []
        self._all_orders = []

    # ----- NAV computation ---------------------------------------------------

    def _compute_nav(self, prices: Dict[str, float]) -> float:
        """Compute Net Asset Value = cash + market value of all positions."""
        nav = self._cash
        for sym, pos in self._positions.items():
            price = prices.get(sym, pos.avg_cost)
            nav += pos.quantity * price
        return nav

    def _update_unrealized(self, prices: Dict[str, float]) -> None:
        """Update unrealized P&L for all positions given current prices."""
        for sym, pos in self._positions.items():
            price = prices.get(sym, pos.avg_cost)
            pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity

    # ----- Order management --------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        asset_type: AssetType = AssetType.EQUITY,
        timestamp: Optional[pd.Timestamp] = None,
        strategy: str = "",
    ) -> Order:
        """Submit an order to the FIFO queue."""
        order = Order(
            order_id=self._next_order_id,
            symbol=symbol,
            side=side,
            quantity=abs(quantity),
            order_type=order_type,
            limit_price=limit_price,
            asset_type=asset_type,
            submitted_at=timestamp,
            strategy=strategy,
        )
        self._next_order_id += 1
        self._order_queue.append(order)
        self._all_orders.append(order)
        return order

    def _process_orders(
        self,
        bar: Dict[str, Dict[str, float]],
        timestamp: pd.Timestamp,
    ) -> List[Order]:
        """Process all pending orders against the current bar (FIFO).

        Market orders: fill at bar open + slippage.
        Limit orders: fill if bar's range includes limit price.

        Parameters
        ----------
        bar : dict
            {symbol: {"open": ..., "high": ..., "low": ..., "close": ...}}
        timestamp : pd.Timestamp
            Current bar timestamp.

        Returns
        -------
        List of filled orders.
        """
        filled = []
        remaining = []

        for order in self._order_queue:
            if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
                continue

            sym_data = bar.get(order.symbol)
            if sym_data is None:
                remaining.append(order)
                continue

            fill_price = self._try_fill(order, sym_data)
            if fill_price is not None:
                commission = self._calc_commission(order, fill_price)
                slippage_cost = abs(fill_price - sym_data.get("open", fill_price)) * order.quantity

                order.fill_price = fill_price
                order.filled_qty = order.quantity
                order.filled_at = timestamp
                order.commission = commission
                order.slippage_cost = slippage_cost
                order.status = OrderStatus.FILLED

                self._apply_fill(order)
                filled.append(order)
            else:
                remaining.append(order)

        self._order_queue = remaining
        return filled

    def _try_fill(
        self,
        order: Order,
        bar_data: Dict[str, float],
    ) -> Optional[float]:
        """Determine fill price for an order against a bar.

        Returns fill price if the order can be filled, None otherwise.
        """
        open_price = bar_data.get("open", 0.0)
        high_price = bar_data.get("high", open_price)
        low_price = bar_data.get("low", open_price)

        if open_price <= 0:
            return None

        if order.order_type == OrderType.MARKET:
            # Market orders fill at open + slippage
            if order.asset_type == AssetType.EQUITY:
                slippage = open_price * self.equity_slippage
            else:
                slippage = self.option_slippage_ticks * 0.01  # tick = 1 cent
            if order.side == OrderSide.BUY:
                return open_price + slippage
            else:
                return max(open_price - slippage, 0.01)

        elif order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                return None
            # Limit buy: fill if bar low <= limit price
            if order.side == OrderSide.BUY and low_price <= order.limit_price:
                return min(order.limit_price, open_price)
            # Limit sell: fill if bar high >= limit price
            if order.side == OrderSide.SELL and high_price >= order.limit_price:
                return max(order.limit_price, open_price)

        return None

    def _calc_commission(self, order: Order, fill_price: float) -> float:
        """Calculate commission for a filled order."""
        if order.asset_type == AssetType.OPTION:
            return CommissionCalculator.option_commission(order.quantity)
        return CommissionCalculator.equity_commission(order.quantity, fill_price)

    def _apply_fill(self, order: Order) -> None:
        """Update cash and positions after a fill."""
        sym = order.symbol
        qty = order.filled_qty if order.side == OrderSide.BUY else -order.filled_qty
        cost = order.fill_price * abs(qty)
        commission = order.commission

        if order.side == OrderSide.BUY:
            self._cash -= cost + commission
        else:
            self._cash += cost - commission

        if sym in self._positions:
            pos = self._positions[sym]
            old_qty = pos.quantity
            new_qty = old_qty + qty

            if old_qty != 0 and np.sign(old_qty) != np.sign(qty):
                # Closing (fully or partially)
                close_qty = min(abs(qty), abs(old_qty))
                pnl = (order.fill_price - pos.avg_cost) * close_qty * np.sign(old_qty)
                pos.realized_pnl += pnl

                # Record trade
                trade_side = "LONG" if old_qty > 0 else "SHORT"
                holding_days = 0
                if pos.entry_date is not None and order.filled_at is not None:
                    holding_days = max(1, (order.filled_at - pos.entry_date).days)

                self._trades.append(Trade(
                    symbol=sym,
                    side=trade_side,
                    entry_date=pos.entry_date or order.filled_at,
                    exit_date=order.filled_at,
                    entry_price=pos.avg_cost,
                    exit_price=order.fill_price,
                    qty=close_qty,
                    pnl=pnl - commission,
                    commission=commission,
                    holding_days=holding_days,
                    strategy=order.strategy,
                    asset_type=order.asset_type.value,
                ))

                if abs(new_qty) < 1e-10:
                    del self._positions[sym]
                    return
                elif np.sign(new_qty) == np.sign(old_qty):
                    # Partial close, same direction
                    pos.quantity = new_qty
                else:
                    # Flipped direction
                    pos.quantity = new_qty
                    pos.avg_cost = order.fill_price
                    pos.entry_date = order.filled_at
            else:
                # Adding to position
                if abs(old_qty) < 1e-10:
                    pos.avg_cost = order.fill_price
                    pos.entry_date = order.filled_at
                else:
                    total_cost = pos.avg_cost * abs(old_qty) + order.fill_price * abs(qty)
                    pos.avg_cost = total_cost / abs(new_qty) if abs(new_qty) > 0 else 0
                pos.quantity = new_qty
        else:
            self._positions[sym] = Position(
                symbol=sym,
                quantity=qty,
                avg_cost=order.fill_price,
                entry_date=order.filled_at,
                asset_type=order.asset_type,
            )

    # ----- Signal processing -------------------------------------------------

    def _process_signals(
        self,
        signals_row: pd.Series,
        timestamp: pd.Timestamp,
    ) -> List[Order]:
        """Convert a row of signals into orders.

        Expected signal columns: symbol, direction (1=long, -1=short, 0=flat),
        strength, strategy.
        """
        orders = []
        if signals_row is None:
            return orders

        for col in signals_row.index:
            if col in ("date", "timestamp"):
                continue
            val = signals_row[col]
            if isinstance(val, (int, float)) and val != 0:
                symbol = col
                side = OrderSide.BUY if val > 0 else OrderSide.SELL
                # Check if we need to close existing position
                if symbol in self._positions:
                    pos = self._positions[symbol]
                    if (val > 0 and pos.quantity < 0) or (val < 0 and pos.quantity > 0):
                        # Close existing position first
                        close_side = OrderSide.BUY if pos.quantity < 0 else OrderSide.SELL
                        order = self.submit_order(
                            symbol=symbol,
                            side=close_side,
                            quantity=abs(pos.quantity),
                            timestamp=timestamp,
                            strategy="close",
                        )
                        orders.append(order)

        return orders

    # ----- Main run method ---------------------------------------------------

    def run(
        self,
        signals: pd.DataFrame,
        price_data: pd.DataFrame,
        initial_capital: float = 444.0,
        position_sizer: Optional[Any] = None,
    ) -> BacktestResult:
        """Run the backtest.

        Parameters
        ----------
        signals : pd.DataFrame
            Indexed by date. Columns are symbols, values are signal direction
            (+1 long, -1 short, 0 flat). Can also have 'strength' and
            'strategy' columns.
        price_data : pd.DataFrame
            OHLCV data with MultiIndex columns: (symbol, field) where field
            is one of 'open', 'high', 'low', 'close', 'volume'.
            OR a dict-of-dicts: {date: {symbol: {"open":..., "close":...}}}
        initial_capital : float
            Starting capital (default $444).
        position_sizer : callable, optional
            Function(signal_value, nav, symbol) → qty. If None, uses a
            default equal-weight sizer.

        Returns
        -------
        BacktestResult
        """
        self._reset(initial_capital)

        # Normalize price data to {date: {symbol: {open, high, low, close}}}
        bar_data = self._normalize_price_data(price_data)

        # Normalize signals
        signal_dates = sorted(signals.index)
        all_dates = sorted(bar_data.keys())

        if not all_dates:
            return self._build_result(initial_capital)

        # Determine symbols from signals and price data
        signal_symbols = [
            c for c in signals.columns
            if c not in ("date", "timestamp", "strength", "strategy")
        ]

        logger.info(
            "Backtest: %d bars, %d signal dates, capital=$%.2f",
            len(all_dates), len(signal_dates), initial_capital,
        )

        prev_date: Optional[pd.Timestamp] = None

        for i, date in enumerate(all_dates):
            bars = bar_data[date]

            # Get current prices for NAV computation
            prices = {sym: d.get("close", 0.0) for sym, d in bars.items()}

            # 1. Process pending orders from previous bar (t+1 execution)
            if self._order_queue:
                self._process_orders(bars, date)

            # 2. Process signals for this date
            if date in signals.index:
                sig_row = signals.loc[date]
                if isinstance(sig_row, pd.DataFrame):
                    sig_row = sig_row.iloc[0]

                for sym in signal_symbols:
                    if sym not in sig_row.index:
                        continue
                    signal_val = sig_row[sym]
                    if pd.isna(signal_val) or signal_val == 0:
                        # Flat signal: close position if exists
                        if sym in self._positions:
                            pos = self._positions[sym]
                            close_side = OrderSide.BUY if pos.quantity < 0 else OrderSide.SELL
                            self.submit_order(
                                symbol=sym,
                                side=close_side,
                                quantity=abs(pos.quantity),
                                timestamp=date,
                                strategy="signal_exit",
                            )
                        continue

                    # Determine desired position
                    direction = 1 if signal_val > 0 else -1
                    nav = self._compute_nav(prices)

                    if position_sizer:
                        qty = position_sizer(signal_val, nav, sym)
                    else:
                        # Default: equal weight, 10% of NAV per position
                        sym_price = prices.get(sym, 0.0)
                        if sym_price > 0:
                            alloc = nav * 0.10
                            qty = round(alloc / sym_price, 4)
                        else:
                            qty = 0

                    if qty <= 0:
                        continue

                    current_qty = 0.0
                    if sym in self._positions:
                        current_qty = self._positions[sym].quantity

                    desired_qty = qty * direction

                    if abs(desired_qty - current_qty) < 1e-10:
                        continue  # Already at target

                    # Close existing position if direction changes
                    if current_qty != 0 and np.sign(current_qty) != np.sign(desired_qty):
                        close_side = OrderSide.BUY if current_qty < 0 else OrderSide.SELL
                        self.submit_order(
                            symbol=sym,
                            side=close_side,
                            quantity=abs(current_qty),
                            timestamp=date,
                            strategy="flip",
                        )

                    # Open new position
                    side = OrderSide.BUY if desired_qty > 0 else OrderSide.SELL
                    open_qty = abs(desired_qty)
                    if current_qty != 0 and np.sign(current_qty) == np.sign(desired_qty):
                        # Adjust existing position
                        delta = abs(desired_qty) - abs(current_qty)
                        if delta > 0:
                            open_qty = delta
                        else:
                            # Reduce position
                            side = OrderSide.SELL if desired_qty > 0 else OrderSide.BUY
                            open_qty = abs(delta)

                    if open_qty > 0:
                        self.submit_order(
                            symbol=sym,
                            side=side,
                            quantity=open_qty,
                            timestamp=date,
                            strategy=str(sig_row.get("strategy", "")) if hasattr(sig_row, "get") else "",
                        )

            # 3. Update unrealized P&L and record equity
            self._update_unrealized(prices)
            nav = self._compute_nav(prices)
            self._equity_curve.append((date, nav))

            # 4. Record position snapshot
            pos_snap = {
                "date": date,
                "cash": self._cash,
                "nav": nav,
                "positions": {
                    sym: {"qty": p.quantity, "avg_cost": p.avg_cost, "unrealized_pnl": p.unrealized_pnl}
                    for sym, p in self._positions.items()
                },
            }
            self._positions_history.append(pos_snap)

            if (i + 1) % 100 == 0:
                logger.debug("Bar %d/%d | NAV=$%.2f | Positions=%d",
                             i + 1, len(all_dates), nav, len(self._positions))

            prev_date = date

        # Close all remaining positions at last close
        if all_dates:
            last_bars = bar_data[all_dates[-1]]
            last_prices = {sym: d.get("close", 0.0) for sym, d in last_bars.items()}
            for sym, pos in list(self._positions.items()):
                close_side = OrderSide.BUY if pos.quantity < 0 else OrderSide.SELL
                self.submit_order(
                    symbol=sym,
                    side=close_side,
                    quantity=abs(pos.quantity),
                    timestamp=all_dates[-1],
                    strategy="eod_close",
                )
            # Process final close orders immediately at close prices
            close_bars = {
                sym: {"open": p, "high": p, "low": p, "close": p}
                for sym, p in last_prices.items()
            }
            self._process_orders(close_bars, all_dates[-1])

        return self._build_result(initial_capital)

    # ----- Data normalization ------------------------------------------------

    @staticmethod
    def _normalize_price_data(
        price_data: pd.DataFrame,
    ) -> Dict[pd.Timestamp, Dict[str, Dict[str, float]]]:
        """Convert price_data into {date: {symbol: {open, high, low, close}}} dict.

        Supports:
        - MultiIndex columns: (symbol, field)
        - Single-symbol DataFrame with open/high/low/close columns
        - Already normalized dict
        """
        if isinstance(price_data, dict):
            return price_data

        result: Dict[pd.Timestamp, Dict[str, Dict[str, float]]] = {}

        if isinstance(price_data.columns, pd.MultiIndex):
            symbols = price_data.columns.get_level_values(0).unique()
            for date in price_data.index:
                result[date] = {}
                for sym in symbols:
                    try:
                        row = price_data.loc[date, sym]
                        result[date][sym] = {
                            "open": float(row.get("open", row.get("Open", 0))),
                            "high": float(row.get("high", row.get("High", 0))),
                            "low": float(row.get("low", row.get("Low", 0))),
                            "close": float(row.get("close", row.get("Close", 0))),
                        }
                    except (KeyError, TypeError):
                        pass
        else:
            # Single-symbol: assume columns are open/high/low/close
            col_map = {}
            for c in price_data.columns:
                cl = c.lower()
                if "open" in cl:
                    col_map["open"] = c
                elif "high" in cl:
                    col_map["high"] = c
                elif "low" in cl:
                    col_map["low"] = c
                elif "close" in cl:
                    col_map["close"] = c

            sym = "ASSET"
            for date in price_data.index:
                row = price_data.loc[date]
                result[date] = {
                    sym: {
                        "open": float(row.get(col_map.get("open", "open"), 0)),
                        "high": float(row.get(col_map.get("high", "high"), 0)),
                        "low": float(row.get(col_map.get("low", "low"), 0)),
                        "close": float(row.get(col_map.get("close", "close"), 0)),
                    },
                }

        return result

    # ----- Result builder ----------------------------------------------------

    def _build_result(self, initial_capital: float) -> BacktestResult:
        """Assemble the BacktestResult from internal state."""
        from backtest.metrics import BacktestMetrics

        if self._equity_curve:
            dates, values = zip(*self._equity_curve)
            eq = pd.Series(values, index=pd.DatetimeIndex(dates), name="equity")
        else:
            eq = pd.Series(dtype=float, name="equity")

        # Build trades DataFrame
        if self._trades:
            trades_df = pd.DataFrame([
                {
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_date": t.entry_date,
                    "exit_date": t.exit_date,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "qty": t.qty,
                    "pnl": t.pnl,
                    "commission": t.commission,
                    "holding_days": t.holding_days,
                    "strategy": t.strategy,
                    "asset_type": t.asset_type,
                }
                for t in self._trades
            ])
        else:
            trades_df = pd.DataFrame(columns=[
                "symbol", "side", "entry_date", "exit_date",
                "entry_price", "exit_price", "qty", "pnl",
                "commission", "holding_days", "strategy", "asset_type",
            ])

        # Compute metrics
        trade_dicts = trades_df.to_dict("records") if len(trades_df) > 0 else []
        metrics = BacktestMetrics.compute_all(eq, trade_dicts, initial_capital)

        return BacktestResult(
            equity_curve=eq,
            trades=trades_df,
            positions_history=self._positions_history,
            metrics=metrics,
            orders=self._all_orders,
            initial_capital=initial_capital,
        )

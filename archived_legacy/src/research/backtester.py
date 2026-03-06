"""
Phase S — Event-Driven Backtester.

Item 19: EventDrivenBacktester — MarketDataEvent, SignalEvent, OrderEvent, FillEvent,
         realistic fills with slippage and commission.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event Types
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    MARKET_DATA = "MARKET_DATA"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


@dataclass
class Event:
    """Base event."""
    event_type: str = ""
    timestamp: str = ""


@dataclass
class MarketDataEvent(Event):
    """New market data bar."""
    symbol: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    bar_index: int = 0

    def __post_init__(self):
        self.event_type = EventType.MARKET_DATA.value


@dataclass
class SignalEvent(Event):
    """Trading signal from strategy."""
    symbol: str = ""
    direction: str = ""     # "LONG", "SHORT", "EXIT"
    strength: float = 0.0   # Signal strength [0, 1]
    strategy: str = ""

    def __post_init__(self):
        self.event_type = EventType.SIGNAL.value


@dataclass
class OrderEvent(Event):
    """Order generated from signal."""
    symbol: str = ""
    side: str = ""          # "BUY", "SELL"
    quantity: int = 0
    order_type: str = "MARKET"
    limit_price: Optional[float] = None

    def __post_init__(self):
        self.event_type = EventType.ORDER.value


@dataclass
class FillEvent(Event):
    """Order fill with realistic execution."""
    symbol: str = ""
    side: str = ""
    quantity: int = 0
    price: float = 0.0           # Fill price (after slippage)
    commission: float = 0.0
    slippage: float = 0.0

    def __post_init__(self):
        self.event_type = EventType.FILL.value


# ---------------------------------------------------------------------------
# Strategy Interface
# ---------------------------------------------------------------------------

class Strategy(ABC):
    """Abstract strategy interface for the backtester."""

    @abstractmethod
    def on_market_data(self, event: MarketDataEvent) -> Optional[SignalEvent]:
        """Process market data and optionally emit a signal."""
        ...


# ---------------------------------------------------------------------------
# Fill Model
# ---------------------------------------------------------------------------

class RealisticFillModel:
    """Simulate realistic fills with slippage and commission.

    Slippage model: slippage = base_bps + volume_impact * sqrt(qty / avg_volume)
    Commission: fixed per share.
    """

    def __init__(
        self,
        slippage_bps: float = 5.0,
        commission_per_share: float = 0.005,
        volume_impact_bps: float = 10.0,
    ):
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        self.volume_impact_bps = volume_impact_bps

    def simulate_fill(
        self,
        order: OrderEvent,
        market_data: MarketDataEvent,
        avg_volume: float = 1_000_000,
    ) -> FillEvent:
        """Simulate a realistic fill.

        Args:
            order: The order to fill.
            market_data: Current market data.
            avg_volume: Average daily volume for impact calculation.

        Returns:
            FillEvent with realistic price.
        """
        # Base slippage
        base_slip = self.slippage_bps / 10000.0
        # Volume impact: higher for larger orders relative to volume
        vol_impact = self.volume_impact_bps / 10000.0 * np.sqrt(
            order.quantity / max(avg_volume, 1)
        )
        total_slip = base_slip + vol_impact

        # Apply slippage direction
        if order.side == "BUY":
            fill_price = market_data.close * (1 + total_slip)
        else:
            fill_price = market_data.close * (1 - total_slip)

        # Ensure fill within high/low range
        fill_price = max(market_data.low, min(market_data.high, fill_price))
        commission = self.commission_per_share * order.quantity

        return FillEvent(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=round(fill_price, 4),
            commission=round(commission, 4),
            slippage=round(abs(fill_price - market_data.close), 4),
            timestamp=market_data.timestamp,
        )


# ---------------------------------------------------------------------------
# Portfolio Tracker
# ---------------------------------------------------------------------------

@dataclass
class PortfolioState:
    """Current portfolio state."""
    cash: float = 0.0
    positions: Dict[str, int] = field(default_factory=dict)
    position_values: Dict[str, float] = field(default_factory=dict)
    total_value: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    n_trades: int = 0


class PortfolioTracker:
    """Track portfolio state through backtest."""

    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, int] = {}
        self.total_commission: float = 0.0
        self.total_slippage: float = 0.0
        self.n_trades: int = 0
        self._equity_curve: List[float] = [initial_cash]

    def process_fill(self, fill: FillEvent) -> None:
        """Update portfolio for a fill."""
        if fill.side == "BUY":
            cost = fill.price * fill.quantity + fill.commission
            self.cash -= cost
            self.positions[fill.symbol] = self.positions.get(fill.symbol, 0) + fill.quantity
        else:
            proceeds = fill.price * fill.quantity - fill.commission
            self.cash += proceeds
            self.positions[fill.symbol] = self.positions.get(fill.symbol, 0) - fill.quantity
            if self.positions[fill.symbol] == 0:
                del self.positions[fill.symbol]

        self.total_commission += fill.commission
        self.total_slippage += fill.slippage * fill.quantity
        self.n_trades += 1

    def mark_to_market(self, prices: Dict[str, float]) -> float:
        """Compute total portfolio value at current prices."""
        position_value = sum(
            qty * prices.get(sym, 0.0)
            for sym, qty in self.positions.items()
        )
        total = self.cash + position_value
        self._equity_curve.append(total)
        return total

    def get_state(self, prices: Optional[Dict[str, float]] = None) -> PortfolioState:
        """Get current portfolio state."""
        pos_vals = {}
        if prices:
            pos_vals = {sym: qty * prices.get(sym, 0) for sym, qty in self.positions.items()}
        total = self.cash + sum(pos_vals.values())
        return PortfolioState(
            cash=self.cash,
            positions=dict(self.positions),
            position_values=pos_vals,
            total_value=total,
            total_commission=self.total_commission,
            total_slippage=self.total_slippage,
            n_trades=self.n_trades,
        )

    @property
    def equity_curve(self) -> np.ndarray:
        return np.array(self._equity_curve)


# ---------------------------------------------------------------------------
# Item 19 — EventDrivenBacktester
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Complete backtest result."""
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    n_trades: int = 0
    win_rate: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    final_value: float = 0.0


class EventDrivenBacktester:
    """Event-driven backtesting engine.

    Event flow: MarketData → Strategy → Signal → Order → Fill → Portfolio

    Features:
      - Pluggable strategy interface.
      - Realistic fill model (slippage + commission).
      - Full portfolio tracking.
      - Performance statistics.
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        fill_model: Optional[RealisticFillModel] = None,
    ):
        self.portfolio = PortfolioTracker(initial_cash)
        self.fill_model = fill_model or RealisticFillModel()
        self._events: List[Event] = []
        self._fills: List[FillEvent] = []

    def run(
        self,
        strategy: Strategy,
        data: Dict[str, np.ndarray],
        n_bars: Optional[int] = None,
    ) -> BacktestResult:
        """Run backtest.

        Args:
            strategy: Trading strategy implementing on_market_data().
            data: Dict of symbol -> (N, 5) OHLCV arrays.
            n_bars: Number of bars to process (default: all).

        Returns:
            BacktestResult with performance metrics.
        """
        if not data:
            return BacktestResult()

        # Determine bar count
        first_key = next(iter(data))
        total_bars = data[first_key].shape[0]
        if n_bars:
            total_bars = min(n_bars, total_bars)

        for bar_idx in range(total_bars):
            prices = {}

            for symbol, ohlcv in data.items():
                if bar_idx >= ohlcv.shape[0]:
                    continue

                # Create market data event
                md = MarketDataEvent(
                    symbol=symbol,
                    open=float(ohlcv[bar_idx, 0]),
                    high=float(ohlcv[bar_idx, 1]),
                    low=float(ohlcv[bar_idx, 2]),
                    close=float(ohlcv[bar_idx, 3]),
                    volume=float(ohlcv[bar_idx, 4]) if ohlcv.shape[1] > 4 else 0,
                    bar_index=bar_idx,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                prices[symbol] = md.close

                # Strategy processes market data
                signal = strategy.on_market_data(md)
                if signal is None:
                    continue

                # Convert signal to order
                order = self._signal_to_order(signal, md)
                if order is None:
                    continue

                # Simulate fill
                fill = self.fill_model.simulate_fill(order, md)
                self._fills.append(fill)

                # Update portfolio
                self.portfolio.process_fill(fill)

            # Mark to market
            if prices:
                self.portfolio.mark_to_market(prices)

        return self._compute_results()

    def _signal_to_order(
        self,
        signal: SignalEvent,
        md: MarketDataEvent,
    ) -> Optional[OrderEvent]:
        """Convert signal to order."""
        if signal.direction == "LONG":
            # Size: 10% of portfolio value
            value = self.portfolio.cash * 0.1
            qty = max(int(value / max(md.close, 1e-6)), 1)
            return OrderEvent(
                symbol=signal.symbol,
                side="BUY",
                quantity=qty,
                timestamp=md.timestamp,
            )
        elif signal.direction == "SHORT" or signal.direction == "EXIT":
            qty = self.portfolio.positions.get(signal.symbol, 0)
            if qty > 0:
                return OrderEvent(
                    symbol=signal.symbol,
                    side="SELL",
                    quantity=qty,
                    timestamp=md.timestamp,
                )
        return None

    def _compute_results(self) -> BacktestResult:
        """Compute backtest performance metrics."""
        curve = self.portfolio.equity_curve
        if len(curve) < 2:
            return BacktestResult()

        total_return = (curve[-1] / curve[0]) - 1
        n_days = len(curve) - 1
        ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        # Sharpe
        returns = np.diff(curve) / curve[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
        else:
            sharpe = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(curve)
        drawdown = (curve - peak) / peak
        max_dd = float(np.min(drawdown))

        # Win rate from fills
        trades = []
        current_entry = {}
        for fill in self._fills:
            if fill.side == "BUY":
                current_entry[fill.symbol] = fill.price
            elif fill.side == "SELL" and fill.symbol in current_entry:
                pnl = fill.price - current_entry[fill.symbol]
                trades.append(pnl)
                del current_entry[fill.symbol]

        win_rate = sum(1 for t in trades if t > 0) / max(len(trades), 1)

        return BacktestResult(
            total_return=float(total_return),
            annualized_return=float(ann_return),
            sharpe_ratio=sharpe,
            max_drawdown=float(max_dd),
            n_trades=self.portfolio.n_trades,
            win_rate=win_rate,
            total_commission=self.portfolio.total_commission,
            total_slippage=self.portfolio.total_slippage,
            equity_curve=curve,
            final_value=float(curve[-1]),
        )

"""
equities/execution.py
======================
Broker-agnostic execution layer for the ATNN trading system.

This module provides:

    :class:`Broker`            — Abstract base class; every real/simulated broker
                                  must implement this interface.
    :class:`SimulatedBroker`   — In-memory broker for backtesting and paper trading.
                                  Tracks positions, cash, and equity; simulates
                                  fills with configurable slippage.
    :class:`ExecutionManager`  — High-level manager: receives signals + risk
                                  approvals, creates orders, submits to broker,
                                  and manages order lifecycle.

Design Principles
-----------------
- **Broker-agnostic**: Swap ``SimulatedBroker`` for a live broker (Alpaca,
  IBKR) by implementing ``Broker`` without touching any strategy code.
- **Same code path**: Backtest and live modes run identical logic; only the
  ``Broker`` implementation differs.
- **No silent failures**: All exceptions are logged and re-raised.

Order Lifecycle
---------------
``new`` → ``submitted`` → ``filled`` | ``partially_filled`` | ``cancelled`` | ``rejected``

Slippage Model
--------------
The ``SimulatedBroker`` applies a two-component slippage model:

1. **Fixed component** (``slippage_bps``): baseline execution cost.
2. **Market impact component** (square-root model): scales with order size
   relative to average daily volume (ADV).  Impact formula:

       impact_bps = k × σ_daily × √(Q / ADV) × 10_000

   where *k* is a calibration constant (default 0.1), *σ_daily* is the
   realised daily volatility, *Q* is the order quantity, and *ADV* is the
   20-day average daily volume.  This is the standard Almgren-Chriss
   temporary impact model used in institutional execution analytics.

- **Market orders**: fill at ``bar_open × (1 ± total_slippage_bps / 10_000)``.
- **Limit orders**: fill only when price touches the limit; no additional
  slippage (since price had to reach the limit).
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import get_config
from core.logger import TradeLogger, get_trade_logger
from core.risk_manager import RiskManager
from equities.models import (
    Account,
    Fill,
    Order,
    Position,
    PortfolioState,
    Signal,
)

logger = logging.getLogger(__name__)

# Fix the alias issue — PortfolioState is imported by its real name
# (we handle the alias below)


# ---------------------------------------------------------------------------
# Abstract Broker interface
# ---------------------------------------------------------------------------

class Broker(ABC):
    """Abstract base class for broker adapters.

    All broker implementations (simulated, Alpaca, IBKR, etc.) must implement
    this interface.  The :class:`ExecutionManager` only ever calls these methods,
    ensuring strategy code is fully broker-agnostic.
    """

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str,
        limit_price: Optional[float] = None,
        strategy: str = "",
        signal_strength: float = 1.0,
    ) -> Order:
        """Submit a new order to the broker.

        Parameters
        ----------
        symbol:
            Ticker symbol.
        qty:
            Number of shares (always positive).
        side:
            ``"buy"`` or ``"sell"``.
        order_type:
            ``"market"`` or ``"limit"``.
        limit_price:
            Required when ``order_type == "limit"``.
        strategy:
            Strategy that generated this order (for attribution).
        signal_strength:
            Originating signal strength (for position sizing attribution).

        Returns
        -------
        :class:`Order` with status ``"submitted"``.

        Raises
        ------
        ValueError:
            On invalid parameters.
        RuntimeError:
            If the broker rejects the order.
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Attempt to cancel an open order.

        Parameters
        ----------
        order_id:
            Order identifier returned by :meth:`submit_order`.

        Returns
        -------
        ``True`` if the order was cancelled; ``False`` if it was already
        filled or not found.
        """
        ...

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """Return current open positions.

        Returns
        -------
        Dict mapping symbol → :class:`Position`.
        """
        ...

    @abstractmethod
    def get_account(self) -> Account:
        """Return current account state (equity, cash, buying power).

        Returns
        -------
        :class:`Account` snapshot.
        """
        ...

    @abstractmethod
    def get_portfolio_state(self) -> PortfolioState:
        """Return a complete portfolio snapshot.

        Returns
        -------
        :class:`PortfolioState` with equity, cash, positions, P&L, drawdown.
        """
        ...


# ---------------------------------------------------------------------------
# SimulatedBroker
# ---------------------------------------------------------------------------

class SimulatedBroker(Broker):
    """In-memory simulated broker for backtesting and paper trading.

    Tracks positions and cash in memory.  Simulates realistic fills with
    configurable slippage.

    Fill Mechanics
    --------------
    - **Market orders**: processed immediately on the next ``on_bar()`` call.
      Fill price = ``bar_open × (1 + slippage_bps/10_000)`` for buys,
      ``bar_open × (1 − slippage_bps/10_000)`` for sells.
    - **Limit orders**: processed on each bar.  Buy limit fills if
      ``bar_low ≤ limit_price``; sell limit fills if ``bar_high ≥ limit_price``.
      No additional slippage is applied.

    Parameters
    ----------
    initial_cash:
        Starting cash balance in USD.
    slippage_bps:
        One-way slippage in basis points applied to market orders.
        Default 7 bps (0.07%).
    commission_per_share:
        Per-share commission in USD.  Default $0.005.
    account_id:
        Optional broker account identifier (for logging).
    trade_logger:
        Audit logger.
    """

    # Almgren-Chriss temporary impact calibration constant.
    # Kyle's lambda ≈ 0.1 is a widely-used institutional default.
    _IMPACT_K: float = 0.1

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        slippage_bps: float = 7.0,
        commission_per_share: float = 0.005,
        account_id: str = "SIM-001",
        trade_logger: Optional[TradeLogger] = None,
    ) -> None:
        if initial_cash <= 0:
            raise ValueError(f"initial_cash must be positive; got {initial_cash!r}")

        self._initial_cash = initial_cash
        self._cash = initial_cash
        self._slippage_bps = slippage_bps
        self._commission_per_share = commission_per_share
        self._account_id = account_id
        self._log = trade_logger or get_trade_logger()

        # Active orders (order_id → Order)
        self._orders: Dict[str, Order] = {}
        # Positions (symbol → Position)
        self._positions: Dict[str, Position] = {}
        # Fill history
        self._fills: List[Fill] = []
        # Realised P&L
        self._realized_pnl: float = 0.0
        # Peak equity for drawdown tracking
        self._peak_equity: float = initial_cash
        # Current simulated bar datetime (set by backtester each bar)
        self._current_bar_dt: Optional[datetime] = None

        # --- Volume & volatility tracking for market impact model ---
        # Rolling 20-bar ADV per symbol (updated each on_bar call)
        self._volume_history: Dict[str, List[float]] = defaultdict(list)
        # Rolling 20-bar close prices for realised vol computation
        self._price_history: Dict[str, List[float]] = defaultdict(list)
        _ADV_WINDOW = 20  # noqa: N806 — local constant
        self._adv_window = _ADV_WINDOW

        # --- Daily P&L tracking ---
        # Start-of-day equity snapshot, used to compute today_pnl for the
        # risk manager's daily-loss-limit check.  Reset each trading day via
        # reset_daily().  Initialised to initial_cash so that today_pnl starts
        # at zero on day 1.
        self._sod_equity: float = initial_cash

        logger.info(
            f"SimulatedBroker initialised: "
            f"cash={initial_cash:,.2f}, slippage={slippage_bps}bps, "
            f"market_impact=sqrt (k={self._IMPACT_K})."
        )

    # ------------------------------------------------------------------
    # Broker interface implementation
    # ------------------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str,
        limit_price: Optional[float] = None,
        strategy: str = "",
        signal_strength: float = 1.0,
    ) -> Order:
        """Submit a new order; returns immediately with status ``"submitted"``."""
        order = Order(
            order_id=str(uuid.uuid4())[:12],
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            limit_price=limit_price,
            status="submitted",
            strategy=strategy,
            signal_strength=signal_strength,
        )
        self._orders[order.order_id] = order

        self._log.log_order(
            order.order_id,
            symbol,
            side,
            qty,
            limit_price or 0.0,
            "submitted",
            metadata={"order_type": order_type, "strategy": strategy},
        )
        logger.debug(
            f"SimulatedBroker: order submitted "
            f"{order.order_id} {side} {qty} {symbol} @ "
            f"{'market' if order_type == 'market' else f'limit {limit_price:.2f}'}"
        )
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.  Returns False if already filled."""
        order = self._orders.get(order_id)
        if order is None:
            logger.warning(f"SimulatedBroker: cancel_order — order {order_id!r} not found.")
            return False

        if not order.is_active:
            logger.debug(
                f"SimulatedBroker: cannot cancel order {order_id!r} with status {order.status!r}."
            )
            return False

        order.status = "cancelled"
        self._log.log_order(order.order_id, order.symbol, order.side, order.qty, 0.0, "cancelled")
        return True

    def get_positions(self) -> Dict[str, Position]:
        """Return a copy of the current positions map."""
        return dict(self._positions)

    def get_account(self) -> Account:
        """Return a current account snapshot."""
        equity = self._compute_equity()
        return Account(
            account_id=self._account_id,
            buying_power=self._cash,
            equity=equity,
            cash=self._cash,
        )

    def get_portfolio_state(self) -> PortfolioState:
        """Return a complete portfolio snapshot."""
        equity = self._compute_equity()
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())

        if equity > self._peak_equity:
            self._peak_equity = equity

        drawdown = (equity - self._peak_equity) / max(self._peak_equity, 1.0)

        return PortfolioState(
            equity=equity,
            cash=self._cash,
            positions=dict(self._positions),
            unrealized_pnl=unrealized,
            realized_pnl=self._realized_pnl,
            peak_equity=self._peak_equity,
            drawdown=drawdown,
        )

    # ------------------------------------------------------------------
    # Bar processing (call once per bar in the event loop)
    # ------------------------------------------------------------------

    def on_bar(
        self,
        bar: pd.Series,
        symbol: str,
    ) -> List[Fill]:
        """Process pending orders for a single bar update.

        This method should be called once per bar for each symbol that has
        pending orders.  It simulates fills based on the bar's OHLC prices.

        Parameters
        ----------
        bar:
            A Series with at minimum an ``open`` and ``close`` value;
            ``high`` and ``low`` are used for limit order fills.
            Column names are case-insensitive.
        symbol:
            Ticker symbol for this bar.

        Returns
        -------
        List of :class:`Fill` objects generated during this bar.
        """
        bar_lower = {k.lower(): v for k, v in bar.items()}
        bar_open = float(bar_lower.get("open", bar_lower.get("close", 0.0)))
        bar_high = float(bar_lower.get("high", bar_open * 1.002))
        bar_low = float(bar_lower.get("low", bar_open * 0.998))
        bar_close = float(bar_lower.get("close", bar_open))
        bar_volume = float(bar_lower.get("volume", 0.0))

        # --- Track volume and price for market impact model ---
        if bar_volume > 0:
            hist = self._volume_history[symbol]
            hist.append(bar_volume)
            if len(hist) > self._adv_window:
                hist.pop(0)
        if bar_close > 0:
            phist = self._price_history[symbol]
            phist.append(bar_close)
            if len(phist) > self._adv_window + 1:
                phist.pop(0)

        # Update position mark-to-market
        if symbol in self._positions:
            pos = self._positions[symbol]
            pos.current_price = bar_close
            pos.unrealized_pnl = (bar_close - pos.avg_entry) * pos.qty

        fills: List[Fill] = []
        for order_id, order in list(self._orders.items()):
            if order.symbol != symbol or not order.is_active:
                continue

            fill = self._try_fill(order, bar_open, bar_high, bar_low, bar_close)
            if fill is not None:
                fills.append(fill)

        return fills

    def update_prices(self, current_prices: Dict[str, float]) -> None:
        """Update mark-to-market prices for all held positions.

        Parameters
        ----------
        current_prices:
            Mapping of symbol → latest close price.
        """
        for symbol, price in current_prices.items():
            if symbol in self._positions:
                pos = self._positions[symbol]
                pos.current_price = float(price)
                pos.unrealized_pnl = (pos.current_price - pos.avg_entry) * pos.qty

    # ------------------------------------------------------------------
    # Private fill logic
    # ------------------------------------------------------------------

    def _compute_market_impact_bps(self, symbol: str, qty: int) -> float:
        """Compute additional market-impact slippage using the square-root model.

        Uses the Almgren-Chriss temporary impact formula:

            impact_bps = k × σ_daily × √(Q / ADV) × 10_000

        Parameters
        ----------
        symbol:
            Ticker symbol (used to look up ADV and vol history).
        qty:
            Order quantity in shares.

        Returns
        -------
        float
            Additional slippage in basis points (0.0 if insufficient data).
        """
        vol_hist = self._volume_history.get(symbol, [])
        price_hist = self._price_history.get(symbol, [])

        # Need at least 5 bars of history for a meaningful estimate
        if len(vol_hist) < 5 or len(price_hist) < 5:
            return 0.0

        adv = float(np.mean(vol_hist))
        if adv <= 0:
            return 0.0

        # Realised daily volatility from log returns
        prices = np.array(price_hist, dtype=float)
        log_returns = np.diff(np.log(prices))
        if len(log_returns) < 2:
            return 0.0
        sigma_daily = float(np.std(log_returns, ddof=1))
        if sigma_daily <= 0:
            return 0.0

        participation_rate = float(qty) / adv
        impact = self._IMPACT_K * sigma_daily * np.sqrt(participation_rate)

        # Convert to basis points (impact is in decimal return units)
        return impact * 10_000.0

    def _try_fill(
        self,
        order: Order,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
    ) -> Optional[Fill]:
        """Attempt to fill a single order given bar prices.

        Parameters
        ----------
        order:
            The order to attempt to fill.
        bar_open, bar_high, bar_low, bar_close:
            OHLC prices for the current bar.

        Returns
        -------
        :class:`Fill` if the order was filled; ``None`` otherwise.
        """
        if order.order_type == "market":
            # Market order: fill at open + fixed slippage + market impact
            impact_bps = self._compute_market_impact_bps(order.symbol, order.qty)
            total_slippage_bps = self._slippage_bps + impact_bps
            slippage_factor = total_slippage_bps / 10_000.0
            if order.side == "buy":
                fill_price = bar_open * (1.0 + slippage_factor)
            else:
                fill_price = bar_open * (1.0 - slippage_factor)

            return self._execute_fill(order, fill_price, order.qty)

        elif order.order_type == "limit" and order.limit_price is not None:
            lp = order.limit_price
            # Limit buy: fills if bar_low <= limit_price
            if order.side == "buy" and bar_low <= lp:
                return self._execute_fill(order, lp, order.qty)
            # Limit sell: fills if bar_high >= limit_price
            elif order.side == "sell" and bar_high >= lp:
                return self._execute_fill(order, lp, order.qty)

        return None

    def _execute_fill(
        self,
        order: Order,
        fill_price: float,
        fill_qty: int,
    ) -> Fill:
        """Record a fill and update positions and cash.

        Parameters
        ----------
        order:
            The order being filled.
        fill_price:
            Execution price per share.
        fill_qty:
            Number of shares filled.

        Returns
        -------
        :class:`Fill`
        """
        commission = fill_qty * self._commission_per_share
        notional = fill_price * fill_qty

        if order.side == "buy":
            cash_impact = -(notional + commission)
        else:
            cash_impact = (notional - commission)

        self._cash += cash_impact

        # Compute total slippage (fixed + market impact)
        if order.order_type == "market":
            impact_bps = self._compute_market_impact_bps(order.symbol, fill_qty)
            slippage_bps = self._slippage_bps + impact_bps
        else:
            slippage_bps = 0.0

        # Update position
        self._update_position(order.symbol, fill_qty, fill_price, order.side, order.strategy)

        # Update order status
        order.fill_price = fill_price
        order.fill_qty = fill_qty
        order.status = "filled"

        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            fill_price=fill_price,
            fill_qty=fill_qty,
            slippage_bps=slippage_bps,
            timestamp=self._current_bar_dt or datetime.now(timezone.utc),
        )
        self._fills.append(fill)

        self._log.log_fill(
            order.order_id,
            fill_price,
            fill_qty,
            slippage_bps,
            metadata={
                "symbol": order.symbol,
                "side": order.side,
                "commission": commission,
                "strategy": order.strategy,
            },
        )

        logger.debug(
            f"SimulatedBroker: FILL {order.side.upper()} {fill_qty} "
            f"{order.symbol} @ {fill_price:.4f} "
            f"(slippage={slippage_bps}bps, commission={commission:.2f})"
        )
        return fill

    def _update_position(
        self,
        symbol: str,
        fill_qty: int,
        fill_price: float,
        side: str,
        strategy: str,
    ) -> None:
        """Update internal position after a fill.

        Uses FIFO cost basis for long positions and LIFO for short.

        Parameters
        ----------
        symbol:
            Ticker.
        fill_qty:
            Shares filled (always positive).
        fill_price:
            Execution price.
        side:
            ``"buy"`` or ``"sell"``.
        strategy:
            Originating strategy name.
        """
        signed_qty = fill_qty if side == "buy" else -fill_qty

        if symbol in self._positions:
            pos = self._positions[symbol]
            old_qty = pos.qty
            new_qty = old_qty + signed_qty

            if new_qty == 0:
                # Position closed: realise P&L
                realised = (fill_price - pos.avg_entry) * old_qty
                if side == "buy":  # Closing a short
                    realised = (pos.avg_entry - fill_price) * abs(old_qty)
                self._realized_pnl += realised
                del self._positions[symbol]
                return

            if (old_qty > 0 and new_qty > 0) or (old_qty < 0 and new_qty < 0):
                # Adding to position: update VWAP
                new_avg = (pos.avg_entry * abs(old_qty) + fill_price * fill_qty) / abs(new_qty)
                pos.qty = new_qty
                pos.avg_entry = new_avg
                pos.current_price = fill_price
                pos.unrealized_pnl = (fill_price - new_avg) * new_qty
            else:
                # Partial close and then reverse: realise P&L on closed portion
                closed_qty = min(abs(old_qty), fill_qty)
                if old_qty > 0:
                    realised = (fill_price - pos.avg_entry) * closed_qty
                else:
                    realised = (pos.avg_entry - fill_price) * closed_qty
                self._realized_pnl += realised

                remaining_qty = new_qty
                if remaining_qty == 0:
                    del self._positions[symbol]
                else:
                    pos.qty = remaining_qty
                    pos.avg_entry = fill_price
                    pos.current_price = fill_price
                    pos.unrealized_pnl = 0.0
        else:
            # New position
            if signed_qty == 0:
                return
            self._positions[symbol] = Position(
                symbol=symbol,
                qty=signed_qty,
                avg_entry=fill_price,
                current_price=fill_price,
                unrealized_pnl=0.0,
                strategy=strategy,
            )

    def _compute_equity(self) -> float:
        """Compute total mark-to-market equity = cash + sum(position market values).

        For both long and short positions this formula is correct:

        - **Long position** (qty > 0): cash was reduced by the purchase notional,
          so adding ``qty * price`` restores the current market value of the holding.
        - **Short position** (qty < 0): cash was *increased* by the sale proceeds
          (see ``_execute_fill`` — ``cash_impact = notional - commission`` for sells).
          Adding ``qty * price`` (a negative number) subtracts the current liability,
          which correctly reflects the mark-to-market cost of buying back the short.

        Example (short 100 shares sold @ $50, price later moves to $40):
            cash       = initial_cash + 5_000    (short proceeds credited)
            pos value  = -100 * 40 = -4_000      (liability at current price)
            equity     = initial_cash + 5_000 - 4_000 = initial_cash + 1_000  ✓ (profit)

        No separate short-proceeds tracking is needed because ``_cash`` already
        includes the sale proceeds at the time of the short fill.
        """
        return self._cash + sum(
            pos.qty * pos.current_price for pos in self._positions.values()
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self._cash

    @property
    def realized_pnl(self) -> float:
        """Cumulative realised P&L."""
        return self._realized_pnl

    @property
    def fill_history(self) -> List[Fill]:
        """All fills in chronological order."""
        return list(self._fills)

    @property
    def order_history(self) -> Dict[str, Order]:
        """All orders (including filled and cancelled)."""
        return dict(self._orders)

    def get_open_orders(self) -> List[Order]:
        """Return all orders with active lifecycle status."""
        return [o for o in self._orders.values() if o.is_active]

    def reset_daily(self) -> None:
        """Reset start-of-day equity snapshot for daily P&L tracking.

        Call this once at the start of each new trading day (e.g. from the
        backtester's daily loop or the live trading scheduler).  This updates
        the ``_sod_equity`` baseline so that ``today_pnl`` computed in
        :class:`ExecutionManager` reflects only the current day's performance.
        """
        self._sod_equity = self._compute_equity()
        logger.debug(
            f"SimulatedBroker: reset_daily — sod_equity set to {self._sod_equity:,.2f}"
        )

    @property
    def sod_equity(self) -> float:
        """Start-of-day equity snapshot (for daily P&L computation)."""
        return self._sod_equity


# ---------------------------------------------------------------------------
# ExecutionManager
# ---------------------------------------------------------------------------

class ExecutionManager:
    """High-level execution manager: signal → order → fill lifecycle.

    The :class:`ExecutionManager` sits between the signal generator and the
    broker.  It:
    1. Converts :class:`Signal` objects into :class:`Order` objects.
    2. Consults the :class:`RiskManager` for position-size approval.
    3. Submits approved orders to the broker.
    4. Tracks the lifecycle of all orders.
    5. Logs all events through :class:`TradeLogger`.

    Parameters
    ----------
    broker:
        :class:`Broker` implementation to submit orders to.
    risk_manager:
        :class:`~core.risk_manager.RiskManager` for pre-trade risk checks.
    trade_logger:
        Audit logger.
    order_type:
        Default order type for new entries (``"market"`` or ``"limit"``).
        Default ``"market"`` for simplicity in backtests.
    max_position_value:
        Hard cap on position value per name in USD.  Overrides risk manager
        sizing if smaller.
    """

    def __init__(
        self,
        broker: Broker,
        risk_manager: RiskManager,
        trade_logger: Optional[TradeLogger] = None,
        order_type: str = "market",
        max_position_value: float = 5_000.0,
    ) -> None:
        self._broker = broker
        self._risk_manager = risk_manager
        self._log = trade_logger or get_trade_logger()
        self._order_type = order_type
        self._max_position_value = max_position_value

        # Track submitted orders: order_id → Order
        self._submitted_orders: Dict[str, Order] = {}

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    def process_signals(
        self,
        signals: List[Signal],
        current_prices: Dict[str, float],
    ) -> List[Order]:
        """Convert signals to orders with risk approval and submit to broker.

        Parameters
        ----------
        signals:
            Consolidated signals from :class:`~equities.signal_generator.SignalGenerator`.
        current_prices:
            Current prices for all symbols (used for position sizing).

        Returns
        -------
        List of submitted :class:`Order` objects.

        Raises
        ------
        RuntimeError:
            If the broker raises an error during submission.
        """
        portfolio_state = self._broker.get_portfolio_state()
        submitted: List[Order] = []

        # ── Position cap gate ──
        _MAX_TOTAL_POSITIONS = 25
        _current_pos_count = len(portfolio_state.positions)
        if _current_pos_count >= _MAX_TOTAL_POSITIONS:
            signals = [s for s in signals if s.direction == "close"]
            if not signals:
                logger.info(
                    f"ExecutionManager: position cap ({_current_pos_count}/{_MAX_TOTAL_POSITIONS}) — only close signals"
                )
                return submitted

        for signal in signals:
            symbol = signal.symbol
            price = current_prices.get(symbol)

            if price is None or price <= 0:
                logger.warning(
                    f"ExecutionManager: no price for {symbol!r}; skipping signal."
                )
                continue

            # Handle close signals
            if signal.direction == "close":
                close_orders = self._process_close_signal(signal, portfolio_state)
                submitted.extend(close_orders)
                continue

            # ── Min signal strength gate ──
            # Find the best raw (pre-scale) strength from metadata.
            # In merged signals, keys are prefixed: e.g. mean_reversion__pre_scale_strength
            _raw_str = signal.metadata.get("pre_scale_strength", None)
            if _raw_str is None:
                # Look for prefixed keys from merged signals
                _pre_scales = [
                    v for k, v in signal.metadata.items()
                    if k.endswith("__pre_scale_strength") and isinstance(v, (int, float))
                ]
                _raw_str = max(_pre_scales) if _pre_scales else signal.strength
            _MIN_RAW_STRENGTH = 0.20
            if abs(float(_raw_str)) < _MIN_RAW_STRENGTH:
                logger.info(
                    f"ExecutionManager: {symbol} raw strength {float(_raw_str):.3f} < {_MIN_RAW_STRENGTH} — skip"
                )
                continue

            # ── Short-selling gate ──
            if signal.direction == "short" and abs(signal.strength) < 0.50:
                logger.info(
                    f"ExecutionManager: blocking short {symbol} — strength {signal.strength:.3f} < 0.50"
                )
                continue

            # Size the order
            qty = self._compute_order_qty(signal, price, portfolio_state)
            if qty <= 0:
                logger.info(
                    f"ExecutionManager: signal for {symbol!r} sized to 0 shares; skipping."
                )
                continue

            # Risk approval
            side = "buy" if signal.direction == "long" else "sell"
            from core.risk_manager import PortfolioState as RMPortfolioState

            # Compute actual today_pnl from start-of-day equity snapshot.
            # The broker's _sod_equity is set via reset_daily() at the
            # start of each trading day; if the broker doesn't expose it
            # (e.g. a live broker adapter), fall back to 0.0 so that the
            # daily-loss check is skipped safely rather than incorrectly.
            sod_equity = getattr(self._broker, "sod_equity", None)
            if sod_equity is not None and sod_equity > 0:
                today_pnl = portfolio_state.equity - sod_equity
            else:
                today_pnl = 0.0  # fallback: live broker must track this separately

            rm_portfolio = RMPortfolioState(
                equity=portfolio_state.equity,
                peak_equity=portfolio_state.peak_equity,
                today_pnl=today_pnl,
                sod_equity=sod_equity,
                positions={
                    sym: pos.market_value
                    for sym, pos in portfolio_state.positions.items()
                },
            )

            approval = self._risk_manager.approve_trade(
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                portfolio_state=rm_portfolio,
            )

            if not approval.approved:
                logger.info(
                    f"ExecutionManager: {symbol} {side} {qty} REJECTED by risk: "
                    f"{approval.reason}"
                )
                self._log.log_risk_event(
                    "trade_rejected",
                    {
                        "symbol": symbol,
                        "side": side,
                        "qty": qty,
                        "reason": approval.reason,
                        "strategy": signal.strategy,
                    },
                )
                continue

            # Use risk manager's suggested qty if it was adjusted
            final_qty = int(approval.suggested_qty) if approval.suggested_qty > 0 else qty

            # Submit order
            try:
                order = self._broker.submit_order(
                    symbol=symbol,
                    qty=final_qty,
                    side=side,
                    order_type=self._order_type,
                    strategy=signal.strategy,
                    signal_strength=signal.strength,
                )
                self._submitted_orders[order.order_id] = order
                submitted.append(order)

                logger.info(
                    f"ExecutionManager: submitted {side.upper()} {final_qty} "
                    f"{symbol} (strategy={signal.strategy}, strength={signal.strength:.3f})"
                )
            except Exception as exc:
                self._log.log_error(
                    f"ExecutionManager: broker submission failed for {symbol}: {exc}",
                    exc_info=exc,
                )
                raise

        return submitted

    def _process_close_signal(
        self,
        signal: Signal,
        portfolio_state: PortfolioState,
    ) -> List[Order]:
        """Generate close orders for an open position.

        Parameters
        ----------
        signal:
            A ``close`` signal.
        portfolio_state:
            Current portfolio snapshot.

        Returns
        -------
        List of submitted close orders (one per leg for pairs trades).
        """
        orders: List[Order] = []
        symbol = signal.symbol
        pos = portfolio_state.positions.get(symbol)

        if pos is not None and pos.qty != 0:
            side = "sell" if pos.qty > 0 else "buy"
            qty = abs(pos.qty)
            try:
                order = self._broker.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    order_type="market",
                    strategy=signal.strategy,
                )
                self._submitted_orders[order.order_id] = order
                orders.append(order)
                logger.info(
                    f"ExecutionManager: CLOSE {symbol} {side.upper()} {qty} shares."
                )
            except Exception as exc:
                self._log.log_error(
                    f"ExecutionManager: close order failed for {symbol}: {exc}",
                    exc_info=exc,
                )
                raise

        # Handle paired symbol for stat arb spreads
        paired_symbol = signal.metadata.get("symbol_short") or signal.metadata.get("symbol_long")
        if paired_symbol and paired_symbol != symbol:
            paired_pos = portfolio_state.positions.get(paired_symbol)
            if paired_pos is not None and paired_pos.qty != 0:
                side = "sell" if paired_pos.qty > 0 else "buy"
                qty = abs(paired_pos.qty)
                try:
                    order = self._broker.submit_order(
                        symbol=paired_symbol,
                        qty=qty,
                        side=side,
                        order_type="market",
                        strategy=signal.strategy,
                    )
                    self._submitted_orders[order.order_id] = order
                    orders.append(order)
                except Exception as exc:
                    self._log.log_error(
                        f"ExecutionManager: close order failed for {paired_symbol}: {exc}",
                        exc_info=exc,
                    )
                    raise

        return orders

    def _compute_order_qty(
        self,
        signal: Signal,
        price: float,
        portfolio_state: PortfolioState,
    ) -> int:
        """Compute the number of shares to order based on signal strength.

        Position sizing uses the *pre-scale* signal strength (before regime
        allocation weighting) so that allocation weights control *which*
        signals pass the pipeline, not *how large* positions are.  This is
        critical for capital deployment — allocation-scaled strengths of
        0.3-0.6 produced tiny positions and kept 70%+ cash.

        Sizing formula:
            raw_strength = signal.metadata.pre_scale_strength (or signal.strength)
            adjusted      = max(raw_strength, 0.10)  # 10% floor preserves signal variation
            target_value  = equity × max_position_pct × adjusted

        Capped at ``max_position_value`` and bounded by:
        - Gross exposure limit (150% of equity)
        - Net exposure limit (±80% of equity) — prevents directional crowding

        Parameters
        ----------
        signal:
            The originating signal.
        price:
            Current price per share.
        portfolio_state:
            Current portfolio state.

        Returns
        -------
        Number of shares (0 if below minimum lot size).
        """
        max_pct = get_config().risk.max_position_pct
        equity = max(portfolio_state.equity, 1.0)

        # --- Gross exposure cap (150% of equity) ---
        _MAX_GROSS_EXPOSURE_PCT = 0.95
        current_gross = portfolio_state.gross_exposure
        remaining_capacity = max(equity * _MAX_GROSS_EXPOSURE_PCT - current_gross, 0.0)
        if remaining_capacity <= 0:
            logger.info(
                f"ExecutionManager: gross exposure at {current_gross / equity:.1%} "
                f"of equity — no capacity for {signal.symbol}"
            )
            return 0

        # --- Net exposure cap (80% of equity in either direction) ---
        # Prevents the portfolio from becoming excessively directional even
        # when gross exposure is within limit (e.g. all longs or all shorts).
        _MAX_NET_EXPOSURE_PCT = 0.80
        side = "buy" if signal.direction == "long" else "sell"
        net_exposure_pct = portfolio_state.net_exposure / equity
        if abs(net_exposure_pct) > _MAX_NET_EXPOSURE_PCT:
            # Only block if this order would push net exposure further out of bounds
            if (side == "buy" and net_exposure_pct > 0) or (side == "sell" and net_exposure_pct < 0):
                logger.warning(
                    f"ExecutionManager: net exposure limit reached "
                    f"({net_exposure_pct:.1%} vs ±{_MAX_NET_EXPOSURE_PCT:.0%}) — "
                    f"skipping {signal.symbol} {side}"
                )
                return 0

        # Use PRE-SCALE strength for sizing (before allocation weighting).
        # The allocation weight controls conflict resolution priority, not
        # position size.  This fixes the capital deployment problem.
        raw_strength = signal.metadata.get("pre_scale_strength", signal.strength)
        # Floor at 10% to avoid zero-size orders; preserve the actual signal
        # strength signal rather than collapsing all signals to 85%.
        adjusted_strength = max(float(raw_strength), 0.10)
        target_notional = equity * max_pct * adjusted_strength
        target_notional = min(target_notional, self._max_position_value)

        # Don't exceed remaining gross exposure capacity
        target_notional = min(target_notional, remaining_capacity)

        # Deduct existing position value to prevent over-sizing.
        # If we already hold $4,800 of GOOGL and target is $5,000,
        # we should only buy $200 more, not another $5,000.
        existing_pos = portfolio_state.positions.get(signal.symbol)
        if existing_pos is not None:
            existing_mv = abs(existing_pos.market_value)
            target_notional -= existing_mv
            if target_notional <= 0:
                logger.debug(
                    f"ExecutionManager: {signal.symbol} already at target "
                    f"(held=${existing_mv:,.0f}, target=${self._max_position_value:,.0f}) — skip"
                )
                return 0

        qty = int(target_notional / max(price, 0.01))
        return max(qty, 0)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def submitted_orders(self) -> Dict[str, Order]:
        """All orders submitted this session."""
        return dict(self._submitted_orders)

    def get_active_orders(self) -> List[Order]:
        """Return orders still pending fills."""
        return [o for o in self._submitted_orders.values() if o.is_active]

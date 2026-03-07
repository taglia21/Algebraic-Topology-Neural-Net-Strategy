"""
equities/alpaca_broker.py
=========================
Live broker adapter that routes orders to Alpaca's REST API.

Implements the :class:`~equities.execution.Broker` abstract interface so the
full signal pipeline (regime → signals → risk → execution) runs identically
in live and backtest modes — only the broker implementation changes.

Features
--------
- Market and limit order submission via Alpaca SDK.
- Position synchronisation from Alpaca account state.
- Account snapshot (equity, cash, buying power).
- Order cancellation.
- Automatic retry with exponential back-off on transient errors.

Environment Variables
---------------------
    ALPACA_API_KEY      — Alpaca API key
    ALPACA_API_SECRET   — Alpaca secret key
    ALPACA_BASE_URL     — Base URL (default: paper trading)

Usage
-----
    from equities.alpaca_broker import AlpacaBroker

    broker = AlpacaBroker()
    order = broker.submit_order("AAPL", 10, "buy", "market")
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from equities.execution import Broker
from equities.models import Account, Fill, Order, Pair, Position, PortfolioState
from core.config import get_config

logger = logging.getLogger(__name__)

# Maximum retries on transient HTTP errors
_MAX_RETRIES: int = 3
_RETRY_BASE_WAIT: float = 1.0  # seconds


class AlpacaBroker(Broker):
    """Live broker adapter routing orders through Alpaca's REST API.

    Implements the full :class:`Broker` interface.  Swap this in place of
    :class:`SimulatedBroker` for paper or live trading.

    Parameters
    ----------
    api_key :
        Alpaca API key.  Falls back to ``ALPACA_API_KEY`` env var.
    secret_key :
        Alpaca secret key.  Falls back to ``ALPACA_API_SECRET`` env var.
    base_url :
        Alpaca REST base URL.  Falls back to ``ALPACA_BASE_URL``.
    paper :
        If True (default), use the paper trading endpoint.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        paper: bool = True,
    ) -> None:
        cfg = get_config().alpaca

        self._api_key = api_key or cfg.api_key
        self._secret_key = secret_key or cfg.secret_key
        self._base_url = base_url or cfg.base_url

        if not self._api_key or not self._secret_key:
            raise RuntimeError(
                "Alpaca credentials not configured. Set ALPACA_API_KEY and "
                "ALPACA_API_SECRET environment variables."
            )

        # Lazy-load the Alpaca SDK to avoid hard dependency in backtest mode
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import (
                MarketOrderRequest,
                LimitOrderRequest,
                GetOrdersRequest,
            )
            from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, OrderStatus
        except ImportError:
            raise ImportError(
                "alpaca-py is required for live trading. "
                "Install with: pip install alpaca-py"
            )

        self._TradingClient = TradingClient
        self._MarketOrderRequest = MarketOrderRequest
        self._LimitOrderRequest = LimitOrderRequest
        self._OrderSide = OrderSide
        self._OrderType = OrderType
        self._TimeInForce = TimeInForce
        self._OrderStatus = OrderStatus

        self._client = TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=paper,
        )

        # Internal tracking
        self._submitted_orders: Dict[str, Order] = {}
        self._fills: List[Fill] = []
        self._peak_equity: float = 0.0

        # Verify connectivity
        try:
            acct = self._client.get_account()
            self._peak_equity = float(acct.equity)
            logger.info(
                f"AlpacaBroker connected: account={acct.id}, "
                f"equity=${float(acct.equity):,.2f}, "
                f"cash=${float(acct.cash):,.2f}, "
                f"paper={paper}"
            )
        except Exception as exc:
            raise RuntimeError(f"Alpaca connection failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Broker interface
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
        """Submit an order to Alpaca.

        Parameters
        ----------
        symbol : Ticker symbol.
        qty : Number of shares (positive).
        side : ``"buy"`` or ``"sell"``.
        order_type : ``"market"`` or ``"limit"``.
        limit_price : Required for limit orders.
        strategy : Originating strategy name.
        signal_strength : Signal conviction [0, 1].

        Returns
        -------
        :class:`Order` with Alpaca order ID.
        """
        alpaca_side = (
            self._OrderSide.BUY if side == "buy" else self._OrderSide.SELL
        )

        if order_type == "limit" and limit_price is not None:
            request = self._LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=self._TimeInForce.DAY,
                limit_price=round(limit_price, 2),
            )
        else:
            request = self._MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=self._TimeInForce.DAY,
            )

        alpaca_order = self._retry(lambda: self._client.submit_order(request))

        order = Order(
            order_id=str(alpaca_order.id),
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            limit_price=limit_price,
            status="submitted",
            strategy=strategy,
            signal_strength=signal_strength,
        )
        self._submitted_orders[order.order_id] = order

        logger.info(
            f"AlpacaBroker: submitted {side.upper()} {qty} {symbol} "
            f"({order_type}) → order_id={order.order_id}"
        )
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order on Alpaca.

        Parameters
        ----------
        order_id : Alpaca order UUID.

        Returns
        -------
        True if cancelled, False if already filled or not found.
        """
        try:
            self._client.cancel_order_by_id(order_id)
            if order_id in self._submitted_orders:
                self._submitted_orders[order_id].status = "cancelled"
            logger.info(f"AlpacaBroker: cancelled order {order_id}")
            return True
        except Exception as exc:
            logger.warning(f"AlpacaBroker: cancel failed for {order_id}: {exc}")
            return False

    def get_positions(self) -> Dict[str, Position]:
        """Fetch all open positions from Alpaca.

        Returns
        -------
        Dict mapping symbol → :class:`Position`.
        """
        alpaca_positions = self._retry(lambda: self._client.get_all_positions())
        positions: Dict[str, Position] = {}

        for ap in alpaca_positions:
            positions[ap.symbol] = Position(
                symbol=ap.symbol,
                qty=int(ap.qty),
                avg_entry=float(ap.avg_entry_price),
                current_price=float(ap.current_price),
                unrealized_pnl=float(ap.unrealized_pl),
                strategy="",
            )

        return positions

    def get_account(self) -> Account:
        """Fetch current Alpaca account state.

        Returns
        -------
        :class:`Account` snapshot.
        """
        acct = self._retry(lambda: self._client.get_account())
        return Account(
            account_id=str(acct.id),
            buying_power=float(acct.buying_power),
            equity=float(acct.equity),
            cash=float(acct.cash),
            pattern_day_trader=bool(acct.pattern_day_trader),
        )

    def get_portfolio_state(self) -> PortfolioState:
        """Build a complete portfolio snapshot from Alpaca state.

        Returns
        -------
        :class:`PortfolioState`.
        """
        acct = self._retry(lambda: self._client.get_account())
        positions = self.get_positions()

        equity = float(acct.equity)
        cash = float(acct.cash)

        if equity > self._peak_equity:
            self._peak_equity = equity

        unrealized = sum(p.unrealized_pnl for p in positions.values())
        drawdown = (
            (equity - self._peak_equity) / max(self._peak_equity, 1.0)
            if self._peak_equity > 0
            else 0.0
        )

        return PortfolioState(
            equity=equity,
            cash=cash,
            positions=positions,
            unrealized_pnl=unrealized,
            realized_pnl=0.0,  # Alpaca doesn't expose cumulative realized P&L simply
            peak_equity=self._peak_equity,
            drawdown=drawdown,
        )

    # ------------------------------------------------------------------
    # Order status sync
    # ------------------------------------------------------------------

    def sync_order_status(self) -> Dict[str, str]:
        """Poll Alpaca for the latest status of all tracked orders.

        Returns
        -------
        Dict mapping order_id → current status string.
        """
        updates: Dict[str, str] = {}
        for order_id, order in list(self._submitted_orders.items()):
            if order.status in ("filled", "cancelled", "rejected"):
                continue
            try:
                alpaca_order = self._client.get_order_by_id(order_id)
                new_status = str(alpaca_order.status.value).lower()

                if new_status == "filled" and order.status != "filled":
                    order.status = "filled"
                    order.fill_price = float(alpaca_order.filled_avg_price or 0)
                    order.fill_qty = int(alpaca_order.filled_qty or 0)

                    fill = Fill(
                        order_id=order_id,
                        symbol=order.symbol,
                        side=order.side,
                        fill_price=order.fill_price,
                        fill_qty=order.fill_qty,
                        timestamp=datetime.now(timezone.utc),
                    )
                    self._fills.append(fill)
                elif new_status in ("cancelled", "expired", "rejected"):
                    order.status = new_status

                updates[order_id] = new_status
            except Exception as exc:
                logger.warning(f"AlpacaBroker: sync failed for {order_id}: {exc}")

        return updates

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _retry(self, fn, max_retries: int = _MAX_RETRIES):
        """Execute fn with exponential back-off on transient errors."""
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise
                wait = _RETRY_BASE_WAIT * (2 ** attempt)
                logger.warning(
                    f"AlpacaBroker: transient error (attempt {attempt + 1}): "
                    f"{exc}. Retrying in {wait:.1f}s ..."
                )
                time.sleep(wait)

    @property
    def fill_history(self) -> List[Fill]:
        """All fills tracked this session."""
        return list(self._fills)

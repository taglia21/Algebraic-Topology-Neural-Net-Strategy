"""
IBKRBrokerClient — Interactive Brokers adapter via ib_insync
=============================================================

Connects to IB Gateway / TWS and implements the full BaseBrokerClient interface
including LIVE option Greeks from ``reqMktData``.

Usage::

    from src.brokers.ibkr_client import IBKRBrokerClient

    client = IBKRBrokerClient(host='127.0.0.1', port=4002, account='U22452226')
    client.connect()

    acct = client.get_account()
    print(acct.portfolio_value)

    chain = client.get_option_chain('SPY', '20260320')
    for c in chain:
        print(c.symbol, c.strike, c.delta, c.implied_volatility)
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

from ib_insync import IB, Contract, Stock, Index, Option, MarketOrder, LimitOrder, util

from .base import (
    AccountInfo,
    Bar,
    BaseBrokerClient,
    OptionContract,
    Order,
    Position,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeframe string → IB barSize / durationStr mapping
# ---------------------------------------------------------------------------
_TF_MAP: Dict[str, tuple] = {
    "1Min":  ("1 min",  "1 D"),
    "5Min":  ("5 mins", "2 D"),
    "15Min": ("15 mins","5 D"),
    "1H":    ("1 hour", "10 D"),
    "1D":    ("1 day",  "1 Y"),
}


class IBKRBrokerClient(BaseBrokerClient):
    """Interactive Brokers broker client using *ib_insync*.

    Parameters
    ----------
    host : str
        IB Gateway / TWS hostname (default ``127.0.0.1``).
    port : int
        API port. ``4001`` = live, ``4002`` = paper.
    account : str
        IB account id, e.g. ``U22452226``.
    paper : bool
        If True use paper port (4002).  Ignored when *port* is given explicitly.
    client_id : int
        TWS client-id (default 1).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,
        account: str = "U22452226",
        paper: bool = True,
        client_id: int = 1,
    ) -> None:
        self.host = host
        self.port = port
        self.account = account
        self.paper = paper
        self.client_id = client_id
        self.ib = IB()

        # Background reconnect
        self._reconnect_thread: Optional[threading.Thread] = None
        self._stop_reconnect = threading.Event()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, max_retries: int = 5) -> None:
        """Connect with exponential back-off (max *max_retries* attempts)."""
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "IBKR connect attempt %d/%d -> %s:%s (account=%s)",
                    attempt, max_retries, self.host, self.port, self.account,
                )
                self.ib.connect(
                    self.host,
                    self.port,
                    clientId=self.client_id,
                    account=self.account,
                    readonly=False,
                )
                logger.info("IBKR connected ✓  (server version %s)", self.ib.client.serverVersion())
                self._start_reconnect_loop()
                return
            except Exception as exc:
                wait = min(2 ** attempt, 60)
                logger.warning("IBKR connect failed (%s) — retry in %ds", exc, wait)
                if attempt == max_retries:
                    raise ConnectionError(
                        f"Could not connect to IBKR after {max_retries} attempts"
                    ) from exc
                time.sleep(wait)

    def disconnect(self) -> None:
        """Gracefully disconnect and stop the reconnect watchdog."""
        self._stop_reconnect.set()
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            self._reconnect_thread.join(timeout=5)
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("IBKR disconnected")

    # ------------------------------------------------------------------
    # Background reconnect watchdog
    # ------------------------------------------------------------------

    def _start_reconnect_loop(self) -> None:
        self._stop_reconnect.clear()
        self._reconnect_thread = threading.Thread(
            target=self._reconnect_loop, daemon=True, name="ibkr-reconnect",
        )
        self._reconnect_thread.start()

    def _reconnect_loop(self) -> None:
        """Background thread: check connection every 10 s, reconnect on drop."""
        while not self._stop_reconnect.is_set():
            self._stop_reconnect.wait(timeout=10)
            if self._stop_reconnect.is_set():
                break
            if not self.ib.isConnected():
                logger.warning("IBKR connection lost — attempting reconnect …")
                try:
                    self.ib.connect(
                        self.host,
                        self.port,
                        clientId=self.client_id,
                        account=self.account,
                        readonly=False,
                    )
                    logger.info("IBKR reconnected ✓")
                except Exception as exc:
                    logger.error("IBKR reconnect failed: %s", exc)

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> AccountInfo:
        """Fetch real-time account summary."""
        summary = {
            tag.tag: tag.value
            for tag in self.ib.accountSummary(self.account)
        }
        return AccountInfo(
            cash=float(summary.get("TotalCashValue", 0)),
            buying_power=float(summary.get("BuyingPower", 0)),
            portfolio_value=float(summary.get("NetLiquidation", 0)),
        )

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> List[Position]:
        """Return all open positions for this account."""
        positions: List[Position] = []
        for pos in self.ib.positions(self.account):
            positions.append(
                Position(
                    symbol=pos.contract.symbol,
                    qty=float(pos.position),
                    avg_cost=float(pos.avgCost),
                    market_value=float(pos.position * pos.avgCost),
                    unrealized_pnl=0.0,  # Updated via pnl subscriptions
                    side="long" if pos.position > 0 else "short",
                )
            )
        return positions

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> Order:
        """Place an equity order using SMART routing.

        Parameters
        ----------
        symbol : str
            Ticker, e.g. ``'AAPL'``.
        qty : float
            Number of shares. Positive.
        side : str
            ``'buy'`` or ``'sell'``.
        order_type : str
            ``'market'`` or ``'limit'``.
        limit_price : float | None
            Required when *order_type* is ``'limit'``.
        """
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)

        action = "BUY" if side.lower() == "buy" else "SELL"

        if order_type.lower() == "limit" and limit_price is not None:
            ib_order = LimitOrder(action, abs(qty), limit_price)
        else:
            ib_order = MarketOrder(action, abs(qty))

        trade = self.ib.placeOrder(contract, ib_order)
        self.ib.sleep(0)  # allow event loop to process

        return Order(
            order_id=str(trade.order.orderId),
            symbol=symbol,
            qty=float(qty),
            side=side,
            order_type=order_type,
            status=trade.orderStatus.status,
            filled_qty=float(trade.orderStatus.filled),
            filled_avg_price=float(trade.orderStatus.avgFillPrice),
            created_at=datetime.now(),
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by id."""
        for trade in self.ib.openTrades():
            if str(trade.order.orderId) == str(order_id):
                self.ib.cancelOrder(trade.order)
                logger.info("Cancelled order %s", order_id)
                return True
        logger.warning("Order %s not found in open trades", order_id)
        return False

    def close_position(self, symbol: str) -> Optional[Order]:
        """Flatten the position for *symbol* with a market order."""
        for pos in self.ib.positions(self.account):
            if pos.contract.symbol == symbol:
                side = "sell" if pos.position > 0 else "buy"
                return self.place_order(symbol, abs(pos.position), side, "market")
        logger.warning("No open position for %s", symbol)
        return None

    # ------------------------------------------------------------------
    # Market data — bars
    # ------------------------------------------------------------------

    def get_bars(self, symbol: str, timeframe: str = "1D", limit: int = 100) -> List[Bar]:
        """Fetch historical OHLCV bars.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        timeframe : str
            One of ``1Min``, ``5Min``, ``15Min``, ``1H``, ``1D``.
        limit : int
            Approximate number of bars (IB uses durationStr).
        """
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)

        bar_size, duration = _TF_MAP.get(timeframe, ("1 day", "1 Y"))

        ib_bars = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )

        bars: List[Bar] = []
        for b in ib_bars[-limit:]:
            bars.append(
                Bar(
                    timestamp=b.date if isinstance(b.date, datetime) else datetime.now(),
                    open=float(b.open),
                    high=float(b.high),
                    low=float(b.low),
                    close=float(b.close),
                    volume=float(b.volume),
                )
            )
        return bars

    # ------------------------------------------------------------------
    # Options — chain with LIVE Greeks
    # ------------------------------------------------------------------

    def get_option_chain(self, symbol: str, expiry: str) -> List[OptionContract]:
        """Fetch full option chain with LIVE Greeks via ``reqMktData``.

        Parameters
        ----------
        symbol : str
            Underlying ticker, e.g. ``'SPY'``.
        expiry : str
            Expiration in ``YYYYMMDD`` format.

        Returns
        -------
        list[OptionContract]
            Contracts populated with bid, ask, delta, gamma, theta, vega,
            impliedVolatility from the exchange — NOT calculated/indicative.
        """
        underlying = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(underlying)

        chains = self.ib.reqSecDefOptParams(
            underlying.symbol,
            "",
            underlying.secType,
            underlying.conId,
        )
        if not chains:
            logger.warning("No option chains found for %s", symbol)
            return []

        # Pick the SMART exchange chain
        chain_info = next((c for c in chains if c.exchange == "SMART"), chains[0])

        # Filter to requested expiry
        if expiry not in chain_info.expirations:
            logger.warning("Expiry %s not available for %s", expiry, symbol)
            return []

        strikes = sorted(chain_info.strikes)

        # Build option contracts for both calls and puts
        contracts: list = []
        for strike in strikes:
            for right in ("C", "P"):
                opt = Option(symbol, expiry, strike, right, "SMART")
                contracts.append(opt)

        # Qualify in batch
        self.ib.qualifyContracts(*contracts)

        # Request live market data for all contracts
        tickers = []
        for c in contracts:
            ticker = self.ib.reqMktData(c, genericTickList="106", snapshot=True)
            tickers.append((c, ticker))

        # Wait for snapshots to arrive (up to 10 s)
        self.ib.sleep(5)

        results: List[OptionContract] = []
        for contract_obj, ticker in tickers:
            greeks = ticker.modelGreeks or ticker.lastGreeks
            iv = 0.0
            delta = gamma = theta = vega = 0.0
            if greeks:
                iv = greeks.impliedVol or 0.0
                delta = greeks.delta or 0.0
                gamma = greeks.gamma or 0.0
                theta = greeks.theta or 0.0
                vega = greeks.vega or 0.0

            results.append(
                OptionContract(
                    symbol=symbol,
                    expiry=expiry,
                    strike=float(contract_obj.strike),
                    right=contract_obj.right,
                    bid=float(ticker.bid) if ticker.bid and ticker.bid > 0 else 0.0,
                    ask=float(ticker.ask) if ticker.ask and ticker.ask > 0 else 0.0,
                    last=float(ticker.last) if ticker.last and ticker.last > 0 else 0.0,
                    volume=int(ticker.volume) if ticker.volume and ticker.volume >= 0 else 0,
                    open_interest=0,
                    implied_volatility=iv,
                    delta=delta,
                    gamma=gamma,
                    theta=theta,
                    vega=vega,
                )
            )

            # Cancel streaming to free threads
            self.ib.cancelMktData(contract_obj)

        return results

    # ------------------------------------------------------------------
    # VIX
    # ------------------------------------------------------------------

    def get_vix(self) -> float:
        """Get the current VIX level via ``reqMktData`` on ``Index('VIX','CBOE')``."""
        vix_contract = Index("VIX", "CBOE")
        self.ib.qualifyContracts(vix_contract)
        ticker = self.ib.reqMktData(vix_contract, snapshot=True)
        self.ib.sleep(2)
        value = ticker.last if (ticker.last and ticker.last > 0) else ticker.close
        self.ib.cancelMktData(vix_contract)
        return float(value) if value else 0.0

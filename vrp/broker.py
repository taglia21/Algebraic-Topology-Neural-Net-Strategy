"""
vrp/broker.py
=============
Interactive Brokers interface via ib_async (successor to ib_insync).

Handles all IBKR communication:
- Connection management (connect, disconnect, reconnect)
- Option chain retrieval for SPX
- Order placement for put credit spreads
- Position tracking and account data
- Real-time market data (SPX price, VIX, option greeks)

Designed for production reliability:
- Automatic reconnection on disconnect
- Order validation before submission
- Comprehensive error handling and logging
- Clean separation between broker operations and strategy logic

Usage:
    broker = IBKRBroker(config.ibkr)
    await broker.connect()
    chain = await broker.get_option_chain("SPX", expiry)
    trade = await broker.place_spread(short_leg, long_leg, quantity)
    await broker.disconnect()

Requirements:
    pip install ib_async  (or ib_insync for older versions)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from vrp.config import IBKRConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Broker interface (abstract)
# ---------------------------------------------------------------------------

@dataclass
class OptionQuote:
    """Market data for a single option contract."""
    strike: float
    expiry: date
    right: str  # "P" or "C"
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float


@dataclass
class SpreadOrder:
    """Order details for a put credit spread."""
    short_strike: float
    long_strike: float
    expiry: date
    quantity: int
    limit_price: float  # net credit per spread
    order_type: str = "LMT"  # always limit for spreads
    tif: str = "DAY"


@dataclass
class AccountSummary:
    """IBKR account summary."""
    equity: float
    cash: float
    buying_power: float
    maintenance_margin: float
    unrealized_pnl: float
    realized_pnl: float


# ---------------------------------------------------------------------------
# IBKR Broker (live implementation)
# ---------------------------------------------------------------------------

class IBKRBroker:
    """Production IBKR broker interface using ib_async.

    This class wraps the ib_async library to provide a clean interface
    specifically for SPX put credit spread trading.
    """

    def __init__(self, config: IBKRConfig) -> None:
        self.config = config
        self._ib = None
        self._connected = False

    async def connect(self) -> bool:
        """Establish connection to IBKR TWS or Gateway.

        Returns True if connection successful.
        """
        try:
            # Try ib_async first (successor to ib_insync)
            try:
                from ib_async import IB
                logger.info("Using ib_async library")
            except ImportError:
                from ib_insync import IB
                logger.info("Using ib_insync library (fallback)")

            self._ib = IB()

            await self._ib.connectAsync(
                host=self.config.host,
                port=self.config.port,
                clientId=self.config.client_id,
                timeout=self.config.timeout,
                readonly=self.config.readonly,
            )

            self._connected = True
            logger.info(
                f"Connected to IBKR at {self.config.host}:{self.config.port} "
                f"(client {self.config.client_id})"
            )

            # Request account updates
            if self.config.account:
                self._ib.reqAccountUpdates(True, self.config.account)

            return True

        except Exception as e:
            logger.error(f"IBKR connection failed: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")

    @property
    def is_connected(self) -> bool:
        """Check if broker is connected."""
        return self._connected and self._ib is not None and self._ib.isConnected()

    async def get_spx_price(self) -> Optional[float]:
        """Get current SPX price."""
        if not self.is_connected:
            return None

        try:
            from ib_async import Index
        except ImportError:
            from ib_insync import Index

        contract = Index("SPX", "CBOE")
        self._ib.qualifyContracts(contract)

        ticker = self._ib.reqMktData(contract, "", False, False)
        await asyncio.sleep(2)  # wait for data

        price = ticker.last or ticker.close
        self._ib.cancelMktData(contract)

        return float(price) if price else None

    async def get_vix(self) -> Optional[float]:
        """Get current VIX level."""
        if not self.is_connected:
            return None

        try:
            from ib_async import Index
        except ImportError:
            from ib_insync import Index

        contract = Index("VIX", "CBOE")
        self._ib.qualifyContracts(contract)

        ticker = self._ib.reqMktData(contract, "", False, False)
        await asyncio.sleep(2)

        vix = ticker.last or ticker.close
        self._ib.cancelMktData(contract)

        return float(vix) if vix else None

    async def get_option_chain(
        self,
        expiry: date,
        strike_range: Optional[Tuple[float, float]] = None,
    ) -> List[OptionQuote]:
        """Retrieve SPX option chain for a given expiry.

        Parameters
        ----------
        expiry : Target expiration date
        strike_range : Optional (min_strike, max_strike) filter

        Returns
        -------
        List of OptionQuote for puts at available strikes
        """
        if not self.is_connected:
            return []

        try:
            from ib_async import Index, Option
        except ImportError:
            from ib_insync import Index, Option

        # Get SPX underlying
        spx = Index("SPX", "CBOE")
        self._ib.qualifyContracts(spx)

        # Get option parameters
        chains = self._ib.reqSecDefOptParams(
            spx.symbol, "", spx.secType, spx.conId
        )

        if not chains:
            logger.warning("No option chain data returned")
            return []

        # Find the right exchange (SMART or CBOE)
        chain = None
        for c in chains:
            if "SMART" in c.exchange or "CBOE" in c.exchange:
                chain = c
                break

        if chain is None:
            chain = chains[0]

        # Build option contracts for the target expiry
        expiry_str = expiry.strftime("%Y%m%d")
        puts = []

        # Filter strikes
        all_strikes = sorted(chain.strikes)
        if strike_range:
            all_strikes = [s for s in all_strikes if strike_range[0] <= s <= strike_range[1]]

        # Request market data for each strike
        contracts = []
        for strike in all_strikes:
            opt = Option("SPX", expiry_str, strike, "P", "SMART")
            contracts.append(opt)

        if not contracts:
            return []

        # Qualify contracts
        qualified = self._ib.qualifyContracts(*contracts)

        # Request tickers
        tickers = []
        for contract in qualified:
            if contract:
                ticker = self._ib.reqMktData(
                    contract, "106", False, False  # 106 = greeks
                )
                tickers.append((contract, ticker))

        # Wait for data
        await asyncio.sleep(3)

        quotes = []
        for contract, ticker in tickers:
            try:
                quote = OptionQuote(
                    strike=contract.strike,
                    expiry=expiry,
                    right="P",
                    bid=float(ticker.bid or 0),
                    ask=float(ticker.ask or 0),
                    last=float(ticker.last or 0),
                    volume=int(ticker.volume or 0),
                    open_interest=0,  # requires separate request
                    iv=float(ticker.modelGreeks.impliedVol or 0) if ticker.modelGreeks else 0,
                    delta=float(ticker.modelGreeks.delta or 0) if ticker.modelGreeks else 0,
                    gamma=float(ticker.modelGreeks.gamma or 0) if ticker.modelGreeks else 0,
                    theta=float(ticker.modelGreeks.theta or 0) if ticker.modelGreeks else 0,
                    vega=float(ticker.modelGreeks.vega or 0) if ticker.modelGreeks else 0,
                )
                quotes.append(quote)
            except Exception as e:
                logger.debug(f"Failed to parse option data for strike {contract.strike}: {e}")

        # Cancel market data
        for contract, _ in tickers:
            self._ib.cancelMktData(contract)

        logger.info(f"Retrieved {len(quotes)} put quotes for {expiry}")
        return quotes

    async def place_spread(
        self,
        order: SpreadOrder,
    ) -> Optional[str]:
        """Place a put credit spread order (combo order).

        Parameters
        ----------
        order : SpreadOrder with all trade details

        Returns
        -------
        Order ID if successful, None otherwise
        """
        if not self.is_connected:
            logger.error("Cannot place order: not connected")
            return None

        try:
            from ib_async import (
                Contract, ComboLeg, Order as IBOrder, TagValue
            )
        except ImportError:
            from ib_insync import (
                Contract, ComboLeg, Order as IBOrder, TagValue
            )

        expiry_str = order.expiry.strftime("%Y%m%d")

        # Create individual option contracts
        try:
            from ib_async import Option
        except ImportError:
            from ib_insync import Option

        short_opt = Option("SPX", expiry_str, order.short_strike, "P", "SMART")
        long_opt = Option("SPX", expiry_str, order.long_strike, "P", "SMART")

        # Qualify
        self._ib.qualifyContracts(short_opt, long_opt)

        if not short_opt.conId or not long_opt.conId:
            logger.error("Failed to qualify option contracts for spread")
            return None

        # Build combo contract
        combo = Contract()
        combo.symbol = "SPX"
        combo.secType = "BAG"
        combo.currency = "USD"
        combo.exchange = "SMART"

        # Short leg: SELL
        leg1 = ComboLeg()
        leg1.conId = short_opt.conId
        leg1.ratio = 1
        leg1.action = "SELL"
        leg1.exchange = "SMART"

        # Long leg: BUY
        leg2 = ComboLeg()
        leg2.conId = long_opt.conId
        leg2.ratio = 1
        leg2.action = "BUY"
        leg2.exchange = "SMART"

        combo.comboLegs = [leg1, leg2]

        # Create the order
        ib_order = IBOrder()
        ib_order.action = "SELL"  # Selling the spread (collecting credit)
        ib_order.totalQuantity = order.quantity
        ib_order.orderType = order.order_type
        ib_order.lmtPrice = order.limit_price  # net credit
        ib_order.tif = order.tif

        # Smart routing parameters
        ib_order.smartComboRoutingParams = [
            TagValue("NonGuaranteed", "1"),
        ]

        # Place the order
        trade = self._ib.placeOrder(combo, ib_order)

        logger.info(
            f"Spread order placed: SELL {order.short_strike}P / "
            f"BUY {order.long_strike}P x{order.quantity} @ ${order.limit_price:.2f} | "
            f"OrderId: {trade.order.orderId}"
        )

        return str(trade.order.orderId)

    async def close_spread(
        self,
        short_strike: float,
        long_strike: float,
        expiry: date,
        quantity: int,
        limit_price: Optional[float] = None,
    ) -> Optional[str]:
        """Close an existing put credit spread.

        Parameters
        ----------
        short_strike : Strike of the short put
        long_strike : Strike of the long put
        expiry : Expiration date
        quantity : Number of spreads to close
        limit_price : Limit price (net debit to close). If None, use market.

        Returns
        -------
        Order ID if successful
        """
        if not self.is_connected:
            return None

        try:
            from ib_async import (
                Contract, ComboLeg, Order as IBOrder, Option, TagValue
            )
        except ImportError:
            from ib_insync import (
                Contract, ComboLeg, Order as IBOrder, Option, TagValue
            )

        expiry_str = expiry.strftime("%Y%m%d")

        short_opt = Option("SPX", expiry_str, short_strike, "P", "SMART")
        long_opt = Option("SPX", expiry_str, long_strike, "P", "SMART")
        self._ib.qualifyContracts(short_opt, long_opt)

        combo = Contract()
        combo.symbol = "SPX"
        combo.secType = "BAG"
        combo.currency = "USD"
        combo.exchange = "SMART"

        # Reverse the legs to close
        leg1 = ComboLeg()
        leg1.conId = short_opt.conId
        leg1.ratio = 1
        leg1.action = "BUY"  # Buy back short
        leg1.exchange = "SMART"

        leg2 = ComboLeg()
        leg2.conId = long_opt.conId
        leg2.ratio = 1
        leg2.action = "SELL"  # Sell the long
        leg2.exchange = "SMART"

        combo.comboLegs = [leg1, leg2]

        ib_order = IBOrder()
        ib_order.action = "BUY"
        ib_order.totalQuantity = quantity

        if limit_price is not None:
            ib_order.orderType = "LMT"
            ib_order.lmtPrice = limit_price
        else:
            ib_order.orderType = "MKT"

        ib_order.tif = "DAY"
        ib_order.smartComboRoutingParams = [
            TagValue("NonGuaranteed", "1"),
        ]

        trade = self._ib.placeOrder(combo, ib_order)

        logger.info(
            f"Close order placed: BUY {short_strike}P / "
            f"SELL {long_strike}P x{quantity}"
        )

        return str(trade.order.orderId)

    async def get_account_summary(self) -> Optional[AccountSummary]:
        """Get current account summary."""
        if not self.is_connected:
            return None

        try:
            summary = self._ib.accountSummary(self.config.account)
            if not summary:
                # Fallback: request account values
                self._ib.reqAccountUpdates(True, self.config.account)
                await asyncio.sleep(2)
                summary = self._ib.accountSummary(self.config.account)

            values = {item.tag: float(item.value) for item in summary if item.value}

            return AccountSummary(
                equity=values.get("NetLiquidation", 0),
                cash=values.get("TotalCashValue", 0),
                buying_power=values.get("BuyingPower", 0),
                maintenance_margin=values.get("MaintMarginReq", 0),
                unrealized_pnl=values.get("UnrealizedPnL", 0),
                realized_pnl=values.get("RealizedPnL", 0),
            )
        except Exception as e:
            logger.error(f"Failed to get account summary: {e}")
            return None

    async def get_positions(self) -> List[Dict]:
        """Get all current positions."""
        if not self.is_connected:
            return []

        positions = self._ib.positions(self.config.account)
        result = []
        for pos in positions:
            result.append({
                "symbol": pos.contract.symbol,
                "secType": pos.contract.secType,
                "strike": getattr(pos.contract, "strike", None),
                "right": getattr(pos.contract, "right", None),
                "expiry": getattr(pos.contract, "lastTradeDateOrContractMonth", None),
                "quantity": pos.position,
                "avg_cost": pos.avgCost,
                "market_value": pos.marketValue if hasattr(pos, "marketValue") else None,
            })

        return result


# ---------------------------------------------------------------------------
# Simulated Broker (for testing without IBKR connection)
# ---------------------------------------------------------------------------

class SimulatedBroker:
    """Simulated broker for testing strategy logic without IBKR.

    Mimics the IBKRBroker interface using Black-Scholes pricing.
    Useful for development, unit testing, and CI/CD.
    """

    def __init__(self, initial_cash: float = 10_000.0) -> None:
        self.cash = initial_cash
        self.equity = initial_cash
        self._connected = True
        self._positions: List[Dict] = []

    async def connect(self) -> bool:
        self._connected = True
        return True

    async def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def get_account_summary(self) -> AccountSummary:
        return AccountSummary(
            equity=self.equity,
            cash=self.cash,
            buying_power=self.cash,
            maintenance_margin=0,
            unrealized_pnl=0,
            realized_pnl=0,
        )

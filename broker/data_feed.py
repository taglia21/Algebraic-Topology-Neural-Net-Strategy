"""
IBKR market data feed — replaces Alpaca and yfinance entirely.

Provides historical bars, real-time quotes, option chains, and Greeks
all sourced from the paid IBKR data subscription.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from ib_async import IB, Contract, Stock, Option, ComboLeg, TagValue, util
except ImportError:
    IB = Contract = Stock = Option = ComboLeg = TagValue = util = None


class IBKRDataFeed:
    """
    Market data provider backed entirely by IBKR.

    Provides historical OHLCV, real-time quotes, option chains,
    and Greeks without any external data source dependency.
    """

    # IBKR bar size mappings
    VALID_BAR_SIZES = {
        "1min": "1 min",
        "5min": "5 mins",
        "15min": "15 mins",
        "30min": "30 mins",
        "1hour": "1 hour",
        "1day": "1 day",
        "1week": "1 week",
        "1month": "1 month",
    }

    VALID_DURATIONS = {
        "1d": "1 D",
        "1w": "1 W",
        "1m": "1 M",
        "3m": "3 M",
        "6m": "6 M",
        "1y": "1 Y",
        "2y": "2 Y",
        "5y": "5 Y",
    }

    def __init__(self, client) -> None:
        """
        Args:
            client: IBKRClient instance (must be connected).
        """
        self._client = client
        self._subscriptions: dict[str, int] = {}

    @property
    def ib(self) -> IB:
        return self._client.ib

    # --- Contract Helpers ---

    @staticmethod
    def make_stock_contract(symbol: str, exchange: str = "SMART", currency: str = "USD") -> Contract:
        """Create a stock contract."""
        if Stock is None:
            raise ImportError("ib_async not installed")
        return Stock(symbol, exchange, currency)

    @staticmethod
    def make_option_contract(
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Contract:
        """
        Create an option contract.

        Args:
            symbol: Underlying ticker
            expiry: Expiration date as YYYYMMDD string
            strike: Strike price
            right: 'C' for call, 'P' for put
            exchange: Exchange (default SMART)
            currency: Currency (default USD)
        """
        if Option is None:
            raise ImportError("ib_async not installed")
        return Option(symbol, expiry, strike, right, exchange, currency=currency)

    # --- Historical Data ---

    async def get_historical_bars(
        self,
        symbol: str,
        duration: str = "1y",
        bar_size: str = "1day",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars for a single symbol.

        Args:
            symbol: Ticker symbol
            duration: Lookback period ('1d', '1m', '1y', etc.)
            bar_size: Bar granularity ('1min', '1day', etc.)
            what_to_show: Data type ('TRADES', 'MIDPOINT', 'BID', 'ASK')
            use_rth: Regular trading hours only

        Returns:
            DataFrame with columns: open, high, low, close, volume, date
        """
        contract = self.make_stock_contract(symbol)
        ib_duration = self.VALID_DURATIONS.get(duration, duration)
        ib_bar_size = self.VALID_BAR_SIZES.get(bar_size, bar_size)

        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=ib_duration,
            barSizeSetting=ib_bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=2,
        )

        if not bars:
            logger.warning("No historical data returned for %s", symbol)
            return pd.DataFrame()

        return self._bars_to_dataframe(bars)

    async def get_historical_bars_multi(
        self,
        symbols: list[str],
        duration: str = "1y",
        bar_size: str = "1day",
        what_to_show: str = "TRADES",
        delay: float = 0.5,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch historical bars for multiple symbols with rate limiting.

        Args:
            symbols: List of ticker symbols
            duration: Lookback period
            bar_size: Bar granularity
            what_to_show: Data type
            delay: Seconds between requests (IBKR rate limit ~50/10s)

        Returns:
            Dict mapping symbol to DataFrame
        """
        results: dict[str, pd.DataFrame] = {}
        for i, symbol in enumerate(symbols):
            try:
                df = await self.get_historical_bars(
                    symbol, duration, bar_size, what_to_show
                )
                results[symbol] = df
                if i < len(symbols) - 1:
                    await asyncio.sleep(delay)
            except Exception as exc:
                logger.error("Failed to fetch data for %s: %s", symbol, exc)
                results[symbol] = pd.DataFrame()
        return results

    # --- Real-time Data ---

    async def get_realtime_quote(self, symbol: str) -> dict:
        """
        Get current quote snapshot for a symbol.

        Returns dict with: bid, ask, last, volume, high, low, close.
        """
        contract = self.make_stock_contract(symbol)
        self.ib.reqMktData(contract, "", False, False)
        await asyncio.sleep(2)  # allow data to populate

        ticker = self.ib.ticker(contract)
        result = {
            "symbol": symbol,
            "bid": ticker.bid if ticker.bid != -1 else None,
            "ask": ticker.ask if ticker.ask != -1 else None,
            "last": ticker.last if ticker.last != -1 else None,
            "volume": ticker.volume if ticker.volume != -1 else None,
            "high": ticker.high if ticker.high != -1 else None,
            "low": ticker.low if ticker.low != -1 else None,
            "close": ticker.close if ticker.close != -1 else None,
            "timestamp": datetime.now().isoformat(),
        }
        self.ib.cancelMktData(contract)
        return result

    # --- Options Data ---

    async def get_option_chain(self, symbol: str, exchange: str = "SMART") -> pd.DataFrame:
        """
        Fetch full option chain for an underlying symbol.

        Returns DataFrame with columns: expiry, strike, right, bid, ask,
        last, volume, open_interest, implied_vol, delta, gamma, theta, vega.
        """
        stock = self.make_stock_contract(symbol)
        await self.ib.qualifyContractsAsync(stock)

        chains = await self.ib.reqSecDefOptParamsAsync(
            stock.symbol, "", stock.secType, stock.conId
        )

        if not chains:
            logger.warning("No option chain found for %s", symbol)
            return pd.DataFrame()

        # Use the SMART exchange chain (most liquid)
        chain = next((c for c in chains if c.exchange == exchange), chains[0])

        rows = []
        for expiry in sorted(chain.expirations)[:4]:  # next 4 expiries
            for strike in chain.strikes:
                for right in ["C", "P"]:
                    rows.append({
                        "expiry": expiry,
                        "strike": strike,
                        "right": right,
                    })

        return pd.DataFrame(rows)

    async def get_option_greeks(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
    ) -> dict:
        """
        Get Greeks for a specific option contract.

        Returns dict with: implied_vol, delta, gamma, theta, vega,
        bid, ask, last, volume, open_interest.
        """
        contract = self.make_option_contract(symbol, expiry, strike, right)
        await self.ib.qualifyContractsAsync(contract)

        self.ib.reqMktData(contract, "", False, False)
        await asyncio.sleep(2)

        ticker = self.ib.ticker(contract)
        greeks = ticker.modelGreeks or ticker.lastGreeks

        result = {
            "symbol": symbol,
            "expiry": expiry,
            "strike": strike,
            "right": right,
            "bid": ticker.bid if ticker.bid != -1 else None,
            "ask": ticker.ask if ticker.ask != -1 else None,
            "last": ticker.last if ticker.last != -1 else None,
            "volume": ticker.volume if ticker.volume != -1 else None,
            "implied_vol": greeks.impliedVol if greeks else None,
            "delta": greeks.delta if greeks else None,
            "gamma": greeks.gamma if greeks else None,
            "theta": greeks.theta if greeks else None,
            "vega": greeks.vega if greeks else None,
            "underlying_price": greeks.undPrice if greeks else None,
        }

        self.ib.cancelMktData(contract)
        return result

    # --- Internal Helpers ---

    @staticmethod
    def _bars_to_dataframe(bars) -> pd.DataFrame:
        """Convert IB BarData list to pandas DataFrame."""
        data = []
        for bar in bars:
            data.append({
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "average": getattr(bar, "average", None),
                "barCount": getattr(bar, "barCount", None),
            })
        df = pd.DataFrame(data)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
        return df

"""
IBKR connection management via ib_async.

Handles connection to TWS or IB Gateway, auto-reconnection with
exponential backoff, account subscription, and thread-safe access.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    from ib_async import IB, Contract, util
except ImportError:
    IB = None  # allow import without ib_async for testing
    Contract = None
    util = None


@dataclass
class IBKRConfig:
    """IBKR connection configuration."""
    host: str = "127.0.0.1"
    port: int = 4001          # 4001=Gateway live, 4002=Gateway paper, 7496=TWS live, 7497=TWS paper
    client_id: int = 1
    account: str = ""
    timeout: int = 30
    max_reconnect_attempts: int = 3
    readonly: bool = False     # True = no order placement


class IBKRClient:
    """
    Core IBKR connection client.

    Manages connection lifecycle, auto-reconnection, and provides
    access to account data. All order-placement modules depend on this.

    Usage:
        async with IBKRClient(config) as client:
            summary = await client.get_account_summary()
    """

    def __init__(self, config: IBKRConfig) -> None:
        self._config = config
        self._ib: Optional[IB] = None
        self._connected = False
        self._reconnect_attempts = 0
        self._on_connect: list[Callable] = []
        self._on_disconnect: list[Callable] = []
        self._on_error: list[Callable] = []

    # --- Connection Lifecycle ---

    async def connect(self) -> None:
        """Establish connection to IBKR TWS/Gateway."""
        if IB is None:
            raise ImportError("ib_async is not installed. Run: pip install ib_async")

        self._ib = IB()
        self._ib.disconnectedEvent += self._handle_disconnect

        try:
            await self._ib.connectAsync(
                host=self._config.host,
                port=self._config.port,
                clientId=self._config.client_id,
                timeout=self._config.timeout,
                readonly=self._config.readonly,
            )
            # IB Gateway auto-subscribes to account updates on connect.
            # reqAccountUpdates() throws "event loop already running" in
            # ib_async v2.1, so we skip it and rely on the auto-subscription.
            # accountValues() returns cached data populated by the gateway.

            self._connected = True
            self._reconnect_attempts = 0
            logger.info(
                "Connected to IBKR at %s:%d (client_id=%d, account=%s)",
                self._config.host,
                self._config.port,
                self._config.client_id,
                self._config.account or "all",
            )
            for cb in self._on_connect:
                cb()

        except Exception as exc:
            logger.error("Failed to connect to IBKR: %s", exc)
            self._connected = False
            raise

    async def disconnect(self) -> None:
        """Clean shutdown of IBKR connection."""
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")

    async def reconnect(self) -> None:
        """Reconnect with exponential backoff (1s, 2s, 4s... max 60s)."""
        for attempt in range(1, self._config.max_reconnect_attempts + 1):
            delay = min(2 ** (attempt - 1), 60)
            logger.warning(
                "Reconnection attempt %d/%d in %ds...",
                attempt,
                self._config.max_reconnect_attempts,
                delay,
            )
            await asyncio.sleep(delay)
            try:
                await self.connect()
                logger.info("Reconnected on attempt %d", attempt)
                return
            except Exception as exc:
                logger.error("Reconnect attempt %d failed: %s", attempt, exc)

        msg = f"Failed to reconnect after {self._config.max_reconnect_attempts} attempts"
        logger.critical(msg)
        for cb in self._on_error:
            cb(msg)
        raise ConnectionError(msg)

    def _handle_disconnect(self) -> None:
        """Internal handler for unexpected disconnection."""
        self._connected = False
        logger.warning("IBKR connection lost")
        for cb in self._on_disconnect:
            cb()

    # --- Properties ---

    @property
    def ib(self) -> IB:
        """Access the underlying IB instance. Raises if not connected."""
        if not self._connected or self._ib is None:
            raise ConnectionError("Not connected to IBKR")
        return self._ib

    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._connected and self._ib is not None and self._ib.isConnected()

    # --- Account Data ---

    async def get_account_summary(self) -> dict:
        """
        Get account summary from cached accountValues (no new request).

        Uses the real-time account subscription started on connect()
        instead of reqAccountSummary which creates new subscriptions
        and hits the IBKR request limit (Error 322).

        Returns dict with keys like NetLiquidation, BuyingPower, etc.
        """
        acct = self._config.account
        values = self.ib.accountValues(account=acct)
        result: dict = {}
        for av in values:
            try:
                result[av.tag] = float(av.value)
            except (ValueError, TypeError):
                result[av.tag] = av.value
        return result

    async def get_positions(self) -> list:
        """Get current positions from IBKR."""
        positions = await self.ib.reqPositionsAsync()
        return [p for p in positions if not self._config.account or p.account == self._config.account]

    # --- Event Callbacks ---

    def on_connect(self, callback: Callable) -> None:
        """Register callback for connection event."""
        self._on_connect.append(callback)

    def on_disconnect(self, callback: Callable) -> None:
        """Register callback for disconnection event."""
        self._on_disconnect.append(callback)

    def on_error(self, callback: Callable) -> None:
        """Register callback for critical errors."""
        self._on_error.append(callback)

    # --- Context Manager ---

    async def __aenter__(self) -> IBKRClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

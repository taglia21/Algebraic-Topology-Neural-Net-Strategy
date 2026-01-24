#!/usr/bin/env python3
"""
V36 Intraday Monitor
====================
Real-time intraday monitoring system for VIX and regime changes.

Features:
- Asyncio-based continuous monitoring during market hours (9:30 AM - 4:00 PM EST)
- VIX spike detection (>20% change triggers rebalance)
- Regime change detection with callback support
- Graceful SIGTERM handling for clean shutdown
- Alpaca API integration for market data

Usage:
    monitor = IntradayMonitor(rebalance_callback=my_rebalance_fn)
    await monitor.run()
"""

import asyncio
import logging
import os
import signal
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Callable, Dict, Optional, Any, Awaitable
from zoneinfo import ZoneInfo

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('V36_Monitor')

EST = ZoneInfo("America/New_York")


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"


@dataclass
class MonitorConfig:
    """Configuration for the intraday monitor."""
    check_interval_minutes: int = 15
    vix_spike_threshold: float = 0.20  # 20% spike
    market_open: time = field(default_factory=lambda: time(9, 30))
    market_close: time = field(default_factory=lambda: time(16, 0))
    alpaca_api_key: str = field(default_factory=lambda: os.getenv('ALPACA_API_KEY', ''))
    alpaca_secret_key: str = field(default_factory=lambda: os.getenv('ALPACA_SECRET_KEY', ''))
    alpaca_base_url: str = field(
        default_factory=lambda: os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    )
    alpaca_data_url: str = field(
        default_factory=lambda: os.getenv('ALPACA_DATA_URL', 'https://data.alpaca.markets')
    )


@dataclass
class MarketState:
    """Current market state snapshot."""
    vix: float = 0.0
    previous_vix: float = 0.0
    regime: MarketRegime = MarketRegime.SIDEWAYS
    previous_regime: MarketRegime = MarketRegime.SIDEWAYS
    spy_price: float = 0.0
    last_update: Optional[datetime] = None

    @property
    def vix_change_pct(self) -> float:
        """Calculate VIX percentage change."""
        if self.previous_vix <= 0:
            return 0.0
        return (self.vix - self.previous_vix) / self.previous_vix

    @property
    def vix_spiked(self) -> bool:
        """Check if VIX spiked beyond threshold (20%)."""
        return self.vix_change_pct > 0.20

    @property
    def regime_changed(self) -> bool:
        """Check if regime changed from previous state."""
        return self.regime != self.previous_regime


RebalanceCallback = Callable[[MarketState, str], Awaitable[None]]


class AlpacaDataClient:
    """Async Alpaca market data client."""

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session exists."""
        if self.session is None or self.session.closed:
            headers = {
                'APCA-API-KEY-ID': self.config.alpaca_api_key,
                'APCA-API-SECRET-KEY': self.config.alpaca_secret_key,
            }
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch latest quote for a symbol."""
        session = await self._ensure_session()
        url = f"{self.config.alpaca_data_url}/v2/stocks/{symbol}/quotes/latest"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('quote', {})
                logger.warning(f"Failed to fetch quote for {symbol}: {resp.status}")
        except aiohttp.ClientError as e:
            logger.error(f"Alpaca API error fetching {symbol}: {e}")
        return None

    async def get_latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch latest bar (OHLCV) for a symbol."""
        session = await self._ensure_session()
        url = f"{self.config.alpaca_data_url}/v2/stocks/{symbol}/bars/latest"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('bar', {})
                logger.warning(f"Failed to fetch bar for {symbol}: {resp.status}")
        except aiohttp.ClientError as e:
            logger.error(f"Alpaca API error fetching bar for {symbol}: {e}")
        return None


class IntradayMonitor:
    """
    Intraday monitoring system for VIX and regime changes.
    
    Runs continuously during market hours, checking VIX levels and market
    regime every 15 minutes. Triggers rebalance callback when VIX spikes
    >20% or regime changes.
    
    Args:
        config: Monitor configuration settings
        rebalance_callback: Async function called on VIX spike or regime change
    
    Example:
        async def on_rebalance(state: MarketState, reason: str):
            print(f"Rebalance triggered: {reason}")
        
        monitor = IntradayMonitor(rebalance_callback=on_rebalance)
        await monitor.run()
    """

    def __init__(
        self,
        config: Optional[MonitorConfig] = None,
        rebalance_callback: Optional[RebalanceCallback] = None,
    ):
        self.config = config or MonitorConfig()
        self.rebalance_callback = rebalance_callback
        self.client = AlpacaDataClient(self.config)
        self.state = MarketState()
        self._shutdown_event = asyncio.Event()
        self._running = False

    def _setup_signal_handlers(self) -> None:
        """Setup SIGTERM and SIGINT handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown_signal, sig)
        logger.info("Signal handlers configured for graceful shutdown")

    def _handle_shutdown_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signal."""
        logger.info(f"Received signal {sig.name}, initiating graceful shutdown...")
        self._shutdown_event.set()

    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours (EST)."""
        now = datetime.now(EST)
        current_time = now.time()
        weekday = now.weekday()
        
        # Markets closed on weekends
        if weekday >= 5:
            return False
        
        return self.config.market_open <= current_time <= self.config.market_close

    def _classify_regime(self, vix: float, spy_change: float = 0.0) -> MarketRegime:
        """
        Classify market regime based on VIX and SPY movement.
        
        Args:
            vix: Current VIX value
            spy_change: SPY percentage change (optional)
        
        Returns:
            MarketRegime classification
        """
        if vix >= 35:
            return MarketRegime.CRISIS
        elif vix >= 25:
            return MarketRegime.BEAR
        elif vix <= 15 and spy_change >= 0:
            return MarketRegime.BULL
        else:
            return MarketRegime.SIDEWAYS

    async def _fetch_market_data(self) -> None:
        """Fetch current VIX and SPY data from Alpaca."""
        # Store previous values
        self.state.previous_vix = self.state.vix
        self.state.previous_regime = self.state.regime
        
        # Fetch VIX (using VIXY as proxy since VIX itself isn't tradeable)
        vix_bar = await self.client.get_latest_bar("VIXY")
        if vix_bar:
            # VIXY is a VIX proxy ETF - approximate VIX level
            self.state.vix = vix_bar.get('c', self.state.vix) * 1.5
        
        # Fetch SPY for regime detection
        spy_bar = await self.client.get_latest_bar("SPY")
        if spy_bar:
            new_spy = spy_bar.get('c', 0.0)
            spy_change = 0.0
            if self.state.spy_price > 0:
                spy_change = (new_spy - self.state.spy_price) / self.state.spy_price
            self.state.spy_price = new_spy
            self.state.regime = self._classify_regime(self.state.vix, spy_change)
        
        self.state.last_update = datetime.now(EST)
        logger.info(
            f"Market data updated: VIX={self.state.vix:.2f} "
            f"(Δ{self.state.vix_change_pct:+.1%}), "
            f"SPY=${self.state.spy_price:.2f}, "
            f"Regime={self.state.regime.value}"
        )

    async def _check_and_trigger(self) -> None:
        """Check conditions and trigger rebalance if needed."""
        if not self.rebalance_callback:
            return

        reasons = []
        
        if self.state.vix_spiked:
            reasons.append(f"VIX spike: {self.state.vix_change_pct:+.1%}")
        
        if self.state.regime_changed:
            reasons.append(
                f"Regime change: {self.state.previous_regime.value} → {self.state.regime.value}"
            )
        
        if reasons:
            reason_str = "; ".join(reasons)
            logger.warning(f"Rebalance triggered: {reason_str}")
            try:
                await self.rebalance_callback(self.state, reason_str)
            except Exception as e:
                logger.error(f"Rebalance callback error: {e}")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        interval_seconds = self.config.check_interval_minutes * 60
        
        while not self._shutdown_event.is_set():
            if self._is_market_hours():
                try:
                    await self._fetch_market_data()
                    await self._check_and_trigger()
                except Exception as e:
                    logger.error(f"Monitor loop error: {e}")
            else:
                logger.debug("Outside market hours, skipping check")
            
            # Wait for next interval or shutdown
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=interval_seconds
                )
                break  # Shutdown signaled
            except asyncio.TimeoutError:
                continue  # Normal timeout, continue loop

    async def run(self) -> None:
        """
        Start the intraday monitor.
        
        Runs continuously until SIGTERM/SIGINT or shutdown() is called.
        """
        if self._running:
            logger.warning("Monitor already running")
            return
        
        self._running = True
        logger.info(
            f"Starting IntradayMonitor: checking every {self.config.check_interval_minutes} min "
            f"during {self.config.market_open} - {self.config.market_close} EST"
        )
        
        try:
            self._setup_signal_handlers()
            await self._monitor_loop()
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully shutdown the monitor."""
        if not self._running:
            return
        
        logger.info("Shutting down IntradayMonitor...")
        self._shutdown_event.set()
        await self.client.close()
        self._running = False
        logger.info("IntradayMonitor shutdown complete")

    async def check_now(self) -> MarketState:
        """
        Perform an immediate market check (for testing or manual trigger).
        
        Returns:
            Current MarketState snapshot
        """
        await self._fetch_market_data()
        await self._check_and_trigger()
        return self.state


async def main() -> None:
    """Example usage of IntradayMonitor."""
    
    async def example_rebalance(state: MarketState, reason: str) -> None:
        logger.info(f"[REBALANCE] {reason}")
        logger.info(f"  VIX: {state.vix:.2f}, Regime: {state.regime.value}")
    
    config = MonitorConfig(check_interval_minutes=15)
    monitor = IntradayMonitor(config=config, rebalance_callback=example_rebalance)
    
    try:
        await monitor.run()
    except KeyboardInterrupt:
        await monitor.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

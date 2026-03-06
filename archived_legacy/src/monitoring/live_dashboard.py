"""
Phase T — Live Dashboard with WebSocket.

Item 23: LiveDashboard — WebSocket /ws endpoint, 5-sec JSON push, docker-compose port 8002.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class DashboardSnapshot:
    """Real-time dashboard data pushed via WebSocket."""
    timestamp: str = ""
    portfolio_value: float = 0.0
    daily_pnl: float = 0.0
    daily_return_pct: float = 0.0
    positions_count: int = 0
    open_orders: int = 0
    regime: str = "unknown"
    var_99_1d: float = 0.0
    sharpe_30d: float = 0.0
    max_drawdown: float = 0.0
    compliance_status: str = "ok"
    uptime_seconds: float = 0.0


class LiveDashboard:
    """WebSocket-based live trading dashboard.

    Pushes DashboardSnapshot every 5 seconds to all connected clients.
    Endpoint: ws://host:8002/ws

    Architecture:
      - Async WebSocket server using asyncio.
      - Connected clients tracked in a set.
      - Data source: pull from portfolio/risk/compliance components.
      - Graceful shutdown support.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8002,
        push_interval: float = 5.0,
    ):
        self.host = host
        self.port = port
        self.push_interval = push_interval
        self._clients: Set[Any] = set()
        self._running: bool = False
        self._start_time: float = time.time()
        self._snapshot: DashboardSnapshot = DashboardSnapshot()
        self._server: Optional[Any] = None

    def update_snapshot(
        self,
        portfolio_value: float = 0.0,
        daily_pnl: float = 0.0,
        daily_return_pct: float = 0.0,
        positions_count: int = 0,
        open_orders: int = 0,
        regime: str = "unknown",
        var_99_1d: float = 0.0,
        sharpe_30d: float = 0.0,
        max_drawdown: float = 0.0,
        compliance_status: str = "ok",
    ) -> DashboardSnapshot:
        """Update the current dashboard snapshot.

        Called by trading engine components to keep data fresh.
        """
        self._snapshot = DashboardSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            portfolio_value=portfolio_value,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
            positions_count=positions_count,
            open_orders=open_orders,
            regime=regime,
            var_99_1d=var_99_1d,
            sharpe_30d=sharpe_30d,
            max_drawdown=max_drawdown,
            compliance_status=compliance_status,
            uptime_seconds=time.time() - self._start_time,
        )
        return self._snapshot

    def get_snapshot_json(self) -> str:
        """Get current snapshot as JSON string."""
        self._snapshot.uptime_seconds = time.time() - self._start_time
        self._snapshot.timestamp = datetime.now(timezone.utc).isoformat()
        return json.dumps(asdict(self._snapshot))

    async def _handle_client(self, websocket: Any, path: str = "/ws") -> None:
        """Handle a WebSocket client connection."""
        self._clients.add(websocket)
        client_addr = getattr(websocket, 'remote_address', 'unknown')
        logger.info("Dashboard client connected: %s (total: %d)", client_addr, len(self._clients))

        try:
            # Send initial snapshot immediately
            await websocket.send(self.get_snapshot_json())

            # Keep connection alive
            async for message in websocket:
                # Handle client messages (e.g., subscription preferences)
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.debug("Client disconnected: %s", e)
        finally:
            self._clients.discard(websocket)
            logger.info("Dashboard client disconnected (remaining: %d)", len(self._clients))

    async def _broadcast_loop(self) -> None:
        """Broadcast snapshot to all clients every push_interval seconds."""
        while self._running:
            if self._clients:
                message = self.get_snapshot_json()
                disconnected = set()
                for client in self._clients.copy():
                    try:
                        await client.send(message)
                    except Exception:
                        disconnected.add(client)
                self._clients -= disconnected
            await asyncio.sleep(self.push_interval)

    async def start_async(self) -> None:
        """Start the WebSocket server (async version)."""
        try:
            import websockets
        except ImportError:
            logger.warning("websockets package not installed — dashboard unavailable")
            return

        self._running = True
        self._start_time = time.time()

        self._server = await websockets.serve(  # type: ignore
            self._handle_client,
            self.host,
            self.port,
        )
        logger.info("Live dashboard started on ws://%s:%d/ws", self.host, self.port)

        # Start broadcast loop
        asyncio.create_task(self._broadcast_loop())

    async def stop_async(self) -> None:
        """Stop the WebSocket server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        # Close all client connections
        for client in self._clients.copy():
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()
        logger.info("Live dashboard stopped")

    def start(self) -> None:
        """Start dashboard in a new event loop (blocking)."""
        asyncio.run(self._run())

    async def _run(self) -> None:
        """Run the server until stopped."""
        await self.start_async()
        while self._running:
            await asyncio.sleep(1)

    @property
    def connected_clients(self) -> int:
        return len(self._clients)

    @property
    def snapshot(self) -> DashboardSnapshot:
        return self._snapshot

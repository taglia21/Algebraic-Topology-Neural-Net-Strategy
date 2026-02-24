"""
Prometheus Metrics Exporter (Phase L, Item 20)
================================================

Expose ``/metrics`` HTTP endpoint (port 8001) with Prometheus gauges:
  - daily_pnl
  - open_positions
  - net_delta
  - net_vega
  - sharpe_30d
  - win_rate_7d
  - model_confidence_avg

Uses a lightweight built-in HTTP server (no prometheus_client dependency
required, but uses it if available).
"""

import json
import logging
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["MetricsExporter"]

DEFAULT_PORT = 8001


class _MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler serving /metrics in Prometheus text format."""

    server: "MetricsHTTPServer"

    def do_GET(self):
        if self.path == "/metrics":
            body = self.server.exporter.render_metrics()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        """Suppress default access logging."""
        pass


class MetricsHTTPServer(HTTPServer):
    """HTTPServer subclass that carries a reference to the exporter."""
    exporter: "MetricsExporter"


class MetricsExporter:
    """Prometheus-compatible metrics exporter.

    Exposes trading metrics at ``http://0.0.0.0:{port}/metrics``.

    Parameters
    ----------
    port : int
        HTTP port (default 8001).
    """

    # Gauge definitions: name → (help_text, initial_value)
    GAUGE_DEFS = {
        "trading_daily_pnl": ("Daily P&L in USD", 0.0),
        "trading_open_positions": ("Number of open option positions", 0),
        "trading_net_delta": ("Net portfolio delta", 0.0),
        "trading_net_vega": ("Net portfolio vega in USD", 0.0),
        "trading_sharpe_30d": ("30-day rolling Sharpe ratio", 0.0),
        "trading_win_rate_7d": ("7-day win rate (0-1)", 0.0),
        "trading_model_confidence_avg": ("Average model confidence (0-1)", 0.0),
    }

    def __init__(self, port: int = DEFAULT_PORT):
        self.port = port
        self._gauges: Dict[str, float] = {
            name: val for name, (_, val) in self.GAUGE_DEFS.items()
        }
        self._server: Optional[MetricsHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def set(self, name: str, value: float) -> None:
        """Set a gauge value.

        Parameters
        ----------
        name : str
            Gauge name (e.g. "trading_daily_pnl").
        value : float
        """
        if name in self._gauges:
            self._gauges[name] = value
        else:
            logger.warning("Unknown metric: %s", name)

    def get(self, name: str) -> float:
        """Get current gauge value."""
        return self._gauges.get(name, 0.0)

    def render_metrics(self) -> str:
        """Render all gauges in Prometheus text exposition format."""
        lines = []
        for name, (help_text, _) in self.GAUGE_DEFS.items():
            value = self._gauges.get(name, 0.0)
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        lines.append("")  # trailing newline
        return "\n".join(lines)

    def start(self) -> None:
        """Start the metrics HTTP server in a background thread."""
        if self._thread and self._thread.is_alive():
            return

        server = MetricsHTTPServer(("0.0.0.0", self.port), _MetricsHandler)
        server.exporter = self
        self._server = server

        self._thread = threading.Thread(
            target=server.serve_forever, daemon=True, name="metrics-exporter",
        )
        self._thread.start()
        logger.info("Prometheus metrics exporter started on port %d", self.port)

    def stop(self) -> None:
        """Stop the HTTP server."""
        if self._server:
            self._server.shutdown()
            self._server = None
        logger.info("Metrics exporter stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

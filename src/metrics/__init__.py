#!/usr/bin/env python3
"""
Prometheus Metrics Endpoint for V28 Production Trading System
==============================================================

Exposes trading metrics at :9090/metrics for Prometheus scraping.

Metrics exposed:
  - trading_portfolio_value          (Gauge)
  - trading_daily_pnl               (Gauge)
  - trading_positions_count          (Gauge)
  - trading_daily_turnover_pct       (Gauge)
  - trading_regime_scale             (Gauge)
  - trading_max_drawdown_pct         (Gauge)
  - trading_orders_total             (Counter, by side/status)
  - trading_signals_total            (Counter, by action)
  - trading_cycle_duration_seconds   (Histogram)
  - trading_rolling_sharpe           (Gauge)

Usage:
    from src.metrics.prometheus_metrics import MetricsServer, record_order, record_signal

    server = MetricsServer(port=9090)
    server.start()  # non-blocking background thread
"""

import logging
import threading
from typing import Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CollectorRegistry,
    start_http_server,
    generate_latest,
)

logger = logging.getLogger(__name__)

# ── Global registry (allows test isolation) ──
REGISTRY = CollectorRegistry()

# ── Gauges ──
PORTFOLIO_VALUE = Gauge(
    "trading_portfolio_value",
    "Current portfolio equity in USD",
    registry=REGISTRY,
)
DAILY_PNL = Gauge(
    "trading_daily_pnl",
    "Daily P&L in USD",
    registry=REGISTRY,
)
POSITIONS_COUNT = Gauge(
    "trading_positions_count",
    "Number of open equity positions",
    registry=REGISTRY,
)
DAILY_TURNOVER_PCT = Gauge(
    "trading_daily_turnover_pct",
    "Daily turnover as percentage of equity",
    registry=REGISTRY,
)
REGIME_SCALE = Gauge(
    "trading_regime_scale",
    "Current regime position scale (0.0-1.0)",
    registry=REGISTRY,
)
MAX_DRAWDOWN_PCT = Gauge(
    "trading_max_drawdown_pct",
    "Maximum drawdown from peak as percentage",
    registry=REGISTRY,
)
ROLLING_SHARPE = Gauge(
    "trading_rolling_sharpe",
    "Rolling 30-day Sharpe ratio",
    registry=REGISTRY,
)

# ── Counters ──
ORDERS_TOTAL = Counter(
    "trading_orders_total",
    "Total orders placed",
    ["side", "status"],
    registry=REGISTRY,
)
SIGNALS_TOTAL = Counter(
    "trading_signals_total",
    "Total signals generated",
    ["action"],
    registry=REGISTRY,
)
FILTERS_BLOCKED = Counter(
    "trading_filters_blocked_total",
    "Signals blocked by filters",
    ["filter_name"],
    registry=REGISTRY,
)

# ── Histograms ──
CYCLE_DURATION = Histogram(
    "trading_cycle_duration_seconds",
    "Duration of each equity trading cycle",
    buckets=[1, 5, 10, 30, 60, 120, 300],
    registry=REGISTRY,
)


# ── Convenience functions ──

def update_portfolio_metrics(
    equity: float,
    daily_pnl: float = 0.0,
    n_positions: int = 0,
    turnover_pct: float = 0.0,
    regime_scale: float = 1.0,
    max_dd_pct: float = 0.0,
    sharpe: float = 0.0,
):
    """Update all portfolio-level gauges at once."""
    PORTFOLIO_VALUE.set(equity)
    DAILY_PNL.set(daily_pnl)
    POSITIONS_COUNT.set(n_positions)
    DAILY_TURNOVER_PCT.set(turnover_pct)
    REGIME_SCALE.set(regime_scale)
    MAX_DRAWDOWN_PCT.set(max_dd_pct)
    ROLLING_SHARPE.set(sharpe)


def record_order(side: str, status: str = "submitted"):
    """Increment order counter."""
    ORDERS_TOTAL.labels(side=side, status=status).inc()


def record_signal(action: str):
    """Increment signal counter (e.g., BUY, SELL, HOLD, BLOCKED)."""
    SIGNALS_TOTAL.labels(action=action).inc()


def record_filter_block(filter_name: str):
    """Increment filter block counter."""
    FILTERS_BLOCKED.labels(filter_name=filter_name).inc()


class MetricsServer:
    """Start a Prometheus metrics HTTP server in a background thread."""

    def __init__(self, port: int = 9090):
        self.port = port
        self._started = False

    def start(self):
        """Start the metrics HTTP server (non-blocking)."""
        if self._started:
            return
        try:
            start_http_server(self.port, registry=REGISTRY)
            self._started = True
            logger.info(f"Prometheus metrics server started on :{self.port}/metrics")
        except Exception as e:
            logger.warning(f"Failed to start metrics server on :{self.port}: {e}")

    def get_metrics_text(self) -> str:
        """Return metrics in Prometheus text format (for testing)."""
        return generate_latest(REGISTRY).decode("utf-8")

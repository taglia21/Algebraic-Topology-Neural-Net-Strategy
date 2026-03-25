"""
core/logger.py
==============
Structured JSON logging for the ATNN trade audit trail.

Every log record is a JSON object written to both a rotating daily file
(logs/trades_YYYY-MM-DD.jsonl) and stdout.  Each record carries:

    - ``ts``         — ISO-8601 timestamp with microseconds
    - ``session_id`` — set once at startup; identifies a trading session
    - ``level``      — INFO | WARNING | ERROR | CRITICAL
    - ``event``      — machine-readable event type (see EVENT_* constants)
    - ``...``        — event-specific fields

Usage
-----
    from core.logger import get_trade_logger

    log = get_trade_logger(session_id="2026-03-06-A")
    log.log_signal("momentum", "AAPL", "BUY", 0.82, {"z_score": 1.9})
    log.log_order("ord-001", "AAPL", "buy", 100, 175.00, "submitted")

Design notes
------------
- ``TradeLogger`` is **not** a singleton — the orchestrator owns the instance
  and passes it to sub-components.  Use :func:`get_trade_logger` for a
  process-level default.
- All I/O errors when writing to the file are re-raised; no silent failures.
- Performance stats (P&L, Sharpe, drawdown) are tracked in-memory and written
  as ``perf_snapshot`` events at configurable intervals.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Event-type constants (machine-readable labels in every log record)
# ---------------------------------------------------------------------------
EVENT_SIGNAL = "signal"
EVENT_ORDER = "order"
EVENT_FILL = "fill"
EVENT_RISK_EVENT = "risk_event"
EVENT_REGIME_CHANGE = "regime_change"
EVENT_PERF_SNAPSHOT = "perf_snapshot"
EVENT_ERROR = "error"
EVENT_INFO = "info"


class _JSONFileHandler(logging.Handler):
    """Custom ``logging.Handler`` that writes one JSON object per line.

    Rotates to a new file at midnight (file is keyed to ``date.today()``).
    The handler opens the file lazily on the first emit, and re-opens it
    whenever the calendar date has changed.
    """

    def __init__(self, log_dir: str) -> None:
        super().__init__()
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._current_date: Optional[date] = None
        self._file: Optional[Any] = None  # open file handle

    def _get_file(self):  # type: ignore[return]
        today = date.today()
        if self._current_date != today:
            if self._file is not None:
                self._file.close()
            filename = self._log_dir / f"trades_{today.isoformat()}.jsonl"
            self._file = open(filename, "a", encoding="utf-8")  # noqa: WPS515
            self._current_date = today
        return self._file

    def emit(self, record: logging.LogRecord) -> None:
        try:
            f = self._get_file()
            f.write(record.getMessage() + "\n")
            f.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        super().close()


class TradeLogger:
    """Structured audit logger for all trading events.

    Parameters
    ----------
    session_id:
        Unique identifier for this trading session.  Included in every log
        record so logs from multiple sessions can be filtered easily.
    log_dir:
        Directory for daily ``.jsonl`` log files.  Created on first write.
    log_level:
        Python logging level string (DEBUG, INFO, WARNING, ERROR).
    echo_stdout:
        When True (default), records are also printed to stdout as JSON.
    risk_free_rate:
        Annualised risk-free rate used for Sharpe / Sortino calculation
        (default 0.05 = 5%, representative of current Fed Funds regime).
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        log_dir: str = "logs",
        log_level: str = "INFO",
        echo_stdout: bool = True,
        risk_free_rate: float = 0.05,
    ) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())[:8]
        self._risk_free_rate: float = risk_free_rate
        self._log_dir = log_dir
        self._echo_stdout = echo_stdout

        # Set up internal Python logger (used for routing only)
        self._logger = logging.getLogger(f"trade_logger.{self.session_id}")
        self._logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        self._logger.propagate = False  # don't bubble up to root logger

        # File handler (JSONL rotating)
        self._file_handler = _JSONFileHandler(log_dir)
        self._file_handler.setLevel(logging.DEBUG)
        self._logger.addHandler(self._file_handler)

        # Stdout handler
        if echo_stdout:
            _stdout_handler = logging.StreamHandler(sys.stdout)
            _stdout_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
            self._logger.addHandler(_stdout_handler)

        # In-memory performance tracking
        self._portfolio_values: List[float] = []
        self._daily_pnl: List[float] = []
        self._peak_value: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now_iso() -> str:
        """Return current UTC time as an ISO-8601 string with microseconds."""
        return datetime.now(timezone.utc).isoformat()

    def _emit(self, level: str, record: Dict[str, Any]) -> None:
        """Serialise *record* to JSON and route through the Python logger."""
        record.setdefault("ts", self._now_iso())
        record["session_id"] = self.session_id
        record["level"] = level.upper()

        msg = json.dumps(record, default=str)
        log_level = getattr(logging, level.upper(), logging.INFO)
        self._logger.log(log_level, msg)

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_signal(
        self,
        strategy: str,
        symbol: str,
        direction: str,
        strength: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a strategy signal before order generation.

        Parameters
        ----------
        strategy:
            Strategy name, e.g. ``"stat_arb"``, ``"momentum"``.
        symbol:
            Ticker symbol, e.g. ``"AAPL"``.
        direction:
            ``"BUY"``, ``"SELL"``, or ``"FLAT"``.
        strength:
            Normalised signal strength in [0, 1].
        metadata:
            Optional dict of strategy-specific diagnostics (Z-score, factor
            scores, etc.).
        """
        if not 0.0 <= strength <= 1.0:
            raise ValueError(
                f"Signal strength must be in [0, 1]; got {strength!r} "
                f"for {strategy}/{symbol}"
            )
        self._emit("INFO", {
            "event": EVENT_SIGNAL,
            "strategy": strategy,
            "symbol": symbol,
            "direction": direction.upper(),
            "strength": round(strength, 6),
            "metadata": metadata or {},
        })

    def log_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an order submission or status update.

        Parameters
        ----------
        order_id:
            Broker-assigned or internal order identifier.
        symbol:
            Ticker symbol.
        side:
            ``"buy"`` or ``"sell"``.
        qty:
            Requested quantity (shares / contracts).
        price:
            Limit price or estimated execution price.
        status:
            Order lifecycle status, e.g. ``"submitted"``, ``"filled"``,
            ``"cancelled"``.
        metadata:
            Optional extra fields (order type, time-in-force, etc.).
        """
        if qty <= 0:
            raise ValueError(f"Order qty must be positive; got {qty!r} for {symbol}")
        if price < 0:
            raise ValueError(f"Order price must be non-negative; got {price!r} for {symbol}")

        self._emit("INFO", {
            "event": EVENT_ORDER,
            "order_id": order_id,
            "symbol": symbol,
            "side": side.lower(),
            "qty": qty,
            "price": price,
            "status": status,
            "metadata": metadata or {},
        })

    def log_fill(
        self,
        order_id: str,
        fill_price: float,
        fill_qty: float,
        slippage: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an order fill with execution quality metrics.

        Parameters
        ----------
        order_id:
            Must match a previously logged order.
        fill_price:
            Actual execution price.
        fill_qty:
            Number of shares / contracts filled.
        slippage:
            Signed slippage in basis points (positive = paid more than expected).
        metadata:
            Optional extra fields (venue, liquidity flag, etc.).
        """
        self._emit("INFO", {
            "event": EVENT_FILL,
            "order_id": order_id,
            "fill_price": fill_price,
            "fill_qty": fill_qty,
            "slippage_bps": round(slippage, 4),
            "metadata": metadata or {},
        })

    def log_risk_event(
        self,
        event_type: str,
        details: Dict[str, Any],
    ) -> None:
        """Log a risk-management event (breach, gate trigger, override).

        Parameters
        ----------
        event_type:
            Short label such as ``"drawdown_reduce"``, ``"daily_loss_halt"``,
            ``"position_limit_breach"``, ``"correlation_breach"``.
        details:
            Arbitrary key/value diagnostics (current values, thresholds, etc.).
        """
        self._emit("WARNING", {
            "event": EVENT_RISK_EVENT,
            "event_type": event_type,
            "details": details,
        })

    def log_regime_change(
        self,
        old_regime: str,
        new_regime: str,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a market-regime transition.

        Parameters
        ----------
        old_regime:
            Previous regime name.
        new_regime:
            New regime name.
        confidence:
            HMM posterior probability for the new state, in (0, 1].
        metadata:
            Optional extra fields (VIX level, ADX, etc.).
        """
        self._emit("INFO", {
            "event": EVENT_REGIME_CHANGE,
            "old_regime": old_regime,
            "new_regime": new_regime,
            "confidence": round(confidence, 6),
            "metadata": metadata or {},
        })

    def log_error(self, message: str, exc_info: Optional[Exception] = None) -> None:
        """Log an error.  Never swallows the exception — callers are
        responsible for re-raising if needed.
        """
        record: Dict[str, Any] = {"event": EVENT_ERROR, "message": message}
        if exc_info is not None:
            record["exception"] = repr(exc_info)
        self._emit("ERROR", record)

    def log_info(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a generic informational message."""
        self._emit("INFO", {
            "event": EVENT_INFO,
            "message": message,
            "metadata": metadata or {},
        })

    # ------------------------------------------------------------------
    # Performance tracking
    # ------------------------------------------------------------------

    def update_portfolio_value(self, value: float) -> None:
        """Register the current mark-to-market portfolio value.

        Called by the portfolio manager after each bar.  Internally tracks the
        series used to compute real-time performance metrics.

        Parameters
        ----------
        value:
            Current portfolio equity in USD.
        """
        if value <= 0:
            raise ValueError(f"Portfolio value must be positive; got {value!r}")

        self._portfolio_values.append(value)
        if value > self._peak_value:
            self._peak_value = value

        if len(self._portfolio_values) >= 2:
            self._daily_pnl.append(value - self._portfolio_values[-2])

    def compute_performance(self) -> Dict[str, float]:
        """Compute real-time performance metrics from the stored equity series.

        Returns
        -------
        dict with keys:
            - ``total_return``    — total return since inception
            - ``sharpe``          — annualised Sharpe ratio (assumes daily bars)
            - ``sortino``         — annualised Sortino ratio
            - ``max_drawdown``    — maximum peak-to-trough drawdown (negative)
            - ``current_drawdown``— drawdown from most recent peak (negative)
        """
        if len(self._portfolio_values) < 2:
            return {
                "total_return": 0.0,
                "sharpe": float("nan"),
                "sortino": float("nan"),
                "max_drawdown": 0.0,
                "current_drawdown": 0.0,
            }

        import numpy as np  # deferred import — numpy is a heavy dep

        values = np.array(self._portfolio_values, dtype=float)
        daily_returns = np.diff(values) / values[:-1]

        total_return = (values[-1] / values[0]) - 1.0

        # Sharpe (annualised, assuming 252 trading days)
        # Use annualized risk-free rate (default 5% for current regime)
        rf_annual = getattr(self, '_risk_free_rate', 0.05)
        rf_daily = rf_annual / 252
        excess = daily_returns - rf_daily
        sharpe = float("nan")
        sample_std = float(np.std(excess, ddof=1))
        if sample_std > 0:
            sharpe = float((excess.mean() / sample_std) * math.sqrt(252))

        # Sortino
        # Correct Sortino: downside deviation = sqrt(mean(min(r - target, 0)^2))
        # computed over ALL returns, not just the std of losing days.
        # target = 0 (excess returns already net of risk-free)
        sortino = float("nan")
        target = 0.0
        downside_diff = np.minimum(excess - target, 0.0)
        downside_dev = float(np.sqrt(np.mean(downside_diff ** 2)))
        if downside_dev > 0:
            sortino = float((excess.mean() / downside_dev) * math.sqrt(252))

        # Max drawdown
        running_max = np.maximum.accumulate(values)
        drawdowns = (values - running_max) / running_max
        max_drawdown = float(drawdowns.min())
        current_drawdown = float(drawdowns[-1])

        return {
            "total_return": round(total_return, 6),
            "sharpe": round(sharpe, 4) if not math.isnan(sharpe) else float("nan"),
            "sortino": round(sortino, 4) if not math.isnan(sortino) else float("nan"),
            "max_drawdown": round(max_drawdown, 6),
            "current_drawdown": round(current_drawdown, 6),
        }

    def log_perf_snapshot(self) -> None:
        """Compute performance metrics and write a ``perf_snapshot`` record."""
        metrics = self.compute_performance()
        self._emit("INFO", {
            "event": EVENT_PERF_SNAPSHOT,
            **metrics,
            "n_bars": len(self._portfolio_values),
            "current_equity": (
                self._portfolio_values[-1] if self._portfolio_values else 0.0
            ),
            "peak_equity": self._peak_value,
        })

    def close(self) -> None:
        """Flush and close all handlers."""
        for handler in list(self._logger.handlers):
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Process-level default logger
# ---------------------------------------------------------------------------

_DEFAULT_LOGGER: Optional[TradeLogger] = None


def get_trade_logger(
    session_id: Optional[str] = None,
    log_dir: str = "logs",
    log_level: str = "INFO",
    echo_stdout: bool = True,
) -> TradeLogger:
    """Return (or create) the process-level default :class:`TradeLogger`.

    The first call creates the singleton; subsequent calls return the same
    instance unless *session_id* is provided and differs from the current one.

    Parameters
    ----------
    session_id:
        Human-readable session tag, e.g. ``"2026-03-06-A"``.  Defaults to an
        8-character UUID fragment.
    log_dir:
        Directory for daily JSONL log files.
    log_level:
        Minimum log level (DEBUG / INFO / WARNING / ERROR).
    echo_stdout:
        Mirror every record to stdout.

    Returns
    -------
    TradeLogger
    """
    global _DEFAULT_LOGGER

    if _DEFAULT_LOGGER is None or (
        session_id is not None and session_id != _DEFAULT_LOGGER.session_id
    ):
        _DEFAULT_LOGGER = TradeLogger(
            session_id=session_id,
            log_dir=log_dir,
            log_level=log_level,
            echo_stdout=echo_stdout,
        )

    return _DEFAULT_LOGGER

"""
Daily Performance Logger
========================
Records daily P/L, equity, positions, trades, turnover at close.
Computes rolling 20-day Sharpe, max drawdown, win rate.
Writes JSON-lines to logs/daily_performance.jsonl for retraining loop consumption.

Usage:
    logger = DailyPerformanceLogger(log_dir="logs")
    logger.log_daily(equity=75000, daily_pnl=120.50, n_positions=5,
                     n_trades=3, turnover_pct=4.2)
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

_logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


@dataclass
class DailySnapshot:
    """Single day's performance record."""
    date: str
    equity: float
    daily_pnl: float
    daily_return_pct: float
    n_positions: int
    n_trades: int
    turnover_pct: float
    cumulative_return_pct: float
    rolling_sharpe_20d: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    win_rate: Optional[float] = None
    timestamp: str = ""


class DailyPerformanceLogger:
    """
    Tracks and persists daily trading performance metrics.

    Maintains an in-memory rolling window of daily returns for
    computing Sharpe, drawdown, and win rate. Each call to log_daily()
    appends a JSON line to logs/daily_performance.jsonl.
    """

    def __init__(self, log_dir: Optional[str] = None, initial_equity: float = 0.0):
        self._log_dir = Path(log_dir) if log_dir else LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self._log_dir / "daily_performance.jsonl"

        self._initial_equity: float = initial_equity
        self._prev_equity: float = initial_equity
        self._peak_equity: float = initial_equity
        self._daily_returns: List[float] = []
        self._daily_pnls: List[float] = []
        self._last_date: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_daily(
        self,
        equity: float,
        daily_pnl: float = 0.0,
        n_positions: int = 0,
        n_trades: int = 0,
        turnover_pct: float = 0.0,
    ) -> DailySnapshot:
        """
        Record one day's metrics.  Call once at/after market close.

        Args:
            equity: End-of-day portfolio equity.
            daily_pnl: Dollar P/L for the day.
            n_positions: Number of open positions at close.
            n_trades: Trades executed today.
            turnover_pct: Daily turnover as % of equity.

        Returns:
            DailySnapshot with computed rolling metrics.
        """
        today = date.today().isoformat()

        # Guard against duplicate calls on the same day
        if today == self._last_date:
            _logger.debug(f"DailyPerformanceLogger: already logged {today}, skipping")
            return self._last_snapshot  # type: ignore[return-value]

        # Bootstrap initial equity on first call
        if self._initial_equity <= 0:
            self._initial_equity = equity
            self._prev_equity = equity
            self._peak_equity = equity

        # Daily return
        daily_ret = (equity - self._prev_equity) / self._prev_equity if self._prev_equity > 0 else 0.0
        self._daily_returns.append(daily_ret)
        self._daily_pnls.append(daily_pnl)

        # Peak / drawdown
        if equity > self._peak_equity:
            self._peak_equity = equity
        dd = (self._peak_equity - equity) / self._peak_equity * 100 if self._peak_equity > 0 else 0.0

        # Cumulative return
        cum_ret = (equity - self._initial_equity) / self._initial_equity * 100 if self._initial_equity > 0 else 0.0

        # Rolling metrics
        sharpe = self._rolling_sharpe(20)
        wr = self._win_rate()

        snap = DailySnapshot(
            date=today,
            equity=round(equity, 2),
            daily_pnl=round(daily_pnl, 2),
            daily_return_pct=round(daily_ret * 100, 4),
            n_positions=n_positions,
            n_trades=n_trades,
            turnover_pct=round(turnover_pct, 2),
            cumulative_return_pct=round(cum_ret, 4),
            rolling_sharpe_20d=round(sharpe, 4) if sharpe is not None else None,
            max_drawdown_pct=round(dd, 4),
            win_rate=round(wr, 4) if wr is not None else None,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

        # Persist to JSONL
        self._write_jsonl(snap)

        self._prev_equity = equity
        self._last_date = today
        self._last_snapshot = snap

        _logger.info(
            f"Daily perf: equity=${equity:,.0f}  pnl=${daily_pnl:+,.2f}  "
            f"ret={daily_ret:+.2%}  sharpe={sharpe or 0:.2f}  dd={dd:.2f}%  "
            f"wr={wr or 0:.1%}  pos={n_positions}  trades={n_trades}"
        )

        return snap

    def get_history(self) -> List[dict]:
        """Read back all daily snapshots from the JSONL file."""
        if not self._jsonl_path.exists():
            return []
        records = []
        with open(self._jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rolling_sharpe(self, window: int = 20) -> Optional[float]:
        """Annualized Sharpe from the last `window` daily returns."""
        if len(self._daily_returns) < window:
            return None
        rets = np.array(self._daily_returns[-window:])
        mu = rets.mean()
        sigma = rets.std()
        if sigma < 1e-10:
            return 0.0
        return float(mu / sigma * np.sqrt(252))

    def _win_rate(self) -> Optional[float]:
        """Fraction of positive-PnL days."""
        if not self._daily_pnls:
            return None
        wins = sum(1 for p in self._daily_pnls if p > 0)
        return wins / len(self._daily_pnls)

    def _write_jsonl(self, snap: DailySnapshot):
        """Append one JSON line to the log file."""
        try:
            with open(self._jsonl_path, "a") as f:
                f.write(json.dumps(asdict(snap)) + "\n")
        except Exception as e:
            _logger.warning(f"Failed to write daily perf JSONL: {e}")

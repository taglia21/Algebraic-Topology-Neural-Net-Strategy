"""
Retraining Scheduler
=====================
Monitors live model performance metrics and triggers retraining when
performance degrades below configurable thresholds.

Features
--------
- Rolling window accuracy / Sharpe tracking
- Automatic retraining trigger on degradation
- Cooldown period to prevent thrashing
- History log of retraining events
- Pluggable retrain callback

Usage::

    from src.ml.retraining_scheduler import RetrainingScheduler, RetrainingConfig

    def my_retrain_fn(reason: str) -> bool:
        '''Retrain the model. Return True on success.'''
        model.fit(new_data)
        return True

    scheduler = RetrainingScheduler(
        config=RetrainingConfig(
            accuracy_floor=0.52,
            sharpe_floor=0.5,
            check_interval_hours=24,
            cooldown_hours=72,
        ),
        retrain_fn=my_retrain_fn,
    )

    # Call periodically (e.g. end-of-day)
    scheduler.record_prediction(predicted=1, actual=1)  # correct
    scheduler.record_prediction(predicted=1, actual=0)  # wrong
    scheduler.record_trade_return(0.012)                # +1.2% return

    action = scheduler.check_and_retrain()
    # action = {"triggered": True, "reason": "accuracy 0.48 < floor 0.52", ...}
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrainingConfig:
    """Thresholds and timing for the retraining scheduler."""
    accuracy_floor: float = 0.52        # retrain if rolling accuracy < this
    sharpe_floor: float = 0.3           # retrain if rolling Sharpe < this
    rolling_window: int = 100           # number of predictions / trades
    check_interval_hours: float = 24    # how often to check metrics
    cooldown_hours: float = 72          # minimum time between retraining
    max_retrains_per_week: int = 2      # safety cap


@dataclass
class RetrainingEvent:
    """Log entry for a retraining event."""
    timestamp: datetime
    reason: str
    accuracy_before: float
    sharpe_before: float
    success: bool
    duration_sec: float = 0.0


class RetrainingScheduler:
    """
    Watches rolling accuracy and Sharpe, triggers retraining when
    performance falls below configured floors.
    """

    def __init__(
        self,
        config: Optional[RetrainingConfig] = None,
        retrain_fn: Optional[Callable[[str], bool]] = None,
    ):
        self.cfg = config or RetrainingConfig()
        self.retrain_fn = retrain_fn

        # Rolling metrics buffers
        self._predictions: deque = deque(maxlen=self.cfg.rolling_window)
        self._trade_returns: deque = deque(maxlen=self.cfg.rolling_window)

        # Timing
        self._last_check: float = 0.0
        self._last_retrain: Optional[datetime] = None

        # History
        self._events: List[RetrainingEvent] = []

    # ── Recording ────────────────────────────────────────────────

    def record_prediction(self, predicted: int, actual: int):
        """Record a binary prediction (1=up, 0=down) vs actual."""
        self._predictions.append(int(predicted == actual))

    def record_trade_return(self, ret: float):
        """Record a single trade's return (e.g. 0.02 = +2%)."""
        self._trade_returns.append(float(ret))

    # ── Metrics ──────────────────────────────────────────────────

    @property
    def rolling_accuracy(self) -> float:
        if len(self._predictions) < 10:
            return 1.0  # not enough data — assume OK
        return float(np.mean(self._predictions))

    @property
    def rolling_sharpe(self) -> float:
        if len(self._trade_returns) < 10:
            return 999.0  # not enough data — assume OK
        rets = np.array(self._trade_returns)
        mu = rets.mean()
        sigma = rets.std()
        if sigma < 1e-8:
            return 0.0
        return float(mu / sigma * np.sqrt(252))

    @property
    def needs_retrain(self) -> tuple[bool, str]:
        """Check if retraining thresholds are breached."""
        acc = self.rolling_accuracy
        sharpe = self.rolling_sharpe

        if acc < self.cfg.accuracy_floor and len(self._predictions) >= 20:
            return True, f"accuracy {acc:.3f} < floor {self.cfg.accuracy_floor}"
        if sharpe < self.cfg.sharpe_floor and len(self._trade_returns) >= 20:
            return True, f"sharpe {sharpe:.2f} < floor {self.cfg.sharpe_floor}"
        return False, "metrics OK"

    # ── Check + trigger retrain ──────────────────────────────────

    def check_and_retrain(self) -> Dict:
        """
        Check metrics and trigger retraining if needed.

        Returns dict with 'triggered', 'reason', 'success' keys.
        """
        now = time.time()

        # Respect check interval
        if now - self._last_check < self.cfg.check_interval_hours * 3600:
            return {"triggered": False, "reason": "check interval not reached"}
        self._last_check = now

        triggered, reason = self.needs_retrain
        if not triggered:
            return {"triggered": False, "reason": reason}

        # Respect cooldown
        if self._last_retrain:
            elapsed = (datetime.now() - self._last_retrain).total_seconds()
            if elapsed < self.cfg.cooldown_hours * 3600:
                remaining_h = (self.cfg.cooldown_hours * 3600 - elapsed) / 3600
                return {
                    "triggered": False,
                    "reason": f"cooldown ({remaining_h:.1f}h remaining)",
                }

        # Weekly cap
        week_ago = datetime.now() - timedelta(days=7)
        recent_count = sum(
            1 for e in self._events if e.timestamp > week_ago
        )
        if recent_count >= self.cfg.max_retrains_per_week:
            return {
                "triggered": False,
                "reason": f"weekly cap ({recent_count}/{self.cfg.max_retrains_per_week})",
            }

        # Execute retraining
        acc_before = self.rolling_accuracy
        sharpe_before = self.rolling_sharpe

        logger.info(f"🔄 RETRAINING triggered: {reason}")
        success = False
        duration = 0.0
        if self.retrain_fn:
            t0 = time.time()
            try:
                success = bool(self.retrain_fn(reason))
                duration = time.time() - t0
            except Exception as e:
                logger.error(f"Retraining failed: {e}")
                duration = time.time() - t0
        else:
            logger.warning("No retrain_fn configured — skipping actual retrain")

        event = RetrainingEvent(
            timestamp=datetime.now(),
            reason=reason,
            accuracy_before=acc_before,
            sharpe_before=sharpe_before,
            success=success,
            duration_sec=duration,
        )
        self._events.append(event)
        self._last_retrain = datetime.now()

        # Clear rolling buffers after retrain
        if success:
            self._predictions.clear()
            self._trade_returns.clear()

        logger.info(
            f"Retraining {'succeeded' if success else 'FAILED'} "
            f"in {duration:.1f}s (acc={acc_before:.3f}, sharpe={sharpe_before:.2f})"
        )

        return {
            "triggered": True,
            "reason": reason,
            "success": success,
            "duration_sec": duration,
            "accuracy_before": acc_before,
            "sharpe_before": sharpe_before,
        }

    # ── Diagnostics ──────────────────────────────────────────────

    def get_status(self) -> Dict:
        return {
            "rolling_accuracy": self.rolling_accuracy,
            "rolling_sharpe": self.rolling_sharpe,
            "predictions_buffered": len(self._predictions),
            "trades_buffered": len(self._trade_returns),
            "total_retrains": len(self._events),
            "last_retrain": str(self._last_retrain) if self._last_retrain else None,
            "needs_retrain": self.needs_retrain[0],
        }

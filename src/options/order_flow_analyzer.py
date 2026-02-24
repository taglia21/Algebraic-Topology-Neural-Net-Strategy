"""
Order Flow Analyzer — Tape Speed Detector (Phase F, Item 3)
============================================================

Measure prints-per-second on the L1 tape.
If tape_speed > 3× 20-period average, flag as institutional flow
and boost signal confidence by 15%.
"""

import logging
import time as _time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["TapeSpeedDetector", "TapeSpeedReading"]

# Defaults
DEFAULT_LOOKBACK = 20
INSTITUTIONAL_MULTIPLIER = 3.0
CONFIDENCE_BOOST = 0.15


@dataclass
class TapeSpeedReading:
    """Result from tape speed measurement."""
    symbol: str
    prints_per_second: float
    avg_prints_per_second: float
    is_institutional: bool
    confidence_boost: float  # 0.0 or CONFIDENCE_BOOST


class TapeSpeedDetector:
    """Measure tape speed (prints/second) and detect institutional flow.

    Maintains a rolling window of trade-print timestamps.  Each time
    ``record_print()`` is called, the detector updates the rolling
    prints-per-second rate.

    Parameters
    ----------
    lookback : int
        Number of periods for rolling average (default 20).
    multiplier : float
        Threshold multiplier (default 3.0 → 3× average = institutional).
    boost : float
        Confidence boost when institutional flow detected (default 0.15).
    window_seconds : float
        Measurement window for prints-per-second (default 1.0).
    """

    def __init__(
        self,
        lookback: int = DEFAULT_LOOKBACK,
        multiplier: float = INSTITUTIONAL_MULTIPLIER,
        boost: float = CONFIDENCE_BOOST,
        window_seconds: float = 1.0,
    ):
        self.lookback = lookback
        self.multiplier = multiplier
        self.boost = boost
        self.window_seconds = window_seconds

        # Per-symbol state
        self._timestamps: dict[str, deque] = {}
        self._pps_history: dict[str, deque] = {}

    def record_print(self, symbol: str, timestamp: Optional[float] = None) -> None:
        """Record a trade print for tape speed calculation.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        timestamp : float or None
            Unix timestamp; defaults to now.
        """
        ts = timestamp or _time.time()
        if symbol not in self._timestamps:
            self._timestamps[symbol] = deque(maxlen=5000)
            self._pps_history[symbol] = deque(maxlen=self.lookback)
        self._timestamps[symbol].append(ts)

    def get_tape_speed(self, symbol: str) -> TapeSpeedReading:
        """Measure current prints-per-second and compare to average.

        Parameters
        ----------
        symbol : str

        Returns
        -------
        TapeSpeedReading
        """
        stamps = self._timestamps.get(symbol)
        if not stamps or len(stamps) < 2:
            return TapeSpeedReading(
                symbol=symbol, prints_per_second=0.0,
                avg_prints_per_second=0.0, is_institutional=False,
                confidence_boost=0.0,
            )

        now = stamps[-1]
        cutoff = now - self.window_seconds
        recent = [t for t in stamps if t > cutoff]
        pps = len(recent) / self.window_seconds if self.window_seconds > 0 else 0

        # Update history
        history = self._pps_history.setdefault(
            symbol, deque(maxlen=self.lookback)
        )
        history.append(pps)

        avg_pps = float(np.mean(history)) if len(history) >= 2 else pps
        is_institutional = (avg_pps > 0) and (pps > self.multiplier * avg_pps)
        boost = self.boost if is_institutional else 0.0

        if is_institutional:
            logger.info(
                "INSTITUTIONAL FLOW %s: %.1f pps > %.1f × %.1f avg → +%.0f%% boost",
                symbol, pps, self.multiplier, avg_pps, boost * 100,
            )

        return TapeSpeedReading(
            symbol=symbol,
            prints_per_second=pps,
            avg_prints_per_second=avg_pps,
            is_institutional=is_institutional,
            confidence_boost=boost,
        )

"""
Mean Reversion Strategy
========================
Bollinger Band squeeze detection, RSI divergence, and z-score
entry/exit signals for range-bound regimes.

Signals:
  LONG  — price at lower band + RSI < 30 + z-score < -2
  SHORT — price at upper band + RSI > 70 + z-score > +2
  EXIT  — z-score crosses back inside ±0.5

Usage:
    from src.mean_reversion_strategy import MeanReversionStrategy, MRSignal
    mr = MeanReversionStrategy()
    sig = mr.generate_signal(bars)
    if sig.direction != 'HOLD':
        print(sig)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────

class MRDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"
    HOLD = "HOLD"


@dataclass
class MRSignal:
    """Signal produced by the mean-reversion engine."""
    symbol: str
    direction: str            # LONG / SHORT / EXIT / HOLD
    z_score: float            # current z-score
    rsi: float                # current RSI(14)
    bb_position: float        # 0 = lower band, 1 = upper band
    squeeze_active: bool      # Bollinger bandwidth < threshold
    confidence: float         # 0-1 composite confidence
    reasons: List[str]


@dataclass
class MRConfig:
    """Tunables for the mean-reversion engine."""
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    squeeze_threshold: float = 0.04   # bandwidth < 4% = squeeze

    # RSI
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # Z-score
    zscore_lookback: int = 20
    zscore_entry: float = 2.0         # |z| > 2 → entry
    zscore_exit: float = 0.5          # |z| < 0.5 → exit

    # Confirmation
    min_bars: int = 50                # need 50 bars minimum
    min_confidence: float = 0.40      # skip if confidence < 0.40


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    """RSI from close array."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses)) + 1e-10
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _bollinger_bands(closes: np.ndarray, period: int = 20, n_std: float = 2.0):
    """Return (upper, middle, lower, bandwidth, bb_position)."""
    if len(closes) < period:
        mid = closes[-1]
        return mid, mid, mid, 0.0, 0.5

    window = closes[-period:]
    mid = float(np.mean(window))
    std = float(np.std(window, ddof=1))
    upper = mid + n_std * std
    lower = mid - n_std * std
    bandwidth = (upper - lower) / mid if mid > 0 else 0.0
    price = closes[-1]
    bb_pos = (price - lower) / (upper - lower) if upper != lower else 0.5
    return upper, mid, lower, bandwidth, float(np.clip(bb_pos, 0.0, 1.0))


def _z_score(closes: np.ndarray, lookback: int = 20) -> float:
    """Rolling z-score of latest close vs lookback mean/std."""
    if len(closes) < lookback:
        return 0.0
    window = closes[-lookback:]
    mu = float(np.mean(window))
    sigma = float(np.std(window, ddof=1)) + 1e-10
    return float((closes[-1] - mu) / sigma)


# ─────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────

class MeanReversionStrategy:
    """
    Mean reversion signal generator.

    Combines Bollinger squeeze detection, RSI extremes, and z-score
    to produce LONG / SHORT / EXIT / HOLD signals.
    """

    def __init__(self, config: MRConfig = None):
        self.cfg = config or MRConfig()

    def generate_signal(
        self, bars: List[dict], symbol: str = "UNK",
    ) -> MRSignal:
        """
        Generate a mean-reversion signal from OHLCV bars.

        Parameters
        ----------
        bars : list[dict]
            Alpaca-style bars ``[{"o", "h", "l", "c", "v"}, ...]``
        symbol : str
            Ticker (for logging / tracking).

        Returns
        -------
        MRSignal
        """
        hold = MRSignal(
            symbol=symbol, direction="HOLD", z_score=0.0,
            rsi=50.0, bb_position=0.5, squeeze_active=False,
            confidence=0.0, reasons=["insufficient data"],
        )

        if not bars or len(bars) < self.cfg.min_bars:
            return hold

        closes = np.array([float(b["c"]) for b in bars])

        # Indicators
        rsi = _compute_rsi(closes, self.cfg.rsi_period)
        upper, mid, lower, bandwidth, bb_pos = _bollinger_bands(
            closes, self.cfg.bb_period, self.cfg.bb_std,
        )
        z = _z_score(closes, self.cfg.zscore_lookback)
        squeeze = bandwidth < self.cfg.squeeze_threshold

        reasons: List[str] = []
        direction = "HOLD"
        confidence = 0.0

        # ── EXIT check (z-score reverted) ────────────────────────
        if abs(z) < self.cfg.zscore_exit:
            direction = "EXIT"
            confidence = 0.6
            reasons.append(f"z={z:+.2f} reverted inside ±{self.cfg.zscore_exit}")
            return MRSignal(
                symbol=symbol, direction=direction, z_score=z,
                rsi=rsi, bb_position=bb_pos, squeeze_active=squeeze,
                confidence=confidence, reasons=reasons,
            )

        # ── LONG conditions ──────────────────────────────────────
        long_score = 0.0
        if z < -self.cfg.zscore_entry:
            long_score += 0.40
            reasons.append(f"z={z:+.2f}<-{self.cfg.zscore_entry}")
        if rsi < self.cfg.rsi_oversold:
            long_score += 0.30
            reasons.append(f"RSI={rsi:.0f}<{self.cfg.rsi_oversold}")
        if bb_pos < 0.05:
            long_score += 0.20
            reasons.append("price at lower BB")
        if squeeze:
            long_score += 0.10
            reasons.append("BB squeeze")

        # ── SHORT conditions ─────────────────────────────────────
        short_score = 0.0
        short_reasons: List[str] = []
        if z > self.cfg.zscore_entry:
            short_score += 0.40
            short_reasons.append(f"z={z:+.2f}>+{self.cfg.zscore_entry}")
        if rsi > self.cfg.rsi_overbought:
            short_score += 0.30
            short_reasons.append(f"RSI={rsi:.0f}>{self.cfg.rsi_overbought}")
        if bb_pos > 0.95:
            short_score += 0.20
            short_reasons.append("price at upper BB")
        if squeeze:
            short_score += 0.10
            short_reasons.append("BB squeeze")

        # Pick the stronger side
        if long_score >= short_score and long_score >= self.cfg.min_confidence:
            direction = "LONG"
            confidence = long_score
        elif short_score > long_score and short_score >= self.cfg.min_confidence:
            direction = "SHORT"
            confidence = short_score
            reasons = short_reasons
        else:
            reasons = ["no strong MR signal"]

        return MRSignal(
            symbol=symbol, direction=direction, z_score=z,
            rsi=rsi, bb_position=bb_pos, squeeze_active=squeeze,
            confidence=confidence, reasons=reasons,
        )

    def scan_universe(
        self, bars_map: Dict[str, List[dict]],
    ) -> List[MRSignal]:
        """Score every symbol and return actionable signals (non-HOLD)."""
        results = []
        for sym, bars in bars_map.items():
            sig = self.generate_signal(bars, symbol=sym)
            if sig.direction != "HOLD":
                results.append(sig)
        results.sort(key=lambda s: s.confidence, reverse=True)
        return results

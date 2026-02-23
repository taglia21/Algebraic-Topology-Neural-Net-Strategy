"""
Alpha Signal Engine — Advanced Options & Equity Signal Generation
==================================================================

Generates composite alpha scores from multiple independent signals:

1. VIX term structure (VIX3M/VIX backwardation)
2. Put/Call ratio from IBKR market data
3. Gamma Exposure (GEX) for SPY/QQQ magnetic strike levels
4. IV momentum: 5-day vs 20-day IV percentile rank
5. Earnings calendar gating: block trades 2 days before earnings

Each signal returns a score in [-1.0, +1.0].
Signals are combined using Kelly-weighted ensemble.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AlphaSignal:
    """Individual alpha signal output."""

    name: str
    score: float  # -1.0 to +1.0
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompositeSignal:
    """Kelly-weighted composite of all alpha signals."""

    overall_score: float  # -1.0 to +1.0
    confidence: float
    signals: List[AlphaSignal] = field(default_factory=list)
    blocked: bool = False
    block_reason: str = ""


# ---------------------------------------------------------------------------
# Individual signal generators
# ---------------------------------------------------------------------------


class VIXTermStructureSignal:
    """
    VIX term structure signal.

    Signal logic:
    - VIX3M / VIX > 1.10  → backwardation → bearish → score < 0 (buy puts)
    - VIX3M / VIX < 0.90  → contango      → bullish → score > 0
    - Between 0.90-1.10    → neutral
    """

    def __init__(self, backwardation_threshold: float = 1.10, contango_threshold: float = 0.90):
        self.backwardation_thresh = backwardation_threshold
        self.contango_thresh = contango_threshold

    def compute(self, vix: float, vix3m: float) -> AlphaSignal:
        if vix <= 0:
            return AlphaSignal(name="vix_term_structure", score=0.0, confidence=0.0)

        ratio = vix3m / vix

        if ratio > self.backwardation_thresh:
            # Backwardation: near-term vol higher expected → bearish
            score = -min(1.0, (ratio - self.backwardation_thresh) / 0.2)
            confidence = min(1.0, abs(ratio - 1.0))
        elif ratio < self.contango_thresh:
            # Steep contango: market complacent → bullish
            score = min(1.0, (self.contango_thresh - ratio) / 0.2)
            confidence = min(1.0, abs(ratio - 1.0))
        else:
            score = 0.0
            confidence = 0.2

        return AlphaSignal(
            name="vix_term_structure",
            score=np.clip(score, -1.0, 1.0),
            confidence=confidence,
            metadata={"vix": vix, "vix3m": vix3m, "ratio": ratio},
        )


class PutCallRatioSignal:
    """
    Put/Call ratio signal.

    - PC ratio > 1.2 → extreme fear → contrarian bullish
    - PC ratio < 0.6 → extreme greed → contrarian bearish
    """

    def __init__(self, fear_threshold: float = 1.2, greed_threshold: float = 0.6):
        self.fear_thresh = fear_threshold
        self.greed_thresh = greed_threshold

    def compute(self, put_call_ratio: float) -> AlphaSignal:
        if put_call_ratio <= 0:
            return AlphaSignal(name="put_call_ratio", score=0.0, confidence=0.0)

        if put_call_ratio > self.fear_thresh:
            # Contrarian bullish
            score = min(1.0, (put_call_ratio - self.fear_thresh) / 0.5)
            confidence = min(1.0, (put_call_ratio - 1.0) / 0.5)
        elif put_call_ratio < self.greed_thresh:
            # Contrarian bearish
            score = -min(1.0, (self.greed_thresh - put_call_ratio) / 0.3)
            confidence = min(1.0, (1.0 - put_call_ratio) / 0.5)
        else:
            score = 0.0
            confidence = 0.2

        return AlphaSignal(
            name="put_call_ratio",
            score=np.clip(score, -1.0, 1.0),
            confidence=confidence,
            metadata={"put_call_ratio": put_call_ratio},
        )


class GammaExposureSignal:
    """
    Gamma Exposure (GEX) signal for SPY/QQQ.

    Identifies magnetic strike levels where market makers have concentrated
    gamma.  Positive GEX → market pinned (mean-revert). Negative GEX → trending.
    """

    def compute(
        self,
        spot_price: float,
        strikes: List[float],
        call_oi: List[int],
        put_oi: List[int],
        call_gamma: List[float],
        put_gamma: List[float],
    ) -> AlphaSignal:
        if not strikes or spot_price <= 0:
            return AlphaSignal(name="gex", score=0.0, confidence=0.0)

        # Net GEX per strike: call_gamma * call_oi - put_gamma * put_oi
        n = min(len(strikes), len(call_oi), len(put_oi), len(call_gamma), len(put_gamma))
        gex_by_strike = []
        for i in range(n):
            net = call_gamma[i] * call_oi[i] * 100 - put_gamma[i] * put_oi[i] * 100
            gex_by_strike.append((strikes[i], net))

        total_gex = sum(g for _, g in gex_by_strike)

        # Find magnetic strike (max absolute GEX)
        if gex_by_strike:
            magnetic_strike, _ = max(gex_by_strike, key=lambda x: abs(x[1]))
        else:
            magnetic_strike = spot_price

        # Score: positive GEX = mean-reverting (buy dips near magnetic), negative = trend-following
        if total_gex > 0:
            # Dealers long gamma → mean reversion environment
            dist = (spot_price - magnetic_strike) / spot_price
            score = -np.clip(dist * 10, -1.0, 1.0)  # push towards magnetic
            confidence = min(1.0, abs(total_gex) / 1e9)
        else:
            # Dealers short gamma → momentum/trend
            score = 0.0
            confidence = 0.3

        return AlphaSignal(
            name="gex",
            score=np.clip(score, -1.0, 1.0),
            confidence=confidence,
            metadata={
                "total_gex": total_gex,
                "magnetic_strike": magnetic_strike,
                "spot": spot_price,
            },
        )


class IVMomentumSignal:
    """
    IV momentum signal: 5-day vs 20-day IV percentile rank.

    - Short-term IV rising faster than long-term → bearish (vol expansion)
    - Short-term IV falling relative to long-term → bullish (vol compression)
    """

    def compute(self, iv_history: List[float]) -> AlphaSignal:
        if len(iv_history) < 20:
            return AlphaSignal(name="iv_momentum", score=0.0, confidence=0.0)

        arr = np.array(iv_history[-20:])
        iv_5d = float(np.mean(arr[-5:]))
        iv_20d = float(np.mean(arr))

        # Percentile rank of current IV in the 20-day window
        current_iv = arr[-1]
        pct_rank = float(np.sum(arr <= current_iv)) / len(arr)

        # If short-term IV accelerating above long-term → bearish signal
        iv_ratio = iv_5d / max(iv_20d, 1e-8)

        if iv_ratio > 1.15:
            score = -min(1.0, (iv_ratio - 1.0) * 3)  # bearish
        elif iv_ratio < 0.85:
            score = min(1.0, (1.0 - iv_ratio) * 3)  # bullish
        else:
            score = 0.0

        confidence = abs(score) * 0.8

        return AlphaSignal(
            name="iv_momentum",
            score=np.clip(score, -1.0, 1.0),
            confidence=confidence,
            metadata={
                "iv_5d": iv_5d,
                "iv_20d": iv_20d,
                "iv_ratio": iv_ratio,
                "pct_rank": pct_rank,
            },
        )


class EarningsGateSignal:
    """
    Earnings calendar gating: block new positions 2 days before earnings.

    This is a binary gate: blocked or not.
    """

    def __init__(self, blackout_days: int = 2):
        self.blackout_days = blackout_days
        self._earnings_cache: Dict[str, datetime] = {}

    def set_earnings_dates(self, earnings_map: Dict[str, str]):
        """Load earnings dates as {symbol: 'YYYY-MM-DD'}."""
        for sym, dt_str in earnings_map.items():
            try:
                self._earnings_cache[sym] = datetime.strptime(dt_str, "%Y-%m-%d")
            except ValueError:
                pass

    def check(self, symbol: str, now: Optional[datetime] = None) -> AlphaSignal:
        """Returns a signal — score=0 if blocked (within blackout), else neutral."""
        now = now or datetime.now()
        earnings_dt = self._earnings_cache.get(symbol)

        blocked = False
        days_until = None
        if earnings_dt:
            delta = (earnings_dt - now).days
            days_until = delta
            if 0 <= delta <= self.blackout_days:
                blocked = True

        if blocked:
            return AlphaSignal(
                name="earnings_gate",
                score=0.0,
                confidence=0.0,  # zero confidence = do not trade
                metadata={"blocked": True, "days_until_earnings": days_until, "symbol": symbol},
            )
        return AlphaSignal(
            name="earnings_gate",
            score=0.0,
            confidence=1.0,  # OK to trade
            metadata={"blocked": False, "days_until_earnings": days_until, "symbol": symbol},
        )


# ---------------------------------------------------------------------------
# Kelly-weighted ensemble combiner
# ---------------------------------------------------------------------------


def kelly_weighted_combine(signals: List[AlphaSignal]) -> CompositeSignal:
    """
    Combine signals using Kelly-style confidence weighting.

    Kelly weight for each signal: w_i = confidence_i / sum(confidences)
    Composite score = sum(w_i * score_i)
    """
    if not signals:
        return CompositeSignal(overall_score=0.0, confidence=0.0, signals=[])

    total_conf = sum(s.confidence for s in signals)
    if total_conf < 1e-10:
        return CompositeSignal(overall_score=0.0, confidence=0.0, signals=signals)

    weighted_score = sum(s.score * s.confidence for s in signals) / total_conf
    avg_confidence = total_conf / len(signals)

    return CompositeSignal(
        overall_score=float(np.clip(weighted_score, -1.0, 1.0)),
        confidence=float(avg_confidence),
        signals=signals,
    )


# ---------------------------------------------------------------------------
# Signal Engine — main facade
# ---------------------------------------------------------------------------


class SignalEngine:
    """
    Unified signal engine that runs all alpha sub-signals and combines them.

    Usage::

        engine = SignalEngine()
        signal = engine.generate_signal(
            symbol='SPY',
            vix=18.0, vix3m=20.0,
            put_call_ratio=1.05,
            iv_history=[...],
        )
        print(signal.overall_score, signal.confidence)
    """

    def __init__(self):
        self.vix_signal = VIXTermStructureSignal()
        self.pc_signal = PutCallRatioSignal()
        self.gex_signal = GammaExposureSignal()
        self.iv_momentum = IVMomentumSignal()
        self.earnings_gate = EarningsGateSignal()

    def set_earnings_dates(self, earnings_map: Dict[str, str]):
        """Load earnings dates for gating."""
        self.earnings_gate.set_earnings_dates(earnings_map)

    def generate_signal(
        self,
        symbol: str,
        vix: float = 0.0,
        vix3m: float = 0.0,
        put_call_ratio: float = 0.0,
        iv_history: Optional[List[float]] = None,
        spot_price: float = 0.0,
        strikes: Optional[List[float]] = None,
        call_oi: Optional[List[int]] = None,
        put_oi: Optional[List[int]] = None,
        call_gamma: Optional[List[float]] = None,
        put_gamma: Optional[List[float]] = None,
    ) -> CompositeSignal:
        """Generate composite signal for given market data."""

        signals: List[AlphaSignal] = []

        # 1. Earnings gate (binary block check)
        earnings = self.earnings_gate.check(symbol)
        if earnings.metadata.get("blocked"):
            return CompositeSignal(
                overall_score=0.0,
                confidence=0.0,
                signals=[earnings],
                blocked=True,
                block_reason=f"Earnings blackout ({earnings.metadata.get('days_until_earnings')}d)",
            )

        # 2. VIX term structure
        if vix > 0 and vix3m > 0:
            signals.append(self.vix_signal.compute(vix, vix3m))

        # 3. Put/Call ratio
        if put_call_ratio > 0:
            signals.append(self.pc_signal.compute(put_call_ratio))

        # 4. GEX
        if spot_price > 0 and strikes:
            signals.append(
                self.gex_signal.compute(
                    spot_price,
                    strikes or [],
                    call_oi or [],
                    put_oi or [],
                    call_gamma or [],
                    put_gamma or [],
                )
            )

        # 5. IV momentum
        if iv_history and len(iv_history) >= 20:
            signals.append(self.iv_momentum.compute(iv_history))

        if not signals:
            return CompositeSignal(overall_score=0.0, confidence=0.0, signals=[])

        return kelly_weighted_combine(signals)

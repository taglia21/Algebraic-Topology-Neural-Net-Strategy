"""
Multi-Timeframe Confluence Analyzer — Tier 2
=============================================

Requires agreement across multiple timeframes before allowing entry.

Timeframes analyzed:
  - 5m   → Micro-structure momentum (entry timing)
  - 15m  → Short-term trend (scalp confirmation)
  - 1h   → Intraday trend (primary signal)
  - 4h   → Swing trend (directional bias)
  - 1D   → Daily trend (macro direction)

Confluence rules:
  - BUY requires >= min_confirming timeframes (default 3/5) bullish
  - Higher timeframes carry more weight (daily >> 5m)
  - Disagreement between 1D and 1h triggers "caution" mode

Data: Alpaca bars at multiple timeframes, or synthetic resampling
from 5m/15m bars to construct 1h/4h/D.

Integration:
    from src.multi_timeframe import MultiTimeframeAnalyzer
    mtf = MultiTimeframeAnalyzer()
    result = mtf.analyze("AAPL", bars_by_tf)
    if result.confirms_long:
        # proceed with buy
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class Trend(Enum):
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1


@dataclass
class TimeframeSignal:
    """Analysis result for one timeframe."""
    timeframe: str
    trend: Trend
    strength: float              # [0, 1]
    momentum: float              # Raw momentum value
    rsi: float = 50.0
    sma_cross: float = 0.0      # Normalized SMA crossover signal
    volume_confirm: bool = True  # Volume supports direction


@dataclass
class ConfluenceResult:
    """Multi-timeframe confluence analysis result."""
    symbol: str
    timestamp: datetime

    # Per-timeframe signals
    signals: Dict[str, TimeframeSignal] = field(default_factory=dict)

    # Confluence metrics
    bullish_count: int = 0       # How many TFs are bullish
    bearish_count: int = 0
    neutral_count: int = 0
    total_timeframes: int = 0

    # Weighted score
    weighted_score: float = 0.0  # [-1, 1]: +1 = all bullish, -1 = all bearish
    confluence_pct: float = 0.0  # % of TFs agreeing with majority

    # Trade decision
    confirms_long: bool = False
    confirms_short: bool = False
    caution: bool = False        # Higher TF disagrees with entry TF
    dominant_trend: Trend = Trend.NEUTRAL

    # Position sizing modifier
    confluence_scale: float = 1.0  # Higher when more TFs agree

    @property
    def is_actionable(self) -> bool:
        return self.confirms_long or self.confirms_short

    def describe(self) -> str:
        """Human-readable summary."""
        parts = []
        for tf in ["5m", "15m", "1h", "4h", "D"]:
            if tf in self.signals:
                sig = self.signals[tf]
                arrow = "↑" if sig.trend == Trend.BULLISH else "↓" if sig.trend == Trend.BEARISH else "→"
                parts.append(f"{tf}:{arrow}{sig.strength:.0%}")
        tf_str = " | ".join(parts)
        decision = "LONG" if self.confirms_long else "SHORT" if self.confirms_short else "HOLD"
        return f"[{decision}] {tf_str} (score={self.weighted_score:+.2f}, conf={self.confluence_pct:.0%})"


# ============================================================================
# TIMEFRAME WEIGHTS
# ============================================================================

# Higher timeframes get more weight in confluence voting
TIMEFRAME_WEIGHTS = {
    "5m": 0.10,
    "15m": 0.15,
    "1h": 0.25,
    "4h": 0.25,
    "D": 0.25,
}

# Minimum bars needed per timeframe for valid analysis
MIN_BARS = {
    "5m": 40,
    "15m": 30,
    "1h": 24,
    "4h": 20,
    "D": 20,
}


# ============================================================================
# CORE ENGINE
# ============================================================================

class MultiTimeframeAnalyzer:
    """
    Requires confluence across 5m/15m/1h/4h/D timeframes.

    Features:
      - SMA crossover trend detection per TF
      - RSI momentum confirmation
      - Volume-weighted conviction
      - Weighted voting with higher TF priority
      - Caution flag when daily trend opposes intraday

    Usage:
        mtf = MultiTimeframeAnalyzer(min_confirming=3)
        result = mtf.analyze("AAPL", {"5m": bars_5m, "1h": bars_1h, "D": bars_d})
        if result.confirms_long:
            execute_buy()
    """

    TIMEFRAMES = ["5m", "15m", "1h", "4h", "D"]

    def __init__(
        self,
        min_confirming: int = 3,
        strong_trend_threshold: float = 0.6,
    ):
        self.min_confirming = min_confirming
        self.strong_trend_threshold = strong_trend_threshold

        # Cache per-symbol results
        self._last_results: Dict[str, ConfluenceResult] = {}

    # ── Public API ──────────────────────────────────────────────────

    def analyze(
        self,
        symbol: str,
        bars_by_tf: Dict[str, List[dict]],
    ) -> ConfluenceResult:
        """
        Analyze multiple timeframes and compute confluence.

        Args:
            symbol: Ticker
            bars_by_tf: Dict mapping timeframe string to OHLCV bars
                        e.g. {"5m": [...], "1h": [...], "D": [...]}

        Returns:
            ConfluenceResult with trade confirmation
        """
        signals: Dict[str, TimeframeSignal] = {}

        for tf in self.TIMEFRAMES:
            bars = bars_by_tf.get(tf)
            if bars is None or len(bars) < MIN_BARS.get(tf, 20):
                continue
            sig = self._analyze_timeframe(tf, bars)
            signals[tf] = sig

        if not signals:
            return ConfluenceResult(
                symbol=symbol,
                timestamp=datetime.now(),
                dominant_trend=Trend.NEUTRAL,
            )

        result = self._compute_confluence(symbol, signals)
        self._last_results[symbol] = result
        return result

    def analyze_from_daily(
        self,
        symbol: str,
        daily_bars: List[dict],
    ) -> ConfluenceResult:
        """
        Analyze using only daily bars — construct synthetic lower TFs.

        When intraday bars aren't available, we approximate lower-TF
        analysis using different lookback windows on daily data:
          - "D"   = full daily series (long-term trend)
          - "4h"  = last 30 bars (swing)
          - "1h"  = last 15 bars (short-term)
          - "15m" = last 10 bars (very short)
          - "5m"  = last 5 bars (micro)
        """
        if not daily_bars or len(daily_bars) < 20:
            return ConfluenceResult(
                symbol=symbol,
                timestamp=datetime.now(),
                dominant_trend=Trend.NEUTRAL,
            )

        # Synthetic timeframe windows
        bars_by_tf = {
            "D": daily_bars,
            "4h": daily_bars[-30:] if len(daily_bars) >= 30 else daily_bars,
            "1h": daily_bars[-15:] if len(daily_bars) >= 15 else daily_bars,
            "15m": daily_bars[-10:] if len(daily_bars) >= 10 else daily_bars,
            "5m": daily_bars[-5:] if len(daily_bars) >= 5 else daily_bars,
        }

        # Override MIN_BARS for synthetic mode
        signals: Dict[str, TimeframeSignal] = {}
        for tf, bars in bars_by_tf.items():
            if len(bars) >= 3:
                sig = self._analyze_timeframe(tf, bars)
                signals[tf] = sig

        if not signals:
            return ConfluenceResult(
                symbol=symbol,
                timestamp=datetime.now(),
                dominant_trend=Trend.NEUTRAL,
            )

        return self._compute_confluence(symbol, signals)

    def confirms_trade(
        self,
        symbol: str,
        direction: int,
        bars_by_tf: Dict[str, List[dict]],
    ) -> bool:
        """
        Check if multi-TF analysis confirms a directional trade.

        Args:
            direction: +1 for long, -1 for short

        Returns:
            True if confluence supports the direction
        """
        result = self.analyze(symbol, bars_by_tf)
        if direction > 0:
            return result.confirms_long
        elif direction < 0:
            return result.confirms_short
        return True

    def get_signal(
        self,
        symbol: str,
        prices_by_tf: Dict[str, np.ndarray],
    ) -> TimeframeSignal:
        """
        Legacy API: get aggregated signal across timeframes.

        Maintained for backward compatibility.
        """
        # Convert price arrays to bar dicts
        bars_by_tf = {}
        for tf, prices in prices_by_tf.items():
            bars = [{"o": float(p), "h": float(p), "l": float(p),
                      "c": float(p), "v": 1e6} for p in prices]
            bars_by_tf[tf] = bars

        result = self.analyze(symbol, bars_by_tf)
        return TimeframeSignal(
            timeframe="multi",
            trend=result.dominant_trend,
            strength=abs(result.weighted_score),
            momentum=result.weighted_score,
        )

    # ── Per-Timeframe Analysis ──────────────────────────────────────

    def _analyze_timeframe(self, tf: str, bars: List[dict]) -> TimeframeSignal:
        """Analyze trend for a single timeframe."""
        closes = np.array([float(b["c"]) for b in bars])
        highs = np.array([float(b["h"]) for b in bars])
        lows = np.array([float(b["l"]) for b in bars])
        volumes = np.array([float(b["v"]) for b in bars])
        n = len(closes)

        # SMA crossover
        short_window = max(3, n // 4)
        long_window = max(6, n // 2)
        sma_short = float(np.mean(closes[-short_window:]))
        sma_long = float(np.mean(closes[-long_window:]))
        price = closes[-1]

        # Normalized cross: (short - long) / price
        sma_cross = (sma_short - sma_long) / max(price, 0.01)

        # Momentum: % return over lookback
        lookback = min(10, n - 1)
        momentum = (price / closes[-lookback - 1] - 1) if lookback > 0 else 0.0

        # RSI (simple)
        rsi = self._compute_rsi(closes)

        # Volume confirmation: recent volume > average
        if len(volumes) >= 10:
            avg_vol = float(np.mean(volumes[-21:-1])) if len(volumes) >= 21 else float(np.mean(volumes[:-1]))
            vol_confirm = float(volumes[-1]) > avg_vol * 0.8
        else:
            vol_confirm = True

        # Determine trend
        trend, strength = self._classify_trend(sma_cross, momentum, rsi, price, sma_short, sma_long)

        return TimeframeSignal(
            timeframe=tf,
            trend=trend,
            strength=strength,
            momentum=round(momentum, 4),
            rsi=round(rsi, 1),
            sma_cross=round(sma_cross, 6),
            volume_confirm=vol_confirm,
        )

    def _classify_trend(
        self,
        sma_cross: float,
        momentum: float,
        rsi: float,
        price: float,
        sma_short: float,
        sma_long: float,
    ) -> Tuple[Trend, float]:
        """Classify trend direction and strength."""
        score = 0.0

        # SMA cross signal
        if sma_cross > 0.002:
            score += 0.35
        elif sma_cross > 0:
            score += 0.15
        elif sma_cross < -0.002:
            score -= 0.35
        elif sma_cross < 0:
            score -= 0.15

        # Momentum
        if momentum > 0.02:
            score += 0.25
        elif momentum > 0:
            score += 0.10
        elif momentum < -0.02:
            score -= 0.25
        elif momentum < 0:
            score -= 0.10

        # RSI
        if rsi > 60:
            score += 0.20
        elif rsi > 50:
            score += 0.05
        elif rsi < 40:
            score -= 0.20
        elif rsi < 50:
            score -= 0.05

        # Price vs SMAs
        if price > sma_short > sma_long:
            score += 0.20
        elif price < sma_short < sma_long:
            score -= 0.20

        # Classify
        if score > 0.25:
            trend = Trend.BULLISH
        elif score < -0.25:
            trend = Trend.BEARISH
        else:
            trend = Trend.NEUTRAL

        strength = float(np.clip(abs(score), 0, 1))
        return trend, strength

    def _compute_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """RSI calculation."""
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses)) + 1e-10
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    # ── Confluence ──────────────────────────────────────────────────

    def _compute_confluence(
        self,
        symbol: str,
        signals: Dict[str, TimeframeSignal],
    ) -> ConfluenceResult:
        """Compute weighted confluence across timeframes."""
        bull_count = 0
        bear_count = 0
        neutral_count = 0
        weighted_sum = 0.0
        total_weight = 0.0

        for tf, sig in signals.items():
            w = TIMEFRAME_WEIGHTS.get(tf, 0.15)
            total_weight += w

            if sig.trend == Trend.BULLISH:
                bull_count += 1
                # Extra weight if volume confirms
                vol_bonus = 1.1 if sig.volume_confirm else 0.9
                weighted_sum += w * sig.strength * vol_bonus
            elif sig.trend == Trend.BEARISH:
                bear_count += 1
                vol_bonus = 1.1 if sig.volume_confirm else 0.9
                weighted_sum -= w * sig.strength * vol_bonus
            else:
                neutral_count += 1

        total_tf = len(signals)

        # Normalize weighted score to [-1, 1]
        if total_weight > 0:
            weighted_score = weighted_sum / total_weight
        else:
            weighted_score = 0.0
        weighted_score = float(np.clip(weighted_score, -1, 1))

        # Dominant trend
        if bull_count > bear_count and bull_count >= self.min_confirming:
            dominant = Trend.BULLISH
        elif bear_count > bull_count and bear_count >= self.min_confirming:
            dominant = Trend.BEARISH
        else:
            dominant = Trend.NEUTRAL

        # Confluence %
        max_agree = max(bull_count, bear_count)
        confluence_pct = (max_agree / total_tf * 100) if total_tf > 0 else 0

        # Confirms long/short
        confirms_long = (
            bull_count >= self.min_confirming and
            weighted_score > 0.15
        )
        confirms_short = (
            bear_count >= self.min_confirming and
            weighted_score < -0.15
        )

        # Caution: daily opposes intraday
        caution = False
        daily_trend = signals.get("D", TimeframeSignal("D", Trend.NEUTRAL, 0, 0)).trend
        hourly_trend = signals.get("1h", TimeframeSignal("1h", Trend.NEUTRAL, 0, 0)).trend
        if daily_trend != Trend.NEUTRAL and hourly_trend != Trend.NEUTRAL:
            if daily_trend != hourly_trend:
                caution = True
                # Reduce confirmation in cross-trend scenarios
                if daily_trend == Trend.BEARISH and confirms_long:
                    # Daily bearish but intraday bullish — risky long
                    confirms_long = bull_count >= self.min_confirming + 1
                elif daily_trend == Trend.BULLISH and confirms_short:
                    confirms_short = bear_count >= self.min_confirming + 1

        # Confluence scale for position sizing
        if total_tf > 0:
            agreement_ratio = max_agree / total_tf
            confluence_scale = 0.5 + agreement_ratio * 0.5  # 0.5 to 1.0
            if confluence_pct >= 80:
                confluence_scale = 1.2  # Bonus for strong agreement
        else:
            confluence_scale = 0.5

        return ConfluenceResult(
            symbol=symbol,
            timestamp=datetime.now(),
            signals=signals,
            bullish_count=bull_count,
            bearish_count=bear_count,
            neutral_count=neutral_count,
            total_timeframes=total_tf,
            weighted_score=round(weighted_score, 4),
            confluence_pct=round(confluence_pct, 1),
            confirms_long=confirms_long,
            confirms_short=confirms_short,
            caution=caution,
            dominant_trend=dominant,
            confluence_scale=round(confluence_scale, 3),
        )

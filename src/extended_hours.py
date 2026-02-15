"""
Extended Hours Trading — Pre/Post Market Scanner
==================================================

Tracks extended hours price action for edge detection:
  1. Gap detection: overnight gaps > 2% signal momentum continuation/reversal
  2. Pre-market momentum scanner: identifies movers before the open
  3. Post-market activity tracking: earnings reactions, after-hours volume
  4. Extended hours signal generation for early positioning

Requires Alpaca API (supports extended hours data).
Falls back to neutral signals when no extended hours data is available.

Usage:
    scanner = ExtendedHoursScanner()

    # Before market open:
    signals = scanner.get_extended_hours_signals(symbols, api_getter)
    for sig in signals:
        print(f"{sig.symbol}: gap={sig.gap_pct:+.2%} pm_momentum={sig.pm_momentum:+.2%}")

    # Gap detection:
    gaps = scanner.detect_gaps(symbols, bars_map)

Author: Tier 1 Implementation — Feb 2026
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


# ============================================================================
# DATA MODELS
# ============================================================================

class GapType(Enum):
    """Type of overnight gap."""
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"
    NO_GAP = "no_gap"


class ExtendedHoursAction(Enum):
    """Recommended action from extended hours analysis."""
    STRONG_BUY = "strong_buy"       # Large gap up + volume confirmation
    BUY = "buy"                     # Moderate gap up or pre-market momentum
    WATCH = "watch"                 # Notable activity, no clear signal
    SELL = "sell"                   # Gap down, negative pre-market
    STRONG_SELL = "strong_sell"     # Large gap down + volume confirmation
    NEUTRAL = "neutral"             # No significant extended hours activity


@dataclass
class GapSignal:
    """Detected overnight gap for a symbol."""
    symbol: str
    gap_type: GapType
    gap_pct: float                  # Signed % gap (positive = up)
    prev_close: float               # Previous day close
    open_price: float               # Current day open (or pre-market)
    volume_ratio: float             # Pre-market volume vs average
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExtendedHoursSignal:
    """Complete extended hours signal for a symbol."""
    symbol: str
    action: ExtendedHoursAction
    gap_pct: float                  # Overnight gap %
    gap_type: GapType
    pm_momentum: float              # Pre-market price change from open
    pm_volume_ratio: float          # Pre-market volume vs normal
    confidence: float               # 0-1
    reasons: List[str]
    position_bias: float            # [-1, 1] position bias from EH analysis
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExtendedHoursConfig:
    """Configuration for extended hours scanner."""
    gap_threshold_pct: float = 0.02     # Min gap to consider (2%)
    large_gap_pct: float = 0.05         # Large gap threshold (5%)
    pm_momentum_threshold: float = 0.01  # Min pre-market move (1%)
    volume_confirmation_ratio: float = 1.5  # Pre-market volume > 1.5x avg
    min_price: float = 5.0              # Skip penny stocks
    max_signals: int = 10               # Max signals to return
    gap_fade_threshold: float = 0.03    # Fade gaps > 3% (mean reversion)
    # Pre-market time window (ET)
    pm_start_hour: int = 4              # Pre-market opens 4:00 AM ET
    pm_end_hour: int = 9               # Regular session 9:30 AM ET
    # Post-market window
    ah_start_hour: int = 16             # After-hours starts 4:00 PM ET
    ah_end_hour: int = 20              # After-hours ends 8:00 PM ET


# ============================================================================
# EXTENDED HOURS SCANNER
# ============================================================================

class ExtendedHoursScanner:
    """
    Scans for pre-market and post-market trading signals.

    Key signals:
      1. Overnight gaps (close-to-open): Large gaps indicate news/earnings
      2. Pre-market momentum: Direction of early trading
      3. Volume anomalies: High pre-market volume = institutional interest
      4. Gap fade candidates: >3% gaps with declining pre-market momentum
    """

    def __init__(self, config: Optional[ExtendedHoursConfig] = None):
        self.cfg = config or ExtendedHoursConfig()
        self._gap_history: Dict[str, List[GapSignal]] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_extended_hours_signals(
        self,
        symbols: List[str],
        bars_getter: Optional[Callable] = None,
        current_bars: Optional[Dict[str, List[dict]]] = None,
    ) -> List[ExtendedHoursSignal]:
        """
        Generate extended hours signals for a list of symbols.

        Args:
            symbols: List of symbols to scan
            bars_getter: Optional callable(symbol, limit) -> list of bar dicts
            current_bars: Optional pre-fetched bars map {symbol: [bars]}

        Returns:
            List of ExtendedHoursSignal, sorted by confidence (highest first)
        """
        signals: List[ExtendedHoursSignal] = []

        for sym in symbols:
            bars = None
            if current_bars and sym in current_bars:
                bars = current_bars[sym]
            elif bars_getter is not None:
                try:
                    bars = bars_getter(sym, 5)  # Last 5 bars
                except Exception as e:
                    logger.debug(f"Failed to get bars for {sym}: {e}")

            if bars is None or len(bars) < 2:
                continue

            sig = self._analyze_symbol(sym, bars)
            if sig is not None and sig.action != ExtendedHoursAction.NEUTRAL:
                signals.append(sig)

        # Sort by confidence, take top N
        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals[:self.cfg.max_signals]

    def detect_gaps(
        self,
        bars_map: Dict[str, List[dict]],
    ) -> List[GapSignal]:
        """
        Detect overnight gaps for all symbols.

        Args:
            bars_map: Dict of symbol -> list of bar dicts (needs ≥2 bars)

        Returns:
            List of GapSignal for symbols with gaps > threshold
        """
        gaps: List[GapSignal] = []

        for sym, bars in bars_map.items():
            if len(bars) < 2:
                continue

            gap = self._compute_gap(sym, bars)
            if gap is not None and abs(gap.gap_pct) >= self.cfg.gap_threshold_pct:
                gaps.append(gap)

                # Track history
                if sym not in self._gap_history:
                    self._gap_history[sym] = []
                self._gap_history[sym].append(gap)
                # Keep last 20 gaps
                self._gap_history[sym] = self._gap_history[sym][-20:]

        # Sort by absolute gap size
        gaps.sort(key=lambda g: abs(g.gap_pct), reverse=True)
        return gaps

    def get_gap_stats(self, symbol: str) -> Dict[str, float]:
        """Get historical gap statistics for a symbol."""
        history = self._gap_history.get(symbol, [])
        if not history:
            return {"avg_gap": 0.0, "max_gap": 0.0, "gap_up_rate": 0.5, "count": 0}

        gaps = [g.gap_pct for g in history]
        up_gaps = [g for g in gaps if g > 0]

        return {
            "avg_gap": float(np.mean(np.abs(gaps))) if np else sum(abs(g) for g in gaps) / len(gaps),
            "max_gap": max(abs(g) for g in gaps),
            "gap_up_rate": len(up_gaps) / len(gaps) if gaps else 0.5,
            "count": len(gaps),
        }

    def is_extended_hours(self) -> bool:
        """Check if current time is in extended hours (pre-market or after-hours)."""
        now = datetime.now()
        hour = now.hour
        # Pre-market: 4 AM - 9:30 AM
        if self.cfg.pm_start_hour <= hour < self.cfg.pm_end_hour:
            return True
        # After-hours: 4 PM - 8 PM
        if self.cfg.ah_start_hour <= hour < self.cfg.ah_end_hour:
            return True
        return False

    def get_pre_market_movers(
        self,
        bars_map: Dict[str, List[dict]],
        top_n: int = 5,
    ) -> List[Dict[str, float]]:
        """
        Identify top pre-market movers by gap size and volume.

        Returns:
            List of dicts with symbol, gap_pct, volume_ratio, score
        """
        movers = []

        for sym, bars in bars_map.items():
            if len(bars) < 2:
                continue

            gap = self._compute_gap(sym, bars)
            if gap is None:
                continue

            price = float(bars[-1].get("c", 0))
            if price < self.cfg.min_price:
                continue

            score = abs(gap.gap_pct) * gap.volume_ratio
            movers.append({
                "symbol": sym,
                "gap_pct": gap.gap_pct,
                "volume_ratio": gap.volume_ratio,
                "price": price,
                "score": score,
            })

        movers.sort(key=lambda m: m["score"], reverse=True)
        return movers[:top_n]

    # ------------------------------------------------------------------ #
    # Internal analysis
    # ------------------------------------------------------------------ #

    def _analyze_symbol(self, symbol: str, bars: List[dict]) -> Optional[ExtendedHoursSignal]:
        """Analyze extended hours activity for a single symbol."""
        if len(bars) < 2:
            return None

        price = float(bars[-1].get("c", 0))
        if price < self.cfg.min_price:
            return None

        # Compute gap
        gap = self._compute_gap(symbol, bars)
        if gap is None:
            return self._neutral_signal(symbol)

        # Compute pre-market momentum (intra-bar: open to close of latest bar)
        latest = bars[-1]
        bar_open = float(latest.get("o", price))
        pm_momentum = (price - bar_open) / bar_open if bar_open > 0 else 0.0

        # Volume analysis
        volumes = [float(b.get("v", 0)) for b in bars]
        avg_vol = float(np.mean(volumes[:-1])) if len(volumes) > 1 and np else (
            sum(volumes[:-1]) / max(len(volumes) - 1, 1)
        )
        current_vol = volumes[-1] if volumes else 0
        vol_ratio = current_vol / max(avg_vol, 1)

        # Classify signal
        action, confidence, reasons, bias = self._classify_signal(
            gap.gap_pct, gap.gap_type, pm_momentum, vol_ratio,
        )

        return ExtendedHoursSignal(
            symbol=symbol,
            action=action,
            gap_pct=gap.gap_pct,
            gap_type=gap.gap_type,
            pm_momentum=pm_momentum,
            pm_volume_ratio=vol_ratio,
            confidence=confidence,
            reasons=reasons,
            position_bias=bias,
        )

    def _compute_gap(self, symbol: str, bars: List[dict]) -> Optional[GapSignal]:
        """Compute overnight gap from bar data."""
        if len(bars) < 2:
            return None

        prev_close = float(bars[-2].get("c", 0))
        curr_open = float(bars[-1].get("o", 0))
        curr_vol = float(bars[-1].get("v", 0))
        prev_vol = float(bars[-2].get("v", 1))

        if prev_close <= 0 or curr_open <= 0:
            return None

        gap_pct = (curr_open - prev_close) / prev_close
        vol_ratio = curr_vol / max(prev_vol, 1)

        if gap_pct > self.cfg.gap_threshold_pct:
            gap_type = GapType.GAP_UP
        elif gap_pct < -self.cfg.gap_threshold_pct:
            gap_type = GapType.GAP_DOWN
        else:
            gap_type = GapType.NO_GAP

        return GapSignal(
            symbol=symbol,
            gap_type=gap_type,
            gap_pct=gap_pct,
            prev_close=prev_close,
            open_price=curr_open,
            volume_ratio=vol_ratio,
        )

    def _classify_signal(
        self,
        gap_pct: float,
        gap_type: GapType,
        pm_momentum: float,
        vol_ratio: float,
    ) -> Tuple[ExtendedHoursAction, float, List[str], float]:
        """
        Classify extended hours signal into action.

        Returns:
            (action, confidence, reasons, position_bias)
        """
        reasons = []
        confidence = 0.0
        bias = 0.0

        abs_gap = abs(gap_pct)

        # Large gap up with volume confirmation
        if gap_pct >= self.cfg.large_gap_pct and vol_ratio >= self.cfg.volume_confirmation_ratio:
            # Check for gap fade (large gaps often reverse)
            if abs_gap >= self.cfg.gap_fade_threshold and pm_momentum < 0:
                reasons.append(f"Large gap up {gap_pct:+.1%} fading (pm={pm_momentum:+.1%})")
                reasons.append(f"Volume {vol_ratio:.1f}x avg")
                return ExtendedHoursAction.WATCH, 0.5, reasons, -0.2

            reasons.append(f"Large gap up {gap_pct:+.1%} with volume {vol_ratio:.1f}x")
            if pm_momentum > 0:
                reasons.append(f"Pre-market momentum confirms ({pm_momentum:+.1%})")
                return ExtendedHoursAction.STRONG_BUY, 0.8, reasons, 0.8

            return ExtendedHoursAction.BUY, 0.6, reasons, 0.5

        # Large gap down with volume
        if gap_pct <= -self.cfg.large_gap_pct and vol_ratio >= self.cfg.volume_confirmation_ratio:
            if abs_gap >= self.cfg.gap_fade_threshold and pm_momentum > 0:
                reasons.append(f"Large gap down {gap_pct:+.1%} recovering (pm={pm_momentum:+.1%})")
                return ExtendedHoursAction.WATCH, 0.5, reasons, 0.2

            reasons.append(f"Large gap down {gap_pct:+.1%} with volume {vol_ratio:.1f}x")
            if pm_momentum < 0:
                reasons.append(f"Pre-market momentum confirms ({pm_momentum:+.1%})")
                return ExtendedHoursAction.STRONG_SELL, 0.8, reasons, -0.8

            return ExtendedHoursAction.SELL, 0.6, reasons, -0.5

        # Moderate gap up
        if gap_pct >= self.cfg.gap_threshold_pct:
            reasons.append(f"Gap up {gap_pct:+.1%}")
            confidence = 0.4 + 0.2 * (vol_ratio > 1.0)
            bias = 0.3

            if pm_momentum > self.cfg.pm_momentum_threshold:
                reasons.append(f"PM momentum {pm_momentum:+.1%}")
                confidence += 0.15
                bias += 0.2
                return ExtendedHoursAction.BUY, min(confidence, 0.9), reasons, min(bias, 1.0)

            return ExtendedHoursAction.WATCH, confidence, reasons, bias

        # Moderate gap down
        if gap_pct <= -self.cfg.gap_threshold_pct:
            reasons.append(f"Gap down {gap_pct:+.1%}")
            confidence = 0.4 + 0.2 * (vol_ratio > 1.0)
            bias = -0.3

            if pm_momentum < -self.cfg.pm_momentum_threshold:
                reasons.append(f"PM momentum {pm_momentum:+.1%}")
                confidence += 0.15
                bias -= 0.2
                return ExtendedHoursAction.SELL, min(confidence, 0.9), reasons, max(bias, -1.0)

            return ExtendedHoursAction.WATCH, confidence, reasons, bias

        # No significant gap — check pre-market momentum only
        if abs(pm_momentum) > self.cfg.pm_momentum_threshold:
            reasons.append(f"Pre-market momentum {pm_momentum:+.1%} (no gap)")
            if pm_momentum > 0:
                return ExtendedHoursAction.WATCH, 0.35, reasons, 0.15
            else:
                return ExtendedHoursAction.WATCH, 0.35, reasons, -0.15

        return ExtendedHoursAction.NEUTRAL, 0.0, [], 0.0

    def _neutral_signal(self, symbol: str) -> ExtendedHoursSignal:
        """Return neutral signal for a symbol."""
        return ExtendedHoursSignal(
            symbol=symbol,
            action=ExtendedHoursAction.NEUTRAL,
            gap_pct=0.0,
            gap_type=GapType.NO_GAP,
            pm_momentum=0.0,
            pm_volume_ratio=1.0,
            confidence=0.0,
            reasons=[],
            position_bias=0.0,
        )

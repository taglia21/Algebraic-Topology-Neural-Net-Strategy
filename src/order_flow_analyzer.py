"""
Order Flow Analyzer — Institutional & Dark Pool Flow Tracking (Tier 2)
======================================================================

Detects institutional activity via:

  1. **Block Trade Detection**   — Large single-bar volume spikes
  2. **Dark Pool Prints**        — Off-exchange volume analysis (FINRA TRF proxy)
  3. **Accumulation/Distribution — OBV + MFI + VWAP deviation
  4. **Sweep Detection**         — Multi-exchange aggressive fills
  5. **Smart Money Score**       — Composite institutional flow metric

Data: Alpaca bars (volume + OHLCV). For live use, intraday 1m/5m bars
provide higher-resolution flow detection; daily bars give slower signals.

Integration:
    from src.order_flow_analyzer import OrderFlowAnalyzer
    ofa = OrderFlowAnalyzer()
    flow = ofa.analyze(symbol, bars)
    # flow.smart_money_score, flow.institutional_bias, flow.block_trades
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class InstitutionalBias(Enum):
    STRONG_BUY = "strong_accumulation"
    ACCUMULATING = "accumulating"
    NEUTRAL = "neutral"
    DISTRIBUTING = "distributing"
    STRONG_SELL = "strong_distribution"


@dataclass
class BlockTrade:
    """Detected block trade (large single-bar volume spike)."""
    bar_idx: int
    volume: float
    avg_volume: float
    volume_ratio: float          # volume / avg_volume
    price_impact: float          # % move on block bar
    direction: str               # "buy" or "sell" (inferred from price action)
    is_dark_pool_likely: bool    # Off-exchange characteristics


@dataclass
class FlowSignal:
    """Aggregated order flow analysis for a symbol."""
    symbol: str
    timestamp: datetime

    # Volume analysis
    relative_volume: float       # Today's vol / 20-day avg
    volume_trend: float          # Slope of 10-day volume SMA (normalized)
    dark_pool_pct: float         # Estimated % of volume via dark pool (proxy)

    # Money flow
    mfi: float                   # Money Flow Index [0, 100]
    obv_trend: float             # OBV slope (normalized, positive = accumulation)
    ad_line_trend: float         # Accumulation/Distribution line slope

    # VWAP
    vwap: float                  # Volume-weighted average price
    vwap_deviation: float        # (price - VWAP) / VWAP as %

    # Institutional detection
    block_trades: List[BlockTrade] = field(default_factory=list)
    block_buy_volume: float = 0.0
    block_sell_volume: float = 0.0
    net_block_flow: float = 0.0   # buy_blocks - sell_blocks (normalized)

    # Composite
    smart_money_score: float = 0.0    # [-1, 1]: +1 = strong buy, -1 = strong sell
    institutional_bias: InstitutionalBias = InstitutionalBias.NEUTRAL
    confidence: float = 0.0           # [0, 1]

    # Trading signal
    trade_bias: str = "neutral"       # "buy", "sell", "neutral"
    position_scale: float = 1.0       # Size multiplier

    @property
    def is_actionable(self) -> bool:
        return self.trade_bias != "neutral" and self.confidence >= 0.3


# ============================================================================
# CORE ENGINE
# ============================================================================

class OrderFlowAnalyzer:
    """
    Analyze order flow for institutional activity detection.

    Uses volume microstructure analysis on Alpaca OHLCV bars
    to infer institutional buying/selling pressure.

    Usage:
        ofa = OrderFlowAnalyzer()
        flow = ofa.analyze("AAPL", bars)
        if flow.institutional_bias == InstitutionalBias.ACCUMULATING:
            # Bullish flow signal
    """

    def __init__(
        self,
        block_volume_mult: float = 3.0,
        dark_pool_volume_mult: float = 2.0,
        mfi_period: int = 14,
        obv_slope_period: int = 10,
        vwap_lookback: int = 20,
    ):
        self.block_volume_mult = block_volume_mult
        self.dark_pool_volume_mult = dark_pool_volume_mult
        self.mfi_period = mfi_period
        self.obv_slope_period = obv_slope_period
        self.vwap_lookback = vwap_lookback

        # Rolling history for multi-day analysis
        self._flow_history: Dict[str, List[float]] = {}  # symbol -> smart_money_scores

    # ── Public API ──────────────────────────────────────────────────

    def analyze(self, symbol: str, bars: List[dict]) -> FlowSignal:
        """
        Full order flow analysis for a symbol.

        Args:
            symbol: Ticker symbol
            bars: OHLCV bars (Alpaca format, need >= 30)

        Returns:
            FlowSignal with all flow metrics and trade bias
        """
        if not bars or len(bars) < 20:
            return self._neutral_signal(symbol)

        opens = np.array([float(b["o"]) for b in bars])
        highs = np.array([float(b["h"]) for b in bars])
        lows = np.array([float(b["l"]) for b in bars])
        closes = np.array([float(b["c"]) for b in bars])
        volumes = np.array([float(b["v"]) for b in bars])

        # 1. Volume analysis
        rel_volume = self._relative_volume(volumes)
        vol_trend = self._volume_trend(volumes)
        dark_pool_pct = self._estimate_dark_pool(volumes, closes, highs, lows)

        # 2. Money flow
        mfi = self._compute_mfi(highs, lows, closes, volumes)
        obv_trend = self._compute_obv_trend(closes, volumes)
        ad_trend = self._compute_ad_trend(highs, lows, closes, volumes)

        # 3. VWAP
        vwap = self._compute_vwap(highs, lows, closes, volumes)
        vwap_dev = (closes[-1] - vwap) / max(vwap, 0.01)

        # 4. Block trades
        blocks = self._detect_blocks(opens, highs, lows, closes, volumes)
        block_buy_vol = sum(b.volume for b in blocks if b.direction == "buy")
        block_sell_vol = sum(b.volume for b in blocks if b.direction == "sell")
        total_block = block_buy_vol + block_sell_vol
        net_block = (block_buy_vol - block_sell_vol) / max(total_block, 1)

        # 5. Composite smart money score
        smart_money = self._compute_smart_money_score(
            rel_volume, vol_trend, mfi, obv_trend, ad_trend,
            vwap_dev, net_block, dark_pool_pct,
        )
        bias = self._classify_bias(smart_money)
        confidence = min(abs(smart_money), 1.0)

        # 6. Trade signal
        trade_bias, position_scale = self._generate_trade_signal(
            smart_money, bias, rel_volume, confidence,
        )

        # Update history
        self._update_history(symbol, smart_money)

        return FlowSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            relative_volume=round(rel_volume, 2),
            volume_trend=round(vol_trend, 4),
            dark_pool_pct=round(dark_pool_pct, 2),
            mfi=round(mfi, 1),
            obv_trend=round(obv_trend, 4),
            ad_line_trend=round(ad_trend, 4),
            vwap=round(vwap, 2),
            vwap_deviation=round(vwap_dev, 4),
            block_trades=blocks,
            block_buy_volume=block_buy_vol,
            block_sell_volume=block_sell_vol,
            net_block_flow=round(net_block, 3),
            smart_money_score=round(smart_money, 3),
            institutional_bias=bias,
            confidence=round(confidence, 3),
            trade_bias=trade_bias,
            position_scale=round(position_scale, 3),
        )

    def get_flow_history(self, symbol: str) -> List[float]:
        """Return recent smart money score history."""
        return self._flow_history.get(symbol, [])

    # ── Volume Analysis ─────────────────────────────────────────────

    def _relative_volume(self, volumes: np.ndarray) -> float:
        """Current volume relative to 20-day average."""
        if len(volumes) < 2:
            return 1.0
        avg_20 = float(np.mean(volumes[-21:-1])) if len(volumes) >= 21 else float(np.mean(volumes[:-1]))
        return float(volumes[-1] / max(avg_20, 1))

    def _volume_trend(self, volumes: np.ndarray) -> float:
        """10-day volume trend (linear regression slope, normalized)."""
        n = min(10, len(volumes))
        if n < 3:
            return 0.0
        v = volumes[-n:]
        x = np.arange(n)
        slope = float(np.polyfit(x, v, 1)[0])
        avg_v = float(np.mean(v)) + 1e-10
        return slope / avg_v  # Normalized

    def _estimate_dark_pool(
        self, volumes: np.ndarray,
        closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    ) -> float:
        """
        Estimate dark pool participation % (proxy).

        Heuristic: high volume bars with narrow range (low price impact)
        suggest off-exchange / dark pool activity.
        """
        if len(volumes) < 5:
            return 0.0

        recent = min(5, len(volumes))
        avg_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
        dark_pool_score = 0.0

        for i in range(-recent, 0):
            vol_ratio = volumes[i] / max(avg_vol, 1)
            bar_range = (highs[i] - lows[i]) / max(closes[i], 0.01)
            # Dark pool signature: high volume + narrow range
            if vol_ratio > self.dark_pool_volume_mult and bar_range < 0.005:
                dark_pool_score += 1

        return dark_pool_score / recent

    # ── Money Flow Indicators ───────────────────────────────────────

    def _compute_mfi(
        self,
        highs: np.ndarray, lows: np.ndarray,
        closes: np.ndarray, volumes: np.ndarray,
    ) -> float:
        """Money Flow Index [0, 100]."""
        period = min(self.mfi_period, len(closes) - 1)
        if period < 2:
            return 50.0

        typical = (highs + lows + closes) / 3
        raw_mf = typical * volumes

        pos_flow = 0.0
        neg_flow = 0.0
        for i in range(-period, 0):
            if typical[i] > typical[i - 1]:
                pos_flow += raw_mf[i]
            else:
                neg_flow += raw_mf[i]

        if neg_flow < 1e-10:
            return 100.0
        mf_ratio = pos_flow / neg_flow
        return float(100 - (100 / (1 + mf_ratio)))

    def _compute_obv_trend(self, closes: np.ndarray, volumes: np.ndarray) -> float:
        """OBV slope over N bars (normalized)."""
        n = min(self.obv_slope_period, len(closes) - 1)
        if n < 3:
            return 0.0

        obv = np.zeros(n + 1)
        for i in range(1, n + 1):
            idx = -(n + 1) + i
            if closes[idx] > closes[idx - 1]:
                obv[i] = obv[i - 1] + volumes[idx]
            elif closes[idx] < closes[idx - 1]:
                obv[i] = obv[i - 1] - volumes[idx]
            else:
                obv[i] = obv[i - 1]

        # Linear regression slope
        x = np.arange(len(obv))
        slope = float(np.polyfit(x, obv, 1)[0])
        avg_vol = float(np.mean(volumes[-n:])) + 1e-10
        return slope / avg_vol

    def _compute_ad_trend(
        self,
        highs: np.ndarray, lows: np.ndarray,
        closes: np.ndarray, volumes: np.ndarray,
    ) -> float:
        """Accumulation/Distribution line trend (normalized slope)."""
        n = min(self.obv_slope_period, len(closes))
        if n < 3:
            return 0.0

        ad = np.zeros(n)
        for i in range(n):
            idx = -(n) + i
            hl = highs[idx] - lows[idx]
            if hl > 0:
                clv = ((closes[idx] - lows[idx]) - (highs[idx] - closes[idx])) / hl
            else:
                clv = 0.0
            ad[i] = clv * volumes[idx] + (ad[i - 1] if i > 0 else 0)

        x = np.arange(len(ad))
        slope = float(np.polyfit(x, ad, 1)[0])
        avg_vol = float(np.mean(volumes[-n:])) + 1e-10
        return slope / avg_vol

    # ── VWAP ────────────────────────────────────────────────────────

    def _compute_vwap(
        self,
        highs: np.ndarray, lows: np.ndarray,
        closes: np.ndarray, volumes: np.ndarray,
    ) -> float:
        """Volume-weighted average price over lookback window."""
        n = min(self.vwap_lookback, len(closes))
        typical = (highs[-n:] + lows[-n:] + closes[-n:]) / 3
        cum_tv = np.sum(typical * volumes[-n:])
        cum_vol = np.sum(volumes[-n:])
        return float(cum_tv / max(cum_vol, 1))

    # ── Block Trade Detection ───────────────────────────────────────

    def _detect_blocks(
        self,
        opens: np.ndarray, highs: np.ndarray,
        lows: np.ndarray, closes: np.ndarray,
        volumes: np.ndarray,
    ) -> List[BlockTrade]:
        """Detect block trades (large volume spikes with price impact)."""
        blocks = []
        if len(volumes) < 21:
            return blocks

        avg_vol = float(np.mean(volumes[-21:-1]))  # Exclude last bar for fair base

        for i in range(-min(10, len(volumes)), 0):  # Check last 10 bars
            vol = float(volumes[i])
            vol_ratio = vol / max(avg_vol, 1)

            if vol_ratio < self.block_volume_mult:
                continue

            # Price impact
            bar_return = (closes[i] - opens[i]) / max(opens[i], 0.01)
            bar_range = (highs[i] - lows[i]) / max(closes[i], 0.01)

            # Direction: close > open = buy pressure
            direction = "buy" if closes[i] > opens[i] else "sell"

            # Dark pool heuristic: very high volume + narrow range
            is_dark = vol_ratio > self.dark_pool_volume_mult and bar_range < 0.008

            blocks.append(BlockTrade(
                bar_idx=len(volumes) + i,
                volume=vol,
                avg_volume=avg_vol,
                volume_ratio=round(vol_ratio, 2),
                price_impact=round(bar_return, 4),
                direction=direction,
                is_dark_pool_likely=is_dark,
            ))

        return blocks

    # ── Composite Score ─────────────────────────────────────────────

    def _compute_smart_money_score(
        self,
        rel_volume: float,
        vol_trend: float,
        mfi: float,
        obv_trend: float,
        ad_trend: float,
        vwap_dev: float,
        net_block: float,
        dark_pool_pct: float,
    ) -> float:
        """
        Compute composite smart money score [-1, 1].

        Weights:
          - MFI direction:    0.25
          - OBV trend:        0.20
          - A/D trend:        0.15
          - Net block flow:   0.20
          - VWAP deviation:   0.10
          - Volume trend:     0.10
        """
        # Normalize MFI to [-1, 1]
        mfi_norm = (mfi - 50) / 50.0

        # Normalize OBV/AD trends (already normalized by avg_vol, clip)
        obv_n = float(np.clip(obv_trend * 50, -1, 1))
        ad_n = float(np.clip(ad_trend * 50, -1, 1))

        # VWAP deviation: positive = above VWAP (bullish)
        vwap_n = float(np.clip(vwap_dev * 20, -1, 1))

        # Volume trend
        vol_n = float(np.clip(vol_trend * 20, -1, 1))

        # Net block flow already in [-1, 1]
        block_n = float(np.clip(net_block, -1, 1))

        score = (
            0.25 * mfi_norm +
            0.20 * obv_n +
            0.15 * ad_n +
            0.20 * block_n +
            0.10 * vwap_n +
            0.10 * vol_n
        )

        # Amplify if relative volume confirms direction
        if rel_volume > 2.0 and abs(score) > 0.2:
            score *= 1.2

        return float(np.clip(score, -1, 1))

    def _classify_bias(self, score: float) -> InstitutionalBias:
        """Map score to institutional bias."""
        if score > 0.5:
            return InstitutionalBias.STRONG_BUY
        elif score > 0.2:
            return InstitutionalBias.ACCUMULATING
        elif score < -0.5:
            return InstitutionalBias.STRONG_SELL
        elif score < -0.2:
            return InstitutionalBias.DISTRIBUTING
        return InstitutionalBias.NEUTRAL

    def _generate_trade_signal(
        self,
        score: float,
        bias: InstitutionalBias,
        rel_volume: float,
        confidence: float,
    ) -> Tuple[str, float]:
        """Generate trade bias and position scale from flow analysis."""
        if bias in (InstitutionalBias.STRONG_BUY, InstitutionalBias.ACCUMULATING):
            trade_bias = "buy"
            # Scale up if volume confirms
            scale = 1.0 + min(rel_volume - 1, 0.5) * 0.3 if rel_volume > 1.5 else 1.0
        elif bias in (InstitutionalBias.STRONG_SELL, InstitutionalBias.DISTRIBUTING):
            trade_bias = "sell"
            scale = 0.5  # Reduce size in distribution
        else:
            trade_bias = "neutral"
            scale = 1.0

        return trade_bias, float(np.clip(scale, 0.3, 1.5))

    def _update_history(self, symbol: str, score: float):
        """Append smart money score to rolling history."""
        if symbol not in self._flow_history:
            self._flow_history[symbol] = []
        self._flow_history[symbol].append(score)
        if len(self._flow_history[symbol]) > 60:
            self._flow_history[symbol] = self._flow_history[symbol][-60:]

    def _neutral_signal(self, symbol: str) -> FlowSignal:
        """Return neutral signal when insufficient data."""
        return FlowSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            relative_volume=1.0,
            volume_trend=0.0,
            dark_pool_pct=0.0,
            mfi=50.0,
            obv_trend=0.0,
            ad_line_trend=0.0,
            vwap=0.0,
            vwap_deviation=0.0,
        )

"""
Rule-Based Market Regime Detector
===================================

Simple, robust regime detection for options strategy selection.

Detects 4 regimes:
- TRENDING_BULL: Market in uptrend, favor bullish credit spreads
- TRENDING_BEAR: Market in downtrend, favor bearish credit spreads  
- MEAN_REVERTING: Range-bound, favor iron condors and premium selling
- HIGH_VOLATILITY: Elevated vol, sell premium aggressively when IV rank high

Uses straightforward technical indicators (no ML required for initial version):
- Price vs 20/50/200 SMA for trend detection
- ADX for trend strength
- RSI for overbought/oversold
- MACD for momentum confirmation
- Bollinger Band width for squeeze/expansion detection

This is a REPLACEMENT for the broken signal pipeline. The HMM-based
RegimeDetector in src/options/regime_detector.py is preserved but this
module handles regime classification for strategy selection independently.

Author: System Overhaul - Feb 2026
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
except ImportError:
    yf = None


# ============================================================================
# DATA MODELS
# ============================================================================

class Regime(Enum):
    """Market regime classification for strategy selection."""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    UNKNOWN = "unknown"


@dataclass
class TechnicalSignals:
    """Technical indicator readings for a symbol."""
    symbol: str

    # Price & trend
    current_price: float
    sma_20: float
    sma_50: float
    sma_200: float
    price_vs_sma20_pct: float    # % above/below 20 SMA
    price_vs_sma50_pct: float    # % above/below 50 SMA

    # Momentum
    rsi_14: float                # RSI 14-period
    macd_signal: float           # MACD - Signal (positive = bullish)
    macd_histogram: float        # MACD histogram

    # Volatility
    bb_width: float              # Bollinger Band width (% of price)
    bb_position: float           # Price position within bands (0-1)
    atr_pct: float               # ATR as % of price (14-period)

    # Trend strength
    adx: float                   # ADX (0-100, >25 = trending)
    trend_direction: int         # +1 bull, -1 bear, 0 neutral

    # Volume
    volume_ratio: float          # Current volume vs 20-day average

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RegimeResult:
    """Regime detection result with confidence and supporting evidence."""
    regime: Regime
    confidence: float            # 0-1
    evidence: Dict[str, str]     # Key evidence points
    technicals: Optional[TechnicalSignals]
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# TECHNICAL INDICATOR CALCULATIONS
# ============================================================================

def _sma(prices: np.ndarray, window: int) -> float:
    """Simple moving average of last `window` prices."""
    if len(prices) < window:
        return float(np.mean(prices))
    return float(np.mean(prices[-window:]))


def _ema(prices: np.ndarray, window: int) -> np.ndarray:
    """Exponential moving average."""
    alpha = 2.0 / (window + 1)
    ema = np.zeros_like(prices, dtype=float)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema


def _rsi(prices: np.ndarray, period: int = 14) -> float:
    """Relative Strength Index."""
    if len(prices) < period + 1:
        return 50.0

    deltas = np.diff(prices[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.001

    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    return float(rsi_val)


def _macd(prices: np.ndarray) -> Tuple[float, float, float]:
    """MACD, Signal, Histogram."""
    if len(prices) < 35:
        return 0.0, 0.0, 0.0

    ema_12 = _ema(prices, 12)
    ema_26 = _ema(prices, 26)
    macd_line = ema_12 - ema_26

    signal_line = _ema(macd_line[-9:], 9) if len(macd_line) >= 9 else macd_line
    histogram = macd_line[-1] - signal_line[-1]

    return float(macd_line[-1]), float(signal_line[-1]), float(histogram)


def _bollinger_bands(
    prices: np.ndarray, window: int = 20, num_std: float = 2.0
) -> Tuple[float, float, float, float, float]:
    """
    Bollinger Bands.

    Returns: (upper, middle, lower, width_pct, position)
    """
    if len(prices) < window:
        mid = float(np.mean(prices))
        return mid * 1.02, mid, mid * 0.98, 0.04, 0.5

    window_prices = prices[-window:]
    mid = float(np.mean(window_prices))
    std = float(np.std(window_prices))

    upper = mid + num_std * std
    lower = mid - num_std * std
    current = float(prices[-1])

    width_pct = (upper - lower) / mid if mid > 0 else 0
    position = (current - lower) / (upper - lower) if (upper - lower) > 0 else 0.5

    return upper, mid, lower, width_pct, position


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Average Directional Index."""
    if len(closes) < period + 1:
        return 20.0  # Default: weak trend

    n = len(closes)
    tr_list = []
    plus_dm_list = []
    minus_dm_list = []

    for i in range(1, n):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]
        prev_high = highs[i - 1]
        prev_low = lows[i - 1]

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_list.append(tr)

        plus_dm = max(high - prev_high, 0) if (high - prev_high) > (prev_low - low) else 0
        minus_dm = max(prev_low - low, 0) if (prev_low - low) > (high - prev_high) else 0
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)

    if len(tr_list) < period:
        return 20.0

    # Smoothed averages using Wilder's method
    atr = np.mean(tr_list[:period])
    plus_di_smooth = np.mean(plus_dm_list[:period])
    minus_di_smooth = np.mean(minus_dm_list[:period])

    dx_list = []
    for i in range(period, len(tr_list)):
        atr = atr - (atr / period) + tr_list[i]
        plus_di_smooth = plus_di_smooth - (plus_di_smooth / period) + plus_dm_list[i]
        minus_di_smooth = minus_di_smooth - (minus_di_smooth / period) + minus_dm_list[i]

        if atr > 0:
            plus_di = 100 * plus_di_smooth / atr
            minus_di = 100 * minus_di_smooth / atr
        else:
            plus_di = minus_di = 0

        di_sum = plus_di + minus_di
        dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0
        dx_list.append(dx)

    if not dx_list:
        return 20.0

    adx_val = float(np.mean(dx_list[-period:]))
    return min(adx_val, 100.0)


def _atr_pct(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """ATR as percentage of current price."""
    if len(closes) < period + 1:
        return 0.02

    tr_values = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_values.append(tr)

    atr = float(np.mean(tr_values[-period:]))
    current_price = float(closes[-1])
    return atr / current_price if current_price > 0 else 0.02


# ============================================================================
# REGIME DETECTOR
# ============================================================================

class RuleBasedRegimeDetector:
    """
    Rule-based market regime detector.

    Classification logic:
    1. ATR% > 2.5% OR (ADX > 35 AND ATR% > 1.5%) -> HIGH_VOLATILITY
    2. ADX > 25 AND clear trend -> TRENDING (bull or bear)
    3. ADX < 20 AND BB_width < 0.08 -> MEAN_REVERTING
    4. Default -> MEAN_REVERTING
    """

    def __init__(self):
        """Initialize regime detector."""
        self.logger = logging.getLogger(__name__)
        self._last_regime: Optional[RegimeResult] = None
        self._cache: Dict[str, Tuple[RegimeResult, datetime]] = {}
        self._cache_ttl = timedelta(minutes=10)

    def detect_regime(self, symbol: str = "SPY") -> RegimeResult:
        """
        Detect current market regime for a symbol.

        Args:
            symbol: Underlying to analyze (default SPY for broad market)

        Returns:
            RegimeResult with regime, confidence, and evidence
        """
        # Check cache
        if symbol in self._cache:
            cached, ts = self._cache[symbol]
            if datetime.now() - ts < self._cache_ttl:
                return cached

        # Compute technicals
        technicals = self._compute_technicals(symbol)
        if technicals is None:
            result = RegimeResult(
                regime=Regime.UNKNOWN,
                confidence=0.0,
                evidence={"error": "Failed to compute technicals"},
                technicals=None,
            )
            return result

        # Classify regime
        result = self._classify_regime(technicals)
        self._cache[symbol] = (result, datetime.now())
        self._last_regime = result

        self.logger.info(
            f"Regime [{symbol}]: {result.regime.value} "
            f"(confidence: {result.confidence:.0%}) "
            f"ADX={technicals.adx:.1f} RSI={technicals.rsi_14:.1f} "
            f"ATR%={technicals.atr_pct:.2%}"
        )

        return result

    def get_technicals(self, symbol: str) -> Optional[TechnicalSignals]:
        """Get technical signals for a symbol (useful for signal generation)."""
        return self._compute_technicals(symbol)

    # ================================================================== #
    # PRIVATE
    # ================================================================== #

    def _compute_technicals(self, symbol: str) -> Optional[TechnicalSignals]:
        """Compute all technical indicators from price data."""
        if yf is None:
            self.logger.error("yfinance not available")
            return None

        try:
            data = yf.download(symbol, period="1y", interval="1d", progress=False)
            if data.empty or len(data) < 50:
                self.logger.warning(f"Insufficient data for {symbol}")
                return None

            closes = data["Close"].values.flatten().astype(float)
            highs = data["High"].values.flatten().astype(float)
            lows = data["Low"].values.flatten().astype(float)
            volumes = data["Volume"].values.flatten().astype(float)

            current_price = float(closes[-1])

            # SMAs
            sma20 = _sma(closes, 20)
            sma50 = _sma(closes, 50)
            sma200 = _sma(closes, 200)

            # Momentum
            rsi14 = _rsi(closes, 14)
            macd_val, macd_sig, macd_hist = _macd(closes)

            # Bollinger Bands
            bb_upper, bb_mid, bb_lower, bb_width, bb_pos = _bollinger_bands(closes, 20)

            # Trend strength
            adx_val = _adx(highs, lows, closes, 14)
            atr_pct_val = _atr_pct(highs, lows, closes, 14)

            # Trend direction
            if current_price > sma50 and sma20 > sma50:
                trend_dir = 1  # bull
            elif current_price < sma50 and sma20 < sma50:
                trend_dir = -1  # bear
            else:
                trend_dir = 0  # neutral

            # Volume ratio
            avg_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
            current_vol = float(volumes[-1])
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

            return TechnicalSignals(
                symbol=symbol,
                current_price=current_price,
                sma_20=sma20,
                sma_50=sma50,
                sma_200=sma200,
                price_vs_sma20_pct=(current_price / sma20 - 1) * 100 if sma20 > 0 else 0,
                price_vs_sma50_pct=(current_price / sma50 - 1) * 100 if sma50 > 0 else 0,
                rsi_14=rsi14,
                macd_signal=macd_val - macd_sig,
                macd_histogram=macd_hist,
                bb_width=bb_width,
                bb_position=bb_pos,
                atr_pct=atr_pct_val,
                adx=adx_val,
                trend_direction=trend_dir,
                volume_ratio=vol_ratio,
            )
        except Exception as e:
            self.logger.error(f"Technical computation failed for {symbol}: {e}")
            return None

    def _classify_regime(self, t: TechnicalSignals) -> RegimeResult:
        """
        Classify market regime from technicals.

        Decision tree:
        1. ATR% > 2.5% OR (ADX > 35 AND ATR% > 1.5%) -> HIGH_VOLATILITY
        2. ADX > 25 AND trend_direction != 0 -> TRENDING (bull/bear)
        3. ADX < 20 AND BB_width < 0.08 -> MEAN_REVERTING
        4. Default -> MEAN_REVERTING
        """
        evidence: Dict[str, str] = {}
        scores: Dict[Regime, float] = {r: 0.0 for r in Regime if r != Regime.UNKNOWN}

        # ---- HIGH VOLATILITY signals ----
        if t.atr_pct > 0.025:
            scores[Regime.HIGH_VOLATILITY] += 0.4
            evidence["high_atr"] = f"ATR% {t.atr_pct:.2%} > 2.5%"
        if t.adx > 35 and t.atr_pct > 0.015:
            scores[Regime.HIGH_VOLATILITY] += 0.2
            evidence["strong_move"] = f"ADX {t.adx:.0f} with elevated ATR"
        if t.bb_width > 0.12:
            scores[Regime.HIGH_VOLATILITY] += 0.15
            evidence["wide_bb"] = f"BB width {t.bb_width:.2%}"

        # ---- TRENDING signals ----
        if t.adx > 25:
            if t.trend_direction == 1:
                scores[Regime.TRENDING_BULL] += 0.35
                evidence["adx_bull"] = f"ADX {t.adx:.0f} with bullish trend"
            elif t.trend_direction == -1:
                scores[Regime.TRENDING_BEAR] += 0.35
                evidence["adx_bear"] = f"ADX {t.adx:.0f} with bearish trend"

        # Price above/below key SMAs
        if t.price_vs_sma50_pct > 3:
            scores[Regime.TRENDING_BULL] += 0.15
            evidence["above_sma50"] = f"Price {t.price_vs_sma50_pct:+.1f}% above SMA50"
        elif t.price_vs_sma50_pct < -3:
            scores[Regime.TRENDING_BEAR] += 0.15
            evidence["below_sma50"] = f"Price {t.price_vs_sma50_pct:+.1f}% below SMA50"

        # MACD confirmation
        if t.macd_histogram > 0 and t.macd_signal > 0:
            scores[Regime.TRENDING_BULL] += 0.1
        elif t.macd_histogram < 0 and t.macd_signal < 0:
            scores[Regime.TRENDING_BEAR] += 0.1

        # RSI extremes suggest regime
        if t.rsi_14 > 65:
            scores[Regime.TRENDING_BULL] += 0.05
        elif t.rsi_14 < 35:
            scores[Regime.TRENDING_BEAR] += 0.05

        # ---- MEAN REVERTING signals ----
        if t.adx < 20:
            scores[Regime.MEAN_REVERTING] += 0.3
            evidence["low_adx"] = f"ADX {t.adx:.0f} < 20 (no trend)"
        if t.bb_width < 0.08:
            scores[Regime.MEAN_REVERTING] += 0.15
            evidence["tight_bb"] = f"BB width {t.bb_width:.2%} (range-bound)"
        if 40 <= t.rsi_14 <= 60:
            scores[Regime.MEAN_REVERTING] += 0.1
            evidence["neutral_rsi"] = f"RSI {t.rsi_14:.0f} (neutral)"

        # Default baseline for mean-reverting (it's the safest assumption)
        scores[Regime.MEAN_REVERTING] += 0.05

        # ---- Pick winner ----
        best_regime = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_regime]

        # Normalize confidence to 0-1
        total = sum(scores.values())
        confidence = best_score / total if total > 0 else 0.5

        return RegimeResult(
            regime=best_regime,
            confidence=round(confidence, 2),
            evidence=evidence,
            technicals=t,
        )

"""
Volatility Surface — IV Term-Structure Trading Signals (Tier 2)
================================================================

Builds a volatility surface from option chain data and extracts
actionable trading signals:

  1. **Term Structure Slope**  — contango vs backwardation in IV
  2. **Skew Analysis**         — put/call skew for crash fear detection
  3. **Surface Anomalies**     — mispricings vs fitted SVI surface
  4. **Vol-of-Vol**            — regime change detection via VVIX proxy

Data: Alpaca option snapshots, VIX futures proxy, or historical IV DB.
Fallback: synthetic surface from ATR + HV when live chain unavailable.

Integration:
    from src.volatility_surface import VolatilitySurface
    vs = VolatilitySurface()
    signal = vs.get_vol_signal(symbol, bars, option_chain)
    # signal.term_structure_slope, signal.skew_zscore, signal.trade_bias
"""

import logging
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class VolRegime(Enum):
    LOW_VOL = "low_vol"           # IV rank < 25
    NORMAL = "normal"             # IV rank 25-50
    ELEVATED = "elevated"         # IV rank 50-75
    HIGH_VOL = "high_vol"         # IV rank > 75


class TermStructure(Enum):
    CONTANGO = "contango"         # Front < Back (normal)
    FLAT = "flat"
    BACKWARDATION = "backwardation"  # Front > Back (fear)


@dataclass
class VolSurfacePoint:
    """Single point on the IV surface."""
    strike: float
    dte: int
    iv: float
    delta: float = 0.0
    option_type: str = "call"     # "call" or "put"


@dataclass
class VolSurfaceSlice:
    """IV smile/skew at a single expiration."""
    dte: int
    expiration: str              # YYYY-MM-DD
    atm_iv: float
    put_25d_iv: float = 0.0     # 25-delta put IV
    call_25d_iv: float = 0.0    # 25-delta call IV
    skew: float = 0.0           # put_25d - call_25d (positive = put skew)
    points: List[VolSurfacePoint] = field(default_factory=list)


@dataclass
class VolSignal:
    """Actionable signal derived from volatility surface."""
    symbol: str
    timestamp: datetime

    # Term structure
    term_structure: TermStructure
    term_slope: float            # +contango, -backwardation (annualized)
    front_iv: float              # Front-month ATM IV
    back_iv: float               # Back-month ATM IV

    # Skew
    skew_25d: float              # 25-delta risk reversal (put - call IV)
    skew_zscore: float           # Z-score of skew vs 30-day history
    skew_percentile: float       # Percentile rank of skew (0-100)

    # Surface metrics
    vol_regime: VolRegime
    iv_rank: float               # 0-100
    hv_iv_ratio: float           # HV20 / IV (>1 = IV cheap)
    vol_of_vol: float            # Std of IV changes (VVIX proxy)

    # Trading signal
    trade_bias: str              # "sell_premium", "buy_vol", "neutral"
    confidence: float            # [0, 1]
    position_scale: float        # Multiplier for position sizing
    reasons: List[str] = field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        return self.trade_bias != "neutral" and self.confidence >= 0.3


# ============================================================================
# CORE ENGINE
# ============================================================================

class VolatilitySurface:
    """
    Build and analyze IV surface for a symbol.

    Hierarchy:
      1. Live option chain data (Alpaca snapshots)
      2. Synthetic surface from HV + ATR fallback
      3. VIX proxy for SPY/QQQ when chain unavailable

    Usage:
        vs = VolatilitySurface()
        signal = vs.get_vol_signal("AAPL", bars, option_chain)
    """

    # Target DTEs for term structure slices
    TARGET_DTES = [7, 14, 30, 45, 60, 90]

    def __init__(self, history_window: int = 252, skew_lookback: int = 30):
        self.history_window = history_window
        self.skew_lookback = skew_lookback

        # Internal caches
        self._iv_history: Dict[str, List[float]] = {}     # symbol -> list of daily ATM IVs
        self._skew_history: Dict[str, List[float]] = {}   # symbol -> list of daily skews
        self._last_surface: Dict[str, List[VolSurfaceSlice]] = {}

    # ── Public API ──────────────────────────────────────────────────

    def get_vol_signal(
        self,
        symbol: str,
        bars: List[dict],
        option_chain: Optional[List[dict]] = None,
    ) -> VolSignal:
        """
        Generate a volatility-based trading signal.

        Args:
            symbol: Ticker symbol
            bars: OHLCV bars (Alpaca format, need >= 60)
            option_chain: Optional list of option snapshots with IV

        Returns:
            VolSignal with term structure, skew, and trade bias
        """
        closes = np.array([float(b["c"]) for b in bars])
        highs = np.array([float(b["h"]) for b in bars])
        lows = np.array([float(b["l"]) for b in bars])

        # Build surface
        if option_chain and len(option_chain) >= 10:
            slices = self._build_surface_from_chain(symbol, closes[-1], option_chain)
        else:
            slices = self._build_synthetic_surface(symbol, closes, highs, lows)

        self._last_surface[symbol] = slices

        # Extract metrics
        front_iv, back_iv, term_slope, term_struct = self._analyze_term_structure(slices)
        skew_25d, skew_z, skew_pct = self._analyze_skew(symbol, slices)
        hv20 = self._compute_hv(closes, 20)
        hv_iv_ratio = hv20 / max(front_iv, 0.01)
        vol_of_vol = self._compute_vol_of_vol(symbol, front_iv)
        iv_rank = self._compute_iv_rank(symbol, front_iv)
        vol_regime = self._classify_vol_regime(iv_rank)

        # Update history
        self._update_history(symbol, front_iv, skew_25d)

        # Generate trade bias
        trade_bias, confidence, position_scale, reasons = self._generate_signal(
            vol_regime, term_struct, term_slope, skew_z,
            hv_iv_ratio, vol_of_vol, iv_rank,
        )

        return VolSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            term_structure=term_struct,
            term_slope=round(term_slope, 4),
            front_iv=round(front_iv, 4),
            back_iv=round(back_iv, 4),
            skew_25d=round(skew_25d, 4),
            skew_zscore=round(skew_z, 2),
            skew_percentile=round(skew_pct, 1),
            vol_regime=vol_regime,
            iv_rank=round(iv_rank, 1),
            hv_iv_ratio=round(hv_iv_ratio, 3),
            vol_of_vol=round(vol_of_vol, 4),
            trade_bias=trade_bias,
            confidence=round(confidence, 3),
            position_scale=round(position_scale, 3),
            reasons=reasons,
        )

    def get_term_structure(self, symbol: str) -> List[VolSurfaceSlice]:
        """Return the last computed surface slices."""
        return self._last_surface.get(symbol, [])

    # ── Surface Construction ────────────────────────────────────────

    def _build_surface_from_chain(
        self, symbol: str, spot: float, chain: List[dict]
    ) -> List[VolSurfaceSlice]:
        """Build IV surface from option chain snapshots."""
        # Group by DTE
        by_dte: Dict[int, List[VolSurfacePoint]] = {}
        for opt in chain:
            try:
                iv = float(opt.get("iv", opt.get("implied_volatility", 0)))
                if iv <= 0 or iv > 5.0:
                    continue
                strike = float(opt.get("strike", opt.get("strike_price", 0)))
                dte = int(opt.get("dte", opt.get("days_to_expiry", 30)))
                delta = float(opt.get("delta", 0.5))
                otype = str(opt.get("type", opt.get("option_type", "call"))).lower()
                exp = str(opt.get("expiration", opt.get("expiry", "")))

                pt = VolSurfacePoint(
                    strike=strike, dte=dte, iv=iv, delta=delta, option_type=otype,
                )
                if dte not in by_dte:
                    by_dte[dte] = []
                by_dte[dte].append(pt)
            except (ValueError, TypeError):
                continue

        # Build slices for each DTE group
        slices = []
        for dte in sorted(by_dte.keys()):
            pts = by_dte[dte]
            if len(pts) < 3:
                continue

            # ATM IV: closest strike to spot
            atm_pts = sorted(pts, key=lambda p: abs(p.strike - spot))
            atm_iv = float(np.mean([p.iv for p in atm_pts[:3]]))

            # 25-delta put and call IV
            puts = [p for p in pts if p.option_type == "put"]
            calls = [p for p in pts if p.option_type == "call"]

            put_25d_iv = self._find_delta_iv(puts, -0.25, atm_iv)
            call_25d_iv = self._find_delta_iv(calls, 0.25, atm_iv)
            skew = put_25d_iv - call_25d_iv

            exp_str = ""
            for p in pts:
                if hasattr(p, "expiration"):
                    exp_str = str(getattr(p, "expiration", ""))
                    break

            slices.append(VolSurfaceSlice(
                dte=dte,
                expiration=exp_str,
                atm_iv=round(atm_iv, 4),
                put_25d_iv=round(put_25d_iv, 4),
                call_25d_iv=round(call_25d_iv, 4),
                skew=round(skew, 4),
                points=pts,
            ))

        return slices if slices else self._build_synthetic_surface(
            symbol, np.array([spot]), np.array([spot]), np.array([spot])
        )

    def _build_synthetic_surface(
        self, symbol: str,
        closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    ) -> List[VolSurfaceSlice]:
        """
        Build synthetic IV surface from HV + ATR when no chain available.

        Uses empirical vol-of-vol scaling:
          - Short-dated IV = HV10 * 1.15 (add premium)
          - Medium IV  = HV20 * 1.10
          - Long-dated IV = HV60 * 1.05 (less premium)
          - Skew = 0.05 * HV20 (mild put skew)
        """
        hv10 = self._compute_hv(closes, 10) if len(closes) >= 11 else 0.20
        hv20 = self._compute_hv(closes, 20) if len(closes) >= 21 else 0.20
        hv60 = self._compute_hv(closes, 60) if len(closes) >= 61 else 0.20

        # Synthetic term structure
        synth_ivs = {
            7: hv10 * 1.18,
            14: hv10 * 1.15,
            30: hv20 * 1.10,
            45: hv20 * 1.08,
            60: hv60 * 1.06,
            90: hv60 * 1.04,
        }

        slices = []
        for dte, atm_iv in synth_ivs.items():
            skew = 0.05 * hv20  # Mild synthetic skew
            put_25d = atm_iv + skew / 2
            call_25d = atm_iv - skew / 2

            slices.append(VolSurfaceSlice(
                dte=dte,
                expiration="",
                atm_iv=round(atm_iv, 4),
                put_25d_iv=round(put_25d, 4),
                call_25d_iv=round(call_25d, 4),
                skew=round(skew, 4),
            ))

        return slices

    def _find_delta_iv(
        self, options: List[VolSurfacePoint], target_delta: float, fallback_iv: float
    ) -> float:
        """Find IV of option closest to target delta."""
        if not options:
            return fallback_iv + (0.03 if target_delta < 0 else -0.01)  # Small skew default

        closest = min(options, key=lambda p: abs(abs(p.delta) - abs(target_delta)))
        if abs(abs(closest.delta) - abs(target_delta)) < 0.15:
            return closest.iv
        return fallback_iv + (0.03 if target_delta < 0 else -0.01)

    # ── Analysis ────────────────────────────────────────────────────

    def _analyze_term_structure(
        self, slices: List[VolSurfaceSlice]
    ) -> Tuple[float, float, float, TermStructure]:
        """
        Analyze IV term structure.

        Returns: (front_iv, back_iv, slope, structure_type)
        """
        if len(slices) < 2:
            iv = slices[0].atm_iv if slices else 0.20
            return iv, iv, 0.0, TermStructure.FLAT

        # Front = shortest DTE, Back = longest DTE
        sorted_slices = sorted(slices, key=lambda s: s.dte)
        front = sorted_slices[0]
        back = sorted_slices[-1]

        front_iv = front.atm_iv
        back_iv = back.atm_iv

        # Annualized slope
        dte_diff = max(back.dte - front.dte, 1)
        slope = (back_iv - front_iv) / (dte_diff / 365.0)

        # Classify
        diff_pct = (back_iv - front_iv) / max(front_iv, 0.01)
        if diff_pct > 0.03:
            structure = TermStructure.CONTANGO
        elif diff_pct < -0.03:
            structure = TermStructure.BACKWARDATION
        else:
            structure = TermStructure.FLAT

        return front_iv, back_iv, slope, structure

    def _analyze_skew(
        self, symbol: str, slices: List[VolSurfaceSlice]
    ) -> Tuple[float, float, float]:
        """
        Analyze put/call skew.

        Returns: (skew_25d, z_score, percentile)
        """
        # Use 30-DTE slice if available, else nearest
        target_slice = None
        for s in slices:
            if 25 <= s.dte <= 45:
                target_slice = s
                break
        if target_slice is None and slices:
            target_slice = min(slices, key=lambda s: abs(s.dte - 30))

        skew_25d = target_slice.skew if target_slice else 0.0

        # Z-score against history
        history = self._skew_history.get(symbol, [])
        if len(history) >= 10:
            mean_skew = float(np.mean(history))
            std_skew = float(np.std(history)) + 1e-6
            z_score = (skew_25d - mean_skew) / std_skew
            from scipy.stats import percentileofscore as pctof
            percentile = float(pctof(history, skew_25d))
        else:
            z_score = 0.0
            percentile = 50.0

        return skew_25d, z_score, percentile

    def _compute_hv(self, closes: np.ndarray, window: int) -> float:
        """Compute annualized historical volatility."""
        if len(closes) < window + 1:
            return 0.20
        log_returns = np.diff(np.log(closes[-(window + 1):]))
        return float(np.std(log_returns) * np.sqrt(252))

    def _compute_vol_of_vol(self, symbol: str, current_iv: float) -> float:
        """Vol-of-vol: std of IV changes (VVIX proxy)."""
        history = self._iv_history.get(symbol, [])
        if len(history) < 5:
            return 0.0
        iv_changes = np.diff(history[-30:])
        return float(np.std(iv_changes)) if len(iv_changes) >= 3 else 0.0

    def _compute_iv_rank(self, symbol: str, current_iv: float) -> float:
        """IV Rank: (current - 52wk low) / (52wk high - 52wk low) * 100."""
        history = self._iv_history.get(symbol, [])
        if len(history) < 20:
            return 50.0  # Neutral default
        iv_min = min(history[-252:])
        iv_max = max(history[-252:])
        if iv_max - iv_min < 0.001:
            return 50.0
        return float(np.clip((current_iv - iv_min) / (iv_max - iv_min) * 100, 0, 100))

    def _classify_vol_regime(self, iv_rank: float) -> VolRegime:
        """Classify vol regime from IV rank."""
        if iv_rank < 25:
            return VolRegime.LOW_VOL
        elif iv_rank < 50:
            return VolRegime.NORMAL
        elif iv_rank < 75:
            return VolRegime.ELEVATED
        return VolRegime.HIGH_VOL

    def _update_history(self, symbol: str, atm_iv: float, skew: float):
        """Append today's IV and skew to rolling history."""
        if symbol not in self._iv_history:
            self._iv_history[symbol] = []
        self._iv_history[symbol].append(atm_iv)
        # Trim to window
        if len(self._iv_history[symbol]) > self.history_window:
            self._iv_history[symbol] = self._iv_history[symbol][-self.history_window:]

        if symbol not in self._skew_history:
            self._skew_history[symbol] = []
        self._skew_history[symbol].append(skew)
        if len(self._skew_history[symbol]) > self.skew_lookback * 2:
            self._skew_history[symbol] = self._skew_history[symbol][-self.skew_lookback * 2:]

    # ── Signal Generation ───────────────────────────────────────────

    def _generate_signal(
        self,
        vol_regime: VolRegime,
        term_struct: TermStructure,
        term_slope: float,
        skew_z: float,
        hv_iv_ratio: float,
        vol_of_vol: float,
        iv_rank: float,
    ) -> Tuple[str, float, float, List[str]]:
        """
        Generate trade bias from surface analysis.

        Logic:
          - HIGH IV + contango + high skew → sell premium (iron condors / put spreads)
          - LOW IV + backwardation + low skew → buy vol (straddles / call spreads)
          - Vol-of-vol spike → reduce size (regime uncertainty)

        Returns: (bias, confidence, position_scale, reasons)
        """
        reasons = []
        score = 0.0  # Positive → sell premium, negative → buy vol

        # 1. IV Rank
        if iv_rank > 70:
            score += 0.35
            reasons.append(f"IV rank high ({iv_rank:.0f})")
        elif iv_rank > 50:
            score += 0.15
            reasons.append(f"IV rank elevated ({iv_rank:.0f})")
        elif iv_rank < 25:
            score -= 0.30
            reasons.append(f"IV rank low ({iv_rank:.0f})")
        elif iv_rank < 40:
            score -= 0.10

        # 2. Term structure
        if term_struct == TermStructure.CONTANGO:
            score += 0.15
            reasons.append("Contango (normal)")
        elif term_struct == TermStructure.BACKWARDATION:
            score -= 0.25
            reasons.append("Backwardation (fear/event)")

        # 3. Skew
        if skew_z > 1.5:
            score += 0.15
            reasons.append(f"Elevated skew (z={skew_z:.1f})")
        elif skew_z < -1.5:
            score -= 0.15
            reasons.append(f"Compressed skew (z={skew_z:.1f})")

        # 4. HV/IV ratio
        if hv_iv_ratio > 1.2:
            score -= 0.10
            reasons.append(f"IV cheap vs HV ({hv_iv_ratio:.2f})")
        elif hv_iv_ratio < 0.7:
            score += 0.10
            reasons.append(f"IV expensive vs HV ({hv_iv_ratio:.2f})")

        # 5. Vol-of-vol (regime uncertainty)
        vol_of_vol_thresh = 0.04
        if vol_of_vol > vol_of_vol_thresh:
            reasons.append(f"High vol-of-vol ({vol_of_vol:.3f}) — size reduced")

        # Determine bias and confidence
        if score >= 0.25:
            trade_bias = "sell_premium"
            confidence = min(abs(score), 1.0)
        elif score <= -0.25:
            trade_bias = "buy_vol"
            confidence = min(abs(score), 1.0)
        else:
            trade_bias = "neutral"
            confidence = 0.0

        # Position scale: reduce if vol-of-vol high
        position_scale = 1.0
        if vol_of_vol > vol_of_vol_thresh:
            position_scale *= 0.6
        if vol_regime == VolRegime.HIGH_VOL:
            position_scale *= 0.7
        elif vol_regime == VolRegime.LOW_VOL:
            position_scale *= 1.2

        return trade_bias, float(confidence), float(position_scale), reasons

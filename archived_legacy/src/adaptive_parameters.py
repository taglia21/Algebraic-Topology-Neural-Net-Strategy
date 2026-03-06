"""
Adaptive Parameters — Auto-Tune Thresholds from Performance (Tier 2)
=====================================================================

Continuously adjusts trading parameters based on recent P&L:

  1. **Composite Threshold** — Lower when recent signals are profitable,
     raise when win rate drops below target
  2. **Position Sizing**     — Scale Kelly fraction by regime confidence
  3. **Stop Distance**       — Widen ATR multiplier after whipsaws,
     tighten after clean trends
  4. **Profit Target**       — Extend in trending regimes, tighten in
     mean-reverting
  5. **Cooldown Periods**    — Increase after consecutive losses

Guardrails: all parameters are clamped to safe ranges to prevent
runaway optimization or overfitting to noise.

Integration:
    from src.adaptive_parameters import AdaptiveParameterTuner
    tuner = AdaptiveParameterTuner()
    tuner.record_trade(pnl_pct=0.032, regime="trending_bull", ...)
    adjustments = tuner.get_adjustments()
    # Apply: cfg.min_composite_score += adjustments.composite_threshold_adj
"""

import logging
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TradeRecord:
    """Record of a completed trade for parameter learning."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    pnl_pct: float
    holding_bars: int
    regime: str
    composite_score: float
    ml_confidence: float
    atr_pct: float
    stop_distance_pct: float
    exit_reason: str             # "stop", "target", "trailing", "manual"


@dataclass
class ParameterAdjustments:
    """Recommended parameter adjustments from adaptive tuner."""
    # Threshold adjustments (additive)
    composite_threshold_adj: float = 0.0    # Add to min_composite_score
    ml_confidence_adj: float = 0.0          # Add to ml_min_confidence

    # Multiplier adjustments
    position_size_mult: float = 1.0         # Multiply position sizing
    kelly_fraction_mult: float = 1.0        # Multiply kelly_fraction
    atr_stop_mult: float = 1.0             # Multiply ATR multiplier
    profit_target_mult: float = 1.0         # Multiply profit_target_pct
    trailing_activation_mult: float = 1.0   # Multiply trailing_stop_activation

    # Cooldown
    skip_next_n_signals: int = 0            # Temporary cooldown after streak

    # Metadata
    win_rate_30d: float = 0.0
    avg_pnl_30d: float = 0.0
    sharpe_30d: float = 0.0
    consecutive_losses: int = 0
    adaptation_reason: str = ""

    def describe(self) -> str:
        """Human-readable summary of adjustments."""
        parts = []
        if self.composite_threshold_adj != 0:
            parts.append(f"threshold {self.composite_threshold_adj:+.3f}")
        if self.position_size_mult != 1.0:
            parts.append(f"size ×{self.position_size_mult:.2f}")
        if self.atr_stop_mult != 1.0:
            parts.append(f"stop ×{self.atr_stop_mult:.2f}")
        if self.profit_target_mult != 1.0:
            parts.append(f"target ×{self.profit_target_mult:.2f}")
        if self.skip_next_n_signals > 0:
            parts.append(f"cooldown {self.skip_next_n_signals}")
        return " | ".join(parts) if parts else "no adjustments"


# ============================================================================
# SAFE PARAMETER RANGES (guardrails)
# ============================================================================

PARAM_BOUNDS = {
    "composite_threshold": (0.30, 0.70),     # Never below 0.30 or above 0.70
    "ml_min_confidence": (0.15, 0.55),        # Never below 0.15
    "position_size_mult": (0.3, 1.8),         # 30% to 180% of base
    "kelly_fraction_mult": (0.25, 1.5),       # Quarter to 1.5x Kelly
    "atr_stop_mult": (0.6, 2.0),             # 60% to 200% of base ATR mult
    "profit_target_mult": (0.5, 2.5),         # 50% to 250% of base target
    "trailing_activation_mult": (0.5, 2.0),
    "max_cooldown": 5,                        # Max 5 signals skipped
}


# ============================================================================
# CORE ENGINE
# ============================================================================

class AdaptiveParameterTuner:
    """
    Auto-tunes trading parameters based on recent performance.

    Learning rules:
      - EMA of win rate drives threshold adjustment
      - Regime-conditioned sizing (learn which regimes are profitable)
      - Whipsaw detection tightens/loosens stops
      - Consecutive loss counter triggers cooldown

    Usage:
        tuner = AdaptiveParameterTuner()
        tuner.record_trade(TradeRecord(...))
        adj = tuner.get_adjustments()
        cfg.min_composite_score = base_threshold + adj.composite_threshold_adj
    """

    def __init__(
        self,
        lookback_trades: int = 50,
        target_win_rate: float = 0.55,
        ema_alpha: float = 0.1,
        state_file: str = "state/adaptive_params.json",
    ):
        self.lookback_trades = lookback_trades
        self.target_win_rate = target_win_rate
        self.ema_alpha = ema_alpha
        self.state_file = Path(state_file)

        # Trade history
        self._trades: deque = deque(maxlen=lookback_trades * 2)
        self._consecutive_losses = 0
        self._max_consecutive_losses = 0

        # EMA state
        self._ema_win_rate = target_win_rate
        self._ema_avg_pnl = 0.0
        self._ema_sharpe = 0.0

        # Regime performance tracking
        self._regime_pnl: Dict[str, List[float]] = {}

        # Exit reason tracking (whipsaw detection)
        self._exit_reasons: deque = deque(maxlen=30)

        # Load persisted state
        self._load_state()

    # ── Public API ──────────────────────────────────────────────────

    def record_trade(self, trade: TradeRecord):
        """Record a completed trade and update adaptive state."""
        self._trades.append(trade)

        # Update consecutive losses
        if trade.pnl_pct < 0:
            self._consecutive_losses += 1
            self._max_consecutive_losses = max(
                self._max_consecutive_losses, self._consecutive_losses
            )
        else:
            self._consecutive_losses = 0

        # Update EMAs
        won = 1.0 if trade.pnl_pct > 0 else 0.0
        self._ema_win_rate = (
            self.ema_alpha * won + (1 - self.ema_alpha) * self._ema_win_rate
        )
        self._ema_avg_pnl = (
            self.ema_alpha * trade.pnl_pct + (1 - self.ema_alpha) * self._ema_avg_pnl
        )

        # Track regime performance
        if trade.regime not in self._regime_pnl:
            self._regime_pnl[trade.regime] = []
        self._regime_pnl[trade.regime].append(trade.pnl_pct)

        # Track exit reasons
        self._exit_reasons.append(trade.exit_reason)

        # Persist state
        self._save_state()

        logger.debug(
            f"Adaptive: recorded {trade.symbol} pnl={trade.pnl_pct:+.2%} "
            f"ema_wr={self._ema_win_rate:.3f} consec_loss={self._consecutive_losses}"
        )

    def get_adjustments(self, current_regime: str = "neutral") -> ParameterAdjustments:
        """
        Compute parameter adjustments based on recent performance.

        Returns ParameterAdjustments with all recommended changes.
        """
        recent = list(self._trades)[-self.lookback_trades:]
        if len(recent) < 5:
            return ParameterAdjustments(
                adaptation_reason="insufficient trade history",
            )

        # Compute rolling metrics
        pnls = [t.pnl_pct for t in recent]
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
        avg_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls)) + 1e-6
        sharpe = avg_pnl / std_pnl * np.sqrt(252) if std_pnl > 0 else 0.0

        reasons = []

        # ── 1. Composite threshold adjustment ───────────────────────
        threshold_adj = 0.0
        if win_rate > self.target_win_rate + 0.10:
            # Winning a lot → can afford to lower threshold (more trades)
            threshold_adj = -0.03
            reasons.append(f"WR high ({win_rate:.0%}) → lower threshold")
        elif win_rate > self.target_win_rate + 0.05:
            threshold_adj = -0.015
        elif win_rate < self.target_win_rate - 0.10:
            # Losing too much → raise threshold (fewer, higher quality)
            threshold_adj = 0.05
            reasons.append(f"WR low ({win_rate:.0%}) → raise threshold")
        elif win_rate < self.target_win_rate - 0.05:
            threshold_adj = 0.025

        # ── 2. Position sizing ──────────────────────────────────────
        size_mult = 1.0
        kelly_mult = 1.0

        # Scale by regime profitability
        regime_hist = self._regime_pnl.get(current_regime, [])
        if len(regime_hist) >= 5:
            regime_wr = sum(1 for p in regime_hist[-20:] if p > 0) / len(regime_hist[-20:])
            if regime_wr > 0.65:
                size_mult = 1.3
                kelly_mult = 1.2
                reasons.append(f"Regime '{current_regime}' profitable ({regime_wr:.0%})")
            elif regime_wr < 0.35:
                size_mult = 0.5
                kelly_mult = 0.5
                reasons.append(f"Regime '{current_regime}' unprofitable ({regime_wr:.0%})")

        # Scale by overall performance
        if sharpe > 2.0:
            size_mult *= 1.15
        elif sharpe < 0:
            size_mult *= 0.7
            reasons.append(f"Negative Sharpe ({sharpe:.2f}) → reduce size")

        # ── 3. ATR stop multiplier ──────────────────────────────────
        atr_mult = 1.0
        stop_exits = sum(1 for r in self._exit_reasons if r == "stop")
        total_recent = len(self._exit_reasons)

        if total_recent >= 5:
            stop_rate = stop_exits / total_recent
            if stop_rate > 0.6:
                # Too many stops → widen (whipsaw detected)
                atr_mult = 1.25
                reasons.append(f"High stop rate ({stop_rate:.0%}) → widen stops")
            elif stop_rate < 0.15:
                # Stops rarely hit → can tighten
                atr_mult = 0.85
                reasons.append("Low stop rate → tighten stops")

        # ── 4. Profit target ────────────────────────────────────────
        target_mult = 1.0
        target_exits = sum(1 for r in self._exit_reasons if r == "target")
        trailing_exits = sum(1 for r in self._exit_reasons if r == "trailing")

        if total_recent >= 5:
            target_rate = target_exits / total_recent
            if target_rate > 0.5:
                # Hitting targets frequently → extend targets
                target_mult = 1.3
                reasons.append("Frequently hitting targets → extend")
            elif target_rate < 0.1 and trailing_exits > target_exits * 2:
                # Trailing stops catching most exits → tighten target
                target_mult = 0.8
                reasons.append("Trailing catching exits → tighten target")

        # ── 5. Cooldown ─────────────────────────────────────────────
        cooldown = 0
        if self._consecutive_losses >= 5:
            cooldown = 3
            reasons.append(f"{self._consecutive_losses} consecutive losses → cooldown")
        elif self._consecutive_losses >= 3:
            cooldown = 1
            reasons.append(f"{self._consecutive_losses} consecutive losses → brief cooldown")

        # ── 6. ML confidence adjustment ─────────────────────────────
        ml_adj = 0.0
        # Check if ML-filtered trades outperform
        ml_trades = [t for t in recent if t.ml_confidence > 0.5]
        low_ml_trades = [t for t in recent if t.ml_confidence <= 0.5]

        if len(ml_trades) >= 5 and len(low_ml_trades) >= 5:
            ml_avg = float(np.mean([t.pnl_pct for t in ml_trades]))
            low_ml_avg = float(np.mean([t.pnl_pct for t in low_ml_trades]))
            if ml_avg > low_ml_avg + 0.005:
                ml_adj = 0.03  # Raise ML threshold
                reasons.append("High-ML trades outperform → raise ML threshold")
            elif low_ml_avg > ml_avg + 0.005:
                ml_adj = -0.02
                reasons.append("Low-ML trades outperform → lower ML threshold")

        # ── Apply guardrails ────────────────────────────────────────
        threshold_adj = float(np.clip(
            threshold_adj,
            PARAM_BOUNDS["composite_threshold"][0] - 0.5,  # Allow reasonable range
            PARAM_BOUNDS["composite_threshold"][1] - 0.3,
        ))
        size_mult = float(np.clip(
            size_mult,
            PARAM_BOUNDS["position_size_mult"][0],
            PARAM_BOUNDS["position_size_mult"][1],
        ))
        kelly_mult = float(np.clip(
            kelly_mult,
            PARAM_BOUNDS["kelly_fraction_mult"][0],
            PARAM_BOUNDS["kelly_fraction_mult"][1],
        ))
        atr_mult = float(np.clip(
            atr_mult,
            PARAM_BOUNDS["atr_stop_mult"][0],
            PARAM_BOUNDS["atr_stop_mult"][1],
        ))
        target_mult = float(np.clip(
            target_mult,
            PARAM_BOUNDS["profit_target_mult"][0],
            PARAM_BOUNDS["profit_target_mult"][1],
        ))
        cooldown = min(cooldown, PARAM_BOUNDS["max_cooldown"])
        ml_adj = float(np.clip(
            ml_adj,
            PARAM_BOUNDS["ml_min_confidence"][0] - 0.3,
            PARAM_BOUNDS["ml_min_confidence"][1] - 0.2,
        ))

        return ParameterAdjustments(
            composite_threshold_adj=round(threshold_adj, 4),
            ml_confidence_adj=round(ml_adj, 4),
            position_size_mult=round(size_mult, 3),
            kelly_fraction_mult=round(kelly_mult, 3),
            atr_stop_mult=round(atr_mult, 3),
            profit_target_mult=round(target_mult, 3),
            trailing_activation_mult=1.0,
            skip_next_n_signals=cooldown,
            win_rate_30d=round(win_rate, 4),
            avg_pnl_30d=round(avg_pnl, 6),
            sharpe_30d=round(sharpe, 2),
            consecutive_losses=self._consecutive_losses,
            adaptation_reason=" | ".join(reasons) if reasons else "parameters stable",
        )

    def get_regime_performance(self) -> Dict[str, Dict]:
        """Return per-regime performance summary."""
        summary = {}
        for regime, pnls in self._regime_pnl.items():
            if not pnls:
                continue
            summary[regime] = {
                "trades": len(pnls),
                "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
                "avg_pnl": float(np.mean(pnls)),
                "total_pnl": float(np.sum(pnls)),
                "best": float(max(pnls)),
                "worst": float(min(pnls)),
            }
        return summary

    def reset(self):
        """Reset all adaptive state."""
        self._trades.clear()
        self._consecutive_losses = 0
        self._max_consecutive_losses = 0
        self._ema_win_rate = self.target_win_rate
        self._ema_avg_pnl = 0.0
        self._regime_pnl.clear()
        self._exit_reasons.clear()
        logger.info("Adaptive parameter tuner reset")

    # ── Persistence ─────────────────────────────────────────────────

    def _save_state(self):
        """Persist state to JSON."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "timestamp": datetime.now().isoformat(),
                "ema_win_rate": self._ema_win_rate,
                "ema_avg_pnl": self._ema_avg_pnl,
                "consecutive_losses": self._consecutive_losses,
                "max_consecutive_losses": self._max_consecutive_losses,
                "regime_pnl": {
                    k: v[-50:] for k, v in self._regime_pnl.items()
                },
                "exit_reasons": list(self._exit_reasons),
                "trade_count": len(self._trades),
            }
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save adaptive state: {e}")

    def _load_state(self):
        """Load persisted state."""
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            self._ema_win_rate = state.get("ema_win_rate", self.target_win_rate)
            self._ema_avg_pnl = state.get("ema_avg_pnl", 0.0)
            self._consecutive_losses = state.get("consecutive_losses", 0)
            self._max_consecutive_losses = state.get("max_consecutive_losses", 0)
            self._regime_pnl = state.get("regime_pnl", {})
            exit_reasons = state.get("exit_reasons", [])
            self._exit_reasons = deque(exit_reasons, maxlen=30)
            logger.info(
                f"Loaded adaptive state: ema_wr={self._ema_win_rate:.3f} "
                f"consec_loss={self._consecutive_losses} "
                f"regimes={list(self._regime_pnl.keys())}"
            )
        except Exception as e:
            logger.warning(f"Failed to load adaptive state: {e}")

"""
Strategy Selector
==================

Selects the optimal options strategy based on:
1. IV Rank/Percentile (from iv_analysis.py)
2. Market Regime (from regime_detector.py)
3. Technical signals (RSI, MACD, BB from regime_detector)

Decision Matrix (research-backed):
┌────────────────────┬──────────────────┬──────────────────────────────────┐
│ IV Rank            │ Regime           │ Strategy                         │
├────────────────────┼──────────────────┼──────────────────────────────────┤
│ > 50 (HIGH)        │ Mean-Reverting   │ Iron Condor / Short Strangle     │
│ > 50 (HIGH)        │ Trending Bull    │ Bull Put Spread (credit)         │
│ > 50 (HIGH)        │ Trending Bear    │ Bear Call Spread (credit)        │
│ > 50 (HIGH)        │ High Vol         │ Iron Condor (wider wings)        │
│ < 30 (LOW)         │ Trending Bull    │ Bull Call Spread (debit)         │
│ < 30 (LOW)         │ Trending Bear    │ Bear Put Spread (debit)          │
│ < 30 (LOW)         │ Breakout signal  │ Long Straddle/Strangle           │
│ 30-50 (NEUTRAL)    │ Any              │ NO TRADE (wait)                  │
└────────────────────┴──────────────────┴──────────────────────────────────┘

Key Parameters:
- 45 DTE for credit strategies (optimal theta decay)
- 21 DTE for debit strategies (less time decay erosion)
- 16-delta for high probability (68% theoretical PoP)
- 30-delta for higher premium when IV is extreme

Author: System Overhaul - Feb 2026
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from src.iv_analysis import IVMetrics, MarketIVSnapshot
from src.regime_detector import Regime, RegimeResult, TechnicalSignals

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

class StrategyType(Enum):
    """Available options strategies (all defined-risk)."""
    IRON_CONDOR = "iron_condor"
    BULL_PUT_SPREAD = "bull_put_spread"       # Credit spread, bullish
    BEAR_CALL_SPREAD = "bear_call_spread"     # Credit spread, bearish
    BULL_CALL_SPREAD = "bull_call_spread"     # Debit spread, bullish
    BEAR_PUT_SPREAD = "bear_put_spread"       # Debit spread, bearish
    LONG_STRADDLE = "long_straddle"           # Buy call + put (volatility play)
    NO_TRADE = "no_trade"                     # DO NOT TRADE


class Direction(Enum):
    """Trade direction."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"     # Non-directional (iron condors, straddles)


@dataclass
class StrategyRecommendation:
    """Complete strategy recommendation with all parameters."""
    symbol: str
    strategy: StrategyType
    direction: Direction
    
    # Options parameters
    target_dte: int                      # Target days to expiration
    target_delta: float                  # Target delta for short strikes
    wing_width: float                    # Width between strikes ($ for spreads)

    # Risk parameters
    max_contracts: int                   # Maximum contracts
    probability_of_profit: float         # Estimated PoP
    risk_reward_ratio: float             # Ratio of max loss to max profit

    # Context
    iv_rank: float                       # IV rank that drove the decision
    regime: Regime                       # Detected regime
    confidence: float                    # Overall confidence (0-1)
    rationale: str                       # Human-readable explanation

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_credit_strategy(self) -> bool:
        """Does this strategy collect premium (credit)?"""
        return self.strategy in (
            StrategyType.IRON_CONDOR,
            StrategyType.BULL_PUT_SPREAD,
            StrategyType.BEAR_CALL_SPREAD,
        )

    @property
    def is_debit_strategy(self) -> bool:
        """Does this strategy pay premium (debit)?"""
        return self.strategy in (
            StrategyType.BULL_CALL_SPREAD,
            StrategyType.BEAR_PUT_SPREAD,
            StrategyType.LONG_STRADDLE,
        )


# ============================================================================
# WING WIDTH CONFIGURATION PER SYMBOL
# ============================================================================

WING_WIDTHS: Dict[str, float] = {
    "SPY": 5.0,
    "QQQ": 5.0,
    "IWM": 3.0,
    "AAPL": 5.0,
    "TSLA": 10.0,
    "NVDA": 10.0,
    "MSFT": 5.0,
    "AMZN": 5.0,
    "META": 5.0,
    "GOOGL": 5.0,
}
DEFAULT_WING_WIDTH = 5.0


# ============================================================================
# STRATEGY SELECTOR
# ============================================================================

class StrategySelector:
    """
    Selects optimal options strategy based on IV + regime + technicals.

    Core principle: SELL premium when IV is HIGH, BUY premium when IV is LOW.
    Let the regime determine the direction.

    NO TRADE when:
    - IV rank is 30-50 (neutral zone, no edge)
    - Regime is UNKNOWN
    - Technical signals are conflicting
    - Confidence is below minimum threshold
    """

    # Minimum confidence to generate a trade recommendation
    MIN_CONFIDENCE = 0.40

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def select_strategy(
        self,
        symbol: str,
        iv_metrics: IVMetrics,
        regime_result: RegimeResult,
        market_snapshot: Optional[MarketIVSnapshot] = None,
    ) -> StrategyRecommendation:
        """
        Select the optimal options strategy.

        Args:
            symbol: Underlying ticker
            iv_metrics: IV analysis results
            regime_result: Regime detection results
            market_snapshot: Broad market IV context (optional)

        Returns:
            StrategyRecommendation (may be NO_TRADE)
        """
        iv_rank = iv_metrics.iv_rank
        regime = regime_result.regime
        technicals = regime_result.technicals

        self.logger.info(
            f"Selecting strategy for {symbol}: "
            f"IV_Rank={iv_rank:.0f} Regime={regime.value} "
            f"RSI={technicals.rsi_14:.0f if technicals else 'N/A'}"
        )

        # ================================================================
        # DECISION TREE
        # ================================================================

        # Gate 1: Unknown regime → NO TRADE
        if regime == Regime.UNKNOWN:
            return self._no_trade(symbol, iv_rank, regime, "Regime unknown - cannot assess")

        # Gate 2: Neutral IV zone (30-50) → NO TRADE (no edge)
        if 30 < iv_rank < 50:
            return self._no_trade(
                symbol, iv_rank, regime,
                f"IV Rank {iv_rank:.0f} in neutral zone (30-50) - no edge"
            )

        # ================================================================
        # HIGH IV (Rank >= 50): SELL PREMIUM
        # ================================================================
        if iv_rank >= 50:
            return self._select_high_iv_strategy(
                symbol, iv_rank, regime, technicals, market_snapshot
            )

        # ================================================================
        # LOW IV (Rank <= 30): BUY PREMIUM (directional or vol expansion)
        # ================================================================
        if iv_rank <= 30:
            return self._select_low_iv_strategy(
                symbol, iv_rank, regime, technicals, market_snapshot
            )

        # Fallback: should not reach here
        return self._no_trade(symbol, iv_rank, regime, "Fallback - no condition matched")

    # ================================================================== #
    # HIGH IV STRATEGIES (SELL PREMIUM)
    # ================================================================== #

    def _select_high_iv_strategy(
        self,
        symbol: str,
        iv_rank: float,
        regime: Regime,
        technicals: Optional[TechnicalSignals],
        market_snapshot: Optional[MarketIVSnapshot],
    ) -> StrategyRecommendation:
        """Select strategy when IV is high (>= 50) - SELL premium."""

        wing = WING_WIDTHS.get(symbol, DEFAULT_WING_WIDTH)

        # Extreme IV (> 80): Use wider wings, more conservative delta
        is_extreme = iv_rank > 80
        base_delta = 0.16 if is_extreme else 0.20

        # ----------------------------------------------------------
        # MEAN REVERTING: Perfect for iron condors
        # ----------------------------------------------------------
        if regime == Regime.MEAN_REVERTING:
            confidence = self._calc_confidence(iv_rank, regime, technicals)
            pop = 0.72 if is_extreme else 0.68  # 16-delta ≈ 68% PoP per side

            return StrategyRecommendation(
                symbol=symbol,
                strategy=StrategyType.IRON_CONDOR,
                direction=Direction.NEUTRAL,
                target_dte=45,
                target_delta=base_delta,
                wing_width=wing,
                max_contracts=3 if is_extreme else 5,
                probability_of_profit=pop,
                risk_reward_ratio=2.5,
                iv_rank=iv_rank,
                regime=regime,
                confidence=confidence,
                rationale=(
                    f"HIGH IV ({iv_rank:.0f}) + MEAN REVERTING regime = Iron Condor. "
                    f"Selling premium at {base_delta:.0%} delta, 45 DTE. "
                    f"Manage at 50% profit or 21 DTE."
                ),
            )

        # ----------------------------------------------------------
        # TRENDING BULL: Sell put credit spread (bullish + sell premium)
        # ----------------------------------------------------------
        if regime == Regime.TRENDING_BULL:
            confidence = self._calc_confidence(iv_rank, regime, technicals)

            # Extra confirmation: RSI not overbought (don't sell puts at RSI > 80)
            if technicals and technicals.rsi_14 > 80:
                confidence *= 0.6
                if confidence < self.MIN_CONFIDENCE:
                    return self._no_trade(
                        symbol, iv_rank, regime,
                        f"RSI {technicals.rsi_14:.0f} overbought - skip bull put spread"
                    )

            return StrategyRecommendation(
                symbol=symbol,
                strategy=StrategyType.BULL_PUT_SPREAD,
                direction=Direction.BULLISH,
                target_dte=45,
                target_delta=0.25 if is_extreme else 0.30,
                wing_width=wing,
                max_contracts=5,
                probability_of_profit=0.65,
                risk_reward_ratio=1.8,
                iv_rank=iv_rank,
                regime=regime,
                confidence=confidence,
                rationale=(
                    f"HIGH IV ({iv_rank:.0f}) + BULL trend = Bull Put Spread (credit). "
                    f"Selling puts in trend direction at 30-delta, 45 DTE."
                ),
            )

        # ----------------------------------------------------------
        # TRENDING BEAR: Sell call credit spread (bearish + sell premium)
        # ----------------------------------------------------------
        if regime == Regime.TRENDING_BEAR:
            confidence = self._calc_confidence(iv_rank, regime, technicals)

            # Extra confirmation: RSI not oversold (don't sell calls at RSI < 20)
            if technicals and technicals.rsi_14 < 20:
                confidence *= 0.6
                if confidence < self.MIN_CONFIDENCE:
                    return self._no_trade(
                        symbol, iv_rank, regime,
                        f"RSI {technicals.rsi_14:.0f} oversold - skip bear call spread"
                    )

            return StrategyRecommendation(
                symbol=symbol,
                strategy=StrategyType.BEAR_CALL_SPREAD,
                direction=Direction.BEARISH,
                target_dte=45,
                target_delta=0.25 if is_extreme else 0.30,
                wing_width=wing,
                max_contracts=5,
                probability_of_profit=0.65,
                risk_reward_ratio=1.8,
                iv_rank=iv_rank,
                regime=regime,
                confidence=confidence,
                rationale=(
                    f"HIGH IV ({iv_rank:.0f}) + BEAR trend = Bear Call Spread (credit). "
                    f"Selling calls in trend direction at 30-delta, 45 DTE."
                ),
            )

        # ----------------------------------------------------------
        # HIGH VOLATILITY: Iron condor with wider wings for safety
        # ----------------------------------------------------------
        if regime == Regime.HIGH_VOLATILITY:
            confidence = self._calc_confidence(iv_rank, regime, technicals)
            # Reduce size in high vol
            max_contracts = 2

            return StrategyRecommendation(
                symbol=symbol,
                strategy=StrategyType.IRON_CONDOR,
                direction=Direction.NEUTRAL,
                target_dte=45,
                target_delta=0.16,  # Wider: 16-delta for safety
                wing_width=wing * 1.5,  # Wider wings
                max_contracts=max_contracts,
                probability_of_profit=0.72,
                risk_reward_ratio=3.0,
                iv_rank=iv_rank,
                regime=regime,
                confidence=confidence,
                rationale=(
                    f"HIGH IV ({iv_rank:.0f}) + HIGH VOL regime = Wide Iron Condor. "
                    f"16-delta with 1.5x wing width, reduced size ({max_contracts} contracts)."
                ),
            )

        # Fallback for high IV
        return self._no_trade(symbol, iv_rank, regime, "High IV but no matching regime rule")

    # ================================================================== #
    # LOW IV STRATEGIES (BUY PREMIUM)
    # ================================================================== #

    def _select_low_iv_strategy(
        self,
        symbol: str,
        iv_rank: float,
        regime: Regime,
        technicals: Optional[TechnicalSignals],
        market_snapshot: Optional[MarketIVSnapshot],
    ) -> StrategyRecommendation:
        """Select strategy when IV is low (<= 30) - BUY premium or directional."""

        wing = WING_WIDTHS.get(symbol, DEFAULT_WING_WIDTH)

        # ----------------------------------------------------------
        # TRENDING BULL + LOW IV: Buy call debit spread
        # ----------------------------------------------------------
        if regime == Regime.TRENDING_BULL:
            confidence = self._calc_confidence(iv_rank, regime, technicals)

            # Only buy if MACD confirms momentum
            if technicals and technicals.macd_histogram <= 0:
                confidence *= 0.7
                if confidence < self.MIN_CONFIDENCE:
                    return self._no_trade(
                        symbol, iv_rank, regime,
                        "Low IV + Bull but MACD not confirming momentum"
                    )

            return StrategyRecommendation(
                symbol=symbol,
                strategy=StrategyType.BULL_CALL_SPREAD,
                direction=Direction.BULLISH,
                target_dte=21,  # Shorter DTE for debit spreads
                target_delta=0.40,  # Closer to ATM for debit
                wing_width=wing,
                max_contracts=3,
                probability_of_profit=0.50,
                risk_reward_ratio=1.5,
                iv_rank=iv_rank,
                regime=regime,
                confidence=confidence,
                rationale=(
                    f"LOW IV ({iv_rank:.0f}) + BULL trend = Bull Call Spread (debit). "
                    f"Cheap options, buy in trend direction at 40-delta, 21 DTE."
                ),
            )

        # ----------------------------------------------------------
        # TRENDING BEAR + LOW IV: Buy put debit spread
        # ----------------------------------------------------------
        if regime == Regime.TRENDING_BEAR:
            confidence = self._calc_confidence(iv_rank, regime, technicals)

            if technicals and technicals.macd_histogram >= 0:
                confidence *= 0.7
                if confidence < self.MIN_CONFIDENCE:
                    return self._no_trade(
                        symbol, iv_rank, regime,
                        "Low IV + Bear but MACD not confirming downtrend"
                    )

            return StrategyRecommendation(
                symbol=symbol,
                strategy=StrategyType.BEAR_PUT_SPREAD,
                direction=Direction.BEARISH,
                target_dte=21,
                target_delta=0.40,
                wing_width=wing,
                max_contracts=3,
                probability_of_profit=0.50,
                risk_reward_ratio=1.5,
                iv_rank=iv_rank,
                regime=regime,
                confidence=confidence,
                rationale=(
                    f"LOW IV ({iv_rank:.0f}) + BEAR trend = Bear Put Spread (debit). "
                    f"Cheap puts, buy in trend direction at 40-delta, 21 DTE."
                ),
            )

        # ----------------------------------------------------------
        # MEAN REVERTING + LOW IV: Usually NO TRADE
        # Low IV + range-bound = minimal edge
        # ----------------------------------------------------------
        if regime == Regime.MEAN_REVERTING:
            # Only trade if BB squeeze detected (potential breakout)
            if technicals and technicals.bb_width < 0.04:
                confidence = self._calc_confidence(iv_rank, regime, technicals)
                confidence *= 0.8  # Still cautious

                if confidence >= self.MIN_CONFIDENCE:
                    return StrategyRecommendation(
                        symbol=symbol,
                        strategy=StrategyType.LONG_STRADDLE,
                        direction=Direction.NEUTRAL,
                        target_dte=30,
                        target_delta=0.50,  # ATM
                        wing_width=0,  # Not applicable
                        max_contracts=2,
                        probability_of_profit=0.35,
                        risk_reward_ratio=0.5,  # Bad r/r but hoping for vol expansion
                        iv_rank=iv_rank,
                        regime=regime,
                        confidence=confidence,
                        rationale=(
                            f"LOW IV ({iv_rank:.0f}) + BB Squeeze ({technicals.bb_width:.2%}). "
                            f"Long straddle for potential breakout. Small size."
                        ),
                    )

            return self._no_trade(
                symbol, iv_rank, regime,
                "Low IV + Mean Reverting = no edge, waiting for breakout or IV expansion"
            )

        # ----------------------------------------------------------
        # HIGH VOLATILITY + LOW IV: Unusual combo, skip
        # ----------------------------------------------------------
        if regime == Regime.HIGH_VOLATILITY:
            return self._no_trade(
                symbol, iv_rank, regime,
                "Low IV rank + High vol regime = contradictory signals, skip"
            )

        return self._no_trade(symbol, iv_rank, regime, "Low IV - no matching rule")

    # ================================================================== #
    # HELPERS
    # ================================================================== #

    def _calc_confidence(
        self,
        iv_rank: float,
        regime: Regime,
        technicals: Optional[TechnicalSignals],
    ) -> float:
        """
        Calculate overall confidence for a trade.

        Components:
        - IV extremeness: how far from 50 the IV rank is (0-0.4)
        - Regime confidence: from the regime detector (0-0.3)
        - Technical alignment: RSI + MACD confirmation (0-0.3)
        """
        confidence = 0.0

        # IV extremeness (0 to 0.4)
        iv_distance = abs(iv_rank - 50) / 50  # 0 at IV=50, 1 at IV=0 or 100
        confidence += iv_distance * 0.4

        # Regime confidence (0 to 0.3)
        if regime != Regime.UNKNOWN:
            confidence += 0.15  # Base for having a regime
            # Add regime-specific bonus
            if regime in (Regime.MEAN_REVERTING, Regime.TRENDING_BULL, Regime.TRENDING_BEAR):
                confidence += 0.15

        # Technical alignment (0 to 0.3)
        if technicals:
            # RSI not at extremes = better entry
            if 25 <= technicals.rsi_14 <= 75:
                confidence += 0.1
            # MACD agreement with direction
            if technicals.macd_histogram != 0:
                confidence += 0.1
            # Volume above average
            if technicals.volume_ratio > 0.8:
                confidence += 0.1

        return min(confidence, 0.95)

    def _no_trade(
        self,
        symbol: str,
        iv_rank: float,
        regime: Regime,
        reason: str,
    ) -> StrategyRecommendation:
        """Generate a NO_TRADE recommendation."""
        self.logger.info(f"NO TRADE for {symbol}: {reason}")
        return StrategyRecommendation(
            symbol=symbol,
            strategy=StrategyType.NO_TRADE,
            direction=Direction.NEUTRAL,
            target_dte=0,
            target_delta=0,
            wing_width=0,
            max_contracts=0,
            probability_of_profit=0,
            risk_reward_ratio=0,
            iv_rank=iv_rank,
            regime=regime,
            confidence=0,
            rationale=f"NO TRADE: {reason}",
        )

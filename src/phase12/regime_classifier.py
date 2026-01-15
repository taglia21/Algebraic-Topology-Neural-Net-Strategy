"""
Regime Classifier for Phase 12
==============================

Multi-signal regime detection system that classifies market conditions into:
- STRONG_BULL: Full long leverage
- MILD_BULL: Reduced long leverage
- NEUTRAL: Cash/T-bills
- MILD_BEAR: Begin inverse positions
- STRONG_BEAR: Full inverse leverage

Uses multiple technical indicators with confirmation logic to avoid whipsaws.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeState(Enum):
    """Market regime classification."""
    STRONG_BULL = "strong_bull"   # 4-5 bullish signals
    MILD_BULL = "mild_bull"       # 3 bullish signals
    NEUTRAL = "neutral"           # Mixed signals
    MILD_BEAR = "mild_bear"       # 3 bearish signals
    STRONG_BEAR = "strong_bear"   # 4-5 bearish signals


@dataclass
class RegimeSignals:
    """Container for all regime signals."""
    # Primary signals
    sma_trend: int = 0           # -2 to +2 (strong bear to strong bull)
    trend_strength: float = 0.0  # ADX-like measure
    market_breadth: float = 0.5  # % stocks above 200-day MA
    
    # Secondary signals
    vix_level: float = 18.0
    vix_term_structure: int = 0  # -1 (inverted/fear), 0 (normal), +1 (contango)
    drawdown_from_peak: float = 0.0
    momentum_score: float = 0.0  # Cross-sectional momentum
    
    # Derived
    bullish_count: int = 0
    bearish_count: int = 0
    regime: RegimeState = RegimeState.NEUTRAL
    confidence: float = 0.0      # 0-1 confidence in regime classification


@dataclass
class RegimeConfig:
    """Configuration for regime classifier."""
    # SMA periods
    sma_short: int = 20
    sma_medium: int = 50
    sma_long: int = 200
    
    # Thresholds
    breadth_bull_threshold: float = 0.60   # >60% stocks above 200 SMA = bullish
    breadth_bear_threshold: float = 0.40   # <40% stocks above 200 SMA = bearish
    
    vix_low_threshold: float = 18.0        # VIX <18 = bullish
    vix_high_threshold: float = 25.0       # VIX >25 = bearish
    vix_crisis_threshold: float = 35.0     # VIX >35 = extreme fear
    
    # Drawdown thresholds for signal generation
    dd_mild_threshold: float = 0.05        # 5% DD = caution
    dd_moderate_threshold: float = 0.10    # 10% DD = concern
    dd_severe_threshold: float = 0.15      # 15% DD = bearish
    
    # Confirmation
    confirmation_days: int = 5             # Days to confirm regime change
    min_trend_strength: float = 20.0       # ADX-like minimum for trend


class RegimeClassifier:
    """
    Multi-signal regime classifier for all-weather strategy.
    
    Classification Process:
    1. Compute each signal independently
    2. Convert to bullish/bearish votes
    3. Aggregate votes to determine regime
    4. Apply confirmation logic to prevent whipsaws
    """
    
    def __init__(self, config: RegimeConfig = None):
        self.config = config or RegimeConfig()
        
        # State tracking
        self.current_regime = RegimeState.NEUTRAL
        self.regime_since = None
        self.regime_history: List[Tuple[str, RegimeState]] = []
        self.confirmation_counter = 0
        self.pending_regime: Optional[RegimeState] = None
        
    def classify(
        self,
        spy_prices: pd.Series,
        vix_prices: pd.Series = None,
        universe_prices: Dict[str, pd.Series] = None,
        current_drawdown: float = 0.0,
        date: pd.Timestamp = None,
    ) -> RegimeSignals:
        """
        Classify current market regime based on multiple signals.
        
        Args:
            spy_prices: SPY close prices (historical)
            vix_prices: VIX close prices
            universe_prices: Dict of ticker -> close prices for breadth
            current_drawdown: Current portfolio drawdown
            date: Current date
            
        Returns:
            RegimeSignals with classification
        """
        signals = RegimeSignals()
        
        # 1. SMA Trend Signal (-2 to +2)
        signals.sma_trend = self._compute_sma_trend(spy_prices)
        
        # 2. Trend Strength (ADX-like)
        signals.trend_strength = self._compute_trend_strength(spy_prices)
        
        # 3. Market Breadth
        if universe_prices:
            signals.market_breadth = self._compute_breadth(universe_prices)
        else:
            # Estimate from SPY trend
            signals.market_breadth = 0.5 + signals.sma_trend * 0.15
        
        # 4. VIX Level
        if vix_prices is not None and len(vix_prices) > 5:
            signals.vix_level = vix_prices.rolling(5).mean().iloc[-1]
        
        # 5. VIX Term Structure (simplified)
        signals.vix_term_structure = self._estimate_vix_structure(vix_prices)
        
        # 6. Drawdown Signal
        signals.drawdown_from_peak = current_drawdown
        
        # 7. Momentum Score
        signals.momentum_score = self._compute_momentum_score(spy_prices)
        
        # Count bullish/bearish signals
        signals.bullish_count, signals.bearish_count = self._count_signals(signals)
        
        # Determine raw regime
        raw_regime = self._determine_regime(signals)
        
        # Apply confirmation logic
        confirmed_regime = self._apply_confirmation(raw_regime, date)
        
        signals.regime = confirmed_regime
        signals.confidence = self._compute_confidence(signals)
        
        return signals
    
    def _compute_sma_trend(self, prices: pd.Series) -> int:
        """
        Compute SMA-based trend signal.
        
        Returns:
            -2: Strong bear (P < SMA20 < SMA50 < SMA200)
            -1: Mild bear (P < SMA50)
            0: Neutral
            +1: Mild bull (P > SMA50 > SMA200)
            +2: Strong bull (P > SMA20 > SMA50 > SMA200)
        """
        if len(prices) < self.config.sma_long:
            return 0
        
        current = prices.iloc[-1]
        sma20 = prices.rolling(self.config.sma_short).mean().iloc[-1]
        sma50 = prices.rolling(self.config.sma_medium).mean().iloc[-1]
        sma200 = prices.rolling(self.config.sma_long).mean().iloc[-1]
        
        # Strong bull
        if current > sma20 > sma50 > sma200:
            return 2
        # Mild bull
        elif current > sma50 > sma200:
            return 1
        # Strong bear
        elif current < sma20 < sma50 < sma200:
            return -2
        # Mild bear
        elif current < sma50 < sma200:
            return -1
        # Neutral
        return 0
    
    def _compute_trend_strength(self, prices: pd.Series) -> float:
        """
        Compute ADX-like trend strength indicator.
        
        Higher values = stronger trend (either direction).
        """
        if len(prices) < 30:
            return 0.0
        
        # Use rate of change volatility as proxy for trend strength
        returns = prices.pct_change().dropna()
        recent = returns.tail(20)
        
        # Directional movement
        positive = recent[recent > 0].sum()
        negative = abs(recent[recent < 0].sum())
        
        total = positive + negative
        if total == 0:
            return 0.0
        
        # DI difference normalized
        di_diff = abs(positive - negative) / total
        
        # Scale to 0-100 like ADX
        strength = di_diff * 100
        
        # Add persistence bonus
        trend_sign = np.sign(recent.mean())
        consecutive = 0
        for r in reversed(recent.values):
            if np.sign(r) == trend_sign:
                consecutive += 1
            else:
                break
        
        strength += consecutive * 2  # Bonus for persistence
        
        return min(strength, 100)
    
    def _compute_breadth(self, universe_prices: Dict[str, pd.Series]) -> float:
        """
        Compute market breadth: % of stocks above their 200-day SMA.
        """
        if not universe_prices:
            return 0.5
        
        above_200 = 0
        total = 0
        
        for ticker, prices in universe_prices.items():
            if len(prices) >= 200:
                current = prices.iloc[-1]
                sma200 = prices.rolling(200).mean().iloc[-1]
                if current > sma200:
                    above_200 += 1
                total += 1
        
        return above_200 / total if total > 0 else 0.5
    
    def _estimate_vix_structure(self, vix_prices: pd.Series) -> int:
        """
        Estimate VIX term structure from spot VIX movements.
        
        Returns:
            -1: Inverted (fear, backwardation)
            0: Normal
            +1: Contango (complacency)
        """
        if vix_prices is None or len(vix_prices) < 30:
            return 0
        
        current = vix_prices.iloc[-1]
        avg_30d = vix_prices.tail(30).mean()
        
        # If spot VIX much higher than average = inverted/fear
        if current > avg_30d * 1.2:
            return -1
        # If spot VIX much lower than average = contango/complacent
        elif current < avg_30d * 0.85:
            return 1
        return 0
    
    def _compute_momentum_score(self, prices: pd.Series) -> float:
        """
        Compute momentum score for SPY.
        
        Returns value between -1 (strong negative) and +1 (strong positive).
        """
        if len(prices) < 60:
            return 0.0
        
        # 3-month momentum
        mom_3m = prices.iloc[-1] / prices.iloc[-63] - 1 if len(prices) > 63 else 0
        
        # 1-month momentum
        mom_1m = prices.iloc[-1] / prices.iloc[-21] - 1 if len(prices) > 21 else 0
        
        # Combine
        score = 0.6 * mom_3m + 0.4 * mom_1m
        
        # Normalize to -1 to +1
        return np.clip(score * 5, -1, 1)  # 20% move = ±1
    
    def _count_signals(self, signals: RegimeSignals) -> Tuple[int, int]:
        """
        Count bullish and bearish signals.
        """
        bullish = 0
        bearish = 0
        
        # 1. SMA Trend
        if signals.sma_trend >= 1:
            bullish += 1
            if signals.sma_trend == 2:
                bullish += 1
        elif signals.sma_trend <= -1:
            bearish += 1
            if signals.sma_trend == -2:
                bearish += 1
        
        # 2. Market Breadth
        if signals.market_breadth > self.config.breadth_bull_threshold:
            bullish += 1
        elif signals.market_breadth < self.config.breadth_bear_threshold:
            bearish += 1
        
        # 3. VIX Level
        if signals.vix_level < self.config.vix_low_threshold:
            bullish += 1
        elif signals.vix_level > self.config.vix_high_threshold:
            bearish += 1
            if signals.vix_level > self.config.vix_crisis_threshold:
                bearish += 1
        
        # 4. VIX Term Structure
        if signals.vix_term_structure == 1:  # Contango
            bullish += 1
        elif signals.vix_term_structure == -1:  # Inverted
            bearish += 1
        
        # 5. Drawdown
        if signals.drawdown_from_peak < 0.02:
            bullish += 1  # Near highs = bullish
        elif signals.drawdown_from_peak > self.config.dd_moderate_threshold:
            bearish += 1
            if signals.drawdown_from_peak > self.config.dd_severe_threshold:
                bearish += 1
        
        # 6. Momentum Score
        if signals.momentum_score > 0.3:
            bullish += 1
        elif signals.momentum_score < -0.3:
            bearish += 1
        
        return bullish, bearish
    
    def _determine_regime(self, signals: RegimeSignals) -> RegimeState:
        """
        Determine regime from signal counts.
        """
        bull = signals.bullish_count
        bear = signals.bearish_count
        net = bull - bear
        
        # Strong signals
        if net >= 4:
            return RegimeState.STRONG_BULL
        elif net >= 2:
            return RegimeState.MILD_BULL
        elif net <= -4:
            return RegimeState.STRONG_BEAR
        elif net <= -2:
            return RegimeState.MILD_BEAR
        else:
            return RegimeState.NEUTRAL
    
    def _apply_confirmation(
        self, 
        raw_regime: RegimeState,
        date: pd.Timestamp = None,
    ) -> RegimeState:
        """
        Apply confirmation logic to prevent whipsaws.
        
        Regime changes require N consecutive days of new regime signals.
        """
        if raw_regime == self.current_regime:
            # Same regime, reset confirmation counter
            self.confirmation_counter = 0
            self.pending_regime = None
            return self.current_regime
        
        # Different regime detected
        if raw_regime == self.pending_regime:
            # Same pending regime, increment counter
            self.confirmation_counter += 1
            
            if self.confirmation_counter >= self.config.confirmation_days:
                # Confirmed! Change regime
                old_regime = self.current_regime
                self.current_regime = raw_regime
                self.regime_since = date
                self.confirmation_counter = 0
                self.pending_regime = None
                
                logger.info(f"REGIME CHANGE: {old_regime.value} → {raw_regime.value} on {date}")
                self.regime_history.append((str(date), raw_regime))
                
                return raw_regime
        else:
            # New pending regime
            self.pending_regime = raw_regime
            self.confirmation_counter = 1
        
        # Return current (unconfirmed)
        return self.current_regime
    
    def _compute_confidence(self, signals: RegimeSignals) -> float:
        """
        Compute confidence in regime classification.
        """
        net = abs(signals.bullish_count - signals.bearish_count)
        max_signals = 8  # Maximum possible signals
        
        # Confidence based on signal agreement
        confidence = net / max_signals
        
        # Boost for trend strength
        if signals.trend_strength > self.config.min_trend_strength:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def get_regime_leverage_multiplier(self, regime: RegimeState) -> float:
        """
        Get base leverage multiplier for regime.
        """
        multipliers = {
            RegimeState.STRONG_BULL: 1.0,   # Full leverage
            RegimeState.MILD_BULL: 0.65,    # Reduced
            RegimeState.NEUTRAL: 0.30,      # Minimal
            RegimeState.MILD_BEAR: 0.65,    # Moderate inverse
            RegimeState.STRONG_BEAR: 1.0,   # Full inverse
        }
        return multipliers.get(regime, 0.30)
    
    def is_bullish(self, regime: RegimeState = None) -> bool:
        """Check if regime is bullish."""
        regime = regime or self.current_regime
        return regime in [RegimeState.STRONG_BULL, RegimeState.MILD_BULL]
    
    def is_bearish(self, regime: RegimeState = None) -> bool:
        """Check if regime is bearish."""
        regime = regime or self.current_regime
        return regime in [RegimeState.STRONG_BEAR, RegimeState.MILD_BEAR]
    
    def reset(self):
        """Reset classifier state."""
        self.current_regime = RegimeState.NEUTRAL
        self.regime_since = None
        self.regime_history = []
        self.confirmation_counter = 0
        self.pending_regime = None

"""Advanced Alpha Engine for Phase 9.

Implements sophisticated alpha generation strategies:
1. Multi-horizon momentum (short/medium/long-term)
2. Mean reversion capture (RSI extremes, Bollinger Bands)
3. TDA-enhanced signals (topological momentum divergence)
4. Cross-sectional momentum (relative strength)
5. Regime-adaptive signal combination

Target: Extract maximum alpha across market conditions
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of alpha signals."""
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    BREAKOUT = "breakout"
    TDA_DIVERGENCE = "tda_divergence"
    CROSS_SECTIONAL = "cross_sectional"


@dataclass
class AlphaSignal:
    """Complete alpha signal with metadata."""
    ticker: str
    signal_type: SignalType
    
    # Signal values (-1 to 1, positive = bullish)
    raw_signal: float
    weighted_signal: float
    
    # Component breakdown
    momentum_signal: float
    reversal_signal: float
    tda_signal: float
    
    # Quality metrics
    signal_strength: float  # 0-1
    confidence: float      # 0-1
    regime_alignment: float  # -1 to 1
    
    # Position sizing hints
    suggested_weight: float
    max_weight: float
    
    # Risk metrics
    expected_return: float  # Annualized
    expected_vol: float
    risk_reward: float


@dataclass
class MomentumComponents:
    """Multi-horizon momentum breakdown."""
    momentum_1w: float   # 5-day
    momentum_1m: float   # 21-day
    momentum_3m: float   # 63-day
    momentum_6m: float   # 126-day
    momentum_12m: float  # 252-day
    momentum_12_1: float # 12-month skipping recent month
    
    # Acceleration
    accel_short: float  # 1-week vs prior week
    accel_medium: float # 1-month vs prior month
    
    # Trend strength
    trend_consistency: float  # % of positive periods
    slope_quality: float      # R² of price trend


class MomentumCalculator:
    """Multi-horizon momentum analysis."""
    
    def __init__(
        self,
        use_log_returns: bool = True,
        winsorize_pct: float = 0.02,
    ):
        self.use_log_returns = use_log_returns
        self.winsorize_pct = winsorize_pct
    
    def compute_momentum_components(
        self,
        prices: np.ndarray,
    ) -> MomentumComponents:
        """Compute all momentum components from price array."""
        # Reduced from 252 to 63 for faster startup
        if len(prices) < 63:
            return self._default_momentum()
        
        current = prices[-1]
        
        # Simple returns at various horizons
        mom_1w = self._safe_return(prices, 5)
        mom_1m = self._safe_return(prices, 21)
        mom_3m = self._safe_return(prices, 63)
        mom_6m = self._safe_return(prices, 126) if len(prices) >= 126 else mom_3m * 2
        mom_12m = self._safe_return(prices, 252) if len(prices) >= 252 else mom_6m * 2
        
        # 12-1 momentum (Jegadeesh-Titman) - adapt to available data
        if len(prices) >= 252:
            price_12m_ago = prices[-252]
            price_1m_ago = prices[-21]
            mom_12_1 = (price_1m_ago / price_12m_ago - 1) if price_12m_ago > 0 else 0
        elif len(prices) >= 84:
            # Use 3-month look-back minus 1 week
            price_3m_ago = prices[-63]
            price_1w_ago = prices[-5]
            mom_12_1 = (price_1w_ago / price_3m_ago - 1) if price_3m_ago > 0 else 0
        else:
            mom_12_1 = mom_3m
        
        # Acceleration
        if len(prices) >= 14:
            ret_week1 = (prices[-1] / prices[-5] - 1) if prices[-5] > 0 else 0
            ret_week2 = (prices[-5] / prices[-10] - 1) if len(prices) >= 10 and prices[-10] > 0 else 0
            accel_short = ret_week1 - ret_week2
        else:
            accel_short = 0.0
        
        if len(prices) >= 42:
            ret_month1 = (prices[-1] / prices[-21] - 1) if prices[-21] > 0 else 0
            ret_month2 = (prices[-21] / prices[-42] - 1) if prices[-42] > 0 else 0
            accel_medium = ret_month1 - ret_month2
        else:
            accel_medium = 0.0
        
        # Trend consistency (% of positive weekly returns over 6 months)
        if len(prices) >= 126:
            weekly_prices = prices[-126::5]  # Sample every 5 days
            weekly_returns = np.diff(weekly_prices) / weekly_prices[:-1]
            trend_consistency = np.mean(weekly_returns > 0) if len(weekly_returns) > 0 else 0.5
        else:
            trend_consistency = 0.5
        
        # Slope quality (R² of linear regression)
        if len(prices) >= 60:
            x = np.arange(60)
            y = np.log(prices[-60:]) if self.use_log_returns else prices[-60:]
            try:
                slope, _, r_value, _, _ = stats.linregress(x, y)
                slope_quality = r_value ** 2
            except:
                slope_quality = 0.0
        else:
            slope_quality = 0.0
        
        return MomentumComponents(
            momentum_1w=mom_1w,
            momentum_1m=mom_1m,
            momentum_3m=mom_3m,
            momentum_6m=mom_6m,
            momentum_12m=mom_12m,
            momentum_12_1=mom_12_1,
            accel_short=accel_short,
            accel_medium=accel_medium,
            trend_consistency=trend_consistency,
            slope_quality=slope_quality,
        )
    
    def compute_momentum_score(
        self,
        components: MomentumComponents,
        weights: Dict[str, float] = None,
    ) -> float:
        """Compute weighted momentum score from components."""
        # Weight shorter-term momentum more heavily for trend-following
        weights = weights or {
            'momentum_12_1': 0.15,  # Classic momentum (reduced)
            'momentum_6m': 0.15,
            'momentum_3m': 0.25,  # Emphasized
            'momentum_1m': 0.20,  # Emphasized
            'accel_medium': 0.10,
            'trend_consistency': 0.10,
            'slope_quality': 0.05,
        }
        
        score = 0.0
        score += weights.get('momentum_12_1', 0) * self._normalize_momentum(components.momentum_12_1)
        score += weights.get('momentum_6m', 0) * self._normalize_momentum(components.momentum_6m)
        score += weights.get('momentum_3m', 0) * self._normalize_momentum(components.momentum_3m)
        score += weights.get('momentum_1m', 0) * self._normalize_momentum(components.momentum_1m)
        score += weights.get('accel_medium', 0) * self._normalize_momentum(components.accel_medium * 5)
        score += weights.get('trend_consistency', 0) * (components.trend_consistency * 2 - 1)
        score += weights.get('slope_quality', 0) * (components.slope_quality * 2 - 1)
        
        return np.clip(score, -1, 1)
    
    def _safe_return(self, prices: np.ndarray, lookback: int) -> float:
        """Safely compute return with bounds checking."""
        if len(prices) < lookback or prices[-lookback] <= 0:
            return 0.0
        return prices[-1] / prices[-lookback] - 1
    
    def _normalize_momentum(self, mom: float, scale: float = 1.0) -> float:
        """Normalize momentum to -1 to 1 range - higher scale to capture big movers."""
        # Increased scale from 0.5 to 1.0 to better differentiate high momentum
        return np.clip(mom / scale, -1, 1)
    
    def _default_momentum(self) -> MomentumComponents:
        """Return neutral momentum components."""
        return MomentumComponents(
            momentum_1w=0, momentum_1m=0, momentum_3m=0, momentum_6m=0,
            momentum_12m=0, momentum_12_1=0, accel_short=0, accel_medium=0,
            trend_consistency=0.5, slope_quality=0,
        )


class ReversalCalculator:
    """Mean reversion signal calculation."""
    
    def __init__(
        self,
        rsi_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
    ):
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def compute_reversal_signals(
        self,
        prices: np.ndarray,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute all reversal indicators."""
        signals = {}
        
        if len(prices) < max(self.rsi_period, self.bb_period) + 1:
            return {'rsi': 50, 'bb_position': 0, 'reversal_score': 0}
        
        # RSI
        rsi = self._compute_rsi(prices, self.rsi_period)
        signals['rsi'] = rsi
        
        # RSI reversal signal (-1 to 1)
        if rsi < 30:
            signals['rsi_signal'] = (30 - rsi) / 30  # Oversold = positive (buy)
        elif rsi > 70:
            signals['rsi_signal'] = (70 - rsi) / 30  # Overbought = negative (sell)
        else:
            signals['rsi_signal'] = 0.0
        
        # Bollinger Band position
        bb_upper, bb_middle, bb_lower = self._compute_bollinger(prices, self.bb_period, self.bb_std)
        current = prices[-1]
        
        bb_width = bb_upper - bb_lower
        if bb_width > 0:
            bb_position = (current - bb_middle) / (bb_width / 2)  # -1 to 1
            signals['bb_position'] = np.clip(bb_position, -2, 2)
        else:
            signals['bb_position'] = 0.0
        
        # BB reversal signal (mean reversion)
        if signals['bb_position'] < -1:
            signals['bb_signal'] = min(1, abs(signals['bb_position']) - 1)  # Below lower band
        elif signals['bb_position'] > 1:
            signals['bb_signal'] = max(-1, 1 - signals['bb_position'])  # Above upper band
        else:
            signals['bb_signal'] = 0.0
        
        # Z-score reversal
        if len(prices) >= 60:
            mean_60 = np.mean(prices[-60:])
            std_60 = np.std(prices[-60:])
            if std_60 > 0:
                z_score = (current - mean_60) / std_60
                signals['z_score'] = z_score
                # Reversal signal: fade extremes
                if z_score < -2:
                    signals['z_signal'] = min(1, (abs(z_score) - 2) / 2)
                elif z_score > 2:
                    signals['z_signal'] = max(-1, (2 - z_score) / 2)
                else:
                    signals['z_signal'] = 0.0
            else:
                signals['z_score'] = 0.0
                signals['z_signal'] = 0.0
        else:
            signals['z_score'] = 0.0
            signals['z_signal'] = 0.0
        
        # Combined reversal score
        signals['reversal_score'] = (
            0.4 * signals.get('rsi_signal', 0) +
            0.35 * signals.get('bb_signal', 0) +
            0.25 * signals.get('z_signal', 0)
        )
        
        return signals
    
    def _compute_rsi(self, prices: np.ndarray, period: int) -> float:
        """Compute RSI."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-(period+1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _compute_bollinger(
        self,
        prices: np.ndarray,
        period: int,
        num_std: float,
    ) -> Tuple[float, float, float]:
        """Compute Bollinger Bands."""
        if len(prices) < period:
            return prices[-1] * 1.02, prices[-1], prices[-1] * 0.98
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + num_std * std
        lower = sma - num_std * std
        
        return upper, sma, lower


class TDAAlphaExtractor:
    """Extract alpha signals from TDA features."""
    
    def __init__(
        self,
        turbulence_weight: float = 0.30,
        persistence_weight: float = 0.40,
        regime_weight: float = 0.30,
    ):
        self.turbulence_weight = turbulence_weight
        self.persistence_weight = persistence_weight
        self.regime_weight = regime_weight
    
    def extract_tda_signal(
        self,
        tda_features: pd.DataFrame,
        ticker: str,
        lookback: int = 10,
    ) -> Dict[str, float]:
        """Extract TDA-based alpha signal."""
        if tda_features is None or len(tda_features) < lookback:
            return {'tda_signal': 0, 'tda_confidence': 0.5}
        
        recent = tda_features.iloc[-lookback:]
        signals = {}
        
        # Turbulence signal (low turbulence = bullish)
        if 'turbulence_index' in recent.columns:
            turb = recent['turbulence_index'].iloc[-1]
            turb_ma = recent['turbulence_index'].mean()
            
            # Below-average turbulence is bullish
            if turb_ma > 0:
                turb_rel = (turb_ma - turb) / turb_ma  # Positive when below average
                signals['turbulence_signal'] = np.clip(turb_rel, -1, 1)
            else:
                signals['turbulence_signal'] = 0.0
        else:
            signals['turbulence_signal'] = 0.0
        
        # Persistence signal (stable topology = trending)
        for col in ['persistence_l0', 'total_persistence_h0', 'max_persistence_h0']:
            if col in recent.columns:
                pers = recent[col].iloc[-1]
                pers_ma = recent[col].mean()
                
                if pers_ma > 0:
                    pers_rel = (pers - pers_ma) / pers_ma
                    signals['persistence_signal'] = np.clip(pers_rel * 2, -1, 1)
                else:
                    signals['persistence_signal'] = 0.0
                break
        else:
            signals['persistence_signal'] = 0.0
        
        # Regime stability (low fragmentation = stable = continuation)
        if 'betti_0' in recent.columns:
            b0 = recent['betti_0'].iloc[-1]
            b0_ma = recent['betti_0'].mean()
            
            # Low fragmentation is bullish
            signals['fragmentation_signal'] = np.clip(1 - b0 / 10, -1, 1)
        else:
            signals['fragmentation_signal'] = 0.0
        
        # Combined TDA signal
        tda_signal = (
            self.turbulence_weight * signals.get('turbulence_signal', 0) +
            self.persistence_weight * signals.get('persistence_signal', 0) +
            self.regime_weight * signals.get('fragmentation_signal', 0)
        )
        
        signals['tda_signal'] = np.clip(tda_signal, -1, 1)
        signals['tda_confidence'] = 0.5 + 0.5 * abs(tda_signal)  # Higher confidence for stronger signals
        
        return signals
    
    def detect_momentum_divergence(
        self,
        prices: np.ndarray,
        tda_features: pd.DataFrame,
        lookback: int = 20,
    ) -> Dict[str, float]:
        """
        Detect divergence between price momentum and TDA topology.
        
        Bullish divergence: Price down, TDA improving (lower turbulence)
        Bearish divergence: Price up, TDA deteriorating
        """
        if len(prices) < lookback or tda_features is None or len(tda_features) < lookback:
            return {'divergence': 0, 'divergence_type': 'none'}
        
        # Price momentum
        price_change = prices[-1] / prices[-lookback] - 1
        
        # TDA momentum (change in turbulence)
        if 'turbulence_index' in tda_features.columns:
            turb_recent = tda_features['turbulence_index'].iloc[-5:].mean()
            turb_prior = tda_features['turbulence_index'].iloc[-lookback:-5].mean()
            turb_change = (turb_recent - turb_prior) / (turb_prior + 1e-6)
        else:
            turb_change = 0.0
        
        # Detect divergence
        if price_change < -0.05 and turb_change < -0.1:
            # Price down, turbulence decreasing = bullish divergence
            divergence = min(1, abs(price_change) + abs(turb_change))
            divergence_type = 'bullish'
        elif price_change > 0.05 and turb_change > 0.1:
            # Price up, turbulence increasing = bearish divergence
            divergence = -min(1, abs(price_change) + abs(turb_change))
            divergence_type = 'bearish'
        else:
            divergence = 0.0
            divergence_type = 'none'
        
        return {'divergence': divergence, 'divergence_type': divergence_type}


class CrossSectionalRanker:
    """Cross-sectional momentum and quality ranking."""
    
    def __init__(
        self,
        universe_size: int = 100,
        top_percentile: float = 0.20,
        bottom_percentile: float = 0.20,
    ):
        self.universe_size = universe_size
        self.top_percentile = top_percentile
        self.bottom_percentile = bottom_percentile
    
    def compute_cross_sectional_ranks(
        self,
        all_scores: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute cross-sectional ranks for all tickers.
        
        Returns dict: {ticker: {'rank': int, 'percentile': float, 'z_score': float}}
        """
        if not all_scores:
            return {}
        
        tickers = list(all_scores.keys())
        scores = np.array(list(all_scores.values()))
        
        # Winsorize extreme values
        lower, upper = np.percentile(scores, [2, 98])
        scores_clipped = np.clip(scores, lower, upper)
        
        # Compute z-scores
        mean_score = np.mean(scores_clipped)
        std_score = np.std(scores_clipped) + 1e-6
        z_scores = (scores_clipped - mean_score) / std_score
        
        # Compute ranks (higher score = lower rank number)
        ranks = len(scores) - stats.rankdata(scores) + 1
        percentiles = 1 - (ranks - 1) / len(scores)
        
        results = {}
        for i, ticker in enumerate(tickers):
            results[ticker] = {
                'rank': int(ranks[i]),
                'percentile': percentiles[i],
                'z_score': z_scores[i],
                'raw_score': scores[i],
                'is_top': percentiles[i] >= (1 - self.top_percentile),
                'is_bottom': percentiles[i] <= self.bottom_percentile,
            }
        
        return results


class AdvancedAlphaEngine:
    """
    Complete advanced alpha generation engine.
    
    Combines momentum, reversal, TDA, and cross-sectional signals
    with regime-adaptive weighting.
    """
    
    def __init__(
        self,
        momentum_calc: Optional[MomentumCalculator] = None,
        reversal_calc: Optional[ReversalCalculator] = None,
        tda_extractor: Optional[TDAAlphaExtractor] = None,
        cross_sectional: Optional[CrossSectionalRanker] = None,
    ):
        self.momentum_calc = momentum_calc or MomentumCalculator()
        self.reversal_calc = reversal_calc or ReversalCalculator()
        self.tda_extractor = tda_extractor or TDAAlphaExtractor()
        self.cross_sectional = cross_sectional or CrossSectionalRanker()
        
        # Default signal weights
        self.signal_weights = {
            'momentum': 0.40,
            'reversal': 0.15,
            'tda': 0.25,
            'cross_sectional': 0.20,
        }
        
        # Position sizing parameters
        self.max_position_weight = 0.10
        self.min_position_weight = 0.02
        self.confidence_scaling = True
    
    def generate_alpha_signal(
        self,
        ticker: str,
        prices: np.ndarray,
        tda_features: Optional[pd.DataFrame] = None,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
        cross_sectional_data: Optional[Dict[str, float]] = None,
        regime_weights: Optional[Dict[str, float]] = None,
    ) -> AlphaSignal:
        """
        Generate complete alpha signal for a ticker.
        
        Args:
            ticker: Stock ticker
            prices: Price array (close prices)
            tda_features: TDA features DataFrame
            high/low: High/low price arrays for reversal
            cross_sectional_data: Cross-sectional scores for all tickers
            regime_weights: Regime-specific signal weights
            
        Returns:
            AlphaSignal with full signal breakdown
        """
        weights = regime_weights or self.signal_weights
        
        # 1. Momentum signal
        mom_components = self.momentum_calc.compute_momentum_components(prices)
        momentum_signal = self.momentum_calc.compute_momentum_score(mom_components)
        
        # 2. Reversal signal
        reversal_data = self.reversal_calc.compute_reversal_signals(prices, high, low)
        reversal_signal = reversal_data.get('reversal_score', 0)
        
        # 3. TDA signal
        tda_data = self.tda_extractor.extract_tda_signal(tda_features, ticker)
        tda_signal = tda_data.get('tda_signal', 0)
        
        # TDA divergence boost
        divergence = self.tda_extractor.detect_momentum_divergence(prices, tda_features)
        if divergence['divergence_type'] != 'none':
            tda_signal = (tda_signal + divergence['divergence']) / 2
        
        # 4. Cross-sectional signal
        if cross_sectional_data:
            cs_ranks = self.cross_sectional.compute_cross_sectional_ranks(cross_sectional_data)
            if ticker in cs_ranks:
                cs_signal = cs_ranks[ticker]['z_score'] / 2  # Scale z-score
            else:
                cs_signal = 0.0
        else:
            cs_signal = 0.0
        
        # Combine signals with regime weights
        raw_signal = (
            weights.get('momentum', 0.4) * momentum_signal +
            weights.get('reversal', 0.15) * reversal_signal +
            weights.get('tda', 0.25) * tda_signal +
            weights.get('cross_sectional', 0.2) * cs_signal
        )
        
        # Compute confidence
        signal_strength = abs(raw_signal)
        confidence = self._compute_confidence(
            momentum_signal, reversal_signal, tda_signal, mom_components
        )
        
        # Regime alignment (how well signals agree)
        signals = [momentum_signal, reversal_signal, tda_signal, cs_signal]
        non_zero = [s for s in signals if abs(s) > 0.05]
        if len(non_zero) >= 2:
            signs = [np.sign(s) for s in non_zero]
            regime_alignment = sum(signs) / len(signs)  # 1 if all agree
        else:
            regime_alignment = 0.0
        
        # Position sizing - more aggressive signal scaling
        # Boost signals when confidence is high, but don't penalize too much when low
        weighted_signal = raw_signal * (0.5 + 0.5 * confidence) * (0.6 + 0.4 * abs(regime_alignment))
        suggested_weight = self._compute_position_weight(weighted_signal, confidence)
        
        # Risk metrics
        expected_return = self._estimate_return(raw_signal, mom_components)
        expected_vol = self._estimate_volatility(prices)
        risk_reward = expected_return / expected_vol if expected_vol > 0 else 0
        
        return AlphaSignal(
            ticker=ticker,
            signal_type=self._classify_signal_type(momentum_signal, reversal_signal),
            raw_signal=raw_signal,
            weighted_signal=weighted_signal,
            momentum_signal=momentum_signal,
            reversal_signal=reversal_signal,
            tda_signal=tda_signal,
            signal_strength=signal_strength,
            confidence=confidence,
            regime_alignment=regime_alignment,
            suggested_weight=suggested_weight,
            max_weight=self.max_position_weight,
            expected_return=expected_return,
            expected_vol=expected_vol,
            risk_reward=risk_reward,
        )
    
    def _compute_confidence(
        self,
        momentum: float,
        reversal: float,
        tda: float,
        mom_components: MomentumComponents,
    ) -> float:
        """Compute signal confidence."""
        confidence = 0.5  # Base confidence
        
        # Higher confidence when signals align
        signals = [momentum, reversal, tda]
        if all(s > 0 for s in signals if abs(s) > 0.1):
            confidence += 0.2
        elif all(s < 0 for s in signals if abs(s) > 0.1):
            confidence += 0.2
        
        # Higher confidence with strong trend
        if mom_components.slope_quality > 0.7:
            confidence += 0.15
        if mom_components.trend_consistency > 0.7:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _compute_position_weight(
        self,
        signal: float,
        confidence: float,
    ) -> float:
        """Compute suggested position weight."""
        # Scale weight by signal strength and confidence
        base_weight = (self.max_position_weight + self.min_position_weight) / 2
        
        if self.confidence_scaling:
            weight = base_weight * abs(signal) * confidence
        else:
            weight = base_weight * abs(signal)
        
        return np.clip(weight, self.min_position_weight, self.max_position_weight)
    
    def _classify_signal_type(
        self,
        momentum: float,
        reversal: float,
    ) -> SignalType:
        """Classify the dominant signal type."""
        if abs(momentum) > abs(reversal) * 2:
            return SignalType.MOMENTUM
        elif abs(reversal) > abs(momentum) * 2:
            return SignalType.REVERSAL
        else:
            return SignalType.MOMENTUM  # Default
    
    def _estimate_return(
        self,
        signal: float,
        mom: MomentumComponents,
    ) -> float:
        """Estimate expected annualized return."""
        # Simple heuristic: scale signal by historical momentum
        base_return = signal * 0.30  # 30% annualized for max signal
        
        # Adjust by trend quality
        quality_adj = 1 + mom.slope_quality * 0.2
        
        return base_return * quality_adj
    
    def _estimate_volatility(self, prices: np.ndarray) -> float:
        """Estimate annualized volatility."""
        if len(prices) < 20:
            return 0.20
        
        returns = np.diff(prices[-60:]) / prices[-60:-1]
        return np.std(returns) * np.sqrt(252)
    
    def update_weights(self, regime_weights: Dict[str, float]):
        """Update signal weights based on regime."""
        self.signal_weights = regime_weights

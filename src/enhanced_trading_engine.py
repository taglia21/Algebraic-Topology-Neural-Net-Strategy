"""
Enhanced Trading Engine - Production Integration Layer
=======================================================

Orchestrates all trading modules into a cohesive decision-making system.

Integrates:
- Risk Manager: Portfolio-level risk controls
- Position Sizer: Kelly Criterion-based sizing
- Multi-Timeframe Analyzer: Trend alignment scoring
- Sentiment Analyzer: News sentiment analysis

Author: Trading System
Version: 1.0.0
"""

import logging
import time
from functools import wraps
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

import yfinance as yf
import numpy as np
import pandas as pd

from src.risk_manager import RiskManager, RiskConfig, Position
from src.position_sizer import PositionSizer, SizingConfig, PerformanceMetrics
from src.multi_timeframe_analyzer import MultiTimeframeAnalyzer, AnalyzerConfig
from src.sentiment_analyzer import SentimentAnalyzer, SentimentConfig
from src.medallion_math import MedallionStrategy

# ==== PHASE 4 & 6: WIRED ORPHANED MODULES ====
try:
    from src.signal_aggregator import SignalAggregator
    AGGREGATOR_AVAILABLE = True
except ImportError:
    SignalAggregator = None
    AGGREGATOR_AVAILABLE = False

try:
    from src.quant_models.capm import CAPMModel
    CAPM_AVAILABLE = True
except ImportError:
    CAPMModel = None
    CAPM_AVAILABLE = False

try:
    from src.quant_models.garch import GARCHModel
    GARCH_AVAILABLE = True
except ImportError:
    GARCHModel = None
    GARCH_AVAILABLE = False

try:
    from src.ml.stacked_ensemble import StackedEnsemble
    from src.ml.transformer_predictor import TransformerPredictor
    ML_AVAILABLE = True
except ImportError:
    StackedEnsemble = None
    TransformerPredictor = None
    ML_AVAILABLE = False

try:
    from src.ml.adaptive_ensemble import AdaptiveEnsemble
    ADAPTIVE_ML_AVAILABLE = True
except ImportError:
    AdaptiveEnsemble = None
    ADAPTIVE_ML_AVAILABLE = False

try:
    from src.ml.continuous_learner import ContinuousLearner
    CONTINUOUS_LEARNER_AVAILABLE = True
except ImportError:
    ContinuousLearner = None
    CONTINUOUS_LEARNER_AVAILABLE = False

try:
    from src.optimization.bayesian_tuner import BayesianTuner
    BAYESIAN_AVAILABLE = True
except ImportError:
    BayesianTuner = None
    BAYESIAN_AVAILABLE = False

logger = logging.getLogger(__name__)


def retry_yfinance(max_retries=3, backoff=2.0):
    """
    CRITICAL FIX: Decorator for yfinance calls with exponential backoff retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff: Backoff multiplier for exponential wait
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"yfinance call failed after {max_retries} attempts: {e}")
                        raise
                    
                    wait_time = backoff ** attempt
                    logger.warning(f"yfinance call failed (attempt {attempt+1}/{max_retries}), "
                                 f"retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
            
            return None
        return wrapper
    return decorator


class TradeSignal(Enum):
    """Trade signal classification."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class TradeDecision:
    """Complete trading decision with all factors."""
    symbol: str
    timestamp: datetime
    signal: TradeSignal
    
    # Scores and metrics
    mtf_score: float
    sentiment_score: float
    combined_score: float
    confidence: float
    
    # Position sizing
    recommended_position_value: float
    recommended_quantity: float
    
    # Risk parameters
    entry_price: float
    stop_loss: float
    take_profits: List[float]
    
    # Decision factors
    is_tradeable: bool
    rejection_reasons: List[str]
    
    # Supporting data
    metadata: Dict


@dataclass
class EngineConfig:
    """Enhanced trading engine configuration."""
    # Module configs
    risk_config: Optional[RiskConfig] = None
    sizing_config: Optional[SizingConfig] = None
    analyzer_config: Optional[AnalyzerConfig] = None
    sentiment_config: Optional[SentimentConfig] = None
    
    # Decision thresholds
    min_mtf_score: float = 40.0  # Minimum multi-timeframe alignment (was 60 — unreachable)
    min_sentiment_score: float = -0.8  # Minimum sentiment (-1 to 1)
    min_combined_score: float = 0.45  # Minimum combined score for trading (was 0.6 — impossible with broken sentiment)
    
    # Scoring weights — technical-first bot, sentiment is supplementary
    mtf_weight: float = 0.80
    sentiment_weight: float = 0.20
    
    # ATR calculation
    atr_period: int = 14
    atr_multiplier: float = 2.0


class EnhancedTradingEngine:
    """
    Production-ready trading engine integrating all modules.
    
    Execution pipeline:
    1. Risk check (portfolio limits)
    2. Multi-timeframe analysis
    3. Sentiment analysis
    4. Combined scoring
    5. Position sizing
    6. Trade execution decision
    """
    
    def __init__(self, config: Optional[EngineConfig] = None):
        """
        Initialize enhanced trading engine.
        
        Args:
            config: Engine configuration (uses defaults if None)
        """
        self.config = config or EngineConfig()
        
        # Initialize modules
        self.risk_manager = RiskManager(self.config.risk_config)
        self.position_sizer = PositionSizer(self.config.sizing_config)
        self.mtf_analyzer = MultiTimeframeAnalyzer(self.config.analyzer_config)
        self.sentiment_analyzer = SentimentAnalyzer(self.config.sentiment_config)
        self.medallion_strategy = MedallionStrategy()

        # ==== PHASE 4 & 6: Wired advanced modules ====
        self.signal_aggregator = None
        if AGGREGATOR_AVAILABLE:
            try:
                self.signal_aggregator = SignalAggregator(min_confidence=0.5)
                self.signal_aggregator.initialize()
                logger.info("✓ SignalAggregator wired into equity engine")
            except Exception as e:
                logger.warning(f"SignalAggregator init failed: {e}")

        self.capm_model = None
        if CAPM_AVAILABLE:
            try:
                self.capm_model = CAPMModel()
                logger.info("✓ CAPM wired into equity engine")
            except Exception as e:
                logger.warning(f"CAPM init failed: {e}")

        self.garch_model = None
        if GARCH_AVAILABLE:
            try:
                self.garch_model = GARCHModel()
                logger.info("✓ GARCH wired into equity engine")
            except Exception as e:
                logger.warning(f"GARCH init failed: {e}")

        # AdaptiveEnsemble — self-training ML pipeline (replaces broken StackedEnsemble)
        self.adaptive_ml = None
        if ADAPTIVE_ML_AVAILABLE:
            try:
                self.adaptive_ml = AdaptiveEnsemble()
                logger.info("✓ AdaptiveEnsemble (self-training ML) wired into equity engine")
            except Exception as e:
                logger.warning(f"AdaptiveEnsemble init failed: {e}")

        # Legacy ML (kept for fallback but no longer primary)
        self.ml_ensemble = None
        self.ml_transformer = None

        self.continuous_learner = None
        if CONTINUOUS_LEARNER_AVAILABLE:
            try:
                self.continuous_learner = ContinuousLearner()
                logger.info("✓ ContinuousLearner wired into equity engine")
            except Exception as e:
                logger.warning(f"ContinuousLearner init failed: {e}")

        self.bayesian_tuner = None
        if BAYESIAN_AVAILABLE:
            try:
                self.bayesian_tuner = BayesianTuner()
                logger.info("✓ BayesianTuner wired into equity engine")
            except Exception as e:
                logger.warning(f"BayesianTuner init failed: {e}")
        
        logger.info("EnhancedTradingEngine initialized with all modules (including Medallion Math)")
        logger.info(f"Phase4 modules: Aggregator={self.signal_aggregator is not None}, "
                   f"CAPM={self.capm_model is not None}, GARCH={self.garch_model is not None}, "
                   f"AdaptiveML={self.adaptive_ml is not None}")
    
    @retry_yfinance(max_retries=3)
    def _calculate_atr(self, symbol: str, period: int = 14) -> float:
        """
        Calculate Average True Range with CRITICAL NaN and inf protection + retry logic.
        
        Args:
            symbol: Stock symbol
            period: ATR period
            
        Returns:
            ATR value
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='3mo', interval='1d')
            
            if data.empty or len(data) < period + 5:  # Extra buffer for shift()
                logger.warning(f"Insufficient data for ATR: {len(data)} bars")
                return 0.0
            
            # Calculate True Range components
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            
            # Combine and calculate ATR
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            # CRITICAL: Get last valid value, skip NaN
            atr_value = atr.dropna().iloc[-1] if not atr.dropna().empty else 0.0
            
            # CRITICAL: Validate result for NaN, inf, or invalid values
            if not np.isfinite(atr_value) or atr_value <= 0:
                logger.error(f"Invalid ATR calculated for {symbol}: {atr_value}")
                return 0.0
            
            logger.debug(f"{symbol} ATR({period}): {atr_value:.2f}")
            return float(atr_value)
        
        except Exception as e:
            logger.error(f"Error calculating ATR for {symbol}: {e}")
            return 0.0
    
    @retry_yfinance(max_retries=3)
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price with HIGH-SEVERITY validation + retry logic.
        MEDIUM-SEVERITY FIX: Returns None on error for explicit failure handling.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price or None if invalid
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            
            if data.empty:
                logger.warning(f"No price data for {symbol}")
                return None  # MEDIUM-SEVERITY FIX: Explicit None instead of 0.0
            
            price = float(data['Close'].iloc[-1])
            
            # Validate price
            if not np.isfinite(price) or price <= 0:
                logger.error(f"Invalid price for {symbol}: {price}")
                return None
            
            logger.debug(f"{symbol} current price: ${price:.2f}")
            return price
        
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None
    
    def _calculate_combined_score(self, mtf_score: float, sentiment_score: float) -> float:
        """
        Calculate combined score from all factors.
        
        Args:
            mtf_score: Multi-timeframe score (0-100)
            sentiment_score: Sentiment score (-1 to 1)
            
        Returns:
            Combined score (0-1)
        """
        # Normalize MTF score to 0-1
        mtf_normalized = mtf_score / 100.0
        
        # Normalize sentiment to 0-1
        sentiment_normalized = (sentiment_score + 1.0) / 2.0
        
        # Weighted combination
        combined = (mtf_normalized * self.config.mtf_weight + 
                   sentiment_normalized * self.config.sentiment_weight)
        
        return combined
    
    def _determine_signal(self, combined_score: float, mtf_score: float, 
                         sentiment_score: float) -> TradeSignal:
        """
        Determine trade signal from scores.
        
        Args:
            combined_score: Combined score (0-1)
            mtf_score: Multi-timeframe score (0-100)
            sentiment_score: Sentiment score (-1 to 1)
            
        Returns:
            TradeSignal enum
        """
        # Signal thresholds calibrated for mtf_weight=0.80, sentiment_weight=0.20
        # With neutral sentiment (0.5 normalized), combined ≈ MTF_norm * 0.8 + 0.1
        # MTF=55 → combined=0.54, MTF=65 → combined=0.62, MTF=40 → combined=0.42
        if combined_score > 0.70 and mtf_score > 70:
            return TradeSignal.STRONG_BUY
        elif combined_score > 0.55:
            return TradeSignal.BUY
        elif combined_score < 0.25:
            return TradeSignal.STRONG_SELL
        elif combined_score < 0.40:
            return TradeSignal.SELL
        else:
            return TradeSignal.HOLD
    
    def analyze_opportunity(self, symbol: str, portfolio_value: float,
                           performance_metrics: Optional[PerformanceMetrics] = None) -> TradeDecision:
        """
        Analyze a trading opportunity through complete pipeline.
        
        Args:
            symbol: Stock symbol to analyze
            portfolio_value: Current portfolio value
            performance_metrics: Historical performance for position sizing
            
        Returns:
            TradeDecision with complete analysis
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing opportunity: {symbol}")
        logger.info(f"{'='*60}")
        
        rejection_reasons = []
        
        # Step 1: Get current price and ATR
        logger.info("Step 1: Fetching market data...")
        entry_price = self._get_current_price(symbol)
        if entry_price is None:  # HIGH-SEVERITY FIX: Check for None instead of <= 0
            rejection_reasons.append("Failed to fetch current price")
            return self._create_rejection_decision(symbol, rejection_reasons)
        
        atr = self._calculate_atr(symbol, self.config.atr_period)
        if atr <= 0:
            rejection_reasons.append("Failed to calculate ATR")
            atr = entry_price * 0.02  # Fallback: 2% of price
        
        # Step 2: Multi-timeframe analysis
        logger.info("Step 2: Multi-timeframe analysis...")
        mtf_analysis = self.mtf_analyzer.analyze(symbol)
        mtf_score = mtf_analysis.alignment_score
        
        logger.info(f"  MTF Score: {mtf_score:.1f}/100")
        logger.info(f"  Dominant Trend: {mtf_analysis.dominant_trend.name}")
        logger.info(f"  Bullish TFs: {mtf_analysis.bullish_timeframes}, "
                   f"Bearish TFs: {mtf_analysis.bearish_timeframes}")
        
        if mtf_score < self.config.min_mtf_score:
            rejection_reasons.append(
                f"MTF score {mtf_score:.1f} below minimum {self.config.min_mtf_score}"
            )
        
        # Step 3: Sentiment analysis
        logger.info("Step 3: Sentiment analysis...")
        sentiment_result = self.sentiment_analyzer.get_sentiment(symbol)
        
        # Sentiment is OPTIONAL — unavailable data should NOT block trades
        # It's an additive signal, not a veto gate
        if not sentiment_result.is_valid:
            logger.info("  ⚠ Sentiment data unavailable — using neutral (not blocking trade)")
            from src.sentiment_analyzer import SentimentResult, SentimentLevel
            sentiment_result = SentimentResult(
                symbol=symbol,
                timestamp=datetime.now(),
                score=0.0,
                level=SentimentLevel.NEUTRAL,
                article_count=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                articles=[],
                data_source='NEUTRAL_FALLBACK',
                is_valid=True,
            )
        
        sentiment_score = sentiment_result.score
        
        logger.info(f"  Sentiment Score: {sentiment_score:+.2f}")
        logger.info(f"  Sentiment Level: {sentiment_result.level.value}")
        logger.info(f"  Articles: {sentiment_result.article_count} "
                   f"({sentiment_result.positive_count}+ / {sentiment_result.negative_count}-)")
        
        if sentiment_score < self.config.min_sentiment_score:
            rejection_reasons.append(
                f"Sentiment {sentiment_score:+.2f} below minimum {self.config.min_sentiment_score:+.2f}"
            )
        
        # Step 3.5: Medallion mathematical analysis
        logger.info("Step 3.5: Medallion mathematical analysis...")
        medallion_analysis = None
        try:
            # Fetch historical price data for Medallion analysis
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period='6mo', interval='1d')
            
            if len(hist_data) >= 100:  # Need sufficient data
                # Convert to writable numpy arrays (yfinance returns read-only)
                prices = np.array(hist_data['Close'].values, dtype=np.float64)
                volumes = np.array(hist_data['Volume'].values, dtype=np.float64) if 'Volume' in hist_data else None
                
                medallion_analysis = self.medallion_strategy.analyze(prices, volumes)
                
                logger.info(f"  Hurst Exponent: {medallion_analysis['hurst_exponent']:.3f}")
                logger.info(f"  Market Regime: {medallion_analysis['regime']}")
                logger.info(f"  Recommended Strategy: {medallion_analysis['recommended_strategy']}")
                logger.info(f"  Strategy Confidence: {medallion_analysis['strategy_confidence']:.1%}")
                logger.info(f"  O-U Z-Score: {medallion_analysis['ou_signal']['z_score']:.2f}")
                logger.info(f"  Half-Life: {medallion_analysis['half_life_days']:.1f} days")
                
                # Reject low confidence trades
                if medallion_analysis['strategy_confidence'] < 0.3:
                    rejection_reasons.append(
                        f"Medallion confidence {medallion_analysis['strategy_confidence']:.1%} too low"
                    )
                
                # Warn if regime suggests caution
                if medallion_analysis['regime'] == 'HighVol':
                    logger.warning("⚠️  High volatility regime detected - will reduce position size")
                elif medallion_analysis['regime'] == 'Bear':
                    logger.warning("⚠️  Bear regime detected - will reduce position size")
            else:
                logger.warning(f"Insufficient historical data for Medallion analysis: {len(hist_data)} bars")
        except Exception as e:
            logger.error(f"Medallion analysis failed: {e}")
            # Don't reject trade if Medallion fails, just log it
        
        # Step 4: Combined scoring (ENHANCED with SignalAggregator, CAPM, GARCH)
        logger.info("Step 4: Combined scoring (enhanced)...")

        # 4a: CAPM expected return screening
        capm_expected_return = None
        if self.capm_model is not None:
            try:
                capm_result = self.capm_model.analyze(symbol)
                capm_expected_return = capm_result.expected_return
                logger.info(f"  CAPM: β={capm_result.beta:.3f}, α={capm_result.alpha:.4f}, "
                           f"E[r]={capm_result.expected_return:.2%}")
                if capm_result.expected_return < 0.02:
                    rejection_reasons.append(
                        f"CAPM expected return {capm_result.expected_return:.2%} too low"
                    )
            except Exception as e:
                logger.debug(f"CAPM analysis failed for {symbol}: {e}")

        # 4b: GARCH vol for position sizing
        garch_vol = None
        if self.garch_model is not None:
            try:
                garch_result = self.garch_model.fit_and_forecast(symbol, horizon=5)
                garch_vol = garch_result.current_vol
                logger.info(f"  GARCH: current_vol={garch_vol:.1%}, VaR95={garch_result.var_95:.2%}")
            except Exception as e:
                logger.debug(f"GARCH failed for {symbol}: {e}")

        # 4c: SignalAggregator ensemble signal
        aggregated_signal = None
        aggregator_boost = 0.0
        if self.signal_aggregator is not None:
            try:
                aggregated_signal = self.signal_aggregator.aggregate(symbol, min_confidence=0.4)
                logger.info(f"  Aggregator: signal={aggregated_signal.signal:.3f} ({aggregated_signal.direction}), "
                           f"confidence={aggregated_signal.confidence:.3f}, regime={aggregated_signal.regime.value}")
                # Boost or dampen combined score based on aggregator
                aggregator_boost = aggregated_signal.signal * 0.15  # ±15% adjustment
            except Exception as e:
                logger.debug(f"SignalAggregator failed for {symbol}: {e}")

        # 4d: AdaptiveEnsemble ML signal (self-training, replaces broken StackedEnsemble + Transformer)
        ml_signal = None
        ml_confidence = 0.0
        if self.adaptive_ml is not None:
            try:
                ml_signal, ml_confidence = self.adaptive_ml.predict(symbol)
                logger.info(f"  AdaptiveML: signal={ml_signal:+.3f}, confidence={ml_confidence:.3f}")
                # ML contributes up to ±15% of combined score, weighted by its own confidence
                aggregator_boost += ml_signal * ml_confidence * 0.15
            except Exception as e:
                logger.debug(f"AdaptiveEnsemble failed for {symbol}: {e}")

        combined_score = self._calculate_combined_score(mtf_score, sentiment_score)
        combined_score = max(0.0, min(1.0, combined_score + aggregator_boost))
        # Confidence: weight toward combined_score so broken sentiment doesn't poison it
        # When sentiment is real, it boosts; when neutral fallback, combined_score dominates
        confidence = combined_score * 0.70 + sentiment_result.confidence * 0.30
        
        logger.info(f"  Combined Score: {combined_score:.2f}")
        logger.info(f"  Confidence: {confidence:.2%}")
        
        if combined_score < self.config.min_combined_score:
            rejection_reasons.append(
                f"Combined score {combined_score:.2f} below minimum {self.config.min_combined_score}"
            )
        
        # Step 5: Risk calculations
        logger.info("Step 5: Risk calculations...")
        stop_loss = self.risk_manager.calculate_stop_loss(entry_price, atr)
        take_profits = self.risk_manager.calculate_take_profits(entry_price, stop_loss)
        
        logger.info(f"  Entry: ${entry_price:.2f}")
        logger.info(f"  Stop Loss: ${stop_loss:.2f} ({((stop_loss/entry_price-1)*100):+.1f}%)")
        logger.info(f"  Take Profits: {[f'${tp:.2f}' for tp in take_profits]}")
        
        # Step 6: Position sizing
        logger.info("Step 6: Position sizing...")
        position_size = self.position_sizer.size_position(
            portfolio_value=portfolio_value,
            confidence=confidence,
            volatility_percentile=50.0,  # Could enhance with actual percentile
            performance_metrics=performance_metrics
        )
        
        if not position_size.is_valid:
            rejection_reasons.append(f"Position sizing: {position_size.rejection_reason}")
        
        # Apply Medallion regime-based adjustments
        original_position_value = position_size.position_value
        if medallion_analysis:
            regime = medallion_analysis['regime']
            if regime == 'HighVol':
                position_size.position_value *= 0.5  # Reduce by 50%
                logger.info(f"  📉 HighVol regime: Position reduced 50% (${original_position_value:,.2f} → ${position_size.position_value:,.2f})")
            elif regime == 'Bear':
                position_size.position_value *= 0.7  # Reduce by 30%
                logger.info(f"  📉 Bear regime: Position reduced 30% (${original_position_value:,.2f} → ${position_size.position_value:,.2f})")
            elif regime == 'Bull':
                logger.info(f"  📈 Bull regime: Position unchanged")

        # GARCH vol-based position adjustment (Phase 6)
        if garch_vol is not None and garch_vol > 0:
            if garch_vol > 0.35:
                position_size.position_value *= 0.4
                logger.info(f"  GARCH extreme vol ({garch_vol:.1%}): Position reduced 60%")
            elif garch_vol > 0.25:
                position_size.position_value *= 0.7
                logger.info(f"  GARCH high vol ({garch_vol:.1%}): Position reduced 30%")
            elif garch_vol < 0.10:
                position_size.position_value *= 1.2
                logger.info(f"  GARCH low vol ({garch_vol:.1%}): Position increased 20%")
        
        logger.info(f"  Position Value: ${position_size.position_value:,.2f}")
        logger.info(f"  Position %: {position_size.position_pct:.2%}")
        logger.info(f"  Kelly Fraction: {position_size.kelly_fraction:.2%}")
        
        # Step 7: Portfolio limit checks
        logger.info("Step 7: Portfolio limit checks...")
        # Simple portfolio limit check (can be enhanced with actual RiskManager method)
        max_position_pct = 0.05  # Max 5% per position (was 20% — way too concentrated)
        position_pct = position_size.position_value / portfolio_value if portfolio_value > 0 else 0
        
        if position_pct > max_position_pct:
            allowed = False
            reason = f"Position {position_pct:.1%} exceeds max {max_position_pct:.1%}"
            rejection_reasons.append(f"Portfolio limits: {reason}")
            logger.info(f"  Limits Check: FAIL")
            logger.info(f"  Reason: {reason}")
        else:
            allowed = True
            reason = "Within limits"
            logger.info(f"  Limits Check: PASS")
        
        # Calculate quantity (HIGH-SEVERITY FIX: round instead of truncate)
        if position_size.position_value > 0 and entry_price > 0:
            # Round to nearest share to avoid underutilization
            quantity = round(position_size.position_value / entry_price)
            
            # Ensure we don't exceed position value (in case price moved)
            actual_value = quantity * entry_price
            if actual_value > position_size.position_value * 1.01:  # 1% tolerance
                quantity -= 1  # Reduce by one share
            
            logger.debug(f"Position sizing: ${position_size.position_value:.2f} @ ${entry_price:.2f} "
                        f"= {quantity} shares (${quantity * entry_price:.2f})")
        else:
            quantity = 0
        
        # Determine signal
        signal = self._determine_signal(combined_score, mtf_score, sentiment_score)
        
        # Final decision
        is_tradeable = (
            len(rejection_reasons) == 0 and
            signal in [TradeSignal.BUY, TradeSignal.STRONG_BUY,
                       TradeSignal.SELL, TradeSignal.STRONG_SELL] and
            quantity > 0
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"DECISION: {signal.value.upper()}")
        logger.info(f"TRADEABLE: {'YES' if is_tradeable else 'NO'}")
        if rejection_reasons:
            logger.info(f"REASONS: {', '.join(rejection_reasons)}")
        logger.info(f"{'='*60}\n")
        
        # Create decision object
        decision = TradeDecision(
            symbol=symbol,
            timestamp=datetime.now(),
            signal=signal,
            mtf_score=mtf_score,
            sentiment_score=sentiment_score,
            combined_score=combined_score,
            confidence=confidence,
            recommended_position_value=position_size.position_value,
            recommended_quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profits=take_profits,
            is_tradeable=is_tradeable,
            rejection_reasons=rejection_reasons,
            metadata={
                'atr': atr,
                'mtf_analysis': mtf_analysis,
                'sentiment_result': sentiment_result,
                'position_sizing': position_size,
                'medallion_analysis': medallion_analysis,
                'capm_expected_return': capm_expected_return,
                'garch_vol': garch_vol,
                'aggregated_signal': aggregated_signal,
            }
        )
        
        return decision
    
    def _create_rejection_decision(self, symbol: str, reasons: List[str]) -> TradeDecision:
        """
        Create a rejection decision when analysis fails.
        
        Args:
            symbol: Stock symbol
            reasons: List of rejection reasons
            
        Returns:
            TradeDecision with HOLD signal
        """
        return TradeDecision(
            symbol=symbol,
            timestamp=datetime.now(),
            signal=TradeSignal.HOLD,
            mtf_score=0.0,
            sentiment_score=0.0,
            combined_score=0.0,
            confidence=0.0,
            recommended_position_value=0.0,
            recommended_quantity=0,
            entry_price=0.0,
            stop_loss=0.0,
            take_profits=[],
            is_tradeable=False,
            rejection_reasons=reasons,
            metadata={}
        )
    
    def batch_analyze(self, symbols: List[str], portfolio_value: float,
                     performance_metrics: Optional[PerformanceMetrics] = None) -> List[TradeDecision]:
        """
        Analyze multiple symbols in batch.
        
        Args:
            symbols: List of stock symbols
            portfolio_value: Current portfolio value
            performance_metrics: Historical performance metrics
            
        Returns:
            List of TradeDecision objects
        """
        logger.info(f"Batch analyzing {len(symbols)} symbols...")
        
        decisions = []
        for symbol in symbols:
            try:
                decision = self.analyze_opportunity(symbol, portfolio_value, performance_metrics)
                decisions.append(decision)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                decisions.append(self._create_rejection_decision(
                    symbol, [f"Analysis error: {str(e)}"]
                ))
        
        # Sort by combined score
        decisions.sort(key=lambda d: d.combined_score, reverse=True)
        
        return decisions

    def analyze(self, symbol: str, portfolio_value: float = 100000) -> Optional[TradeDecision]:
        """Alias for analyze_opportunity, used by run_v28_production.py."""
        try:
            return self.analyze_opportunity(symbol, portfolio_value)
        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {e}")
            return None

    def record_trade_outcome(self, symbol: str, decision: TradeDecision, pnl: float):
        """
        Post-trade feedback to ContinuousLearner and SignalAggregator.
        Called after a trade is closed with the realized P&L.
        """
        # Update ContinuousLearner directly
        if self.continuous_learner is not None:
            try:
                from src.ml.continuous_learner import TradeResult
                agg_signal_obj = decision.metadata.get("aggregated_signal")
                regime_val = "unknown"
                if agg_signal_obj is not None and hasattr(agg_signal_obj, 'regime'):
                    regime_val = agg_signal_obj.regime.value if hasattr(agg_signal_obj.regime, 'value') else str(agg_signal_obj.regime)
                result = TradeResult(
                    timestamp=datetime.now(),
                    ticker=symbol,
                    signal_direction="long" if decision.signal in (TradeSignal.BUY, TradeSignal.STRONG_BUY) else "short",
                    signal_confidence=decision.confidence,
                    predicted_return=decision.combined_score * 0.02,
                    actual_return=pnl / max(decision.recommended_position_value, 1),
                    is_hit=pnl > 0,
                    features={
                        "mtf_score": decision.mtf_score,
                        "sentiment": decision.sentiment_score,
                        "combined": decision.combined_score,
                    },
                    regime=regime_val,
                )
                learner_result = self.continuous_learner.record_trade(result)
                if learner_result.get("needs_recalibration"):
                    logger.warning(f"ContinuousLearner: recalibration triggered after {symbol} trade")
            except Exception as e:
                logger.debug(f"ContinuousLearner trade recording failed: {e}")

        # Update SignalAggregator
        agg_signal = decision.metadata.get("aggregated_signal")
        if self.signal_aggregator is not None and agg_signal is not None:
            try:
                self.signal_aggregator.update_after_trade(symbol, agg_signal, pnl)
            except Exception as e:
                logger.debug(f"SignalAggregator trade update failed: {e}")

        # Feed outcome to AdaptiveEnsemble for online learning
        if self.adaptive_ml is not None:
            try:
                self.adaptive_ml.record_outcome(
                    symbol=symbol,
                    signal=decision.combined_score * 2 - 1,  # map [0,1] → [-1,1]
                    pnl=pnl,
                )
            except Exception as e:
                logger.debug(f"AdaptiveEnsemble outcome recording failed: {e}")

    def _build_ml_features(self, symbol: str) -> Optional[np.ndarray]:
        """Build feature vector for ML models from recent price data."""
        try:
            data = yf.download(symbol, period="6mo", interval="1d", progress=False)
            if data.empty or len(data) < 60:
                return None

            close = data["Close"].values.flatten()
            volume = data["Volume"].values.flatten()

            ret_1d = np.diff(np.log(close))
            ret_5d = close[-1] / close[-5] - 1 if len(close) >= 5 else 0
            ret_20d = close[-1] / close[-20] - 1 if len(close) >= 20 else 0
            vol_20d = np.std(ret_1d[-20:]) * np.sqrt(252) if len(ret_1d) >= 20 else 0
            sma_ratio = close[-1] / np.mean(close[-20:]) if len(close) >= 20 else 1
            rsi = self._compute_rsi(close, 14)
            vol_ratio = np.mean(volume[-5:]) / np.mean(volume[-20:]) if len(volume) >= 20 else 1

            features = np.array([[
                ret_1d[-1] if len(ret_1d) > 0 else 0,
                ret_5d,
                ret_20d,
                vol_20d,
                sma_ratio,
                rsi,
                vol_ratio,
                close[-1] / np.mean(close[-50:]) if len(close) >= 50 else 1,
                np.mean(ret_1d[-5:]) if len(ret_1d) >= 5 else 0,
                np.std(ret_1d[-5:]) * np.sqrt(252) if len(ret_1d) >= 5 else 0,
            ]])
            return features
        except Exception:
            return None

    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int = 14) -> float:
        """Compute RSI."""
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss < 1e-10:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - 100.0 / (1.0 + rs))


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize engine
    engine = EnhancedTradingEngine()
    
    # Analyze a symbol
    symbol = "AMD"
    portfolio_value = 100000
    
    # Sample performance metrics
    metrics = PerformanceMetrics(
        total_trades=100,
        winning_trades=58,
        losing_trades=42,
        total_profit=14500,
        total_loss=-9200
    )
    
    # Run analysis
    decision = engine.analyze_opportunity(symbol, portfolio_value, metrics)
    
    # Display results
    print(f"\n{'='*70}")
    print(f"TRADING DECISION: {symbol}")
    print(f"{'='*70}")
    print(f"Signal: {decision.signal.value.upper()}")
    print(f"Tradeable: {'YES ✓' if decision.is_tradeable else 'NO ✗'}")
    print(f"\nScores:")
    print(f"  MTF Alignment: {decision.mtf_score:.1f}/100")
    print(f"  Sentiment: {decision.sentiment_score:+.2f}")
    print(f"  Combined: {decision.combined_score:.2f}")
    print(f"  Confidence: {decision.confidence:.1%}")
    print(f"\nPosition:")
    print(f"  Value: ${decision.recommended_position_value:,.2f}")
    print(f"  Quantity: {decision.recommended_quantity} shares")
    print(f"  Entry: ${decision.entry_price:.2f}")
    print(f"  Stop Loss: ${decision.stop_loss:.2f}")
    print(f"  Take Profits: {[f'${tp:.2f}' for tp in decision.take_profits]}")
    
    if decision.rejection_reasons:
        print(f"\nRejection Reasons:")
        for reason in decision.rejection_reasons:
            print(f"  - {reason}")
    
    print(f"{'='*70}\n")

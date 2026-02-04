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
    min_mtf_score: float = 60.0  # Minimum multi-timeframe alignment
    min_sentiment_score: float = -0.3  # Minimum sentiment (-1 to 1)
    min_combined_score: float = 0.6  # Minimum combined score for trading
    
    # Scoring weights
    mtf_weight: float = 0.6
    sentiment_weight: float = 0.4
    
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
        
        logger.info("EnhancedTradingEngine initialized with all modules (including Medallion Math)")
    
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
        # Strong signals require high alignment and sentiment
        if combined_score > 0.8 and mtf_score > 80 and sentiment_score > 0.5:
            return TradeSignal.STRONG_BUY
        elif combined_score > 0.7:
            return TradeSignal.BUY
        elif combined_score < 0.3 and mtf_score < 30 and sentiment_score < -0.5:
            return TradeSignal.STRONG_SELL
        elif combined_score < 0.4:
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
        
        # HIGH-SEVERITY FIX: Check sentiment validity
        if not sentiment_result.is_valid:
            rejection_reasons.append("Sentiment data unavailable")
        
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
        
        # Step 4: Combined scoring
        logger.info("Step 4: Combined scoring...")
        combined_score = self._calculate_combined_score(mtf_score, sentiment_score)
        confidence = (combined_score + sentiment_result.confidence) / 2
        
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
        
        logger.info(f"  Position Value: ${position_size.position_value:,.2f}")
        logger.info(f"  Position %: {position_size.position_pct:.2%}")
        logger.info(f"  Kelly Fraction: {position_size.kelly_fraction:.2%}")
        
        # Step 7: Portfolio limit checks
        logger.info("Step 7: Portfolio limit checks...")
        # Simple portfolio limit check (can be enhanced with actual RiskManager method)
        max_position_pct = 0.20  # Max 20% per position
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
            signal in [TradeSignal.BUY, TradeSignal.STRONG_BUY] and
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
                'medallion_analysis': medallion_analysis  # Medallion mathematical analysis
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

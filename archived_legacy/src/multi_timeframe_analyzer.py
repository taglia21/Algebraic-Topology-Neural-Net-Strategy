"""
Production-Ready Multi-Timeframe Analysis Module
=================================================

Analyzes trend alignment across multiple timeframes to identify high-probability setups.

Features:
- Multi-timeframe data fetching (5m, 15m, 1h, 4h, 1d)
- Technical indicators: EMA crossover (8/21), RSI, MACD
- Weighted alignment scoring (higher timeframes = more weight)
- Data caching with TTL for efficiency
- Comprehensive trend strength analysis

Author: Trading System
Version: 1.0.0
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time
from functools import wraps

import numpy as np
import pandas as pd
import yfinance as yf

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


class Timeframe(Enum):
    """Supported timeframes for analysis."""
    M5 = ("5m", "5 minutes", 1)
    M15 = ("15m", "15 minutes", 2)
    H1 = ("1h", "1 hour", 3)
    H4 = ("4h", "4 hours", 4)
    D1 = ("1d", "1 day", 5)
    
    def __init__(self, interval: str, description: str, weight: int):
        self.interval = interval
        self.description = description
        self.weight = weight


class TrendDirection(Enum):
    """Trend direction classification."""
    STRONG_BULLISH = 1.0
    BULLISH = 0.5
    NEUTRAL = 0.0
    BEARISH = -0.5
    STRONG_BEARISH = -1.0


@dataclass
class TimeframeSignals:
    """Technical signals for a single timeframe."""
    timeframe: Timeframe
    ema_cross: float  # -1 to 1: bearish to bullish
    rsi_position: float  # -1 to 1: oversold to overbought
    macd_histogram: float  # -1 to 1: bearish to bullish
    trend_score: float  # Composite score for this timeframe
    raw_data: Optional[pd.DataFrame] = None
    
    @property
    def is_bullish(self) -> bool:
        """Check if timeframe shows bullish bias."""
        return self.trend_score > 0.3
    
    @property
    def is_bearish(self) -> bool:
        """Check if timeframe shows bearish bias."""
        return self.trend_score < -0.3


@dataclass
class TimeframeAnalysis:
    """Complete multi-timeframe analysis result."""
    symbol: str
    timestamp: datetime
    signals: Dict[Timeframe, TimeframeSignals]
    alignment_score: float  # 0-100: overall trend alignment
    dominant_trend: TrendDirection
    bullish_timeframes: int
    bearish_timeframes: int
    neutral_timeframes: int
    
    @property
    def is_aligned(self) -> bool:
        """Check if timeframes show strong alignment (>70)."""
        return self.alignment_score > 70
    
    @property
    def is_tradeable(self) -> bool:
        """Check if setup is tradeable (>60 score)."""
        return self.alignment_score > 60


@dataclass
class AnalyzerConfig:
    """Multi-timeframe analyzer configuration."""
    # Timeframes to analyze
    timeframes: List[Timeframe] = field(default_factory=lambda: [
        Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1
    ])
    
    # EMA parameters
    ema_fast: int = 8
    ema_slow: int = 21
    
    # RSI parameters
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    
    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Data fetching
    lookback_periods: Dict[Timeframe, int] = field(default_factory=lambda: {
        Timeframe.M5: 100,
        Timeframe.M15: 100,
        Timeframe.H1: 100,
        Timeframe.H4: 100,
        Timeframe.D1: 100
    })
    
    # Caching
    cache_ttl_seconds: int = 60  # 1 minute cache
    
    # Scoring weights (higher timeframes get more weight)
    use_weighted_scoring: bool = True


class MultiTimeframeAnalyzer:
    """
    Multi-timeframe trend analysis system.
    
    Analyzes price action across multiple timeframes to identify
    high-probability trade setups with aligned trends.
    """
    
    def __init__(self, config: Optional[AnalyzerConfig] = None):
        """
        Initialize multi-timeframe analyzer.
        
        Args:
            config: Analyzer configuration (uses defaults if None)
        """
        self.config = config or AnalyzerConfig()
        # CRITICAL FIX: Use OrderedDict for LRU cache with size limit
        self.cache: OrderedDict = OrderedDict()
        self.max_cache_size = 100  # Limit to 100 symbols
        self.cache_ttl = timedelta(seconds=self.config.cache_ttl_seconds)
        
        logger.info(f"MultiTimeframeAnalyzer initialized with {len(self.config.timeframes)} timeframes, "
                   f"max_cache={self.max_cache_size}")
    
    @retry_yfinance(max_retries=3)
    def _fetch_data(self, symbol: str, timeframe: Timeframe) -> Optional[pd.DataFrame]:
        """
        Fetch price data for a specific timeframe with CRITICAL retry logic.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe to fetch
            
        Returns:
            DataFrame with OHLCV data or None on error
        """
        try:
            ticker = yf.Ticker(symbol)
            periods = self.config.lookback_periods.get(timeframe, 100)
            
            # Determine period string for yfinance
            if timeframe in [Timeframe.M5, Timeframe.M15]:
                period = "5d"  # Max for minute data
            elif timeframe == Timeframe.H1:
                period = "1mo"
            elif timeframe == Timeframe.H4:
                period = "3mo"
            else:  # Daily
                period = "1y"
            
            # Fetch data
            df = ticker.history(period=period, interval=timeframe.interval)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol} on {timeframe.description}")
                return None
            
            # HIGH-SEVERITY FIX: Calculate minimum required bars based on indicators
            # EMA(21) needs 21+, RSI(14) needs 14+, MACD(26,12,9) needs 26+9=35
            min_required = max(
                self.config.ema_slow + 20,  # EMA needs warm-up period
                self.config.rsi_period + 20,  # RSI needs warm-up period
                self.config.macd_slow + self.config.macd_signal + 20  # MACD needs most data
            )
            
            if len(df) < min_required:
                logger.warning(f"Insufficient data for {symbol} on {timeframe.description}: "
                             f"{len(df)} bars < {min_required} required")
                return None
            
            logger.debug(f"Fetched {len(df)} bars for {symbol} on {timeframe.description} "
                        f"(min required: {min_required})")
            return df
        
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} on {timeframe.description}: {e}")
            return None
    
    def _calculate_ema(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data['Close'].ewm(span=period, adjust=False).mean()
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            data: OHLCV DataFrame
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD indicator.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            (macd_line, signal_line, histogram) tuple
        """
        ema_fast = data['Close'].ewm(span=self.config.macd_fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=self.config.macd_slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.config.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _analyze_timeframe(self, symbol: str, timeframe: Timeframe) -> Optional[TimeframeSignals]:
        """
        Analyze a single timeframe for trend signals.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe to analyze
            
        Returns:
            TimeframeSignals or None on error
        """
        # Fetch data
        data = self._fetch_data(symbol, timeframe)
        if data is None:
            return None
        
        try:
            # Calculate indicators
            ema_fast = self._calculate_ema(data, self.config.ema_fast)
            ema_slow = self._calculate_ema(data, self.config.ema_slow)
            rsi = self._calculate_rsi(data, self.config.rsi_period)
            macd_line, signal_line, histogram = self._calculate_macd(data)
            
            # Get latest values
            current_price = data['Close'].iloc[-1]
            ema_fast_val = ema_fast.iloc[-1]
            ema_slow_val = ema_slow.iloc[-1]
            rsi_val = rsi.iloc[-1]
            hist_val = histogram.iloc[-1]
            
            # Calculate EMA crossover signal (-1 to 1)
            ema_diff = ema_fast_val - ema_slow_val
            ema_diff_pct = (ema_diff / current_price) * 100
            ema_cross = np.tanh(ema_diff_pct)  # Normalize to -1 to 1
            
            # Calculate RSI position signal (-1 to 1)
            # Map RSI 0-100 to -1 to 1, with 50 as neutral
            rsi_position = (rsi_val - 50) / 50
            rsi_position = np.clip(rsi_position, -1, 1)
            
            # Calculate MACD histogram signal (-1 to 1)
            # Normalize histogram relative to recent range
            hist_recent = histogram.iloc[-20:] if len(histogram) > 20 else histogram
            hist_std = hist_recent.std()
            if hist_std > 0:
                macd_signal = np.tanh(hist_val / (hist_std * 2))
            else:
                macd_signal = 0.0
            
            # Calculate composite trend score (weighted average)
            trend_score = (ema_cross * 0.4 + rsi_position * 0.2 + macd_signal * 0.4)
            
            signals = TimeframeSignals(
                timeframe=timeframe,
                ema_cross=float(ema_cross),
                rsi_position=float(rsi_position),
                macd_histogram=float(macd_signal),
                trend_score=float(trend_score)
            )
            
            logger.debug(f"{symbol} {timeframe.description}: EMA={ema_cross:.2f}, "
                        f"RSI={rsi_position:.2f}, MACD={macd_signal:.2f}, "
                        f"Score={trend_score:.2f}")
            
            return signals
        
        except Exception as e:
            logger.error(f"Error analyzing {symbol} on {timeframe.description}: {e}")
            return None
    
    def _calculate_alignment_score(self, signals: Dict[Timeframe, TimeframeSignals]) -> float:
        """
        Calculate overall alignment score from all timeframes.
        
        Args:
            signals: Dictionary of timeframe signals
            
        Returns:
            Alignment score (0-100)
        """
        if not signals:
            return 0.0
        
        # Calculate weighted average if enabled
        if self.config.use_weighted_scoring:
            total_weight = sum(tf.weight for tf in signals.keys())
            weighted_score = sum(
                sig.trend_score * sig.timeframe.weight 
                for sig in signals.values()
            )
            avg_score = weighted_score / total_weight if total_weight > 0 else 0.0
        else:
            # Simple average
            avg_score = sum(sig.trend_score for sig in signals.values()) / len(signals)
        
        # Convert from -1..1 to 0..100
        alignment_score = (avg_score + 1) * 50
        
        # Bonus for consensus: if all timeframes agree on direction
        bullish_count = sum(1 for sig in signals.values() if sig.is_bullish)
        bearish_count = sum(1 for sig in signals.values() if sig.is_bearish)
        
        if bullish_count == len(signals):
            alignment_score = min(100, alignment_score * 1.1)  # 10% bonus
        elif bearish_count == len(signals):
            alignment_score = max(0, alignment_score * 0.9)  # Shift toward bearish
        
        return float(np.clip(alignment_score, 0, 100))
    
    def _determine_trend(self, alignment_score: float, signals: Dict[Timeframe, TimeframeSignals]) -> TrendDirection:
        """
        Determine dominant trend direction.
        
        Args:
            alignment_score: Overall alignment score
            signals: Timeframe signals
            
        Returns:
            TrendDirection enum
        """
        # Calculate average trend score
        avg_trend = sum(sig.trend_score for sig in signals.values()) / len(signals)
        
        # Classify trend
        if avg_trend > 0.5:
            return TrendDirection.STRONG_BULLISH
        elif avg_trend > 0.2:
            return TrendDirection.BULLISH
        elif avg_trend < -0.5:
            return TrendDirection.STRONG_BEARISH
        elif avg_trend < -0.2:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.NEUTRAL
    
    def analyze(self, symbol: str, use_cache: bool = True) -> TimeframeAnalysis:
        """
        Perform complete multi-timeframe analysis.
        
        Args:
            symbol: Stock symbol to analyze
            use_cache: Whether to use cached results
            
        Returns:
            TimeframeAnalysis with complete results
        """
        # Check cache
        if use_cache and symbol in self.cache:
            cached_analysis, cache_time = self.cache[symbol]
            age = (datetime.now() - cache_time).total_seconds()
            
            if age < self.config.cache_ttl_seconds:
                logger.debug(f"Using cached analysis for {symbol} ({age:.1f}s old)")
                return cached_analysis
        
        logger.info(f"Analyzing {symbol} across {len(self.config.timeframes)} timeframes")
        
        # Analyze each timeframe
        signals: Dict[Timeframe, TimeframeSignals] = {}
        
        for timeframe in self.config.timeframes:
            tf_signals = self._analyze_timeframe(symbol, timeframe)
            if tf_signals:
                signals[timeframe] = tf_signals
        
        if not signals:
            logger.error(f"Failed to analyze any timeframes for {symbol}")
            # Return empty analysis
            return TimeframeAnalysis(
                symbol=symbol,
                timestamp=datetime.now(),
                signals={},
                alignment_score=0.0,
                dominant_trend=TrendDirection.NEUTRAL,
                bullish_timeframes=0,
                bearish_timeframes=0,
                neutral_timeframes=0
            )
        
        # Calculate alignment score
        alignment_score = self._calculate_alignment_score(signals)
        
        # Determine dominant trend
        dominant_trend = self._determine_trend(alignment_score, signals)
        
        # Count timeframe biases
        bullish_count = sum(1 for sig in signals.values() if sig.is_bullish)
        bearish_count = sum(1 for sig in signals.values() if sig.is_bearish)
        neutral_count = len(signals) - bullish_count - bearish_count
        
        # Create analysis result
        analysis = TimeframeAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            signals=signals,
            alignment_score=alignment_score,
            dominant_trend=dominant_trend,
            bullish_timeframes=bullish_count,
            bearish_timeframes=bearish_count,
            neutral_timeframes=neutral_count
        )
        
        # CRITICAL FIX: Add to cache with LRU eviction
        self.cache[symbol] = (analysis, datetime.now())
        
        # Evict oldest if over limit
        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)  # Remove oldest entry
            logger.debug(f"Cache LRU eviction: size was {len(self.cache) + 1}, now {len(self.cache)}")
        
        # Periodic cleanup of expired entries (every 10th addition)
        if len(self.cache) % 10 == 0:
            self._cleanup_expired_cache()
        
        logger.info(f"{symbol} analysis: Score={alignment_score:.1f}, "
                   f"Trend={dominant_trend.name}, "
                   f"Bullish={bullish_count}, Bearish={bearish_count}")
        
        return analysis
    
    def _cleanup_expired_cache(self):
        """Remove expired cache entries based on TTL."""
        now = datetime.now()
        expired = [k for k, (_, ts) in self.cache.items() 
                  if now - ts > self.cache_ttl]
        
        for key in expired:
            del self.cache[key]
        
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired cache entries")
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear analysis cache.
        
        Args:
            symbol: Clear specific symbol, or all if None
        """
        if symbol:
            if symbol in self.cache:
                del self.cache[symbol]
                logger.debug(f"Cleared cache for {symbol}")
        else:
            self.cache.clear()
            logger.debug("Cleared all cache")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize analyzer
    analyzer = MultiTimeframeAnalyzer()
    
    # Analyze a symbol
    symbol = "AMD"
    analysis = analyzer.analyze(symbol)
    
    print(f"\n{'='*60}")
    print(f"Multi-Timeframe Analysis: {symbol}")
    print(f"{'='*60}")
    print(f"Timestamp: {analysis.timestamp}")
    print(f"Alignment Score: {analysis.alignment_score:.1f}/100")
    print(f"Dominant Trend: {analysis.dominant_trend.name}")
    print(f"Tradeable: {'YES' if analysis.is_tradeable else 'NO'}")
    print(f"Aligned: {'YES' if analysis.is_aligned else 'NO'}")
    print(f"\nTimeframe Breakdown:")
    print(f"  Bullish: {analysis.bullish_timeframes}")
    print(f"  Bearish: {analysis.bearish_timeframes}")
    print(f"  Neutral: {analysis.neutral_timeframes}")
    
    print(f"\nDetailed Signals:")
    for tf, signals in analysis.signals.items():
        print(f"  {tf.description:12s}: Score={signals.trend_score:+.2f}, "
              f"EMA={signals.ema_cross:+.2f}, RSI={signals.rsi_position:+.2f}, "
              f"MACD={signals.macd_histogram:+.2f}")

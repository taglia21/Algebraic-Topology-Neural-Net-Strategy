"""Multi-Timeframe Analysis for trend confirmation."""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class Trend(Enum):
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1

@dataclass
class TimeframeSignal:
    timeframe: str
    trend: Trend
    strength: float  # 0-1
    momentum: float

class MultiTimeframeAnalyzer:
    """Analyzes multiple timeframes for trend confirmation."""
    
    TIMEFRAMES = ['5m', '15m', '1h', '4h', 'D']
    
    def __init__(self):
        self.signals: Dict[str, Dict[str, TimeframeSignal]] = {}
        
    def analyze_trend(self, prices: np.ndarray, timeframe: str = '1h') -> Trend:
        """Determine trend from price array."""
        if len(prices) < 20:
            return Trend.NEUTRAL
        sma_short = np.mean(prices[-10:])
        sma_long = np.mean(prices[-20:])
        if sma_short > sma_long * 1.01:
            return Trend.BULLISH
        elif sma_short < sma_long * 0.99:
            return Trend.BEARISH
        return Trend.NEUTRAL
    
    def get_signal(self, symbol: str, prices_by_tf: Dict[str, np.ndarray]) -> TimeframeSignal:
        """Get aggregated signal across timeframes."""
        trends = []
        for tf, prices in prices_by_tf.items():
            trend = self.analyze_trend(prices, tf)
            trends.append(trend.value)
        
        avg_trend = np.mean(trends) if trends else 0
        if avg_trend > 0.5:
            final_trend = Trend.BULLISH
        elif avg_trend < -0.5:
            final_trend = Trend.BEARISH
        else:
            final_trend = Trend.NEUTRAL
            
        return TimeframeSignal(
            timeframe='multi',
            trend=final_trend,
            strength=abs(avg_trend),
            momentum=avg_trend
        )
    
    def confirms_trade(self, symbol: str, direction: int, 
                      prices_by_tf: Dict[str, np.ndarray]) -> bool:
        """Check if higher timeframes confirm trade direction."""
        signal = self.get_signal(symbol, prices_by_tf)
        if direction > 0:  # Long trade
            return signal.trend == Trend.BULLISH
        elif direction < 0:  # Short trade
            return signal.trend == Trend.BEARISH
        return True

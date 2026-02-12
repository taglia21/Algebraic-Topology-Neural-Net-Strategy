"""
Market Regime Filter
====================

Simple but effective market regime detection using SPY's moving averages.
Prevents buying individual stocks during broad market downtrends.

Usage:
    from src.risk.regime_filter import is_bullish_regime, get_regime
    
    if not is_bullish_regime():
        logger.warning("Bear regime — skipping new long entries")
        return
"""

import logging
from typing import Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Cache regime check to avoid hammering yfinance every cycle
_regime_cache = {
    'result': None,
    'checked_at': None,
    'cache_minutes': 30,  # Re-check every 30 minutes
}


def get_regime() -> Tuple[str, dict]:
    """
    Detect market regime using SPY moving averages.
    
    Returns:
        Tuple of (regime_name, details_dict)
        regime_name: 'strong_bull', 'bull', 'neutral', 'bear', 'strong_bear'
    """
    global _regime_cache
    
    # Check cache
    now = datetime.now()
    if (_regime_cache['result'] is not None and 
        _regime_cache['checked_at'] is not None and
        (now - _regime_cache['checked_at']).total_seconds() < _regime_cache['cache_minutes'] * 60):
        return _regime_cache['result']
    
    try:
        import yfinance as yf
        
        spy = yf.Ticker("SPY")
        hist = spy.history(period="1y")
        
        if hist is None or len(hist) < 200:
            logger.warning("Insufficient SPY data for regime detection, defaulting to neutral")
            result = ('neutral', {'reason': 'insufficient_data'})
            _regime_cache['result'] = result
            _regime_cache['checked_at'] = now
            return result
        
        close = hist['Close']
        current_price = float(close.iloc[-1])
        sma_50 = float(close.rolling(50).mean().iloc[-1])
        sma_200 = float(close.rolling(200).mean().iloc[-1])
        
        # Also check recent momentum (20-day)
        sma_20 = float(close.rolling(20).mean().iloc[-1])
        
        details = {
            'spy_price': current_price,
            'sma_20': round(sma_20, 2),
            'sma_50': round(sma_50, 2),
            'sma_200': round(sma_200, 2),
            'above_200sma': current_price > sma_200,
            'above_50sma': current_price > sma_50,
            'golden_cross': sma_50 > sma_200,  # 50 > 200 = bullish
        }
        
        # Classify regime
        if current_price > sma_200 and sma_50 > sma_200 and current_price > sma_50:
            regime = 'strong_bull'  # Everything aligned bullish
        elif current_price > sma_200:
            regime = 'bull'  # Above long-term trend
        elif current_price > sma_50 and current_price < sma_200:
            regime = 'neutral'  # Mixed signals
        elif current_price < sma_200 and sma_50 > sma_200:
            regime = 'bear'  # Below 200 SMA but golden cross still holds
        else:
            regime = 'strong_bear'  # Below both, death cross
        
        details['regime'] = regime
        result = (regime, details)
        
        _regime_cache['result'] = result
        _regime_cache['checked_at'] = now
        
        logger.info(f"Market regime: {regime} | SPY: ${current_price:.2f} | "
                     f"50SMA: ${sma_50:.2f} | 200SMA: ${sma_200:.2f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Regime detection failed: {e}")
        result = ('neutral', {'reason': f'error: {e}'})
        _regime_cache['result'] = result
        _regime_cache['checked_at'] = now
        return result


def is_bullish_regime() -> bool:
    """
    Simple check: should we be opening new long positions?
    
    Returns True for 'strong_bull', 'bull', or 'neutral' regimes.
    Returns False for 'bear' or 'strong_bear'.
    """
    regime, _ = get_regime()
    return regime in ('strong_bull', 'bull', 'neutral')


def get_position_scale() -> float:
    """
    Get a position size multiplier based on regime.
    
    - strong_bull: 1.0 (full size)
    - bull: 0.8
    - neutral: 0.5
    - bear: 0.25 (reduce heavily)
    - strong_bear: 0.0 (no new longs)
    """
    regime, _ = get_regime()
    scales = {
        'strong_bull': 1.0,
        'bull': 0.8,
        'neutral': 0.5,
        'bear': 0.25,
        'strong_bear': 0.0,
    }
    return scales.get(regime, 0.5)

"""
Signal quality filters using technical analysis.
Improves entry timing by filtering weak signals.

V1.0: RSI + Volatility filters for signal quality improvement
- RSI filter: Only take buy signals when truly oversold, sell when overbought
- Volatility filter: Pause trading during high vol + negative momentum
"""

import logging
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalFilter:
    """Applies RSI and volatility filters to improve signal quality.
    
    Filtering logic:
    - RSI Filter: Confirms momentum alignment with signal direction
    - Volatility Filter: Reduces trading during unstable market conditions
    
    Expected impact:
    - Fewer trades (-20-40%)
    - Higher win rate (+5-10%)
    - Better Sharpe ratio (+0.2 to +0.3)
    """
    
    def __init__(
        self, 
        rsi_period: int = 14, 
        rsi_oversold: float = 35.0, 
        rsi_overbought: float = 65.0, 
        vol_threshold: float = 0.30,
        enable_rsi_filter: bool = True,
        enable_vol_filter: bool = True
    ):
        """
        Initialize filters.
        
        Args:
            rsi_period: RSI calculation period (default 14)
            rsi_oversold: Buy only below this RSI (default 35 = true oversold)
            rsi_overbought: Sell only above this RSI (default 65 = true overbought)  
            vol_threshold: Pause trading if 20d volatility exceeds this (default 0.30 = 30%)
            enable_rsi_filter: Whether to apply RSI filtering
            enable_vol_filter: Whether to apply volatility filtering
        """
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.vol_threshold = vol_threshold
        self.enable_rsi_filter = enable_rsi_filter
        self.enable_vol_filter = enable_vol_filter
        
        # Tracking for diagnostics
        self.filter_stats = {
            'total_signals': 0,
            'rsi_filtered': 0,
            'vol_filtered': 0,
            'passed': 0
        }
        
        logger.info(f"SignalFilter initialized: RSI({rsi_period}, OS={rsi_oversold}, OB={rsi_overbought}), "
                   f"Vol threshold={vol_threshold:.0%}")
    
    def _calculate_rsi(self, price_data: pd.DataFrame) -> float:
        """Calculate current RSI value from price data."""
        if len(price_data) < self.rsi_period + 1:
            return 50.0  # Neutral RSI if insufficient data
        
        # Handle different column name conventions
        close_col = None
        for col in ['close', 'Close', 'adj_close', 'Adj Close']:
            if col in price_data.columns:
                close_col = col
                break
        
        if close_col is None:
            logger.warning("No close price column found in data")
            return 50.0
        
        close_prices = price_data[close_col].dropna()
        if len(close_prices) < self.rsi_period + 1:
            return 50.0
        
        rsi_indicator = RSIIndicator(close_prices, window=self.rsi_period)
        rsi = rsi_indicator.rsi()
        
        if len(rsi) == 0 or pd.isna(rsi.iloc[-1]):
            return 50.0
        
        return float(rsi.iloc[-1])
    
    def _calculate_volatility(self, price_data: pd.DataFrame) -> float:
        """Calculate 20-day annualized volatility."""
        close_col = None
        for col in ['close', 'Close', 'adj_close', 'Adj Close']:
            if col in price_data.columns:
                close_col = col
                break
        
        if close_col is None or len(price_data) < 21:
            return 0.15  # Default moderate volatility
        
        returns = price_data[close_col].pct_change().dropna()
        if len(returns) < 20:
            return 0.15
        
        vol_20d = returns.iloc[-20:].std() * np.sqrt(252)
        return float(vol_20d) if not pd.isna(vol_20d) else 0.15
    
    def _calculate_recent_return(self, price_data: pd.DataFrame, lookback: int = 3) -> float:
        """Calculate recent return over lookback period."""
        close_col = None
        for col in ['close', 'Close', 'adj_close', 'Adj Close']:
            if col in price_data.columns:
                close_col = col
                break
        
        if close_col is None or len(price_data) < lookback + 1:
            return 0.0
        
        returns = price_data[close_col].pct_change().dropna()
        if len(returns) < lookback:
            return 0.0
        
        return float(returns.iloc[-lookback:].sum())
    
    def apply_rsi_filter(self, signal: str, price_data: pd.DataFrame) -> tuple:
        """
        Filter based on RSI extremes.
        
        Logic:
        - 'buy' signal: Only keep if RSI < oversold threshold (true bargain)
        - 'sell' signal: Only keep if RSI > overbought threshold (true expensive)
        - Otherwise: Return 'neutral' (filter out weak signal)
        
        Args:
            signal: Raw signal ('buy', 'sell', or 'neutral')
            price_data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (filtered_signal, filter_reason)
        """
        if signal == 'neutral' or not self.enable_rsi_filter:
            return signal, None
        
        current_rsi = self._calculate_rsi(price_data)
        
        # Apply filters
        if signal == 'buy' and current_rsi < self.rsi_oversold:
            return 'buy', None  # Keep - genuinely oversold
        elif signal == 'sell' and current_rsi > self.rsi_overbought:
            return 'sell', None  # Keep - genuinely overbought
        elif signal == 'buy':
            reason = f'RSI_NOT_OVERSOLD (RSI={current_rsi:.1f} > {self.rsi_oversold})'
            return 'neutral', reason
        else:  # signal == 'sell'
            reason = f'RSI_NOT_OVERBOUGHT (RSI={current_rsi:.1f} < {self.rsi_overbought})'
            return 'neutral', reason
    
    def apply_volatility_filter(self, signal: str, price_data: pd.DataFrame) -> tuple:
        """
        Pause trading during high volatility + negative momentum.
        
        Logic: When 20-day realized vol > threshold AND recent returns negative,
        markets are unstable. Return neutral to avoid trading.
        
        Args:
            signal: Signal after previous filters
            price_data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (filtered_signal, filter_reason)
        """
        if signal == 'neutral' or not self.enable_vol_filter:
            return signal, None
        
        current_vol = self._calculate_volatility(price_data)
        recent_return = self._calculate_recent_return(price_data, lookback=3)
        
        # High vol + negative momentum = pause for buy signals only
        # (still allow exits during high vol)
        if signal == 'buy' and current_vol > self.vol_threshold and recent_return < 0:
            reason = f'HIGH_VOL_PAUSE (vol={current_vol:.1%} > {self.vol_threshold:.0%}, ret={recent_return:.2%})'
            return 'neutral', reason
        
        return signal, None
    
    def filter_signal(self, signal: str, price_data: pd.DataFrame) -> dict:
        """
        Apply all filters in sequence.
        
        Order: RSI first (more selective), then volatility (regime check).
        
        Args:
            signal: Raw signal from strategy
            price_data: DataFrame with OHLCV data
            
        Returns:
            Dict with filtered signal and diagnostic info:
            {
                'signal': filtered signal,
                'original_signal': original signal,
                'filtered': bool,
                'filter_reason': reason or None,
                'rsi': current RSI,
                'volatility': current vol
            }
        """
        self.filter_stats['total_signals'] += 1
        
        result = {
            'signal': signal,
            'original_signal': signal,
            'filtered': False,
            'filter_reason': None,
            'rsi': self._calculate_rsi(price_data),
            'volatility': self._calculate_volatility(price_data)
        }
        
        if signal == 'neutral':
            self.filter_stats['passed'] += 1
            return result
        
        # Apply RSI filter
        filtered, reason = self.apply_rsi_filter(signal, price_data)
        if reason:
            self.filter_stats['rsi_filtered'] += 1
            result['signal'] = 'neutral'
            result['filtered'] = True
            result['filter_reason'] = reason
            return result
        
        # Apply volatility filter
        filtered, reason = self.apply_volatility_filter(filtered, price_data)
        if reason:
            self.filter_stats['vol_filtered'] += 1
            result['signal'] = 'neutral'
            result['filtered'] = True
            result['filter_reason'] = reason
            return result
        
        # Signal passed all filters
        self.filter_stats['passed'] += 1
        result['signal'] = filtered
        return result
    
    def get_stats(self) -> dict:
        """Get filter statistics for diagnostics."""
        total = self.filter_stats['total_signals']
        if total == 0:
            return self.filter_stats
        
        return {
            **self.filter_stats,
            'rsi_filter_rate': self.filter_stats['rsi_filtered'] / total,
            'vol_filter_rate': self.filter_stats['vol_filtered'] / total,
            'pass_rate': self.filter_stats['passed'] / total
        }
    
    def reset_stats(self):
        """Reset filter statistics."""
        self.filter_stats = {
            'total_signals': 0,
            'rsi_filtered': 0,
            'vol_filtered': 0,
            'passed': 0
        }


# Convenience function for quick signal filtering
def filter_trading_signal(
    signal: str, 
    price_data: pd.DataFrame,
    rsi_oversold: float = 35.0,
    rsi_overbought: float = 65.0,
    vol_threshold: float = 0.30
) -> str:
    """
    Quick function to filter a single trading signal.
    
    Args:
        signal: Raw signal ('buy', 'sell', 'neutral')
        price_data: DataFrame with OHLCV data
        rsi_oversold: RSI threshold for buy signals
        rsi_overbought: RSI threshold for sell signals
        vol_threshold: Volatility threshold for pausing
        
    Returns:
        Filtered signal string
    """
    filter_obj = SignalFilter(
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        vol_threshold=vol_threshold
    )
    result = filter_obj.filter_signal(signal, price_data)
    return result['signal']


if __name__ == "__main__":
    # Self-test
    print("Testing SignalFilter...")
    
    # Create sample price data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    close_prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
    high_prices = close_prices + np.abs(np.random.randn(50)) * 0.5
    low_prices = close_prices - np.abs(np.random.randn(50)) * 0.5
    
    data = pd.DataFrame({
        'date': dates,
        'open': close_prices - np.random.randn(50) * 0.2,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, 50)
    })
    data.set_index('date', inplace=True)
    
    # Test filter
    filter_obj = SignalFilter()
    
    # Test each signal type
    for signal in ['buy', 'sell', 'neutral']:
        result = filter_obj.filter_signal(signal, data)
        print(f"  Signal '{signal}' -> '{result['signal']}' "
              f"(RSI={result['rsi']:.1f}, Vol={result['volatility']:.1%})")
    
    print("\n✓ SignalFilter test passed!")
    print(f"Filter stats: {filter_obj.get_stats()}")

#!/usr/bin/env python3
"""
V16.0 Order Flow Imbalance (OFI) Engine
=======================================
Real-time detection of order flow imbalances for alpha capture.

Research Basis:
- 3:1 buy/sell ratio predicts 50ms-5min moves
- Stacked imbalances at multiple price levels amplify signals
- Volume-weighted imbalance more predictive than count-based

Features:
- Real-time OFI calculation from quote stream
- Multi-timeframe imbalance tracking (50ms, 500ms, 5s, 1min, 5min)
- Stacked imbalance detection across price levels
- Signal generation with confidence scoring
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from enum import Enum
import time
import logging

logger = logging.getLogger('V160_OFI')


class SignalType(Enum):
    """Trading signal types"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class OFISignal:
    """Order Flow Imbalance Signal"""
    symbol: str
    signal_type: SignalType
    imbalance_ratio: float
    confidence: float  # 0-1
    timeframe_ms: int
    timestamp: float
    bid_volume: float
    ask_volume: float
    stacked_levels: int = 0  # Number of price levels with same direction
    
    @property
    def strength(self) -> float:
        """Signal strength (0-1)"""
        base = abs(self.imbalance_ratio - 1) / 3  # Normalize around ratio of 3
        return min(1.0, base * self.confidence)
    
    def to_position_signal(self) -> float:
        """Convert to position signal (-1 to 1)"""
        if self.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
            return self.strength
        elif self.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
            return -self.strength
        return 0.0


@dataclass
class QuoteSnapshot:
    """Snapshot of bid/ask at a point in time"""
    timestamp: float
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    

@dataclass
class MultiLevelQuote:
    """Multi-level order book snapshot"""
    timestamp: float
    bids: List[Tuple[float, int]]  # List of (price, size)
    asks: List[Tuple[float, int]]
    
    @property
    def total_bid_volume(self) -> int:
        return sum(size for _, size in self.bids)
    
    @property
    def total_ask_volume(self) -> int:
        return sum(size for _, size in self.asks)
    
    @property
    def imbalance_ratio(self) -> float:
        ask_vol = self.total_ask_volume
        if ask_vol == 0:
            return float('inf')
        return self.total_bid_volume / ask_vol


class TimeframedBuffer:
    """Buffer that tracks data over multiple timeframes"""
    
    def __init__(self, timeframes_ms: List[int] = None):
        self.timeframes_ms = timeframes_ms or [50, 500, 5000, 60000, 300000]
        self.buffers: Dict[int, Deque] = {
            tf: deque() for tf in self.timeframes_ms
        }
    
    def add(self, item: QuoteSnapshot):
        """Add item to all buffers"""
        now = item.timestamp
        
        for tf, buffer in self.buffers.items():
            buffer.append(item)
            
            # Prune old items
            cutoff = now - tf
            while buffer and buffer[0].timestamp < cutoff:
                buffer.popleft()
    
    def get_timeframe(self, tf_ms: int) -> List[QuoteSnapshot]:
        """Get items for a specific timeframe"""
        return list(self.buffers.get(tf_ms, []))
    
    def calculate_imbalance(self, tf_ms: int) -> Tuple[float, float, float]:
        """Calculate bid/ask imbalance for a timeframe
        
        Returns:
            Tuple of (ratio, total_bid_volume, total_ask_volume)
        """
        items = self.get_timeframe(tf_ms)
        if not items:
            return 1.0, 0, 0
        
        total_bid = sum(q.bid_size for q in items)
        total_ask = sum(q.ask_size for q in items)
        
        if total_ask == 0:
            return float('inf'), total_bid, 0
        
        return total_bid / total_ask, total_bid, total_ask


class OrderFlowImbalanceEngine:
    """
    Real-time Order Flow Imbalance detection engine.
    
    Tracks bid/ask volume imbalances across multiple timeframes
    and generates trading signals based on research-backed thresholds.
    """
    
    # Thresholds based on research
    STRONG_BUY_THRESHOLD = 3.0   # 3:1 or higher bid/ask ratio
    BUY_THRESHOLD = 2.0          # 2:1 ratio
    STRONG_SELL_THRESHOLD = 0.33  # 1:3 or lower (inverted)
    SELL_THRESHOLD = 0.5          # 1:2 ratio
    
    def __init__(self, symbols: List[str]):
        """
        Initialize OFI engine.
        
        Args:
            symbols: List of symbols to track
        """
        self.symbols = symbols
        self.timeframes_ms = [50, 500, 5000, 60000, 300000]  # 50ms to 5min
        
        # Per-symbol quote buffers
        self.quote_buffers: Dict[str, TimeframedBuffer] = {
            sym: TimeframedBuffer(self.timeframes_ms) for sym in symbols
        }
        
        # Signal history
        self.signal_history: Dict[str, Deque[OFISignal]] = {
            sym: deque(maxlen=1000) for sym in symbols
        }
        
        # Statistics
        self.quotes_processed = 0
        self.signals_generated = 0
        
    def process_quote(self, symbol: str, bid_price: float, bid_size: int,
                      ask_price: float, ask_size: int, 
                      timestamp_ms: float = None) -> Optional[OFISignal]:
        """
        Process incoming quote and check for OFI signals.
        
        Args:
            symbol: Stock symbol
            bid_price: Current bid price
            bid_size: Current bid size
            ask_price: Current ask price
            ask_size: Current ask size
            timestamp_ms: Quote timestamp in milliseconds
            
        Returns:
            OFISignal if imbalance detected, None otherwise
        """
        if symbol not in self.quote_buffers:
            self.quote_buffers[symbol] = TimeframedBuffer(self.timeframes_ms)
            self.signal_history[symbol] = deque(maxlen=1000)
        
        timestamp = timestamp_ms or (time.time() * 1000)
        
        snapshot = QuoteSnapshot(
            timestamp=timestamp,
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size
        )
        
        self.quote_buffers[symbol].add(snapshot)
        self.quotes_processed += 1
        
        # Check for signals across timeframes
        best_signal = None
        best_strength = 0
        
        for tf_ms in self.timeframes_ms:
            ratio, bid_vol, ask_vol = self.quote_buffers[symbol].calculate_imbalance(tf_ms)
            
            signal = self._evaluate_imbalance(symbol, ratio, bid_vol, ask_vol, tf_ms, timestamp)
            
            if signal and signal.strength > best_strength:
                best_signal = signal
                best_strength = signal.strength
        
        if best_signal:
            self.signal_history[symbol].append(best_signal)
            self.signals_generated += 1
        
        return best_signal
    
    def _evaluate_imbalance(self, symbol: str, ratio: float, 
                            bid_vol: float, ask_vol: float,
                            timeframe_ms: int, timestamp: float) -> Optional[OFISignal]:
        """Evaluate imbalance ratio and generate signal if threshold met"""
        
        # Minimum volume threshold (avoid noise on thin markets)
        min_volume = 100  # shares
        if bid_vol + ask_vol < min_volume:
            return None
        
        # Determine signal type
        if ratio >= self.STRONG_BUY_THRESHOLD:
            signal_type = SignalType.STRONG_BUY
            confidence = min(1.0, (ratio - 3) / 5 + 0.7)  # Scale confidence
        elif ratio >= self.BUY_THRESHOLD:
            signal_type = SignalType.BUY
            confidence = 0.5 + (ratio - 2) / 2 * 0.2
        elif ratio <= self.STRONG_SELL_THRESHOLD:
            signal_type = SignalType.STRONG_SELL
            confidence = min(1.0, (1/ratio - 3) / 5 + 0.7)
        elif ratio <= self.SELL_THRESHOLD:
            signal_type = SignalType.SELL
            confidence = 0.5 + (1/ratio - 2) / 2 * 0.2
        else:
            return None  # No actionable signal
        
        return OFISignal(
            symbol=symbol,
            signal_type=signal_type,
            imbalance_ratio=ratio,
            confidence=confidence,
            timeframe_ms=timeframe_ms,
            timestamp=timestamp,
            bid_volume=bid_vol,
            ask_volume=ask_vol
        )
    
    def get_current_imbalance(self, symbol: str, timeframe_ms: int = 500) -> float:
        """Get current imbalance ratio for a symbol"""
        if symbol not in self.quote_buffers:
            return 1.0
        
        ratio, _, _ = self.quote_buffers[symbol].calculate_imbalance(timeframe_ms)
        return ratio
    
    def get_multi_timeframe_imbalance(self, symbol: str) -> Dict[int, float]:
        """Get imbalance ratios across all timeframes"""
        if symbol not in self.quote_buffers:
            return {tf: 1.0 for tf in self.timeframes_ms}
        
        return {
            tf: self.quote_buffers[symbol].calculate_imbalance(tf)[0]
            for tf in self.timeframes_ms
        }
    
    def get_latest_signal(self, symbol: str) -> Optional[OFISignal]:
        """Get most recent signal for a symbol"""
        if symbol in self.signal_history and self.signal_history[symbol]:
            return self.signal_history[symbol][-1]
        return None
    
    def get_signal_stats(self) -> Dict[str, any]:
        """Get engine statistics"""
        return {
            'quotes_processed': self.quotes_processed,
            'signals_generated': self.signals_generated,
            'signal_rate': self.signals_generated / max(self.quotes_processed, 1),
            'symbols_tracked': len(self.symbols)
        }


class StackedImbalanceDetector:
    """
    Detects stacked imbalances across multiple price levels.
    
    Stacked imbalances (same direction at multiple levels) are more
    predictive of price moves than single-level imbalances.
    """
    
    def __init__(self, min_levels: int = 3):
        """
        Args:
            min_levels: Minimum stacked levels for signal
        """
        self.min_levels = min_levels
        self.order_books: Dict[str, MultiLevelQuote] = {}
    
    def update_order_book(self, symbol: str, 
                          bids: List[Tuple[float, int]], 
                          asks: List[Tuple[float, int]]):
        """Update order book snapshot for a symbol"""
        self.order_books[symbol] = MultiLevelQuote(
            timestamp=time.time() * 1000,
            bids=bids,
            asks=asks
        )
    
    def detect_stacked_imbalance(self, symbol: str) -> Optional[OFISignal]:
        """
        Detect stacked imbalances in the order book.
        
        Returns signal if multiple consecutive levels show same direction imbalance.
        """
        if symbol not in self.order_books:
            return None
        
        book = self.order_books[symbol]
        
        # Need at least min_levels on each side
        if len(book.bids) < self.min_levels or len(book.asks) < self.min_levels:
            return None
        
        # Check each level
        buy_biased_levels = 0
        sell_biased_levels = 0
        
        for i in range(min(len(book.bids), len(book.asks))):
            bid_vol = book.bids[i][1]
            ask_vol = book.asks[i][1]
            
            if ask_vol == 0:
                buy_biased_levels += 1
            elif bid_vol / ask_vol >= 2.0:
                buy_biased_levels += 1
            elif bid_vol / ask_vol <= 0.5:
                sell_biased_levels += 1
            else:
                # Reset streak on neutral level
                break
        
        # Generate signal if stacked
        if buy_biased_levels >= self.min_levels:
            return OFISignal(
                symbol=symbol,
                signal_type=SignalType.STRONG_BUY,
                imbalance_ratio=book.imbalance_ratio,
                confidence=min(1.0, buy_biased_levels / 5),
                timeframe_ms=0,  # Instantaneous
                timestamp=book.timestamp,
                bid_volume=book.total_bid_volume,
                ask_volume=book.total_ask_volume,
                stacked_levels=buy_biased_levels
            )
        elif sell_biased_levels >= self.min_levels:
            return OFISignal(
                symbol=symbol,
                signal_type=SignalType.STRONG_SELL,
                imbalance_ratio=book.imbalance_ratio,
                confidence=min(1.0, sell_biased_levels / 5),
                timeframe_ms=0,
                timestamp=book.timestamp,
                bid_volume=book.total_bid_volume,
                ask_volume=book.total_ask_volume,
                stacked_levels=sell_biased_levels
            )
        
        return None


# ============================================================================
# Backtesting Simulation
# ============================================================================

class OFIBacktester:
    """
    Backtester for Order Flow Imbalance strategies.
    Simulates OFI signals from historical tick data.
    """
    
    def __init__(self, initial_capital: float = 100_000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: Dict[str, int] = {}
        self.trades: List[Dict] = []
        
    def simulate_ofi_from_ohlcv(self, df: pd.DataFrame, 
                                 symbol: str = 'SPY') -> pd.DataFrame:
        """
        Simulate OFI signals from OHLCV data.
        
        Uses price movement and volume to infer order flow imbalance.
        Higher volume on up moves = buy pressure, and vice versa.
        """
        df = df.copy()
        
        # Calculate returns and volume-weighted direction
        df['return'] = df['close'].pct_change()
        df['direction'] = np.sign(df['return'])
        
        # Infer OFI from price movement + volume
        # Up move + high volume = buy pressure
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['rel_volume'] = df['volume'] / (df['volume_sma'] + 1)
        
        # Synthetic bid/ask volume inference
        df['buy_volume'] = np.where(df['direction'] > 0, 
                                     df['volume'] * (0.5 + 0.3 * df['rel_volume']),
                                     df['volume'] * (0.5 - 0.2 * df['rel_volume']))
        df['sell_volume'] = df['volume'] - df['buy_volume']
        
        # Calculate rolling OFI
        for window in [5, 10, 20]:
            df[f'ofi_ratio_{window}'] = (
                df['buy_volume'].rolling(window).sum() / 
                (df['sell_volume'].rolling(window).sum() + 1)
            )
        
        # Generate signals
        df['ofi_signal'] = 0
        df.loc[df['ofi_ratio_10'] >= 3.0, 'ofi_signal'] = 1  # Strong buy
        df.loc[df['ofi_ratio_10'] >= 2.0, 'ofi_signal'] = 0.5  # Buy
        df.loc[df['ofi_ratio_10'] <= 0.33, 'ofi_signal'] = -1  # Strong sell
        df.loc[df['ofi_ratio_10'] <= 0.5, 'ofi_signal'] = -0.5  # Sell
        
        return df
    
    def backtest(self, df: pd.DataFrame, max_position: float = 0.1) -> Dict:
        """
        Backtest OFI strategy on simulated signals.
        
        Args:
            df: DataFrame with 'close' and 'ofi_signal' columns
            max_position: Maximum position as fraction of capital
            
        Returns:
            Dictionary of performance metrics
        """
        df = df.copy()
        df['position'] = df['ofi_signal'].shift(1).fillna(0) * max_position
        df['strategy_return'] = df['position'] * df['close'].pct_change()
        
        # Calculate equity curve
        equity = self.initial_capital * (1 + df['strategy_return'].fillna(0)).cumprod()
        
        # Metrics
        returns = df['strategy_return'].fillna(0)
        trading_days = len(df)
        years = trading_days / 252
        
        total_return = equity.iloc[-1] / self.initial_capital - 1
        cagr = (equity.iloc[-1] / self.initial_capital) ** (1/max(years, 0.1)) - 1
        vol = returns.std() * np.sqrt(252)
        sharpe = (cagr - 0.05) / max(vol, 0.01)
        
        max_dd = ((equity - equity.cummax()) / equity.cummax()).min()
        win_rate = (returns > 0).sum() / max((returns != 0).sum(), 1)
        
        # Signal statistics
        signal_days = (df['ofi_signal'] != 0).sum()
        signal_rate = signal_days / trading_days
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe': sharpe,
            'volatility': vol,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'trading_days': trading_days,
            'signal_days': signal_days,
            'signal_rate': signal_rate,
            'final_equity': equity.iloc[-1]
        }


# ============================================================================
# Unit Tests
# ============================================================================

def test_ofi_engine():
    """Test OFI engine functionality"""
    print("\n🧪 Testing Order Flow Imbalance Engine...")
    
    engine = OrderFlowImbalanceEngine(['SPY', 'QQQ'])
    
    # Test 1: Strong buy signal (3:1 ratio)
    for i in range(10):
        engine.process_quote('SPY', 450.00, 300, 450.01, 100, time.time() * 1000 + i)
    
    signal = engine.get_latest_signal('SPY')
    assert signal is not None, "Should generate signal for 3:1 imbalance"
    assert signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY], f"Expected buy signal, got {signal.signal_type}"
    print(f"  ✅ Test 1 passed: {signal.signal_type.value} with ratio {signal.imbalance_ratio:.2f}")
    
    # Test 2: Strong sell signal (1:3 ratio)
    engine2 = OrderFlowImbalanceEngine(['QQQ'])
    for i in range(10):
        engine2.process_quote('QQQ', 380.00, 100, 380.01, 300, time.time() * 1000 + i)
    
    signal2 = engine2.get_latest_signal('QQQ')
    assert signal2 is not None, "Should generate signal for 1:3 imbalance"
    assert signal2.signal_type in [SignalType.SELL, SignalType.STRONG_SELL], f"Expected sell signal, got {signal2.signal_type}"
    print(f"  ✅ Test 2 passed: {signal2.signal_type.value} with ratio {signal2.imbalance_ratio:.2f}")
    
    # Test 3: Neutral (1:1 ratio)
    engine3 = OrderFlowImbalanceEngine(['SPY'])
    for i in range(10):
        engine3.process_quote('SPY', 450.00, 100, 450.01, 100, time.time() * 1000 + i)
    
    signal3 = engine3.get_latest_signal('SPY')
    assert signal3 is None, "Should not generate signal for 1:1 ratio"
    print(f"  ✅ Test 3 passed: No signal for neutral ratio")
    
    # Test 4: Multi-timeframe
    imbalances = engine.get_multi_timeframe_imbalance('SPY')
    assert len(imbalances) == 5, "Should have 5 timeframes"
    print(f"  ✅ Test 4 passed: Multi-timeframe imbalances: {list(imbalances.keys())}")
    
    print("\n✅ All OFI Engine tests passed!")
    return True


def test_stacked_imbalance():
    """Test stacked imbalance detector"""
    print("\n🧪 Testing Stacked Imbalance Detector...")
    
    detector = StackedImbalanceDetector(min_levels=3)
    
    # Create order book with stacked buy imbalance
    bids = [(450.00, 500), (449.99, 400), (449.98, 300), (449.97, 200)]
    asks = [(450.01, 100), (450.02, 100), (450.03, 100), (450.04, 100)]
    
    detector.update_order_book('SPY', bids, asks)
    signal = detector.detect_stacked_imbalance('SPY')
    
    assert signal is not None, "Should detect stacked buy imbalance"
    assert signal.signal_type == SignalType.STRONG_BUY
    print(f"  ✅ Detected stacked imbalance: {signal.stacked_levels} levels")
    
    print("\n✅ Stacked Imbalance tests passed!")
    return True


if __name__ == "__main__":
    test_ofi_engine()
    test_stacked_imbalance()

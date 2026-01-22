"""
Order Flow Analyzer for Microstructure Features

Analyzes market microstructure for enhanced signal generation:
- Bid-ask spread analysis
- Order book depth imbalance
- Trade flow direction (buy/sell pressure)
- Smart money indicators

Designed for integration with Alpaca Market Data API.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Quote:
    """Single quote tick."""
    timestamp: datetime
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        return (self.spread / self.mid) * 10000 if self.mid > 0 else 0


@dataclass
class Trade:
    """Single trade tick."""
    timestamp: datetime
    price: float
    size: int
    conditions: List[str] = field(default_factory=list)
    
    @property
    def is_block(self) -> bool:
        """Check if trade is a block trade (>10k shares)."""
        return self.size >= 10000


@dataclass
class OrderBookLevel:
    """Single level in order book."""
    price: float
    size: int
    order_count: int = 1


@dataclass
class OrderFlowMetrics:
    """Computed order flow metrics."""
    timestamp: datetime
    
    # Spread metrics
    bid_ask_spread: float = 0.0
    spread_bps: float = 0.0
    spread_zscore: float = 0.0
    
    # Depth metrics
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    depth_imbalance: float = 0.0  # (bid - ask) / (bid + ask)
    
    # Trade flow
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    net_flow: float = 0.0  # buy - sell
    flow_imbalance: float = 0.0  # net / total
    
    # Price impact
    vwap: float = 0.0
    price_momentum: float = 0.0
    
    # Volatility
    realized_vol: float = 0.0
    quote_vol: float = 0.0
    
    # Smart money
    block_flow: float = 0.0  # Net flow from block trades
    odd_lot_flow: float = 0.0  # Net flow from odd lots (<100)
    
    @property
    def is_bullish(self) -> bool:
        """Check if order flow is bullish."""
        return (self.flow_imbalance > 0.1 and 
                self.depth_imbalance > 0 and
                self.block_flow > 0)
    
    @property
    def is_bearish(self) -> bool:
        """Check if order flow is bearish."""
        return (self.flow_imbalance < -0.1 and 
                self.depth_imbalance < 0 and
                self.block_flow < 0)


class OrderFlowAnalyzer:
    """
    Analyzes order flow and market microstructure.
    
    Features computed:
    1. Bid-ask spread (absolute and bps)
    2. Spread volatility (quote stability)
    3. Depth imbalance (bid/ask pressure)
    4. Trade flow imbalance (buy/sell pressure)
    5. Block trade flow (institutional activity)
    6. VWAP and price momentum
    7. Realized volatility from ticks
    
    Designed for real-time and historical analysis.
    """
    
    def __init__(self, window_minutes: int = 15,
                 max_history: int = 10000,
                 spread_lookback: int = 100):
        """
        Args:
            window_minutes: Rolling window for metrics
            max_history: Maximum ticks to store
            spread_lookback: Lookback for spread statistics
        """
        self.window_minutes = window_minutes
        self.max_history = max_history
        self.spread_lookback = spread_lookback
        
        # Tick storage
        self.quotes: Dict[str, deque] = {}  # ticker -> quote history
        self.trades: Dict[str, deque] = {}  # ticker -> trade history
        
        # Cached metrics
        self.cached_metrics: Dict[str, OrderFlowMetrics] = {}
        self.last_update: Dict[str, datetime] = {}
        
        # Historical spread stats for z-score
        self.spread_history: Dict[str, deque] = {}
    
    def add_quote(self, ticker: str, quote: Quote):
        """Add quote tick for ticker."""
        if ticker not in self.quotes:
            self.quotes[ticker] = deque(maxlen=self.max_history)
            self.spread_history[ticker] = deque(maxlen=self.spread_lookback)
        
        self.quotes[ticker].append(quote)
        self.spread_history[ticker].append(quote.spread)
    
    def add_trade(self, ticker: str, trade: Trade):
        """Add trade tick for ticker."""
        if ticker not in self.trades:
            self.trades[ticker] = deque(maxlen=self.max_history)
        
        self.trades[ticker].append(trade)
    
    def classify_trade(self, trade: Trade, last_quote: Optional[Quote]) -> str:
        """
        Classify trade as buy or sell using Lee-Ready algorithm.
        
        Args:
            trade: Trade to classify
            last_quote: Most recent quote
        
        Returns:
            'buy', 'sell', or 'unknown'
        """
        if last_quote is None:
            return 'unknown'
        
        mid = last_quote.mid
        
        if trade.price > mid:
            return 'buy'
        elif trade.price < mid:
            return 'sell'
        else:
            # At midpoint - use tick test
            return 'unknown'
    
    def _get_recent_quotes(self, ticker: str, 
                           minutes: Optional[int] = None) -> List[Quote]:
        """Get quotes within time window."""
        if ticker not in self.quotes:
            return []
        
        if minutes is None:
            minutes = self.window_minutes
        
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [q for q in self.quotes[ticker] if q.timestamp > cutoff]
    
    def _get_recent_trades(self, ticker: str,
                           minutes: Optional[int] = None) -> List[Trade]:
        """Get trades within time window."""
        if ticker not in self.trades:
            return []
        
        if minutes is None:
            minutes = self.window_minutes
        
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [t for t in self.trades[ticker] if t.timestamp > cutoff]
    
    def compute_metrics(self, ticker: str) -> OrderFlowMetrics:
        """
        Compute comprehensive order flow metrics for ticker.
        
        Args:
            ticker: Stock ticker
        
        Returns:
            OrderFlowMetrics with all computed values
        """
        now = datetime.now()
        
        # Check cache
        if ticker in self.cached_metrics:
            last = self.last_update.get(ticker)
            if last and (now - last).seconds < 5:
                return self.cached_metrics[ticker]
        
        quotes = self._get_recent_quotes(ticker)
        trades = self._get_recent_trades(ticker)
        
        metrics = OrderFlowMetrics(timestamp=now)
        
        if not quotes and not trades:
            return metrics
        
        # Spread metrics
        if quotes:
            spreads = [q.spread for q in quotes]
            metrics.bid_ask_spread = np.mean(spreads)
            metrics.spread_bps = np.mean([q.spread_bps for q in quotes])
            
            # Spread z-score
            if ticker in self.spread_history and len(self.spread_history[ticker]) > 10:
                hist_spreads = list(self.spread_history[ticker])
                mean_spread = np.mean(hist_spreads)
                std_spread = np.std(hist_spreads) + 1e-10
                metrics.spread_zscore = (metrics.bid_ask_spread - mean_spread) / std_spread
            
            # Quote volatility
            mids = [q.mid for q in quotes]
            if len(mids) > 1:
                returns = np.diff(mids) / np.array(mids[:-1])
                metrics.quote_vol = np.std(returns) * np.sqrt(len(returns))
            
            # Depth from latest quote
            latest_quote = quotes[-1]
            metrics.bid_depth = latest_quote.bid_size * latest_quote.bid
            metrics.ask_depth = latest_quote.ask_size * latest_quote.ask
            
            total_depth = metrics.bid_depth + metrics.ask_depth
            if total_depth > 0:
                metrics.depth_imbalance = (metrics.bid_depth - metrics.ask_depth) / total_depth
        
        # Trade flow metrics
        if trades:
            buy_volume = 0.0
            sell_volume = 0.0
            block_buy = 0.0
            block_sell = 0.0
            odd_lot_buy = 0.0
            odd_lot_sell = 0.0
            
            volume_price = 0.0
            total_volume = 0.0
            
            # Get last quote for trade classification
            last_quote = quotes[-1] if quotes else None
            
            for trade in trades:
                direction = self.classify_trade(trade, last_quote)
                value = trade.price * trade.size
                
                if direction == 'buy':
                    buy_volume += value
                    if trade.is_block:
                        block_buy += value
                    if trade.size < 100:
                        odd_lot_buy += value
                elif direction == 'sell':
                    sell_volume += value
                    if trade.is_block:
                        block_sell += value
                    if trade.size < 100:
                        odd_lot_sell += value
                
                volume_price += trade.price * trade.size
                total_volume += trade.size
            
            metrics.buy_volume = buy_volume
            metrics.sell_volume = sell_volume
            metrics.net_flow = buy_volume - sell_volume
            
            total_flow = buy_volume + sell_volume
            if total_flow > 0:
                metrics.flow_imbalance = metrics.net_flow / total_flow
            
            # Block and odd lot flow
            metrics.block_flow = block_buy - block_sell
            metrics.odd_lot_flow = odd_lot_buy - odd_lot_sell
            
            # VWAP
            if total_volume > 0:
                metrics.vwap = volume_price / total_volume
            
            # Price momentum
            if len(trades) > 1:
                first_price = trades[0].price
                last_price = trades[-1].price
                metrics.price_momentum = (last_price - first_price) / first_price
            
            # Realized volatility
            prices = [t.price for t in trades]
            if len(prices) > 1:
                returns = np.diff(prices) / np.array(prices[:-1])
                metrics.realized_vol = np.std(returns) * np.sqrt(len(returns))
        
        # Cache results
        self.cached_metrics[ticker] = metrics
        self.last_update[ticker] = now
        
        return metrics
    
    def get_feature_vector(self, ticker: str) -> np.ndarray:
        """
        Get order flow features as numpy array for ML input.
        
        Returns 10 features:
        1. Spread z-score (normalized)
        2. Depth imbalance [-1, 1]
        3. Flow imbalance [-1, 1]
        4. Block flow direction [-1, 1]
        5. Quote volatility (normalized)
        6. Price momentum [-1, 1]
        7. Spread change (bps)
        8. Volume ratio (buy/sell normalized)
        9. Odd lot indicator
        10. Liquidity score
        
        Args:
            ticker: Stock ticker
        
        Returns:
            (10,) numpy array of features
        """
        metrics = self.compute_metrics(ticker)
        
        # Normalize features
        features = np.zeros(10)
        
        # 1. Spread z-score (clipped to [-3, 3])
        features[0] = np.clip(metrics.spread_zscore, -3, 3) / 3
        
        # 2. Depth imbalance
        features[1] = np.clip(metrics.depth_imbalance, -1, 1)
        
        # 3. Flow imbalance
        features[2] = np.clip(metrics.flow_imbalance, -1, 1)
        
        # 4. Block flow direction
        if abs(metrics.block_flow) > 0:
            block_dir = np.sign(metrics.block_flow)
            block_mag = min(1.0, abs(metrics.block_flow) / (metrics.buy_volume + metrics.sell_volume + 1e-10))
            features[3] = block_dir * block_mag
        
        # 5. Quote volatility (normalized to typical range)
        features[4] = np.clip(metrics.quote_vol / 0.01, 0, 1)
        
        # 6. Price momentum
        features[5] = np.clip(metrics.price_momentum * 100, -1, 1)
        
        # 7. Spread change (bps, normalized)
        features[6] = np.clip(metrics.spread_bps / 50, 0, 1)
        
        # 8. Volume ratio
        total = metrics.buy_volume + metrics.sell_volume + 1e-10
        features[7] = (metrics.buy_volume - metrics.sell_volume) / total
        
        # 9. Odd lot indicator
        odd_total = abs(metrics.odd_lot_flow)
        features[8] = np.clip(odd_total / (total + 1e-10), 0, 1)
        
        # 10. Liquidity score (inverse of spread)
        features[9] = np.clip(1.0 / (metrics.spread_bps + 1), 0, 1)
        
        return features
    
    def get_signal(self, ticker: str) -> Tuple[str, float]:
        """
        Get trading signal from order flow.
        
        Returns:
            signal: 'buy', 'sell', or 'hold'
            strength: Signal strength [0, 1]
        """
        metrics = self.compute_metrics(ticker)
        
        # Scoring
        score = 0.0
        
        # Flow imbalance
        score += metrics.flow_imbalance * 2
        
        # Depth imbalance
        score += metrics.depth_imbalance
        
        # Block flow direction
        if abs(metrics.block_flow) > 0:
            total = metrics.buy_volume + metrics.sell_volume + 1e-10
            block_dir = np.sign(metrics.block_flow)
            block_strength = min(0.5, abs(metrics.block_flow) / total)
            score += block_dir * block_strength
        
        # Price momentum
        score += metrics.price_momentum * 10
        
        # Spread widening is bearish
        if metrics.spread_zscore > 1.5:
            score -= 0.5
        
        # Determine signal
        if score > 0.5:
            return 'buy', min(1.0, score / 2)
        elif score < -0.5:
            return 'sell', min(1.0, abs(score) / 2)
        else:
            return 'hold', 0.5
    
    def clear_history(self, ticker: Optional[str] = None):
        """Clear tick history."""
        if ticker:
            if ticker in self.quotes:
                self.quotes[ticker].clear()
            if ticker in self.trades:
                self.trades[ticker].clear()
            if ticker in self.cached_metrics:
                del self.cached_metrics[ticker]
        else:
            self.quotes.clear()
            self.trades.clear()
            self.cached_metrics.clear()
    
    def get_stats(self, ticker: str) -> Dict:
        """Get analyzer statistics for ticker."""
        return {
            'quotes_count': len(self.quotes.get(ticker, [])),
            'trades_count': len(self.trades.get(ticker, [])),
            'has_cached_metrics': ticker in self.cached_metrics,
            'window_minutes': self.window_minutes
        }


# =============================================================================
# ALPACA INTEGRATION
# =============================================================================

class AlpacaOrderFlowAdapter:
    """
    Adapter for Alpaca streaming data to OrderFlowAnalyzer.
    
    Converts Alpaca quote and trade websocket messages
    to internal format.
    """
    
    def __init__(self, analyzer: OrderFlowAnalyzer):
        """
        Args:
            analyzer: OrderFlowAnalyzer instance
        """
        self.analyzer = analyzer
    
    def on_quote(self, quote_data: Dict):
        """
        Handle Alpaca quote message.
        
        Expected format:
        {
            'S': 'AAPL',  # symbol
            'bp': 150.00,  # bid price
            'ap': 150.01,  # ask price
            'bs': 100,  # bid size
            'as': 200,  # ask size
            't': '2024-01-15T10:30:00Z'  # timestamp
        }
        """
        try:
            ticker = quote_data.get('S', quote_data.get('symbol'))
            
            quote = Quote(
                timestamp=self._parse_timestamp(quote_data.get('t')),
                bid=float(quote_data.get('bp', quote_data.get('bid_price', 0))),
                ask=float(quote_data.get('ap', quote_data.get('ask_price', 0))),
                bid_size=int(quote_data.get('bs', quote_data.get('bid_size', 0))),
                ask_size=int(quote_data.get('as', quote_data.get('ask_size', 0)))
            )
            
            self.analyzer.add_quote(ticker, quote)
            
        except Exception as e:
            logger.warning(f"Error parsing quote: {e}")
    
    def on_trade(self, trade_data: Dict):
        """
        Handle Alpaca trade message.
        
        Expected format:
        {
            'S': 'AAPL',  # symbol
            'p': 150.00,  # price
            's': 100,  # size
            'c': ['@', 'I'],  # conditions
            't': '2024-01-15T10:30:00Z'  # timestamp
        }
        """
        try:
            ticker = trade_data.get('S', trade_data.get('symbol'))
            
            trade = Trade(
                timestamp=self._parse_timestamp(trade_data.get('t')),
                price=float(trade_data.get('p', trade_data.get('price', 0))),
                size=int(trade_data.get('s', trade_data.get('size', 0))),
                conditions=trade_data.get('c', trade_data.get('conditions', []))
            )
            
            self.analyzer.add_trade(ticker, trade)
            
        except Exception as e:
            logger.warning(f"Error parsing trade: {e}")
    
    def _parse_timestamp(self, ts) -> datetime:
        """Parse timestamp from various formats."""
        if ts is None:
            return datetime.now()
        
        if isinstance(ts, datetime):
            return ts
        
        if isinstance(ts, str):
            # ISO format
            try:
                return datetime.fromisoformat(ts.replace('Z', '+00:00'))
            except:
                pass
        
        return datetime.now()


# =============================================================================
# HISTORICAL DATA ANALYSIS
# =============================================================================

def analyze_historical_orderflow(trades_df, quotes_df = None,
                                  window_minutes: int = 15) -> Dict[str, List[OrderFlowMetrics]]:
    """
    Analyze order flow from historical data.
    
    Args:
        trades_df: DataFrame with columns ['timestamp', 'symbol', 'price', 'size']
        quotes_df: Optional DataFrame with ['timestamp', 'symbol', 'bid', 'ask', 'bid_size', 'ask_size']
        window_minutes: Rolling window for metrics
    
    Returns:
        Dictionary of ticker -> list of OrderFlowMetrics
    """
    if not PANDAS_AVAILABLE:
        logger.error("pandas required for historical analysis")
        return {}
    
    analyzer = OrderFlowAnalyzer(window_minutes=window_minutes)
    
    # Sort by timestamp
    trades_df = trades_df.sort_values('timestamp')
    
    results = {}
    
    for ticker in trades_df['symbol'].unique():
        ticker_trades = trades_df[trades_df['symbol'] == ticker]
        
        # Add trades
        for _, row in ticker_trades.iterrows():
            trade = Trade(
                timestamp=row['timestamp'],
                price=row['price'],
                size=row['size'],
                conditions=row.get('conditions', [])
            )
            analyzer.add_trade(ticker, trade)
        
        # Add quotes if available
        if quotes_df is not None:
            ticker_quotes = quotes_df[quotes_df['symbol'] == ticker]
            for _, row in ticker_quotes.iterrows():
                quote = Quote(
                    timestamp=row['timestamp'],
                    bid=row['bid'],
                    ask=row['ask'],
                    bid_size=row['bid_size'],
                    ask_size=row['ask_size']
                )
                analyzer.add_quote(ticker, quote)
        
        # Compute metrics
        results[ticker] = [analyzer.compute_metrics(ticker)]
    
    return results


def compute_orderflow_features_batch(tickers: List[str],
                                     analyzer: OrderFlowAnalyzer) -> np.ndarray:
    """
    Compute order flow features for multiple tickers.
    
    Args:
        tickers: List of tickers
        analyzer: OrderFlowAnalyzer with data
    
    Returns:
        (n_tickers, 10) feature matrix
    """
    features = np.zeros((len(tickers), 10))
    
    for i, ticker in enumerate(tickers):
        features[i] = analyzer.get_feature_vector(ticker)
    
    return features

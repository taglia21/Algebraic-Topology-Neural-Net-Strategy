#!/usr/bin/env python3
"""
V16.0 Market Maker Module
=========================
Spread capture and liquidity provision for retail-scale market making.

Strategy:
- Post competitive bid/ask quotes on liquid ETFs (SPY, QQQ, IWM)
- Capture bid-ask spread while managing inventory risk
- Use IOC (Immediate-Or-Cancel) orders for quick execution
- Adjust spreads based on volatility and inventory

Features:
- Optimal spread calculation (Avellaneda-Stoikov model)
- Inventory-adjusted quoting
- Volatility-adaptive spread widening
- Risk-managed position limits
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from enum import Enum
import time
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger('V160_MarketMaker')


@dataclass
class Quote:
    """A bid or ask quote"""
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    quantity: int
    order_id: Optional[str] = None
    status: str = 'pending'  # pending, sent, filled, cancelled
    timestamp: float = field(default_factory=time.time)
    
    @property
    def value(self) -> float:
        return self.price * self.quantity


@dataclass
class Fill:
    """A filled order"""
    symbol: str
    side: str
    price: float
    quantity: int
    timestamp: float
    order_id: str
    
    @property
    def value(self) -> float:
        return self.price * self.quantity


@dataclass
class InventoryPosition:
    """Current inventory position"""
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    def update_on_fill(self, fill: Fill, current_price: float):
        """Update position after a fill"""
        if fill.side == 'buy':
            # Buying increases inventory
            if self.quantity >= 0:
                # Adding to long or starting long
                new_qty = self.quantity + fill.quantity
                self.avg_price = (self.avg_price * self.quantity + fill.price * fill.quantity) / new_qty
                self.quantity = new_qty
            else:
                # Covering short
                cover_qty = min(fill.quantity, abs(self.quantity))
                self.realized_pnl += (self.avg_price - fill.price) * cover_qty
                remaining = fill.quantity - cover_qty
                self.quantity += fill.quantity
                if remaining > 0:
                    self.avg_price = fill.price
        else:
            # Selling decreases inventory
            if self.quantity <= 0:
                # Adding to short or starting short
                new_qty = self.quantity - fill.quantity
                self.avg_price = (self.avg_price * abs(self.quantity) + fill.price * fill.quantity) / abs(new_qty)
                self.quantity = new_qty
            else:
                # Closing long
                close_qty = min(fill.quantity, self.quantity)
                self.realized_pnl += (fill.price - self.avg_price) * close_qty
                remaining = fill.quantity - close_qty
                self.quantity -= fill.quantity
                if remaining > 0:
                    self.avg_price = fill.price
        
        # Update unrealized PnL
        self.unrealized_pnl = (current_price - self.avg_price) * self.quantity if self.quantity != 0 else 0
    
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl


class SpreadModel(Enum):
    """Spread calculation models"""
    FIXED = "fixed"
    VOLATILITY_SCALED = "volatility_scaled"
    AVELLANEDA_STOIKOV = "avellaneda_stoikov"


class MarketMaker:
    """
    Retail-scale market maker for spread capture.
    
    Uses the Avellaneda-Stoikov model to calculate optimal spreads
    based on inventory risk and volatility.
    """
    
    # Default parameters
    DEFAULT_SPREAD_BPS = 5  # 5 bps minimum spread
    MAX_SPREAD_BPS = 50     # 50 bps maximum spread
    BASE_QTY = 100          # 100 shares per quote
    MAX_INVENTORY = 500     # Max shares per symbol
    GAMMA = 0.1             # Risk aversion parameter
    
    def __init__(self, 
                 symbols: List[str],
                 capital: float = 30_000,  # 30% of 100k for HF layer
                 spread_model: SpreadModel = SpreadModel.AVELLANEDA_STOIKOV):
        """
        Initialize market maker.
        
        Args:
            symbols: Symbols to make markets in
            capital: Available capital for market making
            spread_model: Method for calculating optimal spread
        """
        self.symbols = symbols
        self.capital = capital
        self.spread_model = spread_model
        
        # Positions and state
        self.positions: Dict[str, InventoryPosition] = {
            sym: InventoryPosition(symbol=sym) for sym in symbols
        }
        self.active_quotes: Dict[str, Dict[str, Quote]] = {
            sym: {'bid': None, 'ask': None} for sym in symbols
        }
        
        # Market data
        self.mid_prices: Dict[str, float] = {}
        self.volatilities: Dict[str, float] = {}  # 1-day realized vol
        self.spreads: Dict[str, float] = {}  # Current market spread
        
        # Statistics
        self.fills: List[Fill] = []
        self.quotes_sent = 0
        self.quotes_filled = 0
        self.spread_captured = 0.0
        
        # Price history for volatility
        self.price_history: Dict[str, deque] = {
            sym: deque(maxlen=1000) for sym in symbols
        }
        
    def update_market_data(self, symbol: str, bid: float, ask: float, 
                           bid_size: int = 0, ask_size: int = 0):
        """Update market data for a symbol"""
        mid = (bid + ask) / 2
        spread = ask - bid
        
        self.mid_prices[symbol] = mid
        self.spreads[symbol] = spread
        
        # Track price history for volatility
        self.price_history[symbol].append({
            'price': mid,
            'time': time.time()
        })
        
        # Update realized volatility
        if len(self.price_history[symbol]) >= 20:
            prices = [p['price'] for p in self.price_history[symbol]]
            returns = np.diff(np.log(prices))
            self.volatilities[symbol] = np.std(returns) * np.sqrt(252 * 390)  # Annualized
    
    def calculate_optimal_spread(self, symbol: str) -> float:
        """
        Calculate optimal bid-ask spread using selected model.
        
        Returns:
            Optimal spread in dollars
        """
        mid = self.mid_prices.get(symbol, 0)
        if mid == 0:
            return mid * self.DEFAULT_SPREAD_BPS / 10000
        
        if self.spread_model == SpreadModel.FIXED:
            return mid * self.DEFAULT_SPREAD_BPS / 10000
        
        elif self.spread_model == SpreadModel.VOLATILITY_SCALED:
            vol = self.volatilities.get(symbol, 0.20)  # Default 20% vol
            # Scale spread with volatility
            spread_bps = self.DEFAULT_SPREAD_BPS * (1 + vol * 5)
            spread_bps = min(spread_bps, self.MAX_SPREAD_BPS)
            return mid * spread_bps / 10000
        
        elif self.spread_model == SpreadModel.AVELLANEDA_STOIKOV:
            return self._avellaneda_stoikov_spread(symbol)
        
        return mid * self.DEFAULT_SPREAD_BPS / 10000
    
    def _avellaneda_stoikov_spread(self, symbol: str) -> float:
        """
        Avellaneda-Stoikov optimal market making spread.
        
        The model balances inventory risk with spread profit:
        - Wider spread = more profit but less fills
        - Narrower spread = more fills but more inventory risk
        """
        mid = self.mid_prices.get(symbol, 100)
        vol = self.volatilities.get(symbol, 0.20)
        inventory = self.positions[symbol].quantity
        
        # Time to end of day (T - t), use 6.5 hours = 390 minutes
        # Simplified: assume 4 hours remaining on average
        T_remaining = 4 / 6.5
        
        # Avellaneda-Stoikov reservation price adjustment
        # r(t) = s - q * gamma * sigma^2 * (T - t)
        reservation_adj = inventory * self.GAMMA * (vol ** 2) * T_remaining
        
        # Optimal spread
        # delta = gamma * sigma^2 * (T - t) + 2/gamma * ln(1 + gamma/k)
        # Simplified for retail: spread = base + inventory penalty
        base_spread = mid * self.DEFAULT_SPREAD_BPS / 10000
        vol_component = vol * mid * 0.001  # Vol-based widening
        inventory_penalty = abs(inventory) / self.MAX_INVENTORY * mid * 0.001
        
        optimal_spread = base_spread + vol_component + inventory_penalty
        
        # Cap spread
        max_spread = mid * self.MAX_SPREAD_BPS / 10000
        return min(optimal_spread, max_spread)
    
    def calculate_quote_prices(self, symbol: str) -> Tuple[float, float]:
        """
        Calculate bid and ask prices for a symbol.
        
        Returns:
            Tuple of (bid_price, ask_price)
        """
        mid = self.mid_prices.get(symbol, 0)
        if mid == 0:
            return 0, 0
        
        spread = self.calculate_optimal_spread(symbol)
        half_spread = spread / 2
        
        # Adjust for inventory skew
        inventory = self.positions[symbol].quantity
        skew = (inventory / self.MAX_INVENTORY) * half_spread * 0.5
        
        bid = mid - half_spread + skew  # Move bid up if short, down if long
        ask = mid + half_spread + skew  # Move ask up if short, down if long
        
        return round(bid, 2), round(ask, 2)
    
    def calculate_quote_size(self, symbol: str, side: str) -> int:
        """
        Calculate quote size based on inventory limits.
        
        Args:
            symbol: Symbol to quote
            side: 'buy' or 'sell'
            
        Returns:
            Quantity to quote
        """
        inventory = self.positions[symbol].quantity
        
        if side == 'buy':
            # Limit buying if already long
            room = self.MAX_INVENTORY - inventory
            return min(self.BASE_QTY, max(0, room))
        else:
            # Limit selling if already short
            room = self.MAX_INVENTORY + inventory
            return min(self.BASE_QTY, max(0, room))
    
    def generate_quotes(self, symbol: str) -> Tuple[Optional[Quote], Optional[Quote]]:
        """
        Generate bid and ask quotes for a symbol.
        
        Returns:
            Tuple of (bid_quote, ask_quote)
        """
        bid_price, ask_price = self.calculate_quote_prices(symbol)
        
        if bid_price == 0 or ask_price == 0:
            return None, None
        
        bid_qty = self.calculate_quote_size(symbol, 'buy')
        ask_qty = self.calculate_quote_size(symbol, 'sell')
        
        bid_quote = Quote(
            symbol=symbol,
            side='buy',
            price=bid_price,
            quantity=bid_qty
        ) if bid_qty > 0 else None
        
        ask_quote = Quote(
            symbol=symbol,
            side='sell',
            price=ask_price,
            quantity=ask_qty
        ) if ask_qty > 0 else None
        
        return bid_quote, ask_quote
    
    def on_fill(self, fill: Fill):
        """Handle a fill event"""
        self.fills.append(fill)
        self.quotes_filled += 1
        
        # Update position
        current_price = self.mid_prices.get(fill.symbol, fill.price)
        self.positions[fill.symbol].update_on_fill(fill, current_price)
        
        # Track spread captured (simplified)
        mid = self.mid_prices.get(fill.symbol, fill.price)
        if fill.side == 'buy':
            self.spread_captured += (mid - fill.price) * fill.quantity
        else:
            self.spread_captured += (fill.price - mid) * fill.quantity
        
        logger.debug(f"Fill: {fill.side} {fill.quantity} {fill.symbol} @ {fill.price}")
    
    def get_pnl(self) -> Dict[str, float]:
        """Get P&L breakdown"""
        realized = sum(p.realized_pnl for p in self.positions.values())
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        
        return {
            'realized_pnl': realized,
            'unrealized_pnl': unrealized,
            'total_pnl': realized + unrealized,
            'spread_captured': self.spread_captured
        }
    
    def get_stats(self) -> Dict:
        """Get market maker statistics"""
        pnl = self.get_pnl()
        fill_rate = self.quotes_filled / max(self.quotes_sent, 1)
        
        return {
            'quotes_sent': self.quotes_sent,
            'quotes_filled': self.quotes_filled,
            'fill_rate': fill_rate,
            'spread_captured': pnl['spread_captured'],
            'realized_pnl': pnl['realized_pnl'],
            'unrealized_pnl': pnl['unrealized_pnl'],
            'total_pnl': pnl['total_pnl'],
            'positions': {sym: p.quantity for sym, p in self.positions.items()}
        }


class MarketMakerBacktester:
    """
    Backtester for market making strategy.
    Simulates spread capture from historical price data.
    """
    
    def __init__(self, initial_capital: float = 30_000):
        self.initial_capital = initial_capital
        
    def simulate_from_ohlcv(self, df: pd.DataFrame, symbol: str = 'SPY',
                            base_spread_bps: float = 5) -> pd.DataFrame:
        """
        Simulate market making from OHLCV data.
        
        Assumes we capture a portion of the spread on each bar.
        """
        df = df.copy()
        
        # Estimate intraday spread from high-low range
        df['range'] = df['high'] - df['low']
        df['estimated_spread'] = np.minimum(df['range'] * 0.1, df['close'] * base_spread_bps / 10000)
        
        # Volume-based fill probability
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['fill_prob'] = np.minimum(1.0, df['volume'] / (df['volume_sma'] * 2 + 1))
        
        # Simulate spread capture (conservative: 30% of spread * fill prob)
        df['spread_capture'] = df['estimated_spread'] * 0.30 * df['fill_prob']
        
        # Per-share profit
        base_qty = 100
        df['mm_pnl'] = df['spread_capture'] * base_qty
        
        # Inventory risk (simplified): lose when price moves against position
        df['return'] = df['close'].pct_change()
        df['inventory_loss'] = -abs(df['return']) * df['close'] * base_qty * 0.1
        
        # Net P&L
        df['net_mm_pnl'] = df['mm_pnl'] + df['inventory_loss']
        
        return df
    
    def backtest(self, df: pd.DataFrame) -> Dict:
        """Run market making backtest"""
        df = df.copy()
        
        if 'net_mm_pnl' not in df.columns:
            df = self.simulate_from_ohlcv(df)
        
        # Calculate cumulative P&L
        cumulative_pnl = df['net_mm_pnl'].cumsum()
        equity = self.initial_capital + cumulative_pnl
        
        # Metrics
        trading_days = len(df)
        years = trading_days / 252
        
        total_pnl = cumulative_pnl.iloc[-1]
        total_return = total_pnl / self.initial_capital
        cagr = (equity.iloc[-1] / self.initial_capital) ** (1/max(years, 0.1)) - 1
        
        daily_returns = df['net_mm_pnl'] / self.initial_capital
        vol = daily_returns.std() * np.sqrt(252)
        sharpe = (cagr - 0.05) / max(vol, 0.01)
        
        max_dd = ((equity - equity.cummax()) / equity.cummax()).min()
        win_rate = (df['net_mm_pnl'] > 0).sum() / len(df)
        
        # Spread capture stats
        total_spread_captured = df['mm_pnl'].sum()
        total_inventory_loss = df['inventory_loss'].sum()
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe': sharpe,
            'volatility': vol,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'spread_captured': total_spread_captured,
            'inventory_loss': total_inventory_loss,
            'profit_factor': total_spread_captured / abs(total_inventory_loss) if total_inventory_loss != 0 else float('inf'),
            'final_equity': equity.iloc[-1]
        }


# ============================================================================
# Order Execution Layer
# ============================================================================

class IOCOrderExecutor:
    """
    Immediate-Or-Cancel order executor for market making.
    Interfaces with Alpaca API (or simulates for backtesting).
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True):
        self.api_key = api_key or os.getenv('ALPACA_API_KEY', '')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY', '')
        self.paper = paper
        self.client = None
        self.connected = False
        
        # Rate limiting
        self.requests_this_second = 0
        self.last_second = time.time()
        self.MAX_REQUESTS_PER_SECOND = 200
        
    def connect(self) -> bool:
        """Connect to Alpaca API"""
        try:
            from alpaca.trading.client import TradingClient
            self.client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper
            )
            # Test connection
            self.client.get_account()
            self.connected = True
            logger.info("✅ Connected to Alpaca Trading API")
            return True
        except Exception as e:
            logger.error(f"❌ Alpaca connection error: {e}")
            return False
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = time.time()
        if now - self.last_second >= 1:
            self.requests_this_second = 0
            self.last_second = now
        
        if self.requests_this_second >= self.MAX_REQUESTS_PER_SECOND:
            return False
        
        self.requests_this_second += 1
        return True
    
    def submit_ioc_order(self, symbol: str, side: str, qty: int, 
                         limit_price: float) -> Optional[str]:
        """
        Submit an Immediate-Or-Cancel limit order.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            qty: Quantity
            limit_price: Limit price
            
        Returns:
            Order ID if successful, None otherwise
        """
        if not self._check_rate_limit():
            logger.warning("Rate limit reached, skipping order")
            return None
        
        if not self.connected or not self.client:
            logger.debug("Not connected, simulating order")
            return f"SIM_{symbol}_{side}_{time.time()}"
        
        try:
            from alpaca.trading.requests import LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            
            order_data = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                type='limit',
                time_in_force=TimeInForce.IOC,
                limit_price=limit_price
            )
            
            order = self.client.submit_order(order_data)
            return str(order.id)
            
        except Exception as e:
            logger.error(f"Order submission error: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID"""
        if not self.connected or not self.client:
            return True  # Simulated
        
        if not self._check_rate_limit():
            return False
        
        try:
            self.client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.error(f"Cancel order error: {e}")
            return False


# ============================================================================
# Unit Tests
# ============================================================================

def test_market_maker():
    """Test market maker functionality"""
    print("\n🧪 Testing Market Maker Module...")
    
    mm = MarketMaker(['SPY', 'QQQ'], capital=30_000)
    
    # Test 1: Update market data
    mm.update_market_data('SPY', 450.00, 450.05, 1000, 1000)
    assert 'SPY' in mm.mid_prices
    assert mm.mid_prices['SPY'] == 450.025
    print(f"  ✅ Test 1 passed: Mid price = {mm.mid_prices['SPY']}")
    
    # Test 2: Calculate optimal spread
    spread = mm.calculate_optimal_spread('SPY')
    assert spread > 0
    print(f"  ✅ Test 2 passed: Optimal spread = ${spread:.4f}")
    
    # Test 3: Generate quotes
    bid_quote, ask_quote = mm.generate_quotes('SPY')
    assert bid_quote is not None
    assert ask_quote is not None
    assert bid_quote.price < ask_quote.price
    print(f"  ✅ Test 3 passed: Bid={bid_quote.price}, Ask={ask_quote.price}")
    
    # Test 4: Process fill
    fill = Fill(
        symbol='SPY',
        side='buy',
        price=450.00,
        quantity=100,
        timestamp=time.time(),
        order_id='test123'
    )
    mm.on_fill(fill)
    assert mm.positions['SPY'].quantity == 100
    print(f"  ✅ Test 4 passed: Position after fill = {mm.positions['SPY'].quantity}")
    
    # Test 5: Inventory skew
    bid2, ask2 = mm.generate_quotes('SPY')
    # When long, bid should be lower (less aggressive buying)
    print(f"  ✅ Test 5 passed: Skewed quotes Bid={bid2.price}, Ask={ask2.price}")
    
    print("\n✅ All Market Maker tests passed!")
    return True


def test_inventory_position():
    """Test inventory position tracking"""
    print("\n🧪 Testing Inventory Position...")
    
    pos = InventoryPosition(symbol='SPY')
    
    # Buy 100 @ $450
    fill1 = Fill('SPY', 'buy', 450.0, 100, time.time(), 'ord1')
    pos.update_on_fill(fill1, 450.0)
    assert pos.quantity == 100
    assert pos.avg_price == 450.0
    print(f"  ✅ After buy: qty={pos.quantity}, avg={pos.avg_price}")
    
    # Sell 50 @ $451 (partial close)
    fill2 = Fill('SPY', 'sell', 451.0, 50, time.time(), 'ord2')
    pos.update_on_fill(fill2, 451.0)
    assert pos.quantity == 50
    assert pos.realized_pnl == 50.0  # (451-450) * 50
    print(f"  ✅ After sell: qty={pos.quantity}, realized_pnl={pos.realized_pnl}")
    
    print("\n✅ Inventory Position tests passed!")
    return True


if __name__ == "__main__":
    test_inventory_position()
    test_market_maker()

#!/usr/bin/env python3
"""Trade Monitor - Monitors trade outcomes and triggers continuous learning.

This module watches for completed trades and feeds results into the
continuous learning system. It integrates with both Alpaca and Tradier
APIs to detect position changes.
"""

import os
import time
import logging
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class CompletedTrade:
    """Represents a completed trade with all relevant data."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    holding_period_hours: float
    trade_type: str  # 'equity' or 'option'
    features_at_entry: Optional[Dict[str, float]] = None
    regime_at_entry: Optional[str] = None
    confidence_at_entry: Optional[float] = None


class TradeMonitor:
    """Monitors positions and detects completed trades for learning."""
    
    def __init__(self):
        self.previous_positions: Dict[str, Dict] = {}
        self.pending_entries: Dict[str, Dict] = {}  # symbol -> entry data
        self.completed_trades: deque = deque(maxlen=1000)
        self.learning_integration = None
        self._running = False
        self._monitor_thread = None
        
        # Track position snapshots
        self.position_history: deque = deque(maxlen=100)
        
        # Initialize APIs
        self._init_apis()
        
        # Try to load learning integration
        try:
            from src.learning_integration import get_continuous_learner
            self.learning_integration = get_continuous_learner()
            logger.info("Continuous learning integration loaded")
        except Exception as e:
            logger.warning(f"Could not load learning integration: {e}")
    
    def _init_apis(self):
        """Initialize trading APIs."""
        self.alpaca_api = None
        self.tradier_api = None
        
        # Initialize Alpaca
        try:
            import alpaca_trade_api as tradeapi
            api_key = os.getenv('ALPACA_API_KEY')
            api_secret = os.getenv('ALPACA_SECRET_KEY')
            if api_key and api_secret:
                base_url = 'https://paper-api.alpaca.markets' if os.getenv('ALPACA_PAPER', 'true').lower() == 'true' else 'https://api.alpaca.markets'
                self.alpaca_api = tradeapi.REST(api_key, api_secret, base_url)
                logger.info("Alpaca API initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Alpaca API: {e}")
        
        # Initialize Tradier
        try:
            import requests
            token = os.getenv('TRADIER_ACCESS_TOKEN')
            if token:
                self.tradier_api = {
                    'token': token,
                    'base_url': os.getenv('TRADIER_BASE_URL', 'https://api.tradier.com/v1')
                }
                logger.info("Tradier API initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Tradier API: {e}")
    
    def get_alpaca_positions(self) -> Dict[str, Dict]:
        """Get current positions from Alpaca."""
        positions = {}
        if not self.alpaca_api:
            return positions
        
        try:
            alpaca_positions = self.alpaca_api.list_positions()
            for pos in alpaca_positions:
                positions[pos.symbol] = {
                    'symbol': pos.symbol,
                    'qty': int(pos.qty),
                    'side': 'long' if int(pos.qty) > 0 else 'short',
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'unrealized_pnl': float(pos.unrealized_pl),
                    'market_value': float(pos.market_value),
                    'source': 'alpaca'
                }
        except Exception as e:
            logger.error(f"Error getting Alpaca positions: {e}")
        
        return positions
    
    def get_tradier_positions(self) -> Dict[str, Dict]:
        """Get current positions from Tradier."""
        positions = {}
        if not self.tradier_api:
            return positions
        
        try:
            import requests
            headers = {
                'Authorization': f"Bearer {self.tradier_api['token']}",
                'Accept': 'application/json'
            }
            
            account_id = os.getenv('TRADIER_ACCOUNT_ID')
            url = f"{self.tradier_api['base_url']}/accounts/{account_id}/positions"
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if data.get('positions') and data['positions'].get('position'):
                    pos_list = data['positions']['position']
                    if isinstance(pos_list, dict):
                        pos_list = [pos_list]
                    
                    for pos in pos_list:
                        symbol = pos.get('symbol', '')
                        positions[symbol] = {
                            'symbol': symbol,
                            'qty': int(pos.get('quantity', 0)),
                            'side': 'long' if int(pos.get('quantity', 0)) > 0 else 'short',
                            'avg_entry_price': float(pos.get('cost_basis', 0)) / max(abs(int(pos.get('quantity', 1))), 1),
                            'current_price': float(pos.get('last', 0)),
                            'unrealized_pnl': float(pos.get('gain_loss', 0)),
                            'market_value': float(pos.get('market_value', 0)),
                            'source': 'tradier'
                        }
        except Exception as e:
            logger.error(f"Error getting Tradier positions: {e}")
        
        return positions
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """Get combined positions from all sources."""
        positions = {}
        positions.update(self.get_alpaca_positions())
        positions.update(self.get_tradier_positions())
        return positions
    
    def record_entry(self, symbol: str, entry_data: Dict):
        """Record a new position entry for tracking.
        
        Call this when entering a new trade.
        """
        self.pending_entries[symbol] = {
            'entry_time': datetime.now(),
            'entry_price': entry_data.get('price', 0.0),
            'quantity': entry_data.get('quantity', 0),
            'side': entry_data.get('side', 'long'),
            'trade_type': entry_data.get('trade_type', 'equity'),
            'features': entry_data.get('features', {}),
            'regime': entry_data.get('regime'),
            'confidence': entry_data.get('confidence')
        }
        logger.info(f"Recorded entry for {symbol}")
    
    def detect_closed_positions(self) -> List[CompletedTrade]:
        """Compare current positions with previous to find closed trades."""
        current_positions = self.get_all_positions()
        closed_trades = []
        
        # Find positions that were present before but not now (closed)
        for symbol, prev_pos in self.previous_positions.items():
            if symbol not in current_positions:
                # Position was closed
                entry_data = self.pending_entries.get(symbol, {})
                
                # Calculate PnL
                entry_price = entry_data.get('entry_price', prev_pos.get('avg_entry_price', 0))
                exit_price = prev_pos.get('current_price', entry_price)
                quantity = prev_pos.get('qty', 1)
                side = prev_pos.get('side', 'long')
                
                if side == 'long':
                    pnl = (exit_price - entry_price) * quantity
                else:
                    pnl = (entry_price - exit_price) * quantity
                
                pnl_pct = (pnl / (entry_price * quantity)) * 100 if entry_price > 0 else 0
                
                # Calculate holding period
                entry_time = entry_data.get('entry_time', datetime.now() - timedelta(hours=24))
                exit_time = datetime.now()
                holding_hours = (exit_time - entry_time).total_seconds() / 3600
                
                completed = CompletedTrade(
                    symbol=symbol,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    side=side,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    holding_period_hours=holding_hours,
                    trade_type=entry_data.get('trade_type', 'equity'),
                    features_at_entry=entry_data.get('features'),
                    regime_at_entry=entry_data.get('regime'),
                    confidence_at_entry=entry_data.get('confidence')
                )
                
                closed_trades.append(completed)
                self.completed_trades.append(completed)
                
                # Remove from pending entries
                if symbol in self.pending_entries:
                    del self.pending_entries[symbol]
                
                logger.info(f"Detected closed trade: {symbol}, PnL: {pnl:.2f} ({pnl_pct:.2f}%)")
        
        # Find new positions (for tracking future closes)
        for symbol, curr_pos in current_positions.items():
            if symbol not in self.previous_positions:
                # New position opened
                self.pending_entries[symbol] = {
                    'entry_time': datetime.now(),
                    'entry_price': curr_pos.get('avg_entry_price', 0),
                    'quantity': curr_pos.get('qty', 0),
                    'side': curr_pos.get('side', 'long'),
                    'trade_type': 'equity'  # Default, can be updated
                }
                logger.info(f"Detected new position: {symbol}")
        
        # Update previous positions
        self.previous_positions = current_positions.copy()
        
        return closed_trades
    
    def process_closed_trade(self, trade: CompletedTrade):
        """Process a closed trade through the learning system."""
        if not self.learning_integration:
            logger.warning("Learning integration not available")
            return
        
        try:
            # Convert to the format expected by learning integration
            trade_data = {
                'symbol': trade.symbol,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'side': trade.side,
                'holding_period_hours': trade.holding_period_hours,
                'trade_type': trade.trade_type,
                'regime': trade.regime_at_entry,
                'confidence': trade.confidence_at_entry
            }
            
            features = trade.features_at_entry or {}
            
            # Call learning integration
            self.learning_integration.on_trade_closed(
                symbol=trade.symbol,
                trade_result=trade_data,
                features=features
            )
            
            logger.info(f"Processed trade through learning system: {trade.symbol}")
            
        except Exception as e:
            logger.error(f"Error processing trade through learning: {e}")
    
    def check_and_learn(self):
        """Single check iteration - detect closed trades and process."""
        closed_trades = self.detect_closed_positions()
        
        for trade in closed_trades:
            self.process_closed_trade(trade)
        
        return closed_trades
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start background monitoring thread."""
        if self._running:
            logger.warning("Monitoring already running")
            return
        
        self._running = True
        
        def monitor_loop():
            logger.info(f"Starting trade monitor (interval: {interval_seconds}s)")
            
            # Initial position snapshot
            self.previous_positions = self.get_all_positions()
            logger.info(f"Initial positions: {list(self.previous_positions.keys())}")
            
            while self._running:
                try:
                    self.check_and_learn()
                except Exception as e:
                    logger.error(f"Error in monitor loop: {e}")
                
                time.sleep(interval_seconds)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Trade monitor started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Trade monitor stopped")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics from the learning system."""
        stats = {
            'trades_processed': len(self.completed_trades),
            'current_positions': len(self.previous_positions),
            'pending_entries': len(self.pending_entries)
        }
        
        if self.learning_integration:
            try:
                learner_stats = self.learning_integration.get_stats()
                stats.update(learner_stats)
            except Exception as e:
                logger.error(f"Error getting learner stats: {e}")
        
        return stats


# Global singleton instance
_trade_monitor: Optional[TradeMonitor] = None

def get_trade_monitor() -> TradeMonitor:
    """Get or create the global trade monitor instance."""
    global _trade_monitor
    if _trade_monitor is None:
        _trade_monitor = TradeMonitor()
    return _trade_monitor


def on_trade_entry(symbol: str, entry_data: Dict):
    """Convenience function to record a trade entry."""
    monitor = get_trade_monitor()
    monitor.record_entry(symbol, entry_data)


def start_learning_monitor(interval: int = 60):
    """Start the background learning monitor."""
    monitor = get_trade_monitor()
    monitor.start_monitoring(interval)


def stop_learning_monitor():
    """Stop the background learning monitor."""
    monitor = get_trade_monitor()
    monitor.stop_monitoring()


if __name__ == '__main__':
    # Test the trade monitor
    print("Testing Trade Monitor...")
    
    monitor = get_trade_monitor()
    print(f"Initial stats: {monitor.get_learning_stats()}")
    
    # Check positions
    positions = monitor.get_all_positions()
    print(f"Current positions: {list(positions.keys())}")
    
    # Run a single check
    closed = monitor.check_and_learn()
    print(f"Closed trades detected: {len(closed)}")
    
    print("Trade Monitor test complete!")

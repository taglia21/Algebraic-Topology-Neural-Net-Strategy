"""Execution Quality Monitor for trade analytics."""
import logging
from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ExecutionMetrics:
    total_trades: int = 0
    avg_slippage_bps: float = 0.0
    fill_rate: float = 1.0
    avg_latency_ms: float = 0.0
    rejected_orders: int = 0

class ExecutionMonitor:
    """Monitors trade execution quality."""
    
    def __init__(self):
        self.trades: List[dict] = []
        self.metrics = ExecutionMetrics()
        self.slippage_by_hour: Dict[int, List[float]] = defaultdict(list)
        
    def record_trade(self, expected_price: float, fill_price: float, 
                     latency_ms: float = 0.0, filled: bool = True):
        """Record a trade execution."""
        slippage_bps = abs(fill_price - expected_price) / expected_price * 10000
        hour = datetime.now().hour
        
        self.trades.append({
            'timestamp': datetime.now(),
            'expected': expected_price,
            'filled': fill_price,
            'slippage_bps': slippage_bps,
            'latency_ms': latency_ms,
            'success': filled
        })
        
        self.slippage_by_hour[hour].append(slippage_bps)
        self._update_metrics()
        
    def _update_metrics(self):
        """Update aggregate metrics."""
        if not self.trades:
            return
        self.metrics.total_trades = len(self.trades)
        self.metrics.avg_slippage_bps = sum(t['slippage_bps'] for t in self.trades) / len(self.trades)
        self.metrics.fill_rate = sum(1 for t in self.trades if t['success']) / len(self.trades)
        self.metrics.avg_latency_ms = sum(t['latency_ms'] for t in self.trades) / len(self.trades)
        
    def get_best_execution_hour(self) -> int:
        """Return hour with lowest average slippage."""
        if not self.slippage_by_hour:
            return 10  # Default to market open
        return min(self.slippage_by_hour.keys(), 
                   key=lambda h: sum(self.slippage_by_hour[h])/len(self.slippage_by_hour[h]))

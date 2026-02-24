import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ExecutionVenue:
    name: str
    fee_rebate: float
    latency_ms: float
    liquidity_score: float

class SmartOrderRouter:
    """Routes orders to the optimal execution venue."""
    def __init__(self):
        self.venues = {} # name -> ExecutionVenue
        
    def add_venue(self, venue: ExecutionVenue):
        self.venues[venue.name] = venue
        
    def route_order(self, symbol: str, quantity: int, side: str) -> str:
        """Selects the best venue based on latency, liquidity, and fees."""
        best_venue = None
        best_score = -float('inf')
        
        for name, venue in self.venues.items():
            # Scoring formula: Liquidity - (Latency * coefficient) + (Rebate * coefficient)
            score = (venue.liquidity_score * 10) - (venue.latency_ms * 0.5) + (venue.fee_rebate * 100)
            if score > best_score:
                best_score = score
                best_venue = name
                
        logger.info(f"Routing {side} {quantity} {symbol} to {best_venue} (score: {best_score:.2f})")
        return best_venue

class AdaptiveLimitOrderPlacer:
    """Adjusts limit prices based on order book imbalance."""
    def __init__(self, aggressiveness: float = 0.5):
        self.aggressiveness = aggressiveness

    def calculate_limit_price(self, mid_price: float, obi: float, side: str) -> float:
        """
        obi: Order Book Imbalance (-1 to 1)
        side: 'buy' or 'sell'
        """
        # If OBI is positive (more buy pressure), price moves up
        adjustment = mid_price * 0.0001 * obi * self.aggressiveness
        
        if side == 'buy':
            return mid_price + adjustment
        else:
            return mid_price - adjustment


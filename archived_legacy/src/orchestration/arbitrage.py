import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ArbitrageOpportunity:
    symbol_a: str
    symbol_b: str
    price_a: float
    price_b: float
    spread: float
    expected_profit: float

class ArbitrageEngine:
    """Detects and executes arbitrage opportunities between correlated assets."""
    def __init__(self, min_profit_threshold: float = 0.005):
        self.min_profit_threshold = min_profit_threshold
        self.correlated_pairs = [] # List of tuples (symbol_a, symbol_b, hedge_ratio)

    def add_pair(self, symbol_a: str, symbol_b: str, hedge_ratio: float = 1.0):
        self.correlated_pairs.append((symbol_a, symbol_b, hedge_ratio))

    def scan(self, market_data: Dict[str, Dict[str, float]]) -> List[ArbitrageOpportunity]:
        opportunities = []
        for sym_a, sym_b, ratio in self.correlated_pairs:
            if sym_a in market_data and sym_b in market_data:
                price_a = market_data[sym_a]['price']
                price_b = market_data[sym_b]['price']
                
                # Simple price discrepancy (normalized by ratio)
                spread = (price_a / ratio) - price_b
                expected_profit = abs(spread) / price_b
                
                if expected_profit > self.min_profit_threshold:
                    opportunities.append(ArbitrageOpportunity(
                        symbol_a=sym_a,
                        symbol_b=sym_b,
                        price_a=price_a,
                        price_b=price_b,
                        spread=spread,
                        expected_profit=expected_profit
                    ))
        return opportunities


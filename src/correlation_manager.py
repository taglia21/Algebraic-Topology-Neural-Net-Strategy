"""Position Correlation Manager for portfolio risk."""
import logging
from typing import Dict, List, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Sector mappings for correlation
SECTOR_MAP = {
    'SPY': 'broad_market', 'QQQ': 'tech', 'IWM': 'small_cap',
    'XLK': 'tech', 'XLF': 'financials', 'XLE': 'energy',
    'XLV': 'healthcare', 'XLI': 'industrials', 'XLP': 'consumer_staples',
    'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech', 'AMZN': 'tech',
    'NVDA': 'tech', 'META': 'tech', 'TSLA': 'tech',
    'JPM': 'financials', 'BAC': 'financials', 'GS': 'financials',
}

class CorrelationManager:
    """Manages position correlation and sector exposure."""
    
    def __init__(self, max_sector_exposure: float = 0.3, max_correlation: float = 0.3):
        self.max_sector_exposure = max_sector_exposure
        self.max_correlation = max_correlation
        self.positions: Dict[str, float] = {}  # symbol -> weight
        
    def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        return SECTOR_MAP.get(symbol, 'unknown')
    
    def get_sector_exposure(self, sector: str) -> float:
        """Calculate current exposure to a sector."""
        total = sum(self.positions.values()) or 1.0
        sector_weight = sum(
            weight for sym, weight in self.positions.items()
            if self.get_sector(sym) == sector
        )
        return sector_weight / total
    
    def can_add_position(self, symbol: str, weight: float) -> bool:
        """Check if adding position would exceed correlation limits."""
        sector = self.get_sector(symbol)
        current_exposure = self.get_sector_exposure(sector)
        return (current_exposure + weight) <= self.max_sector_exposure
    
    def update_position(self, symbol: str, weight: float):
        """Update position weight."""
        if weight > 0:
            self.positions[symbol] = weight
        elif symbol in self.positions:
            del self.positions[symbol]

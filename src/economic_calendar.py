"""Economic Calendar for high-impact event awareness."""
import logging
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, date
from enum import Enum

logger = logging.getLogger(__name__)

class EventImpact(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

@dataclass
class EconomicEvent:
    name: str
    date: date
    impact: EventImpact
    description: str = ""

class EconomicCalendar:
    """Tracks high-impact economic events."""
    
    # Known high-impact events (static for reliability)
    HIGH_IMPACT_EVENTS = [
        'FOMC', 'Federal Reserve', 'Interest Rate Decision',
        'Non-Farm Payrolls', 'NFP', 'Jobs Report',
        'CPI', 'Consumer Price Index', 'Inflation',
        'GDP', 'Gross Domestic Product',
        'Retail Sales', 'ISM Manufacturing'
    ]
    
    def __init__(self):
        self.events: List[EconomicEvent] = []
        
    def is_high_impact_day(self, check_date: Optional[date] = None) -> bool:
        """Check if today is a high-impact event day."""
        if check_date is None:
            check_date = date.today()
        # Check day of week - avoid FOMC Wednesdays, NFP Fridays
        weekday = check_date.weekday()
        # First Friday of month is typically NFP
        if weekday == 4 and check_date.day <= 7:
            return True
        return False
    
    def get_event_risk_level(self) -> EventImpact:
        """Get current event risk level."""
        if self.is_high_impact_day():
            return EventImpact.HIGH
        return EventImpact.LOW
    
    def get_position_size_multiplier(self) -> float:
        """Return position size multiplier based on event risk."""
        risk = self.get_event_risk_level()
        if risk == EventImpact.HIGH:
            return 0.5  # Reduce position size by 50%
        elif risk == EventImpact.MEDIUM:
            return 0.75
        return 1.0

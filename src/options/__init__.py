"""
Medallion-Grade Options Trading Engine
======================================

Production-ready options trading system with institutional-grade risk management.

Key Components:
- Theta decay optimization
- IV rank/percentile analysis
- Portfolio Greeks management
- 15-minute delay compensation
- Premium selling strategies (Wheel, Spreads, Iron Condors)
- Comprehensive risk controls

Author: Medallion Options Team
Version: 1.0.0
Date: February 4, 2026
"""

from .theta_decay_engine import ThetaDecayEngine, ThetaMetrics
from .iv_analyzer import IVAnalyzer, IVMetrics, IVRegime
from .greeks_manager import (
    GreeksManager, 
    PortfolioGreeks, 
    PositionGreeks,
    GreeksViolation,
    GreeksViolationType,
    HedgeRecommendation
)
from .delay_adapter import DelayAdapter, DelayedPrice, EntryAdjustment, MarketPeriod
from .position_manager import PositionManager, Position, PositionSizing, PositionStatus
from .strategy_engine import (
    StrategyEngine,
    StrategyType,
    SpreadType,
    OptionCandidate,
    SpreadCandidate,
    IronCondorCandidate
)
from .tradier_executor import (
    TradierExecutor, 
    OrderResult, 
    OptionLeg,
    OrderSide,
    OrderType as OrderTypeEnum,
    OrderStatus
)

__version__ = "1.0.0"
__all__ = [
    # Theta
    "ThetaDecayEngine",
    "ThetaMetrics",
    # IV Analysis
    "IVAnalyzer",
    "IVMetrics",
    "IVRegime",
    # Greeks
    "GreeksManager",
    "PortfolioGreeks",
    "PositionGreeks",
    "GreeksViolation",
    "GreeksViolationType",
    "HedgeRecommendation",
    # Delay
    "DelayAdapter",
    "DelayedPrice",
    "EntryAdjustment",
    "MarketPeriod",
    # Position
    "PositionManager",
    "Position",
    "PositionSizing",
    "PositionStatus",
    # Strategy
    "StrategyEngine",
    "StrategyType",
    "SpreadType",
    "OptionCandidate",
    "SpreadCandidate",
    "IronCondorCandidate",
    # Execution
    "TradierExecutor",
    "OrderResult",
    "OptionLeg",
    "OrderSide",
    "OrderTypeEnum",
    "OrderStatus",
]

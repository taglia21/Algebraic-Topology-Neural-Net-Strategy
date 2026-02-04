"""Enhanced Trading System - Production Modules

LOW-SEVERITY FIX: Proper module exports for clean imports.
"""

from .risk_manager import RiskManager, RiskConfig, Position
from .position_sizer import PositionSizer, SizingConfig, PerformanceMetrics
from .multi_timeframe_analyzer import MultiTimeframeAnalyzer, AnalyzerConfig
from .sentiment_analyzer import SentimentAnalyzer, SentimentConfig
from .enhanced_trading_engine import EnhancedTradingEngine, EngineConfig, TradeDecision

__version__ = "1.0.0"

__all__ = [
    "RiskManager",
    "RiskConfig",
    "Position",
    "PositionSizer",
    "SizingConfig",
    "PerformanceMetrics",
    "MultiTimeframeAnalyzer",
    "AnalyzerConfig",
    "SentimentAnalyzer",
    "SentimentConfig",
    "EnhancedTradingEngine",
    "EngineConfig",
    "TradeDecision",
]

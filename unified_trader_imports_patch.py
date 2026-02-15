# Phase 3-9 Component Imports (add after line ~750 in try/except block)
try:
    from src.news_sentiment import NewsSentimentAnalyzer
    from src.economic_calendar import EconomicCalendar
    from src.correlation_manager import CorrelationManager
    from src.execution_monitor import ExecutionMonitor
    from src.walk_forward import WalkForwardOptimizer
    from src.ml_retrainer import MLRetrainer
    from src.multi_timeframe import MultiTimeframeAnalyzer
    PHASE3_9_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Phase 3-9 components not available: {e}")
    PHASE3_9_AVAILABLE = False

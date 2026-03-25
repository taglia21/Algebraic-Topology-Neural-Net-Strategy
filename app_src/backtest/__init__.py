"""
backtest/
=========
Event-driven backtesting engine for the ATNN v2 Quant Powerhouse.

Modules
-------
engine              — :class:`BacktestEngine` event loop + :class:`BacktestResult`
metrics             — :class:`PerformanceMetrics` & :class:`BacktestMetrics` analytics
walk_forward        — :class:`WalkForwardOptimizer` rolling walk-forward framework
options_backtester  — :class:`OptionsBacktester` defined-risk options strategies
report              — :class:`BacktestReport` HTML report generation
"""

from backtest.engine import (
    AssetType,
    BacktestEngine,
    BacktestResult,
    CommissionCalculator,
    Event,
    EventType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Trade,
)
from backtest.metrics import BacktestMetrics, PerformanceMetrics
from backtest.options_backtester import (
    BlackScholes,
    OptionLeg,
    OptionPosition,
    OptionsBacktestResult,
    OptionsBacktester,
)
from backtest.report import BacktestReport
from backtest.walk_forward import (
    WalkForwardOptimizer,
    WalkForwardResult,
    WindowResult,
    WindowSchedule,
)

__all__ = [
    # Engine
    "BacktestEngine",
    "BacktestResult",
    "CommissionCalculator",
    "Event",
    "EventType",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Position",
    "Trade",
    "AssetType",
    # Metrics
    "PerformanceMetrics",
    "BacktestMetrics",
    # Walk-forward
    "WalkForwardOptimizer",
    "WalkForwardResult",
    "WindowResult",
    "WindowSchedule",
    # Options
    "OptionsBacktester",
    "OptionsBacktestResult",
    "BlackScholes",
    "OptionLeg",
    "OptionPosition",
    # Report
    "BacktestReport",
]

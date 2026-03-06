"""
backtest/
=========
Event-driven backtesting engine for the ATNN Quant Powerhouse.

Modules
-------
backtester  — :class:`Backtester` event loop + :class:`BacktestResult`
metrics     — :class:`PerformanceMetrics` analytics suite
"""

from backtest.metrics import BacktestResult, PerformanceMetrics
from backtest.backtester import Backtester

__all__ = ["Backtester", "BacktestResult", "PerformanceMetrics"]

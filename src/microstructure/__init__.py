"""
Market Microstructure Module

This module contains:
- OrderFlowAnalyzer: Bid-ask spread, order book depth, trade imbalance analysis
- AlpacaOrderFlowAdapter: Adapter for Alpaca streaming data
- Quote, Trade: Data structures for market ticks
"""

try:
    from .order_flow_analyzer import (
        OrderFlowAnalyzer,
        AlpacaOrderFlowAdapter,
        Quote,
        Trade,
        OrderFlowMetrics
    )
except ImportError as e:
    OrderFlowAnalyzer = None
    AlpacaOrderFlowAdapter = None
    Quote = None
    Trade = None
    OrderFlowMetrics = None

__all__ = [
    'OrderFlowAnalyzer',
    'AlpacaOrderFlowAdapter',
    'Quote',
    'Trade',
    'OrderFlowMetrics'
]

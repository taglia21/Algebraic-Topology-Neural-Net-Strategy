"""
ensemble/
=========
Meta-classifier ensemble for dynamic capital allocation between
TDA arbitrage and NN directional strategies.
"""

from ensemble.meta_allocator import AllocationResult, MetaAllocator
from ensemble.risk_manager import EnsembleRiskManager, PositionSize, RiskReport
from ensemble.signal_aggregator import SignalAggregator
from ensemble.strategy_nn import NNDirectionalStrategy
from ensemble.strategy_tda import TDADiffusionStrategy

__all__ = [
    "TDADiffusionStrategy",
    "NNDirectionalStrategy",
    "MetaAllocator",
    "AllocationResult",
    "SignalAggregator",
    "EnsembleRiskManager",
    "PositionSize",
    "RiskReport",
]

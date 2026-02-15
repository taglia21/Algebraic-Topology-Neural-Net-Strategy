"""Options trading module for Algebraic-Topology-Neural-Net-Strategy."""
from .signal_generator import IVRankStrategy
from .contract_resolver import OptionContractResolver
from .weight_optimizer import DynamicWeightOptimizer

# Convenience aliases
ContractResolver = OptionContractResolver
WeightOptimizer = DynamicWeightOptimizer

__all__ = [
    'IVRankStrategy',
    'OptionContractResolver',
    'ContractResolver',
    'DynamicWeightOptimizer',
    'WeightOptimizer',
]

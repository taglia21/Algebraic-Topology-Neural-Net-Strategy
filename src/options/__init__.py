"""Options trading module for Algebraic-Topology-Neural-Net-Strategy."""
from .signal_generator import (
    IVRankStrategy,
    ThetaDecayStrategy,
    MeanReversionStrategy,
    DeltaHedgingStrategy,
    VolDivergenceStrategy,
    VRPStrategy,
    IVCrushStrategy,
    EarningsIVCrushStrategy,
    ZeroDTEIronButterflyStrategy,
    SignalGenerator,
    Signal,
    SignalType,
    SignalSource,
    bayesian_combine_confidence,
)
from .contract_resolver import OptionContractResolver
from .weight_optimizer import DynamicWeightOptimizer

# Convenience aliases
ContractResolver = OptionContractResolver
WeightOptimizer = DynamicWeightOptimizer

__all__ = [
    'IVRankStrategy',
    'ThetaDecayStrategy',
    'MeanReversionStrategy',
    'DeltaHedgingStrategy',
    'VolDivergenceStrategy',
    'VRPStrategy',
    'IVCrushStrategy',
    'EarningsIVCrushStrategy',
    'ZeroDTEIronButterflyStrategy',
    'SignalGenerator',
    'Signal',
    'SignalType',
    'SignalSource',
    'bayesian_combine_confidence',
    'OptionContractResolver',
    'ContractResolver',
    'DynamicWeightOptimizer',
    'WeightOptimizer',
]

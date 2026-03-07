"""
equities/strategies/__init__.py
================================
Strategy sub-package for the equities trading engine.

Available strategies
--------------------
StatArbStrategy      — Statistical arbitrage / pairs trading with OU model and Kalman filter.
MomentumStrategy     — Cross-sectional residual momentum with sector-neutral construction.
FactorModelStrategy  — Multi-factor alpha model (Quality, Value, Low-Vol, Momentum).
MeanReversionStrategy — Short-term mean reversion based on 5-day return z-scores (Lehmann 1990).
"""

from equities.strategies.stat_arb import StatArbStrategy
from equities.strategies.momentum import MomentumStrategy
from equities.strategies.factor_model import FactorModelStrategy
from equities.strategies.mean_reversion import MeanReversionStrategy

__all__ = [
    "StatArbStrategy",
    "MomentumStrategy",
    "FactorModelStrategy",
    "MeanReversionStrategy",
]

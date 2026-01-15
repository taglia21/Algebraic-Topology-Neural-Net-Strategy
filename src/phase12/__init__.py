"""
Phase 12: All-Weather Regime-Switching Strategy
================================================

Profit in BOTH bull and bear markets through:
1. Multi-signal regime classification
2. Long leveraged ETFs in bull regimes (TQQQ, SPXL, SOXL)
3. Inverse leveraged ETFs in bear regimes (SQQQ, SPXU, SOXS)
4. Adaptive risk management with regime-aware controls
"""

from .regime_classifier import RegimeClassifier, RegimeState, RegimeSignals
from .inverse_allocator import InverseAllocator, AllocationResult
from .adaptive_risk_manager import AdaptiveRiskManager, RiskState
from .all_weather_orchestrator import AllWeatherOrchestrator, Phase12Config

__all__ = [
    'RegimeClassifier',
    'RegimeState',
    'RegimeSignals',
    'InverseAllocator',
    'AllocationResult',
    'AdaptiveRiskManager',
    'RiskState',
    'AllWeatherOrchestrator',
    'Phase12Config',
]

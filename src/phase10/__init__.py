"""Phase 10: Aggressive Alpha Amplification.

This module transforms Phase 9's risk-adjusted performance into aggressive
absolute returns (25-35% CAGR) using dynamic leverage, high-beta concentration,
and convex payoff structures.

Components:
- dynamic_leverage: Kelly-based regime-adaptive leverage calculator
- leverage_optimizer: Optimal leverage with drawdown controls
- phase10_orchestrator: Main integration layer
"""

from .dynamic_leverage import (
    KellyLeverageCalculator,
    RegimeLeverageScaler,
    LeverageAdjuster,
    DynamicLeverageEngine,
    LeverageState,
)

from .phase10_orchestrator import (
    Phase10Orchestrator,
    Phase10Config,
    Phase10State,
)

__all__ = [
    # Dynamic Leverage
    'KellyLeverageCalculator',
    'RegimeLeverageScaler', 
    'LeverageAdjuster',
    'DynamicLeverageEngine',
    'LeverageState',
    
    # Orchestrator
    'Phase10Orchestrator',
    'Phase10Config',
    'Phase10State',
]

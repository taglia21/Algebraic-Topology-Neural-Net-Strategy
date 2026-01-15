"""
Phase 13: Validation + Options Amplification
=============================================

PART 1: Rigorous Validation
- Walk-forward analysis
- Monte Carlo simulation
- Transaction cost sensitivity
- Parameter stability testing

PART 2: Options Amplification
- Covered calls for income
- Protective puts for downside protection
- Long calls/puts for directional bets
- Straddles for volatility plays

PART 3: Production Deployment
- Real-time monitoring
- Circuit breakers
- Position limits
- Execution quality checks
"""

from .validation import WalkForwardValidator, MonteCarloSimulator, CostSensitivityAnalyzer
from .options_pricer import BlackScholes, OptionsGreeks
from .options_overlay import OptionsOverlay, OptionsSignal, OptionsPosition
from .production_controls import ProductionController, CircuitBreaker, PositionLimits

__all__ = [
    'WalkForwardValidator',
    'MonteCarloSimulator', 
    'CostSensitivityAnalyzer',
    'BlackScholes',
    'OptionsGreeks',
    'OptionsOverlay',
    'OptionsSignal',
    'OptionsPosition',
    'ProductionController',
    'CircuitBreaker',
    'PositionLimits',
]

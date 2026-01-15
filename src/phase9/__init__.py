"""Phase 9: Advanced Alpha Generation & Regime-Adaptive Optimization.

This module implements a comprehensive 5-pillar transformation system targeting
institutional-grade alpha generation with:
- CAGR: 30-50%+
- Sharpe Ratio: > 2.0
- Max Drawdown: < 15%

Components:
1. Hierarchical Regime Meta-Strategy (HMM + TDA + Dynamic Allocation)
2. Advanced Alpha Engine (Multi-horizon momentum, reversal capture)
3. Adaptive Universe Screener (Quality + Momentum + TDA filtering)
4. Dynamic Position Optimizer (Kelly-based with regime scaling)
5. Risk-Adjusted Ensemble (Combine signals across timeframes)
"""

from .regime_meta_strategy import HierarchicalRegimeStrategy, RegimeMeta
from .alpha_engine import AdvancedAlphaEngine, AlphaSignal
from .adaptive_screener import AdaptiveUniverseScreener
from .dynamic_optimizer import DynamicPositionOptimizer
from .phase9_orchestrator import Phase9Orchestrator, Phase9Config

__all__ = [
    'HierarchicalRegimeStrategy',
    'RegimeMeta', 
    'AdvancedAlphaEngine',
    'AlphaSignal',
    'AdaptiveUniverseScreener',
    'DynamicPositionOptimizer',
    'Phase9Orchestrator',
    'Phase9Config',
]

"""
Phase 11: Total Market Domination
=================================

Scalable multi-factor stock selection across the entire US market.

Key Components:
- UniverseManager: Get all US stocks from Polygon.io
- FactorEngine: 5-factor composite scoring (Momentum, Quality, Vol-Adj, RelStrength, Liquidity)
- PortfolioConstructor: Select top 30-50 stocks with concentration
- SectorLeverageManager: Dynamic 3x sector ETF allocation
- RiskController: Drawdown protection and position limits
"""

from .universe_manager import UniverseManager, StockFilter
from .factor_engine import FactorEngine, FactorWeights
from .portfolio_constructor import PortfolioConstructor, PortfolioConfig
from .sector_leverage import SectorLeverageManager, LeverageConfig
from .risk_controller import Phase11RiskController, RiskConfig, compute_portfolio_stats

__all__ = [
    'UniverseManager',
    'StockFilter', 
    'FactorEngine',
    'FactorWeights',
    'PortfolioConstructor',
    'PortfolioConfig',
    'SectorLeverageManager',
    'LeverageConfig',
    'Phase11RiskController',
    'RiskConfig',
    'compute_portfolio_stats',
]

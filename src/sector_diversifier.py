"""
Sector Diversifier - Strategic Sector Allocation for Risk-Adjusted Returns

Implements research-backed sector diversification:
1. Defensive Core (30%): Utilities, Consumer Staples, Healthcare
2. Growth Engine (50%): Technology, Financials, Industrials  
3. Tactical Rotation (20%): Top momentum sectors

Academic research shows:
- Defensive sectors outperform by 15-25% in bear markets (Stanford, Vanguard)
- Sector momentum captures persistent trends (SSRN 2024)
- Volatility parity reduces tail risk (AlphaArchitect 2022)

Target: No sector >25%, defensive allocation 30%, tactical 20%
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# GICS Sector ETFs and their characteristics
SECTOR_ETFS = {
    'Technology': {'etf': 'XLK', 'category': 'growth', 'base_weight': 0.25},
    'Financials': {'etf': 'XLF', 'category': 'growth', 'base_weight': 0.15},
    'Industrials': {'etf': 'XLI', 'category': 'growth', 'base_weight': 0.10},
    'Healthcare': {'etf': 'XLV', 'category': 'defensive', 'base_weight': 0.10},
    'Consumer Staples': {'etf': 'XLP', 'category': 'defensive', 'base_weight': 0.10},
    'Utilities': {'etf': 'XLU', 'category': 'defensive', 'base_weight': 0.10},
    'Energy': {'etf': 'XLE', 'category': 'tactical', 'base_weight': 0.05},
    'Materials': {'etf': 'XLB', 'category': 'tactical', 'base_weight': 0.05},
    'Real Estate': {'etf': 'XLRE', 'category': 'tactical', 'base_weight': 0.05},
    'Communication Services': {'etf': 'XLC', 'category': 'tactical', 'base_weight': 0.05},
}

# Defensive sectors - prioritized in bear markets
DEFENSIVE_SECTORS = ['Utilities', 'Consumer Staples', 'Healthcare']

# Growth sectors - prioritized in bull markets
GROWTH_SECTORS = ['Technology', 'Financials', 'Industrials']

# Tactical sectors - momentum-based rotation
TACTICAL_SECTORS = ['Energy', 'Materials', 'Real Estate', 'Communication Services']


@dataclass
class SectorAllocation:
    """Target allocation for a sector."""
    sector: str
    etf: str
    category: str
    base_weight: float
    momentum_score: float
    adjusted_weight: float
    n_stocks: int


@dataclass
class SectorConfig:
    """Configuration for sector diversifier."""
    # Allocation limits
    max_sector_weight: float = 0.25       # No sector >25%
    min_sector_weight: float = 0.05       # Minimum 5% per sector
    defensive_target: float = 0.30        # 30% defensive allocation
    growth_target: float = 0.50           # 50% growth allocation
    tactical_target: float = 0.20         # 20% tactical rotation
    
    # Momentum calculation
    momentum_3m_weight: float = 0.40      # 3-month return weight
    momentum_6m_weight: float = 0.30      # 6-month return weight
    sharpe_weight: float = 0.20           # Volatility-adjusted weight
    relative_strength_weight: float = 0.10  # Relative vs SPY
    
    # Universe construction
    stocks_per_sector: int = 20           # Top 20 stocks per sector
    min_avg_volume: float = 5_000_000     # $5M daily volume minimum
    min_price: float = 5.0                # $5 minimum price
    
    # Tactical rotation
    top_tactical_sectors: int = 2         # Top 2 momentum sectors get allocation


class SectorMomentum:
    """Calculate sector momentum scores."""
    
    def __init__(self, config: Optional[SectorConfig] = None):
        self.config = config or SectorConfig()
    
    def calculate_momentum_score(
        self,
        prices: pd.DataFrame,
        spy_prices: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        Calculate momentum score for a sector ETF.
        
        Args:
            prices: OHLCV DataFrame for sector ETF
            spy_prices: SPY prices for relative strength
            
        Returns:
            Composite momentum score (0-100)
        """
        if prices is None or len(prices) < 126:  # Need 6 months of data
            return 0.0
        
        # Get close prices
        close = self._get_close(prices)
        
        # 3-month return (63 trading days)
        ret_3m = (close.iloc[-1] / close.iloc[-63] - 1) if len(close) >= 63 else 0
        
        # 6-month return (126 trading days)
        ret_6m = (close.iloc[-1] / close.iloc[-126] - 1) if len(close) >= 126 else 0
        
        # Volatility-adjusted return (Sharpe-like)
        returns = close.pct_change().dropna()
        if len(returns) >= 60:
            recent_returns = returns.iloc[-60:]
            sharpe = recent_returns.mean() / recent_returns.std() * np.sqrt(252)
        else:
            sharpe = 0
        
        # Relative strength vs SPY
        rel_strength = 0
        if spy_prices is not None and len(spy_prices) >= 63:
            spy_close = self._get_close(spy_prices)
            spy_ret = (spy_close.iloc[-1] / spy_close.iloc[-63] - 1)
            rel_strength = ret_3m - spy_ret
        
        # Normalize components to 0-100 scale
        ret_3m_score = min(100, max(0, (ret_3m + 0.3) / 0.6 * 100))
        ret_6m_score = min(100, max(0, (ret_6m + 0.4) / 0.8 * 100))
        sharpe_score = min(100, max(0, (sharpe + 2) / 4 * 100))
        rel_score = min(100, max(0, (rel_strength + 0.2) / 0.4 * 100))
        
        # Weighted composite
        score = (
            self.config.momentum_3m_weight * ret_3m_score +
            self.config.momentum_6m_weight * ret_6m_score +
            self.config.sharpe_weight * sharpe_score +
            self.config.relative_strength_weight * rel_score
        )
        
        return score
    
    def _get_close(self, df: pd.DataFrame) -> pd.Series:
        """Safely get close price series."""
        for col in ['close', 'Close', 'Adj Close']:
            if col in df.columns:
                return df[col]
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                return df['Close'].iloc[:, 0]
        return df.iloc[:, 3]  # Assume 4th column is close


class SectorDiversifier:
    """
    Strategic sector diversification system.
    
    Features:
    - Defensive core allocation (30%) for bear market protection
    - Growth engine (50%) for upside capture
    - Tactical rotation (20%) based on sector momentum
    - Dynamic rebalancing with sector limits
    """
    
    def __init__(self, config: Optional[SectorConfig] = None):
        self.config = config or SectorConfig()
        self.momentum_calculator = SectorMomentum(self.config)
        
        # Track allocations
        self.current_allocations: Dict[str, SectorAllocation] = {}
        self.sector_momentum_scores: Dict[str, float] = {}
        
        logger.info(f"Initialized SectorDiversifier: defensive={self.config.defensive_target:.0%}, "
                   f"growth={self.config.growth_target:.0%}, tactical={self.config.tactical_target:.0%}")
    
    def calculate_sector_allocations(
        self,
        sector_prices: Dict[str, pd.DataFrame],
        spy_prices: Optional[pd.DataFrame] = None,
        regime: str = 'neutral',
    ) -> Dict[str, SectorAllocation]:
        """
        Calculate optimal sector allocations.
        
        Args:
            sector_prices: Dict of {sector_name: price_df}
            spy_prices: SPY prices for relative strength
            regime: Current market regime ('bull', 'neutral', 'bear')
            
        Returns:
            Dict of sector allocations
        """
        allocations = {}
        
        # Calculate momentum scores for all sectors
        for sector, info in SECTOR_ETFS.items():
            etf = info['etf']
            if etf in sector_prices:
                score = self.momentum_calculator.calculate_momentum_score(
                    sector_prices[etf], spy_prices
                )
            else:
                score = 50  # Neutral score if no data
            self.sector_momentum_scores[sector] = score
        
        # Rank tactical sectors by momentum
        tactical_scores = [(s, self.sector_momentum_scores.get(s, 0)) 
                          for s in TACTICAL_SECTORS]
        tactical_scores.sort(key=lambda x: x[1], reverse=True)
        top_tactical = [s for s, _ in tactical_scores[:self.config.top_tactical_sectors]]
        
        # Calculate base allocations with regime adjustment
        regime_multipliers = self._get_regime_multipliers(regime)
        
        for sector, info in SECTOR_ETFS.items():
            base_weight = info['base_weight']
            category = info['category']
            
            # Apply regime multiplier
            multiplier = regime_multipliers.get(category, 1.0)
            adjusted_weight = base_weight * multiplier
            
            # Boost top tactical sectors
            if category == 'tactical':
                if sector in top_tactical:
                    adjusted_weight = self.config.tactical_target / 2  # Split between top 2
                else:
                    adjusted_weight = 0.02  # Minimal allocation for non-top tactical
            
            # Apply limits
            adjusted_weight = min(adjusted_weight, self.config.max_sector_weight)
            adjusted_weight = max(adjusted_weight, self.config.min_sector_weight)
            
            allocations[sector] = SectorAllocation(
                sector=sector,
                etf=info['etf'],
                category=category,
                base_weight=base_weight,
                momentum_score=self.sector_momentum_scores.get(sector, 0),
                adjusted_weight=adjusted_weight,
                n_stocks=self.config.stocks_per_sector,
            )
        
        # Normalize weights to sum to 1
        total_weight = sum(a.adjusted_weight for a in allocations.values())
        if total_weight > 0:
            for sector in allocations:
                allocations[sector].adjusted_weight /= total_weight
        
        self.current_allocations = allocations
        return allocations
    
    def _get_regime_multipliers(self, regime: str) -> Dict[str, float]:
        """Get category multipliers based on market regime."""
        if regime == 'bull':
            return {
                'growth': 1.2,     # Boost growth in bull markets
                'defensive': 0.8,  # Reduce defensive
                'tactical': 1.0,
            }
        elif regime == 'bear':
            return {
                'growth': 0.7,     # Reduce growth in bear markets
                'defensive': 1.4,  # Boost defensive
                'tactical': 0.9,
            }
        else:  # neutral
            return {
                'growth': 1.0,
                'defensive': 1.0,
                'tactical': 1.0,
            }
    
    def build_diversified_universe(
        self,
        stock_sectors: Dict[str, str],
        stock_scores: Dict[str, float],
        sector_allocations: Optional[Dict[str, SectorAllocation]] = None,
    ) -> Dict[str, float]:
        """
        Build diversified stock universe from sector allocations.
        
        Args:
            stock_sectors: Dict of {ticker: sector}
            stock_scores: Dict of {ticker: factor_score}
            sector_allocations: Sector allocation targets
            
        Returns:
            Dict of {ticker: weight}
        """
        if sector_allocations is None:
            sector_allocations = self.current_allocations
        
        portfolio_weights = {}
        
        for sector, allocation in sector_allocations.items():
            # Get stocks in this sector
            sector_stocks = [t for t, s in stock_sectors.items() if s == sector]
            
            if not sector_stocks:
                continue
            
            # Rank by score
            scored_stocks = [(t, stock_scores.get(t, 0)) for t in sector_stocks]
            scored_stocks.sort(key=lambda x: x[1], reverse=True)
            
            # Take top N stocks
            n_stocks = min(allocation.n_stocks, len(scored_stocks))
            selected = scored_stocks[:n_stocks]
            
            if selected:
                # Equal weight within sector
                stock_weight = allocation.adjusted_weight / len(selected)
                for ticker, score in selected:
                    portfolio_weights[ticker] = stock_weight
        
        # Normalize to sum to 1
        total = sum(portfolio_weights.values())
        if total > 0:
            portfolio_weights = {k: v/total for k, v in portfolio_weights.items()}
        
        return portfolio_weights
    
    def get_sector_exposure(
        self,
        portfolio_weights: Dict[str, float],
        stock_sectors: Dict[str, str],
    ) -> Dict[str, float]:
        """Calculate current sector exposures."""
        exposures = {}
        for ticker, weight in portfolio_weights.items():
            sector = stock_sectors.get(ticker, 'Other')
            exposures[sector] = exposures.get(sector, 0) + weight
        return exposures
    
    def check_sector_limits(
        self,
        portfolio_weights: Dict[str, float],
        stock_sectors: Dict[str, str],
    ) -> Tuple[bool, Dict[str, str]]:
        """
        Check if portfolio meets sector diversification limits.
        
        Returns:
            (is_compliant, violations_dict)
        """
        exposures = self.get_sector_exposure(portfolio_weights, stock_sectors)
        violations = {}
        
        for sector, exposure in exposures.items():
            if exposure > self.config.max_sector_weight:
                violations[sector] = f"Over limit: {exposure:.1%} > {self.config.max_sector_weight:.1%}"
        
        # Check defensive allocation
        defensive_total = sum(exposures.get(s, 0) for s in DEFENSIVE_SECTORS)
        if defensive_total < self.config.defensive_target * 0.8:  # 80% of target minimum
            violations['Defensive'] = f"Under target: {defensive_total:.1%} < {self.config.defensive_target:.1%}"
        
        return len(violations) == 0, violations
    
    def print_allocation_summary(self, allocations: Optional[Dict[str, SectorAllocation]] = None):
        """Print allocation summary."""
        if allocations is None:
            allocations = self.current_allocations
        
        print("\n" + "="*60)
        print("SECTOR ALLOCATION SUMMARY")
        print("="*60)
        
        # Group by category
        for category in ['defensive', 'growth', 'tactical']:
            cat_allocs = [a for a in allocations.values() if a.category == category]
            cat_total = sum(a.adjusted_weight for a in cat_allocs)
            
            print(f"\n{category.upper()} ({cat_total:.1%}):")
            for alloc in sorted(cat_allocs, key=lambda x: -x.adjusted_weight):
                print(f"  {alloc.sector:25s} {alloc.etf:5s} {alloc.adjusted_weight:6.1%} "
                      f"(momentum: {alloc.momentum_score:.0f})")
        
        print("\n" + "="*60)


def test_sector_diversifier():
    """Test sector diversification system."""
    print("\n" + "="*60)
    print("Testing Sector Diversifier")
    print("="*60)
    
    diversifier = SectorDiversifier()
    
    # Create mock sector ETF data
    import numpy as np
    dates = pd.date_range('2024-01-01', periods=252, freq='B')
    
    sector_prices = {}
    for sector, info in SECTOR_ETFS.items():
        # Simulate different sector behaviors
        if sector in DEFENSIVE_SECTORS:
            returns = np.random.normal(0.0003, 0.008, 252)  # Lower vol
        elif sector in GROWTH_SECTORS:
            returns = np.random.normal(0.0005, 0.015, 252)  # Higher vol, higher return
        else:
            returns = np.random.normal(0.0004, 0.012, 252)
        
        prices = 100 * np.cumprod(1 + returns)
        sector_prices[info['etf']] = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 252),
        }, index=dates)
    
    # Test neutral regime
    print("\n[Neutral Regime Allocation]")
    allocations = diversifier.calculate_sector_allocations(sector_prices, regime='neutral')
    diversifier.print_allocation_summary()
    
    # Test bear regime
    print("\n[Bear Regime Allocation]")
    allocations = diversifier.calculate_sector_allocations(sector_prices, regime='bear')
    diversifier.print_allocation_summary()
    
    # Test bull regime
    print("\n[Bull Regime Allocation]")
    allocations = diversifier.calculate_sector_allocations(sector_prices, regime='bull')
    diversifier.print_allocation_summary()


if __name__ == "__main__":
    test_sector_diversifier()

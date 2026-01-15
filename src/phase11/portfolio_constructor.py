"""
Portfolio Constructor for Phase 11
====================================

Constructs concentrated portfolios from factor rankings.

Features:
- Select top 30-50 stocks by composite score
- Position sizing based on score and liquidity
- Sector limits for diversification
- Integrate leveraged ETFs for amplification
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass 
class PortfolioConfig:
    """Configuration for portfolio construction."""
    n_stocks: int = 40                    # Number of stocks to hold
    max_position_weight: float = 0.05     # 5% max per stock
    min_position_weight: float = 0.015    # 1.5% min per stock
    max_sector_weight: float = 0.35       # 35% max per sector
    leverage_allocation: float = 0.25     # 25% to leveraged ETFs
    score_weighting: bool = True          # Weight by composite score
    equal_weight_fallback: bool = True    # Fall back to equal weight if issues


class PortfolioConstructor:
    """
    Constructs portfolios from factor rankings.
    
    Process:
    1. Take top N stocks by composite score
    2. Apply sector limits
    3. Weight by score (or equal weight)
    4. Reserve allocation for leveraged ETFs
    """
    
    def __init__(self, config: PortfolioConfig = None):
        self.config = config or PortfolioConfig()
        
        # Sector mapping (simplified)
        self.sector_map = {}
    
    def construct_portfolio(
        self,
        factor_df: pd.DataFrame,
        sector_map: Dict[str, str] = None,
        leveraged_weights: Dict[str, float] = None,
        current_drawdown: float = 0.0,
    ) -> Dict[str, float]:
        """
        Construct portfolio from factor rankings.
        
        Args:
            factor_df: DataFrame with factor scores (from FactorEngine)
            sector_map: Dict of ticker -> sector
            leveraged_weights: Pre-computed weights for leveraged ETFs
            current_drawdown: Current portfolio drawdown for risk adjustment
            
        Returns:
            Dict of ticker -> portfolio weight
        """
        if factor_df.empty:
            logger.warning("Empty factor DataFrame, returning empty portfolio")
            return {}
        
        self.sector_map = sector_map or {}
        
        # Step 1: Get top stocks (exclude leveraged ETFs, they're handled separately)
        stock_df = self._filter_stocks(factor_df)
        
        # Step 2: Select top N by composite score
        top_stocks = stock_df.head(self.config.n_stocks)
        
        if len(top_stocks) == 0:
            logger.warning("No stocks passed filter")
            return leveraged_weights or {}
        
        # Step 3: Apply sector limits
        diversified = self._apply_sector_limits(top_stocks)
        
        # Step 4: Compute weights
        stock_allocation = 1.0 - self.config.leverage_allocation
        
        # Adjust for drawdown
        if current_drawdown > 0.10:
            stock_allocation *= 0.7
        elif current_drawdown > 0.05:
            stock_allocation *= 0.85
        
        weights = self._compute_weights(diversified, stock_allocation)
        
        # Step 5: Add leveraged ETF weights
        if leveraged_weights:
            for ticker, weight in leveraged_weights.items():
                weights[ticker] = weight
        
        # Validate
        total = sum(weights.values())
        if total > 1.05:
            # Normalize
            weights = {k: v / total for k, v in weights.items()}
        
        logger.info(f"Constructed portfolio: {len(weights)} positions, "
                   f"{sum(weights.values()):.1%} exposure")
        
        return weights
    
    def _filter_stocks(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """Filter out leveraged ETFs and low-quality stocks."""
        leveraged = {'TQQQ', 'SPXL', 'UPRO', 'TNA', 'SOXL', 'TECL', 'FAS', 
                    'FNGU', 'LABU', 'QLD', 'SSO', 'UWM', 'ROM', 'USD',
                    'ERX', 'CURE', 'DPST', 'RETL', 'NAIL', 'DFEN'}
        
        etfs = {'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'XLK', 'XLF', 
                'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE',
                'SMH', 'IBB', 'XBI', 'IYR', 'VNQ'}
        
        # Exclude leveraged and sector ETFs (we want individual stocks)
        mask = ~factor_df['ticker'].isin(leveraged | etfs)
        
        return factor_df[mask].copy()
    
    def _apply_sector_limits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply sector concentration limits."""
        if not self.sector_map:
            return df
        
        # Add sector column
        df = df.copy()
        df['sector'] = df['ticker'].map(self.sector_map).fillna('Unknown')
        
        # Track sector counts
        sector_counts = {}
        max_per_sector = int(self.config.n_stocks * self.config.max_sector_weight / 
                            (1 / self.config.n_stocks))
        
        selected = []
        for _, row in df.iterrows():
            sector = row['sector']
            current_count = sector_counts.get(sector, 0)
            
            if current_count < max_per_sector:
                selected.append(row)
                sector_counts[sector] = current_count + 1
        
        return pd.DataFrame(selected)
    
    def _compute_weights(
        self,
        df: pd.DataFrame,
        total_allocation: float,
    ) -> Dict[str, float]:
        """Compute position weights based on composite scores."""
        if df.empty:
            return {}
        
        weights = {}
        
        if self.config.score_weighting and 'composite' in df.columns:
            # Score-weighted with concentration
            scores = df['composite'].values
            
            # Shift to positive (z-scores can be negative)
            shifted = scores - scores.min() + 0.1
            
            # Power transform for concentration (favor top picks)
            powered = shifted ** 1.3
            
            # Normalize
            total_score = powered.sum()
            
            for i, row in df.iterrows():
                ticker = row['ticker']
                idx = df.index.get_loc(i)
                raw_weight = (powered[idx] / total_score) * total_allocation
                
                # Apply limits
                weight = np.clip(raw_weight, 
                               self.config.min_position_weight,
                               self.config.max_position_weight)
                weights[ticker] = weight
        else:
            # Equal weight fallback
            n = len(df)
            weight = total_allocation / n
            weight = np.clip(weight, 
                           self.config.min_position_weight,
                           self.config.max_position_weight)
            
            for ticker in df['ticker']:
                weights[ticker] = weight
        
        # Normalize to target allocation
        current_total = sum(weights.values())
        if current_total > 0:
            scale = total_allocation / current_total
            weights = {k: v * scale for k, v in weights.items()}
        
        return weights
    
    def get_sector_exposure(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Get sector-level exposure from portfolio weights."""
        sector_weights = {}
        
        for ticker, weight in weights.items():
            sector = self.sector_map.get(ticker, 'Unknown')
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        return sector_weights

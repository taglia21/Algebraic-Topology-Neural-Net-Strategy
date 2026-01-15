"""
Sector Leverage Manager for Phase 11
======================================

Manages dynamic allocation to leveraged sector ETFs.

Strategy:
- Identify top-performing sectors from factor analysis
- Allocate to sector-specific 3x ETFs
- VIX-based risk overlay
- Regime detection for leverage timing
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LeverageConfig:
    """Configuration for leverage management."""
    max_leverage_allocation: float = 0.30  # 30% max to leveraged
    min_leverage_allocation: float = 0.10  # 10% min when conditions favorable
    vix_threshold_reduce: float = 22.0     # Reduce leverage when VIX > 22
    vix_threshold_exit: float = 30.0       # Minimal leverage when VIX > 30
    n_sectors: int = 3                      # Number of sector ETFs to hold
    momentum_lookback: int = 60            # Days for sector momentum


class SectorLeverageManager:
    """
    Manages leveraged ETF allocation based on sector momentum.
    
    Key principle: Concentrate leverage in the strongest sectors.
    """
    
    # Sector -> Leveraged ETF mapping
    SECTOR_ETF_MAP = {
        'Technology': 'TECL',        # 3x Technology
        'Semiconductors': 'SOXL',    # 3x Semiconductors  
        'Financials': 'FAS',         # 3x Financials
        'Healthcare': 'CURE',        # 3x Healthcare
        'Industrials': 'DUSL',       # 3x Industrials (if available)
        'Consumer': 'WANT',          # 3x Consumer (if available)
        'Energy': 'ERX',             # 3x Energy
        'Real Estate': 'DRN',        # 3x Real Estate
        'Broad Market': 'SPXL',      # 3x S&P 500
        'NASDAQ': 'TQQQ',            # 3x NASDAQ
        'Small Cap': 'TNA',          # 3x Small Cap
    }
    
    # Core leveraged ETFs to always consider
    CORE_LEVERAGED = ['TQQQ', 'SPXL', 'UPRO', 'SOXL']
    
    # Unleveraged sector ETFs for momentum calculation
    SECTOR_TRACKERS = {
        'Technology': 'XLK',
        'Semiconductors': 'SMH',
        'Financials': 'XLF',
        'Healthcare': 'XLV',
        'Industrials': 'XLI',
        'Consumer': 'XLY',
        'Energy': 'XLE',
        'Real Estate': 'XLRE',
        'Utilities': 'XLU',
        'Materials': 'XLB',
        'Communication': 'XLC',
    }
    
    def __init__(self, config: LeverageConfig = None):
        self.config = config or LeverageConfig()
    
    def compute_leverage_weights(
        self,
        price_data: Dict[str, pd.DataFrame],
        vix_level: float = 15.0,
        spy_trend: str = 'up',  # 'up', 'down', or 'neutral'
        current_drawdown: float = 0.0,
    ) -> Dict[str, float]:
        """
        Compute optimal leverage allocation.
        
        Args:
            price_data: Dict of ticker -> OHLCV DataFrame
            vix_level: Current VIX level
            spy_trend: SPY trend direction
            current_drawdown: Current portfolio drawdown
            
        Returns:
            Dict of leveraged ETF -> weight
        """
        # Step 1: Compute sector momentum
        sector_momentum = self._compute_sector_momentum(price_data)
        
        # Step 2: Determine base allocation
        base_allocation = self._get_base_allocation(vix_level, spy_trend, current_drawdown)
        
        if base_allocation < 0.05:
            logger.info("Low allocation regime, minimal leverage")
            return {'TQQQ': 0.05}  # Minimal position
        
        # Step 3: Rank sectors and select top N
        top_sectors = sorted(sector_momentum.items(), key=lambda x: x[1], reverse=True)
        top_sectors = top_sectors[:self.config.n_sectors]
        
        # Step 4: Allocate to sector ETFs
        weights = {}
        
        # Always include core (TQQQ) as primary
        weights['TQQQ'] = base_allocation * 0.50
        
        # Allocate rest to top sector ETFs
        remaining = base_allocation * 0.50
        if top_sectors:
            per_sector = remaining / len(top_sectors)
            
            for sector, _ in top_sectors:
                etf = self.SECTOR_ETF_MAP.get(sector)
                if etf and etf != 'TQQQ':
                    weights[etf] = per_sector
        
        # If not enough sectors, put rest in SPXL
        allocated = sum(weights.values())
        if allocated < base_allocation - 0.01:
            weights['SPXL'] = base_allocation - allocated
        
        logger.info(f"Leverage allocation: {base_allocation:.1%} total, "
                   f"VIX={vix_level:.1f}, sectors={[s for s, _ in top_sectors]}")
        
        return weights
    
    def _compute_sector_momentum(
        self, 
        price_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """Compute momentum for each sector."""
        momentum = {}
        
        for sector, etf in self.SECTOR_TRACKERS.items():
            if etf in price_data and len(price_data[etf]) > self.config.momentum_lookback:
                df = price_data[etf]
                prices = df['Close'].values if 'Close' in df.columns else df['close'].values
                
                # Momentum = return over lookback period
                if len(prices) >= self.config.momentum_lookback:
                    ret = prices[-1] / prices[-self.config.momentum_lookback] - 1
                    momentum[sector] = ret
        
        return momentum
    
    def _get_base_allocation(
        self,
        vix_level: float,
        spy_trend: str,
        current_drawdown: float,
    ) -> float:
        """Determine base leverage allocation."""
        base = self.config.max_leverage_allocation
        
        # VIX adjustment
        if vix_level > self.config.vix_threshold_exit:
            base *= 0.2  # 80% reduction
        elif vix_level > self.config.vix_threshold_reduce:
            base *= 0.5  # 50% reduction
        elif vix_level > 18:
            base *= 0.8  # 20% reduction
        
        # Trend adjustment
        if spy_trend == 'down':
            base *= 0.5
        elif spy_trend == 'neutral':
            base *= 0.75
        
        # Drawdown adjustment
        if current_drawdown > 0.15:
            base *= 0.3
        elif current_drawdown > 0.10:
            base *= 0.5
        elif current_drawdown > 0.05:
            base *= 0.75
        
        # Ensure within bounds
        base = max(self.config.min_leverage_allocation * 0.5, 
                  min(base, self.config.max_leverage_allocation))
        
        return base
    
    def get_spy_trend(
        self, 
        spy_prices: pd.Series,
        short_window: int = 20,
        long_window: int = 50,
    ) -> str:
        """Determine SPY trend using moving averages."""
        if len(spy_prices) < long_window:
            return 'neutral'
        
        sma_short = spy_prices.rolling(short_window).mean().iloc[-1]
        sma_long = spy_prices.rolling(long_window).mean().iloc[-1]
        current = spy_prices.iloc[-1]
        
        if current > sma_short > sma_long:
            return 'up'
        elif current < sma_short < sma_long:
            return 'down'
        else:
            return 'neutral'
    
    def get_vix_level(self, vix_prices: pd.Series) -> float:
        """Get current VIX level with smoothing."""
        if len(vix_prices) < 5:
            return 20.0  # Default to moderate
        
        # Use 5-day average for stability
        return vix_prices.rolling(5).mean().iloc[-1]

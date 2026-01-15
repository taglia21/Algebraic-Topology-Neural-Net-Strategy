"""
Inverse Allocator for Phase 12
==============================

Manages allocation to inverse leveraged ETFs during bear regimes:
- SQQQ: 3x inverse NASDAQ-100
- SPXU: 3x inverse S&P 500
- SOXS: 3x inverse Semiconductors

Key considerations:
- Volatility decay in leveraged products
- Daily rebalancing effects
- Regime transition timing
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .regime_classifier import RegimeState

logger = logging.getLogger(__name__)


@dataclass
class AllocationResult:
    """Result of allocation decision."""
    weights: Dict[str, float]
    direction: str  # 'long', 'short', 'neutral'
    total_exposure: float
    leverage_type: str  # 'long_3x', 'inverse_3x', 'none'
    regime: RegimeState
    confidence: float


class InverseAllocator:
    """
    Allocates to inverse ETFs during bear regimes.
    
    Long ETFs (bull regime): TQQQ, SPXL, SOXL
    Inverse ETFs (bear regime): SQQQ, SPXU, SOXS
    
    Allocation is scaled by regime confidence and VIX level.
    """
    
    # Long 3x ETFs
    LONG_ETFS = {
        'TQQQ': {'weight': 0.50, 'sector': 'nasdaq', 'leverage': 3},
        'SPXL': {'weight': 0.30, 'sector': 'sp500', 'leverage': 3},
        'SOXL': {'weight': 0.20, 'sector': 'semis', 'leverage': 3},
    }
    
    # Inverse 3x ETFs
    INVERSE_ETFS = {
        'SQQQ': {'weight': 0.50, 'sector': 'nasdaq', 'leverage': -3},
        'SPXU': {'weight': 0.30, 'sector': 'sp500', 'leverage': -3},
        'SOXS': {'weight': 0.20, 'sector': 'semis', 'leverage': -3},
    }
    
    def __init__(
        self,
        max_leverage_allocation: float = 0.65,
        min_leverage_allocation: float = 0.10,
        vix_scale_threshold: float = 25.0,
        vix_crisis_threshold: float = 40.0,
    ):
        self.max_leverage_allocation = max_leverage_allocation
        self.min_leverage_allocation = min_leverage_allocation
        self.vix_scale_threshold = vix_scale_threshold
        self.vix_crisis_threshold = vix_crisis_threshold
    
    def allocate(
        self,
        regime: RegimeState,
        regime_confidence: float,
        vix_level: float = 18.0,
        current_drawdown: float = 0.0,
        sector_momentum: Dict[str, float] = None,
    ) -> AllocationResult:
        """
        Determine leveraged ETF allocation based on regime.
        
        Args:
            regime: Current market regime
            regime_confidence: Confidence in regime (0-1)
            vix_level: Current VIX level
            current_drawdown: Current portfolio drawdown
            sector_momentum: Optional sector momentum for tilting
            
        Returns:
            AllocationResult with weights and metadata
        """
        # Determine direction
        if regime in [RegimeState.STRONG_BULL, RegimeState.MILD_BULL]:
            direction = 'long'
            etf_map = self.LONG_ETFS
            leverage_type = 'long_3x'
        elif regime in [RegimeState.STRONG_BEAR, RegimeState.MILD_BEAR]:
            direction = 'short'
            etf_map = self.INVERSE_ETFS
            leverage_type = 'inverse_3x'
        else:  # NEUTRAL
            return AllocationResult(
                weights={},
                direction='neutral',
                total_exposure=0.0,
                leverage_type='none',
                regime=regime,
                confidence=regime_confidence,
            )
        
        # Calculate base allocation
        base_allocation = self._compute_base_allocation(
            regime, regime_confidence, vix_level, current_drawdown
        )
        
        # Apply sector tilting if momentum data available
        weights = self._apply_sector_tilt(etf_map, sector_momentum)
        
        # Scale to base allocation
        weights = {ticker: w * base_allocation for ticker, w in weights.items()}
        
        return AllocationResult(
            weights=weights,
            direction=direction,
            total_exposure=base_allocation,
            leverage_type=leverage_type,
            regime=regime,
            confidence=regime_confidence,
        )
    
    def _compute_base_allocation(
        self,
        regime: RegimeState,
        confidence: float,
        vix_level: float,
        current_drawdown: float,
    ) -> float:
        """
        Compute base allocation to leveraged ETFs.
        """
        # Start with regime-based allocation
        regime_multipliers = {
            RegimeState.STRONG_BULL: 1.0,
            RegimeState.MILD_BULL: 0.70,
            RegimeState.NEUTRAL: 0.0,
            RegimeState.MILD_BEAR: 0.70,
            RegimeState.STRONG_BEAR: 1.0,
        }
        
        base = self.max_leverage_allocation * regime_multipliers.get(regime, 0.0)
        
        # Scale by confidence
        base *= (0.5 + 0.5 * confidence)  # 50-100% based on confidence
        
        # VIX scaling (reduce in high vol)
        if vix_level > self.vix_crisis_threshold:
            base *= 0.30  # 70% reduction in crisis
        elif vix_level > self.vix_scale_threshold:
            scale = 1.0 - (vix_level - self.vix_scale_threshold) / 25
            base *= max(0.50, scale)
        
        # Drawdown protection
        if current_drawdown > 0.15:
            base *= 0.40
        elif current_drawdown > 0.10:
            base *= 0.60
        elif current_drawdown > 0.05:
            base *= 0.80
        
        # Ensure within bounds
        base = max(self.min_leverage_allocation, min(base, self.max_leverage_allocation))
        
        return base
    
    def _apply_sector_tilt(
        self,
        etf_map: Dict,
        sector_momentum: Dict[str, float] = None,
    ) -> Dict[str, float]:
        """
        Apply sector momentum tilting to weights.
        """
        weights = {ticker: info['weight'] for ticker, info in etf_map.items()}
        
        if not sector_momentum:
            return weights
        
        # Map sectors to ETFs
        sector_etf = {
            'nasdaq': list(etf_map.keys())[0],  # First ETF
            'sp500': list(etf_map.keys())[1],   # Second ETF  
            'semis': list(etf_map.keys())[2],   # Third ETF
        }
        
        # Tilt based on momentum (subtle, max ±20%)
        for sector, etf in sector_etf.items():
            if sector in sector_momentum and etf in weights:
                mom = sector_momentum.get(sector, 0)
                tilt = np.clip(mom * 0.5, -0.20, 0.20)
                weights[etf] *= (1 + tilt)
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def get_all_tickers(self) -> List[str]:
        """Get all ETF tickers (long and inverse)."""
        return list(self.LONG_ETFS.keys()) + list(self.INVERSE_ETFS.keys())
    
    def get_long_tickers(self) -> List[str]:
        """Get long ETF tickers."""
        return list(self.LONG_ETFS.keys())
    
    def get_inverse_tickers(self) -> List[str]:
        """Get inverse ETF tickers."""
        return list(self.INVERSE_ETFS.keys())

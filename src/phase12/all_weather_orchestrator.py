"""
All-Weather Orchestrator for Phase 12
======================================

Main coordinator that integrates:
- Regime classification
- Directional allocation (long vs inverse)
- Adaptive risk management
- Execution signals

Provides unified interface for strategy execution.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .regime_classifier import RegimeClassifier, RegimeState, RegimeConfig, RegimeSignals
from .inverse_allocator import InverseAllocator, AllocationResult
from .adaptive_risk_manager import AdaptiveRiskManager, RiskConfig, RiskMetrics, RiskState

logger = logging.getLogger(__name__)


@dataclass
class Phase12Config:
    """Configuration for Phase 12 strategy."""
    # Regime classification
    regime_config: RegimeConfig = None
    
    # Risk management
    risk_config: RiskConfig = None
    
    # Allocation parameters
    max_leverage_allocation: float = 0.65
    min_leverage_allocation: float = 0.10
    
    # Market data parameters
    spy_ticker: str = 'SPY'
    vix_ticker: str = 'VIX'
    
    # Timing
    rebalance_frequency: str = 'weekly'  # 'daily', 'weekly', 'monthly'
    
    def __post_init__(self):
        if self.regime_config is None:
            self.regime_config = RegimeConfig()
        if self.risk_config is None:
            self.risk_config = RiskConfig()


@dataclass
class StrategySignal:
    """Output signal from the orchestrator."""
    date: datetime
    regime: RegimeState
    regime_signals: RegimeSignals
    allocation: AllocationResult
    risk_metrics: RiskMetrics
    final_weights: Dict[str, float]
    action: str  # 'hold', 'rebalance', 'exit'
    notes: List[str]


class AllWeatherOrchestrator:
    """
    Main orchestrator for the Phase 12 all-weather strategy.
    
    Coordinates regime detection, allocation, and risk management
    to produce actionable portfolio weights.
    """
    
    def __init__(self, config: Phase12Config = None):
        self.config = config or Phase12Config()
        
        # Initialize components
        self.regime_classifier = RegimeClassifier(self.config.regime_config)
        self.allocator = InverseAllocator(
            max_leverage_allocation=self.config.max_leverage_allocation,
            min_leverage_allocation=self.config.min_leverage_allocation,
        )
        self.risk_manager = AdaptiveRiskManager(self.config.risk_config)
        
        # State tracking
        self.last_signal: Optional[StrategySignal] = None
        self.last_rebalance_date: Optional[datetime] = None
        self.signal_history: List[StrategySignal] = []
        
    def generate_signal(
        self,
        market_data: pd.DataFrame,
        portfolio_return: float = 0.0,
        current_date: datetime = None,
        force_rebalance: bool = False,
    ) -> StrategySignal:
        """
        Generate portfolio signal for the current period.
        
        Args:
            market_data: DataFrame with OHLCV data for SPY and VIX
            portfolio_return: Previous period portfolio return
            current_date: Current date for signal
            force_rebalance: Force rebalance regardless of timing
            
        Returns:
            StrategySignal with weights and metadata
        """
        if current_date is None:
            current_date = market_data.index[-1] if isinstance(market_data.index, pd.DatetimeIndex) else datetime.now()
        
        notes = []
        
        # Extract market data
        spy_data = self._extract_spy_data(market_data)
        vix_level = self._get_vix_level(market_data)
        
        # Classify regime
        regime, regime_signals = self.regime_classifier.classify(
            spy_data, date=current_date
        )
        notes.append(f"Regime: {regime.value}")
        
        # Update risk manager
        risk_metrics = self.risk_manager.update(
            daily_return=portfolio_return,
            vix_level=vix_level,
            regime=regime,
            date=current_date,
        )
        notes.append(f"Risk state: {risk_metrics.risk_state.value}")
        
        # Check for emergency exit
        if self.risk_manager.should_exit_all():
            notes.append("EMERGENCY EXIT - all positions closed")
            return StrategySignal(
                date=current_date,
                regime=regime,
                regime_signals=regime_signals,
                allocation=AllocationResult({}, 'neutral', 0.0, 'none', regime, 0.0),
                risk_metrics=risk_metrics,
                final_weights={},
                action='exit',
                notes=notes,
            )
        
        # Determine if we should rebalance
        should_rebalance = self._should_rebalance(current_date) or force_rebalance
        
        # Get allocation from inverse allocator
        sector_momentum = self._compute_sector_momentum(market_data)
        allocation = self.allocator.allocate(
            regime=regime,
            regime_confidence=regime_signals.confidence if regime_signals else 0.5,
            vix_level=vix_level,
            current_drawdown=risk_metrics.portfolio_drawdown,
            sector_momentum=sector_momentum,
        )
        notes.append(f"Direction: {allocation.direction}, Exposure: {allocation.total_exposure:.1%}")
        
        # Apply risk scaling
        risk_scale = self.risk_manager.get_allocation_multiplier()
        final_weights = {
            ticker: weight * risk_scale 
            for ticker, weight in allocation.weights.items()
        }
        notes.append(f"Risk scale: {risk_scale:.1%}")
        
        # Determine action
        if should_rebalance:
            action = 'rebalance'
            self.last_rebalance_date = current_date
        else:
            action = 'hold'
        
        signal = StrategySignal(
            date=current_date,
            regime=regime,
            regime_signals=regime_signals,
            allocation=allocation,
            risk_metrics=risk_metrics,
            final_weights=final_weights,
            action=action,
            notes=notes,
        )
        
        self.last_signal = signal
        self.signal_history.append(signal)
        
        return signal
    
    def _extract_spy_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Extract SPY data from market data."""
        # Handle different data formats
        if 'close' in market_data.columns or 'Close' in market_data.columns:
            return market_data
        
        # Multi-asset format
        spy_ticker = self.config.spy_ticker
        if spy_ticker in market_data.columns:
            return market_data[[spy_ticker]].rename(columns={spy_ticker: 'close'})
        
        return market_data
    
    def _get_vix_level(self, market_data: pd.DataFrame) -> float:
        """Get current VIX level from data."""
        vix_ticker = self.config.vix_ticker
        
        # Check for VIX column
        if vix_ticker in market_data.columns:
            return market_data[vix_ticker].iloc[-1]
        
        if 'VIX' in market_data.columns:
            return market_data['VIX'].iloc[-1]
        
        # Default VIX level
        return 18.0
    
    def _compute_sector_momentum(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Compute sector momentum for allocation tilting."""
        # Simplified sector momentum
        # In production, would use sector ETFs (XLK, XLF, etc.)
        return {
            'nasdaq': 0.0,
            'sp500': 0.0,
            'semis': 0.0,
        }
    
    def _should_rebalance(self, current_date: datetime) -> bool:
        """Check if we should rebalance based on frequency."""
        if self.last_rebalance_date is None:
            return True
        
        days_since = (current_date - self.last_rebalance_date).days
        
        if self.config.rebalance_frequency == 'daily':
            return days_since >= 1
        elif self.config.rebalance_frequency == 'weekly':
            return days_since >= 5
        elif self.config.rebalance_frequency == 'monthly':
            return days_since >= 20
        
        return days_since >= 5  # Default weekly
    
    def get_current_regime(self) -> Optional[RegimeState]:
        """Get the current regime."""
        return self.regime_classifier.current_regime
    
    def get_regime_history(self) -> List[Tuple[datetime, RegimeState]]:
        """Get history of regime changes."""
        return [
            (signal.date, signal.regime) 
            for signal in self.signal_history
        ]
    
    def reset(self):
        """Reset orchestrator state."""
        self.regime_classifier.reset()
        self.risk_manager.reset()
        self.last_signal = None
        self.last_rebalance_date = None
        self.signal_history = []
    
    def get_all_tickers(self) -> List[str]:
        """Get all tickers used by the strategy."""
        return self.allocator.get_all_tickers()

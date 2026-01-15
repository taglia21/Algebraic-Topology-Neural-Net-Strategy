"""Phase 10 Orchestrator: Aggressive Alpha Amplification.

Integrates Phase 9 components with dynamic leverage for 25-35% CAGR target.

Key Enhancements:
- Dynamic leverage (0.5x - 1.5x) based on regime and drawdown
- Kelly-optimal position sizing  
- Leveraged ETF integration for aggressive phases
- Higher concentration in top signals
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

# Phase 9 imports
from src.phase9 import (
    Phase9Orchestrator,
    Phase9Config,
    HierarchicalRegimeStrategy,
    AdvancedAlphaEngine,
    AdaptiveUniverseScreener,
    DynamicPositionOptimizer,
)
from src.phase9.phase9_orchestrator import DailyState
from src.phase9.regime_meta_strategy import MacroRegime, TDARegime, RegimeMeta
from src.phase9.dynamic_optimizer import PortfolioState

# Phase 10 imports
from .dynamic_leverage import (
    DynamicLeverageEngine,
    LeverageState,
    LeverageRegime,
    KellyConfig,
    RegimeScaleConfig,
    AdjusterConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class Phase10Config:
    """Configuration for Phase 10 aggressive alpha amplification."""
    # Leverage settings
    max_leverage: float = 1.5           # Maximum leverage
    min_leverage: float = 0.5           # Minimum leverage (de-lever)
    kelly_fraction: float = 0.45        # 45% fractional Kelly
    
    # Regime-based leverage multipliers
    bull_leverage: float = 1.45         # Bull momentum leverage
    moderate_leverage: float = 1.25     # Normal conditions
    defensive_leverage: float = 0.70    # Bear/risk-off
    
    # Position concentration
    max_positions: int = 15             # Concentrated portfolio
    max_position_weight: float = 0.15   # 15% max single position
    min_signal_threshold: float = 0.05  # Minimum signal for inclusion
    
    # Drawdown controls
    dd_start_reduction: float = 0.06    # Start reducing at 6% DD
    dd_aggressive_reduction: float = 0.12  # Aggressive reduction at 12%
    max_acceptable_dd: float = 0.22     # Hard stop at 22% DD
    
    # Risk parameters
    target_volatility: float = 0.22     # Higher vol target for higher returns
    vix_high_threshold: float = 25.0    # Reduce leverage above this
    vix_crisis_threshold: float = 30.0  # Crisis mode above this
    
    # Rebalance
    rebalance_frequency: int = 2        # Every 2 days
    
    # Capital
    initial_capital: float = 100000.0
    
    # Phase 9 base config
    momentum_weight: float = 0.60       # Heavier momentum weight
    tda_weight: float = 0.20
    reversal_weight: float = 0.05       # Less reversal (trend-following)
    cross_sectional_weight: float = 0.15


@dataclass
class Phase10State:
    """Complete state for Phase 10 daily processing."""
    date: str
    
    # Leverage state
    leverage_state: Optional[LeverageState] = None
    target_leverage: float = 1.0
    actual_leverage: float = 1.0
    
    # Phase 9 state
    phase9_state: Optional[DailyState] = None
    
    # Combined signals (leverage-adjusted)
    adjusted_signals: Dict[str, float] = field(default_factory=dict)
    adjusted_weights: Dict[str, float] = field(default_factory=dict)
    
    # Portfolio state
    portfolio_value: float = 100000.0
    peak_value: float = 100000.0
    current_drawdown: float = 0.0
    
    # Performance
    daily_return: float = 0.0
    cumulative_return: float = 0.0


class Phase10Orchestrator:
    """
    Phase 10 Orchestrator: Aggressive Alpha Amplification.
    
    Wraps Phase 9 with dynamic leverage layer for 25-35% CAGR.
    """
    
    def __init__(self, config: Optional[Phase10Config] = None):
        self.config = config or Phase10Config()
        
        # Initialize Phase 9 base
        phase9_config = Phase9Config(
            target_universe_size=self.config.max_positions,
            min_universe_size=10,
            momentum_weight=self.config.momentum_weight,
            tda_weight=self.config.tda_weight,
            reversal_weight=self.config.reversal_weight,
            cross_sectional_weight=self.config.cross_sectional_weight,
            kelly_fraction=self.config.kelly_fraction,
            target_volatility=self.config.target_volatility,
            max_position=self.config.max_position_weight,
            initial_capital=self.config.initial_capital,
        )
        self.phase9 = Phase9Orchestrator(phase9_config)
        
        # Initialize leverage engine
        kelly_config = KellyConfig(
            kelly_fraction=self.config.kelly_fraction,
            min_sample_size=40,  # Faster startup
            lookback_days=180,
        )
        
        regime_config = RegimeScaleConfig(
            aggressive_leverage=self.config.bull_leverage,
            moderate_leverage=self.config.moderate_leverage,
            defensive_leverage=self.config.defensive_leverage,
            vix_high=self.config.vix_high_threshold,
            vix_crisis=self.config.vix_crisis_threshold,
        )
        
        adjuster_config = AdjusterConfig(
            dd_start_threshold=self.config.dd_start_reduction,
            dd_reduction_rate=2.5,  # Aggressive reduction
            dd_min_leverage=self.config.min_leverage,
        )
        
        self.leverage_engine = DynamicLeverageEngine(
            kelly_config=kelly_config,
            regime_config=regime_config,
            adjuster_config=adjuster_config,
            max_leverage=self.config.max_leverage,
            min_leverage=self.config.min_leverage,
        )
        
        # State tracking
        self.returns_history = []
        self.portfolio_value = self.config.initial_capital
        self.peak_value = self.config.initial_capital
        self.current_state: Optional[Phase10State] = None
        
    def process_day(
        self,
        date: str,
        spy_prices: pd.DataFrame,
        universe_prices: Dict[str, np.ndarray],
        universe_volumes: Dict[str, np.ndarray],
        sector_map: Dict[str, str],
        vix_level: float = 15.0,
        tda_data: Optional[Dict[str, pd.DataFrame]] = None,
        current_portfolio: Optional[PortfolioState] = None,
    ) -> Phase10State:
        """
        Process a single trading day with leverage overlay.
        
        Args:
            date: Current date string
            spy_prices: SPY price DataFrame
            universe_prices: {ticker: price_array}
            universe_volumes: {ticker: volume_array}
            sector_map: {ticker: sector}
            vix_level: Current VIX level
            tda_data: TDA features by ticker
            current_portfolio: Current portfolio state
            
        Returns:
            Phase10State with leverage-adjusted recommendations
        """
        # Get portfolio value from current_portfolio or use tracked
        if current_portfolio:
            self.portfolio_value = current_portfolio.total_value
        
        # Update peak value
        self.peak_value = max(self.peak_value, self.portfolio_value)
        
        # Get returns history as numpy array
        returns_array = np.array(self.returns_history) if self.returns_history else np.array([0.0])
        
        # Step 1: Run Phase 9 processing
        phase9_state = self.phase9.process_day(
            date=date,
            spy_prices=spy_prices,
            universe_prices=universe_prices,
            universe_volumes=universe_volumes,
            sector_map=sector_map,
            vix_data=None,  # We use vix_level directly
            tda_data=tda_data,
            current_portfolio=current_portfolio,
        )
        
        # Get regime info
        macro_regime = phase9_state.regime_meta.macro_regime.name if phase9_state.regime_meta else 'NEUTRAL'
        tda_regime = phase9_state.regime_meta.tda_regime.name if phase9_state.regime_meta else None
        
        # Calculate aggregate momentum score
        momentum_score = self._compute_aggregate_momentum(phase9_state.signals)
        
        # Count high-confidence signals
        signal_count, avg_confidence = self._count_strong_signals(phase9_state.signals)
        
        # Step 2: Compute leverage
        leverage_state = self.leverage_engine.compute_target_leverage(
            date=date,
            portfolio_value=self.portfolio_value,
            peak_value=self.peak_value,
            returns_history=returns_array,
            macro_regime=macro_regime,
            vix_level=vix_level,
            momentum_score=momentum_score,
            signal_count=signal_count,
            avg_confidence=avg_confidence,
            tda_regime=tda_regime,
        )
        
        # Step 3: Apply leverage to signals
        adjusted_signals = self._apply_leverage_to_signals(
            phase9_state.signals,
            leverage_state.actual_leverage,
        )
        
        # Step 4: Compute concentrated weights
        adjusted_weights = self._compute_concentrated_weights(
            adjusted_signals,
            leverage_state.actual_leverage,
        )
        
        # Step 5: Apply drawdown protection
        current_dd = 1 - (self.portfolio_value / self.peak_value) if self.peak_value > 0 else 0
        adjusted_weights = self._apply_drawdown_protection(
            adjusted_weights,
            current_dd,
        )
        
        # Build state
        state = Phase10State(
            date=date,
            leverage_state=leverage_state,
            target_leverage=leverage_state.target_leverage,
            actual_leverage=leverage_state.actual_leverage,
            phase9_state=phase9_state,
            adjusted_signals=adjusted_signals,
            adjusted_weights=adjusted_weights,
            portfolio_value=self.portfolio_value,
            peak_value=self.peak_value,
            current_drawdown=current_dd,
        )
        
        self.current_state = state
        return state
    
    def update_portfolio_value(self, new_value: float):
        """Update tracked portfolio value and returns history."""
        if self.portfolio_value > 0:
            daily_return = new_value / self.portfolio_value - 1
            self.returns_history.append(daily_return)
        self.portfolio_value = new_value
        self.peak_value = max(self.peak_value, new_value)
    
    def _compute_aggregate_momentum(self, signals: Dict[str, float]) -> float:
        """Compute aggregate momentum score from signals."""
        if not signals:
            return 0.0
        
        values = list(signals.values())
        positive = [v for v in values if v > 0]
        
        if not positive:
            return -0.3 if values else 0.0
        
        # Weighted average favoring strong signals
        weights = [v for v in positive]
        weighted_avg = sum(v * w for v, w in zip(positive, weights)) / sum(weights)
        
        return np.clip(weighted_avg * 2, -1, 1)
    
    def _count_strong_signals(self, signals: Dict[str, float]) -> Tuple[int, float]:
        """Count high-confidence signals."""
        if not signals:
            return 0, 0.0
        
        strong_signals = [v for v in signals.values() if v > 0.1]
        count = len(strong_signals)
        avg_confidence = np.mean(strong_signals) if strong_signals else 0.0
        
        return count, avg_confidence
    
    def _apply_leverage_to_signals(
        self,
        signals: Dict[str, float],
        leverage: float,
    ) -> Dict[str, float]:
        """
        Apply leverage to signals.
        
        Higher leverage → boost strong signals, keep weak ones same.
        """
        if not signals:
            return {}
        
        adjusted = {}
        for ticker, signal in signals.items():
            if signal > 0.1:
                # Boost strong positive signals with leverage
                adjusted[ticker] = signal * (0.8 + 0.4 * leverage)
            elif signal > 0:
                # Slight boost for weak positive
                adjusted[ticker] = signal * (0.9 + 0.2 * leverage)
            else:
                # Keep negative signals unchanged
                adjusted[ticker] = signal
        
        return adjusted
    
    def _compute_concentrated_weights(
        self,
        signals: Dict[str, float],
        leverage: float,
    ) -> Dict[str, float]:
        """
        Compute concentrated position weights.
        
        Focus on top signals with higher weights.
        For aggressive alpha, we use leverage to boost position weights.
        """
        if not signals:
            return {}
        
        # Filter and sort
        positive = {k: v for k, v in signals.items() 
                   if v > self.config.min_signal_threshold}
        
        if not positive:
            return {}
        
        # Sort by signal strength
        sorted_signals = sorted(positive.items(), key=lambda x: x[1], reverse=True)
        
        # Take top N positions - fewer for concentration
        n_positions = min(self.config.max_positions, len(sorted_signals))
        top_signals = dict(sorted_signals[:n_positions])
        
        # Normalize to sum to 1.0 (fully invested)
        total = sum(top_signals.values())
        if total <= 0:
            return {}
        
        # For aggressive returns, be fully invested (100%)
        # Leverage effect comes from position concentration and signal strength
        target_exposure = 0.98  # 98% invested
        
        weights = {}
        for ticker, signal in top_signals.items():
            weight = (signal / total) * target_exposure
            # Apply position limit
            weight = min(weight, self.config.max_position_weight)
            weights[ticker] = weight
        
        # Re-normalize if we hit position limits
        total_weight = sum(weights.values())
        if total_weight > target_exposure:
            scale = target_exposure / total_weight
            weights = {k: v * scale for k, v in weights.items()}
        
        return weights
    
    def _apply_drawdown_protection(
        self,
        weights: Dict[str, float],
        current_dd: float,
    ) -> Dict[str, float]:
        """
        Apply drawdown-based position reduction.
        More permissive for aggressive alpha targeting.
        """
        # For aggressive returns, only start reducing at higher drawdowns
        if current_dd <= 0.10:  # No reduction until 10% DD
            return weights
        
        if current_dd >= 0.22:  # Max acceptable DD
            # Emergency - reduce to minimum
            scale = 0.3
        elif current_dd >= 0.16:
            # Moderate reduction above 16%
            excess = current_dd - 0.16
            scale = max(0.4, 0.8 - excess * 4)
        else:
            # Gradual reduction from 10-16%
            excess = current_dd - 0.10
            scale = max(0.6, 1.0 - excess * 2.5)
        
        return {k: v * scale for k, v in weights.items()}
    
    def get_leverage_summary(self) -> Dict:
        """Get current leverage state summary."""
        return self.leverage_engine.get_leverage_summary()
    
    def get_performance_summary(self) -> Dict:
        """Get overall performance summary."""
        if not self.returns_history:
            return {}
        
        returns = np.array(self.returns_history)
        total_return = (self.portfolio_value / self.config.initial_capital) - 1
        
        return {
            'portfolio_value': self.portfolio_value,
            'peak_value': self.peak_value,
            'total_return': total_return,
            'current_drawdown': 1 - (self.portfolio_value / self.peak_value),
            'days_traded': len(self.returns_history),
            'avg_daily_return': np.mean(returns),
            'daily_volatility': np.std(returns),
            'current_leverage': self.current_state.actual_leverage if self.current_state else 1.0,
        }

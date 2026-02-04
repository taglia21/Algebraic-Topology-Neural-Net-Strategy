"""
Dynamic Strategy Weight Optimizer
==================================

Bridgewater-style dynamic capital allocation across multiple strategies.
Continuously adjusts weights based on:
- Recent strategy performance (Sharpe ratios)
- Current market regime
- Bayesian belief updating
- Regime-specific constraints

Features:
- Rolling Sharpe ratio calculation
- Quadratic programming for weight optimization
- Bayesian prior/posterior updating
- Automatic rebalancing triggers
- Regime-aware weight bounds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.optimize import minimize
import logging


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class StrategyPerformance:
    """Strategy performance metrics."""
    strategy_name: str
    sharpe_ratio: float
    cumulative_return: float
    volatility: float
    win_rate: float
    avg_trade_return: float
    total_trades: int
    last_updated: datetime


@dataclass
class WeightRecommendation:
    """Weight allocation recommendation."""
    strategy_weights: Dict[str, float]
    regime_adjusted: bool
    sharpe_adjusted: bool
    rebalance_required: bool
    expected_portfolio_sharpe: float
    rationale: str
    timestamp: datetime


# ============================================================================
# DYNAMIC WEIGHT OPTIMIZER
# ============================================================================

class DynamicWeightOptimizer:
    """
    Dynamically allocate capital across strategies.
    
    Optimization approach:
    1. Calculate rolling Sharpe ratios for each strategy
    2. Get regime-specific weight bounds
    3. Solve quadratic program to maximize expected Sharpe
    4. Apply Bayesian updates as new data arrives
    5. Rebalance when weights drift > threshold
    """
    
    # Strategy list
    STRATEGIES = ["iv_rank", "theta_decay", "mean_reversion", "delta_hedging"]
    
    # Default weight bounds (can be overridden by regime)
    DEFAULT_BOUNDS = {
        "iv_rank": (0.15, 0.50),
        "theta_decay": (0.10, 0.45),
        "mean_reversion": (0.10, 0.35),
        "delta_hedging": (0.05, 0.40),
    }
    
    # Rebalancing thresholds
    MIN_WEIGHT_CHANGE = 0.05  # 5% minimum change to trigger rebalance
    REBALANCE_COOLDOWN_DAYS = 1  # Minimum 1 day between rebalances
    
    def __init__(
        self,
        strategies: List[str],
        regime_detector,  # RegimeDetector instance
        sharpe_window: int = 20,
        min_trades_for_sharpe: int = 10,
    ):
        """
        Initialize weight optimizer.
        
        Args:
            strategies: List of strategy names
            regime_detector: RegimeDetector instance for regime-based adjustments
            sharpe_window: Rolling window for Sharpe calculation (default 20 days)
            min_trades_for_sharpe: Minimum trades before using Sharpe (default 10)
        """
        self.strategies = strategies
        self.regime_detector = regime_detector
        self.sharpe_window = sharpe_window
        self.min_trades_for_sharpe = min_trades_for_sharpe
        self.logger = logging.getLogger(__name__)
        
        # Current weights
        self.current_weights: Dict[str, float] = {
            s: 1.0 / len(strategies) for s in strategies
        }
        
        # Performance tracking
        self.strategy_returns: Dict[str, List[float]] = {s: [] for s in strategies}
        self.strategy_trades: Dict[str, int] = {s: 0 for s in strategies}
        
        # Bayesian priors (start with equal belief)
        self.bayesian_priors: Dict[str, float] = {
            s: 1.0 / len(strategies) for s in strategies
        }
        
        # Rebalancing state
        self.last_rebalance: Optional[datetime] = None
        self.rebalance_count = 0
        
        self.logger.info(
            f"Initialized DynamicWeightOptimizer "
            f"(strategies={len(strategies)}, sharpe_window={sharpe_window})"
        )
    
    async def calculate_strategy_sharpe(
        self, 
        window: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate Sharpe ratio for each strategy.
        
        Args:
            window: Lookback window (default uses self.sharpe_window)
        
        Returns:
            Dict mapping strategy name to Sharpe ratio
        """
        if window is None:
            window = self.sharpe_window
        
        sharpe_ratios: Dict[str, float] = {}
        
        for strategy in self.strategies:
            returns = self.strategy_returns.get(strategy, [])
            
            # Need minimum number of trades
            if len(returns) < self.min_trades_for_sharpe:
                # Default to zero Sharpe if insufficient data
                sharpe_ratios[strategy] = 0.0
                continue
            
            # Use recent window
            recent_returns = returns[-window:] if len(returns) > window else returns
            
            # Calculate Sharpe
            if len(recent_returns) > 1:
                mean_return = np.mean(recent_returns)
                std_return = np.std(recent_returns)
                
                if std_return > 1e-8:
                    # Annualized Sharpe (assuming daily returns)
                    sharpe = (mean_return / std_return) * np.sqrt(252)
                else:
                    sharpe = 0.0
            else:
                sharpe = 0.0
            
            sharpe_ratios[strategy] = sharpe
        
        self.logger.info("Strategy Sharpe ratios:")
        for strategy, sharpe in sharpe_ratios.items():
            self.logger.info(f"  {strategy}: {sharpe:.2f}")
        
        return sharpe_ratios
    
    def optimize_weights(
        self,
        regime,  # MarketRegime enum
        sharpe_ratios: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Optimize strategy weights given regime and performance.
        
        Uses quadratic programming to maximize expected Sharpe subject to:
        - Sum of weights = 1.0
        - Regime-specific weight bounds
        - Sharpe-based ranking
        
        Args:
            regime: Current market regime
            sharpe_ratios: Optional precomputed Sharpe ratios
        
        Returns:
            Optimized weight dictionary
        """
        # Get regime-specific bounds
        bounds = self._get_regime_bounds(regime)
        
        # If no Sharpe ratios provided, use equal weighting within bounds
        if sharpe_ratios is None or all(s == 0.0 for s in sharpe_ratios.values()):
            self.logger.info("No Sharpe data, using regime-based weights")
            return self.regime_detector.get_strategy_weights(regime)
        
        # Normalize Sharpe ratios (handle negative Sharpes)
        sharpe_array = np.array([sharpe_ratios[s] for s in self.strategies])
        
        # Shift to positive (min Sharpe -> 0.1, max Sharpe -> 1.0)
        min_sharpe = np.min(sharpe_array)
        if min_sharpe < 0:
            sharpe_array = sharpe_array - min_sharpe + 0.1
        
        # Normalize to sum to 1
        sharpe_sum = np.sum(sharpe_array)
        if sharpe_sum > 1e-8:
            target_weights = sharpe_array / sharpe_sum
        else:
            target_weights = np.ones(len(self.strategies)) / len(self.strategies)
        
        # Apply regime bounds via optimization
        def objective(w):
            # Minimize distance from target weights
            return np.sum((w - target_weights) ** 2)
        
        # Constraints: sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds for each strategy
        weight_bounds = [bounds[s] for s in self.strategies]
        
        # Initial guess: current weights
        x0 = np.array([self.current_weights[s] for s in self.strategies])
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=weight_bounds,
            constraints=constraints,
        )
        
        if result.success:
            optimized_weights = {
                self.strategies[i]: float(result.x[i])
                for i in range(len(self.strategies))
            }
            
            self.logger.info("Optimized weights:")
            for strategy, weight in optimized_weights.items():
                self.logger.info(f"  {strategy}: {weight:.1%}")
            
            return optimized_weights
        else:
            self.logger.warning(f"Optimization failed: {result.message}")
            # Fallback to regime weights
            return self.regime_detector.get_strategy_weights(regime)
    
    def apply_bayesian_update(
        self, 
        strategy: str, 
        realized_return: float
    ) -> None:
        """
        Apply Bayesian update to strategy belief.
        
        Updates prior belief based on realized trade outcome.
        
        Args:
            strategy: Strategy name
            realized_return: Realized return from trade (percentage)
        """
        if strategy not in self.strategies:
            return
        
        # Record return
        self.strategy_returns[strategy].append(realized_return)
        self.strategy_trades[strategy] += 1
        
        # Bayesian update (simplified: weight by success/failure)
        success = 1.0 if realized_return > 0 else 0.0
        
        # Update prior using exponential moving average
        alpha = 0.1  # Learning rate
        old_prior = self.bayesian_priors[strategy]
        new_prior = (1 - alpha) * old_prior + alpha * success
        
        self.bayesian_priors[strategy] = new_prior
        
        self.logger.debug(
            f"Bayesian update for {strategy}: "
            f"return={realized_return:.2%}, new_prior={new_prior:.2f}"
        )
    
    async def rebalance(
        self,
        regime,  # MarketRegime enum
        force: bool = False,
    ) -> Dict[str, float]:
        """
        Rebalance strategy weights if needed.
        
        Rebalances when:
        - Regime changes
        - Weight drift > MIN_WEIGHT_CHANGE
        - Force flag set
        - Cooldown period elapsed
        
        Args:
            regime: Current market regime
            force: Force rebalance regardless of conditions
        
        Returns:
            New weight allocation
        """
        # Check cooldown
        if self.last_rebalance is not None and not force:
            days_since_rebalance = (datetime.now() - self.last_rebalance).days
            if days_since_rebalance < self.REBALANCE_COOLDOWN_DAYS:
                self.logger.debug("Rebalance on cooldown")
                return self.current_weights
        
        # Calculate new weights
        sharpe_ratios = await self.calculate_strategy_sharpe()
        new_weights = self.optimize_weights(regime, sharpe_ratios)
        
        # Check if significant change
        max_change = max(
            abs(new_weights[s] - self.current_weights[s])
            for s in self.strategies
        )
        
        if max_change < self.MIN_WEIGHT_CHANGE and not force:
            self.logger.info(f"Weight change ({max_change:.1%}) below threshold, skipping rebalance")
            return self.current_weights
        
        # Apply rebalance
        old_weights = self.current_weights.copy()
        self.current_weights = new_weights
        self.last_rebalance = datetime.now()
        self.rebalance_count += 1
        
        self.logger.info(f"REBALANCE #{self.rebalance_count}:")
        for strategy in self.strategies:
            change = new_weights[strategy] - old_weights[strategy]
            self.logger.info(
                f"  {strategy}: {old_weights[strategy]:.1%} -> {new_weights[strategy]:.1%} "
                f"({'+'if change >= 0 else ''}{change:.1%})"
            )
        
        return new_weights
    
    def get_current_weights(self) -> Dict[str, float]:
        """
        Get current strategy weights.
        
        Returns:
            Current weight dictionary
        """
        return self.current_weights.copy()
    
    def get_performance_summary(self) -> List[StrategyPerformance]:
        """
        Get performance summary for all strategies.
        
        Returns:
            List of StrategyPerformance objects
        """
        summary = []
        
        for strategy in self.strategies:
            returns = self.strategy_returns.get(strategy, [])
            trades = self.strategy_trades.get(strategy, 0)
            
            if len(returns) > 0:
                cumulative_return = np.sum(returns)
                volatility = np.std(returns)
                win_rate = sum(1 for r in returns if r > 0) / len(returns)
                avg_return = np.mean(returns)
                
                # Calculate Sharpe
                if volatility > 1e-8:
                    sharpe = (avg_return / volatility) * np.sqrt(252)
                else:
                    sharpe = 0.0
            else:
                cumulative_return = 0.0
                volatility = 0.0
                win_rate = 0.0
                avg_return = 0.0
                sharpe = 0.0
            
            summary.append(StrategyPerformance(
                strategy_name=strategy,
                sharpe_ratio=sharpe,
                cumulative_return=cumulative_return,
                volatility=volatility,
                win_rate=win_rate,
                avg_trade_return=avg_return,
                total_trades=trades,
                last_updated=datetime.now(),
            ))
        
        return summary
    
    def _get_regime_bounds(self, regime) -> Dict[str, Tuple[float, float]]:
        """
        Get weight bounds for current regime.
        
        Args:
            regime: MarketRegime enum
        
        Returns:
            Dict of (min, max) bounds per strategy
        """
        # Get regime weights as center point
        regime_weights = self.regime_detector.get_strategy_weights(regime)
        
        # Allow +/- 10% around regime weights, but respect DEFAULT_BOUNDS
        bounds = {}
        for strategy in self.strategies:
            center = regime_weights.get(strategy, 0.25)
            min_bound = max(self.DEFAULT_BOUNDS[strategy][0], center - 0.10)
            max_bound = min(self.DEFAULT_BOUNDS[strategy][1], center + 0.10)
            bounds[strategy] = (min_bound, max_bound)
        
        return bounds


# ============================================================================
# TESTING HELPER
# ============================================================================

async def test_weight_optimizer():
    """Test the weight optimizer."""
    import logging
    from .regime_detector import RegimeDetector, MarketRegime
    
    logging.basicConfig(level=logging.INFO)
    
    # Create regime detector
    detector = RegimeDetector()
    await detector.fit()
    
    # Create optimizer
    optimizer = DynamicWeightOptimizer(
        strategies=["iv_rank", "theta_decay", "mean_reversion", "delta_hedging"],
        regime_detector=detector,
    )
    
    print("\n" + "="*60)
    print("TESTING DYNAMIC WEIGHT OPTIMIZER")
    print("="*60)
    
    # Detect current regime
    regime_state = await detector.detect_current_regime()
    current_regime = regime_state.current_regime
    
    print(f"\nCurrent Regime: {current_regime.value}")
    
    # Test 1: Calculate Sharpe ratios (should be zero initially)
    print("\n1. Initial Sharpe ratios (no data):")
    sharpe_ratios = await optimizer.calculate_strategy_sharpe()
    for strategy, sharpe in sharpe_ratios.items():
        print(f"  {strategy}: {sharpe:.2f}")
    
    # Test 2: Optimize weights without Sharpe data
    print("\n2. Optimize weights (regime-based):")
    weights = optimizer.optimize_weights(current_regime)
    for strategy, weight in weights.items():
        print(f"  {strategy}: {weight:.1%}")
    
    # Test 3: Add some sample returns
    print("\n3. Adding sample trade returns...")
    sample_returns = {
        "iv_rank": [0.05, 0.03, -0.02, 0.04, 0.06, 0.02, 0.05, 0.07, 0.03, 0.04],
        "theta_decay": [0.02, 0.03, 0.02, 0.01, 0.03, 0.02, 0.02, 0.03, 0.02, 0.01],
        "mean_reversion": [0.08, -0.03, 0.10, -0.05, 0.12, 0.03, -0.02, 0.09, 0.04, 0.06],
        "delta_hedging": [0.01, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01],
    }
    
    for strategy, returns in sample_returns.items():
        for ret in returns:
            optimizer.apply_bayesian_update(strategy, ret)
    
    # Test 4: Calculate Sharpe with data
    print("\n4. Sharpe ratios after trades:")
    sharpe_ratios = await optimizer.calculate_strategy_sharpe()
    for strategy, sharpe in sharpe_ratios.items():
        print(f"  {strategy}: {sharpe:.2f}")
    
    # Test 5: Optimize with Sharpe data
    print("\n5. Optimize weights (Sharpe-adjusted):")
    new_weights = optimizer.optimize_weights(current_regime, sharpe_ratios)
    for strategy, weight in new_weights.items():
        print(f"  {strategy}: {weight:.1%}")
    
    # Test 6: Rebalance
    print("\n6. Rebalancing...")
    rebalanced_weights = await optimizer.rebalance(current_regime, force=True)
    
    # Test 7: Performance summary
    print("\n7. Performance summary:")
    summary = optimizer.get_performance_summary()
    for perf in summary:
        print(f"\n  {perf.strategy_name}:")
        print(f"    Sharpe: {perf.sharpe_ratio:.2f}")
        print(f"    Cumulative Return: {perf.cumulative_return:.2%}")
        print(f"    Win Rate: {perf.win_rate:.1%}")
        print(f"    Trades: {perf.total_trades}")
    
    # Validate
    assert abs(sum(weights.values()) - 1.0) < 0.01
    assert abs(sum(new_weights.values()) - 1.0) < 0.01
    assert abs(sum(rebalanced_weights.values()) - 1.0) < 0.01
    assert all(0.0 <= w <= 1.0 for w in weights.values())
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_weight_optimizer())

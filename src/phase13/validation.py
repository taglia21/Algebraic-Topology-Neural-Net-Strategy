"""
Validation Framework for Phase 13
==================================

Walk-forward analysis, Monte Carlo simulation, and cost sensitivity testing.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results from a validation test."""
    test_name: str
    passed: bool
    score: float
    metrics: Dict
    details: str


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis."""
    window_results: List[Dict]
    aggregate_cagr: float
    win_rate: float
    avg_return: float
    std_return: float
    worst_window: Dict
    best_window: Dict
    passed: bool


class WalkForwardValidator:
    """
    Walk-forward validation with rolling windows.
    
    Tests if strategy performance is consistent across multiple
    independent time periods (not just lucky on one period).
    """
    
    def __init__(
        self,
        window_months: int = 6,
        min_windows: int = 6,
        min_win_rate: float = 0.70,
    ):
        self.window_months = window_months
        self.min_windows = min_windows
        self.min_win_rate = min_win_rate
    
    def validate(
        self,
        strategy_func: Callable,
        data: Dict[str, pd.DataFrame],
        spy_data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.
        
        Args:
            strategy_func: Function that runs strategy and returns dict with 'total_return'
            data: Dict of ticker -> price DataFrame
            spy_data: SPY price data
            start_date: Start of test period
            end_date: End of test period
            
        Returns:
            WalkForwardResult with window-by-window performance
        """
        # Generate windows
        windows = self._generate_windows(start_date, end_date)
        
        if len(windows) < self.min_windows:
            logger.warning(f"Only {len(windows)} windows (need {self.min_windows})")
        
        window_results = []
        
        for i, (win_start, win_end) in enumerate(windows):
            try:
                # Run strategy on this window
                result = strategy_func(
                    data=data,
                    spy_data=spy_data,
                    start_date=win_start,
                    end_date=win_end,
                )
                
                window_results.append({
                    'window': i + 1,
                    'start': win_start,
                    'end': win_end,
                    'total_return': result.get('total_return', 0),
                    'cagr': result.get('cagr', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'sharpe': result.get('sharpe', 0),
                    'profitable': result.get('total_return', 0) > 0,
                })
            except Exception as e:
                logger.error(f"Window {i+1} failed: {e}")
                window_results.append({
                    'window': i + 1,
                    'start': win_start,
                    'end': win_end,
                    'total_return': 0,
                    'cagr': 0,
                    'max_drawdown': 0,
                    'sharpe': 0,
                    'profitable': False,
                    'error': str(e),
                })
        
        # Aggregate metrics
        returns = [w['total_return'] for w in window_results]
        win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0
        
        avg_return = np.mean(returns) if returns else 0
        std_return = np.std(returns) if returns else 0
        
        # Find best/worst
        if window_results:
            best_idx = np.argmax(returns)
            worst_idx = np.argmin(returns)
            best_window = window_results[best_idx]
            worst_window = window_results[worst_idx]
        else:
            best_window = worst_window = {}
        
        # Calculate aggregate CAGR
        compound_return = np.prod([1 + r for r in returns]) - 1 if returns else 0
        years = len(returns) * self.window_months / 12
        aggregate_cagr = (1 + compound_return) ** (1/years) - 1 if years > 0 else 0
        
        return WalkForwardResult(
            window_results=window_results,
            aggregate_cagr=aggregate_cagr,
            win_rate=win_rate,
            avg_return=avg_return,
            std_return=std_return,
            worst_window=worst_window,
            best_window=best_window,
            passed=win_rate >= self.min_win_rate,
        )
    
    def _generate_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Tuple[datetime, datetime]]:
        """Generate rolling windows."""
        windows = []
        current_start = start_date
        
        while current_start < end_date:
            current_end = current_start + timedelta(days=self.window_months * 30)
            if current_end > end_date:
                current_end = end_date
            
            if (current_end - current_start).days >= 60:  # Min 2 months
                windows.append((current_start, current_end))
            
            current_start = current_start + timedelta(days=self.window_months * 30)
        
        return windows


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    n_simulations: int
    returns_distribution: np.ndarray
    mean_return: float
    median_return: float
    std_return: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    prob_positive: float
    prob_above_target: float
    passed: bool


class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy validation.
    
    Bootstraps historical returns to generate thousands of
    alternate market scenarios.
    """
    
    def __init__(
        self,
        n_simulations: int = 5000,
        target_return: float = 1.50,  # 150%
        min_prob_above_target: float = 0.50,
    ):
        self.n_simulations = n_simulations
        self.target_return = target_return
        self.min_prob_above_target = min_prob_above_target
    
    def simulate(
        self,
        daily_returns: pd.Series,
        n_days: int = 850,  # ~3.4 years
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation by bootstrapping daily returns.
        
        Args:
            daily_returns: Historical daily returns series
            n_days: Number of days to simulate
            
        Returns:
            MonteCarloResult with distribution statistics
        """
        returns_clean = daily_returns.dropna().values
        
        if len(returns_clean) < 50:
            logger.warning("Insufficient data for Monte Carlo")
            return MonteCarloResult(
                n_simulations=0,
                returns_distribution=np.array([]),
                mean_return=0,
                median_return=0,
                std_return=0,
                percentile_5=0,
                percentile_25=0,
                percentile_75=0,
                percentile_95=0,
                prob_positive=0,
                prob_above_target=0,
                passed=False,
            )
        
        # Run simulations
        simulation_returns = []
        
        for _ in range(self.n_simulations):
            # Bootstrap: randomly sample with replacement
            sampled_returns = np.random.choice(returns_clean, size=n_days, replace=True)
            
            # Compound returns
            total_return = np.prod(1 + sampled_returns) - 1
            simulation_returns.append(total_return)
        
        simulation_returns = np.array(simulation_returns)
        
        # Calculate statistics
        prob_positive = np.mean(simulation_returns > 0)
        prob_above_target = np.mean(simulation_returns > self.target_return)
        
        return MonteCarloResult(
            n_simulations=self.n_simulations,
            returns_distribution=simulation_returns,
            mean_return=np.mean(simulation_returns),
            median_return=np.median(simulation_returns),
            std_return=np.std(simulation_returns),
            percentile_5=np.percentile(simulation_returns, 5),
            percentile_25=np.percentile(simulation_returns, 25),
            percentile_75=np.percentile(simulation_returns, 75),
            percentile_95=np.percentile(simulation_returns, 95),
            prob_positive=prob_positive,
            prob_above_target=prob_above_target,
            passed=prob_above_target >= self.min_prob_above_target,
        )
    
    def simulate_with_regime(
        self,
        bull_returns: pd.Series,
        bear_returns: pd.Series,
        regime_sequence: List[str],
        n_days: int = 850,
    ) -> MonteCarloResult:
        """
        Monte Carlo with regime-aware bootstrapping.
        
        Samples from bull returns when regime is bull,
        from bear returns when regime is bear.
        """
        bull_clean = bull_returns.dropna().values
        bear_clean = bear_returns.dropna().values
        
        simulation_returns = []
        
        for _ in range(self.n_simulations):
            simulated_daily = []
            
            for day in range(n_days):
                # Determine regime for this day (cycle through pattern)
                regime = regime_sequence[day % len(regime_sequence)]
                
                if 'bull' in regime.lower():
                    ret = np.random.choice(bull_clean)
                else:
                    ret = np.random.choice(bear_clean)
                
                simulated_daily.append(ret)
            
            total_return = np.prod(1 + np.array(simulated_daily)) - 1
            simulation_returns.append(total_return)
        
        simulation_returns = np.array(simulation_returns)
        
        prob_positive = np.mean(simulation_returns > 0)
        prob_above_target = np.mean(simulation_returns > self.target_return)
        
        return MonteCarloResult(
            n_simulations=self.n_simulations,
            returns_distribution=simulation_returns,
            mean_return=np.mean(simulation_returns),
            median_return=np.median(simulation_returns),
            std_return=np.std(simulation_returns),
            percentile_5=np.percentile(simulation_returns, 5),
            percentile_25=np.percentile(simulation_returns, 25),
            percentile_75=np.percentile(simulation_returns, 75),
            percentile_95=np.percentile(simulation_returns, 95),
            prob_positive=prob_positive,
            prob_above_target=prob_above_target,
            passed=prob_above_target >= self.min_prob_above_target,
        )


@dataclass
class CostSensitivityResult:
    """Results from cost sensitivity analysis."""
    cost_scenarios: List[Dict]
    base_return: float
    worst_case_return: float
    cost_impact_range: Tuple[float, float]
    still_profitable: bool
    passed: bool


class CostSensitivityAnalyzer:
    """
    Analyze strategy sensitivity to transaction costs.
    
    Tests how performance degrades with various slippage
    and commission levels.
    """
    
    def __init__(
        self,
        slippage_levels: List[float] = None,
        commission_levels: List[float] = None,
        min_return_threshold: float = 3.0,  # 300%
    ):
        self.slippage_levels = slippage_levels or [0.001, 0.002, 0.005, 0.01]
        self.commission_levels = commission_levels or [1.0, 5.0, 10.0]
        self.min_return_threshold = min_return_threshold
    
    def analyze(
        self,
        base_return: float,
        n_trades: int,
        avg_trade_size: float = 10000,
        initial_capital: float = 100000,
    ) -> CostSensitivityResult:
        """
        Analyze impact of transaction costs on returns.
        
        Args:
            base_return: Base strategy return (before costs)
            n_trades: Number of trades over the period
            avg_trade_size: Average trade size in dollars
            initial_capital: Starting capital
            
        Returns:
            CostSensitivityResult with cost scenarios
        """
        cost_scenarios = []
        
        for slippage in self.slippage_levels:
            for commission in self.commission_levels:
                # Calculate total cost
                slippage_cost = slippage * avg_trade_size * n_trades * 2  # Round trip
                commission_cost = commission * n_trades * 2  # Round trip
                total_cost = slippage_cost + commission_cost
                
                # Adjust return
                cost_as_pct = total_cost / initial_capital
                adjusted_return = base_return - cost_as_pct
                
                cost_scenarios.append({
                    'slippage': slippage,
                    'commission': commission,
                    'total_cost': total_cost,
                    'cost_pct': cost_as_pct,
                    'adjusted_return': adjusted_return,
                    'return_pct': adjusted_return * 100,
                    'profitable': adjusted_return > 0,
                })
        
        # Find worst case
        worst_return = min(s['adjusted_return'] for s in cost_scenarios)
        best_return = max(s['adjusted_return'] for s in cost_scenarios)
        
        return CostSensitivityResult(
            cost_scenarios=cost_scenarios,
            base_return=base_return,
            worst_case_return=worst_return,
            cost_impact_range=(best_return, worst_return),
            still_profitable=worst_return > 0,
            passed=worst_return >= self.min_return_threshold,
        )


class ParameterStabilityTester:
    """
    Test parameter stability (is strategy robust to parameter changes?).
    """
    
    def __init__(self, variation_range: float = 0.20):
        self.variation_range = variation_range
    
    def test_parameter(
        self,
        strategy_func: Callable,
        param_name: str,
        base_value: float,
        data: Dict[str, pd.DataFrame],
        spy_data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict:
        """
        Test sensitivity to a single parameter.
        """
        # Test values
        test_values = [
            base_value * (1 - self.variation_range),
            base_value * (1 - self.variation_range / 2),
            base_value,
            base_value * (1 + self.variation_range / 2),
            base_value * (1 + self.variation_range),
        ]
        
        results = []
        
        for value in test_values:
            try:
                result = strategy_func(
                    data=data,
                    spy_data=spy_data,
                    start_date=start_date,
                    end_date=end_date,
                    **{param_name: value},
                )
                results.append({
                    'param_value': value,
                    'return': result.get('total_return', 0),
                    'sharpe': result.get('sharpe', 0),
                    'max_dd': result.get('max_drawdown', 0),
                })
            except Exception as e:
                logger.error(f"Parameter test failed for {param_name}={value}: {e}")
                results.append({
                    'param_value': value,
                    'return': 0,
                    'sharpe': 0,
                    'max_dd': 1.0,
                    'error': str(e),
                })
        
        returns = [r['return'] for r in results]
        variance = np.var(returns) if returns else 0
        mean = np.mean(returns) if returns else 0
        coef_of_var = np.sqrt(variance) / abs(mean) if mean != 0 else float('inf')
        
        return {
            'param_name': param_name,
            'base_value': base_value,
            'test_results': results,
            'return_variance': variance,
            'return_mean': mean,
            'coefficient_of_variation': coef_of_var,
            'stable': coef_of_var < 0.30,  # <30% variation is stable
        }

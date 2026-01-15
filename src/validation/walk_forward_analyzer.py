"""
Walk-Forward Analyzer for Phase 7.

Implements rolling window walk-forward analysis:
- 12-month optimization windows
- 3-month out-of-sample test periods
- Composite equity curve from all OOS periods
- Parameter stability analysis across windows

Key metrics:
- In-sample vs out-of-sample performance degradation
- Consistency of returns across windows
- Parameter stability (do optimal params change?)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WFAWindow:
    """A single walk-forward window."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Results
    train_sharpe: float = 0.0
    train_cagr: float = 0.0
    train_max_dd: float = 0.0
    
    test_sharpe: float = 0.0
    test_cagr: float = 0.0
    test_max_dd: float = 0.0
    test_return: float = 0.0
    
    # Parameters used
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Equity curve segment
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)


@dataclass
class WFAResult:
    """Complete walk-forward analysis result."""
    windows: List[WFAWindow]
    
    # Aggregate metrics
    composite_sharpe: float = 0.0
    composite_cagr: float = 0.0
    composite_max_dd: float = 0.0
    composite_return: float = 0.0
    
    # Comparison to simple backtest
    backtest_sharpe: float = 0.0
    backtest_cagr: float = 0.0
    backtest_max_dd: float = 0.0
    
    # Stability metrics
    sharpe_consistency: float = 0.0  # Pct of windows with positive Sharpe
    return_consistency: float = 0.0  # Pct of windows with positive return
    param_stability: float = 0.0  # How stable are optimal params
    
    # Degradation
    oos_degradation: float = 0.0  # Avg (IS Sharpe - OOS Sharpe) / IS Sharpe


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis framework.
    
    Process:
    1. Split data into rolling windows
    2. For each window:
       a. Optimize on training period (find best params)
       b. Test on out-of-sample period (with fixed params)
    3. Chain OOS periods to build composite equity curve
    4. Compare composite to full-period backtest
    """
    
    def __init__(
        self,
        train_months: int = 12,
        test_months: int = 3,
        step_months: int = 3,  # How far to advance each window
        min_train_months: int = 6,
        optimize_params: bool = True,
    ):
        """
        Initialize WFA.
        
        Args:
            train_months: Length of training/optimization window
            test_months: Length of out-of-sample test window
            step_months: Months to advance between windows
            min_train_months: Minimum training data required
            optimize_params: Whether to optimize params each window
        """
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        self.min_train_months = min_train_months
        self.optimize_params = optimize_params
        
        logger.info(f"WFA initialized: {train_months}m train, {test_months}m test, {step_months}m step")
    
    def generate_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Dict[str, datetime]]:
        """
        Generate train/test windows.
        
        Returns:
            List of dicts with train_start, train_end, test_start, test_end
        """
        windows = []
        window_id = 0
        
        current_train_start = start_date
        
        while True:
            train_end = current_train_start + relativedelta(months=self.train_months)
            test_start = train_end
            test_end = test_start + relativedelta(months=self.test_months)
            
            # Check if we have enough room for test period
            if test_end > end_date:
                break
            
            windows.append({
                'window_id': window_id,
                'train_start': current_train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
            })
            
            window_id += 1
            current_train_start = current_train_start + relativedelta(months=self.step_months)
        
        logger.info(f"Generated {len(windows)} WFA windows")
        return windows
    
    def run_analysis(
        self,
        data: Dict[str, pd.DataFrame],  # ticker -> OHLCV
        strategy_fn: Callable,  # Function to run backtest
        param_grid: Dict[str, List[Any]] = None,  # Params to optimize
        start_date: str = None,
        end_date: str = None,
        initial_capital: float = 100000,
    ) -> WFAResult:
        """
        Run complete walk-forward analysis.
        
        Args:
            data: Dict of ticker -> OHLCV DataFrame
            strategy_fn: Function that runs backtest with signature:
                         strategy_fn(data, start, end, params) -> Dict with metrics
            param_grid: Optional parameter grid for optimization
            start_date: Override start date
            end_date: Override end date
            initial_capital: Starting capital
            
        Returns:
            WFAResult with all metrics
        """
        # Determine date range from data
        sample_df = list(data.values())[0]
        data_start = sample_df.index.min()
        data_end = sample_df.index.max()
        
        if start_date:
            data_start = pd.Timestamp(start_date)
        if end_date:
            data_end = pd.Timestamp(end_date)
        
        # Generate windows
        windows = self.generate_windows(data_start, data_end)
        
        if len(windows) == 0:
            raise ValueError("Not enough data for walk-forward analysis")
        
        wfa_windows = []
        composite_equity = []
        current_capital = initial_capital
        
        for w in windows:
            logger.info(f"Processing window {w['window_id']}: "
                       f"Train {w['train_start'].date()} to {w['train_end'].date()}, "
                       f"Test {w['test_start'].date()} to {w['test_end'].date()}")
            
            # Run on training period (with optimization if enabled)
            if self.optimize_params and param_grid:
                best_params, train_metrics = self._optimize_params(
                    data, strategy_fn, w['train_start'], w['train_end'],
                    param_grid, initial_capital
                )
            else:
                best_params = {}
                train_result = strategy_fn(
                    data, 
                    str(w['train_start'].date()), 
                    str(w['train_end'].date()),
                    best_params,
                    initial_capital,
                )
                train_metrics = train_result
            
            # Run on test period (with fixed params from training)
            test_result = strategy_fn(
                data,
                str(w['test_start'].date()),
                str(w['test_end'].date()),
                best_params,
                current_capital,  # Use rolling capital
            )
            
            # Create window result
            wfa_window = WFAWindow(
                window_id=w['window_id'],
                train_start=w['train_start'],
                train_end=w['train_end'],
                test_start=w['test_start'],
                test_end=w['test_end'],
                train_sharpe=train_metrics.get('sharpe_ratio', 0),
                train_cagr=train_metrics.get('cagr_pct', 0) / 100,
                train_max_dd=train_metrics.get('max_drawdown_pct', 0) / 100,
                test_sharpe=test_result.get('sharpe_ratio', 0),
                test_cagr=test_result.get('cagr_pct', 0) / 100,
                test_max_dd=test_result.get('max_drawdown_pct', 0) / 100,
                test_return=test_result.get('total_return_pct', 0) / 100,
                params=best_params,
                equity_curve=test_result.get('equity_curve', []),
            )
            
            wfa_windows.append(wfa_window)
            
            # Update rolling capital
            current_capital = current_capital * (1 + wfa_window.test_return)
            
            # Append to composite equity curve
            if wfa_window.equity_curve:
                composite_equity.extend(wfa_window.equity_curve)
        
        # Calculate composite metrics
        result = self._calculate_composite_metrics(
            wfa_windows, 
            composite_equity,
            initial_capital,
        )
        
        return result
    
    def _optimize_params(
        self,
        data: Dict[str, pd.DataFrame],
        strategy_fn: Callable,
        train_start: datetime,
        train_end: datetime,
        param_grid: Dict[str, List[Any]],
        initial_capital: float,
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Optimize parameters on training period.
        
        Returns:
            Tuple of (best_params, best_metrics)
        """
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        best_sharpe = -np.inf
        best_params = {}
        best_metrics = {}
        
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            
            try:
                result = strategy_fn(
                    data,
                    str(train_start.date()),
                    str(train_end.date()),
                    params,
                    initial_capital,
                )
                
                sharpe = result.get('sharpe_ratio', 0)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
                    best_metrics = result
                    
            except Exception as e:
                logger.warning(f"Param optimization failed for {params}: {e}")
                continue
        
        logger.info(f"Best params: {best_params} -> Sharpe {best_sharpe:.2f}")
        return best_params, best_metrics
    
    def _calculate_composite_metrics(
        self,
        windows: List[WFAWindow],
        equity_curve: List[Tuple[datetime, float]],
        initial_capital: float,
    ) -> WFAResult:
        """Calculate aggregate metrics from all windows."""
        if not windows:
            return WFAResult(windows=[])
        
        # Aggregate from windows
        test_sharpes = [w.test_sharpe for w in windows]
        test_returns = [w.test_return for w in windows]
        test_dds = [abs(w.test_max_dd) for w in windows]
        
        train_sharpes = [w.train_sharpe for w in windows]
        
        # Composite metrics
        composite_return = np.prod([1 + r for r in test_returns]) - 1
        years = len(windows) * self.test_months / 12
        composite_cagr = (1 + composite_return) ** (1/years) - 1 if years > 0 else 0
        
        # Composite Sharpe (from OOS returns)
        monthly_returns = test_returns  # Each is already a period return
        if len(monthly_returns) > 1:
            ret_mean = np.mean(monthly_returns)
            ret_std = np.std(monthly_returns)
            annualized_factor = 12 / self.test_months
            composite_sharpe = (ret_mean / ret_std) * np.sqrt(annualized_factor) if ret_std > 0 else 0
        else:
            composite_sharpe = test_sharpes[0] if test_sharpes else 0
        
        # Max drawdown from composite equity
        composite_max_dd = max(test_dds) if test_dds else 0
        
        # Consistency metrics
        sharpe_consistency = sum(1 for s in test_sharpes if s > 0) / len(test_sharpes)
        return_consistency = sum(1 for r in test_returns if r > 0) / len(test_returns)
        
        # OOS degradation
        degradations = []
        for w in windows:
            if w.train_sharpe > 0:
                deg = (w.train_sharpe - w.test_sharpe) / w.train_sharpe
                degradations.append(deg)
        oos_degradation = np.mean(degradations) if degradations else 0
        
        # Parameter stability (how often do optimal params change?)
        if windows[0].params:
            param_changes = 0
            for i in range(1, len(windows)):
                if windows[i].params != windows[i-1].params:
                    param_changes += 1
            param_stability = 1 - (param_changes / max(1, len(windows) - 1))
        else:
            param_stability = 1.0  # No optimization = fully stable
        
        return WFAResult(
            windows=windows,
            composite_sharpe=composite_sharpe,
            composite_cagr=composite_cagr,
            composite_max_dd=composite_max_dd,
            composite_return=composite_return,
            sharpe_consistency=sharpe_consistency,
            return_consistency=return_consistency,
            param_stability=param_stability,
            oos_degradation=oos_degradation,
        )


def run_quick_wfa(
    data: Dict[str, pd.DataFrame],
    strategy_fn: Callable,
    train_months: int = 12,
    test_months: int = 3,
    initial_capital: float = 100000,
) -> WFAResult:
    """
    Quick helper to run WFA with default settings.
    
    Args:
        data: Dict of ticker -> OHLCV DataFrame
        strategy_fn: Strategy backtest function
        train_months: Training window length
        test_months: Test window length
        initial_capital: Starting capital
        
    Returns:
        WFAResult
    """
    analyzer = WalkForwardAnalyzer(
        train_months=train_months,
        test_months=test_months,
        step_months=test_months,  # Non-overlapping test periods
        optimize_params=False,  # Use fixed params
    )
    
    return analyzer.run_analysis(
        data=data,
        strategy_fn=strategy_fn,
        initial_capital=initial_capital,
    )


def print_wfa_report(result: WFAResult) -> None:
    """Print formatted WFA report."""
    print("\n" + "="*60)
    print("WALK-FORWARD ANALYSIS REPORT")
    print("="*60)
    
    print(f"\nWindows Analyzed: {len(result.windows)}")
    print(f"\n{'Window':<8} {'Train Sharpe':<14} {'Test Sharpe':<14} {'Test Return':<14}")
    print("-"*50)
    
    for w in result.windows:
        print(f"{w.window_id:<8} {w.train_sharpe:>10.2f}    {w.test_sharpe:>10.2f}    {w.test_return*100:>10.1f}%")
    
    print("\n" + "-"*50)
    print("COMPOSITE METRICS (Out-of-Sample)")
    print("-"*50)
    print(f"  Total Return:    {result.composite_return*100:.1f}%")
    print(f"  CAGR:            {result.composite_cagr*100:.1f}%")
    print(f"  Sharpe Ratio:    {result.composite_sharpe:.2f}")
    print(f"  Max Drawdown:    {result.composite_max_dd*100:.1f}%")
    
    print("\n" + "-"*50)
    print("CONSISTENCY METRICS")
    print("-"*50)
    print(f"  Sharpe > 0:      {result.sharpe_consistency*100:.0f}% of windows")
    print(f"  Return > 0:      {result.return_consistency*100:.0f}% of windows")
    print(f"  Param Stability: {result.param_stability*100:.0f}%")
    print(f"  OOS Degradation: {result.oos_degradation*100:.1f}%")
    
    print("="*60)


if __name__ == "__main__":
    # Test window generation
    print("Testing Walk-Forward Analyzer")
    print("="*50)
    
    analyzer = WalkForwardAnalyzer(
        train_months=12,
        test_months=3,
        step_months=3,
    )
    
    # Generate windows for 4-year period
    start = datetime(2021, 1, 1)
    end = datetime(2024, 12, 31)
    
    windows = analyzer.generate_windows(start, end)
    
    print(f"Generated {len(windows)} windows:")
    for w in windows:
        print(f"  Window {w['window_id']}: "
              f"Train {w['train_start'].date()} to {w['train_end'].date()}, "
              f"Test {w['test_start'].date()} to {w['test_end'].date()}")
    
    print("\nWalk-Forward Analyzer tests complete!")

"""Walk-Forward Analysis for strategy validation.

Implements rolling window walk-forward testing to detect overfitting
and validate out-of-sample performance.

Design:
- Rolling training window (default 2 years)
- Non-overlapping test periods (default 6 months)
- Step forward by configurable interval (default 3 months)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple, Any, Optional
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis for systematic strategy validation.
    
    Walk-forward design:
    1. Split data into rolling windows
    2. For each window:
       - Train model on optimization window
       - Optimize thresholds
       - Freeze parameters
       - Test on validation window (out-of-sample)
    3. Combine all out-of-sample results
    4. Calculate overfitting metrics
    """
    
    def __init__(
        self,
        optimization_window_days: int = 730,  # 2 years
        validation_window_days: int = 180,     # 6 months
        step_size_days: int = 90,              # 3 months
        min_trades_per_window: int = 5,
        results_dir: str = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results'
    ):
        """
        Initialize WalkForwardAnalyzer.
        
        Args:
            optimization_window_days: Training window size in days
            validation_window_days: Test window size in days
            step_size_days: How many days to step forward between folds
            min_trades_per_window: Minimum trades required for valid window
            results_dir: Directory to save results
        """
        self.optimization_window_days = optimization_window_days
        self.validation_window_days = validation_window_days
        self.step_size_days = step_size_days
        self.min_trades_per_window = min_trades_per_window
        self.results_dir = results_dir
        
        # Results storage
        self.window_results: List[Dict] = []
        self.aggregate_metrics: Dict = {}
        
        logger.info(f"WalkForwardAnalyzer initialized: "
                   f"opt={optimization_window_days}d, val={validation_window_days}d, step={step_size_days}d")
    
    def generate_windows(
        self,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, str]]:
        """
        Generate walk-forward window schedule.
        
        Args:
            start_date: Overall start date (YYYY-MM-DD)
            end_date: Overall end date (YYYY-MM-DD)
            
        Returns:
            List of window dicts with train_start, train_end, test_start, test_end
        """
        windows = []
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        window_idx = 0
        current_start = start
        
        while True:
            # Calculate window boundaries
            train_start = current_start
            train_end = train_start + timedelta(days=self.optimization_window_days - 1)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.validation_window_days - 1)
            
            # Check if test window exceeds end date
            if test_end > end:
                break
            
            window = {
                'window_idx': window_idx,
                'train_start': train_start.strftime("%Y-%m-%d"),
                'train_end': train_end.strftime("%Y-%m-%d"),
                'test_start': test_start.strftime("%Y-%m-%d"),
                'test_end': test_end.strftime("%Y-%m-%d"),
                'train_days': self.optimization_window_days,
                'test_days': self.validation_window_days
            }
            
            windows.append(window)
            window_idx += 1
            
            # Step forward
            current_start += timedelta(days=self.step_size_days)
        
        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows
    
    def run_walk_forward(
        self,
        data_dict: Dict[str, pd.DataFrame],
        model_factory,
        tda_generator,
        preprocessor,
        train_func,
        backtest_func,
        start_date: str = '2015-01-01',
        end_date: str = '2025-12-31',
        tickers: List[str] = None
    ) -> List[Dict]:
        """
        Run full walk-forward analysis.
        
        Args:
            data_dict: Dict of {ticker: DataFrame} with full date range
            model_factory: Function to create new model instance
            tda_generator: TDA feature generator instance
            preprocessor: Data preprocessor instance
            train_func: Function(model, X, y) -> trained model
            backtest_func: Function(model, data, ticker, start, end) -> metrics
            start_date: Walk-forward start date
            end_date: Walk-forward end date
            tickers: List of tickers to test
            
        Returns:
            List of window results
        """
        if tickers is None:
            tickers = list(data_dict.keys())
        
        windows = self.generate_windows(start_date, end_date)
        
        print("\n" + "=" * 70)
        print("WALK-FORWARD ANALYSIS")
        print("=" * 70)
        print(f"  Date Range: {start_date} to {end_date}")
        print(f"  Training Window: {self.optimization_window_days} days")
        print(f"  Validation Window: {self.validation_window_days} days")
        print(f"  Step Size: {self.step_size_days} days")
        print(f"  Total Windows: {len(windows)}")
        print(f"  Tickers: {tickers}")
        print("-" * 70)
        
        self.window_results = []
        
        for window in windows:
            print(f"\n  Window {window['window_idx']}: "
                  f"Train {window['train_start']} to {window['train_end']}, "
                  f"Test {window['test_start']} to {window['test_end']}")
            
            try:
                result = self._run_single_window(
                    window=window,
                    data_dict=data_dict,
                    model_factory=model_factory,
                    tda_generator=tda_generator,
                    preprocessor=preprocessor,
                    train_func=train_func,
                    backtest_func=backtest_func,
                    tickers=tickers
                )
                
                self.window_results.append(result)
                
                print(f"    In-sample Sharpe: {result['in_sample_sharpe']:.3f}, "
                      f"Out-of-sample Sharpe: {result['out_sample_sharpe']:.3f}")
                
            except Exception as e:
                logger.error(f"Window {window['window_idx']} failed: {e}")
                self.window_results.append({
                    'window_idx': window['window_idx'],
                    'error': str(e),
                    'in_sample_sharpe': 0,
                    'out_sample_sharpe': 0,
                    'in_sample_return': 0,
                    'out_sample_return': 0
                })
        
        # Calculate aggregate metrics
        self._calculate_aggregate_metrics()
        
        return self.window_results
    
    def _run_single_window(
        self,
        window: Dict,
        data_dict: Dict[str, pd.DataFrame],
        model_factory,
        tda_generator,
        preprocessor,
        train_func,
        backtest_func,
        tickers: List[str]
    ) -> Dict:
        """Run a single walk-forward window."""
        train_start = window['train_start']
        train_end = window['train_end']
        test_start = window['test_start']
        test_end = window['test_end']
        
        # Prepare training data
        train_X_list = []
        train_y_list = []
        
        for ticker in tickers:
            df = data_dict.get(ticker)
            if df is None or df.empty:
                continue
            
            # Filter to training window
            train_df = df[(df.index >= train_start) & (df.index <= train_end)].copy()
            
            if len(train_df) < 50:
                continue
            
            # Generate features
            try:
                tda_features = tda_generator.generate_features(train_df)
                X, y = preprocessor.prepare_sequences(train_df, tda_features)
                
                if len(X) > 0:
                    train_X_list.append(X)
                    train_y_list.append(y)
            except Exception as e:
                logger.warning(f"Feature generation failed for {ticker}: {e}")
                continue
        
        if not train_X_list:
            raise ValueError("No training data available for this window")
        
        # Concatenate training data
        train_X = np.concatenate(train_X_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)
        
        # Create and train model
        model = model_factory()
        trained_model = train_func(model, train_X, train_y)
        
        # In-sample evaluation
        train_predictions = trained_model.predict(train_X, verbose=0).flatten()
        in_sample_accuracy = np.mean((train_predictions > 0.5) == train_y)
        
        # Out-of-sample backtest
        out_sample_results = []
        in_sample_results = []
        
        for ticker in tickers:
            df = data_dict.get(ticker)
            if df is None or df.empty:
                continue
            
            # In-sample backtest
            train_df = df[(df.index >= train_start) & (df.index <= train_end)].copy()
            if len(train_df) >= 50:
                try:
                    is_metrics = backtest_func(trained_model, train_df, ticker, train_start, train_end)
                    in_sample_results.append(is_metrics)
                except Exception as e:
                    logger.warning(f"In-sample backtest failed for {ticker}: {e}")
            
            # Out-of-sample backtest
            test_df = df[(df.index >= test_start) & (df.index <= test_end)].copy()
            if len(test_df) >= 30:
                try:
                    oos_metrics = backtest_func(trained_model, test_df, ticker, test_start, test_end)
                    out_sample_results.append(oos_metrics)
                except Exception as e:
                    logger.warning(f"Out-of-sample backtest failed for {ticker}: {e}")
        
        # Aggregate metrics
        in_sample_sharpe = np.mean([r.get('sharpe_ratio', 0) for r in in_sample_results]) if in_sample_results else 0
        in_sample_return = np.mean([r.get('total_return', 0) for r in in_sample_results]) if in_sample_results else 0
        
        out_sample_sharpe = np.mean([r.get('sharpe_ratio', 0) for r in out_sample_results]) if out_sample_results else 0
        out_sample_return = np.mean([r.get('total_return', 0) for r in out_sample_results]) if out_sample_results else 0
        
        total_trades = sum(r.get('num_trades', 0) for r in out_sample_results)
        
        return {
            'window_idx': window['window_idx'],
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'in_sample_sharpe': round(in_sample_sharpe, 4),
            'in_sample_return': round(in_sample_return, 4),
            'out_sample_sharpe': round(out_sample_sharpe, 4),
            'out_sample_return': round(out_sample_return, 4),
            'out_sample_trades': total_trades,
            'in_sample_accuracy': round(in_sample_accuracy, 4),
            'num_tickers': len(out_sample_results)
        }
    
    def _calculate_aggregate_metrics(self):
        """Calculate aggregate metrics from all window results."""
        if not self.window_results:
            self.aggregate_metrics = {}
            return
        
        valid_results = [r for r in self.window_results if 'error' not in r]
        
        if not valid_results:
            self.aggregate_metrics = {}
            return
        
        in_sample_sharpes = [r['in_sample_sharpe'] for r in valid_results]
        out_sample_sharpes = [r['out_sample_sharpe'] for r in valid_results]
        in_sample_returns = [r['in_sample_return'] for r in valid_results]
        out_sample_returns = [r['out_sample_return'] for r in valid_results]
        
        # Calculate degradation
        avg_in_sharpe = np.mean(in_sample_sharpes)
        avg_out_sharpe = np.mean(out_sample_sharpes)
        
        if avg_in_sharpe != 0:
            degradation_pct = (avg_in_sharpe - avg_out_sharpe) / abs(avg_in_sharpe) * 100
        else:
            degradation_pct = 0
        
        # Calculate consistency
        profitable_windows = sum(1 for r in valid_results if r['out_sample_return'] > 0)
        consistency_pct = profitable_windows / len(valid_results) * 100
        
        # Calculate stability
        return_std = np.std(out_sample_returns) if len(out_sample_returns) > 1 else 0
        sharpe_std = np.std(out_sample_sharpes) if len(out_sample_sharpes) > 1 else 0
        
        # Overfitting score (0 = no overfitting, 100 = complete overfitting)
        if avg_in_sharpe > 0:
            overfitting_score = max(0, min(100, degradation_pct))
        else:
            overfitting_score = 0
        
        self.aggregate_metrics = {
            'num_windows': len(valid_results),
            'num_failed_windows': len(self.window_results) - len(valid_results),
            'avg_in_sample_sharpe': round(avg_in_sharpe, 4),
            'avg_out_sample_sharpe': round(avg_out_sharpe, 4),
            'avg_in_sample_return': round(np.mean(in_sample_returns), 4),
            'avg_out_sample_return': round(np.mean(out_sample_returns), 4),
            'degradation_pct': round(degradation_pct, 2),
            'consistency_pct': round(consistency_pct, 2),
            'return_stability_std': round(return_std, 4),
            'sharpe_stability_std': round(sharpe_std, 4),
            'overfitting_score': round(overfitting_score, 2),
            'total_out_sample_trades': sum(r.get('out_sample_trades', 0) for r in valid_results)
        }
    
    def generate_report(
        self,
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive walk-forward report.
        
        Args:
            output_path: Path to save JSON report
            
        Returns:
            Report dictionary
        """
        if output_path is None:
            output_path = os.path.join(self.results_dir, 'walk_forward_detailed.json')
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'optimization_window_days': self.optimization_window_days,
                'validation_window_days': self.validation_window_days,
                'step_size_days': self.step_size_days
            },
            'aggregate_metrics': self.aggregate_metrics,
            'window_results': self.window_results,
            'interpretation': self._generate_interpretation()
        }
        
        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Walk-forward report saved to: {output_path}")
        
        # Print summary
        self._print_summary()
        
        return report
    
    def _generate_interpretation(self) -> Dict[str, str]:
        """Generate human-readable interpretation of results."""
        interp = {}
        
        if not self.aggregate_metrics:
            return {'status': 'No results available'}
        
        # Overfitting assessment
        ofs = self.aggregate_metrics.get('overfitting_score', 0)
        if ofs < 20:
            interp['overfitting'] = 'LOW - Strategy shows minimal degradation from in-sample to out-of-sample'
        elif ofs < 40:
            interp['overfitting'] = 'MODERATE - Some performance degradation, but within acceptable limits'
        elif ofs < 60:
            interp['overfitting'] = 'CONCERNING - Significant performance drop in out-of-sample testing'
        else:
            interp['overfitting'] = 'HIGH - Strategy is likely overfit to training data'
        
        # Consistency assessment
        cons = self.aggregate_metrics.get('consistency_pct', 0)
        if cons >= 70:
            interp['consistency'] = f'GOOD - {cons:.0f}% of windows were profitable'
        elif cons >= 50:
            interp['consistency'] = f'FAIR - {cons:.0f}% of windows were profitable'
        else:
            interp['consistency'] = f'POOR - Only {cons:.0f}% of windows were profitable'
        
        # Stability assessment
        std = self.aggregate_metrics.get('return_stability_std', 0)
        if std < 0.02:
            interp['stability'] = 'STABLE - Returns are consistent across windows'
        elif std < 0.05:
            interp['stability'] = 'MODERATE - Some variability in returns'
        else:
            interp['stability'] = 'UNSTABLE - High variability in returns across windows'
        
        # Overall recommendation
        if ofs < 30 and cons >= 60:
            interp['recommendation'] = 'PROCEED - Strategy shows acceptable robustness'
        elif ofs < 50 and cons >= 50:
            interp['recommendation'] = 'CAUTION - Consider parameter simplification or additional validation'
        else:
            interp['recommendation'] = 'REDESIGN - Strategy requires fundamental changes to reduce overfitting'
        
        return interp
    
    def _print_summary(self):
        """Print summary of walk-forward results."""
        print("\n" + "=" * 70)
        print("WALK-FORWARD ANALYSIS SUMMARY")
        print("=" * 70)
        
        if not self.aggregate_metrics:
            print("  No results available")
            return
        
        am = self.aggregate_metrics
        
        print(f"  Windows Analyzed: {am.get('num_windows', 0)} (Failed: {am.get('num_failed_windows', 0)})")
        print(f"  Total Out-of-Sample Trades: {am.get('total_out_sample_trades', 0)}")
        print("-" * 70)
        print(f"  {'Metric':<30} {'In-Sample':>15} {'Out-of-Sample':>15}")
        print("-" * 70)
        print(f"  {'Average Sharpe':<30} {am.get('avg_in_sample_sharpe', 0):>15.3f} {am.get('avg_out_sample_sharpe', 0):>15.3f}")
        print(f"  {'Average Return':<30} {am.get('avg_in_sample_return', 0)*100:>14.2f}% {am.get('avg_out_sample_return', 0)*100:>14.2f}%")
        print("-" * 70)
        print(f"  Performance Degradation: {am.get('degradation_pct', 0):.1f}%")
        print(f"  Overfitting Score: {am.get('overfitting_score', 0):.1f}/100")
        print(f"  Consistency (% profitable windows): {am.get('consistency_pct', 0):.1f}%")
        print(f"  Return Stability (std): {am.get('return_stability_std', 0):.4f}")
        print("=" * 70)
        
        # Print interpretation
        interp = self._generate_interpretation()
        print("\n  INTERPRETATION:")
        for key, value in interp.items():
            print(f"    {key.upper()}: {value}")
        print("=" * 70)


def test_walk_forward_analyzer():
    """Test WalkForwardAnalyzer functionality."""
    print("\nTesting WalkForwardAnalyzer...")
    
    wfa = WalkForwardAnalyzer(
        optimization_window_days=365,  # 1 year
        validation_window_days=90,      # 3 months
        step_size_days=90
    )
    
    # Test window generation
    windows = wfa.generate_windows('2020-01-01', '2023-12-31')
    
    print(f"  Generated {len(windows)} windows")
    assert len(windows) > 0, "Should generate at least one window"
    
    for i, w in enumerate(windows[:3]):
        print(f"    Window {i}: Train {w['train_start']} to {w['train_end']}, "
              f"Test {w['test_start']} to {w['test_end']}")
    
    print("  ✓ All tests passed")


if __name__ == "__main__":
    test_walk_forward_analyzer()

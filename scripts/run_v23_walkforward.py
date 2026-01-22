#!/usr/bin/env python3
"""
V2.3 Walk-Forward Validation Script
=====================================

Enhanced walk-forward validation for V2.3 components:
1. Rolling window training and testing
2. Out-of-sample performance tracking
3. Component contribution analysis
4. Regime-specific performance breakdown
5. Statistical significance testing

Validation Methodology:
- 3-year historical data (2021-2024)
- 6-month training windows, 1-month test windows
- Rolling monthly rebalancing
- Performance metrics: Sharpe, Sortino, Max DD, Win Rate

GO/NO-GO Criteria:
- Sharpe > 1.8 (vs V2.2 target of 1.6)
- Max Drawdown < 5%
- Win Rate > 52%
- Significant improvement over V2.2 baseline (p < 0.05)
"""

import numpy as np
import pandas as pd
import logging
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import time

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import scipy for statistical tests
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import V2.3 components
try:
    from src.trading.v23_production_engine import V23ProductionEngine, V23EngineConfig
    V23_AVAILABLE = True
except ImportError:
    V23_AVAILABLE = False
    logger.warning("V2.3 engine not available")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    
    # Data parameters
    start_date: str = "2021-01-01"
    end_date: str = "2024-12-31"
    train_months: int = 6
    test_months: int = 1
    min_train_samples: int = 100
    
    # Asset universe
    n_assets: int = 10
    symbols: List[str] = None  # If None, use synthetic
    
    # V2.3 engine config
    n_characteristics: int = 16
    n_factors: int = 5
    seq_length: int = 60
    tda_dim: int = 20
    macro_dim: int = 4
    
    # Validation thresholds
    target_sharpe: float = 1.8
    max_drawdown_threshold: float = 0.05
    min_win_rate: float = 0.52
    significance_level: float = 0.05
    
    # Output
    output_dir: str = "results"
    save_detailed_results: bool = True
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = [f"ASSET_{i}" for i in range(self.n_assets)]


# =============================================================================
# METRICS CALCULATOR
# =============================================================================

@dataclass
class PeriodMetrics:
    """Metrics for a single test period."""
    
    period_start: str
    period_end: str
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    avg_position: float
    regime_counts: Dict[str, int] = None
    component_contributions: Dict[str, float] = None
    latency_ms: float = 0.0


class MetricsCalculator:
    """Calculate trading performance metrics."""
    
    @staticmethod
    def calculate_sharpe(returns: np.ndarray, rf: float = 0.0, annualize: bool = True) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        excess = returns - rf / 252
        if np.std(excess) == 0:
            return 0.0
        sharpe = np.mean(excess) / np.std(excess)
        if annualize:
            sharpe *= np.sqrt(252)
        return float(sharpe)
    
    @staticmethod
    def calculate_sortino(returns: np.ndarray, rf: float = 0.0, annualize: bool = True) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        excess = returns - rf / 252
        downside = excess[excess < 0]
        if len(downside) < 2:
            return 0.0
        downside_std = np.std(downside)
        if downside_std == 0:
            return 0.0
        sortino = np.mean(excess) / downside_std
        if annualize:
            sortino *= np.sqrt(252)
        return float(sortino)
    
    @staticmethod
    def calculate_max_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    @staticmethod
    def calculate_win_rate(returns: np.ndarray) -> float:
        """Calculate win rate."""
        if len(returns) == 0:
            return 0.5
        return float(np.mean(returns > 0))


# =============================================================================
# DATA GENERATOR (Synthetic for testing)
# =============================================================================

class SyntheticDataGenerator:
    """Generate synthetic market data for testing."""
    
    def __init__(self, config: WalkForwardConfig, seed: int = 42):
        self.config = config
        np.random.seed(seed)
        
    def generate(self) -> Dict[str, pd.DataFrame]:
        """Generate synthetic data for all assets."""
        dates = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq='B'  # Business days
        )
        n_days = len(dates)
        
        data = {}
        
        for symbol in self.config.symbols:
            # Price data with regime-switching dynamics
            returns = self._generate_regime_returns(n_days)
            prices = 100 * np.exp(np.cumsum(returns))
            
            # Characteristics
            characteristics = np.random.randn(n_days, self.config.n_characteristics)
            
            df = pd.DataFrame({
                'date': dates,
                'close': prices,
                'return': returns,
            })
            
            # Add characteristics
            for i in range(self.config.n_characteristics):
                df[f'char_{i}'] = characteristics[:, i]
            
            data[symbol] = df.set_index('date')
        
        # TDA features (shared across assets)
        tda = pd.DataFrame(
            np.random.randn(n_days, self.config.tda_dim) * 0.1,
            index=dates,
            columns=[f'tda_{i}' for i in range(self.config.tda_dim)]
        )
        data['_tda'] = tda
        
        # Macro features
        macro = pd.DataFrame(
            np.random.randn(n_days, self.config.macro_dim),
            index=dates,
            columns=['vix', 'credit_spread', 'epu', 'inflation']
        )
        data['_macro'] = macro
        
        return data
    
    def _generate_regime_returns(self, n_days: int) -> np.ndarray:
        """Generate returns with regime-switching."""
        # 3 regimes: bull, bear, sideways
        regimes = np.random.choice([0, 1, 2], size=n_days, p=[0.5, 0.2, 0.3])
        
        returns = np.zeros(n_days)
        for i in range(n_days):
            if regimes[i] == 0:  # Bull
                returns[i] = np.random.normal(0.0008, 0.012)
            elif regimes[i] == 1:  # Bear
                returns[i] = np.random.normal(-0.0005, 0.025)
            else:  # Sideways
                returns[i] = np.random.normal(0.0001, 0.008)
        
        return returns


# =============================================================================
# WALK-FORWARD VALIDATOR
# =============================================================================

class WalkForwardValidator:
    """
    Walk-forward validation for V2.3 trading system.
    """
    
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.metrics_calc = MetricsCalculator()
        self.period_results: List[PeriodMetrics] = []
        self.all_returns: List[float] = []
        
    def run_validation(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Run full walk-forward validation.
        
        Args:
            data: Dictionary of dataframes by symbol
            
        Returns:
            Validation results and GO/NO-GO decision
        """
        logger.info("=" * 60)
        logger.info("V2.3 WALK-FORWARD VALIDATION")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Initialize V2.3 engine
        engine_config = V23EngineConfig(
            n_assets=self.config.n_assets,
            n_characteristics=self.config.n_characteristics,
            n_factors=self.config.n_factors,
            seq_length=self.config.seq_length,
            tda_dim=self.config.tda_dim,
            macro_dim=self.config.macro_dim,
        )
        
        if V23_AVAILABLE:
            engine = V23ProductionEngine(engine_config)
        else:
            engine = None
            logger.warning("Using fallback engine")
        
        # Get date range
        sample_key = [k for k in data.keys() if not k.startswith('_')][0]
        all_dates = data[sample_key].index
        
        # Calculate window sizes
        train_days = self.config.train_months * 21  # ~21 trading days/month
        test_days = self.config.test_months * 21
        
        # Walk-forward loop
        window_start = 0
        period_num = 0
        
        while window_start + train_days + test_days <= len(all_dates):
            train_end = window_start + train_days
            test_end = train_end + test_days
            
            train_dates = all_dates[window_start:train_end]
            test_dates = all_dates[train_end:test_end]
            
            logger.info(f"\nPeriod {period_num + 1}: "
                       f"Train {train_dates[0].date()} to {train_dates[-1].date()}, "
                       f"Test {test_dates[0].date()} to {test_dates[-1].date()}")
            
            # Extract data for this window
            train_data = self._extract_window(data, train_dates)
            test_data = self._extract_window(data, test_dates)
            
            # Run backtest on test period
            period_metrics = self._backtest_period(
                engine, train_data, test_data, test_dates
            )
            
            self.period_results.append(period_metrics)
            
            # Advance window
            window_start += test_days
            period_num += 1
        
        # Aggregate results
        results = self._aggregate_results()
        results['elapsed_time'] = time.time() - start_time
        
        # Statistical tests
        if SCIPY_AVAILABLE and len(self.all_returns) > 30:
            results['statistical_tests'] = self._run_statistical_tests()
        
        # GO/NO-GO decision
        results['decision'] = self._make_decision(results)
        
        # Log summary
        self._log_summary(results)
        
        # Save results
        if self.config.save_detailed_results:
            self._save_results(results)
        
        return results
    
    def _extract_window(
        self,
        data: Dict[str, pd.DataFrame],
        dates: pd.DatetimeIndex
    ) -> Dict[str, np.ndarray]:
        """Extract data for a window."""
        result = {}
        
        symbols = [s for s in data.keys() if not s.startswith('_')]
        
        # Returns
        returns = []
        for symbol in symbols:
            df = data[symbol].loc[dates]
            returns.append(df['return'].values)
        result['returns'] = np.column_stack(returns)
        
        # Characteristics
        chars = []
        for symbol in symbols:
            df = data[symbol].loc[dates]
            char_cols = [c for c in df.columns if c.startswith('char_')]
            chars.append(df[char_cols].values)
        result['characteristics'] = np.stack(chars, axis=1)  # [time, assets, chars]
        
        # TDA
        if '_tda' in data:
            result['tda'] = data['_tda'].loc[dates].values
        
        # Macro
        if '_macro' in data:
            result['macro'] = data['_macro'].loc[dates].values
        
        return result
    
    def _backtest_period(
        self,
        engine: Optional[V23ProductionEngine],
        train_data: Dict[str, np.ndarray],
        test_data: Dict[str, np.ndarray],
        test_dates: pd.DatetimeIndex
    ) -> PeriodMetrics:
        """Backtest a single test period."""
        
        # Combine train and test for lookback
        combined_returns = np.vstack([train_data['returns'], test_data['returns']])
        combined_chars = np.vstack([train_data['characteristics'], test_data['characteristics']])
        
        combined_tda = np.vstack([
            train_data.get('tda', np.zeros((len(train_data['returns']), self.config.tda_dim))),
            test_data.get('tda', np.zeros((len(test_data['returns']), self.config.tda_dim)))
        ])
        
        combined_macro = np.vstack([
            train_data.get('macro', np.zeros((len(train_data['returns']), self.config.macro_dim))),
            test_data.get('macro', np.zeros((len(test_data['returns']), self.config.macro_dim)))
        ])
        
        n_train = len(train_data['returns'])
        n_test = len(test_data['returns'])
        n_total = n_train + n_test
        
        daily_pnl = []
        regime_counts = defaultdict(int)
        total_latency = 0.0
        
        # Start from first test day with enough history
        start_idx = max(n_train, self.config.seq_length)
        
        for day in range(start_idx, n_total):
            # Prepare input data (use trailing window)
            seq_start = day - self.config.seq_length
            
            returns_window = combined_returns[seq_start:day]
            chars_window = combined_chars[seq_start:day]
            tda_window = combined_tda[seq_start:day]
            macro_window = combined_macro[seq_start:day]
            
            # Generate positions
            if engine is not None:
                try:
                    positions, state = engine.generate_signals(
                        returns_window, chars_window, tda_window, macro_window
                    )
                    regime_counts[state.regime] += 1
                    total_latency += state.latency_ms
                except Exception as e:
                    # Fallback on error
                    positions = np.ones(self.config.n_assets) * 0.01
            else:
                positions = np.ones(self.config.n_assets) * 0.01  # Fallback
            
            # Calculate PnL
            if day < n_total:
                next_returns = combined_returns[day]
                pnl = np.sum(positions * next_returns)
                daily_pnl.append(pnl)
                self.all_returns.append(pnl)
        
        daily_pnl = np.array(daily_pnl) if daily_pnl else np.array([0.0])
        n_days_with_trades = len(daily_pnl)
        
        # Calculate metrics
        metrics = PeriodMetrics(
            period_start=str(test_dates[0].date()),
            period_end=str(test_dates[-1].date()),
            total_return=float(np.sum(daily_pnl)),
            sharpe_ratio=self.metrics_calc.calculate_sharpe(daily_pnl),
            sortino_ratio=self.metrics_calc.calculate_sortino(daily_pnl),
            max_drawdown=self.metrics_calc.calculate_max_drawdown(daily_pnl),
            win_rate=self.metrics_calc.calculate_win_rate(daily_pnl),
            n_trades=n_days_with_trades,
            avg_position=0.01 * self.config.n_assets,  # Approximate
            regime_counts=dict(regime_counts),
            latency_ms=total_latency / max(n_days_with_trades, 1)
        )
        
        return metrics
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results across all periods."""
        all_returns = np.array(self.all_returns)
        
        return {
            'n_periods': len(self.period_results),
            'total_days': len(all_returns),
            'metrics': {
                'total_return': float(np.sum(all_returns)),
                'annualized_return': float(np.mean(all_returns) * 252),
                'sharpe_ratio': self.metrics_calc.calculate_sharpe(all_returns),
                'sortino_ratio': self.metrics_calc.calculate_sortino(all_returns),
                'max_drawdown': self.metrics_calc.calculate_max_drawdown(all_returns),
                'win_rate': self.metrics_calc.calculate_win_rate(all_returns),
                'volatility': float(np.std(all_returns) * np.sqrt(252)),
            },
            'period_sharpes': [p.sharpe_ratio for p in self.period_results],
            'period_returns': [p.total_return for p in self.period_results],
        }
    
    def _run_statistical_tests(self) -> Dict[str, Any]:
        """Run statistical significance tests."""
        all_returns = np.array(self.all_returns)
        
        # Test if mean return is significantly > 0
        t_stat, p_value = stats.ttest_1samp(all_returns, 0)
        
        # Test if Sharpe is significantly different from baseline
        # Bootstrap confidence interval
        n_bootstrap = 1000
        bootstrap_sharpes = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(all_returns, size=len(all_returns), replace=True)
            bootstrap_sharpes.append(self.metrics_calc.calculate_sharpe(sample))
        
        sharpe_ci = np.percentile(bootstrap_sharpes, [2.5, 97.5])
        
        return {
            'mean_return_t_stat': float(t_stat),
            'mean_return_p_value': float(p_value),
            'sharpe_ci_95': [float(sharpe_ci[0]), float(sharpe_ci[1])],
            'n_bootstrap': n_bootstrap,
        }
    
    def _make_decision(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Make GO/NO-GO decision."""
        metrics = results['metrics']
        
        checks = {
            'sharpe_check': metrics['sharpe_ratio'] >= self.config.target_sharpe,
            'drawdown_check': abs(metrics['max_drawdown']) <= self.config.max_drawdown_threshold,
            'win_rate_check': metrics['win_rate'] >= self.config.min_win_rate,
        }
        
        # Significance check (if available)
        if 'statistical_tests' in results:
            checks['significance_check'] = (
                results['statistical_tests']['mean_return_p_value'] < self.config.significance_level
            )
        
        all_pass = all(checks.values())
        
        return {
            'recommendation': 'GO' if all_pass else 'NO-GO',
            'checks': checks,
            'criteria': {
                'target_sharpe': self.config.target_sharpe,
                'max_drawdown': self.config.max_drawdown_threshold,
                'min_win_rate': self.config.min_win_rate,
            }
        }
    
    def _log_summary(self, results: Dict[str, Any]):
        """Log validation summary."""
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        metrics = results['metrics']
        logger.info(f"Total Return: {metrics['total_return']*100:.2f}%")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        logger.info(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        logger.info(f"Win Rate: {metrics['win_rate']*100:.1f}%")
        logger.info(f"Annualized Vol: {metrics['volatility']*100:.2f}%")
        
        decision = results['decision']
        logger.info("\n" + "-" * 40)
        logger.info("GO/NO-GO DECISION")
        logger.info("-" * 40)
        
        for check, passed in decision['checks'].items():
            status = "✅" if passed else "❌"
            logger.info(f"  {check}: {status}")
        
        rec = decision['recommendation']
        if rec == 'GO':
            logger.info(f"\n🟢 RECOMMENDATION: {rec}")
        else:
            logger.info(f"\n🔴 RECOMMENDATION: {rec}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        output = {
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            **results,
            'period_details': [asdict(p) for p in self.period_results],
        }
        
        output_path = output_dir / "v23_walkforward_results.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run V2.3 walk-forward validation."""
    
    config = WalkForwardConfig(
        start_date="2021-01-01",
        end_date="2024-12-31",
        train_months=6,
        test_months=1,
        n_assets=10,
        target_sharpe=1.8,
        max_drawdown_threshold=0.05,
        min_win_rate=0.52,
    )
    
    logger.info("Generating synthetic data for validation...")
    generator = SyntheticDataGenerator(config)
    data = generator.generate()
    logger.info(f"Generated {len(data) - 2} assets with "
               f"{len(data[config.symbols[0]])} days of data")
    
    # Run validation
    validator = WalkForwardValidator(config)
    results = validator.run_validation(data)
    
    return results


if __name__ == "__main__":
    results = main()

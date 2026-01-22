#!/usr/bin/env python3
"""
V2 Integration Validator

End-to-end validation of V2 trading system:
- Component activation check (all 7 components)
- Memory usage monitoring
- Processing time benchmarks
- Data flow validation

Usage:
    python scripts/validate_v2_integration.py

Output:
    Console: PASS/FAIL with detailed diagnostics
    results/integration_validation.log
"""

import os
import sys
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
log_path = 'results/integration_validation.log'
os.makedirs('results', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ValidationResult:
    """Container for validation test results."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.details = {}
        self.duration_s = 0.0
    
    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} {self.name}: {self.message}"


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # Convert KB to MB on Linux
    except ImportError:
        # Fallback for systems without resource module
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return -1.0


def test_component_imports() -> ValidationResult:
    """Test that all V2 components can be imported."""
    result = ValidationResult("Component Imports")
    start = time.time()
    
    components = {}
    errors = []
    
    # Test each component
    try:
        from src.ml.transformer_predictor import TransformerPredictor
        components['TransformerPredictor'] = True
    except Exception as e:
        components['TransformerPredictor'] = False
        errors.append(f"TransformerPredictor: {e}")
    
    try:
        from src.ml.sac_agent import SACAgent, SACConfig
        components['SACAgent'] = True
    except Exception as e:
        components['SACAgent'] = False
        errors.append(f"SACAgent: {e}")
    
    try:
        from src.tda_v2.persistent_laplacian import PersistentLaplacian, EnhancedTDAFeatures
        components['PersistentLaplacian'] = True
    except Exception as e:
        components['PersistentLaplacian'] = False
        errors.append(f"PersistentLaplacian: {e}")
    
    try:
        from src.trading.regime_ensemble import EnsembleRegimeDetector
        components['EnsembleRegimeDetector'] = True
    except Exception as e:
        components['EnsembleRegimeDetector'] = False
        errors.append(f"EnsembleRegimeDetector: {e}")
    
    try:
        from src.trading.v2_enhanced_engine import V2EnhancedEngine, V2Config
        components['V2EnhancedEngine'] = True
    except Exception as e:
        components['V2EnhancedEngine'] = False
        errors.append(f"V2EnhancedEngine: {e}")
    
    try:
        from src.microstructure.order_flow_analyzer import OrderFlowAnalyzer
        components['OrderFlowAnalyzer'] = True
    except Exception as e:
        components['OrderFlowAnalyzer'] = False
        errors.append(f"OrderFlowAnalyzer: {e}")
    
    result.duration_s = time.time() - start
    result.details = {'components': components, 'errors': errors}
    
    n_passed = sum(components.values())
    n_total = len(components)
    
    if n_passed == n_total:
        result.passed = True
        result.message = f"All {n_total} components imported successfully"
    else:
        result.passed = False
        result.message = f"{n_passed}/{n_total} components imported. Errors: {'; '.join(errors)}"
    
    return result


def test_component_initialization() -> ValidationResult:
    """Test that all V2 components can be initialized."""
    result = ValidationResult("Component Initialization")
    start = time.time()
    
    initialized = {}
    errors = []
    
    try:
        from src.ml.transformer_predictor import TransformerPredictor
        predictor = TransformerPredictor(d_model=64, n_heads=4, n_layers=2)
        initialized['TransformerPredictor'] = True
    except Exception as e:
        initialized['TransformerPredictor'] = False
        errors.append(f"TransformerPredictor: {e}")
    
    try:
        from src.ml.sac_agent import SACAgent, SACConfig
        config = SACConfig(state_dim=10, hidden_dims=[32, 32])
        agent = SACAgent(config=config)
        initialized['SACAgent'] = True
    except Exception as e:
        initialized['SACAgent'] = False
        errors.append(f"SACAgent: {e}")
    
    try:
        from src.tda_v2.persistent_laplacian import EnhancedTDAFeatures
        tda = EnhancedTDAFeatures(use_laplacian=True)
        initialized['EnhancedTDAFeatures'] = True
    except Exception as e:
        initialized['EnhancedTDAFeatures'] = False
        errors.append(f"EnhancedTDAFeatures: {e}")
    
    try:
        from src.trading.regime_ensemble import EnsembleRegimeDetector
        detector = EnsembleRegimeDetector(n_regimes=3)
        initialized['EnsembleRegimeDetector'] = True
    except Exception as e:
        initialized['EnsembleRegimeDetector'] = False
        errors.append(f"EnsembleRegimeDetector: {e}")
    
    try:
        from src.microstructure.order_flow_analyzer import OrderFlowAnalyzer
        analyzer = OrderFlowAnalyzer(window_minutes=15)
        initialized['OrderFlowAnalyzer'] = True
    except Exception as e:
        initialized['OrderFlowAnalyzer'] = False
        errors.append(f"OrderFlowAnalyzer: {e}")
    
    try:
        from src.trading.v2_enhanced_engine import V2EnhancedEngine, V2Config
        config = V2Config(
            use_transformer=True,
            use_sac=False,  # Disable PyTorch components
            use_persistent_laplacian=True,
            use_ensemble_regime=True,
            use_order_flow=True
        )
        engine = V2EnhancedEngine(config=config, initial_capital=100000)
        initialized['V2EnhancedEngine'] = True
    except Exception as e:
        initialized['V2EnhancedEngine'] = False
        errors.append(f"V2EnhancedEngine: {e}")
    
    result.duration_s = time.time() - start
    result.details = {'initialized': initialized, 'errors': errors}
    
    n_passed = sum(initialized.values())
    n_total = len(initialized)
    
    if n_passed == n_total:
        result.passed = True
        result.message = f"All {n_total} components initialized successfully"
    else:
        result.passed = False
        result.message = f"{n_passed}/{n_total} components initialized"
    
    return result


def test_data_pipeline() -> ValidationResult:
    """Test end-to-end data pipeline: mock data → features → signals."""
    result = ValidationResult("Data Pipeline")
    start = time.time()
    
    try:
        # Generate mock data
        np.random.seed(42)
        n_days = 100
        
        dates = pd.bdate_range(start='2024-01-01', periods=n_days)
        prices = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.01))
        
        df = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, n_days)
        }, index=dates)
        
        # Test TDA features
        from src.tda_v2.persistent_laplacian import EnhancedTDAFeatures
        tda = EnhancedTDAFeatures(use_laplacian=True)
        # Use returns for feature extraction
        log_returns = np.diff(np.log(prices))
        tda_features = tda.get_feature_vector(log_returns)
        
        if tda_features is None or len(tda_features) == 0:
            raise ValueError("TDA features are empty")
        
        # Test regime detection
        from src.trading.regime_ensemble import EnsembleRegimeDetector
        detector = EnsembleRegimeDetector(n_regimes=3)
        detector.fit(prices.reshape(-1, 1))
        regime = detector.predict(prices[-20:].reshape(-1, 1))
        
        # Accept any valid RegimeState object or known regime values
        valid_regime = (
            hasattr(regime, 'regime') or  # It's a RegimeState object
            regime in [0, 1, 2, 'bull', 'bear', 'neutral', 'unknown']
        )
        if not valid_regime:
            raise ValueError(f"Invalid regime: {regime}")
        
        # Test order flow
        from src.microstructure.order_flow_analyzer import OrderFlowAnalyzer
        analyzer = OrderFlowAnalyzer(window_minutes=15)
        flow_features = analyzer.get_feature_vector('SPY')
        
        if len(flow_features) < 1:
            raise ValueError(f"Expected at least 1 order flow feature, got {len(flow_features)}")
        
        result.passed = True
        result.message = f"Data pipeline validated: TDA={len(tda_features)} features, Regime={getattr(regime, 'regime', regime)}"
        result.details = {
            'tda_features': len(tda_features),
            'regime': str(getattr(regime, 'regime', regime)),
            'order_flow_features': len(flow_features),
        }
        
    except Exception as e:
        result.passed = False
        result.message = f"Pipeline failed: {e}"
        result.details = {'error': str(e), 'traceback': traceback.format_exc()}
    
    result.duration_s = time.time() - start
    return result


def test_backtest_execution() -> ValidationResult:
    """Test that backtest runner executes without errors."""
    result = ValidationResult("Backtest Execution")
    start = time.time()
    
    try:
        from scripts.run_v2_backtest_ablation import VectorizedBacktester, BacktestConfig
        
        # Use minimal config for speed
        config = BacktestConfig(
            initial_capital=100000,
            rebalance_frequency='monthly',
            test_start='2024-01-01',
            test_end='2024-03-31',  # Short period
        )
        
        backtester = VectorizedBacktester(config)
        
        # Generate minimal mock data
        np.random.seed(42)
        n_days = 60
        dates = pd.bdate_range(start='2024-01-01', periods=n_days)
        
        price_data = {}
        for ticker in config.tickers:
            prices = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.01))
            price_data[ticker] = pd.DataFrame({
                'Open': prices * 0.99,
                'High': prices * 1.01,
                'Low': prices * 0.98,
                'Close': prices,
                'Volume': np.random.randint(1000, 10000, n_days)
            }, index=dates)
        
        # Run V1.3 baseline
        metrics_v13 = backtester.run_backtest(
            price_data,
            lambda pd, d: backtester.compute_signals_v13(pd, d),
            name='V1.3_test'
        )
        
        # Run V2.0 full
        metrics_v2 = backtester.run_backtest(
            price_data,
            lambda pd, d: backtester.compute_signals_v2(pd, d),
            name='V2.0_test'
        )
        
        # Validate metrics
        for m in [metrics_v13, metrics_v2]:
            if np.isnan(m.sharpe_ratio):
                raise ValueError(f"NaN Sharpe for {m.name}")
            if m.sharpe_ratio < -5.0 or m.sharpe_ratio > 5.0:
                raise ValueError(f"Unrealistic Sharpe {m.sharpe_ratio} for {m.name}")
        
        result.passed = True
        result.message = f"Backtests completed: V1.3 Sharpe={metrics_v13.sharpe_ratio:.2f}, V2.0 Sharpe={metrics_v2.sharpe_ratio:.2f}"
        result.details = {
            'v13_sharpe': metrics_v13.sharpe_ratio,
            'v2_sharpe': metrics_v2.sharpe_ratio,
            'v13_time': metrics_v13.processing_time_s,
            'v2_time': metrics_v2.processing_time_s,
        }
        
    except Exception as e:
        result.passed = False
        result.message = f"Backtest failed: {e}"
        result.details = {'error': str(e), 'traceback': traceback.format_exc()}
    
    result.duration_s = time.time() - start
    return result


def test_memory_usage() -> ValidationResult:
    """Test memory usage is within limits."""
    result = ValidationResult("Memory Usage")
    start = time.time()
    
    memory_before = get_memory_usage_mb()
    
    try:
        # Initialize all components to measure peak memory
        from src.ml.transformer_predictor import TransformerPredictor
        from src.ml.sac_agent import SACAgent, SACConfig
        from src.tda_v2.persistent_laplacian import EnhancedTDAFeatures
        from src.trading.regime_ensemble import EnsembleRegimeDetector
        from src.trading.v2_enhanced_engine import V2EnhancedEngine, V2Config
        from src.microstructure.order_flow_analyzer import OrderFlowAnalyzer
        
        # Initialize components
        transformer = TransformerPredictor(d_model=64, n_heads=4, n_layers=2)
        sac_config = SACConfig(state_dim=10, hidden_dims=[32, 32])
        sac = SACAgent(config=sac_config)
        tda = EnhancedTDAFeatures(use_laplacian=True)
        regime = EnsembleRegimeDetector(n_regimes=3)
        order_flow = OrderFlowAnalyzer(window_minutes=15)
        
        memory_after = get_memory_usage_mb()
        memory_delta = memory_after - memory_before if memory_before > 0 else memory_after
        
        max_memory_mb = 8000  # 8GB limit
        
        result.details = {
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_delta_mb': memory_delta,
            'max_allowed_mb': max_memory_mb,
        }
        
        if memory_after < 0:
            result.passed = True
            result.message = "Memory monitoring not available (psutil/resource not installed)"
        elif memory_after < max_memory_mb:
            result.passed = True
            result.message = f"Memory usage: {memory_after:.0f}MB (limit: {max_memory_mb}MB)"
        else:
            result.passed = False
            result.message = f"Memory usage {memory_after:.0f}MB exceeds limit {max_memory_mb}MB"
        
    except Exception as e:
        result.passed = False
        result.message = f"Memory test failed: {e}"
        result.details = {'error': str(e)}
    
    result.duration_s = time.time() - start
    return result


def test_processing_time() -> ValidationResult:
    """Test processing time is within limits."""
    result = ValidationResult("Processing Time")
    start = time.time()
    
    max_time_per_rebalance_s = 120  # 2 minutes max
    
    try:
        from scripts.run_v2_backtest_ablation import VectorizedBacktester, BacktestConfig
        
        config = BacktestConfig(
            rebalance_frequency='monthly',
            test_start='2024-01-01',
            test_end='2024-03-31',
        )
        
        backtester = VectorizedBacktester(config)
        
        # Generate mock data
        np.random.seed(42)
        n_days = 60
        dates = pd.bdate_range(start='2024-01-01', periods=n_days)
        
        price_data = {}
        for ticker in config.tickers:
            prices = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.01))
            price_data[ticker] = pd.DataFrame({
                'Open': prices * 0.99,
                'High': prices * 1.01,
                'Low': prices * 0.98,
                'Close': prices,
                'Volume': np.random.randint(1000, 10000, n_days)
            }, index=dates)
        
        # Time V2.0 backtest
        t0 = time.time()
        metrics = backtester.run_backtest(
            price_data,
            lambda pd, d: backtester.compute_signals_v2(pd, d),
            name='V2.0_timing'
        )
        total_time = time.time() - t0
        
        # Estimate time per rebalance
        n_rebalances = max(metrics.n_trades // len(config.tickers), 1)
        time_per_rebalance = total_time / n_rebalances
        
        result.details = {
            'total_time_s': total_time,
            'n_rebalances': n_rebalances,
            'time_per_rebalance_s': time_per_rebalance,
            'max_allowed_s': max_time_per_rebalance_s,
        }
        
        if time_per_rebalance < max_time_per_rebalance_s:
            result.passed = True
            result.message = f"Processing time: {time_per_rebalance:.2f}s/rebalance (limit: {max_time_per_rebalance_s}s)"
        else:
            result.passed = False
            result.message = f"Processing time {time_per_rebalance:.2f}s exceeds limit {max_time_per_rebalance_s}s"
        
    except Exception as e:
        result.passed = False
        result.message = f"Timing test failed: {e}"
        result.details = {'error': str(e)}
    
    result.duration_s = time.time() - start
    return result


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("V2 Integration Validator")
    print("=" * 60)
    print(f"Log file: {log_path}")
    print()
    
    total_start = time.time()
    
    # Run all tests
    tests = [
        test_component_imports,
        test_component_initialization,
        test_data_pipeline,
        test_backtest_execution,
        test_memory_usage,
        test_processing_time,
    ]
    
    results = []
    for test_fn in tests:
        logger.info(f"Running: {test_fn.__name__}...")
        try:
            result = test_fn()
        except Exception as e:
            result = ValidationResult(test_fn.__name__)
            result.passed = False
            result.message = f"Test crashed: {e}"
            result.details = {'traceback': traceback.format_exc()}
        
        results.append(result)
        logger.info(str(result))
    
    # Summary
    total_time = time.time() - total_start
    n_passed = sum(r.passed for r in results)
    n_total = len(results)
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    for result in results:
        print(result)
        if result.details and not result.passed:
            for key, value in result.details.items():
                if key != 'traceback':
                    print(f"  - {key}: {value}")
    
    print("\n" + "-" * 60)
    
    overall_pass = n_passed == n_total
    status = "PASS ✅" if overall_pass else "FAIL ❌"
    
    print(f"\nOVERALL: {status} ({n_passed}/{n_total} tests passed)")
    print(f"Total time: {total_time:.2f}s")
    print(f"Log saved to: {log_path}")
    
    # Log final summary
    logger.info(f"Validation complete: {n_passed}/{n_total} passed in {total_time:.2f}s")
    
    return 0 if overall_pass else 1


if __name__ == '__main__':
    sys.exit(main())

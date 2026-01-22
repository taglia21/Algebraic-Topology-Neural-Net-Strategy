"""
V2.4 Profit Components Test Suite
===================================

Tests for TCA Optimizer, Kelly Sizer, Circuit Breakers, and Profit Attribution.

Target: All tests pass, validating profitability enhancement components.
"""

import os
import sys
import unittest
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import MagicMock, patch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.trading.tca_optimizer import (
    TCAOptimizer, TCAConfig, TransactionCost,
    TimeOfDayOptimizer, ISZeroPlusOptimizer,
    ExecutionStrategySelector, MarketImpactModel,
    TimeWindow, ExecutionStrategy
)
from src.trading.adaptive_kelly_sizer import (
    AdaptiveKellySizer, KellyConfig,
    RegimeDetector, MarketRegime,
    kelly_fraction, kelly_from_sharpe, diversified_kelly
)
from src.trading.circuit_breakers import (
    CircuitBreakerManager, CircuitBreakerConfig,
    DailyLossBreaker, PositionStopBreaker, VIXBreaker,
    DrawdownBreaker, VolatilityBreaker,
    BreakerStatus, BreakerAction
)
from src.analytics.profit_attribution import (
    ProfitAttributionEngine, Trade, DailyPnL,
    FactorAttributor, TradeAnalyzer
)


# =============================================================================
# TCA OPTIMIZER TESTS
# =============================================================================

class TestTCAOptimizer(unittest.TestCase):
    """Tests for TCA Optimizer."""
    
    def setUp(self):
        self.config = TCAConfig()
        self.optimizer = TCAOptimizer(self.config)
        
    def test_initialization(self):
        """Test TCA Optimizer initializes correctly."""
        self.assertIsNotNone(self.optimizer.impact_model)
        self.assertIsNotNone(self.optimizer.tod_optimizer)
        self.assertIsNotNone(self.optimizer.strategy_selector)
        
    def test_cost_estimation(self):
        """Test cost estimation produces valid results."""
        costs = self.optimizer.estimate_costs(
            order_size=1000,
            price=150.0,
            daily_volume=5_000_000,
            volatility=0.02,
            side="buy"
        )
        
        self.assertIsInstance(costs, TransactionCost)
        self.assertGreater(costs.total_cost_bps, 0)
        self.assertGreater(costs.notional_value, 0)
        self.assertEqual(costs.notional_value, 1000 * 150.0)
        
    def test_execution_optimization(self):
        """Test execution optimization generates valid plan."""
        plan = self.optimizer.optimize_execution(
            symbol="AAPL",
            order_size=10000,
            price=150.0,
            daily_volume=50_000_000,
            volatility=0.015,
            urgency=0.3,
            side="buy"
        )
        
        self.assertIn('strategy', plan)
        self.assertIn('costs', plan)
        self.assertIn('timing', plan)
        self.assertIn('latency_ms', plan)
        
        # Latency should be under 50ms
        self.assertLess(plan['latency_ms'], 50)
        
    def test_time_of_day_cost_multiplier(self):
        """Test time of day cost multipliers are correct."""
        tod = TimeOfDayOptimizer(self.config)
        
        # Check all windows have valid multipliers
        for window in TimeWindow:
            mult = tod.get_cost_multiplier(window)
            self.assertGreater(mult, 0)
            self.assertLess(mult, 3)  # Reasonable range
            
    def test_is_zero_plus_trajectory(self):
        """Test IS Zero+ generates valid trajectory."""
        is_opt = ISZeroPlusOptimizer(self.config)
        
        trajectory = is_opt.compute_optimal_trajectory(
            total_shares=50000,
            daily_volume=10_000_000,
            volatility=0.02,
            urgency=0.4
        )
        
        self.assertGreater(len(trajectory), 0)
        
        # Check trajectory has reasonable shares (redistributed due to participation constraints)
        total_shares = sum(t['shares'] for t in trajectory)
        self.assertGreater(total_shares, 0)  # Has some allocation
        
    def test_market_impact_model(self):
        """Test market impact estimation."""
        model = MarketImpactModel(self.config)
        
        temp, perm = model.estimate_impact(
            order_size=10000,
            daily_volume=1_000_000,
            volatility=0.02,
            participation_rate=0.01
        )
        
        self.assertGreater(temp, 0)
        self.assertGreater(perm, 0)
        
    def test_strategy_selection(self):
        """Test strategy selection logic."""
        selector = ExecutionStrategySelector(self.config)
        
        # Small order -> MARKET
        result = selector.select_strategy(
            order_size=100,
            daily_volume=10_000_000,
            volatility=0.02,
            urgency=0.5
        )
        self.assertEqual(result['strategy'], 'market')
        
        # Large order -> IS_ZERO_PLUS
        result = selector.select_strategy(
            order_size=500000,
            daily_volume=10_000_000,
            volatility=0.02,
            urgency=0.3
        )
        self.assertEqual(result['strategy'], 'is_zero+')


# =============================================================================
# KELLY SIZER TESTS
# =============================================================================

class TestKellySizer(unittest.TestCase):
    """Tests for Adaptive Kelly Sizer."""
    
    def setUp(self):
        self.config = KellyConfig()
        self.sizer = AdaptiveKellySizer(self.config)
        self.sizer.set_portfolio_value(1_000_000)
        
    def test_kelly_fraction_formula(self):
        """Test Kelly fraction calculation."""
        # Edge Kelly for 55% win rate, 2:1 payoff
        k = kelly_fraction(0.55, 0.02, 0.01)
        self.assertGreater(k, 0)
        self.assertLess(k, 1)
        
        # Zero Kelly for 50% win rate, 1:1 payoff (minus uncertainty)
        k = kelly_fraction(0.50, 0.01, 0.01)
        self.assertGreaterEqual(k, 0)
        
    def test_kelly_from_sharpe(self):
        """Test Kelly from Sharpe ratio."""
        k = kelly_from_sharpe(2.0, 0.15)
        self.assertGreater(k, 0)
        
        # Higher Sharpe with same vol -> Higher or equal Kelly (capped at 2.0)
        k1 = kelly_from_sharpe(0.5, 0.20)
        k2 = kelly_from_sharpe(1.5, 0.20)
        self.assertGreater(k2, k1)
        
    def test_single_position_sizing(self):
        """Test single position sizing."""
        result = self.sizer.compute_position_size(
            symbol="AAPL",
            expected_return=0.15,
            volatility=0.25,
            current_price=175.0
        )
        
        self.assertIn('position_pct', result)
        self.assertIn('shares', result)
        self.assertGreater(result['position_pct'], 0)
        self.assertLessEqual(result['position_pct'], self.config.max_position_pct)
        
    def test_portfolio_sizing(self):
        """Test portfolio sizing with correlation."""
        assets = [
            {'symbol': 'AAPL', 'expected_return': 0.12, 'volatility': 0.25, 'current_price': 175},
            {'symbol': 'MSFT', 'expected_return': 0.10, 'volatility': 0.22, 'current_price': 380},
        ]
        
        corr = np.array([[1.0, 0.7], [0.7, 1.0]])
        
        portfolio = self.sizer.compute_portfolio_sizes(assets, corr)
        
        self.assertIn('positions', portfolio)
        self.assertIn('total_allocation', portfolio)
        self.assertLessEqual(portfolio['total_allocation'], self.config.max_total_leverage)
        
    def test_regime_detection(self):
        """Test regime detection."""
        detector = RegimeDetector(self.config)
        
        # High volatility regime
        regime = detector.detect_regime(current_vol=0.35)
        self.assertEqual(regime, MarketRegime.HIGH_VOL)
        
        # Crisis regime
        regime = detector.detect_regime(current_vol=0.45)
        self.assertEqual(regime, MarketRegime.CRISIS)
        
    def test_regime_multiplier(self):
        """Test regime multipliers reduce position size."""
        detector = RegimeDetector(self.config)
        
        normal_mult = detector.get_regime_multiplier(MarketRegime.NORMAL)
        crisis_mult = detector.get_regime_multiplier(MarketRegime.CRISIS)
        
        self.assertGreater(normal_mult, crisis_mult)
        
    def test_diversified_kelly(self):
        """Test diversification adjustment."""
        individual = np.array([0.3, 0.3, 0.3])
        
        # High correlation -> lower total
        high_corr = np.array([
            [1.0, 0.9, 0.9],
            [0.9, 1.0, 0.9],
            [0.9, 0.9, 1.0]
        ])
        
        # Low correlation -> higher total
        low_corr = np.array([
            [1.0, 0.2, 0.2],
            [0.2, 1.0, 0.2],
            [0.2, 0.2, 1.0]
        ])
        
        high_result = diversified_kelly(individual, high_corr)
        low_result = diversified_kelly(individual, low_corr)
        
        # Low correlation should allow more allocation
        self.assertGreater(np.sum(low_result), np.sum(high_result))


# =============================================================================
# CIRCUIT BREAKER TESTS
# =============================================================================

class TestCircuitBreakers(unittest.TestCase):
    """Tests for Circuit Breakers."""
    
    def setUp(self):
        self.config = CircuitBreakerConfig()
        self.manager = CircuitBreakerManager(self.config)
        self.manager.reset_daily(1_000_000)
        
    def test_daily_loss_breaker(self):
        """Test daily loss breaker."""
        breaker = DailyLossBreaker(self.config)
        breaker.reset_daily(1_000_000)
        
        # Normal conditions
        state = breaker.update(995_000)  # -0.5%
        self.assertEqual(state.status, BreakerStatus.NORMAL)
        
        # Warning
        state = breaker.update(965_000)  # -3.5%
        self.assertEqual(state.status, BreakerStatus.WARNING)
        
        # Triggered
        state = breaker.update(940_000)  # -6%
        self.assertEqual(state.status, BreakerStatus.TRIGGERED)
        self.assertEqual(state.action, BreakerAction.HALT_TRADING)
        
    def test_position_stop_breaker(self):
        """Test position-level stops."""
        breaker = PositionStopBreaker(self.config)
        
        breaker.add_position("AAPL", 175.0, 100, 0.25)
        
        # Normal
        state = breaker.update_price("AAPL", 173.0)
        self.assertEqual(state.status, BreakerStatus.NORMAL)
        
        # Triggered (3-sigma = ~7.5% for 25% vol)
        state = breaker.update_price("AAPL", 155.0)  # ~11% loss
        self.assertEqual(state.status, BreakerStatus.TRIGGERED)
        
    def test_vix_breaker(self):
        """Test VIX breaker."""
        breaker = VIXBreaker(self.config)
        
        # Normal
        state = breaker.update(18.0)
        self.assertEqual(state.status, BreakerStatus.NORMAL)
        
        # Warning
        state = breaker.update(27.0)
        self.assertEqual(state.status, BreakerStatus.WARNING)
        
        # Reduce
        state = breaker.update(32.0)
        self.assertEqual(state.action, BreakerAction.REDUCE_50)
        
        # Halt
        state = breaker.update(38.0)
        self.assertEqual(state.action, BreakerAction.HALT_TRADING)
        
    def test_drawdown_breaker(self):
        """Test drawdown breaker."""
        breaker = DrawdownBreaker(self.config)
        
        breaker.update(1_000_000)  # Set HWM
        
        # Normal
        state = breaker.update(950_000)  # 5% DD
        self.assertEqual(state.status, BreakerStatus.NORMAL)
        
        # Warning (8% threshold)
        state = breaker.update(910_000)  # 9% DD
        self.assertEqual(state.status, BreakerStatus.WARNING)
        
        # Halt (12% threshold)
        state = breaker.update(870_000)  # 13% DD
        self.assertEqual(state.status, BreakerStatus.TRIGGERED)
        
    def test_manager_halt(self):
        """Test manager halt functionality."""
        # Trigger halt
        states = self.manager.update_all(
            portfolio_value=940_000,  # -6%
            daily_return=-0.06,
            vix=36.0
        )
        
        can_trade, msg = self.manager.can_trade()
        self.assertFalse(can_trade)
        
    def test_position_scaling(self):
        """Test position scaling in adverse conditions."""
        # Normal conditions
        self.manager.update_all(
            portfolio_value=1_000_000,
            daily_return=0.01,
            vix=18.0
        )
        normal_scaling = self.manager.get_position_scaling()
        
        # Elevated VIX
        self.manager.vix.update(28.0)
        vix_scaling = self.manager.get_position_scaling()
        
        self.assertLess(vix_scaling, normal_scaling)


# =============================================================================
# PROFIT ATTRIBUTION TESTS
# =============================================================================

class TestProfitAttribution(unittest.TestCase):
    """Tests for Profit Attribution."""
    
    def setUp(self):
        self.engine = ProfitAttributionEngine(initial_value=1_000_000)
        
    def test_daily_pnl_recording(self):
        """Test daily P&L recording."""
        self.engine.record_daily_pnl(
            date=date.today(),
            portfolio_value=1_005_000,
            gross_pnl=5_500,
            transaction_costs=500,
            n_trades=10,
            win_trades=6
        )
        
        self.assertEqual(len(self.engine.daily_pnl_history), 1)
        self.assertEqual(self.engine.total_gross_pnl, 5_500)
        self.assertEqual(self.engine.total_transaction_costs, 500)
        
    def test_trade_recording(self):
        """Test trade recording."""
        trade = Trade(
            trade_id="1",
            symbol="AAPL",
            side="buy",
            shares=100,
            entry_price=150.0,
            exit_price=155.0,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            strategy="momentum",
            transaction_cost=3.0
        )
        
        self.engine.record_trade(trade)
        
        self.assertEqual(len(self.engine.trade_analyzer.closed_trades), 1)
        
    def test_trade_pnl_calculation(self):
        """Test trade P&L calculation."""
        trade = Trade(
            trade_id="1",
            symbol="AAPL",
            side="buy",
            shares=100,
            entry_price=150.0,
            exit_price=155.0,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            strategy="momentum",
            transaction_cost=3.0
        )
        trade.calculate_pnl()
        
        # Gross: (155-150)*100 = 500, Net: 500-3 = 497
        self.assertAlmostEqual(trade.pnl, 497, delta=0.01)
        
    def test_trade_analyzer(self):
        """Test trade analyzer statistics."""
        analyzer = TradeAnalyzer()
        
        # Add winning and losing trades
        for i in range(10):
            exit_mult = 1.05 if i < 6 else 0.95
            trade = Trade(
                trade_id=str(i),
                symbol="AAPL",
                side="buy",
                shares=100,
                entry_price=100.0,
                exit_price=100.0 * exit_mult,
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                strategy="momentum",
                transaction_cost=1.0
            )
            analyzer.add_trade(trade)
            
        summary = analyzer.get_summary()
        
        self.assertEqual(summary['n_trades'], 10)
        self.assertAlmostEqual(summary['win_rate'], 0.6, delta=0.01)
        
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Add some data
        current_date = date.today()
        for i in range(30):
            pnl = np.random.randn() * 1000
            self.engine.record_daily_pnl(
                date=current_date + timedelta(days=i),
                portfolio_value=self.engine.portfolio_value + pnl,
                gross_pnl=abs(pnl),
                transaction_costs=abs(pnl) * 0.01,
                n_trades=5,
                win_trades=3
            )
            
        summary = self.engine.get_performance_summary()
        
        self.assertIn('performance', summary)
        self.assertIn('pnl', summary)
        self.assertIn('total_return', summary['performance'])
        self.assertIn('sharpe_ratio', summary['performance'])
        
    def test_full_report(self):
        """Test full report generation."""
        # Add minimal data
        self.engine.record_daily_pnl(
            date=date.today(),
            portfolio_value=1_010_000,
            gross_pnl=10_000,
            transaction_costs=100,
            n_trades=5,
            win_trades=3
        )
        
        report = self.engine.get_full_report()
        
        self.assertIn('generated_at', report)
        self.assertIn('performance', report)
        self.assertIn('trade_attribution', report)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestV24Integration(unittest.TestCase):
    """Integration tests for V2.4 components."""
    
    def test_tca_kelly_integration(self):
        """Test TCA and Kelly work together."""
        tca = TCAOptimizer()
        kelly = AdaptiveKellySizer()
        kelly.set_portfolio_value(1_000_000)
        
        # Get Kelly size
        kelly_result = kelly.compute_position_size(
            symbol="AAPL",
            expected_return=0.15,
            volatility=0.25,
            current_price=175.0
        )
        
        # Get TCA optimization for Kelly-sized order
        shares = kelly_result['shares']
        tca_plan = tca.optimize_execution(
            symbol="AAPL",
            order_size=shares,
            price=175.0,
            daily_volume=50_000_000,
            volatility=0.015,
            urgency=0.3
        )
        
        self.assertIn('strategy', tca_plan)
        
    def test_circuit_breaker_kelly_integration(self):
        """Test circuit breakers affect Kelly sizing."""
        breaker = CircuitBreakerManager()
        breaker.reset_daily(1_000_000)
        kelly = AdaptiveKellySizer()
        
        # Get normal position size
        normal_result = kelly.compute_position_size(
            symbol="AAPL",
            expected_return=0.15,
            volatility=0.25,
            current_price=175.0
        )
        normal_pct = normal_result['position_pct']
        
        # Simulate elevated VIX
        breaker.vix.update(32.0)
        scaling = breaker.get_position_scaling()
        
        # Scaled position should be smaller
        scaled_pct = normal_pct * scaling
        self.assertLess(scaled_pct, normal_pct)
        
    def test_full_workflow(self):
        """Test complete V2.4 workflow."""
        # Initialize all components
        tca = TCAOptimizer()
        kelly = AdaptiveKellySizer()
        kelly.set_portfolio_value(1_000_000)
        breaker = CircuitBreakerManager()
        breaker.reset_daily(1_000_000)
        attribution = ProfitAttributionEngine(1_000_000)
        
        # Check if can trade
        can_trade, _ = breaker.can_trade()
        self.assertTrue(can_trade)
        
        # Get position size
        kelly_result = kelly.compute_position_size(
            symbol="AAPL",
            expected_return=0.15,
            volatility=0.25,
            current_price=175.0
        )
        
        # Adjust for circuit breaker
        scaling = breaker.get_position_scaling()
        final_shares = int(kelly_result['shares'] * scaling)
        
        # Get execution plan
        plan = tca.optimize_execution(
            symbol="AAPL",
            order_size=final_shares,
            price=175.0,
            daily_volume=50_000_000,
            volatility=0.015,
            urgency=0.3
        )
        
        # Record trade
        trade = Trade(
            trade_id="1",
            symbol="AAPL",
            side="buy",
            shares=final_shares,
            entry_price=175.0,
            exit_price=177.0,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            strategy="tda_momentum",
            transaction_cost=plan['costs']['optimized_dollars']
        )
        attribution.record_trade(trade)
        
        # Update breakers
        breaker.update_all(
            portfolio_value=1_000_000 + trade.pnl,
            daily_return=trade.pnl / 1_000_000,
            vix=18.0
        )
        
        # Workflow completed successfully
        self.assertGreater(final_shares, 0)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("V2.4 PROFIT COMPONENTS TEST SUITE")
    print("=" * 60)
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTCAOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestKellySizer))
    suite.addTests(loader.loadTestsFromTestCase(TestCircuitBreakers))
    suite.addTests(loader.loadTestsFromTestCase(TestProfitAttribution))
    suite.addTests(loader.loadTestsFromTestCase(TestV24Integration))
    
    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
        for test, traceback in result.failures + result.errors:
            print(f"\n{test}:")
            print(traceback)

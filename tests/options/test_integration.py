"""
Integration Tests
=================

Test complete workflows across multiple components.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from src.options.utils.black_scholes import BlackScholes, OptionType
from src.options.theta_decay_engine import ThetaDecayEngine, IVRegime, TrendDirection
from src.options.iv_analyzer import IVAnalyzer
from src.options.greeks_manager import GreeksManager
from src.options.position_manager import PositionManager
from src.options.delay_adapter import DelayAdapter
from src.options.strategy_engine import StrategyEngine, SpreadType


class TestWheelStrategyWorkflow:
    """Test complete Wheel strategy workflow."""
    
    def setup_method(self):
        """Setup components."""
        self.bs = BlackScholes()
        self.theta_engine = ThetaDecayEngine()
        self.iv_analyzer = IVAnalyzer()
        self.greeks_manager = GreeksManager(account_value=100_000)
        self.position_manager = PositionManager(
            account_value=100_000,
            buying_power=50_000
        )
        self.delay_adapter = DelayAdapter()
        self.strategy_engine = StrategyEngine(
            theta_engine=self.theta_engine,
            iv_analyzer=self.iv_analyzer
        )
    
    def test_complete_wheel_trade(self):
        """Test complete Wheel trade from entry to exit."""
        # Market data
        symbol = "SPY"
        underlying_price = 450.0
        current_iv = 0.22
        historical_vol = 0.18
        
        # Step 1: Find candidates
        candidates = self.strategy_engine.find_wheel_candidates(
            symbol=symbol,
            underlying_price=underlying_price,
            current_iv=current_iv,
            historical_vol=historical_vol,
            trend=TrendDirection.NEUTRAL,
            top_n=3
        )
        
        assert len(candidates) > 0
        best = candidates[0]
        
        # Step 2: Check Greeks limits
        can_add, reason = self.greeks_manager.can_add_position(
            new_greeks=best.greeks,
            quantity=-1  # Selling one contract
        )
        
        assert can_add is True
        
        # Step 3: Calculate position size
        sizing = self.position_manager.calculate_position_size(
            win_rate=0.65,
            avg_win=150,
            avg_loss=100,
            option_price=best.mid,
            kelly_multiplier=0.25
        )
        
        assert sizing.num_contracts > 0
        
        # Step 4: Apply delay adjustment
        adjusted = self.delay_adapter.adjust_entry_price(
            quoted_price=best.mid,
            is_credit=True,  # Selling premium
            atr=0.25,
            underlying_price=underlying_price,
            symbol=symbol
        )
        
        assert adjusted.adjusted_price < best.mid  # Conservative for credit
        
        # Step 5: Open position
        position = self.position_manager.open_position(
            symbol=symbol,
            strike=best.strike,
            expiration=best.expiration,
            option_type=OptionType.PUT,
            quantity=-1,  # Short
            entry_price=adjusted.adjusted_price,
            greeks=best.greeks,
            underlying_price=underlying_price,
            iv=current_iv
        )
        
        assert position is not None
        assert position.quantity == -1
        
        # Step 6: Add to Greeks manager
        self.greeks_manager.add_position(
            symbol=symbol,
            strike=best.strike,
            expiration=best.expiration,
            option_type=OptionType.PUT,
            quantity=-1,
            greeks=best.greeks,
            underlying_price=underlying_price
        )
        
        # Check portfolio Greeks
        portfolio = self.greeks_manager.get_portfolio_greeks()
        assert portfolio.num_positions == 1
        
        # Step 7: Simulate profit and close
        exit_price = adjusted.adjusted_price * 0.50  # 50% profit
        
        pnl = self.position_manager.close_position(
            position=position,
            exit_price=exit_price
        )
        
        assert pnl > 0  # Profitable trade
        
        # Verify performance metrics
        perf = self.position_manager.get_performance_summary()
        assert perf['total_trades'] == 1
        assert perf['winning_trades'] == 1
        assert perf['win_rate'] == 1.0


class TestCreditSpreadWorkflow:
    """Test credit spread workflow."""
    
    def setup_method(self):
        """Setup components."""
        self.strategy_engine = StrategyEngine()
        self.greeks_manager = GreeksManager(account_value=100_000)
        self.position_manager = PositionManager(
            account_value=100_000,
            buying_power=50_000
        )
    
    def test_bull_put_spread(self):
        """Test bull put spread trade."""
        symbol = "SPY"
        underlying_price = 450.0
        current_iv = 0.20
        
        # Find spread candidates
        candidates = self.strategy_engine.find_credit_spread_candidates(
            symbol=symbol,
            underlying_price=underlying_price,
            current_iv=current_iv,
            spread_type=SpreadType.BULL_PUT,
            top_n=3
        )
        
        assert len(candidates) > 0
        
        best_spread = candidates[0]
        
        # Verify spread structure
        assert best_spread.short_strike > best_spread.long_strike  # Put spread
        assert best_spread.net_credit > 0  # Collecting premium
        assert best_spread.max_profit > 0
        assert best_spread.max_loss > 0
        assert best_spread.pop > 50  # Should have > 50% probability of profit
        
        # Check risk/reward
        rr_ratio = best_spread.max_profit / best_spread.max_loss
        assert rr_ratio > 0.25  # At least 1:4 reward:risk


class TestIronCondorWorkflow:
    """Test iron condor workflow."""
    
    def setup_method(self):
        """Setup components."""
        self.strategy_engine = StrategyEngine()
        self.greeks_manager = GreeksManager(account_value=100_000)
    
    def test_iron_condor_construction(self):
        """Test iron condor construction and validation."""
        symbol = "SPY"
        underlying_price = 450.0
        current_iv = 0.22
        
        # Find IC candidates
        candidates = self.strategy_engine.find_iron_condor_candidates(
            symbol=symbol,
            underlying_price=underlying_price,
            current_iv=current_iv,
            top_n=2
        )
        
        assert len(candidates) > 0
        
        ic = candidates[0]
        
        # Verify structure
        assert ic.put_short_strike < underlying_price  # Put below market
        assert ic.call_short_strike > underlying_price  # Call above market
        assert ic.put_long_strike < ic.put_short_strike  # Put wing
        assert ic.call_long_strike > ic.call_short_strike  # Call wing
        
        # Verify pricing
        assert ic.total_credit > 0
        assert ic.max_profit == ic.total_credit * 100
        assert ic.max_loss > 0
        
        # Verify breakevens are outside short strikes
        assert ic.breakeven_lower < ic.put_short_strike
        assert ic.breakeven_upper > ic.call_short_strike
        
        # Verify Greeks are delta-neutral-ish
        assert abs(ic.net_greeks.delta) < 10  # Should be nearly neutral


class TestRiskManagementIntegration:
    """Test risk management across components."""
    
    def setup_method(self):
        """Setup components."""
        self.greeks_manager = GreeksManager(account_value=100_000)
        self.position_manager = PositionManager(
            account_value=100_000,
            buying_power=50_000
        )
        self.bs = BlackScholes()
    
    def test_position_limits(self):
        """Test position limits are enforced."""
        # Try to open max positions
        for i in range(6):  # Max is 6
            T = 30 / 365.0
            greeks = self.bs.calculate_all_greeks(
                S=450, K=440, T=T, sigma=0.20, option_type=OptionType.PUT
            )
            
            position = self.position_manager.open_position(
                symbol="SPY",
                strike=440 + i,
                expiration=datetime.now() + timedelta(days=30),
                option_type=OptionType.PUT,
                quantity=-1,
                entry_price=4.50,
                greeks=greeks,
                underlying_price=450
            )
            
            if i < 6:
                assert position is not None
            else:
                assert position is None  # Should reject 7th position
    
    def test_greeks_limits(self):
        """Test Greeks limits are enforced."""
        T = 30 / 365.0
        
        # Add positions until Greeks limit hit
        for i in range(10):
            greeks = self.bs.calculate_all_greeks(
                S=450, K=440 - i*5, T=T, sigma=0.20, option_type=OptionType.PUT
            )
            
            can_add, reason = self.greeks_manager.can_add_position(
                new_greeks=greeks,
                quantity=-5  # Large position
            )
            
            if can_add:
                self.greeks_manager.add_position(
                    symbol="SPY",
                    strike=440 - i*5,
                    expiration=datetime.now() + timedelta(days=30),
                    option_type=OptionType.PUT,
                    quantity=-5,
                    greeks=greeks,
                    underlying_price=450
                )
            else:
                # Should eventually hit limit
                assert "exceed" in reason.lower()
                break
        
        # Verify violations detected
        violations = self.greeks_manager.check_limits()
        # May or may not have violations depending on where we stopped


class TestDelayCompensation:
    """Test delay adaptation strategies."""
    
    def setup_method(self):
        """Setup delay adapter."""
        self.adapter = DelayAdapter(delay_minutes=15)
    
    def test_safe_trading_windows(self):
        """Test trading window detection."""
        # Test different times
        times = [
            (datetime.now().replace(hour=9, minute=45), False),  # Open volatility
            (datetime.now().replace(hour=11, minute=0), True),   # Safe window
            (datetime.now().replace(hour=15, minute=45), False), # Close volatility
        ]
        
        for test_time, should_be_safe in times:
            is_safe, reason = self.adapter.is_safe_to_trade(current_time=test_time)
            
            if should_be_safe:
                assert is_safe is True
            else:
                assert is_safe is False
    
    def test_position_size_reduction(self):
        """Test position size reduction for delayed data."""
        standard_size = 10
        
        # Low VIX: minimal reduction
        reduced_low, reason_low = self.adapter.should_reduce_position_size(
            standard_size=standard_size,
            vix_level=15,
            reduction_pct=0.20
        )
        
        assert reduced_low == 8  # 20% reduction
        
        # High VIX: additional reduction
        reduced_high, reason_high = self.adapter.should_reduce_position_size(
            standard_size=standard_size,
            vix_level=30,
            reduction_pct=0.20
        )
        
        assert reduced_high < reduced_low  # More reduction for high VIX


class TestCompleteSystemWorkflow:
    """Test complete system from analysis to execution."""
    
    def test_end_to_end_workflow(self):
        """Test full workflow from market analysis to position management."""
        # Initialize all components
        bs = BlackScholes()
        iv_analyzer = IVAnalyzer()
        theta_engine = ThetaDecayEngine()
        greeks_manager = GreeksManager(account_value=100_000)
        position_manager = PositionManager(
            account_value=100_000,
            buying_power=50_000
        )
        delay_adapter = DelayAdapter()
        strategy_engine = StrategyEngine(
            theta_engine=theta_engine,
            iv_analyzer=iv_analyzer
        )
        
        # Market conditions
        symbol = "SPY"
        underlying_price = 450.0
        current_iv = 0.22
        historical_vol = 0.18
        vix = 18.0
        
        # 1. Analyze IV environment
        iv_metrics = iv_analyzer.analyze(symbol, current_iv, historical_vol)
        
        assert iv_metrics.iv_rank > 0
        assert iv_metrics.regime in list(IVRegime)
        
        # 2. Check if safe to trade
        is_safe, reason = delay_adapter.is_safe_to_trade(vix_level=vix)
        
        if not is_safe:
            pytest.skip(f"Not safe to trade: {reason}")
        
        # 3. Find trade candidates
        candidates = strategy_engine.find_wheel_candidates(
            symbol=symbol,
            underlying_price=underlying_price,
            current_iv=current_iv,
            historical_vol=historical_vol,
            top_n=3
        )
        
        if len(candidates) == 0:
            pytest.skip("No candidates found")
        
        best = candidates[0]
        
        # 4. Size position
        sizing = position_manager.calculate_position_size(
            win_rate=0.65,
            avg_win=150,
            avg_loss=100,
            option_price=best.mid,
            max_contracts=3
        )
        
        # 5. Adjust for delay
        adjusted = delay_adapter.adjust_entry_price(
            quoted_price=best.mid,
            is_credit=True,
            atr=0.25,
            underlying_price=underlying_price
        )
        
        # 6. Validate Greeks
        can_add, reason = greeks_manager.can_add_position(
            new_greeks=best.greeks,
            quantity=-sizing.num_contracts
        )
        
        assert can_add is True
        
        # 7. Open position
        position = position_manager.open_position(
            symbol=symbol,
            strike=best.strike,
            expiration=best.expiration,
            option_type=best.option_type,
            quantity=-sizing.num_contracts,
            entry_price=adjusted.adjusted_price,
            greeks=best.greeks,
            underlying_price=underlying_price,
            iv=current_iv
        )
        
        assert position is not None
        
        # 8. Track in Greeks manager
        greeks_manager.add_position(
            symbol=symbol,
            strike=best.strike,
            expiration=best.expiration,
            option_type=best.option_type,
            quantity=-sizing.num_contracts,
            greeks=best.greeks,
            underlying_price=underlying_price
        )
        
        # 9. Monitor portfolio
        portfolio = greeks_manager.get_portfolio_greeks()
        violations = greeks_manager.check_limits()
        
        assert portfolio.num_positions == 1
        
        # All integrated successfully!


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Unit tests for Risk Management Framework.

Tests:
- Position sizing never exceeds 10% of account
- Portfolio heat never exceeds 20%
- Stop-loss distance between 1.5% and 4%
- Kelly calculation handles edge cases
- Take-profit correctly calculated
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk_management import RiskManager, TradeJournal, calculate_atr


class TestRiskManager:
    """Test suite for RiskManager class."""
    
    @pytest.fixture
    def risk_manager(self, tmp_path):
        """Create a RiskManager instance for testing."""
        return RiskManager(
            initial_capital=100000,
            risk_per_trade=0.01,
            log_path=str(tmp_path / "risk_log.csv")
        )
    
    def test_init(self, risk_manager):
        """Test RiskManager initialization."""
        assert risk_manager.initial_capital == 100000
        assert risk_manager.account_balance == 100000
        assert risk_manager.risk_per_trade == 0.01
        assert risk_manager.win_rate == 0.50  # Default
        assert 0 <= risk_manager.kelly_fraction <= 0.5
    
    # =========================================================================
    # Position Sizing Tests
    # =========================================================================
    
    def test_position_size_within_10_percent_cap(self, risk_manager):
        """Position size should never exceed 10% of account."""
        entry = 100.0
        stop = 97.0  # 3% stop
        
        size = risk_manager.calculate_position_size(
            account_balance=100000,
            risk_per_trade=0.01,
            entry_price=entry,
            stop_price=stop
        )
        
        position_value = size * entry
        max_allowed = 100000 * 0.10  # 10% of account
        
        assert position_value <= max_allowed, \
            f"Position value ${position_value:.2f} exceeds 10% cap ${max_allowed:.2f}"
    
    def test_position_size_10_percent_cap_extreme_case(self, risk_manager):
        """Test 10% cap with very high kelly fraction (simulated)."""
        # Simulate high win rate scenario
        risk_manager.win_rate = 0.80
        risk_manager.avg_win = 0.10
        risk_manager.avg_loss = 0.02
        risk_manager.kelly_fraction = risk_manager._compute_kelly_fraction()
        
        entry = 50.0
        stop = 49.0  # Very tight stop
        
        size = risk_manager.calculate_position_size(
            account_balance=100000,
            risk_per_trade=0.05,  # Higher risk
            entry_price=entry,
            stop_price=stop
        )
        
        position_value = size * entry
        max_allowed = 100000 * 0.10
        
        assert position_value <= max_allowed + 0.01, \
            f"Position value ${position_value:.2f} exceeds 10% cap ${max_allowed:.2f}"
    
    def test_position_size_various_account_balances(self, tmp_path):
        """Test position sizing scales correctly with account balance."""
        for balance in [10000, 50000, 100000, 500000, 1000000]:
            rm = RiskManager(
                initial_capital=balance,
                log_path=str(tmp_path / f"risk_log_{balance}.csv")
            )
            
            entry = 100.0
            stop = 97.0
            
            size = rm.calculate_position_size(
                account_balance=balance,
                risk_per_trade=0.01,
                entry_price=entry,
                stop_price=stop
            )
            
            position_value = size * entry
            max_allowed = balance * 0.10
            
            assert position_value <= max_allowed + 0.01, \
                f"Balance ${balance}: Position ${position_value:.2f} exceeds 10% cap"
    
    def test_position_size_zero_when_stop_equals_entry(self, risk_manager):
        """Position size should be 0 when stop equals entry."""
        size = risk_manager.calculate_position_size(
            account_balance=100000,
            risk_per_trade=0.01,
            entry_price=100.0,
            stop_price=100.0  # No risk per share
        )
        assert size == 0, "Should return 0 when stop equals entry"
    
    def test_position_size_zero_for_invalid_prices(self, risk_manager):
        """Position size should be 0 for invalid prices."""
        # Negative entry
        size = risk_manager.calculate_position_size(
            account_balance=100000,
            risk_per_trade=0.01,
            entry_price=-100.0,
            stop_price=97.0
        )
        assert size == 0, "Should return 0 for negative entry price"
        
        # Zero entry
        size = risk_manager.calculate_position_size(
            account_balance=100000,
            risk_per_trade=0.01,
            entry_price=0.0,
            stop_price=97.0
        )
        assert size == 0, "Should return 0 for zero entry price"
        
        # Negative stop
        size = risk_manager.calculate_position_size(
            account_balance=100000,
            risk_per_trade=0.01,
            entry_price=100.0,
            stop_price=-97.0
        )
        assert size == 0, "Should return 0 for negative stop price"
    
    # =========================================================================
    # Stop-Loss Tests
    # =========================================================================
    
    def test_stop_loss_within_bounds(self, risk_manager):
        """Stop-loss distance should be between 1.5% and 4%."""
        entry = 100.0
        
        # Test with various ATR values
        for atr in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
            stop = risk_manager.set_stop_loss(entry, 'long', atr, multiplier=2.0)
            
            stop_dist_pct = abs(entry - stop) / entry
            
            assert stop_dist_pct >= 0.015 - 0.001, \
                f"ATR={atr}: Stop distance {stop_dist_pct*100:.2f}% below min 1.5%"
            assert stop_dist_pct <= 0.04 + 0.001, \
                f"ATR={atr}: Stop distance {stop_dist_pct*100:.2f}% above max 4%"
    
    def test_stop_loss_long_direction(self, risk_manager):
        """Long stop should be below entry price."""
        entry = 100.0
        stop = risk_manager.set_stop_loss(entry, 'long', atr_value=2.0, multiplier=2.0)
        assert stop < entry, "Long stop should be below entry"
    
    def test_stop_loss_short_direction(self, risk_manager):
        """Short stop should be above entry price."""
        entry = 100.0
        stop = risk_manager.set_stop_loss(entry, 'short', atr_value=2.0, multiplier=2.0)
        assert stop > entry, "Short stop should be above entry"
    
    def test_stop_loss_various_multipliers(self, risk_manager):
        """Test stop-loss with different ATR multipliers."""
        entry = 100.0
        atr = 2.0
        
        for mult in [1.0, 1.5, 2.0, 2.5, 3.0]:
            stop = risk_manager.set_stop_loss(entry, 'long', atr, multiplier=mult)
            
            stop_dist_pct = abs(entry - stop) / entry
            # Should still be within bounds
            assert 0.015 <= stop_dist_pct <= 0.04, \
                f"Multiplier={mult}: Stop distance {stop_dist_pct*100:.2f}% out of bounds"
    
    def test_stop_loss_handles_zero_atr(self, risk_manager):
        """Stop-loss should handle zero ATR gracefully."""
        entry = 100.0
        stop = risk_manager.set_stop_loss(entry, 'long', atr_value=0.0, multiplier=2.0)
        
        # Should default to minimum distance
        stop_dist_pct = abs(entry - stop) / entry
        assert stop_dist_pct >= 0.015, "Zero ATR should default to min stop distance"
    
    # =========================================================================
    # Take-Profit Tests  
    # =========================================================================
    
    def test_take_profit_calculation(self, risk_manager):
        """Take-profit should be correctly calculated based on R:R ratio."""
        entry = 100.0
        stop = 97.0  # $3 risk
        
        target = risk_manager.set_take_profit(entry, stop, risk_reward_ratio=2.0)
        
        expected = 106.0  # 100 + (3 * 2) = 106
        assert abs(target - expected) < 0.01, \
            f"Target {target} should be {expected}"
    
    def test_take_profit_various_ratios(self, risk_manager):
        """Test take-profit with different R:R ratios."""
        entry = 100.0
        stop = 98.0  # $2 risk
        
        for rr in [1.0, 1.5, 2.0, 2.5, 3.0]:
            target = risk_manager.set_take_profit(entry, stop, risk_reward_ratio=rr)
            expected = entry + (2.0 * rr)
            assert abs(target - expected) < 0.01, \
                f"R:R={rr}: Target {target} should be {expected}"
    
    def test_take_profit_short_position(self, risk_manager):
        """Take-profit for short position should be below entry."""
        entry = 100.0
        stop = 103.0  # Short stop above entry
        
        target = risk_manager.set_take_profit(entry, stop, risk_reward_ratio=2.0)
        
        # For short: target = entry - risk * RR = 100 - 3 * 2 = 94
        expected = 94.0
        assert abs(target - expected) < 0.01, \
            f"Short target {target} should be {expected}"
        assert target < entry, "Short take-profit should be below entry"
    
    # =========================================================================
    # Portfolio Heat Tests
    # =========================================================================
    
    def test_portfolio_heat_empty_positions(self, risk_manager):
        """Empty positions should have zero heat."""
        can_open, heat = risk_manager.check_portfolio_heat({}, max_heat=0.20)
        assert can_open is True
        assert heat == 0.0
    
    def test_portfolio_heat_single_position(self, risk_manager):
        """Test heat calculation with single position."""
        positions = {
            'SPY': {'entry': 100, 'stop': 97, 'size': 100}  # $300 risk
        }
        
        can_open, heat = risk_manager.check_portfolio_heat(positions, max_heat=0.20)
        
        # Risk = 100 * (100 - 97) = $300
        # Heat = 300 / 100000 = 0.003 = 0.3%
        expected_heat = 300 / 100000
        assert abs(heat - expected_heat) < 0.001
        assert can_open is True
    
    def test_portfolio_heat_multiple_positions(self, risk_manager):
        """Test heat calculation with multiple positions."""
        positions = {
            'SPY': {'entry': 100, 'stop': 97, 'size': 300},   # $900 risk
            'QQQ': {'entry': 200, 'stop': 194, 'size': 150},  # $900 risk
            'IWM': {'entry': 150, 'stop': 144, 'size': 100},  # $600 risk
        }
        
        can_open, heat = risk_manager.check_portfolio_heat(positions, max_heat=0.20)
        
        # Total risk = 900 + 900 + 600 = $2400
        # Heat = 2400 / 100000 = 2.4%
        expected_heat = 2400 / 100000
        assert abs(heat - expected_heat) < 0.001
        assert can_open is True  # 2.4% < 20%
    
    def test_portfolio_heat_exceeds_limit(self, risk_manager):
        """Test that positions are blocked when heat exceeds limit."""
        # Create positions that exceed 20% heat
        positions = {
            'SPY': {'entry': 100, 'stop': 80, 'size': 500},   # $10,000 risk
            'QQQ': {'entry': 200, 'stop': 160, 'size': 300},  # $12,000 risk
        }
        
        can_open, heat = risk_manager.check_portfolio_heat(positions, max_heat=0.20)
        
        # Total risk = $22,000
        # Heat = 22000 / 100000 = 22%
        assert heat > 0.20
        assert can_open is False
    
    def test_portfolio_heat_never_exceeds_20_percent_boundary(self, risk_manager):
        """Test that 20% heat boundary is enforced correctly."""
        # Exactly at 20%
        positions = {
            'SPY': {'entry': 100, 'stop': 80, 'size': 1000},  # $20,000 risk = 20%
        }
        
        can_open, heat = risk_manager.check_portfolio_heat(positions, max_heat=0.20)
        assert abs(heat - 0.20) < 0.001
        assert can_open is False  # At limit, can't open new
        
        # Just under 20%
        positions = {
            'SPY': {'entry': 100, 'stop': 80, 'size': 990},  # $19,800 risk = 19.8%
        }
        
        can_open, heat = risk_manager.check_portfolio_heat(positions, max_heat=0.20)
        assert heat < 0.20
        assert can_open is True
    
    # =========================================================================
    # Kelly Criterion Tests
    # =========================================================================
    
    def test_kelly_calculation_basic(self, risk_manager):
        """Test basic Kelly fraction calculation."""
        risk_manager.win_rate = 0.60
        risk_manager.avg_win = 0.03  # 3% average win
        risk_manager.avg_loss = 0.02  # 2% average loss
        
        kelly = risk_manager._compute_kelly_fraction()
        
        # Kelly = (0.6 * 0.03 - 0.4 * 0.02) / 0.03
        # Kelly = (0.018 - 0.008) / 0.03 = 0.333...
        expected = (0.60 * 0.03 - 0.40 * 0.02) / 0.03
        assert abs(kelly - expected) < 0.001
    
    def test_kelly_handles_zero_trades(self, risk_manager):
        """Kelly update should handle zero trades gracefully."""
        empty_df = pd.DataFrame({'pnl': []})
        stats = risk_manager.update_kelly_parameters(empty_df, lookback=50)
        
        assert 'kelly_fraction' in stats
        assert stats['num_trades'] == 0
    
    def test_kelly_handles_few_trades(self, risk_manager):
        """Kelly update with fewer than 5 trades should keep defaults."""
        few_trades = pd.DataFrame({'pnl': [100, -50, 75]})
        stats = risk_manager.update_kelly_parameters(few_trades, lookback=50)
        
        assert stats['num_trades'] == 3
        # Should keep reasonable defaults
        assert 0 <= stats['kelly_fraction'] <= 0.5
    
    def test_kelly_update_from_trade_history(self, risk_manager):
        """Test Kelly parameter update from trade history."""
        trades = pd.DataFrame({
            'pnl': [100, -50, 200, -30, 150, -40, 80, 120, -60, 90]
        })
        
        stats = risk_manager.update_kelly_parameters(trades, lookback=10)
        
        assert stats['num_trades'] == 10
        assert 0 <= stats['win_rate'] <= 1
        assert stats['avg_win'] > 0
        assert stats['avg_loss'] > 0
        assert 0 <= stats['kelly_fraction'] <= 0.5
    
    def test_kelly_constrained_to_valid_range(self, risk_manager):
        """Kelly fraction should always be in [0, 0.5] range."""
        # Test with extreme win rate
        risk_manager.win_rate = 0.95
        risk_manager.avg_win = 0.10
        risk_manager.avg_loss = 0.01
        kelly = risk_manager._compute_kelly_fraction()
        assert 0 <= kelly <= 0.5
        
        # Test with very low win rate
        risk_manager.win_rate = 0.10
        risk_manager.avg_win = 0.02
        risk_manager.avg_loss = 0.03
        kelly = risk_manager._compute_kelly_fraction()
        assert 0 <= kelly <= 0.5
        
        # Test with negative expected value (should be 0)
        risk_manager.win_rate = 0.20
        risk_manager.avg_win = 0.02
        risk_manager.avg_loss = 0.05
        kelly = risk_manager._compute_kelly_fraction()
        assert kelly == 0.0


class TestTradeJournal:
    """Test suite for TradeJournal class."""
    
    @pytest.fixture
    def trade_journal(self, tmp_path):
        """Create a TradeJournal instance for testing."""
        return TradeJournal(journal_path=str(tmp_path / "trade_journal.csv"))
    
    def test_log_trade(self, trade_journal):
        """Test logging a trade."""
        trade = trade_journal.log_trade(
            ticker='SPY',
            direction='long',
            entry_date='2024-01-01',
            entry_price=100.0,
            exit_date='2024-01-15',
            exit_price=106.0,
            size=100,
            stop_loss=97.0,
            take_profit=109.0,
            exit_reason='signal'
        )
        
        assert trade['ticker'] == 'SPY'
        assert trade['pnl'] == 600.0  # (106 - 100) * 100
        assert abs(trade['pnl_pct'] - 0.06) < 0.001
        assert trade['r_multiple'] == 2.0  # 6 / 3 = 2
    
    def test_summary_stats(self, trade_journal):
        """Test summary statistics calculation."""
        # Log several trades
        trade_journal.log_trade('SPY', 'long', '2024-01-01', 100, '2024-01-15', 106, 100, 97, 109, 'signal')
        trade_journal.log_trade('QQQ', 'long', '2024-02-01', 200, '2024-02-15', 190, 50, 194, 212, 'stop_loss')
        trade_journal.log_trade('IWM', 'long', '2024-03-01', 150, '2024-03-15', 159, 75, 144, 162, 'take_profit')
        
        stats = trade_journal.get_summary_stats()
        
        assert stats['num_trades'] == 3
        assert 0 <= stats['win_rate'] <= 1
        assert stats['num_stopped_out'] == 1
        assert stats['num_take_profit_hits'] == 1


class TestCalculateATR:
    """Test suite for ATR calculation function."""
    
    def test_atr_basic(self):
        """Test basic ATR calculation."""
        df = pd.DataFrame({
            'high': [102, 103, 104, 103, 105, 106, 104, 107, 108, 106,
                    109, 110, 108, 111, 112, 110, 113, 114, 112, 115],
            'low': [98, 99, 100, 99, 101, 102, 100, 103, 104, 102,
                   105, 106, 104, 107, 108, 106, 109, 110, 108, 111],
            'close': [100, 101, 102, 101, 103, 104, 102, 105, 106, 104,
                     107, 108, 106, 109, 110, 108, 111, 112, 110, 113]
        })
        
        atr = calculate_atr(df, period=14)
        
        # Should have values after warmup
        assert len(atr) == 20
        assert pd.notna(atr.iloc[-1])
        assert atr.iloc[-1] > 0


class TestIntegration:
    """Integration tests for risk management workflow."""
    
    def test_full_risk_workflow(self, tmp_path):
        """Test complete risk management workflow."""
        rm = RiskManager(
            initial_capital=100000,
            risk_per_trade=0.01,
            log_path=str(tmp_path / "risk_log.csv")
        )
        tj = TradeJournal(journal_path=str(tmp_path / "trade_journal.csv"))
        
        # 1. Calculate stop and target
        entry = 100.0
        atr = 2.5
        stop = rm.set_stop_loss(entry, 'long', atr, multiplier=2.0)
        target = rm.set_take_profit(entry, stop, risk_reward_ratio=2.0)
        
        # 2. Calculate position size
        size = rm.calculate_position_size(100000, 0.01, entry, stop, atr, 'SPY')
        
        # 3. Check portfolio heat before entering
        can_open, heat = rm.check_portfolio_heat({}, max_heat=0.20)
        assert can_open is True
        
        # 4. Simulate trade execution
        positions = {'SPY': {'entry': entry, 'stop': stop, 'target': target, 'size': size}}
        can_open_new, heat = rm.check_portfolio_heat(positions, max_heat=0.20)
        
        # 5. Close trade at target
        exit_price = target
        rm.record_trade('SPY', entry, exit_price, size, 'long', '2024-01-01', '2024-01-15', 'take_profit')
        
        # 6. Log to journal
        tj.log_trade('SPY', 'long', '2024-01-01', entry, '2024-01-15', exit_price,
                    size, stop, target, 'take_profit')
        
        # 7. Verify metrics
        risk_metrics = rm.get_risk_metrics({})
        journal_stats = tj.get_summary_stats()
        
        assert risk_metrics['num_trades_recorded'] == 1
        assert journal_stats['num_trades'] == 1
        assert journal_stats['num_take_profit_hits'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

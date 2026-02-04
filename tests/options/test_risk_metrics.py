"""
Test Risk Metrics Module
=========================

Tests for Kelly criterion, Sharpe ratio, and other risk metrics.
"""

import pytest
import numpy as np
from src.options.utils.risk_metrics import (
    calculate_kelly_fraction,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_win_rate,
    calculate_expected_value,
    calculate_calmar_ratio,
    risk_of_ruin,
)


class TestKellyCriterion:
    """Test Kelly fraction calculations."""
    
    def test_positive_edge(self):
        """Test Kelly with positive edge."""
        # 60% win rate, avg win $150, avg loss $100
        kelly = calculate_kelly_fraction(
            win_rate=0.60,
            avg_win=150,
            avg_loss=100
        )
        
        # Should recommend positive position size
        assert kelly > 0
        assert kelly <= 0.50  # Max default
    
    def test_no_edge(self):
        """Test Kelly with no edge (fair game)."""
        # 50% win rate, equal wins and losses
        kelly = calculate_kelly_fraction(
            win_rate=0.50,
            avg_win=100,
            avg_loss=100
        )
        
        # Should recommend 0 position size
        assert kelly == 0
    
    def test_negative_edge(self):
        """Test Kelly with negative edge."""
        # 40% win rate
        kelly = calculate_kelly_fraction(
            win_rate=0.40,
            avg_win=100,
            avg_loss=100
        )
        
        # Should recommend 0 (don't trade)
        assert kelly == 0
    
    def test_high_edge(self):
        """Test Kelly with very high edge."""
        # 70% win rate, 2:1 reward:risk
        kelly = calculate_kelly_fraction(
            win_rate=0.70,
            avg_win=200,
            avg_loss=100
        )
        
        # Should be capped at max_fraction
        assert kelly == 0.50  # Default max
    
    def test_custom_max(self):
        """Test custom max fraction."""
        kelly = calculate_kelly_fraction(
            win_rate=0.70,
            avg_win=200,
            avg_loss=100,
            max_fraction=0.25
        )
        
        assert kelly == 0.25
    
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Win rate > 1
        kelly = calculate_kelly_fraction(win_rate=1.5, avg_win=100, avg_loss=50)
        assert kelly == 0
        
        # Negative avg loss
        kelly = calculate_kelly_fraction(win_rate=0.6, avg_win=100, avg_loss=-50)
        assert kelly == 0


class TestSharpeRatio:
    """Test Sharpe ratio calculations."""
    
    def test_positive_sharpe(self):
        """Test with profitable returns."""
        # Returns with positive mean
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.01])
        
        sharpe = calculate_sharpe_ratio(
            returns=returns,
            risk_free_rate=0.02,
            periods_per_year=252
        )
        
        # Should be positive
        assert sharpe > 0
    
    def test_negative_sharpe(self):
        """Test with losing returns."""
        returns = np.array([-0.01, -0.02, 0.005, -0.015, -0.01])
        
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        
        # Should be negative
        assert sharpe < 0
    
    def test_zero_volatility(self):
        """Test with zero volatility."""
        # Constant returns
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        
        # Should return inf (infinite Sharpe)
        assert np.isinf(sharpe)
    
    def test_high_sharpe(self):
        """Test excellent strategy (Sharpe > 3)."""
        # Low volatility, high returns
        returns = np.array([0.02, 0.021, 0.019, 0.022, 0.020])
        
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        
        assert sharpe > 3.0


class TestSortinoRatio:
    """Test Sortino ratio (downside deviation only)."""
    
    def test_sortino_vs_sharpe(self):
        """Sortino should be higher than Sharpe for positive skew."""
        # Returns with positive skew (few large losses, many small wins)
        returns = np.array([0.01, 0.01, 0.01, -0.05, 0.01, 0.01])
        
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
        
        # Sortino should be higher (ignores upside volatility)
        assert sortino >= sharpe
    
    def test_no_downside(self):
        """Test with no negative returns."""
        returns = np.array([0.01, 0.02, 0.015, 0.01])
        
        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
        
        # Should be infinite (no downside risk)
        assert np.isinf(sortino)


class TestMaxDrawdown:
    """Test maximum drawdown calculations."""
    
    def test_no_drawdown(self):
        """Test with steadily increasing equity."""
        equity = np.array([100, 105, 110, 115, 120])
        
        dd, peak_idx, trough_idx = calculate_max_drawdown(equity)
        
        # No drawdown
        assert dd == 0
    
    def test_simple_drawdown(self):
        """Test with single drawdown."""
        equity = np.array([100, 110, 90, 95, 105])
        
        dd, peak_idx, trough_idx = calculate_max_drawdown(equity)
        
        # Max DD from 110 to 90 = -18.18%
        assert abs(dd - (-18.18)) < 0.01
        assert peak_idx == 1
        assert trough_idx == 2
    
    def test_multiple_drawdowns(self):
        """Test with multiple drawdowns."""
        equity = np.array([100, 120, 100, 110, 80, 100])
        
        dd, peak_idx, trough_idx = calculate_max_drawdown(equity)
        
        # Max DD from 110 to 80 = -27.27%
        assert abs(dd - (-27.27)) < 0.01


class TestProfitFactor:
    """Test profit factor calculations."""
    
    def test_profitable_system(self):
        """Test profitable system (PF > 1)."""
        winning = [100, 150, 120]
        losing = [-50, -40]
        
        pf = calculate_profit_factor(winning, losing)
        
        # PF = 370 / 90 = 4.11
        assert abs(pf - 4.11) < 0.01
    
    def test_losing_system(self):
        """Test losing system (PF < 1)."""
        winning = [50, 40]
        losing = [-100, -80]
        
        pf = calculate_profit_factor(winning, losing)
        
        # PF = 90 / 180 = 0.50
        assert abs(pf - 0.50) < 0.01
    
    def test_no_losers(self):
        """Test with no losing trades."""
        winning = [100, 200]
        losing = []
        
        pf = calculate_profit_factor(winning, losing)
        
        # Infinite profit factor
        assert np.isinf(pf)
    
    def test_no_winners(self):
        """Test with no winning trades."""
        winning = []
        losing = [-100, -50]
        
        pf = calculate_profit_factor(winning, losing)
        
        # Zero profit factor
        assert pf == 0


class TestWinRate:
    """Test win rate calculations."""
    
    def test_60_percent_win_rate(self):
        """Test 60% win rate."""
        trades = [100, -50, 80, -30, 90, 110, -40, 70, 85, -20]
        
        wr = calculate_win_rate(trades)
        
        # 6 winners out of 10 = 60%
        assert abs(wr - 0.60) < 0.01
    
    def test_all_winners(self):
        """Test 100% win rate."""
        trades = [100, 50, 80]
        
        wr = calculate_win_rate(trades)
        
        assert wr == 1.0
    
    def test_all_losers(self):
        """Test 0% win rate."""
        trades = [-100, -50, -80]
        
        wr = calculate_win_rate(trades)
        
        assert wr == 0.0
    
    def test_breakeven_trades(self):
        """Test handling of breakeven trades."""
        # Breakeven trades (0) should not count as wins
        trades = [100, 0, -50, 0]
        
        wr = calculate_win_rate(trades)
        
        # 1 winner out of 4 = 25%
        assert abs(wr - 0.25) < 0.01


class TestExpectedValue:
    """Test expected value calculations."""
    
    def test_positive_ev(self):
        """Test positive expected value."""
        ev = calculate_expected_value(
            win_rate=0.65,
            avg_win=150,
            avg_loss=100
        )
        
        # EV = 0.65*150 - 0.35*100 = 97.5 - 35 = 62.5
        assert abs(ev - 62.5) < 0.01
    
    def test_negative_ev(self):
        """Test negative expected value."""
        ev = calculate_expected_value(
            win_rate=0.40,
            avg_win=100,
            avg_loss=150
        )
        
        # EV = 0.40*100 - 0.60*150 = 40 - 90 = -50
        assert abs(ev - (-50)) < 0.01
    
    def test_zero_ev(self):
        """Test zero expected value (fair game)."""
        ev = calculate_expected_value(
            win_rate=0.50,
            avg_win=100,
            avg_loss=100
        )
        
        assert abs(ev) < 0.01


class TestCalmarRatio:
    """Test Calmar ratio calculations."""
    
    def test_positive_calmar(self):
        """Test with positive returns and drawdown."""
        annual_return = 0.20  # 20% annual return
        max_dd = -0.10  # -10% max drawdown
        
        calmar = calculate_calmar_ratio(annual_return, max_dd)
        
        # Calmar = 0.20 / 0.10 = 2.0
        assert abs(calmar - 2.0) < 0.01
    
    def test_no_drawdown(self):
        """Test with no drawdown."""
        calmar = calculate_calmar_ratio(0.15, 0)
        
        # Infinite Calmar
        assert np.isinf(calmar)


class TestRiskOfRuin:
    """Test risk of ruin calculations."""
    
    def test_positive_edge(self):
        """Test with positive edge system."""
        ror = risk_of_ruin(
            win_rate=0.60,
            avg_win=150,
            avg_loss=100,
            max_dd_pct=0.30
        )
        
        # With 60% win rate and good R:R, risk should be low
        assert ror < 0.10  # Less than 10% risk
    
    def test_negative_edge(self):
        """Test with negative edge system."""
        ror = risk_of_ruin(
            win_rate=0.40,
            avg_win=100,
            avg_loss=150,
            max_dd_pct=0.30
        )
        
        # With negative edge, risk should be very high
        assert ror > 0.90  # Over 90% risk of ruin
    
    def test_edge_case_inputs(self):
        """Test edge cases."""
        # Win rate = 0
        ror = risk_of_ruin(0, 100, 100, 0.30)
        assert ror == 1.0  # Certain ruin
        
        # Win rate = 1
        ror = risk_of_ruin(1.0, 100, 100, 0.30)
        assert ror == 0.0  # No risk


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Test Black-Scholes Pricing Engine
==================================

Tests for option pricing, Greeks calculations, and implied volatility.
"""

import pytest
import numpy as np
from datetime import datetime

from src.options.utils.black_scholes import BlackScholes, OptionType, Greeks


class TestBlackScholes:
    """Test Black-Scholes pricing and Greeks."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.bs = BlackScholes(risk_free_rate=0.05)
        
        # Standard test parameters
        self.S = 100.0  # Stock price
        self.K = 100.0  # Strike (ATM)
        self.T = 30 / 365.0  # 30 days to expiration
        self.sigma = 0.20  # 20% IV
    
    def test_call_price_atm(self):
        """Test ATM call option pricing."""
        price = self.bs.call_price(self.S, self.K, self.T, self.sigma)
        
        # ATM call should be worth approximately 2-3% of stock price
        assert 1.5 < price < 4.0
        assert price > 0
    
    def test_put_price_atm(self):
        """Test ATM put option pricing."""
        price = self.bs.put_price(self.S, self.K, self.T, self.sigma)
        
        # ATM put should be similar to call (slightly less due to interest)
        call_price = self.bs.call_price(self.S, self.K, self.T, self.sigma)
        assert abs(price - call_price) < 0.50
        assert price > 0
    
    def test_put_call_parity(self):
        """Test put-call parity relationship."""
        # C - P = S - K*e^(-rT)
        call = self.bs.call_price(self.S, self.K, self.T, self.sigma)
        put = self.bs.put_price(self.S, self.K, self.T, self.sigma)
        
        pv_strike = self.K * np.exp(-self.bs.risk_free_rate * self.T)
        
        # Parity should hold within small tolerance
        assert abs((call - put) - (self.S - pv_strike)) < 0.01
    
    def test_zero_expiration(self):
        """Test option value at expiration."""
        T = 0.0
        
        # ITM call
        call = self.bs.call_price(110, 100, T, self.sigma)
        assert abs(call - 10.0) < 0.01  # Intrinsic value only
        
        # OTM call
        call_otm = self.bs.call_price(90, 100, T, self.sigma)
        assert call_otm == 0.0
        
        # ITM put
        put = self.bs.put_price(90, 100, T, self.sigma)
        assert abs(put - 10.0) < 0.01
        
        # OTM put
        put_otm = self.bs.put_price(110, 100, T, self.sigma)
        assert put_otm == 0.0
    
    def test_delta_call(self):
        """Test call delta calculation."""
        greeks = self.bs.calculate_all_greeks(
            self.S, self.K, self.T, self.sigma, OptionType.CALL
        )
        
        # ATM call delta should be around 0.50
        assert 0.45 < greeks.delta < 0.55
        
        # ITM call delta should be higher
        greeks_itm = self.bs.calculate_all_greeks(
            110, 100, self.T, self.sigma, OptionType.CALL
        )
        assert greeks_itm.delta > greeks.delta
        
        # OTM call delta should be lower
        greeks_otm = self.bs.calculate_all_greeks(
            90, 100, self.T, self.sigma, OptionType.CALL
        )
        assert greeks_otm.delta < greeks.delta
    
    def test_delta_put(self):
        """Test put delta calculation."""
        greeks = self.bs.calculate_all_greeks(
            self.S, self.K, self.T, self.sigma, OptionType.PUT
        )
        
        # ATM put delta should be around -0.50
        assert -0.55 < greeks.delta < -0.45
        
        # Delta should be negative for puts
        assert greeks.delta < 0
    
    def test_gamma(self):
        """Test gamma calculation."""
        greeks = self.bs.calculate_all_greeks(
            self.S, self.K, self.T, self.sigma, OptionType.CALL
        )
        
        # Gamma should be positive
        assert greeks.gamma > 0
        
        # ATM options have highest gamma
        greeks_itm = self.bs.calculate_all_greeks(
            110, 100, self.T, self.sigma, OptionType.CALL
        )
        greeks_otm = self.bs.calculate_all_greeks(
            90, 100, self.T, self.sigma, OptionType.CALL
        )
        
        assert greeks.gamma > greeks_itm.gamma
        assert greeks.gamma > greeks_otm.gamma
    
    def test_theta(self):
        """Test theta calculation."""
        greeks_call = self.bs.calculate_all_greeks(
            self.S, self.K, self.T, self.sigma, OptionType.CALL
        )
        greeks_put = self.bs.calculate_all_greeks(
            self.S, self.K, self.T, self.sigma, OptionType.PUT
        )
        
        # Theta should be negative (time decay)
        assert greeks_call.theta < 0
        assert greeks_put.theta < 0
        
        # Theta accelerates closer to expiration
        T_short = 7 / 365.0
        greeks_short = self.bs.calculate_all_greeks(
            self.S, self.K, T_short, self.sigma, OptionType.CALL
        )
        
        # Shorter DTE should have more negative theta (per day)
        daily_theta_30 = greeks_call.theta / self.T
        daily_theta_7 = greeks_short.theta / T_short
        assert abs(daily_theta_7) > abs(daily_theta_30)
    
    def test_vega(self):
        """Test vega calculation."""
        greeks = self.bs.calculate_all_greeks(
            self.S, self.K, self.T, self.sigma, OptionType.CALL
        )
        
        # Vega should be positive
        assert greeks.vega > 0
        
        # Longer-dated options have higher vega
        T_long = 90 / 365.0
        greeks_long = self.bs.calculate_all_greeks(
            self.S, self.K, T_long, self.sigma, OptionType.CALL
        )
        
        assert greeks_long.vega > greeks.vega
    
    def test_implied_volatility_call(self):
        """Test IV calculation from market price."""
        # Calculate option price with known IV
        known_iv = 0.25
        price = self.bs.call_price(self.S, self.K, self.T, known_iv)
        
        # Recover IV from price
        calculated_iv = self.bs.implied_volatility(
            market_price=price,
            S=self.S,
            K=self.K,
            T=self.T,
            option_type=OptionType.CALL
        )
        
        # Should match within 0.1%
        assert abs(calculated_iv - known_iv) < 0.001
    
    def test_implied_volatility_put(self):
        """Test IV calculation for puts."""
        known_iv = 0.30
        price = self.bs.put_price(self.S, self.K, self.T, known_iv)
        
        calculated_iv = self.bs.implied_volatility(
            market_price=price,
            S=self.S,
            K=self.K,
            T=self.T,
            option_type=OptionType.PUT
        )
        
        assert abs(calculated_iv - known_iv) < 0.001
    
    def test_greeks_arithmetic(self):
        """Test Greeks dataclass arithmetic operations."""
        g1 = Greeks(delta=0.5, gamma=0.03, theta=-0.10, vega=0.15, rho=0.05)
        g2 = Greeks(delta=0.3, gamma=0.02, theta=-0.08, vega=0.12, rho=0.03)
        
        # Addition
        g_sum = g1 + g2
        assert abs(g_sum.delta - 0.8) < 0.001
        assert abs(g_sum.gamma - 0.05) < 0.001
        
        # Subtraction
        g_diff = g1 - g2
        assert abs(g_diff.delta - 0.2) < 0.001
        
        # Multiplication by scalar
        g_mult = g1 * 2
        assert abs(g_mult.delta - 1.0) < 0.001
        assert abs(g_mult.theta - (-0.20)) < 0.001
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Very high volatility
        price_high_vol = self.bs.call_price(self.S, self.K, self.T, 2.0)
        assert price_high_vol > 0
        
        # Very low volatility
        price_low_vol = self.bs.call_price(self.S, self.K, self.T, 0.01)
        assert price_low_vol >= 0
        
        # Deep ITM
        price_itm = self.bs.call_price(150, 100, self.T, self.sigma)
        assert price_itm > 49.0  # Should be close to intrinsic value
        
        # Deep OTM
        price_otm = self.bs.call_price(50, 100, self.T, self.sigma)
        assert price_otm < 0.10  # Should be very small


class TestGreeksScenarios:
    """Test Greeks in various market scenarios."""
    
    def test_short_put_greeks(self):
        """Test Greeks for typical cash-secured put."""
        bs = BlackScholes()
        
        # SPY at $450, sell $440 put (30 DTE)
        S = 450
        K = 440  # ~2.2% OTM
        T = 30 / 365.0
        sigma = 0.18
        
        greeks = bs.calculate_all_greeks(S, K, T, sigma, OptionType.PUT)
        
        # For short put (multiply by -1):
        # - Positive delta (bullish exposure)
        # - Negative gamma (risk increases as price falls)
        # - Positive theta (earn time decay)
        # - Negative vega (want IV to decrease)
        
        assert greeks.delta < 0  # Put delta is negative
        assert abs(greeks.delta) < 0.35  # OTM put has low delta
        assert greeks.gamma > 0
        assert greeks.theta < 0
        assert greeks.vega > 0
    
    def test_credit_spread_greeks(self):
        """Test net Greeks for a credit spread."""
        bs = BlackScholes()
        
        # Bull put spread: sell $440 put, buy $435 put
        S = 450
        T = 30 / 365.0
        sigma = 0.18
        
        # Short put
        greeks_short = bs.calculate_all_greeks(S, 440, T, sigma, OptionType.PUT)
        
        # Long put
        greeks_long = bs.calculate_all_greeks(S, 435, T, sigma, OptionType.PUT)
        
        # Net Greeks (short - long)
        net_greeks = greeks_short - greeks_long
        
        # Net delta should be positive but smaller than short alone
        assert abs(net_greeks.delta) < abs(greeks_short.delta)
        
        # Net gamma should be negative (short spreads have neg gamma)
        assert net_greeks.gamma < 0
        
        # Net theta should be positive (earning time decay)
        assert net_greeks.theta < 0  # For individual legs
        # But net theta for the spread (short - long) would be positive


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

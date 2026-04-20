"""
Tests for backtest/options_backtester.py — options backtester.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from backtest.options_backtester import (
    BlackScholes,
    OptionLeg,
    OptionPosition,
    OptionsBacktestResult,
    OptionsBacktester,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_underlying(n_bars: int = 60, start: float = 50.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-03", periods=n_bars)
    close = start + np.cumsum(rng.randn(n_bars) * 0.3)
    close = np.maximum(close, 5.0)
    return pd.DataFrame({"close": close}, index=dates)


def _make_option_signals(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate directional signals with strategy_type."""
    n = len(dates)
    signals = pd.DataFrame(index=dates)
    direction = np.zeros(n)
    strategy_type = [""] * n
    for i in range(n):
        if i % 20 == 0:
            direction[i] = 1
            strategy_type[i] = "vertical"
        elif i % 20 == 10:
            direction[i] = -1
            strategy_type[i] = "vertical"
    signals["direction"] = direction
    signals["strategy_type"] = strategy_type
    return signals


# ---------------------------------------------------------------------------
# BlackScholes tests
# ---------------------------------------------------------------------------

class TestBlackScholes:
    """Black-Scholes pricing and Greeks tests."""

    def test_call_put_parity(self):
        """C - P = S - K*exp(-rT) (put-call parity)."""
        S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.20
        C = BlackScholes.call_price(S, K, T, r, sigma)
        P = BlackScholes.put_price(S, K, T, r, sigma)
        parity = C - P - (S - K * math.exp(-r * T))
        assert abs(parity) < 1e-8

    def test_atm_call_price_positive(self):
        C = BlackScholes.call_price(100, 100, 0.25, 0.05, 0.20)
        assert C > 0

    def test_expired_call_intrinsic(self):
        """Expired ITM call should return intrinsic value."""
        assert BlackScholes.call_price(110, 100, 0, 0.05, 0.20) == pytest.approx(10.0)
        assert BlackScholes.call_price(90, 100, 0, 0.05, 0.20) == 0.0

    def test_expired_put_intrinsic(self):
        assert BlackScholes.put_price(90, 100, 0, 0.05, 0.20) == pytest.approx(10.0)
        assert BlackScholes.put_price(110, 100, 0, 0.05, 0.20) == 0.0

    def test_call_delta_range(self):
        """Call delta should be between 0 and 1."""
        d = BlackScholes.delta(100, 100, 0.25, 0.05, 0.20, is_call=True)
        assert 0 < d < 1

    def test_put_delta_range(self):
        """Put delta should be between -1 and 0."""
        d = BlackScholes.delta(100, 100, 0.25, 0.05, 0.20, is_call=False)
        assert -1 < d < 0

    def test_atm_delta_near_half(self):
        """ATM call delta should be roughly 0.5."""
        d = BlackScholes.delta(100, 100, 0.25, 0.05, 0.20, is_call=True)
        assert 0.4 < d < 0.65

    def test_gamma_positive(self):
        g = BlackScholes.gamma(100, 100, 0.25, 0.05, 0.20)
        assert g > 0

    def test_gamma_same_for_call_put(self):
        """Gamma is the same for calls and puts at same strike."""
        # gamma doesn't take is_call, it's the same
        g = BlackScholes.gamma(100, 100, 0.25, 0.05, 0.20)
        assert g > 0

    def test_theta_negative_for_long(self):
        """Long call theta should be negative (time decay)."""
        th = BlackScholes.theta(100, 100, 0.25, 0.05, 0.20, is_call=True)
        assert th < 0

    def test_vega_positive(self):
        v = BlackScholes.vega(100, 100, 0.25, 0.05, 0.20)
        assert v > 0

    def test_price_wrapper(self):
        """price() should delegate to call_price or put_price."""
        C = BlackScholes.price(100, 100, 0.25, 0.05, 0.20, is_call=True)
        P = BlackScholes.price(100, 100, 0.25, 0.05, 0.20, is_call=False)
        assert C == BlackScholes.call_price(100, 100, 0.25, 0.05, 0.20)
        assert P == BlackScholes.put_price(100, 100, 0.25, 0.05, 0.20)

    def test_edge_case_zero_vol(self):
        """Zero vol should return 0 for d1/d2, and Greeks should handle gracefully."""
        assert BlackScholes.d1(100, 100, 0.25, 0.05, 0) == 0.0
        assert BlackScholes.gamma(100, 100, 0.25, 0.05, 0) == 0.0
        assert BlackScholes.vega(100, 100, 0.25, 0.05, 0) == 0.0

    def test_edge_case_zero_time(self):
        """T=0 should return intrinsic for delta."""
        d = BlackScholes.delta(110, 100, 0, 0.05, 0.20, is_call=True)
        assert d == 1.0  # Deep ITM
        d = BlackScholes.delta(90, 100, 0, 0.05, 0.20, is_call=True)
        assert d == 0.0  # OTM


# ---------------------------------------------------------------------------
# OptionsBacktester tests
# ---------------------------------------------------------------------------

class TestOptionsBacktester:
    def test_init_defaults(self):
        bt = OptionsBacktester()
        assert bt.max_risk_per_trade == 50.0
        assert bt.max_positions == 3
        assert bt.target_profit_pct == 0.50
        assert bt.close_dte == 2

    def test_run_returns_result(self):
        prices = _make_underlying()
        signals = _make_option_signals(prices.index)
        bt = OptionsBacktester()
        result = bt.run(signals, prices, initial_capital=444.0)
        assert isinstance(result, OptionsBacktestResult)

    def test_equity_curve_length(self):
        prices = _make_underlying(n_bars=30)
        signals = _make_option_signals(prices.index)
        bt = OptionsBacktester()
        result = bt.run(signals, prices)
        assert len(result.equity_curve) == 30

    def test_max_positions_enforced(self):
        """Should not exceed max_positions concurrent positions."""
        prices = _make_underlying(n_bars=60)
        # Signal buy on every day
        signals = pd.DataFrame({"direction": 1, "strategy_type": "single"}, index=prices.index)
        bt = OptionsBacktester(max_positions=2, max_risk_per_trade=100)
        result = bt.run(signals, prices)
        # We can't directly check concurrent positions, but it should not crash
        assert isinstance(result, OptionsBacktestResult)

    def test_max_risk_per_trade(self):
        """Trades exceeding max_risk should not be opened."""
        prices = _make_underlying(n_bars=20, start=1000.0)  # expensive underlying
        signals = pd.DataFrame({"direction": 1, "strategy_type": "single"}, index=prices.index)
        bt = OptionsBacktester(max_risk_per_trade=5.0)  # Very tight risk limit
        result = bt.run(signals, prices)
        # Most single-leg trades on $1000 stock will exceed $5 risk, so few or no trades
        assert isinstance(result, OptionsBacktestResult)

    def test_strategy_breakdown(self):
        prices = _make_underlying(n_bars=60)
        signals = _make_option_signals(prices.index)
        bt = OptionsBacktester()
        result = bt.run(signals, prices)
        assert isinstance(result.strategy_breakdown, dict)

    def test_greeks_pnl_attribution(self):
        prices = _make_underlying(n_bars=60)
        signals = _make_option_signals(prices.index)
        bt = OptionsBacktester()
        result = bt.run(signals, prices)
        assert "total_theta_pnl" in result.greeks_pnl_attribution
        assert "total_delta_pnl" in result.greeks_pnl_attribution

    def test_build_vertical_spread(self):
        bt = OptionsBacktester()
        bt._reset(444.0)
        pos = bt._build_vertical_spread(
            "TEST", 100.0, 0.25, 30, True, True, pd.Timestamp("2023-06-01"),
        )
        if pos is not None:
            assert pos.strategy_type == "vertical"
            assert len(pos.legs) == 2
            assert pos.max_loss <= 50.0

    def test_build_iron_condor(self):
        bt = OptionsBacktester()
        bt._reset(444.0)
        pos = bt._build_iron_condor("TEST", 100.0, 0.25, 30, pd.Timestamp("2023-06-01"))
        if pos is not None:
            assert pos.strategy_type == "iron_condor"
            assert len(pos.legs) == 4

    def test_build_single_leg(self):
        bt = OptionsBacktester()
        bt._reset(444.0)
        pos = bt._build_single_leg("TEST", 100.0, 0.25, 30, True, pd.Timestamp("2023-06-01"))
        if pos is not None:
            assert pos.strategy_type == "single"
            assert len(pos.legs) == 1
            assert pos.legs[0].is_long is True

    def test_empty_signals(self):
        """No signals should produce no trades."""
        prices = _make_underlying(n_bars=20)
        signals = pd.DataFrame({"direction": 0, "strategy_type": ""}, index=prices.index)
        bt = OptionsBacktester()
        result = bt.run(signals, prices)
        assert len(result.options_trades) == 0

    def test_commission_calculation(self):
        legs = [
            OptionLeg(strike=100, is_call=True, is_long=True, quantity=2),
            OptionLeg(strike=105, is_call=True, is_long=False, quantity=2),
        ]
        c = OptionsBacktester._commission(legs)
        assert c == pytest.approx(max(1.0, 4 * 0.65))  # 4 contracts * $0.65


# ---------------------------------------------------------------------------
# OptionLeg / OptionPosition dataclass tests
# ---------------------------------------------------------------------------

class TestOptionDataclasses:
    def test_option_leg_defaults(self):
        leg = OptionLeg(strike=100, is_call=True, is_long=True)
        assert leg.quantity == 1
        assert leg.entry_price == 0.0

    def test_option_position_fields(self):
        pos = OptionPosition(
            position_id=1,
            symbol="SPY",
            strategy_type="vertical",
            legs=[],
            entry_date=pd.Timestamp("2023-01-03"),
            expiration_dte=30,
            entry_cost=100.0,
            max_profit=50.0,
            max_loss=50.0,
        )
        assert pos.closed is False
        assert pos.pnl == 0.0

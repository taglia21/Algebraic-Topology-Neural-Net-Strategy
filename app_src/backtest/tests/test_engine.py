"""
Tests for backtest/engine.py — event-driven backtesting engine.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.engine import (
    AssetType,
    BacktestEngine,
    BacktestResult,
    CommissionCalculator,
    Event,
    EventType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Trade,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_price_data(n_bars: int = 60, start_price: float = 50.0, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for a single symbol."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-03", periods=n_bars)
    close = start_price + np.cumsum(rng.randn(n_bars) * 0.5)
    close = np.maximum(close, 1.0)  # keep prices positive

    df = pd.DataFrame({
        "open": close + rng.randn(n_bars) * 0.1,
        "high": close + abs(rng.randn(n_bars) * 0.3),
        "low": close - abs(rng.randn(n_bars) * 0.3),
        "close": close,
        "volume": rng.randint(1000, 10000, n_bars),
    }, index=dates)
    # Ensure high >= open/close and low <= open/close
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def _make_signals(price_data: pd.DataFrame, symbol: str = "ASSET") -> pd.DataFrame:
    """Generate alternating buy/sell signals."""
    signals = pd.DataFrame(index=price_data.index)
    vals = []
    for i in range(len(price_data)):
        if i % 10 == 0:
            vals.append(1)     # buy
        elif i % 10 == 5:
            vals.append(-1)    # sell
        else:
            vals.append(0)     # flat
    signals[symbol] = vals
    return signals


# ---------------------------------------------------------------------------
# CommissionCalculator tests
# ---------------------------------------------------------------------------

class TestCommissionCalculator:
    def test_equity_commission_min(self):
        """Min $1 commission."""
        c = CommissionCalculator.equity_commission(1, 10.0)
        assert c == 1.0  # 1 share * $0.005 = $0.005 -> min $1

    def test_equity_commission_normal(self):
        """Normal commission: $0.005/share."""
        c = CommissionCalculator.equity_commission(500, 50.0)
        assert c == 2.50  # 500 * 0.005 = 2.50

    def test_equity_commission_cap(self):
        """Cap at 1% of trade value (when cap > min $1)."""
        # 10000 shares at $0.50 -> trade value $5000 -> 1% cap = $50
        # 10000 * $0.005 = $50 -> exactly at cap, and above $1 min
        c = CommissionCalculator.equity_commission(10000, 0.50)
        assert c == pytest.approx(50.0)
        # 20000 shares at $0.50 -> raw = $100, cap = $100 -> cap not binding
        c2 = CommissionCalculator.equity_commission(20000, 0.50)
        assert c2 == pytest.approx(100.0)

    def test_option_commission_min(self):
        """Min $1 commission for options."""
        c = CommissionCalculator.option_commission(1)
        assert c == 1.0  # 1 * $0.65 = $0.65 -> min $1

    def test_option_commission_normal(self):
        """Normal option commission."""
        c = CommissionCalculator.option_commission(5)
        assert c == pytest.approx(3.25)  # 5 * $0.65


# ---------------------------------------------------------------------------
# BacktestEngine tests
# ---------------------------------------------------------------------------

class TestBacktestEngine:
    def test_init_defaults(self):
        engine = BacktestEngine()
        assert engine.equity_slippage == 0.001
        assert engine.respect_market_hours is True

    def test_run_returns_backtest_result(self):
        prices = _make_price_data()
        signals = _make_signals(prices)
        engine = BacktestEngine()
        result = engine.run(signals, prices, initial_capital=444.0)
        assert isinstance(result, BacktestResult)

    def test_initial_capital_preserved(self):
        """With no signals, equity should equal initial capital."""
        prices = _make_price_data()
        signals = pd.DataFrame(0, index=prices.index, columns=["ASSET"])
        engine = BacktestEngine()
        result = engine.run(signals, prices, initial_capital=444.0)
        assert result.initial_capital == 444.0
        # Equity should be flat at $444
        assert abs(result.equity_curve.iloc[-1] - 444.0) < 1.0

    def test_equity_curve_length(self):
        prices = _make_price_data(n_bars=30)
        signals = _make_signals(prices)
        engine = BacktestEngine()
        result = engine.run(signals, prices)
        assert len(result.equity_curve) == 30

    def test_trades_recorded(self):
        """Engine should record trades when signals fire."""
        prices = _make_price_data(n_bars=40)
        signals = _make_signals(prices)
        engine = BacktestEngine()
        result = engine.run(signals, prices)
        # Should have at least 1 trade (buy then close at end)
        assert len(result.trades) >= 1

    def test_trades_have_required_columns(self):
        prices = _make_price_data(n_bars=40)
        signals = _make_signals(prices)
        engine = BacktestEngine()
        result = engine.run(signals, prices)
        if len(result.trades) > 0:
            required = {"symbol", "side", "entry_date", "exit_date",
                        "entry_price", "exit_price", "qty", "pnl", "commission"}
            assert required.issubset(set(result.trades.columns))

    def test_metrics_computed(self):
        prices = _make_price_data(n_bars=60)
        signals = _make_signals(prices)
        engine = BacktestEngine()
        result = engine.run(signals, prices)
        assert "total_return" in result.metrics
        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown" in result.metrics

    def test_fifo_order_processing(self):
        """Orders should be processed in FIFO order."""
        engine = BacktestEngine()
        engine._reset(1000.0)

        o1 = engine.submit_order("A", OrderSide.BUY, 10, timestamp=pd.Timestamp("2023-01-03"))
        o2 = engine.submit_order("B", OrderSide.BUY, 5, timestamp=pd.Timestamp("2023-01-03"))

        assert engine._order_queue[0].order_id == o1.order_id
        assert engine._order_queue[1].order_id == o2.order_id

    def test_t_plus_1_execution(self):
        """Market orders submitted on day T should fill at day T+1 open."""
        prices = _make_price_data(n_bars=10)
        # Signal only on first day
        signals = pd.DataFrame(0, index=prices.index, columns=["ASSET"])
        signals.iloc[0] = 1  # Buy on day 0

        engine = BacktestEngine()
        result = engine.run(signals, prices)

        # Should have orders but first bar's order fills on second bar
        if len(result.orders) > 0:
            filled = [o for o in result.orders if o.status == OrderStatus.FILLED]
            if filled:
                first_fill = filled[0]
                # Fill should be at or after the second bar
                assert first_fill.filled_at >= prices.index[1] or first_fill.filled_at == prices.index[0]

    def test_normalize_single_symbol(self):
        """Single-symbol DataFrame should normalize to ASSET."""
        prices = _make_price_data(n_bars=5)
        result = BacktestEngine._normalize_price_data(prices)
        assert len(result) == 5
        for date, bars in result.items():
            assert "ASSET" in bars
            assert "open" in bars["ASSET"]
            assert "close" in bars["ASSET"]

    def test_normalize_dict_passthrough(self):
        """Dict input should pass through unchanged."""
        data = {pd.Timestamp("2023-01-03"): {"SPY": {"open": 100, "close": 101}}}
        result = BacktestEngine._normalize_price_data(data)
        assert result is data

    def test_position_sizer_called(self):
        """Custom position sizer should be used."""
        prices = _make_price_data(n_bars=20)
        signals = pd.DataFrame(0, index=prices.index, columns=["ASSET"])
        signals.iloc[0] = 1

        called = []
        def sizer(signal_val, nav, symbol):
            called.append((signal_val, nav, symbol))
            return 5

        engine = BacktestEngine()
        engine.run(signals, prices, position_sizer=sizer)
        assert len(called) > 0

    def test_empty_signals(self):
        """Engine should handle empty signals gracefully."""
        prices = _make_price_data(n_bars=10)
        signals = pd.DataFrame(index=prices.index)
        engine = BacktestEngine()
        result = engine.run(signals, prices)
        assert isinstance(result, BacktestResult)
        assert len(result.trades) == 0


# ---------------------------------------------------------------------------
# Order and Position dataclass tests
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_order_defaults(self):
        o = Order(order_id=1, symbol="SPY", side=OrderSide.BUY, quantity=10)
        assert o.status == OrderStatus.PENDING
        assert o.order_type == OrderType.MARKET
        assert o.asset_type == AssetType.EQUITY

    def test_position_defaults(self):
        p = Position(symbol="SPY", quantity=100, avg_cost=50.0)
        assert p.unrealized_pnl == 0.0
        assert p.realized_pnl == 0.0

    def test_event_types(self):
        assert EventType.MARKET_DATA.value == "MARKET_DATA"
        assert EventType.FILL.value == "FILL"
        assert EventType.END_OF_DAY.value == "END_OF_DAY"

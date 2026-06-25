from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd


def test_execution_manager_forwards_returns_and_sector_context():
    from core.risk_manager import TradeApproval
    from equities.execution import ExecutionManager, SimulatedBroker
    from equities.models import Signal

    class _RiskStub:
        def __init__(self) -> None:
            self.last_kwargs = None

        def approve_trade(self, **kwargs):
            self.last_kwargs = kwargs
            return TradeApproval(approved=True, reason="", suggested_qty=kwargs["qty"])

    broker = SimulatedBroker(initial_cash=100_000.0)
    risk = _RiskStub()
    exec_mgr = ExecutionManager(broker=broker, risk_manager=risk, order_type="market")

    signal = Signal(
        symbol="AAPL",
        direction="long",
        strength=0.8,
        strategy="test",
        metadata={"sector": "Technology"},
    )
    returns_data = pd.DataFrame(
        {
            "AAPL": [0.01, -0.005, 0.004],
            "MSFT": [0.009, -0.004, 0.003],
        }
    )

    orders = exec_mgr.process_signals(
        [signal],
        current_prices={"AAPL": 100.0},
        returns_data=returns_data,
    )

    assert len(orders) == 1
    assert risk.last_kwargs is not None
    assert risk.last_kwargs["returns_data"] is returns_data
    assert risk.last_kwargs["portfolio_state"].sector_map["AAPL"] == "Technology"


def test_execution_manager_uses_configured_gross_exposure_cap():
    from unittest.mock import patch

    from equities.execution import ExecutionManager, SimulatedBroker
    from equities.models import PortfolioState, Position, Signal

    broker = SimulatedBroker(initial_cash=100_000.0)
    exec_mgr = ExecutionManager(broker=broker, risk_manager=SimpleNamespace())

    # Simulate a portfolio above a 150% gross exposure cap.
    portfolio = PortfolioState(
        equity=100_000.0,
        cash=0.0,
        positions={
            "AAPL": Position("AAPL", qty=600, avg_entry=250.0, current_price=250.0, unrealized_pnl=0.0),
        },
        peak_equity=100_000.0,
    )
    signal = Signal(symbol="MSFT", direction="long", strength=0.9, strategy="test")

    cfg = SimpleNamespace(risk=SimpleNamespace(max_position_pct=0.2, max_gross_exposure=1.5))
    with patch("equities.execution.get_config", return_value=cfg):
        qty = exec_mgr._compute_order_qty(signal, price=100.0, portfolio_state=portfolio)

    assert qty == 0


def test_simulated_broker_applies_short_borrow_once_per_day():
    from equities.execution import SimulatedBroker
    from equities.models import Position

    broker = SimulatedBroker(initial_cash=100_000.0, short_borrow_rate=0.252)
    broker._positions["AAPL"] = Position(
        symbol="AAPL",
        qty=-100,
        avg_entry=100.0,
        current_price=100.0,
        unrealized_pnl=0.0,
    )

    broker._current_bar_dt = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    cash_before = broker.cash
    broker.update_prices({"AAPL": 100.0})
    cash_after_first = broker.cash

    # Same trading day: no second borrow deduction.
    broker._current_bar_dt = datetime(2026, 1, 2, 20, 0, tzinfo=timezone.utc)
    broker.update_prices({"AAPL": 100.0})
    cash_after_second = broker.cash

    assert cash_after_first < cash_before
    assert cash_after_second == cash_after_first


def test_backtester_trade_reconstruction_handles_partial_reversal():
    from backtest.backtester import Backtester
    from equities.models import Fill

    bt = Backtester(verbose=False)
    ts1 = datetime(2026, 1, 2, tzinfo=timezone.utc)
    ts2 = datetime(2026, 1, 3, tzinfo=timezone.utc)
    ts3 = datetime(2026, 1, 4, tzinfo=timezone.utc)

    bt.broker._fills = [
        Fill(order_id="1", symbol="AAPL", side="buy", fill_price=10.0, fill_qty=100, timestamp=ts1),
        Fill(order_id="2", symbol="AAPL", side="sell", fill_price=12.0, fill_qty=150, timestamp=ts2),
        Fill(order_id="3", symbol="AAPL", side="buy", fill_price=11.0, fill_qty=50, timestamp=ts3),
    ]

    trades = bt._build_trades_from_fills()

    assert len(trades) == 2
    assert trades[0]["side"] == "long"
    assert trades[0]["qty"] == 100
    assert trades[0]["pnl"] == 200.0
    assert trades[1]["side"] == "short"
    assert trades[1]["qty"] == 50
    assert trades[1]["pnl"] == 50.0


def test_count_out_of_hours_handles_dst_correctly():
    from data.data_manager import _count_out_of_hours

    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-01-15T14:30:00Z"), "AAPL"),  # 09:30 ET (winter) inside
            (pd.Timestamp("2026-07-15T13:30:00Z"), "AAPL"),  # 09:30 ET (summer) inside
            (pd.Timestamp("2026-07-15T21:30:00Z"), "AAPL"),  # 17:30 ET (summer) outside
        ],
        names=["datetime", "symbol"],
    )
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=idx)

    assert _count_out_of_hours(df) == 1


def test_walk_forward_validation_uses_forward_returns_when_provided():
    from ml.validation import walk_forward_validate

    class _AlwaysLongModel:
        def train(self, X_train, y_train, X_val=None, y_val=None):
            return None

        def predict(self, X):
            return np.ones(len(X), dtype=float)

        def get_feature_importance(self):
            return pd.Series({c: 1.0 for c in ["f1", "f2"]})

    n = 180
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    feats = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)}, index=idx)
    labels = pd.Series(np.ones(n), index=idx)
    # True forward returns are consistently negative.
    fwd = pd.Series(-0.01 - 0.001 * np.sin(np.arange(n)), index=idx)

    result = walk_forward_validate(
        model_factory=lambda: _AlwaysLongModel(),
        features=feats,
        labels=labels,
        forward_returns=fwd,
        train_window=80,
        test_window=20,
        step=20,
        min_windows=3,
    )

    assert result["n_windows"] >= 3
    assert float(np.mean(result["oos_returns"])) < 0.0


def test_backtest_config_reads_impact_and_borrow_env(monkeypatch):
    from core.config import get_config

    monkeypatch.setenv("BACKTEST_MARKET_IMPACT_FACTOR", "0.27")
    monkeypatch.setenv("BACKTEST_SHORT_BORROW_RATE", "0.05")

    cfg = get_config(reload=True)

    assert cfg.backtest.market_impact_factor == 0.27
    assert cfg.backtest.short_borrow_rate == 0.05


def test_execution_manager_raises_strength_floor_when_costs_rise(monkeypatch):
    from core.config import get_config
    from equities.execution import ExecutionManager, SimulatedBroker

    low_cost_cfg = get_config(reload=True)
    low_cost_cfg.backtest.slippage_bps = 7.0
    low_cost_cfg.backtest.commission_per_share = 0.005

    high_cost_cfg = get_config(reload=True)
    high_cost_cfg.backtest.slippage_bps = 20.0
    high_cost_cfg.backtest.commission_per_share = 0.02

    broker = SimulatedBroker(initial_cash=100_000.0)
    exec_mgr = ExecutionManager(broker=broker, risk_manager=SimpleNamespace())

    monkeypatch.setattr("equities.execution.get_config", lambda: low_cost_cfg)
    low_floor = exec_mgr._cost_adjusted_strength_floor(price=100.0)

    monkeypatch.setattr("equities.execution.get_config", lambda: high_cost_cfg)
    high_floor = exec_mgr._cost_adjusted_strength_floor(price=100.0)

    assert high_floor > low_floor
    assert low_floor >= 0.20


def test_execution_manager_strength_floor_uses_liquidity_context(monkeypatch):
    from core.config import get_config
    from core.risk_manager import TradeApproval
    from equities.execution import ExecutionManager, SimulatedBroker
    from equities.models import Signal

    class _ApproveAllRisk:
        def approve_trade(self, **kwargs):
            return TradeApproval(approved=True, reason="", suggested_qty=kwargs["qty"])

    broker = SimulatedBroker(initial_cash=100_000.0)
    exec_mgr = ExecutionManager(broker=broker, risk_manager=_ApproveAllRisk())

    cfg = get_config(reload=True)
    cfg.backtest.slippage_bps = 7.0
    cfg.backtest.commission_per_share = 0.005
    cfg.backtest.market_impact_factor = 0.10
    monkeypatch.setattr("equities.execution.get_config", lambda: cfg)

    signal = Signal(
        symbol="AAPL",
        direction="long",
        strength=0.27,
        strategy="test",
        metadata={"pre_scale_strength": 0.27},
    )

    high_liquidity = pd.DataFrame(
        {"AAPL": [100_000.0] * 25},
        index=pd.date_range("2024-01-01", periods=25, freq="B"),
    )
    low_liquidity = pd.DataFrame(
        {"AAPL": [100.0] * 25},
        index=pd.date_range("2024-01-01", periods=25, freq="B"),
    )

    orders_high = exec_mgr.process_signals(
        [signal],
        current_prices={"AAPL": 100.0},
        volume_data=high_liquidity,
    )

    orders_low = exec_mgr.process_signals(
        [signal],
        current_prices={"AAPL": 100.0},
        volume_data=low_liquidity,
    )

    assert len(orders_high) == 1
    assert len(orders_low) == 0

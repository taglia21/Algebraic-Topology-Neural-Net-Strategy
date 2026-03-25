"""
Tests for backtest/walk_forward.py — walk-forward optimization framework.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.walk_forward import (
    WalkForwardOptimizer,
    WalkForwardResult,
    WindowResult,
    WindowSchedule,
    _grid_search_combos,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _trading_dates(n: int = 1000) -> pd.DatetimeIndex:
    """Generate n business days."""
    return pd.bdate_range("2019-01-02", periods=n)


def _make_features(dates: pd.DatetimeIndex, n_features: int = 5) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    data = rng.randn(len(dates), n_features)
    return pd.DataFrame(data, index=dates, columns=[f"f{i}" for i in range(n_features)])


def _make_prices(dates: pd.DatetimeIndex, start: float = 100.0) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    close = start + np.cumsum(rng.randn(len(dates)) * 0.5)
    close = np.maximum(close, 1.0)
    return pd.DataFrame({"close": close}, index=dates)


# ---------------------------------------------------------------------------
# WindowSchedule tests
# ---------------------------------------------------------------------------

class TestWindowSchedule:
    def test_default_schedule(self):
        """Default 756/21/5/5 should generate windows from 1000 bars."""
        opt = WalkForwardOptimizer()
        dates = _trading_dates(1000)
        schedule = opt.get_window_schedule(dates[0], dates[-1], dates)
        assert len(schedule) > 0
        assert all(isinstance(w, WindowSchedule) for w in schedule)

    def test_insufficient_data(self):
        """Not enough data should return empty schedule."""
        opt = WalkForwardOptimizer(train_window=756)
        dates = _trading_dates(100)  # far too few
        schedule = opt.get_window_schedule(dates[0], dates[-1], dates)
        assert len(schedule) == 0

    def test_train_end_before_test_start(self):
        """Train end should always be before test start."""
        opt = WalkForwardOptimizer()
        dates = _trading_dates(1000)
        schedule = opt.get_window_schedule(dates[0], dates[-1], dates)
        for w in schedule:
            assert w.train_end < w.test_start

    def test_purge_gap_respected(self):
        """Gap between train_end and test_start should be >= purge_gap."""
        opt = WalkForwardOptimizer(train_window=100, test_window=21, purge_gap=5, embargo=5)
        dates = _trading_dates(300)
        schedule = opt.get_window_schedule(dates[0], dates[-1], dates)
        for w in schedule:
            train_end_pos = dates.get_loc(w.train_end)
            test_start_pos = dates.get_loc(w.test_start)
            assert test_start_pos - train_end_pos >= 5  # purge_gap

    def test_windows_non_overlapping_test_periods(self):
        """Test periods should not overlap."""
        opt = WalkForwardOptimizer(train_window=100, test_window=21, purge_gap=5, embargo=5)
        dates = _trading_dates(500)
        schedule = opt.get_window_schedule(dates[0], dates[-1], dates)
        for i in range(1, len(schedule)):
            assert schedule[i].test_start > schedule[i-1].test_end

    def test_window_ids_sequential(self):
        opt = WalkForwardOptimizer(train_window=100, test_window=21, purge_gap=5, embargo=5)
        dates = _trading_dates(500)
        schedule = opt.get_window_schedule(dates[0], dates[-1], dates)
        for i, w in enumerate(schedule):
            assert w.window_id == i

    def test_custom_parameters(self):
        opt = WalkForwardOptimizer(train_window=200, test_window=10, purge_gap=3, embargo=3)
        assert opt.train_window == 200
        assert opt.test_window == 10
        assert opt.purge_gap == 3
        assert opt.embargo == 3

    def test_generated_business_days(self):
        """If trading_dates not provided, should use bdate_range."""
        opt = WalkForwardOptimizer(train_window=100, test_window=21, purge_gap=5, embargo=5)
        start = pd.Timestamp("2020-01-02")
        end = pd.Timestamp("2021-06-30")
        schedule = opt.get_window_schedule(start, end)
        assert len(schedule) > 0


# ---------------------------------------------------------------------------
# WalkForwardOptimizer.run tests
# ---------------------------------------------------------------------------

class TestWalkForwardRun:
    def test_run_returns_result(self):
        """Full run should return WalkForwardResult."""
        dates = _trading_dates(300)
        features = _make_features(dates)
        prices = _make_prices(dates)

        opt = WalkForwardOptimizer(train_window=100, test_window=21, purge_gap=5, embargo=5)

        def model_factory(**kwargs):
            return {"type": "dummy"}

        def train_fn(model, feats, price, targets):
            return model

        def predict_fn(model, feats):
            return pd.DataFrame({"prediction": np.zeros(len(feats))}, index=feats.index)

        result = opt.run(prices, features, model_factory, predict_fn, train_fn)
        assert isinstance(result, WalkForwardResult)
        assert result.aggregate_metrics["total_windows"] > 0

    def test_oos_predictions_only(self):
        """Result should only contain out-of-sample predictions."""
        dates = _trading_dates(300)
        features = _make_features(dates)
        prices = _make_prices(dates)

        opt = WalkForwardOptimizer(train_window=100, test_window=21, purge_gap=5, embargo=5)

        def model_factory(**kwargs):
            return {}

        def train_fn(model, feats, price, targets):
            return model

        def predict_fn(model, feats):
            return pd.DataFrame({"prediction": np.ones(len(feats))}, index=feats.index)

        result = opt.run(prices, features, model_factory, predict_fn, train_fn)
        # All predictions should come from test windows
        assert len(result.oos_predictions) > 0
        assert "window_id" in result.oos_predictions.columns

    def test_empty_data_returns_empty(self):
        """Run with too little data should return empty result."""
        dates = _trading_dates(10)
        features = _make_features(dates)
        prices = _make_prices(dates)

        opt = WalkForwardOptimizer(train_window=100)
        result = opt.run(
            prices, features,
            model_factory=lambda **kw: {},
            predict_fn=lambda m, f: pd.DataFrame(),
            train_fn=lambda m, f, p, t: m,
        )
        assert isinstance(result, WalkForwardResult)
        assert len(result.oos_predictions) == 0

    def test_param_grid_optimization(self):
        """Model factory should be called with grid parameters."""
        dates = _trading_dates(300)
        features = _make_features(dates)
        prices = _make_prices(dates)

        opt = WalkForwardOptimizer(train_window=100, test_window=21, purge_gap=5, embargo=5)
        received_params = []

        def model_factory(**kwargs):
            received_params.append(kwargs)
            return {"lr": kwargs.get("lr", 0.01)}

        def train_fn(model, feats, price, targets):
            return model

        def predict_fn(model, feats):
            return pd.DataFrame({"prediction": np.zeros(len(feats))}, index=feats.index)

        def metric_fn(preds, actuals):
            return {"score": 0.5}

        grid = {"lr": [0.01, 0.1]}
        result = opt.run(
            prices, features, model_factory, predict_fn, train_fn,
            param_grid=grid, metric_fn=metric_fn,
        )
        # Should have tried both lr values at least once
        lr_values = {p.get("lr") for p in received_params}
        assert 0.01 in lr_values
        assert 0.1 in lr_values


# ---------------------------------------------------------------------------
# Grid search helper tests
# ---------------------------------------------------------------------------

class TestGridSearchCombos:
    def test_empty_grid(self):
        assert _grid_search_combos({}) == [{}]

    def test_single_param(self):
        combos = _grid_search_combos({"a": [1, 2, 3]})
        assert len(combos) == 3
        assert {"a": 1} in combos
        assert {"a": 2} in combos
        assert {"a": 3} in combos

    def test_two_params(self):
        combos = _grid_search_combos({"a": [1, 2], "b": [10, 20]})
        assert len(combos) == 4
        assert {"a": 1, "b": 10} in combos
        assert {"a": 2, "b": 20} in combos

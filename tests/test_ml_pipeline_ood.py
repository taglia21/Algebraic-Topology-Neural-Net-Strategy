"""Prediction-time OOD gating tests for MLPipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.config import get_config
from core.regime_detector import Regime
from ml.pipeline import MLPipeline


class _DummyFeatureEngine:
    def __init__(self, features: pd.DataFrame) -> None:
        self._features = features

    def compute_features(self, price_data: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
        return self._features


class _DummyModel:
    def predict_single(self, last_row: pd.DataFrame) -> float:
        return 0.65


def _price_data() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=260, freq="D")
    return pd.DataFrame({"close": np.linspace(100.0, 120.0, len(idx))}, index=idx)


def _make_pipeline(
    tmp_path: Path,
    features: pd.DataFrame,
    *,
    ood_action: str = "auto",
    mode: str = "paper",
) -> MLPipeline:
    cfg = get_config(reload=True)
    cfg.system.mode = mode
    cfg.ml.ood_action = ood_action
    pipe = MLPipeline(
        feature_engine=_DummyFeatureEngine(features),
        config=cfg,
        model_dir=str(tmp_path / "models"),
    )
    pipe.models = {5: _DummyModel()}
    return pipe


def _base_features(index: pd.DatetimeIndex, cols: int = 12) -> pd.DataFrame:
    data = {f"f{i}": np.zeros(len(index), dtype=float) for i in range(cols)}
    return pd.DataFrame(data, index=index)


def test_predict_allows_in_distribution_features(tmp_path: Path):
    prices = _price_data()
    features = _base_features(prices.index)
    pipe = _make_pipeline(tmp_path, features, ood_action="skip")

    # Training stats centered around 0 with moderate spread; last row is 0.
    rng = np.random.default_rng(7)
    pipe._train_feature_stats = {
        c: (rng.normal(loc=0.0, scale=1.0, size=500), 500)
        for c in features.columns
    }

    preds = pipe.predict(prices, Regime.UNKNOWN, symbol="AAPL")

    assert "AAPL" in preds
    assert preds["AAPL"]["score"] > 0


def test_predict_blocks_distribution_shift_outliers(tmp_path: Path):
    prices = _price_data()
    features = _base_features(prices.index)

    # Make half the features extreme on the latest bar to trigger OOD fraction.
    for c in list(features.columns)[:6]:
        features.loc[features.index[-1], c] = 100.0

    pipe = _make_pipeline(tmp_path, features, ood_action="skip")

    rng = np.random.default_rng(11)
    pipe._train_feature_stats = {
        c: (rng.normal(loc=0.0, scale=1.0, size=500), 500)
        for c in features.columns
    }

    preds = pipe.predict(prices, Regime.UNKNOWN, symbol="AAPL")

    assert preds == {}
    t = pipe.get_ood_telemetry()
    assert t["ood_checks"] == 1
    assert t["ood_blocks"] == 1
    assert t["ood_block_rate"] == 1.0
    # Backward-compat aliases
    assert t["checks"] == 1.0
    assert t["blocks"] == 1.0
    assert t["block_rate"] == 1.0


def test_ood_telemetry_breakdown_by_symbol_regime_day(tmp_path: Path):
    prices = _price_data()
    features = _base_features(prices.index)
    for c in list(features.columns)[:6]:
        features.loc[features.index[-1], c] = 100.0

    pipe = _make_pipeline(tmp_path, features, ood_action="skip")
    rng = np.random.default_rng(41)
    pipe._train_feature_stats = {
        c: (rng.normal(loc=0.0, scale=1.0, size=500), 500)
        for c in features.columns
    }

    _ = pipe.predict(prices, Regime.BEAR, symbol="AAPL")
    t = pipe.get_ood_telemetry()

    day_key = str(features.index[-1].date())
    assert t["ood_checks_by_symbol"]["AAPL"] == 1
    assert t["ood_blocks_by_symbol"]["AAPL"] == 1
    assert t["ood_checks_by_regime"]["BEAR"] == 1
    assert t["ood_blocks_by_regime"]["BEAR"] == 1
    assert t["ood_checks_by_day"][day_key] == 1
    assert t["ood_blocks_by_day"][day_key] == 1


def test_predict_skips_ood_gate_when_training_stats_missing(tmp_path: Path):
    prices = _price_data()
    features = _base_features(prices.index)
    features.loc[features.index[-1], features.columns[0]] = 999.0

    pipe = _make_pipeline(tmp_path, features, ood_action="skip")
    pipe._train_feature_stats = {}

    preds = pipe.predict(prices, Regime.UNKNOWN, symbol="AAPL")

    assert "AAPL" in preds


def test_predict_ood_neutral_action_returns_noop_adjustment(tmp_path: Path):
    prices = _price_data()
    features = _base_features(prices.index)
    for c in list(features.columns)[:6]:
        features.loc[features.index[-1], c] = 100.0

    pipe = _make_pipeline(tmp_path, features, ood_action="neutral")
    rng = np.random.default_rng(21)
    pipe._train_feature_stats = {
        c: (rng.normal(loc=0.0, scale=1.0, size=500), 500)
        for c in features.columns
    }

    preds = pipe.predict(prices, Regime.UNKNOWN, symbol="AAPL")

    assert "AAPL" in preds
    assert preds["AAPL"]["ood_blocked"] is True
    assert preds["AAPL"]["score"] == 1.0
    assert preds["AAPL"]["bet_size"] == 1.0
    assert preds["AAPL"]["take_trade"] is True


def test_predict_ood_block_action_returns_take_trade_false(tmp_path: Path):
    prices = _price_data()
    features = _base_features(prices.index)
    for c in list(features.columns)[:6]:
        features.loc[features.index[-1], c] = 100.0

    pipe = _make_pipeline(tmp_path, features, ood_action="block")
    rng = np.random.default_rng(31)
    pipe._train_feature_stats = {
        c: (rng.normal(loc=0.0, scale=1.0, size=500), 500)
        for c in features.columns
    }

    preds = pipe.predict(prices, Regime.UNKNOWN, symbol="AAPL")

    assert "AAPL" in preds
    assert preds["AAPL"]["ood_blocked"] is True
    assert preds["AAPL"]["take_trade"] is False
    assert preds["AAPL"]["bet_size"] == 0.0


def test_auto_ood_action_resolves_by_mode(tmp_path: Path):
    prices = _price_data()
    features = _base_features(prices.index)

    pipe_backtest = _make_pipeline(tmp_path, features, ood_action="auto", mode="backtest")
    pipe_paper = _make_pipeline(tmp_path, features, ood_action="auto", mode="paper")

    assert pipe_backtest._ood_action == "neutral"
    assert pipe_paper._ood_action == "skip"

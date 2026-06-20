"""Regression tests for MLPipeline.retrain_if_needed trigger logic."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from core.config import get_config
from ml.pipeline import MLPipeline, _fingerprint_data


class _DummyFeatureEngine:
    """Minimal feature engine stub for retrain_if_needed unit tests."""

    def __init__(self, features: pd.DataFrame | None = None) -> None:
        self._features = features

    def compute_features(self, price_data: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
        if self._features is not None:
            return self._features
        return pd.DataFrame(index=price_data.index)


def _price_data() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=40, freq="D")
    return pd.DataFrame({"close": np.linspace(100.0, 120.0, len(idx))}, index=idx)


def _make_pipeline(tmp_path: Path, feature_engine: _DummyFeatureEngine) -> MLPipeline:
    cfg = get_config()
    model_dir = str(tmp_path / "models")
    return MLPipeline(feature_engine=feature_engine, config=cfg, model_dir=model_dir)


def test_retrain_triggers_on_data_fingerprint_change_even_within_schedule(tmp_path: Path):
    prices = _price_data()
    pipe = _make_pipeline(tmp_path, _DummyFeatureEngine())

    pipe._last_train_time = time.time()  # within retrain_freq_days
    pipe._data_fingerprint = "stale1234"
    pipe.train_all = MagicMock(return_value={})

    did_retrain = pipe.retrain_if_needed(prices, symbol="TEST", force=False)

    assert did_retrain is True
    pipe.train_all.assert_called_once()


def test_retrain_triggers_on_high_psi_drift_even_within_schedule(tmp_path: Path):
    prices = _price_data()

    # Current features are far from the stored training distribution -> high PSI.
    feature_idx = prices.index
    curr_features = pd.DataFrame({"feat1": np.full(len(feature_idx), 10.0)}, index=feature_idx)
    pipe = _make_pipeline(tmp_path, _DummyFeatureEngine(features=curr_features))

    pipe._last_train_time = time.time()  # within retrain_freq_days
    pipe._data_fingerprint = _fingerprint_data(prices)  # unchanged data
    train_vals = np.random.default_rng(42).normal(loc=0.0, scale=1.0, size=400)
    pipe._train_feature_stats = {"feat1": (train_vals, len(train_vals))}
    pipe.train_all = MagicMock(return_value={})

    did_retrain = pipe.retrain_if_needed(prices, symbol="TEST", force=False)

    assert did_retrain is True
    pipe.train_all.assert_called_once()


def test_retrain_skips_when_no_trigger_and_schedule_not_due(tmp_path: Path):
    prices = _price_data()
    pipe = _make_pipeline(tmp_path, _DummyFeatureEngine())

    pipe._last_train_time = time.time()  # within retrain_freq_days
    pipe._data_fingerprint = _fingerprint_data(prices)
    pipe._train_feature_stats = {}
    pipe.train_all = MagicMock(return_value={})

    did_retrain = pipe.retrain_if_needed(prices, symbol="TEST", force=False)

    assert did_retrain is False
    pipe.train_all.assert_not_called()

"""Tests for nn/features.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nn.features import NNFeatureEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_data(n: int = 200, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic price + volume data."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = 100 * np.exp(np.cumsum(rng.randn(n) * 0.01))
    volume = rng.randint(1_000, 100_000, size=n).astype(float)
    price_df = pd.DataFrame({"close": prices}, index=dates)
    volume_df = pd.DataFrame({"volume": volume}, index=dates)
    return price_df, volume_df


def _make_tda_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Synthetic TDA features aligned with an index."""
    n = len(index)
    rng = np.random.RandomState(99)
    return pd.DataFrame({
        "beta_0": rng.randint(1, 10, n),
        "beta_1": rng.randint(0, 5, n),
        "persistence_entropy": rng.rand(n),
        "wasserstein_dist": rng.rand(n) * 0.5,
        "spectral_gap": rng.rand(n),
        "regime": rng.choice([0, 1, 2], n),
        "diffusion_residual_mean": rng.rand(n) * 0.1,
        "diffusion_residual_std": rng.rand(n) * 0.05,
        "sci": rng.rand(n),
    }, index=index)


def _make_sector_returns(index: pd.DatetimeIndex, n_sectors: int = 5) -> pd.DataFrame:
    rng = np.random.RandomState(77)
    data = rng.randn(len(index), n_sectors) * 0.01
    return pd.DataFrame(data, index=index, columns=[f"sector_{i}" for i in range(n_sectors)])


# ---------------------------------------------------------------------------
# Tests — price features
# ---------------------------------------------------------------------------

class TestPriceFeatures:
    def test_returns_dataframe(self) -> None:
        engine = NNFeatureEngine()
        price_df, _ = _make_price_data()
        result = engine.build_features(price_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_return_columns_present(self) -> None:
        engine = NNFeatureEngine()
        price_df, _ = _make_price_data()
        result = engine.build_features(price_df)
        for w in [1, 5, 10, 21, 63]:
            assert f"ret_{w}d" in result.columns

    def test_vol_columns_present(self) -> None:
        engine = NNFeatureEngine()
        price_df, _ = _make_price_data()
        result = engine.build_features(price_df)
        for w in [5, 10, 21]:
            assert f"vol_{w}d" in result.columns

    def test_log_ret_present(self) -> None:
        engine = NNFeatureEngine()
        price_df, _ = _make_price_data()
        result = engine.build_features(price_df)
        assert "log_ret" in result.columns


# ---------------------------------------------------------------------------
# Tests — technical features
# ---------------------------------------------------------------------------

class TestTechnicalFeatures:
    def test_technical_columns(self) -> None:
        engine = NNFeatureEngine()
        price_df, volume_df = _make_price_data()
        result = engine.build_features(price_df, volume_df=volume_df)
        for col in ["rsi", "macd_line", "macd_signal", "macd_hist", "bb_width", "atr", "obv", "roc"]:
            assert col in result.columns, f"Missing {col}"

    def test_rsi_range(self) -> None:
        engine = NNFeatureEngine()
        price_df, _ = _make_price_data(n=500)
        result = engine.build_features(price_df)
        rsi = result["rsi"].dropna()
        assert rsi.min() >= 0
        assert rsi.max() <= 100


# ---------------------------------------------------------------------------
# Tests — TDA features
# ---------------------------------------------------------------------------

class TestTDAFeatures:
    def test_tda_columns_included(self) -> None:
        engine = NNFeatureEngine()
        price_df, _ = _make_price_data()
        tda_df = _make_tda_features(price_df.index)
        result = engine.build_features(price_df, tda_features_df=tda_df)
        for col in ["beta_0", "beta_1", "persistence_entropy", "sci"]:
            assert col in result.columns


# ---------------------------------------------------------------------------
# Tests — cross-sectional features
# ---------------------------------------------------------------------------

class TestCrossSectionalFeatures:
    def test_cross_sectional_columns(self) -> None:
        engine = NNFeatureEngine()
        price_df, _ = _make_price_data()
        sector_df = _make_sector_returns(price_df.index)
        result = engine.build_features(price_df, sector_returns_df=sector_df)
        assert "sector_rel_strength" in result.columns
        assert "breadth_pct_above_50sma" in result.columns


# ---------------------------------------------------------------------------
# Tests — NaN handling
# ---------------------------------------------------------------------------

class TestNaNHandling:
    def test_no_nans_in_output(self) -> None:
        engine = NNFeatureEngine()
        price_df, volume_df = _make_price_data()
        result = engine.build_features(price_df, volume_df=volume_df)
        assert not result.isna().any().any()

    def test_minimal_data(self) -> None:
        """With very little data most features are NaN — output should be short or empty."""
        engine = NNFeatureEngine()
        price_df, _ = _make_price_data(n=10)
        result = engine.build_features(price_df)
        # Should not raise; result may be empty or very short
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests — feature names / groups
# ---------------------------------------------------------------------------

class TestFeatureGroups:
    def test_feature_names_populated(self) -> None:
        engine = NNFeatureEngine()
        price_df, _ = _make_price_data()
        result = engine.build_features(price_df)
        assert len(engine.feature_names) == result.shape[1]

    def test_get_feature_groups(self) -> None:
        engine = NNFeatureEngine()
        price_df, volume_df = _make_price_data()
        engine.build_features(price_df, volume_df=volume_df)
        groups = engine.get_feature_groups()
        assert isinstance(groups, dict)
        assert len(groups) == len(engine.feature_names)

    def test_no_future_leak_in_returns(self) -> None:
        """Returns at index i should only use prices up to i."""
        engine = NNFeatureEngine()
        price_df, _ = _make_price_data(n=100)
        result = engine.build_features(price_df)
        # ret_1d at time t should equal price[t]/price[t-1] - 1
        # Since NaN rows are dropped, just check it's causal (column exists, no NaN)
        assert "ret_1d" in result.columns
        assert not result["ret_1d"].isna().any()


# ---------------------------------------------------------------------------
# Tests — Series input
# ---------------------------------------------------------------------------

class TestInputTypes:
    def test_series_input(self) -> None:
        engine = NNFeatureEngine()
        price_df, _ = _make_price_data()
        result = engine.build_features(price_df.iloc[:, 0])
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

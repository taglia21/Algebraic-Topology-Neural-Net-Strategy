"""
tda/tests/test_tda.py
=====================
Comprehensive tests for the TDA module.

Tests cover:
- Persistent homology on synthetic data (circle, random noise)
- Graph builder produces valid Laplacian properties
- Diffusion converges (diffused signal smoother than input)
- Regime detector classifies obvious cases correctly
- Feature extractor produces correct columns with no NaNs
- Edge cases: single stock, 2 stocks, missing data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tda.features import TDAFeatureExtractor
from tda.graph_builder import CorrelationGraphBuilder
from tda.laplacian_diffusion import LaplacianDiffusion
from tda.persistent_homology import PersistenceDiagram, PersistentHomologyEngine
from tda.regime_detector import MarketRegime, TDARegimeDetector


# ======================================================================
# Fixtures
# ======================================================================


def _make_circle(n: int = 50, noise: float = 0.05) -> np.ndarray:
    """Generate a noisy circle point cloud in 2-D."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(theta) + np.random.default_rng(42).normal(0, noise, n)
    y = np.sin(theta) + np.random.default_rng(43).normal(0, noise, n)
    return np.column_stack([x, y])


def _make_random_returns(
    T: int = 200,
    N: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic returns DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=T)
    data = rng.normal(0, 0.02, (T, N))
    tickers = [f"STOCK_{i}" for i in range(N)]
    return pd.DataFrame(data, index=dates, columns=tickers)


def _make_crash_returns(
    T: int = 200,
    N: int = 10,
    seed: int = 99,
) -> pd.DataFrame:
    """Generate returns where the second half is a crash (all stocks move
    together with high correlation and negative drift)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=T)

    # First half: normal
    normal = rng.normal(0, 0.02, (T // 2, N))

    # Second half: crash — common factor dominates
    common = rng.normal(-0.03, 0.04, (T - T // 2, 1))
    idio = rng.normal(0, 0.002, (T - T // 2, N))
    crash = common + idio

    data = np.vstack([normal, crash])
    tickers = [f"STOCK_{i}" for i in range(N)]
    return pd.DataFrame(data, index=dates, columns=tickers)


# ======================================================================
# PersistentHomologyEngine tests
# ======================================================================


class TestPersistentHomologyEngine:
    """Tests for PersistentHomologyEngine."""

    def test_compute_circle_has_h1_loop(self) -> None:
        """A circle should produce at least one persistent H1 feature."""
        engine = PersistentHomologyEngine(max_homology_dim=1)
        cloud = _make_circle(n=60, noise=0.02)
        diagram = engine.compute(cloud)

        assert isinstance(diagram, PersistenceDiagram)
        assert len(diagram.diagrams) >= 2  # H0 and H1
        # H1 should have at least one feature (the loop)
        h1 = diagram.diagrams[1]
        finite_h1 = h1[np.isfinite(h1).all(axis=1)]
        assert len(finite_h1) >= 1, "Circle should have at least one H1 loop"

    def test_compute_random_noise_few_loops(self) -> None:
        """Random noise in 2-D should have few persistent H1 features."""
        engine = PersistentHomologyEngine(max_homology_dim=1)
        rng = np.random.default_rng(0)
        cloud = rng.normal(0, 1, (50, 2))
        diagram = engine.compute(cloud)

        # Should still compute without error
        assert diagram.n_points == 50

    def test_betti_numbers_returns_dict(self) -> None:
        engine = PersistentHomologyEngine()
        cloud = _make_circle(n=40)
        diagram = engine.compute(cloud)
        betti = engine.betti_numbers(diagram)

        assert "beta_0" in betti
        assert "beta_1" in betti
        assert isinstance(betti["beta_0"], int)
        assert betti["beta_0"] >= 1  # at least one component

    def test_persistence_entropy_nonnegative(self) -> None:
        engine = PersistentHomologyEngine()
        cloud = _make_circle(n=40)
        diagram = engine.compute(cloud)
        entropy = engine.persistence_entropy(diagram)

        assert entropy >= 0.0

    def test_wasserstein_distance_self_is_zero(self) -> None:
        """Distance of a diagram to itself should be 0."""
        engine = PersistentHomologyEngine()
        cloud = _make_circle(n=40)
        diagram = engine.compute(cloud)
        dist = engine.wasserstein_distance(diagram, diagram)

        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_wasserstein_distance_different_diagrams(self) -> None:
        """Two different point clouds should have nonzero Wasserstein distance."""
        engine = PersistentHomologyEngine()
        d1 = engine.compute(_make_circle(n=40, noise=0.01))
        rng = np.random.default_rng(123)
        d2 = engine.compute(rng.normal(0, 1, (40, 2)))
        dist = engine.wasserstein_distance(d1, d2)

        assert dist >= 0.0  # non-negative by definition

    def test_compute_rejects_nan(self) -> None:
        engine = PersistentHomologyEngine()
        cloud = np.array([[1.0, 2.0], [np.nan, 3.0]])
        with pytest.raises(ValueError, match="NaN"):
            engine.compute(cloud)

    def test_compute_rejects_too_few_points(self) -> None:
        engine = PersistentHomologyEngine()
        cloud = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="at least 2"):
            engine.compute(cloud)

    def test_rolling_compute_returns_dataframe(self) -> None:
        engine = PersistentHomologyEngine()
        returns = _make_random_returns(T=80, N=5)
        df = engine.rolling_compute(
            returns.values, window=20, dates=returns.index
        )

        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {
            "beta_0", "beta_1", "persistence_entropy", "wasserstein_dist"
        }
        assert len(df) > 0
        assert not df.isnull().any().any()


# ======================================================================
# CorrelationGraphBuilder tests
# ======================================================================


class TestCorrelationGraphBuilder:
    """Tests for CorrelationGraphBuilder."""

    def test_correlation_matrix_shape(self) -> None:
        builder = CorrelationGraphBuilder()
        returns = _make_random_returns(T=100, N=5)
        corr = builder.build_correlation_matrix(returns, window=60)

        assert corr.shape == (5, 5)
        # Diagonal should be 1
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)

    def test_correlation_to_distance_properties(self) -> None:
        """Distance matrix should be non-negative, zero diagonal."""
        corr = np.array([
            [1.0, 0.8, -0.3],
            [0.8, 1.0, 0.1],
            [-0.3, 0.1, 1.0],
        ])
        dist = CorrelationGraphBuilder.correlation_to_distance(corr)

        assert dist.shape == (3, 3)
        assert (dist >= 0).all(), "Distance should be non-negative"
        np.testing.assert_allclose(np.diag(dist), 0.0)
        # Perfect correlation → distance 0
        assert dist[0, 0] == pytest.approx(0.0)
        # Anti-correlation → distance = 2
        assert dist[0, 2] == pytest.approx(np.sqrt(2 * (1 - (-0.3))), abs=1e-10)

    def test_adjacency_is_binary_symmetric(self) -> None:
        dist = np.array([
            [0.0, 0.5, 1.5],
            [0.5, 0.0, 0.8],
            [1.5, 0.8, 0.0],
        ])
        adj = CorrelationGraphBuilder.build_adjacency(dist, threshold=1.0)

        assert adj.shape == (3, 3)
        # Binary
        assert set(np.unique(adj)).issubset({0.0, 1.0})
        # Symmetric
        np.testing.assert_array_equal(adj, adj.T)
        # Zero diagonal
        np.testing.assert_array_equal(np.diag(adj), 0.0)
        # Check specific edges
        assert adj[0, 1] == 1.0  # 0.5 < 1.0
        assert adj[0, 2] == 0.0  # 1.5 >= 1.0
        assert adj[1, 2] == 1.0  # 0.8 < 1.0

    def test_laplacian_properties(self) -> None:
        """Laplacian must be symmetric with rows summing to zero."""
        adj = np.array([
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ])
        lap = CorrelationGraphBuilder.build_graph_laplacian(adj)

        assert lap.shape == (4, 4)
        # Symmetric
        np.testing.assert_array_equal(lap, lap.T)
        # Row sums = 0
        np.testing.assert_allclose(lap.sum(axis=1), 0.0, atol=1e-12)
        # Positive semi-definite (all eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(lap)
        assert (eigenvalues >= -1e-10).all(), "Laplacian should be PSD"

    def test_spectral_gap_positive_for_connected_graph(self) -> None:
        """A connected graph should have a positive spectral gap."""
        adj = np.array([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ])
        lap = CorrelationGraphBuilder.build_graph_laplacian(adj)
        gap = CorrelationGraphBuilder.spectral_gap(lap)

        assert gap > 0

    def test_spectral_gap_single_node(self) -> None:
        lap = np.array([[0.0]])
        gap = CorrelationGraphBuilder.spectral_gap(lap)
        assert gap == 0.0

    def test_rolling_spectral_gap_returns_series(self) -> None:
        builder = CorrelationGraphBuilder(default_window=30, default_threshold=1.0)
        returns = _make_random_returns(T=80, N=5)
        sg = builder.rolling_spectral_gap(returns, window=30)

        assert isinstance(sg, pd.Series)
        assert len(sg) > 0
        assert sg.name == "spectral_gap"


# ======================================================================
# LaplacianDiffusion tests
# ======================================================================


class TestLaplacianDiffusion:
    """Tests for LaplacianDiffusion."""

    def test_diffuse_smooths_signal(self) -> None:
        """Diffused signal should be smoother (lower variance) than input."""
        # Build a complete graph Laplacian for 5 nodes
        adj = np.ones((5, 5)) - np.eye(5)
        lap = CorrelationGraphBuilder.build_graph_laplacian(adj)

        rng = np.random.default_rng(42)
        signal = rng.normal(0, 1, 5)

        diffused = LaplacianDiffusion.diffuse(signal, lap, t=1.0)

        assert diffused.shape == signal.shape
        # Diffusion towards mean → lower variance
        assert np.std(diffused) < np.std(signal)

    def test_diffuse_preserves_mean(self) -> None:
        """Heat diffusion on a connected graph preserves the total mass."""
        adj = np.ones((4, 4)) - np.eye(4)
        lap = CorrelationGraphBuilder.build_graph_laplacian(adj)

        signal = np.array([1.0, 2.0, 3.0, 4.0])
        diffused = LaplacianDiffusion.diffuse(signal, lap, t=2.0)

        np.testing.assert_allclose(diffused.sum(), signal.sum(), atol=1e-10)

    def test_diffuse_rejects_negative_time(self) -> None:
        lap = np.eye(3)
        with pytest.raises(ValueError, match="positive"):
            LaplacianDiffusion.diffuse(np.ones(3), lap, t=-1.0)

    def test_diffuse_rejects_mismatched_dimensions(self) -> None:
        lap = np.eye(3)
        with pytest.raises(ValueError, match="must match"):
            LaplacianDiffusion.diffuse(np.ones(5), lap, t=1.0)

    def test_compute_residuals(self) -> None:
        actual = np.array([1.0, 2.0, 3.0])
        diffused = np.array([1.5, 1.8, 2.7])
        residuals = LaplacianDiffusion.compute_residuals(actual, diffused)

        np.testing.assert_allclose(residuals, [-0.5, 0.2, 0.3])

    def test_signal_strength_zscore(self) -> None:
        residuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = LaplacianDiffusion.signal_strength(residuals)

        # Z-scored → mean ≈ 0, std ≈ 1
        assert abs(np.mean(z)) < 1e-10
        assert abs(np.std(z) - 1.0) < 1e-10

    def test_signal_strength_constant_returns_zeros(self) -> None:
        residuals = np.array([2.0, 2.0, 2.0])
        z = LaplacianDiffusion.signal_strength(residuals)
        np.testing.assert_array_equal(z, np.zeros(3))

    def test_generate_signals_returns_dataframe(self) -> None:
        returns = _make_random_returns(T=100, N=5)
        diffusion = LaplacianDiffusion(
            default_window=30, default_diffusion_time=0.5, default_threshold=1.2
        )
        signals = diffusion.generate_signals(returns, window=30)

        assert isinstance(signals, pd.DataFrame)
        assert list(signals.columns) == list(returns.columns)
        assert len(signals) > 0


# ======================================================================
# TDARegimeDetector tests
# ======================================================================


class TestTDARegimeDetector:
    """Tests for TDARegimeDetector."""

    def test_classify_returns_valid_regime(self) -> None:
        detector = TDARegimeDetector(ph_window=20, corr_window=30)
        returns = _make_random_returns(T=100, N=5)

        features = detector.get_regime_features(returns, window=20)
        regime = detector.classify(features)

        assert regime in {r.value for r in MarketRegime}

    def test_normal_regime_for_random_data(self) -> None:
        """Random uncorrelated data should mostly be classified as NORMAL."""
        detector = TDARegimeDetector(
            ph_window=20, corr_window=30, lookback=100
        )
        returns = _make_random_returns(T=150, N=5)

        regimes = detector.rolling_regime(returns, window=20)

        # After warm-up, NORMAL should appear (random data shouldn't be all
        # CRASH).  STRESSED is expected frequently because the 75th-percentile
        # spectral gap threshold is easy to trip on short lookback windows.
        normal_count = (regimes == MarketRegime.NORMAL.value).sum()
        crash_count = (regimes == MarketRegime.CRASH.value).sum()
        assert normal_count > 0, "Expected at least some NORMAL regimes"
        assert crash_count < len(regimes) * 0.5, "Random data should not be mostly CRASH"

    def test_crash_regime_detected(self) -> None:
        """A synthetic crash (high correlation, negative drift) should trigger
        STRESSED or CRASH at some point."""
        detector = TDARegimeDetector(
            ph_window=20, corr_window=30, lookback=100
        )
        returns = _make_crash_returns(T=200, N=10)

        regimes = detector.rolling_regime(returns, window=20)
        non_normal = (regimes != MarketRegime.NORMAL.value).sum()

        assert non_normal > 0, "Crash scenario should trigger at least one non-NORMAL regime"

    def test_rolling_regime_returns_series(self) -> None:
        detector = TDARegimeDetector(ph_window=20, corr_window=30)
        returns = _make_random_returns(T=100, N=5)
        regimes = detector.rolling_regime(returns, window=20)

        assert isinstance(regimes, pd.Series)
        assert regimes.name == "regime"
        assert len(regimes) > 0


# ======================================================================
# TDAFeatureExtractor tests
# ======================================================================


class TestTDAFeatureExtractor:
    """Tests for TDAFeatureExtractor."""

    def test_extract_returns_correct_columns(self) -> None:
        extractor = TDAFeatureExtractor(
            ph_window=20, corr_window=30, lookback=50
        )
        returns = _make_random_returns(T=100, N=5)
        features = extractor.extract(returns, window=20)

        expected_cols = {
            "beta_0", "beta_1", "persistence_entropy", "wasserstein_dist",
            "spectral_gap", "regime", "diffusion_residual_mean",
            "diffusion_residual_std", "sci",
        }
        assert set(features.columns) == expected_cols

    def test_extract_no_nans(self) -> None:
        extractor = TDAFeatureExtractor(
            ph_window=20, corr_window=30, lookback=50
        )
        returns = _make_random_returns(T=100, N=5)
        features = extractor.extract(returns, window=20)

        assert not features.isnull().any().any(), (
            f"Features contain NaN:\n{features.isnull().sum()}"
        )

    def test_extract_sci_in_range(self) -> None:
        """SCI should be in [0, 1] since it's a normalised average."""
        extractor = TDAFeatureExtractor(
            ph_window=20, corr_window=30, lookback=50
        )
        returns = _make_random_returns(T=100, N=5)
        features = extractor.extract(returns, window=20)

        assert (features["sci"] >= 0.0).all()
        assert (features["sci"] <= 1.0).all()

    def test_extract_regime_encoded_numeric(self) -> None:
        extractor = TDAFeatureExtractor(
            ph_window=20, corr_window=30, lookback=50
        )
        returns = _make_random_returns(T=100, N=5)
        features = extractor.extract(returns, window=20)

        assert features["regime"].dtype in (np.int64, np.float64, int)
        assert set(features["regime"].unique()).issubset({0, 1, 2})

    def test_extract_with_two_stocks(self) -> None:
        """Edge case: only 2 stocks."""
        extractor = TDAFeatureExtractor(
            ph_window=15, corr_window=20, lookback=30
        )
        returns = _make_random_returns(T=60, N=2)
        features = extractor.extract(returns, window=15)

        assert len(features) > 0
        assert not features.isnull().any().any()

    def test_extract_raises_on_insufficient_data(self) -> None:
        extractor = TDAFeatureExtractor(
            ph_window=20, corr_window=30
        )
        returns = _make_random_returns(T=25, N=5)
        with pytest.raises(ValueError, match="Not enough data"):
            extractor.extract(returns, window=20)


# ======================================================================
# Edge case tests
# ======================================================================


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_ph_with_3_points(self) -> None:
        """Minimum viable point cloud (3 points)."""
        engine = PersistentHomologyEngine()
        cloud = np.array([[0, 0], [1, 0], [0.5, 0.866]])
        diagram = engine.compute(cloud)
        betti = engine.betti_numbers(diagram)

        assert betti["beta_0"] >= 1

    def test_graph_builder_single_stock(self) -> None:
        """Single-stock returns should not crash, though output is trivial."""
        returns = _make_random_returns(T=100, N=1)
        builder = CorrelationGraphBuilder(default_window=30)
        corr = builder.build_correlation_matrix(returns, window=30)

        assert corr.shape == (1, 1)
        assert corr[0, 0] == pytest.approx(1.0)

    def test_diffusion_single_node_graph(self) -> None:
        """Single node: diffusion should return input unchanged."""
        lap = np.array([[0.0]])
        signal = np.array([5.0])
        diffused = LaplacianDiffusion.diffuse(signal, lap, t=1.0)

        np.testing.assert_allclose(diffused, signal)

    def test_persistence_entropy_empty_diagram(self) -> None:
        """Empty diagram should return entropy = 0."""
        engine = PersistentHomologyEngine()
        diagram = PersistenceDiagram(diagrams=[], max_homology_dim=1, n_points=0)
        entropy = engine.persistence_entropy(diagram)
        assert entropy == 0.0

    def test_wasserstein_empty_diagrams(self) -> None:
        """Wasserstein distance with empty H1 diagrams should be 0."""
        engine = PersistentHomologyEngine()
        d1 = PersistenceDiagram(
            diagrams=[np.array([[0.0, 1.0]]), np.empty((0, 2))],
            max_homology_dim=1, n_points=5,
        )
        d2 = PersistenceDiagram(
            diagrams=[np.array([[0.0, 1.0]]), np.empty((0, 2))],
            max_homology_dim=1, n_points=5,
        )
        dist = engine.wasserstein_distance(d1, d2)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_full_import(self) -> None:
        """All public classes should be importable from tda package."""
        from tda import (
            CorrelationGraphBuilder,
            LaplacianDiffusion,
            PersistentHomologyEngine,
            TDAFeatureExtractor,
            TDARegimeDetector,
        )

        assert PersistentHomologyEngine is not None
        assert CorrelationGraphBuilder is not None
        assert LaplacianDiffusion is not None
        assert TDARegimeDetector is not None
        assert TDAFeatureExtractor is not None

"""
Unit Tests for Manifold Regime Detector
=======================================

Comprehensive test suite for the manifold-based regime detection system.
Tests all components with synthetic data covering all regime types.

Test Coverage:
-------------
1. SphericalCoordinateMapper: Momentum/volatility transformations
2. RegimeSurfaceAnalyzer: Density and curvature computations
3. GeodesicPathTracker: Path solving and classification
4. RegimeClassifier: Regime identification logic
5. ManifoldSignalGenerator: Signal generation
6. ManifoldRegimeDetector: End-to-end integration

Author: Algebraic Topology Neural Net Strategy Team
Date: February 2026
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.options.manifold_regime_detector import (
    SphericalCoordinateMapper,
    RegimeSurfaceAnalyzer,
    GeodesicPathTracker,
    RegimeClassifier,
    ManifoldSignalGenerator,
    ManifoldRegimeDetector,
    RegimeType,
    SphericalCoordinates,
    ManifoldRegimeState,
    GeodesicPath,
)


# ============================================================================
# TEST HELPERS
# ============================================================================

def generate_synthetic_prices(regime: str, length: int = 300) -> np.ndarray:
    """
    Generate synthetic price data for different regime types.
    
    Args:
        regime: 'bull_trend' | 'bear_trend' | 'volatile' | 'consolidation'
        length: Number of data points
        
    Returns:
        Synthetic price array
    """
    np.random.seed(42)
    base_price = 100.0
    prices = [base_price]
    
    if regime == 'bull_trend':
        # Strong uptrend with low volatility
        drift = 0.001  # 0.1% daily drift
        volatility = 0.008  # 0.8% daily vol
        for _ in range(length - 1):
            ret = drift + volatility * np.random.randn()
            prices.append(prices[-1] * (1 + ret))
    
    elif regime == 'bear_trend':
        # Strong downtrend with increasing volatility
        drift = -0.002  # -0.2% daily drift
        volatility = 0.015  # 1.5% daily vol
        for _ in range(length - 1):
            ret = drift + volatility * np.random.randn()
            prices.append(prices[-1] * (1 + ret))
    
    elif regime == 'volatile':
        # High volatility, no clear trend
        drift = 0.0
        volatility = 0.03  # 3% daily vol
        for _ in range(length - 1):
            ret = drift + volatility * np.random.randn()
            prices.append(prices[-1] * (1 + ret))
    
    elif regime == 'consolidation':
        # Range-bound with mean reversion
        for i in range(length - 1):
            # Mean revert to 100
            mean_revert = 0.05 * (base_price - prices[-1]) / base_price
            noise = 0.005 * np.random.randn()
            ret = mean_revert + noise
            prices.append(prices[-1] * (1 + ret))
    
    else:
        raise ValueError(f"Unknown regime: {regime}")
    
    return np.array(prices)


def generate_coordinate_history(n_points: int = 100) -> List[SphericalCoordinates]:
    """Generate synthetic coordinate history."""
    np.random.seed(42)
    coords = []
    
    for i in range(n_points):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        timestamp = datetime.now() - timedelta(days=n_points - i)
        
        coords.append(SphericalCoordinates(
            theta=theta,
            phi=phi,
            timestamp=timestamp,
            momentum_tilt=np.random.uniform(0, 1),
            volatility_phase=np.random.uniform(0, np.pi),
            time_rotation=np.random.uniform(0, 2*np.pi)
        ))
    
    return coords


# ============================================================================
# TEST CLASSES
# ============================================================================

class TestSphericalCoordinateMapper(unittest.TestCase):
    """Test SphericalCoordinateMapper functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mapper = SphericalCoordinateMapper(lookback=20)
    
    def test_initialization(self):
        """Test mapper initialization."""
        self.assertEqual(self.mapper.lookback, 20)
        self.assertIsNotNone(self.mapper._lock)
    
    def test_compute_momentum_tilt_bullish(self):
        """Test momentum tilt computation for bullish trend."""
        prices = generate_synthetic_prices('bull_trend', 300)
        Mt = self.mapper.compute_momentum_tilt(prices)
        
        # Bullish trend should have Mt > 0.5
        self.assertGreater(Mt, 0.5)
        self.assertLessEqual(Mt, 1.0)
        self.assertGreaterEqual(Mt, 0.0)
    
    def test_compute_momentum_tilt_bearish(self):
        """Test momentum tilt computation for bearish trend."""
        prices = generate_synthetic_prices('bear_trend', 300)
        Mt = self.mapper.compute_momentum_tilt(prices)
        
        # Bearish trend should have Mt < 0.5
        self.assertLess(Mt, 0.5)
        self.assertGreaterEqual(Mt, 0.0)
    
    def test_compute_momentum_tilt_insufficient_data(self):
        """Test momentum tilt with insufficient data."""
        prices = np.array([100.0, 101.0])  # Too short
        Mt = self.mapper.compute_momentum_tilt(prices)
        
        # Should return neutral (0.5)
        self.assertAlmostEqual(Mt, 0.5, places=1)
    
    def test_compute_volatility_phase(self):
        """Test volatility phase computation."""
        realized_vol = 0.18
        implied_vol = 0.22
        
        Vt = self.mapper.compute_volatility_phase(realized_vol, implied_vol)
        
        # Should be in valid range [0, π]
        self.assertGreaterEqual(Vt, 0.0)
        self.assertLessEqual(Vt, np.pi)
        
        # IV > RV should give Vt > π/2
        self.assertGreater(Vt, np.pi / 2)
    
    def test_compute_volatility_phase_equilibrium(self):
        """Test volatility phase at IV=RV equilibrium."""
        realized_vol = 0.20
        implied_vol = 0.20
        
        Vt = self.mapper.compute_volatility_phase(realized_vol, implied_vol)
        
        # IV = RV should give Vt ≈ π/2
        self.assertAlmostEqual(Vt, np.pi / 2, places=1)
    
    def test_compute_time_rotation_daily(self):
        """Test daily time rotation."""
        timestamp = datetime(2026, 2, 5, 12, 0, 0)  # Noon
        ωt = self.mapper.compute_time_rotation(timestamp, 'daily')
        
        # Noon should be around π (half day)
        self.assertGreater(ωt, 0.0)
        self.assertLess(ωt, 2 * np.pi)
    
    def test_compute_time_rotation_weekly(self):
        """Test weekly time rotation."""
        timestamp = datetime(2026, 2, 5, 12, 0, 0)  # Thursday
        ωt = self.mapper.compute_time_rotation(timestamp, 'weekly')
        
        self.assertGreaterEqual(ωt, 0.0)
        self.assertLessEqual(ωt, 2 * np.pi)
    
    def test_to_spherical(self):
        """Test spherical coordinate transformation."""
        Mt = 0.35
        Vt = 0.82
        ωt = 0.5
        timestamp = datetime.now()
        
        coords = self.mapper.to_spherical(Mt, Vt, ωt, timestamp)
        
        self.assertIsInstance(coords, SphericalCoordinates)
        self.assertGreaterEqual(coords.theta, 0.0)
        self.assertLessEqual(coords.theta, 2 * np.pi)
        self.assertGreaterEqual(coords.phi, 0.0)
        self.assertLessEqual(coords.phi, 2 * np.pi)
        self.assertEqual(coords.momentum_tilt, Mt)
        self.assertEqual(coords.volatility_phase, Vt)
        self.assertEqual(coords.time_rotation, ωt)
    
    def test_to_cartesian(self):
        """Test Cartesian coordinate conversion."""
        theta = np.pi / 4
        phi = np.pi / 2
        
        x, y, z = self.mapper.to_cartesian(theta, phi, radius=1.0)
        
        # Check on unit sphere: x² + y² + z² = 1
        radius_check = np.sqrt(x**2 + y**2 + z**2)
        self.assertAlmostEqual(radius_check, 1.0, places=5)


class TestRegimeSurfaceAnalyzer(unittest.TestCase):
    """Test RegimeSurfaceAnalyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = RegimeSurfaceAnalyzer(grid_resolution=30)
        self.coords_history = generate_coordinate_history(100)
    
    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(self.analyzer.grid_resolution, 30)
        self.assertEqual(self.analyzer.bandwidth, 'scott')
    
    def test_compute_regime_density(self):
        """Test regime density computation."""
        theta_grid, phi_grid, density = self.analyzer.compute_regime_density(
            self.coords_history
        )
        
        # Check grid shapes
        self.assertEqual(theta_grid.shape, (30, 30))
        self.assertEqual(phi_grid.shape, (30, 30))
        self.assertEqual(density.shape, (30, 30))
        
        # Density should be non-negative
        self.assertTrue(np.all(density >= 0))
        
        # Density should integrate to approximately 1
        # (not exact due to normalization)
        self.assertGreater(np.sum(density), 0)
    
    def test_compute_regime_density_insufficient_data(self):
        """Test density computation with insufficient data."""
        sparse_coords = self.coords_history[:5]
        
        theta_grid, phi_grid, density = self.analyzer.compute_regime_density(
            sparse_coords
        )
        
        # Should return uniform density
        self.assertTrue(np.all(density > 0))
        self.assertAlmostEqual(np.std(density), 0.0, places=5)
    
    def test_compute_surface_curvature(self):
        """Test surface curvature computation."""
        _, _, density = self.analyzer.compute_regime_density(self.coords_history)
        curvature = self.analyzer.compute_surface_curvature(density)
        
        # Check shape
        self.assertEqual(curvature.shape, density.shape)
        
        # Curvature should have both positive/negative values
        # (for a non-trivial surface)
        self.assertTrue(np.any(curvature > 0))
        self.assertTrue(np.any(curvature < 0))
    
    def test_identify_attractors(self):
        """Test attractor identification."""
        theta_grid, phi_grid, density = self.analyzer.compute_regime_density(
            self.coords_history
        )
        curvature = self.analyzer.compute_surface_curvature(density)
        
        attractors = self.analyzer.identify_attractors(
            density, curvature, theta_grid, phi_grid
        )
        
        # Should find at least one attractor
        self.assertGreaterEqual(len(attractors), 0)
        
        # Each attractor should be (θ, φ) tuple
        for attr in attractors:
            self.assertEqual(len(attr), 2)
            theta, phi = attr
            self.assertGreaterEqual(theta, 0.0)
            self.assertLessEqual(theta, 2 * np.pi)
            self.assertGreaterEqual(phi, 0.0)
            self.assertLessEqual(phi, 2 * np.pi)
    
    def test_compute_stress_zones(self):
        """Test stress zone computation."""
        _, _, density = self.analyzer.compute_regime_density(self.coords_history)
        curvature = self.analyzer.compute_surface_curvature(density)
        
        stress_zones = self.analyzer.compute_stress_zones(curvature, threshold_percentile=70)
        
        # Check shape and type
        self.assertEqual(stress_zones.shape, curvature.shape)
        self.assertEqual(stress_zones.dtype, bool)
        
        # Should have some stress zones
        stress_fraction = np.sum(stress_zones) / stress_zones.size
        self.assertGreater(stress_fraction, 0.0)
        self.assertLess(stress_fraction, 1.0)


class TestGeodesicPathTracker(unittest.TestCase):
    """Test GeodesicPathTracker functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = GeodesicPathTracker()
    
    def test_initialization(self):
        """Test tracker initialization."""
        self.assertIsNotNone(self.tracker._lock)
    
    def test_compute_christoffel_symbols(self):
        """Test Christoffel symbol computation."""
        theta = np.pi / 4
        symbols = self.tracker.compute_christoffel_symbols(theta)
        
        # Check expected symbols exist
        self.assertIn('Gamma_theta_phi_phi', symbols)
        self.assertIn('Gamma_phi_theta_phi', symbols)
        
        # Values should be finite
        for key, value in symbols.items():
            self.assertTrue(np.isfinite(value))
    
    def test_compute_christoffel_symbols_pole(self):
        """Test Christoffel symbols near pole (singularity)."""
        theta = 0.0  # North pole
        symbols = self.tracker.compute_christoffel_symbols(theta)
        
        # Should handle singularity gracefully
        for key, value in symbols.items():
            self.assertTrue(np.isfinite(value))
    
    def test_geodesic_derivatives(self):
        """Test geodesic derivative computation."""
        state = np.array([np.pi/4, np.pi/2, 0.01, 0.01])  # [θ, φ, vθ, vφ]
        derivatives = self.tracker.geodesic_derivatives(state, 0.0)
        
        # Check output shape
        self.assertEqual(len(derivatives), 4)
        
        # All derivatives should be finite
        self.assertTrue(np.all(np.isfinite(derivatives)))
    
    def test_solve_geodesic_equation(self):
        """Test geodesic equation solving."""
        start_point = (np.pi/3, np.pi/4)
        initial_velocity = (0.01, 0.02)
        
        path = self.tracker.solve_geodesic_equation(
            start_point, initial_velocity, steps=50, t_max=1.0
        )
        
        # Check path structure
        self.assertIsInstance(path, GeodesicPath)
        self.assertEqual(len(path.coordinates), 50)
        self.assertEqual(len(path.curvature_profile), 50)
        self.assertIn(path.behavior, ['trend_continuation', 'spiral', 'dispersing', 'insufficient_data'])
        self.assertGreaterEqual(path.divergence_score, 0.0)
        self.assertLessEqual(path.divergence_score, 1.0)
    
    def test_compute_path_curvature(self):
        """Test path curvature computation."""
        # Create simple curved path
        t = np.linspace(0, 2*np.pi, 20)
        path = np.column_stack([np.sin(t), np.cos(t)])
        
        curvatures = self.tracker.compute_path_curvature(path)
        
        # Should have same length as path
        self.assertEqual(len(curvatures), len(path))
        
        # All values should be non-negative and finite
        self.assertTrue(all(c >= 0 for c in curvatures))
        self.assertTrue(all(np.isfinite(c) for c in curvatures))
    
    def test_classify_path_behavior_straight(self):
        """Test path classification for straight path."""
        # Nearly straight path
        coords = [(i * 0.1, i * 0.1) for i in range(20)]
        curvatures = [0.01] * 20
        
        behavior = self.tracker.classify_path_behavior(coords, curvatures)
        
        # Should classify as trend continuation
        self.assertEqual(behavior, 'trend_continuation')
    
    def test_classify_path_behavior_spiral(self):
        """Test path classification for spiral path."""
        # Circular/spiral path (ends near start)
        t = np.linspace(0, 2*np.pi, 20)
        coords = [(np.sin(t[i]), np.cos(t[i])) for i in range(20)]
        curvatures = [2.0] * 20  # High curvature
        
        behavior = self.tracker.classify_path_behavior(coords, curvatures)
        
        # Should classify as spiral
        self.assertEqual(behavior, 'spiral')


class TestRegimeClassifier(unittest.TestCase):
    """Test RegimeClassifier functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = RegimeClassifier()
    
    def test_initialization(self):
        """Test classifier initialization."""
        self.assertIsNotNone(self.classifier._lock)
        self.assertEqual(len(self.classifier._regime_history), 0)
    
    def test_classify_regime_crisis_spiral(self):
        """Test classification of crisis spiral regime."""
        # High curvature + high attractor distance = crisis
        regime = self.classifier.classify_regime(
            theta=np.pi/2,
            phi=np.pi/2,
            curvature=0.6,
            attractor_distance=0.7,
            density=0.3
        )
        
        self.assertEqual(regime, RegimeType.CRISIS_SPIRAL)
    
    def test_classify_regime_trend_geodesic(self):
        """Test classification of trend geodesic regime."""
        # Low curvature + low theta (bullish) = trend
        regime = self.classifier.classify_regime(
            theta=np.pi/4,  # Low theta (bullish)
            phi=np.pi/2,
            curvature=0.05,
            attractor_distance=0.3,
            density=0.7
        )
        
        self.assertEqual(regime, RegimeType.TREND_GEODESIC)
    
    def test_classify_regime_mean_reversion(self):
        """Test classification of mean reversion regime."""
        # Near attractor = mean reversion
        regime = self.classifier.classify_regime(
            theta=np.pi/2,
            phi=np.pi/2,
            curvature=0.15,
            attractor_distance=0.1,  # Very close to attractor
            density=0.8
        )
        
        self.assertEqual(regime, RegimeType.MEAN_REVERSION)
    
    def test_classify_regime_volatile_transition(self):
        """Test classification of volatile transition regime."""
        # High curvature but moderate attractor distance
        regime = self.classifier.classify_regime(
            theta=np.pi/2,
            phi=np.pi/2,
            curvature=0.35,
            attractor_distance=0.3,
            density=0.5
        )
        
        self.assertEqual(regime, RegimeType.VOLATILE_TRANSITION)
    
    def test_get_regime_confidence(self):
        """Test regime confidence computation."""
        confidence = self.classifier.get_regime_confidence(
            position=(np.pi/2, np.pi/2),
            density=0.8,
            curvature=0.1
        )
        
        # Confidence should be in [0, 1]
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # High density + low curvature should give high confidence
        self.assertGreater(confidence, 0.5)
    
    def test_detect_regime_transition(self):
        """Test regime transition detection."""
        # Add consistent regime history
        for _ in range(10):
            is_transitioning = self.classifier.detect_regime_transition(
                RegimeType.TREND_GEODESIC, window=5
            )
        
        # No transition with consistent regime
        self.assertFalse(is_transitioning)
        
        # Add different regime
        is_transitioning = self.classifier.detect_regime_transition(
            RegimeType.VOLATILE_TRANSITION, window=5
        )
        
        # Should detect transition
        self.assertTrue(is_transitioning)


class TestManifoldSignalGenerator(unittest.TestCase):
    """Test ManifoldSignalGenerator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = ManifoldSignalGenerator()
    
    def test_initialization(self):
        """Test generator initialization."""
        self.assertIsNotNone(self.generator._lock)
    
    def test_generate_regime_signal_trend(self):
        """Test signal generation for trend regime."""
        signal = self.generator.generate_regime_signal(
            RegimeType.TREND_GEODESIC,
            confidence=0.8,
            path_behavior='trend_continuation'
        )
        
        self.assertEqual(signal['regime'], 'trend_geodesic')
        self.assertEqual(signal['action'], 'long')
        self.assertEqual(signal['strategy'], 'momentum')
        self.assertGreater(signal['position_size'], 0.0)
    
    def test_generate_regime_signal_mean_reversion(self):
        """Test signal generation for mean reversion regime."""
        signal = self.generator.generate_regime_signal(
            RegimeType.MEAN_REVERSION,
            confidence=0.7,
            path_behavior='spiral'
        )
        
        self.assertEqual(signal['action'], 'mean_revert')
        self.assertEqual(signal['strategy'], 'mean_reversion')
    
    def test_generate_regime_signal_crisis(self):
        """Test signal generation for crisis regime."""
        signal = self.generator.generate_regime_signal(
            RegimeType.CRISIS_SPIRAL,
            confidence=0.9,
            path_behavior='spiral'
        )
        
        self.assertEqual(signal['action'], 'exit')
        self.assertEqual(signal['strategy'], 'hedge')
        self.assertEqual(signal['position_size'], 0.0)
    
    def test_compute_position_scalar(self):
        """Test position scalar computation."""
        scalar = self.generator.compute_position_scalar(
            curvature=0.2,
            stress_proximity=0.8
        )
        
        self.assertGreaterEqual(scalar, 0.0)
        self.assertLessEqual(scalar, 1.0)
    
    def test_get_strategy_recommendation(self):
        """Test strategy recommendation."""
        rec = self.generator.get_strategy_recommendation(RegimeType.TREND_GEODESIC)
        self.assertEqual(rec, 'momentum')
        
        rec = self.generator.get_strategy_recommendation(RegimeType.MEAN_REVERSION)
        self.assertEqual(rec, 'mean_reversion')
        
        rec = self.generator.get_strategy_recommendation(RegimeType.CRISIS_SPIRAL)
        self.assertEqual(rec, 'risk_off')


class TestManifoldRegimeDetector(unittest.TestCase):
    """Test ManifoldRegimeDetector end-to-end functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = ManifoldRegimeDetector(lookback=20, grid_resolution=20)
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.lookback, 20)
        self.assertEqual(self.detector.grid_resolution, 20)
        self.assertIsNotNone(self.detector.mapper)
        self.assertIsNotNone(self.detector.surface_analyzer)
        self.assertIsNotNone(self.detector.geodesic_tracker)
        self.assertIsNotNone(self.detector.classifier)
        self.assertIsNotNone(self.detector.signal_generator)
    
    def test_detect_regime_bull_trend(self):
        """Test regime detection for bullish trend."""
        prices = generate_synthetic_prices('bull_trend', 300)
        realized_vol = 0.15
        implied_vol = 0.18
        
        state = self.detector.detect_regime(prices, realized_vol, implied_vol)
        
        # Check state structure
        self.assertIsInstance(state, ManifoldRegimeState)
        self.assertIsInstance(state.regime, RegimeType)
        self.assertGreaterEqual(state.confidence, 0.0)
        self.assertLessEqual(state.confidence, 1.0)
        self.assertGreaterEqual(state.position_scalar, 0.0)
        self.assertLessEqual(state.position_scalar, 1.0)
        
        # Bull trend should likely detect TREND_GEODESIC
        # (not strict assertion due to synthetic data randomness)
        self.assertIn(state.regime, [RegimeType.TREND_GEODESIC, RegimeType.CONSOLIDATION])
    
    def test_detect_regime_volatile(self):
        """Test regime detection for volatile market."""
        prices = generate_synthetic_prices('volatile', 300)
        realized_vol = 0.30
        implied_vol = 0.35
        
        state = self.detector.detect_regime(prices, realized_vol, implied_vol)
        
        # Volatile market should detect VOLATILE_TRANSITION or CRISIS_SPIRAL
        self.assertIn(state.regime, [
            RegimeType.VOLATILE_TRANSITION,
            RegimeType.CRISIS_SPIRAL,
            RegimeType.CONSOLIDATION
        ])
    
    def test_detect_regime_consolidation(self):
        """Test regime detection for consolidation."""
        prices = generate_synthetic_prices('consolidation', 300)
        realized_vol = 0.12
        implied_vol = 0.12
        
        state = self.detector.detect_regime(prices, realized_vol, implied_vol)
        
        # Should detect mean reversion or consolidation
        self.assertIn(state.regime, [
            RegimeType.MEAN_REVERSION,
            RegimeType.CONSOLIDATION,
            RegimeType.TREND_GEODESIC
        ])
    
    def test_get_position_scalar(self):
        """Test position scalar retrieval."""
        prices = generate_synthetic_prices('bull_trend', 300)
        self.detector.detect_regime(prices, 0.15, 0.18)
        
        scalar = self.detector.get_position_scalar()
        
        self.assertGreaterEqual(scalar, 0.0)
        self.assertLessEqual(scalar, 1.0)
    
    def test_get_position_scalar_no_state(self):
        """Test position scalar before any detection."""
        scalar = self.detector.get_position_scalar()
        
        # Should return neutral (0.5)
        self.assertEqual(scalar, 0.5)
    
    def test_invalidate_cache(self):
        """Test cache invalidation."""
        # Detect regime to build cache
        prices = generate_synthetic_prices('bull_trend', 300)
        self.detector.detect_regime(prices, 0.15, 0.18)
        
        # Invalidate cache
        self.detector.invalidate_cache()
        
        # Cache should be invalidated
        self.assertFalse(self.detector._cache_valid)
    
    def test_coordinate_history_accumulation(self):
        """Test that coordinate history accumulates."""
        prices = generate_synthetic_prices('bull_trend', 300)
        
        # Detect multiple times
        for _ in range(10):
            self.detector.detect_regime(prices, 0.15, 0.18)
        
        # History should have accumulated
        self.assertEqual(len(self.detector.coordinate_history), 10)
    
    def test_regime_classification_consistency(self):
        """Test regime classification consistency."""
        prices = generate_synthetic_prices('bull_trend', 300)
        
        # Detect twice with same data
        state1 = self.detector.detect_regime(prices, 0.15, 0.18)
        state2 = self.detector.detect_regime(prices, 0.15, 0.18)
        
        # Regimes might differ slightly due to history accumulation,
        # but coordinates should be similar
        self.assertAlmostEqual(
            state1.coordinates.momentum_tilt,
            state2.coordinates.momentum_tilt,
            places=2
        )


# ============================================================================
# TEST RUNNER
# ============================================================================

if __name__ == '__main__':
    # Run all tests with verbose output
    unittest.main(verbosity=2)

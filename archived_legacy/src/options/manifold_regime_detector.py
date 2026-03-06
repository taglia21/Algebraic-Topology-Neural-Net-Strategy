"""
Continuous Nonlinear Stochastic Dynamical System for Market Regime Detection
============================================================================

A production-grade manifold-based regime detection system that embeds market
momentum and volatility into a spherical coordinate system, computes geodesic
strategy paths, and generates regime-aware trading signals.

Mathematical Foundation:
-----------------------
Market states are mapped to a spherical manifold using:
    θc = (1 - Mt/2) · (π/2) + π/4  (mod 2π)  # Colatitude from momentum
    φc = (Vt - 0.5π) + ωt          (mod 2π)  # Longitude from volatility

Where:
    Mt ∈ [0,1] = normalized momentum tilt factor
    Vt = volatility phase shift from realized/implied ratio
    ωt = time rotation factor for cyclical behavior

Geodesic Dynamics:
-----------------
Paths through regime space follow geodesic equations:
    d²x^k/dt² + Γ^k_ij (dx^i/dt)(dx^j/dt) = 0

For spherical metric ds² = dθ² + sin²(θ)dφ², Christoffel symbols:
    Γ^θ_φφ = -sin(θ)cos(θ)
    Γ^φ_θφ = Γ^φ_φθ = cot(θ)

Surface Curvature:
-----------------
Gaussian curvature K measures regime surface stress:
    K = (f_θθ · f_φφ - f²_θφ) / (1 + f²_θ + f²_φ)²

Regime Types:
------------
- TREND_GEODESIC: Strong momentum, compressed vol, clean trends
- MEAN_REVERSION: Near attractor, low momentum, stable vol
- VOLATILE_TRANSITION: High curvature, regime boundary crossing
- CONSOLIDATION: Low density, weak alignment, ranging
- CRISIS_SPIRAL: Tight spiral paths, extreme curvature, risk-off

Author: Algebraic Topology Neural Net Strategy Team
Date: February 2026
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.integrate import odeint
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import threading
from collections import deque

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND DATA MODELS
# ============================================================================

class RegimeType(Enum):
    """Manifold-based regime classification."""
    TREND_GEODESIC = "trend_geodesic"          # Strong directional, low vol
    MEAN_REVERSION = "mean_reversion"          # Near attractor, stable
    VOLATILE_TRANSITION = "volatile_transition" # High curvature, boundary
    CONSOLIDATION = "consolidation"            # Low density, ranging
    CRISIS_SPIRAL = "crisis_spiral"            # Extreme curvature, risk-off


@dataclass
class SphericalCoordinates:
    """Spherical manifold coordinates."""
    theta: float  # Colatitude [0, 2π]
    phi: float    # Longitude [0, 2π]
    timestamp: datetime
    momentum_tilt: float  # Mt ∈ [0,1]
    volatility_phase: float  # Vt
    time_rotation: float  # ωt


@dataclass
class ManifoldRegimeState:
    """Current regime state from manifold analysis."""
    regime: RegimeType
    confidence: float  # [0, 1]
    position_scalar: float  # Recommended position size multiplier [0, 1]
    attractor_distance: float  # Distance to nearest attractor
    curvature: float  # Gaussian curvature at current position
    path_behavior: str  # 'trend_continuation' | 'spiral' | 'dispersing'
    recommendation: str  # 'momentum' | 'mean_reversion' | 'reduce_risk'
    coordinates: SphericalCoordinates
    metadata: Dict = field(default_factory=dict)


@dataclass
class GeodesicPath:
    """Geodesic path through manifold."""
    coordinates: List[Tuple[float, float]]  # [(θ, φ), ...]
    curvature_profile: List[float]  # Curvature along path
    behavior: str  # Classification of path behavior
    divergence_score: float  # Instability measure


# ============================================================================
# SPHERICAL COORDINATE MAPPER
# ============================================================================

class SphericalCoordinateMapper:
    """
    Transform market observables to spherical manifold coordinates.
    
    Maps momentum and volatility to spherical coordinates (θ, φ) on a
    unit sphere, where θ (colatitude) encodes momentum direction and
    φ (longitude) encodes volatility phase.
    """
    
    def __init__(self, lookback: int = 20):
        """
        Initialize mapper.
        
        Args:
            lookback: Lookback period for momentum/volatility calculations
        """
        self.lookback = lookback
        self._lock = threading.Lock()
        logger.debug(f"SphericalCoordinateMapper initialized with lookback={lookback}")
    
    def compute_momentum_tilt(
        self,
        prices: Union[pd.Series, np.ndarray],
        lookback: Optional[int] = None
    ) -> float:
        """
        Compute momentum tilt factor Mt ∈ [0, 1].
        
        Mt = 0: Strong bearish momentum (θ → π)
        Mt = 1: Strong bullish momentum (θ → 0)
        
        Args:
            prices: Price series (most recent last)
            lookback: Override default lookback period
            
        Returns:
            Momentum tilt Mt ∈ [0, 1]
            
        Formula:
            Mt = sigmoid(normalized_momentum) where
            momentum = (price_now - price_lookback) / ATR
        """
        if lookback is None:
            lookback = self.lookback
            
        prices = np.asarray(prices)
        if len(prices) < lookback + 1:
            logger.warning(f"Insufficient price data: {len(prices)} < {lookback + 1}")
            return 0.5  # Neutral
        
        # Calculate returns for ATR
        returns = np.diff(prices[-lookback-1:]) / prices[-lookback-1:-1]
        atr = np.std(returns) * np.sqrt(252)  # Annualized
        
        if atr < 1e-10:  # Avoid division by zero
            return 0.5
        
        # Normalized momentum
        momentum = (prices[-1] - prices[-lookback-1]) / (prices[-lookback-1] * atr)
        
        # Sigmoid transformation to [0, 1]
        Mt = 1 / (1 + np.exp(-momentum))
        
        logger.debug(f"Momentum tilt Mt={Mt:.4f} (momentum={momentum:.4f}, atr={atr:.4f})")
        return float(np.clip(Mt, 0, 1))
    
    def compute_volatility_phase(
        self,
        realized_vol: float,
        implied_vol: float,
        lookback: Optional[int] = None
    ) -> float:
        """
        Compute volatility phase shift Vt.
        
        Vt encodes the relationship between realized and implied volatility.
        
        Args:
            realized_vol: Recent realized volatility (annualized)
            implied_vol: Current implied volatility (annualized)
            lookback: Override default lookback (unused, for interface consistency)
            
        Returns:
            Volatility phase shift Vt
            
        Formula:
            Vt = arctan(IV/RV - 1) + π/2
            Maps to [0, π] where π/2 is IV=RV equilibrium
        """
        if realized_vol < 1e-10 or implied_vol < 1e-10:
            logger.warning("Near-zero volatility detected")
            return np.pi / 2  # Neutral
        
        # Volatility ratio
        vol_ratio = implied_vol / realized_vol
        
        # Phase encoding: arctan maps (-∞,∞) → (-π/2, π/2)
        # Add π/2 to shift to [0, π]
        Vt = np.arctan(vol_ratio - 1) + np.pi / 2
        
        logger.debug(f"Volatility phase Vt={Vt:.4f} (RV={realized_vol:.4f}, IV={implied_vol:.4f})")
        return float(Vt)
    
    def compute_time_rotation(
        self,
        timestamp: datetime,
        frequency: str = 'daily'
    ) -> float:
        """
        Compute time rotation factor ωt for cyclical market behavior.
        
        Args:
            timestamp: Current timestamp
            frequency: 'daily', 'weekly', 'monthly' rotation period
            
        Returns:
            Time rotation ωt ∈ [0, 2π]
            
        Formula:
            ωt = 2π · (day_of_period / period_length)
        """
        if frequency == 'daily':
            # Daily rotation (intraday cycles)
            seconds_in_day = 24 * 3600
            seconds_now = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
            ωt = 2 * np.pi * (seconds_now / seconds_in_day)
        elif frequency == 'weekly':
            # Weekly rotation (0=Monday, 6=Sunday)
            ωt = 2 * np.pi * (timestamp.weekday() / 7)
        elif frequency == 'monthly':
            # Monthly rotation
            days_in_month = 30  # Approximation
            ωt = 2 * np.pi * (timestamp.day / days_in_month)
        else:
            logger.warning(f"Unknown frequency {frequency}, using daily")
            return self.compute_time_rotation(timestamp, 'daily')
        
        logger.debug(f"Time rotation ωt={ωt:.4f} (freq={frequency})")
        return float(ωt % (2 * np.pi))
    
    def to_spherical(
        self,
        Mt: float,
        Vt: float,
        ωt: float,
        timestamp: Optional[datetime] = None
    ) -> SphericalCoordinates:
        """
        Transform market observables to spherical coordinates.
        
        Args:
            Mt: Momentum tilt ∈ [0, 1]
            Vt: Volatility phase
            ωt: Time rotation
            timestamp: Optional timestamp for record
            
        Returns:
            SphericalCoordinates object
            
        Formula:
            θc = (1 - Mt/2) · (π/2) + π/4  (mod 2π)
            φc = (Vt - 0.5π) + ωt          (mod 2π)
        """
        # Colatitude from momentum (lower θ = bullish)
        theta = ((1 - Mt / 2) * (np.pi / 2) + np.pi / 4) % (2 * np.pi)
        
        # Longitude from volatility phase + time rotation
        phi = (Vt - 0.5 * np.pi + ωt) % (2 * np.pi)
        
        coords = SphericalCoordinates(
            theta=theta,
            phi=phi,
            timestamp=timestamp or datetime.now(),
            momentum_tilt=Mt,
            volatility_phase=Vt,
            time_rotation=ωt
        )
        
        logger.debug(f"Spherical coords: θ={theta:.4f}, φ={phi:.4f}")
        return coords
    
    def to_cartesian(
        self,
        theta: float,
        phi: float,
        radius: float = 1.0
    ) -> Tuple[float, float, float]:
        """
        Convert spherical to Cartesian coordinates for 3D embedding.
        
        Args:
            theta: Colatitude [0, 2π]
            phi: Longitude [0, 2π]
            radius: Sphere radius (default 1.0)
            
        Returns:
            (x, y, z) Cartesian coordinates
            
        Formula:
            x = r · sin(θ) · cos(φ)
            y = r · sin(θ) · sin(φ)
            z = r · cos(θ)
        """
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        
        return (float(x), float(y), float(z))


# ============================================================================
# REGIME SURFACE ANALYZER
# ============================================================================

class RegimeSurfaceAnalyzer:
    """
    Compute regime density heatmap and curvature metrics.
    
    Analyzes historical manifold positions to identify regime attractors,
    compute surface curvature, and detect stress zones.
    """
    
    def __init__(self, grid_resolution: int = 50, bandwidth: str = 'scott'):
        """
        Initialize surface analyzer.
        
        Args:
            grid_resolution: Resolution of density/curvature grids
            bandwidth: KDE bandwidth selection ('scott', 'silverman', or float)
        """
        self.grid_resolution = grid_resolution
        self.bandwidth = bandwidth
        self._lock = threading.Lock()
        logger.debug(f"RegimeSurfaceAnalyzer initialized (res={grid_resolution})")
    
    def compute_regime_density(
        self,
        historical_coords: List[SphericalCoordinates],
        bandwidth: Optional[Union[str, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 2D kernel density estimate over (θ, φ) space.
        
        Args:
            historical_coords: List of historical spherical coordinates
            bandwidth: Override default bandwidth
            
        Returns:
            (theta_grid, phi_grid, density_grid)
            theta_grid, phi_grid: 2D meshgrids
            density_grid: KDE values at each grid point
        """
        if bandwidth is None:
            bandwidth = self.bandwidth
        
        # Extract θ, φ coordinates
        theta_vals = np.array([c.theta for c in historical_coords])
        phi_vals = np.array([c.phi for c in historical_coords])
        
        if len(theta_vals) < 10:
            logger.warning(f"Insufficient historical data: {len(theta_vals)} points")
            # Return uniform density
            theta_grid, phi_grid = np.meshgrid(
                np.linspace(0, 2*np.pi, self.grid_resolution),
                np.linspace(0, 2*np.pi, self.grid_resolution)
            )
            density_grid = np.ones_like(theta_grid) / (4 * np.pi**2)
            return theta_grid, phi_grid, density_grid
        
        # Create grid
        theta_grid, phi_grid = np.meshgrid(
            np.linspace(0, 2*np.pi, self.grid_resolution),
            np.linspace(0, 2*np.pi, self.grid_resolution)
        )
        
        # Stack grid points
        positions = np.vstack([theta_grid.ravel(), phi_grid.ravel()])
        
        # KDE using scipy
        values = np.vstack([theta_vals, phi_vals])
        kernel = stats.gaussian_kde(values, bw_method=bandwidth)
        density = kernel(positions).reshape(theta_grid.shape)
        
        logger.debug(f"Computed density grid from {len(theta_vals)} points")
        return theta_grid, phi_grid, density
    
    def compute_surface_curvature(
        self,
        density_grid: np.ndarray,
        smooth: bool = True
    ) -> np.ndarray:
        """
        Compute Gaussian curvature of regime surface.
        
        Args:
            density_grid: 2D density values
            smooth: Apply Gaussian smoothing before computing derivatives
            
        Returns:
            Curvature grid (same shape as density_grid)
            
        Formula:
            K = (f_θθ · f_φφ - f²_θφ) / (1 + f²_θ + f²_φ)²
            
        Uses finite differences for Hessian approximation.
        """
        f = density_grid.copy()
        
        # Apply smoothing to reduce noise
        if smooth:
            f = gaussian_filter(f, sigma=1.0)
        
        # Compute first derivatives using central differences
        f_theta = np.gradient(f, axis=0)
        f_phi = np.gradient(f, axis=1)
        
        # Compute second derivatives (Hessian)
        f_theta_theta = np.gradient(f_theta, axis=0)
        f_phi_phi = np.gradient(f_phi, axis=1)
        f_theta_phi = np.gradient(f_theta, axis=1)
        
        # Gaussian curvature
        numerator = f_theta_theta * f_phi_phi - f_theta_phi**2
        denominator = (1 + f_theta**2 + f_phi**2)**2
        
        # Avoid division by zero
        curvature = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-10
        )
        
        logger.debug(f"Computed curvature (mean={np.mean(curvature):.6f})")
        return curvature
    
    def identify_attractors(
        self,
        density_grid: np.ndarray,
        curvature: np.ndarray,
        theta_grid: np.ndarray,
        phi_grid: np.ndarray,
        min_density_percentile: float = 70.0
    ) -> List[Tuple[float, float]]:
        """
        Identify regime attractor centers.
        
        Attractors are local maxima in density with negative curvature.
        
        Args:
            density_grid: Density values
            curvature: Curvature values
            theta_grid: θ meshgrid
            phi_grid: φ meshgrid
            min_density_percentile: Minimum density percentile for attractors
            
        Returns:
            List of (θ, φ) attractor centers
        """
        # Threshold for high density regions
        density_threshold = np.percentile(density_grid, min_density_percentile)
        
        # Find local maxima in density
        from scipy.ndimage import maximum_filter
        local_max = (density_grid == maximum_filter(density_grid, size=3))
        
        # Attractors: high density + negative curvature (stable equilibria)
        attractor_mask = local_max & (density_grid > density_threshold) & (curvature < 0)
        
        # Extract coordinates
        attractor_indices = np.argwhere(attractor_mask)
        attractors = [
            (theta_grid[i, j], phi_grid[i, j])
            for i, j in attractor_indices
        ]
        
        logger.info(f"Identified {len(attractors)} regime attractors")
        return attractors
    
    def compute_stress_zones(
        self,
        curvature: np.ndarray,
        threshold_percentile: float = 70.0
    ) -> np.ndarray:
        """
        Identify high-curvature stress zones.
        
        Args:
            curvature: Curvature grid
            threshold_percentile: Percentile threshold for stress classification
            
        Returns:
            Boolean mask of stress zones
        """
        # High absolute curvature indicates stress
        abs_curvature = np.abs(curvature)
        threshold = np.percentile(abs_curvature, threshold_percentile)
        stress_zones = abs_curvature > threshold
        
        logger.debug(f"Identified stress zones (threshold={threshold:.6f})")
        return stress_zones


# ============================================================================
# GEODESIC PATH TRACKER
# ============================================================================

class GeodesicPathTracker:
    """
    Compute and track geodesic paths through regime manifold.
    
    Geodesics represent natural trajectories in regime space, revealing
    market dynamics and regime transitions.
    """
    
    def __init__(self):
        """Initialize geodesic tracker."""
        self._lock = threading.Lock()
        logger.debug("GeodesicPathTracker initialized")
    
    def compute_christoffel_symbols(
        self,
        theta: float
    ) -> Dict[str, float]:
        """
        Compute Christoffel symbols for spherical metric.
        
        For metric ds² = dθ² + sin²(θ)dφ²:
            Γ^θ_φφ = -sin(θ)cos(θ)
            Γ^φ_θφ = Γ^φ_φθ = cot(θ)
            All others = 0
            
        Args:
            theta: Current colatitude
            
        Returns:
            Dictionary of non-zero Christoffel symbols
        """
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Avoid singularities at poles
        if abs(sin_theta) < 1e-10:
            sin_theta = 1e-10 if sin_theta >= 0 else -1e-10
        
        symbols = {
            'Gamma_theta_phi_phi': -sin_theta * cos_theta,
            'Gamma_phi_theta_phi': cos_theta / sin_theta,  # cot(θ)
        }
        
        return symbols
    
    def geodesic_derivatives(
        self,
        state: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        Compute derivatives for geodesic equation.
        
        State vector: [θ, φ, dθ/dt, dφ/dt]
        
        Geodesic equation:
            d²θ/dt² = -Γ^θ_φφ (dφ/dt)²
            d²φ/dt² = -2Γ^φ_θφ (dθ/dt)(dφ/dt)
            
        Args:
            state: [θ, φ, vθ, vφ]
            t: Time parameter (unused, for odeint interface)
            
        Returns:
            [dθ/dt, dφ/dt, d²θ/dt², d²φ/dt²]
        """
        theta, phi, v_theta, v_phi = state
        
        # Compute Christoffel symbols
        symbols = self.compute_christoffel_symbols(theta)
        
        # Geodesic accelerations
        a_theta = -symbols['Gamma_theta_phi_phi'] * v_phi**2
        a_phi = -2 * symbols['Gamma_phi_theta_phi'] * v_theta * v_phi
        
        return np.array([v_theta, v_phi, a_theta, a_phi])
    
    def solve_geodesic_equation(
        self,
        start_point: Tuple[float, float],
        initial_velocity: Tuple[float, float],
        steps: int = 100,
        t_max: float = 1.0
    ) -> GeodesicPath:
        """
        Solve geodesic equation from initial conditions.
        
        Args:
            start_point: (θ₀, φ₀) starting coordinates
            initial_velocity: (vθ₀, vφ₀) initial velocity
            steps: Number of integration steps
            t_max: Maximum integration time
            
        Returns:
            GeodesicPath object with trajectory
        """
        theta0, phi0 = start_point
        v_theta0, v_phi0 = initial_velocity
        
        # Initial state
        state0 = np.array([theta0, phi0, v_theta0, v_phi0])
        
        # Time points
        t = np.linspace(0, t_max, steps)
        
        # Integrate geodesic equation
        solution = odeint(self.geodesic_derivatives, state0, t)
        
        # Extract coordinates
        coordinates = [(solution[i, 0] % (2*np.pi), solution[i, 1] % (2*np.pi))
                      for i in range(steps)]
        
        # Compute path curvature
        curvature_profile = self.compute_path_curvature(solution[:, :2])
        
        # Classify behavior
        behavior = self.classify_path_behavior(coordinates, curvature_profile)
        
        # Compute divergence score
        divergence_score = self.detect_path_divergence_score(coordinates)
        
        path = GeodesicPath(
            coordinates=coordinates,
            curvature_profile=curvature_profile,
            behavior=behavior,
            divergence_score=divergence_score
        )
        
        logger.debug(f"Solved geodesic: {steps} steps, behavior={behavior}")
        return path
    
    def compute_path_curvature(
        self,
        path: np.ndarray
    ) -> List[float]:
        """
        Compute local curvature along path.
        
        Args:
            path: Array of shape (N, 2) with (θ, φ) coordinates
            
        Returns:
            List of curvature values at each point
        """
        if len(path) < 3:
            return [0.0] * len(path)
        
        curvatures = []
        
        for i in range(len(path)):
            if i == 0 or i == len(path) - 1:
                curvatures.append(0.0)
                continue
            
            # Three consecutive points
            p0 = path[i-1]
            p1 = path[i]
            p2 = path[i+1]
            
            # Vectors
            v1 = p1 - p0
            v2 = p2 - p1
            
            # Curvature approximation: change in direction
            cross = v1[0]*v2[1] - v1[1]*v2[0]  # 2D cross product magnitude
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 < 1e-10 or norm2 < 1e-10:
                curvatures.append(0.0)
            else:
                kappa = abs(cross) / (norm1 * norm2)
                curvatures.append(kappa)
        
        return curvatures
    
    def detect_path_divergence_score(
        self,
        coordinates: List[Tuple[float, float]],
        window: int = 10
    ) -> float:
        """
        Compute divergence score for path instability.
        
        Args:
            coordinates: List of (θ, φ) points
            window: Window size for divergence calculation
            
        Returns:
            Divergence score [0, 1]
        """
        if len(coordinates) < window * 2:
            return 0.0
        
        coords_array = np.array(coordinates)
        
        # Compare path spread in first vs last window
        first_window = coords_array[:window]
        last_window = coords_array[-window:]
        
        first_spread = np.std(first_window, axis=0).mean()
        last_spread = np.std(last_window, axis=0).mean()
        
        # Divergence: increase in spread
        if first_spread < 1e-10:
            return 0.0
        
        divergence = (last_spread - first_spread) / (first_spread + 1e-10)
        divergence_score = 1 / (1 + np.exp(-divergence))  # Sigmoid
        
        return float(divergence_score)
    
    def classify_path_behavior(
        self,
        coordinates: List[Tuple[float, float]],
        curvature_profile: List[float]
    ) -> str:
        """
        Classify geodesic path behavior.
        
        Args:
            coordinates: Path coordinates
            curvature_profile: Curvature along path
            
        Returns:
            'trend_continuation' | 'spiral' | 'dispersing'
        """
        if len(coordinates) < 10:
            return 'insufficient_data'
        
        # Mean curvature
        mean_curvature = np.mean(curvature_profile)
        
        # Path length vs displacement
        coords_array = np.array(coordinates)
        path_length = np.sum(np.linalg.norm(np.diff(coords_array, axis=0), axis=1))
        displacement = np.linalg.norm(coords_array[-1] - coords_array[0])
        
        if displacement < 1e-10:
            return 'spiral'
        
        # Efficiency ratio
        efficiency = displacement / (path_length + 1e-10)
        
        if efficiency > 0.7:
            return 'trend_continuation'
        elif mean_curvature > 1.0:
            return 'spiral'
        else:
            return 'dispersing'


# ============================================================================
# REGIME CLASSIFIER
# ============================================================================

class RegimeClassifier:
    """
    Classify current market regime from manifold position.
    
    Uses position, curvature, and attractor proximity to determine regime type.
    """
    
    def __init__(self):
        """Initialize regime classifier."""
        self._lock = threading.Lock()
        self._regime_history: deque = deque(maxlen=50)
        logger.debug("RegimeClassifier initialized")
    
    def classify_regime(
        self,
        theta: float,
        phi: float,
        curvature: float,
        attractor_distance: float,
        density: float = 0.5
    ) -> RegimeType:
        """
        Classify current regime from manifold metrics.
        
        Args:
            theta: Current colatitude
            phi: Current longitude
            curvature: Gaussian curvature at position
            attractor_distance: Distance to nearest attractor
            density: Regime density at position
            
        Returns:
            RegimeType classification
            
        Logic:
            - CRISIS_SPIRAL: High curvature + high attractor distance
            - TREND_GEODESIC: Low curvature + low theta variance
            - MEAN_REVERSION: Near attractor + stable
            - VOLATILE_TRANSITION: High curvature + boundary crossing
            - CONSOLIDATION: Low density + weak alignment
        """
        # Crisis detection: extreme curvature
        if abs(curvature) > 0.5 and attractor_distance > 0.5:
            return RegimeType.CRISIS_SPIRAL
        
        # Trend detection: low curvature, directional momentum
        # Low theta indicates bullish (momentum tilt high)
        if abs(curvature) < 0.1 and (theta < np.pi/3 or theta > 5*np.pi/3):
            return RegimeType.TREND_GEODESIC
        
        # Mean reversion: near attractor
        if attractor_distance < 0.2:
            return RegimeType.MEAN_REVERSION
        
        # Volatile transition: high curvature
        if abs(curvature) > 0.3:
            return RegimeType.VOLATILE_TRANSITION
        
        # Default: consolidation
        return RegimeType.CONSOLIDATION
    
    def get_regime_confidence(
        self,
        position: Tuple[float, float],
        density: float,
        curvature: float
    ) -> float:
        """
        Compute confidence in regime classification.
        
        Args:
            position: (θ, φ) coordinates
            density: Density at position
            curvature: Curvature at position
            
        Returns:
            Confidence [0, 1]
            
        Higher confidence when:
            - High density (well-visited regime)
            - Stable curvature (not transitioning)
            - Consistent with recent history
        """
        # Density component (normalized)
        density_confidence = min(density * 2, 1.0)
        
        # Stability component (low curvature = high stability)
        stability_confidence = 1 / (1 + abs(curvature))
        
        # Combined confidence
        confidence = 0.6 * density_confidence + 0.4 * stability_confidence
        
        return float(np.clip(confidence, 0, 1))
    
    def detect_regime_transition(
        self,
        current_regime: RegimeType,
        window: int = 5
    ) -> bool:
        """
        Detect if regime is transitioning.
        
        Args:
            current_regime: Current regime classification
            window: Lookback window for transition detection
            
        Returns:
            True if regime is transitioning, False otherwise
        """
        self._regime_history.append(current_regime)
        
        if len(self._regime_history) < window:
            return False
        
        # Check consistency over window
        recent = list(self._regime_history)[-window:]
        unique_regimes = set(recent)
        
        # Transition if multiple regimes in window
        is_transitioning = len(unique_regimes) > 1
        
        if is_transitioning:
            logger.info(f"Regime transition detected: {unique_regimes}")
        
        return is_transitioning


# ============================================================================
# MANIFOLD SIGNAL GENERATOR
# ============================================================================

class ManifoldSignalGenerator:
    """
    Generate trading signals from manifold regime analysis.
    
    Translates regime classification into actionable trading recommendations.
    """
    
    def __init__(self):
        """Initialize signal generator."""
        self._lock = threading.Lock()
        logger.debug("ManifoldSignalGenerator initialized")
    
    def generate_regime_signal(
        self,
        regime: RegimeType,
        confidence: float,
        path_behavior: str
    ) -> Dict:
        """
        Generate trading signal from regime.
        
        Args:
            regime: Current regime classification
            confidence: Regime confidence [0, 1]
            path_behavior: Geodesic path behavior
            
        Returns:
            Signal dictionary with action and parameters
        """
        signal = {
            'regime': regime.value,
            'confidence': confidence,
            'path_behavior': path_behavior,
            'action': None,
            'position_size': 0.0,
            'strategy': None,
        }
        
        # TREND_GEODESIC: Full momentum allocation
        if regime == RegimeType.TREND_GEODESIC and confidence > 0.6:
            signal['action'] = 'long' if path_behavior == 'trend_continuation' else 'reduce'
            signal['position_size'] = confidence
            signal['strategy'] = 'momentum'
        
        # MEAN_REVERSION: Contrarian entries
        elif regime == RegimeType.MEAN_REVERSION and confidence > 0.5:
            signal['action'] = 'mean_revert'
            signal['position_size'] = 0.7 * confidence
            signal['strategy'] = 'mean_reversion'
        
        # VOLATILE_TRANSITION: Reduce exposure
        elif regime == RegimeType.VOLATILE_TRANSITION:
            signal['action'] = 'reduce'
            signal['position_size'] = 0.3
            signal['strategy'] = 'defensive'
        
        # CRISIS_SPIRAL: Exit and hedge
        elif regime == RegimeType.CRISIS_SPIRAL:
            signal['action'] = 'exit'
            signal['position_size'] = 0.0
            signal['strategy'] = 'hedge'
        
        # CONSOLIDATION: Wait or small positions
        else:
            signal['action'] = 'wait'
            signal['position_size'] = 0.2
            signal['strategy'] = 'neutral'
        
        logger.debug(f"Generated signal: {signal['action']} (regime={regime.value})")
        return signal
    
    def compute_position_scalar(
        self,
        curvature: float,
        stress_proximity: float
    ) -> float:
        """
        Compute position size scalar from manifold metrics.
        
        Args:
            curvature: Current Gaussian curvature
            stress_proximity: Distance to nearest stress zone [0, 1]
            
        Returns:
            Position scalar [0, 1]
            
        Logic:
            Scale down near stress zones and high curvature areas.
        """
        # Curvature penalty (higher curvature = lower size)
        curvature_factor = 1 / (1 + abs(curvature))
        
        # Stress proximity penalty
        stress_factor = stress_proximity  # Higher proximity = lower size
        
        # Combined scalar
        position_scalar = curvature_factor * stress_factor
        
        return float(np.clip(position_scalar, 0, 1))
    
    def get_strategy_recommendation(
        self,
        regime: RegimeType
    ) -> str:
        """
        Get high-level strategy recommendation for regime.
        
        Args:
            regime: Current regime type
            
        Returns:
            Strategy recommendation string
        """
        recommendations = {
            RegimeType.TREND_GEODESIC: 'momentum',
            RegimeType.MEAN_REVERSION: 'mean_reversion',
            RegimeType.VOLATILE_TRANSITION: 'reduce_risk',
            RegimeType.CONSOLIDATION: 'range_trading',
            RegimeType.CRISIS_SPIRAL: 'risk_off',
        }
        
        return recommendations.get(regime, 'neutral')


# ============================================================================
# MAIN MANIFOLD REGIME DETECTOR
# ============================================================================

class ManifoldRegimeDetector:
    """
    Main orchestrator for manifold-based regime detection.
    
    Coordinates all components to provide real-time regime analysis
    and trading signals based on spherical manifold embedding.
    """
    
    def __init__(
        self,
        lookback: int = 252,
        grid_resolution: int = 50,
        history_size: int = 1000
    ):
        """
        Initialize manifold regime detector.
        
        Args:
            lookback: Lookback period for momentum/volatility
            grid_resolution: Resolution for density/curvature grids
            history_size: Maximum history to maintain
        """
        self.lookback = lookback
        self.grid_resolution = grid_resolution
        
        # Initialize components
        self.mapper = SphericalCoordinateMapper(lookback=lookback)
        self.surface_analyzer = RegimeSurfaceAnalyzer(
            grid_resolution=grid_resolution
        )
        self.geodesic_tracker = GeodesicPathTracker()
        self.classifier = RegimeClassifier()
        self.signal_generator = ManifoldSignalGenerator()
        
        # State
        self.coordinate_history: deque = deque(maxlen=history_size)
        self.current_state: Optional[ManifoldRegimeState] = None
        self._lock = threading.Lock()
        
        # Cached surface analysis
        self._cached_density = None
        self._cached_curvature = None
        self._cached_attractors = None
        self._cache_valid = False
        
        logger.info(f"ManifoldRegimeDetector initialized (lookback={lookback})")
    
    def detect_regime(
        self,
        prices: Union[pd.Series, np.ndarray],
        realized_vol: float,
        implied_vol: float,
        timestamp: Optional[datetime] = None
    ) -> ManifoldRegimeState:
        """
        Detect current market regime from price and volatility data.
        
        Args:
            prices: Price history (most recent last)
            realized_vol: Current realized volatility (annualized)
            implied_vol: Current implied volatility (annualized)
            timestamp: Current timestamp
            
        Returns:
            ManifoldRegimeState with full regime analysis
        """
        with self._lock:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Step 1: Compute spherical coordinates
            Mt = self.mapper.compute_momentum_tilt(prices)
            Vt = self.mapper.compute_volatility_phase(realized_vol, implied_vol)
            ωt = self.mapper.compute_time_rotation(timestamp)
            coords = self.mapper.to_spherical(Mt, Vt, ωt, timestamp)
            
            # Add to history
            self.coordinate_history.append(coords)
            
            # Step 2: Update surface analysis (if enough history)
            if len(self.coordinate_history) >= 20 and not self._cache_valid:
                self._update_surface_cache()
            
            # Step 3: Get current position metrics
            theta_idx = int((coords.theta / (2*np.pi)) * self.grid_resolution)
            phi_idx = int((coords.phi / (2*np.pi)) * self.grid_resolution)
            theta_idx = np.clip(theta_idx, 0, self.grid_resolution - 1)
            phi_idx = np.clip(phi_idx, 0, self.grid_resolution - 1)
            
            if self._cached_density is not None:
                _, _, density_grid = self._cached_density
                curvature = self._cached_curvature[theta_idx, phi_idx]
                density = density_grid[theta_idx, phi_idx]
            else:
                curvature = 0.0
                density = 0.5
            
            # Step 4: Compute attractor distance
            attractor_distance = self._compute_attractor_distance(coords)
            
            # Step 5: Classify regime
            regime = self.classifier.classify_regime(
                coords.theta, coords.phi, curvature, attractor_distance, density
            )
            confidence = self.classifier.get_regime_confidence(
                (coords.theta, coords.phi), density, curvature
            )
            
            # Step 6: Compute geodesic path
            path = self.geodesic_tracker.solve_geodesic_equation(
                (coords.theta, coords.phi),
                (0.01, 0.01),  # Small initial velocity
                steps=50
            )
            
            # Step 7: Generate signal
            position_scalar = self.signal_generator.compute_position_scalar(
                curvature, 1 - attractor_distance
            )
            recommendation = self.signal_generator.get_strategy_recommendation(regime)
            
            # Step 8: Create state
            state = ManifoldRegimeState(
                regime=regime,
                confidence=confidence,
                position_scalar=position_scalar,
                attractor_distance=attractor_distance,
                curvature=curvature,
                path_behavior=path.behavior,
                recommendation=recommendation,
                coordinates=coords,
                metadata={
                    'density': density,
                    'geodesic_divergence': path.divergence_score,
                }
            )
            
            self.current_state = state
            
            logger.info(
                f"Regime detected: {regime.value} "
                f"(confidence={confidence:.2f}, scalar={position_scalar:.2f})"
            )
            
            return state
    
    def _update_surface_cache(self):
        """Update cached surface analysis from coordinate history."""
        coords_list = list(self.coordinate_history)
        
        # Compute density
        self._cached_density = self.surface_analyzer.compute_regime_density(coords_list)
        theta_grid, phi_grid, density_grid = self._cached_density
        
        # Compute curvature
        self._cached_curvature = self.surface_analyzer.compute_surface_curvature(
            density_grid
        )
        
        # Identify attractors
        self._cached_attractors = self.surface_analyzer.identify_attractors(
            density_grid,
            self._cached_curvature,
            theta_grid,
            phi_grid
        )
        
        self._cache_valid = True
        logger.debug("Updated surface cache")
    
    def _compute_attractor_distance(self, coords: SphericalCoordinates) -> float:
        """Compute distance to nearest attractor."""
        if not self._cached_attractors or len(self._cached_attractors) == 0:
            return 1.0  # Maximum distance if no attractors
        
        # Spherical distance to each attractor
        distances = []
        for attr_theta, attr_phi in self._cached_attractors:
            # Great circle distance on sphere
            dtheta = coords.theta - attr_theta
            dphi = coords.phi - attr_phi
            dist = np.sqrt(dtheta**2 + (np.sin(coords.theta) * dphi)**2)
            distances.append(dist)
        
        min_distance = min(distances)
        
        # Normalize to [0, 1] (assuming max distance ~ π)
        normalized_distance = min(min_distance / np.pi, 1.0)
        
        return normalized_distance
    
    def get_position_scalar(self) -> float:
        """
        Get current position sizing scalar.
        
        Returns:
            Position scalar [0, 1]
        """
        if self.current_state is None:
            return 0.5  # Neutral if no state
        
        return self.current_state.position_scalar
    
    def invalidate_cache(self):
        """Invalidate surface cache to force recomputation."""
        with self._lock:
            self._cache_valid = False
            logger.debug("Surface cache invalidated")

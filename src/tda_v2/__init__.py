"""
TDA V2 Module - Enhanced Topological Data Analysis

This module contains:
- PersistentLaplacian: Graph Laplacian + Persistent Laplacian features
- EnhancedTDAFeatures: Combined V1.3 + V2.0 TDA features
"""

try:
    from .persistent_laplacian import (
        PersistentLaplacian,
        EnhancedTDAFeatures,
        compute_rolling_laplacian_features,
        detect_topological_regime_change
    )
except ImportError as e:
    PersistentLaplacian = None
    EnhancedTDAFeatures = None
    compute_rolling_laplacian_features = None
    detect_topological_regime_change = None

__all__ = [
    'PersistentLaplacian',
    'EnhancedTDAFeatures',
    'compute_rolling_laplacian_features',
    'detect_topological_regime_change'
]

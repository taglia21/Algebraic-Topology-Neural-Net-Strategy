"""
Trading Module for V2.0 Enhanced System

This module contains:
- EnsembleRegimeDetector: HMM + GMM + Clustering regime detection
- V2EnhancedEngine: Orchestrator for all V2 components
- RegimeType: Enumeration of market regimes
"""

try:
    from .regime_ensemble import (
        EnsembleRegimeDetector,
        HMMRegimeDetector,
        GMMRegimeDetector,
        ClusterRegimeDetector,
        RegimeType,
        RegimeState,
        create_regime_detector,
        detect_regime_from_returns
    )
except ImportError as e:
    EnsembleRegimeDetector = None
    HMMRegimeDetector = None
    GMMRegimeDetector = None
    ClusterRegimeDetector = None
    RegimeType = None
    RegimeState = None
    create_regime_detector = None
    detect_regime_from_returns = None

try:
    from .v2_enhanced_engine import (
        V2EnhancedEngine,
        V2Config,
        create_v2_engine,
        create_minimal_engine
    )
except ImportError as e:
    V2EnhancedEngine = None
    V2Config = None
    create_v2_engine = None
    create_minimal_engine = None

__all__ = [
    # Regime detection
    'EnsembleRegimeDetector',
    'HMMRegimeDetector',
    'GMMRegimeDetector',
    'ClusterRegimeDetector',
    'RegimeType',
    'RegimeState',
    'create_regime_detector',
    'detect_regime_from_returns',
    # V2 Engine
    'V2EnhancedEngine',
    'V2Config',
    'create_v2_engine',
    'create_minimal_engine'
]

"""
Hierarchical Regime Detection and Control
==========================================

V2.2 Multi-chain regime switching with CUSUM detection.
"""

from .hierarchical_controller import (
    HierarchicalController,
    RegimeState,
    SubPolicy,
    CUSUMDetector,
)

__all__ = [
    'HierarchicalController',
    'RegimeState',
    'SubPolicy',
    'CUSUMDetector',
]

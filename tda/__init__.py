"""
tda/
====
Topological Data Analysis module.

Implements persistent homology, Betti curve computation, graph Laplacian
diffusion, and regime detection via topological features.
"""

from tda.features import TDAFeatureExtractor
from tda.graph_builder import CorrelationGraphBuilder
from tda.laplacian_diffusion import LaplacianDiffusion
from tda.persistent_homology import PersistentHomologyEngine
from tda.regime_detector import TDARegimeDetector

__all__ = [
    "PersistentHomologyEngine",
    "CorrelationGraphBuilder",
    "LaplacianDiffusion",
    "TDARegimeDetector",
    "TDAFeatureExtractor",
]

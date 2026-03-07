"""
ml/
===
Machine-learning pipeline for the ATNN Quant Powerhouse.

Modules
-------
feature_engine  — Technical, fundamental, and cross-sectional feature engineering.
pipeline        — MLPipeline orchestrator: training, validation, drift detection.
validation      — Walk-forward and CPCV validation framework.
models          — Model implementations (LightGBM gradient boost).
"""

from ml.feature_engine import FeatureEngine
from ml.pipeline import MLPipeline

__all__ = ["FeatureEngine", "MLPipeline"]

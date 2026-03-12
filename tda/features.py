"""
tda/features.py
===============
TDA feature extraction for the neural network input pipeline.

Computes a complete feature vector for each trading day by combining outputs
from persistent homology, graph Laplacian analysis, diffusion residuals,
and regime detection.

Output columns per date:
    beta_0, beta_1, persistence_entropy, wasserstein_dist, spectral_gap,
    regime, diffusion_residual_mean, diffusion_residual_std, sci

The **Structural Change Indicator (SCI)** is defined as:

    SCI = (normalised_beta0 + normalised_entropy + normalised_h1_loops) / 3

where normalisation is min-max over the expanding window.  SCI rises when
the topology is stable and drops when the market undergoes structural change.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from tda.graph_builder import CorrelationGraphBuilder
from tda.laplacian_diffusion import LaplacianDiffusion
from tda.persistent_homology import PersistenceDiagram, PersistentHomologyEngine
from tda.regime_detector import MarketRegime, TDARegimeDetector

logger = logging.getLogger(__name__)

# Regime → numeric encoding for the neural network
_REGIME_ENCODING: Dict[str, int] = {
    MarketRegime.NORMAL.value: 0,
    MarketRegime.STRESSED.value: 1,
    MarketRegime.CRASH.value: 2,
}


class TDAFeatureExtractor:
    """Compute all TDA-derived features for the neural network.

    Parameters
    ----------
    ph_window : int
        Rolling window for persistent homology (default 30).
    corr_window : int
        Rolling window for correlation / spectral gap (default 60).
    diffusion_time : float
        Diffusion time parameter for Laplacian diffusion (default 1.0).
    spectral_threshold : float
        Distance threshold for graph adjacency (default 1.0).
    lookback : int
        Calibration lookback for regime detector (default 252).
    """

    def __init__(
        self,
        ph_window: int = 30,
        corr_window: int = 60,
        diffusion_time: float = 1.0,
        spectral_threshold: float = 1.0,
        lookback: int = 252,
    ) -> None:
        self.ph_window = ph_window
        self.corr_window = corr_window
        self.diffusion_time = diffusion_time
        self.spectral_threshold = spectral_threshold

        self.ph_engine = PersistentHomologyEngine()
        self.graph_builder = CorrelationGraphBuilder(
            default_window=corr_window,
            default_threshold=spectral_threshold,
        )
        self.diffusion = LaplacianDiffusion(
            graph_builder=self.graph_builder,
            default_window=corr_window,
            default_diffusion_time=diffusion_time,
            default_threshold=spectral_threshold,
        )
        self.regime_detector = TDARegimeDetector(
            ph_engine=self.ph_engine,
            graph_builder=self.graph_builder,
            lookback=lookback,
            ph_window=ph_window,
            corr_window=corr_window,
            spectral_threshold=spectral_threshold,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _minmax_normalize(values: np.ndarray) -> np.ndarray:
        """Min-max normalise an array to [0, 1]. Returns zeros if constant."""
        vmin = values.min()
        vmax = values.max()
        if vmax - vmin == 0:
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)

    # ------------------------------------------------------------------
    # Main extraction
    # ------------------------------------------------------------------

    def extract(
        self,
        returns: pd.DataFrame,
        window: int = 0,
    ) -> pd.DataFrame:
        """Compute all TDA features for each date in the returns history.

        Parameters
        ----------
        returns : pd.DataFrame
            (T, N) daily returns with DatetimeIndex and ticker columns.
        window : int
            PH rolling window (0 → ``self.ph_window``).

        Returns
        -------
        pd.DataFrame
            One row per date (after warm-up), columns:
            beta_0, beta_1, persistence_entropy, wasserstein_dist,
            spectral_gap, regime, diffusion_residual_mean,
            diffusion_residual_std, sci.
        """
        window = window or self.ph_window
        T, N = returns.shape
        min_rows = max(window, self.corr_window)

        if T < min_rows + 1:
            raise ValueError(
                f"Not enough data: {T} rows; need at least {min_rows + 1}"
            )

        # Reset regime detector for a clean pass
        self.regime_detector._beta0_history = []
        self.regime_detector._entropy_history = []
        self.regime_detector._spectral_gap_history = []

        records = []
        dates = []
        prev_diagram: Optional[PersistenceDiagram] = None

        returns_values = returns.values.astype(np.float64)

        for end in range(min_rows, T):
            date = returns.index[end]

            # --- PH features ---
            cloud = returns_values[end - window + 1 : end + 1]
            std = cloud.std(axis=0)
            std[std == 0] = 1.0
            cloud_z = (cloud - cloud.mean(axis=0)) / std

            if np.isnan(cloud_z).any():
                cloud_z = np.nan_to_num(cloud_z, nan=0.0)

            try:
                diagram = self.ph_engine.compute(cloud_z)
                betti = self.ph_engine.betti_numbers(diagram)
                entropy = self.ph_engine.persistence_entropy(diagram)
                w_dist = (
                    self.ph_engine.wasserstein_distance(prev_diagram, diagram)
                    if prev_diagram is not None
                    else 0.0
                )
                prev_diagram = diagram
            except Exception as exc:
                logger.warning("PH failed at %s: %s", date, exc)
                betti = {"beta_0": 0, "beta_1": 0}
                entropy = 0.0
                w_dist = 0.0

            beta_0 = float(betti.get("beta_0", 0))
            beta_1 = float(betti.get("beta_1", 0))

            # --- Spectral gap ---
            try:
                corr_chunk = returns.iloc[max(0, end - self.corr_window + 1) : end + 1]
                graph_result = self.graph_builder.compute_from_returns(
                    corr_chunk, window=min(len(corr_chunk), self.corr_window)
                )
                spectral_gap = graph_result["spectral_gap"]
            except Exception as exc:
                logger.warning("Spectral gap failed at %s: %s", date, exc)
                spectral_gap = 0.0

            # --- Regime ---
            features_dict = {
                "beta_0": beta_0,
                "beta_1": beta_1,
                "persistence_entropy": entropy,
                "spectral_gap": spectral_gap,
            }
            try:
                regime = self.regime_detector.classify(features_dict)
            except Exception:
                regime = MarketRegime.NORMAL.value

            # --- Diffusion residuals ---
            try:
                current_returns = returns_values[end]
                corr_data = returns.iloc[max(0, end - self.corr_window + 1) : end + 1]
                corr = corr_data.corr().values
                np.fill_diagonal(corr, 1.0)
                corr = np.nan_to_num(corr, nan=0.0)
                dist_mat = CorrelationGraphBuilder.correlation_to_distance(corr)
                adj = CorrelationGraphBuilder.build_adjacency(
                    dist_mat, threshold=self.spectral_threshold
                )
                lap = CorrelationGraphBuilder.build_graph_laplacian(adj)
                diffused = LaplacianDiffusion.diffuse(
                    current_returns, lap, t=self.diffusion_time
                )
                residuals = LaplacianDiffusion.compute_residuals(
                    current_returns, diffused
                )
                diff_mean = float(np.mean(np.abs(residuals)))
                diff_std = float(np.std(residuals))
            except Exception as exc:
                logger.warning("Diffusion failed at %s: %s", date, exc)
                diff_mean = 0.0
                diff_std = 0.0

            records.append({
                "beta_0": beta_0,
                "beta_1": beta_1,
                "persistence_entropy": entropy,
                "wasserstein_dist": w_dist,
                "spectral_gap": spectral_gap,
                "regime": _REGIME_ENCODING.get(regime, 0),
                "diffusion_residual_mean": diff_mean,
                "diffusion_residual_std": diff_std,
            })
            dates.append(date)

        df = pd.DataFrame(records, index=pd.DatetimeIndex(dates))

        # --- Structural Change Indicator (SCI) ---
        # SCI = mean of normalised beta_0, entropy, beta_1 over expanding window
        if len(df) > 0:
            b0_norm = self._minmax_normalize(df["beta_0"].values)
            ent_norm = self._minmax_normalize(df["persistence_entropy"].values)
            h1_norm = self._minmax_normalize(df["beta_1"].values)
            df["sci"] = (b0_norm + ent_norm + h1_norm) / 3.0
        else:
            df["sci"] = pd.Series(dtype=float)

        return df

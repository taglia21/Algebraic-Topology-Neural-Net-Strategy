"""
tda/regime_detector.py
======================
TDA-based market regime classification.

Classifies the market into three regimes using topological features:

- **NORMAL**: High dispersion (beta_0 high), complex topology (entropy high),
  weak coupling (spectral gap low).
- **STRESSED**: Dispersion declining, entropy dropping, coupling rising.
  Transition state that often precedes a crash.
- **CRASH**: Everything collapses into one cluster (beta_0 → 1), topology
  simplifies (entropy minimal), strong herding (spectral gap extreme).

Thresholds are calibrated from a lookback period (default 252 trading days)
so the detector adapts to the evolving statistical properties of the market.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tda.graph_builder import CorrelationGraphBuilder
from tda.persistent_homology import PersistentHomologyEngine

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime labels."""

    NORMAL = "NORMAL"
    STRESSED = "STRESSED"
    CRASH = "CRASH"


class TDARegimeDetector:
    """Classify market regime from topological features.

    Parameters
    ----------
    ph_engine : PersistentHomologyEngine, optional
        Pre-configured PH engine.  Created with defaults if None.
    graph_builder : CorrelationGraphBuilder, optional
        Pre-configured graph builder.  Created with defaults if None.
    lookback : int
        Number of trading days used to calibrate percentile thresholds
        (default 252 = ~1 year).
    ph_window : int
        Rolling window for persistent homology computation.
    corr_window : int
        Rolling window for correlation / spectral gap computation.
    spectral_threshold : float
        Distance threshold for adjacency matrix.
    """

    def __init__(
        self,
        ph_engine: Optional[PersistentHomologyEngine] = None,
        graph_builder: Optional[CorrelationGraphBuilder] = None,
        lookback: int = 252,
        ph_window: int = 30,
        corr_window: int = 60,
        spectral_threshold: float = 1.0,
    ) -> None:
        self.ph_engine = ph_engine or PersistentHomologyEngine()
        self.graph_builder = graph_builder or CorrelationGraphBuilder(
            default_window=corr_window,
            default_threshold=spectral_threshold,
        )
        self.lookback = lookback
        self.ph_window = ph_window
        self.corr_window = corr_window
        self.spectral_threshold = spectral_threshold

        # History buffers for calibration
        self._beta0_history: List[float] = []
        self._entropy_history: List[float] = []
        self._spectral_gap_history: List[float] = []

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def get_regime_features(
        self,
        returns: pd.DataFrame,
        window: int = 0,
    ) -> Dict[str, float]:
        """Compute all regime-relevant TDA features from a returns window.

        Parameters
        ----------
        returns : pd.DataFrame
            (T, N) daily returns. Uses the last *window* rows for PH and
            the last ``corr_window`` rows for the spectral gap.
        window : int
            PH window size (0 → ``self.ph_window``).

        Returns
        -------
        dict
            Keys: beta_0, beta_1, persistence_entropy, spectral_gap.
        """
        window = window or self.ph_window

        # --- Persistent homology features ---
        cloud = returns.iloc[-window:].values.astype(np.float64)

        # Z-score standardise within window
        std = cloud.std(axis=0)
        std[std == 0] = 1.0
        cloud_z = (cloud - cloud.mean(axis=0)) / std

        if np.isnan(cloud_z).any():
            cloud_z = np.nan_to_num(cloud_z, nan=0.0)

        try:
            diagram = self.ph_engine.compute(cloud_z)
            betti = self.ph_engine.betti_numbers(diagram)
            entropy = self.ph_engine.persistence_entropy(diagram)
        except Exception as exc:
            logger.warning("PH computation failed: %s — using fallback values", exc)
            betti = {"beta_0": 0, "beta_1": 0}
            entropy = 0.0

        # --- Spectral gap ---
        corr_data = returns.iloc[-self.corr_window :]
        try:
            result = self.graph_builder.compute_from_returns(
                corr_data, window=self.corr_window
            )
            spectral_gap = result["spectral_gap"]
        except Exception as exc:
            logger.warning("Spectral gap computation failed: %s", exc)
            spectral_gap = 0.0

        return {
            "beta_0": float(betti.get("beta_0", 0)),
            "beta_1": float(betti.get("beta_1", 0)),
            "persistence_entropy": entropy,
            "spectral_gap": spectral_gap,
        }

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, ph_features: Dict[str, float]) -> str:
        """Classify the current market regime from TDA features.

        Uses percentile-based thresholds calibrated from the internal
        history buffers.  If the history is too short (< 30 observations),
        falls back to NORMAL.

        Classification rules:
        - CRASH:    beta_0 <= 10th pctl AND entropy <= 10th pctl
                    AND spectral_gap >= 95th pctl
        - STRESSED: beta_0 <= 25th pctl OR entropy <= 25th pctl
                    OR spectral_gap >= 75th pctl
        - NORMAL:   otherwise

        Parameters
        ----------
        ph_features : dict
            Must contain: beta_0, persistence_entropy, spectral_gap.

        Returns
        -------
        str
            One of ``"NORMAL"``, ``"STRESSED"``, ``"CRASH"``.
        """
        beta0 = ph_features["beta_0"]
        entropy = ph_features["persistence_entropy"]
        sg = ph_features["spectral_gap"]

        # Update history
        self._beta0_history.append(beta0)
        self._entropy_history.append(entropy)
        self._spectral_gap_history.append(sg)

        # Trim to lookback
        self._beta0_history = self._beta0_history[-self.lookback :]
        self._entropy_history = self._entropy_history[-self.lookback :]
        self._spectral_gap_history = self._spectral_gap_history[-self.lookback :]

        # Need minimum history to calibrate
        if len(self._beta0_history) < 30:
            return MarketRegime.NORMAL.value

        b0_arr = np.array(self._beta0_history)
        ent_arr = np.array(self._entropy_history)
        sg_arr = np.array(self._spectral_gap_history)

        # Percentile thresholds
        b0_10 = np.percentile(b0_arr, 10)
        b0_25 = np.percentile(b0_arr, 25)
        ent_10 = np.percentile(ent_arr, 10)
        ent_25 = np.percentile(ent_arr, 25)
        sg_75 = np.percentile(sg_arr, 75)
        sg_95 = np.percentile(sg_arr, 95)

        # CRASH: extreme values on all three indicators
        if beta0 <= b0_10 and entropy <= ent_10 and sg >= sg_95:
            return MarketRegime.CRASH.value

        # STRESSED: at least one indicator is in warning zone
        if beta0 <= b0_25 or entropy <= ent_25 or sg >= sg_75:
            return MarketRegime.STRESSED.value

        return MarketRegime.NORMAL.value

    # ------------------------------------------------------------------
    # Rolling regime series
    # ------------------------------------------------------------------

    def rolling_regime(
        self,
        returns: pd.DataFrame,
        window: int = 0,
    ) -> pd.Series:
        """Compute rolling regime classification over the full returns history.

        Parameters
        ----------
        returns : pd.DataFrame
            (T, N) daily returns.
        window : int
            PH window (0 → default).

        Returns
        -------
        pd.Series
            Regime label for each date (after initial warm-up period).
        """
        window = window or self.ph_window
        T = len(returns)
        min_rows = max(window, self.corr_window)

        if T < min_rows:
            raise ValueError(
                f"Not enough data: {T} rows < minimum required {min_rows}"
            )

        # Reset history for a clean rolling computation
        self._beta0_history = []
        self._entropy_history = []
        self._spectral_gap_history = []

        regimes = []
        dates = []

        for end in range(min_rows, T + 1):
            chunk = returns.iloc[:end]

            try:
                features = self.get_regime_features(chunk, window=window)
                regime = self.classify(features)
            except Exception as exc:
                logger.warning("Regime detection failed at row %d: %s", end, exc)
                regime = MarketRegime.NORMAL.value

            regimes.append(regime)
            dates.append(returns.index[end - 1])

        return pd.Series(regimes, index=pd.Index(dates), name="regime")

"""
tda/persistent_homology.py
==========================
Persistent Homology computation for financial time series.

Takes a point cloud (rolling window of standardized asset returns) and computes
persistent homology using Vietoris-Rips filtration via Ripser.

Key outputs:
- Persistence diagrams (birth-death pairs for H0, H1)
- Betti numbers (beta_0 = connected components, beta_1 = loops)
- Persistence entropy
- Wasserstein distance between consecutive diagrams

Point cloud construction: each row of the window = one day's standardized returns
vector across N assets. So a 30-day window of 50 stocks = 30 points in
50-dimensional space.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PersistenceDiagram:
    """Container for the output of a persistent homology computation.

    Attributes
    ----------
    diagrams : list[np.ndarray]
        diagrams[k] is an (n_k, 2) array of (birth, death) pairs for H_k.
    max_homology_dim : int
        Maximum homology dimension computed.
    n_points : int
        Number of points in the input point cloud.
    """

    diagrams: List[np.ndarray] = field(default_factory=list)
    max_homology_dim: int = 1
    n_points: int = 0


class PersistentHomologyEngine:
    """Compute persistent homology on point clouds built from financial returns.

    Parameters
    ----------
    max_homology_dim : int
        Maximum homology dimension to compute (default 1 → H0, H1).
    max_edge_length : float
        Maximum edge length (filtration value) for the Rips complex.
        ``np.inf`` keeps all edges (default).
    n_threads : int
        Number of threads for Ripser (default 1).
    """

    def __init__(
        self,
        max_homology_dim: int = 1,
        max_edge_length: float = np.inf,
        n_threads: int = 1,
    ) -> None:
        self.max_homology_dim = max_homology_dim
        self.max_edge_length = max_edge_length
        self.n_threads = n_threads

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(self, point_cloud: np.ndarray) -> PersistenceDiagram:
        """Run Vietoris-Rips persistent homology on *point_cloud*.

        Parameters
        ----------
        point_cloud : np.ndarray
            (n_points, n_dims) array. Each row is one observation.

        Returns
        -------
        PersistenceDiagram
            Persistence diagram with birth-death pairs for each dimension.

        Raises
        ------
        ValueError
            If the point cloud has fewer than 2 points or contains NaN.
        """
        from ripser import ripser  # deferred import — heavy C extension

        point_cloud = np.asarray(point_cloud, dtype=np.float64)

        if point_cloud.ndim != 2:
            raise ValueError(
                f"point_cloud must be 2-D; got shape {point_cloud.shape}"
            )
        if point_cloud.shape[0] < 2:
            raise ValueError(
                f"Need at least 2 points; got {point_cloud.shape[0]}"
            )
        if np.isnan(point_cloud).any():
            raise ValueError("point_cloud contains NaN values")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ripser(
                point_cloud,
                maxdim=self.max_homology_dim,
                thresh=self.max_edge_length,
            )

        diagrams = result["dgms"]
        return PersistenceDiagram(
            diagrams=diagrams,
            max_homology_dim=self.max_homology_dim,
            n_points=point_cloud.shape[0],
        )

    # ------------------------------------------------------------------
    # Feature extraction from diagrams
    # ------------------------------------------------------------------

    def betti_numbers(self, diagram: PersistenceDiagram) -> Dict[str, int]:
        """Extract Betti numbers from a persistence diagram.

        Betti numbers are counted at a representative filtration value
        (median of all death values in H0) so that the count is meaningful
        for the typical scale of the data.

        Parameters
        ----------
        diagram : PersistenceDiagram

        Returns
        -------
        dict
            ``{"beta_0": int, "beta_1": int, ...}``
        """
        result: Dict[str, int] = {}
        # Choose a representative threshold: median death time in H0
        if len(diagram.diagrams) > 0 and len(diagram.diagrams[0]) > 0:
            h0 = diagram.diagrams[0]
            finite_deaths = h0[:, 1][np.isfinite(h0[:, 1])]
            threshold = float(np.median(finite_deaths)) if len(finite_deaths) > 0 else 0.0
        else:
            threshold = 0.0

        for dim in range(diagram.max_homology_dim + 1):
            key = f"beta_{dim}"
            if dim >= len(diagram.diagrams) or len(diagram.diagrams[dim]) == 0:
                result[key] = 0
                continue
            pairs = diagram.diagrams[dim]
            # A feature is alive at *threshold* if birth <= threshold < death
            alive = np.sum(
                (pairs[:, 0] <= threshold)
                & ((pairs[:, 1] > threshold) | np.isinf(pairs[:, 1]))
            )
            result[key] = int(alive)

        return result

    def persistence_entropy(self, diagram: PersistenceDiagram) -> float:
        """Compute Shannon entropy of the persistence lifetime distribution.

        Lifetimes are pooled across all dimensions (excluding infinite
        features). The entropy captures how "spread out" the topological
        features are — high entropy = many features with similar lifetimes;
        low entropy = dominated by a few long-lived features.

        Parameters
        ----------
        diagram : PersistenceDiagram

        Returns
        -------
        float
            Persistence entropy in nats. Returns 0.0 if there are no finite
            features.
        """
        lifetimes: List[float] = []
        for dgm in diagram.diagrams:
            if len(dgm) == 0:
                continue
            finite_mask = np.isfinite(dgm[:, 1])
            lt = dgm[finite_mask, 1] - dgm[finite_mask, 0]
            lifetimes.extend(lt[lt > 0].tolist())

        if len(lifetimes) == 0:
            return 0.0

        lifetimes_arr = np.array(lifetimes, dtype=np.float64)
        total = lifetimes_arr.sum()
        if total <= 0:
            return 0.0

        probs = lifetimes_arr / total
        # Shannon entropy: -sum(p * log(p)), skip zeros
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    def wasserstein_distance(
        self,
        diag1: PersistenceDiagram,
        diag2: PersistenceDiagram,
        homology_dim: int = 1,
    ) -> float:
        """Compute the Wasserstein distance between two persistence diagrams.

        Uses persim's implementation of the 1-Wasserstein distance on the
        specified homology dimension.

        Parameters
        ----------
        diag1, diag2 : PersistenceDiagram
        homology_dim : int
            Which homology dimension to compare (default 1 = loops).

        Returns
        -------
        float
            Wasserstein distance. Returns 0.0 if either diagram has no
            features in the requested dimension.
        """
        from persim import wasserstein as persim_wasserstein

        if homology_dim >= len(diag1.diagrams) or homology_dim >= len(diag2.diagrams):
            return 0.0

        d1 = diag1.diagrams[homology_dim]
        d2 = diag2.diagrams[homology_dim]

        # Filter out infinite features (persim can't handle them)
        d1 = d1[np.isfinite(d1).all(axis=1)]
        d2 = d2[np.isfinite(d2).all(axis=1)]

        if len(d1) == 0 and len(d2) == 0:
            return 0.0

        # persim expects at least 1 point in each diagram; if one is empty,
        # add a dummy diagonal point.
        if len(d1) == 0:
            d1 = np.array([[0.0, 0.0]])
        if len(d2) == 0:
            d2 = np.array([[0.0, 0.0]])

        return float(persim_wasserstein(d1, d2))

    # ------------------------------------------------------------------
    # Rolling computation over a returns matrix
    # ------------------------------------------------------------------

    def rolling_compute(
        self,
        returns_matrix: np.ndarray,
        window: int = 30,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """Compute persistent homology over rolling windows of returns.

        Each window of *window* consecutive rows becomes a point cloud.
        Standardisation (z-score) is applied within each window.

        Parameters
        ----------
        returns_matrix : np.ndarray
            (T, N) array of daily returns for N assets over T days.
        window : int
            Rolling window size in days (default 30).
        dates : pd.DatetimeIndex, optional
            Index for the output DataFrame. Must have length T.

        Returns
        -------
        pd.DataFrame
            Columns: beta_0, beta_1, persistence_entropy, wasserstein_dist.
            Index aligned to the *end* of each window.
        """
        returns_matrix = np.asarray(returns_matrix, dtype=np.float64)
        T, N = returns_matrix.shape

        if window < 3:
            raise ValueError(f"window must be >= 3; got {window}")
        if T < window:
            raise ValueError(
                f"Not enough data: {T} rows < window {window}"
            )

        records: List[Dict[str, float]] = []
        idx_list: List[int] = []
        prev_diagram: Optional[PersistenceDiagram] = None

        for end in range(window, T + 1):
            start = end - window
            cloud = returns_matrix[start:end]

            # Skip windows with NaN
            if np.isnan(cloud).any():
                logger.warning("Skipping window [%d:%d] — contains NaN", start, end)
                continue

            # Z-score standardize within the window (per-asset)
            std = cloud.std(axis=0)
            std[std == 0] = 1.0  # avoid division by zero for flat series
            cloud = (cloud - cloud.mean(axis=0)) / std

            try:
                diagram = self.compute(cloud)
            except Exception as exc:
                logger.warning("PH computation failed at window [%d:%d]: %s", start, end, exc)
                continue

            betti = self.betti_numbers(diagram)
            entropy = self.persistence_entropy(diagram)
            w_dist = (
                self.wasserstein_distance(prev_diagram, diagram)
                if prev_diagram is not None
                else 0.0
            )

            records.append({
                "beta_0": betti.get("beta_0", 0),
                "beta_1": betti.get("beta_1", 0),
                "persistence_entropy": entropy,
                "wasserstein_dist": w_dist,
            })
            idx_list.append(end - 1)
            prev_diagram = diagram

        df = pd.DataFrame(records)
        if dates is not None and len(idx_list) > 0:
            df.index = dates[idx_list]
        elif len(idx_list) > 0:
            df.index = pd.Index(idx_list, name="row")

        return df

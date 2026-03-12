"""
tda/graph_builder.py
====================
Build correlation-based graphs from stock return data.

Converts rolling cross-correlation matrices into distance matrices, builds
threshold-based adjacency graphs, and computes the graph Laplacian and its
spectral properties.

The spectral gap of the graph Laplacian is a key regime indicator:
- Small spectral gap → stocks are loosely coupled (normal market)
- Large spectral gap → strong clustering / herding (stressed / crash)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CorrelationGraphBuilder:
    """Build and analyse correlation-based graphs from asset returns.

    Parameters
    ----------
    default_window : int
        Default rolling window for correlation computation (trading days).
    default_threshold : float
        Default distance threshold for building the adjacency matrix.
        Edges with distance < threshold are kept.
    """

    def __init__(
        self,
        default_window: int = 60,
        default_threshold: float = 1.0,
    ) -> None:
        self.default_window = default_window
        self.default_threshold = default_threshold

    # ------------------------------------------------------------------
    # Correlation → Distance → Adjacency
    # ------------------------------------------------------------------

    def build_correlation_matrix(
        self,
        returns: pd.DataFrame,
        window: int = 0,
    ) -> np.ndarray:
        """Compute Pearson correlation matrix from a returns DataFrame.

        Parameters
        ----------
        returns : pd.DataFrame
            (T, N) DataFrame of daily returns.  Uses the last *window* rows.
        window : int
            Number of trailing days to use. 0 → ``self.default_window``.

        Returns
        -------
        np.ndarray
            (N, N) correlation matrix.  NaN entries (e.g. from constant
            columns) are replaced with 0.
        """
        window = window or self.default_window
        data = returns.iloc[-window:]
        corr = data.corr().values
        np.fill_diagonal(corr, 1.0)
        corr = np.nan_to_num(corr, nan=0.0)
        return corr

    @staticmethod
    def correlation_to_distance(corr_matrix: np.ndarray) -> np.ndarray:
        """Convert a correlation matrix to a distance matrix.

        Uses the standard metric: d_ij = sqrt(2 * (1 - rho_ij)).

        Parameters
        ----------
        corr_matrix : np.ndarray
            (N, N) correlation matrix with values in [-1, 1].

        Returns
        -------
        np.ndarray
            (N, N) non-negative distance matrix.
        """
        # Clamp to [-1, 1] for numerical safety
        rho = np.clip(corr_matrix, -1.0, 1.0)
        dist = np.sqrt(2.0 * (1.0 - rho))
        np.fill_diagonal(dist, 0.0)
        return dist

    @staticmethod
    def build_adjacency(
        distance_matrix: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Build a binary adjacency matrix by thresholding distances.

        An edge exists between nodes i and j if ``distance_matrix[i, j] < threshold``
        and ``i != j``.

        Parameters
        ----------
        distance_matrix : np.ndarray
            (N, N) distance matrix.
        threshold : float
            Maximum distance for an edge.

        Returns
        -------
        np.ndarray
            (N, N) binary adjacency matrix (symmetric, zero diagonal).
        """
        adj = (distance_matrix < threshold).astype(np.float64)
        np.fill_diagonal(adj, 0.0)
        return adj

    @staticmethod
    def build_graph_laplacian(adjacency: np.ndarray) -> np.ndarray:
        """Compute the combinatorial graph Laplacian L = D - A.

        Parameters
        ----------
        adjacency : np.ndarray
            (N, N) adjacency matrix (assumed symmetric).

        Returns
        -------
        np.ndarray
            (N, N) graph Laplacian.  Symmetric, positive semi-definite,
            rows and columns sum to zero.
        """
        degree = np.diag(adjacency.sum(axis=1))
        return degree - adjacency

    @staticmethod
    def spectral_gap(laplacian: np.ndarray) -> float:
        """Compute the maximum spectral gap of the Laplacian eigenvalues.

        The spectral gap is the largest consecutive difference between the
        sorted eigenvalues of L.  A large spectral gap indicates strong
        cluster separation in the graph (i.e. market herding).

        Parameters
        ----------
        laplacian : np.ndarray
            (N, N) graph Laplacian.

        Returns
        -------
        float
            Maximum spectral gap.  Returns 0.0 for degenerate cases.
        """
        if laplacian.shape[0] < 2:
            return 0.0

        eigenvalues = np.linalg.eigvalsh(laplacian)
        eigenvalues = np.sort(np.real(eigenvalues))

        if len(eigenvalues) < 2:
            return 0.0

        gaps = np.diff(eigenvalues)
        return float(np.max(gaps))

    # ------------------------------------------------------------------
    # Full pipeline: returns → spectral gap
    # ------------------------------------------------------------------

    def compute_from_returns(
        self,
        returns: pd.DataFrame,
        window: int = 0,
        threshold: Optional[float] = None,
    ) -> dict:
        """Full pipeline: compute correlation, distance, adjacency, Laplacian,
        and spectral gap from a returns DataFrame.

        Parameters
        ----------
        returns : pd.DataFrame
            (T, N) daily returns.
        window : int
            Correlation window (0 → default).
        threshold : float, optional
            Distance threshold (None → default).

        Returns
        -------
        dict
            Keys: corr_matrix, distance_matrix, adjacency, laplacian,
            spectral_gap.
        """
        window = window or self.default_window
        threshold = threshold if threshold is not None else self.default_threshold

        corr = self.build_correlation_matrix(returns, window=window)
        dist = self.correlation_to_distance(corr)
        adj = self.build_adjacency(dist, threshold=threshold)
        lap = self.build_graph_laplacian(adj)
        sg = self.spectral_gap(lap)

        return {
            "corr_matrix": corr,
            "distance_matrix": dist,
            "adjacency": adj,
            "laplacian": lap,
            "spectral_gap": sg,
        }

    def rolling_spectral_gap(
        self,
        returns: pd.DataFrame,
        window: int = 0,
        threshold: Optional[float] = None,
    ) -> pd.Series:
        """Compute the spectral gap over a rolling window.

        Parameters
        ----------
        returns : pd.DataFrame
            (T, N) daily returns with a DatetimeIndex.
        window : int
            Correlation window (0 → default).
        threshold : float, optional
            Distance threshold (None → default).

        Returns
        -------
        pd.Series
            Spectral gap time series, indexed by date.
        """
        window = window or self.default_window
        threshold = threshold if threshold is not None else self.default_threshold

        T = len(returns)
        if T < window:
            raise ValueError(
                f"Not enough data: {T} rows < window {window}"
            )

        gaps = []
        dates = []

        for end in range(window, T + 1):
            chunk = returns.iloc[end - window : end]

            if chunk.isnull().any().any():
                logger.warning(
                    "Skipping window ending at %s — contains NaN",
                    returns.index[end - 1],
                )
                continue

            corr = chunk.corr().values
            np.fill_diagonal(corr, 1.0)
            corr = np.nan_to_num(corr, nan=0.0)

            dist = self.correlation_to_distance(corr)
            adj = self.build_adjacency(dist, threshold=threshold)
            lap = self.build_graph_laplacian(adj)
            sg = self.spectral_gap(lap)

            gaps.append(sg)
            dates.append(returns.index[end - 1])

        return pd.Series(gaps, index=pd.Index(dates), name="spectral_gap")

"""
tda/laplacian_diffusion.py
==========================
Graph Laplacian diffusion for arbitrage signal generation.

Core idea: diffuse recent returns over the correlation graph so that
topologically close stocks converge toward a consensus value.  Stocks
whose *actual* returns deviate from the diffused consensus are considered
mispriced:

- Negative residual → stock under-performed its neighbourhood → **buy**
- Positive residual → stock over-performed its neighbourhood → **sell**

The diffusion is governed by the heat equation on graphs:

    x(t) = exp(-t L) x(0)

where L is the graph Laplacian and t is the diffusion time parameter.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.linalg import expm

from tda.graph_builder import CorrelationGraphBuilder

logger = logging.getLogger(__name__)


class LaplacianDiffusion:
    """Generate mean-reversion signals via heat diffusion on a correlation graph.

    Parameters
    ----------
    graph_builder : CorrelationGraphBuilder, optional
        Pre-configured graph builder.  Created with defaults if not provided.
    default_window : int
        Default rolling window for correlation computation.
    default_diffusion_time : float
        Default diffusion time parameter *t*.  Larger values produce
        smoother (more consensus-like) output.
    default_threshold : float
        Distance threshold for adjacency construction.
    """

    def __init__(
        self,
        graph_builder: Optional[CorrelationGraphBuilder] = None,
        default_window: int = 60,
        default_diffusion_time: float = 1.0,
        default_threshold: float = 1.0,
    ) -> None:
        self.graph_builder = graph_builder or CorrelationGraphBuilder(
            default_window=default_window,
            default_threshold=default_threshold,
        )
        self.default_window = default_window
        self.default_diffusion_time = default_diffusion_time
        self.default_threshold = default_threshold

    # ------------------------------------------------------------------
    # Core diffusion
    # ------------------------------------------------------------------

    @staticmethod
    def diffuse(
        signal: np.ndarray,
        laplacian: np.ndarray,
        t: float = 1.0,
    ) -> np.ndarray:
        """Diffuse *signal* over the graph defined by *laplacian*.

        Computes  x(t) = exp(-t L) @ x(0).

        Parameters
        ----------
        signal : np.ndarray
            (N,) vector of initial values (e.g. today's returns per asset).
        laplacian : np.ndarray
            (N, N) graph Laplacian.
        t : float
            Diffusion time (> 0).

        Returns
        -------
        np.ndarray
            (N,) diffused signal.
        """
        if t <= 0:
            raise ValueError(f"Diffusion time must be positive; got {t}")

        signal = np.asarray(signal, dtype=np.float64).ravel()
        N = laplacian.shape[0]
        if signal.shape[0] != N:
            raise ValueError(
                f"Signal length ({signal.shape[0]}) must match "
                f"Laplacian size ({N})"
            )

        # exp(-t L) via scipy matrix exponential (Padé approximation)
        heat_kernel = expm(-t * laplacian)
        return heat_kernel @ signal

    @staticmethod
    def compute_residuals(
        actual_returns: np.ndarray,
        diffused_returns: np.ndarray,
    ) -> np.ndarray:
        """Compute the deviation of actual returns from topological consensus.

        Parameters
        ----------
        actual_returns : np.ndarray
            (N,) observed returns.
        diffused_returns : np.ndarray
            (N,) consensus returns after diffusion.

        Returns
        -------
        np.ndarray
            (N,) residuals = actual - diffused.  Positive means the stock
            out-performed its topological neighbourhood.
        """
        return np.asarray(actual_returns, dtype=np.float64) - np.asarray(
            diffused_returns, dtype=np.float64
        )

    @staticmethod
    def signal_strength(residuals: np.ndarray) -> np.ndarray:
        """Normalise residuals to z-scores for cross-sectional comparability.

        Parameters
        ----------
        residuals : np.ndarray
            (N,) raw residuals.

        Returns
        -------
        np.ndarray
            (N,) z-scored residuals.  Returns zeros if std is zero.
        """
        residuals = np.asarray(residuals, dtype=np.float64)
        std = residuals.std()
        if std == 0:
            return np.zeros_like(residuals)
        return (residuals - residuals.mean()) / std

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        returns: pd.DataFrame,
        window: int = 0,
        diffusion_time: float = 0.0,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """End-to-end signal generation: build graph → diffuse → rank.

        For each day (after the initial window), builds a correlation graph
        from the trailing window, diffuses the current day's returns, and
        computes the mispricing residual for each stock.

        Parameters
        ----------
        returns : pd.DataFrame
            (T, N) daily returns with DatetimeIndex and ticker columns.
        window : int
            Correlation window (0 → default).
        diffusion_time : float
            Diffusion parameter *t* (0 → default).
        threshold : float, optional
            Adjacency distance threshold (None → default).

        Returns
        -------
        pd.DataFrame
            (T - window, N) DataFrame of z-scored residual signals.
            Columns match the input tickers.  Negative = buy, positive = sell.
        """
        window = window or self.default_window
        diffusion_time = diffusion_time or self.default_diffusion_time
        threshold = threshold if threshold is not None else self.default_threshold

        T = len(returns)
        if T <= window:
            raise ValueError(
                f"Not enough data: {T} rows <= window {window}"
            )

        tickers = returns.columns.tolist()
        signal_records = []
        signal_dates = []

        for end in range(window, T):
            # Trailing window for graph construction
            window_data = returns.iloc[end - window : end]
            # Current day's returns (the signal we'll diffuse)
            current_returns = returns.iloc[end].values.astype(np.float64)

            # Skip if any NaN in the window or current day
            if window_data.isnull().any().any() or np.isnan(current_returns).any():
                logger.warning(
                    "Skipping date %s — NaN in window or current returns",
                    returns.index[end],
                )
                continue

            try:
                # Build graph
                corr = window_data.corr().values
                np.fill_diagonal(corr, 1.0)
                corr = np.nan_to_num(corr, nan=0.0)
                dist = CorrelationGraphBuilder.correlation_to_distance(corr)
                adj = CorrelationGraphBuilder.build_adjacency(dist, threshold=threshold)
                lap = CorrelationGraphBuilder.build_graph_laplacian(adj)

                # Diffuse
                diffused = self.diffuse(current_returns, lap, t=diffusion_time)

                # Residuals and z-score
                residuals = self.compute_residuals(current_returns, diffused)
                z_scores = self.signal_strength(residuals)

                signal_records.append(z_scores)
                signal_dates.append(returns.index[end])

            except Exception as exc:
                logger.warning(
                    "Signal generation failed at %s: %s",
                    returns.index[end],
                    exc,
                )
                continue

        if len(signal_records) == 0:
            return pd.DataFrame(columns=tickers)

        return pd.DataFrame(
            signal_records,
            index=pd.DatetimeIndex(signal_dates),
            columns=tickers,
        )

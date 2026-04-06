"""
tda/extractor.py
=================
Real Topological Data Analysis feature extraction using persistent homology.

This is what was MISSING from the original system. The nn/features.py had
a stub that expected pre-computed TDA features — this module computes them.

Methodology (following Gidea & Katz 2018, "Topological Data Analysis of
Financial Time Series"):
  1. Build a delay-embedding point cloud from a rolling price window
  2. Compute Vietoris-Rips persistent homology (H0 and H1)
  3. Extract Betti numbers, persistence entropy, Wasserstein distance
  4. Compute spectral gap from correlation graph Laplacian

Features extracted per window:
  - beta_0: number of connected components (drops in trends)
  - beta_1: number of loops (rises in mean-reverting regimes)
  - persistence_entropy: complexity of topology (Shannon entropy of lifetimes)
  - wasserstein_dist: topology change vs prior window (regime shift detector)
  - max_persistence_h0: longest-lived H0 feature (dominant cluster scale)
  - max_persistence_h1: longest-lived H1 feature (dominant cycle scale)
  - spectral_gap: smallest non-zero eigenvalue of correlation Laplacian
  - sci: Spread Complexity Index — variance of persistence lifetimes
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd
from ripser import ripser
from persim import wasserstein

logger = logging.getLogger(__name__)


def _delay_embedding(
    prices: np.ndarray,
    dim: int = 3,
    delay: int = 1,
) -> np.ndarray:
    """
    Build a delay-embedding point cloud from a 1D time series.

    For prices [p1, p2, ..., pN], with dim=3, delay=1:
    Returns point cloud where each row is [p_i, p_{i+1}, p_{i+2}].

    This transforms the time series into a geometric object whose
    topology reflects market structure.
    """
    # Ensure strictly 1D float64
    prices = np.array(prices, dtype=np.float64).ravel()
    n = len(prices)
    max_start = n - (dim - 1) * delay
    if max_start <= 0:
        raise ValueError(f"Not enough data: need {(dim-1)*delay + 1} points, got {n}")

    # Efficient column-stack implementation
    cloud = np.column_stack([
        prices[d * delay: max_start + d * delay]
        for d in range(dim)
    ])

    # Normalize to unit cube
    cloud_range = cloud.max(axis=0) - cloud.min(axis=0)
    cloud_range[cloud_range == 0] = 1.0
    cloud = (cloud - cloud.min(axis=0)) / cloud_range

    return cloud


def _persistence_entropy(diagram: np.ndarray) -> float:
    """
    Shannon entropy of persistence lifetimes.
    High entropy = complex, disordered topology.
    Low entropy = simple, structured topology.
    """
    if len(diagram) == 0:
        return 0.0

    lifetimes = diagram[:, 1] - diagram[:, 0]
    # Remove infinite lifetimes
    finite = lifetimes[np.isfinite(lifetimes)]
    if len(finite) == 0:
        return 0.0

    total = finite.sum()
    if total <= 0:
        return 0.0

    probs = finite / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _betti_number(diagram: np.ndarray, epsilon: float = 0.1) -> int:
    """Count features alive at threshold epsilon."""
    if len(diagram) == 0:
        return 0
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    # Replace inf with large number
    deaths = np.where(np.isinf(deaths), 1e10, deaths)
    return int(np.sum((births <= epsilon) & (deaths > epsilon)))


def _spectral_gap(returns: np.ndarray, n_assets: int = 5) -> float:
    """
    Compute spectral gap of the correlation matrix graph Laplacian.

    Small gap → assets highly correlated → trending market.
    Large gap → assets decorrelated → mean-reverting market.

    Uses rolling windows of single-asset returns reshaped as pseudo-multi-asset.
    """
    if len(returns) < n_assets + 5:
        return 0.5

    # Reshape single-asset returns into pseudo-multivariate by lagging
    lag_returns = np.column_stack([
        returns[i:len(returns) - n_assets + i + 1]
        for i in range(n_assets)
    ])

    if lag_returns.shape[0] < 5:
        return 0.5

    corr = np.corrcoef(lag_returns.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    # Graph Laplacian
    degree = corr.sum(axis=1)
    laplacian = np.diag(degree) - corr

    try:
        eigvals = np.linalg.eigvalsh(laplacian)
        eigvals_sorted = np.sort(np.abs(eigvals))
        # Second-smallest eigenvalue (Fiedler value)
        gap = float(eigvals_sorted[1]) if len(eigvals_sorted) > 1 else 0.5
    except Exception:
        gap = 0.5

    return min(gap, 5.0)  # cap for normalization


class TDAFeatureExtractor:
    """
    Compute TDA features from a rolling window of price data.

    Parameters
    ----------
    window : int
        Number of bars per rolling window (default 60).
    delay : int
        Delay embedding step size (default 1).
    dim : int
        Embedding dimension (default 3).
    stride : int
        How many bars between consecutive extractions (default 1).
    maxdim : int
        Maximum homology dimension to compute (default 1 → H0 and H1).
    """

    def __init__(
        self,
        window: int = 60,
        delay: int = 1,
        dim: int = 3,
        stride: int = 1,
        maxdim: int = 1,
    ):
        self.window = window
        self.delay = delay
        self.dim = dim
        self.stride = stride
        self.maxdim = maxdim
        self._prev_h1_diagram: Optional[np.ndarray] = None

    def extract_window(self, prices: np.ndarray) -> dict[str, float]:
        """
        Extract TDA features from a single window of prices.

        Parameters
        ----------
        prices : np.ndarray
            1D array of prices (length = self.window).

        Returns
        -------
        dict with keys: beta_0, beta_1, persistence_entropy, wasserstein_dist,
                        max_persistence_h0, max_persistence_h1, spectral_gap, sci
        """
        if len(prices) < self.window // 2:
            return self._zero_features()

        # Normalize
        p = prices.astype(np.float64)
        if p.std() < 1e-10:
            return self._zero_features()

        # Build point cloud
        try:
            cloud = _delay_embedding(p, dim=self.dim, delay=self.delay)
        except ValueError:
            return self._zero_features()

        # Compute persistent homology
        try:
            result = ripser(cloud, maxdim=self.maxdim, thresh=2.0)
            dgms = result["dgms"]
        except Exception as e:
            logger.debug("ripser failed: %s", e)
            return self._zero_features()

        h0_diagram = dgms[0] if len(dgms) > 0 else np.empty((0, 2))
        h1_diagram = dgms[1] if len(dgms) > 1 else np.empty((0, 2))

        # --- H0 features ---
        # Remove the infinite bar (the one connected component that never dies)
        h0_finite = h0_diagram[np.isfinite(h0_diagram[:, 1])]

        beta_0 = _betti_number(h0_diagram)
        max_h0 = float(np.max(h0_finite[:, 1] - h0_finite[:, 0])) if len(h0_finite) > 0 else 0.0
        entropy_h0 = _persistence_entropy(h0_finite)

        # --- H1 features ---
        h1_finite = h1_diagram[np.isfinite(h1_diagram[:, 1])]

        beta_1 = _betti_number(h1_diagram)
        max_h1 = float(np.max(h1_finite[:, 1] - h1_finite[:, 0])) if len(h1_finite) > 0 else 0.0
        entropy_h1 = _persistence_entropy(h1_finite)

        # Combined persistence entropy
        all_finite = np.vstack([h0_finite, h1_finite]) if len(h1_finite) > 0 else h0_finite
        pers_entropy = _persistence_entropy(all_finite)

        # --- Wasserstein distance to previous window (regime change detector) ---
        w_dist = 0.0
        if self._prev_h1_diagram is not None and len(h1_finite) > 0:
            try:
                prev_finite = self._prev_h1_diagram[np.isfinite(self._prev_h1_diagram[:, 1])]
                if len(prev_finite) > 0:
                    w_dist = float(wasserstein(h1_finite, prev_finite))
            except Exception:
                w_dist = 0.0

        self._prev_h1_diagram = h1_diagram.copy()

        # --- Spectral gap ---
        log_returns = np.diff(np.log(p + 1e-10))
        spec_gap = _spectral_gap(log_returns)

        # --- Spread Complexity Index ---
        all_lifetimes = []
        for finite_diag in [h0_finite, h1_finite]:
            if len(finite_diag) > 0:
                all_lifetimes.extend((finite_diag[:, 1] - finite_diag[:, 0]).tolist())

        sci = float(np.var(all_lifetimes)) if len(all_lifetimes) > 1 else 0.0

        return {
            "beta_0": float(beta_0),
            "beta_1": float(beta_1),
            "persistence_entropy": float(pers_entropy),
            "wasserstein_dist": float(min(w_dist, 10.0)),  # cap outliers
            "max_persistence_h0": float(min(max_h0, 2.0)),
            "max_persistence_h1": float(min(max_h1, 2.0)),
            "spectral_gap": float(spec_gap),
            "sci": float(sci),
            "entropy_h0": float(entropy_h0),
            "entropy_h1": float(entropy_h1),
        }

    def extract_series(self, prices: pd.Series) -> pd.DataFrame:
        """
        Compute TDA features for every bar in a price series.

        Uses a rolling window of self.window bars.

        Parameters
        ----------
        prices : pd.Series
            DatetimeIndexed price series.

        Returns
        -------
        pd.DataFrame with TDA features, same index as prices (NaN for warm-up).
        """
        n = len(prices)
        rows = []
        idx = []

        px = prices.values.astype(np.float64)
        self._prev_h1_diagram = None  # reset state

        for i in range(self.window, n, self.stride):
            window_data = px[i - self.window:i]
            feats = self.extract_window(window_data)
            rows.append(feats)
            idx.append(prices.index[i])

        if not rows:
            return pd.DataFrame()

        result = pd.DataFrame(rows, index=idx)
        return result

    @staticmethod
    def _zero_features() -> dict[str, float]:
        return {
            "beta_0": 0.0,
            "beta_1": 0.0,
            "persistence_entropy": 0.0,
            "wasserstein_dist": 0.0,
            "max_persistence_h0": 0.0,
            "max_persistence_h1": 0.0,
            "spectral_gap": 0.5,
            "sci": 0.0,
            "entropy_h0": 0.0,
            "entropy_h1": 0.0,
        }

"""
ml/hrp.py
=========
Hierarchical Risk Parity (HRP) Portfolio Construction.

Implements the HRP algorithm from Lopez de Prado's
*Advances in Financial Machine Learning* (Ch. 16).

HRP solves the fundamental problem with mean-variance (Markowitz)
optimisation: instability.  Instead of inverting the covariance matrix
(which is poorly conditioned for realistic portfolios), HRP uses:

    1. **Tree clustering** — group correlated assets into clusters using
       hierarchical (agglomerative) clustering on a correlation-based
       distance metric.
    2. **Quasi-diagonalisation** — reorder the covariance matrix so that
       correlated assets are adjacent.
    3. **Recursive bisection** — allocate weights top-down through the
       dendrogram, splitting risk equally at each level.

The result is a portfolio that:
- Is fully invested (weights sum to 1)
- Has better out-of-sample Sharpe than mean-variance
- Is stable under small perturbations of the correlation matrix
- Does not require expected return estimates

Usage
-----
    from ml.hrp import hierarchical_risk_parity

    weights = hierarchical_risk_parity(returns_df)
    # weights → {"AAPL": 0.12, "MSFT": 0.08, ...}

References
----------
- Lopez de Prado (2016), "Building Diversified Portfolios that Outperform
  Out-of-Sample", Journal of Portfolio Management.
- Lopez de Prado (2018), AFML, Ch. 16 — Portfolio Construction.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


def _correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Convert a correlation matrix to a proper distance matrix.

    Uses the distance metric from Lopez de Prado (2016):
        d(i,j) = sqrt(0.5 * (1 - corr(i,j)))

    This maps correlation in [-1, 1] to distance in [0, 1]:
    - corr = 1  → d = 0
    - corr = 0  → d ≈ 0.707
    - corr = -1 → d = 1

    Parameters
    ----------
    corr : pd.DataFrame
        Correlation matrix (symmetric, ones on diagonal).

    Returns
    -------
    pd.DataFrame
        Distance matrix.
    """
    return ((1.0 - corr) / 2.0).clip(lower=0.0) ** 0.5


def _quasi_diag(link: np.ndarray) -> List[int]:
    """Quasi-diagonalisation: reorder assets by the dendrogram leaf order.

    This places correlated assets adjacent in the covariance matrix,
    creating a quasi-diagonal structure.

    Parameters
    ----------
    link : np.ndarray
        Linkage matrix from scipy.cluster.hierarchy.linkage.

    Returns
    -------
    List[int]
        Reordered asset indices.
    """
    return list(leaves_list(link))


def _recursive_bisection(
    cov: pd.DataFrame,
    sorted_indices: List[int],
) -> pd.Series:
    """Recursive bisection: top-down weight allocation through the dendrogram.

    Splits the asset list in half at each level.  For each split, the
    weight of each half is proportional to the inverse variance of the
    cluster:

        w_left  ∝ 1 / var(cluster_left)
        w_right ∝ 1 / var(cluster_right)

    This allocates more weight to lower-risk clusters (risk parity).

    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix (quasi-diagonalised order).
    sorted_indices : List[int]
        Asset indices in quasi-diagonalised order.

    Returns
    -------
    pd.Series
        Weights indexed by the original column names.
    """
    w = pd.Series(1.0, index=cov.columns)
    items = [sorted_indices]

    while items:
        new_items = []
        for subset in items:
            if len(subset) <= 1:
                continue

            mid = len(subset) // 2
            left = subset[:mid]
            right = subset[mid:]

            # Cluster variance = w' Σ w / (w'w), using inverse-vol weights
            left_cols = cov.columns[left]
            right_cols = cov.columns[right]

            left_var = _cluster_var(cov, left_cols)
            right_var = _cluster_var(cov, right_cols)

            # Inverse variance allocation between left and right
            total_var = left_var + right_var
            if total_var > 0:
                alpha = 1.0 - left_var / total_var
            else:
                alpha = 0.5

            w[left_cols] *= alpha
            w[right_cols] *= (1.0 - alpha)

            new_items.append(left)
            new_items.append(right)

        items = new_items

    return w


def _cluster_var(cov: pd.DataFrame, assets: pd.Index) -> float:
    """Compute the variance of an equal-weight portfolio of assets.

    Parameters
    ----------
    cov : pd.DataFrame
        Full covariance matrix.
    assets : pd.Index
        Asset names in the cluster.

    Returns
    -------
    float
        Portfolio variance.
    """
    sub_cov = cov.loc[assets, assets]
    n = len(assets)
    if n == 0:
        return 0.0
    # Inverse-vol weights within cluster
    ivp = 1.0 / np.diag(sub_cov)
    ivp = ivp / ivp.sum()
    return float(ivp @ sub_cov.values @ ivp)


def hierarchical_risk_parity(
    returns: pd.DataFrame,
    min_weight: float = 0.01,
    max_weight: float = 0.25,
) -> Dict[str, float]:
    """Compute HRP portfolio weights from a returns matrix.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns matrix (rows = dates, columns = symbols).
        Requires at least 20 observations.
    min_weight : float
        Minimum weight per asset (floor).
    max_weight : float
        Maximum weight per asset (cap).

    Returns
    -------
    Dict[str, float]
        Symbol → weight mapping, summing to 1.0.
    """
    # Clean data
    returns = returns.dropna(axis=1, how="all").dropna(axis=0, how="any")

    if returns.empty or len(returns) < 20 or returns.shape[1] < 2:
        logger.warning(
            f"HRP: insufficient data ({returns.shape}); returning equal weights."
        )
        if returns.shape[1] > 0:
            n = returns.shape[1]
            return {col: 1.0 / n for col in returns.columns}
        return {}

    # Step 1: Compute correlation and distance matrices
    corr = returns.corr()
    dist = _correlation_distance(corr)

    # Step 2: Hierarchical clustering
    # Convert to condensed distance matrix for scipy
    dist_condensed = squareform(dist.values, checks=False)
    # Replace any NaN/inf with maximum distance
    dist_condensed = np.nan_to_num(dist_condensed, nan=1.0, posinf=1.0, neginf=0.0)
    link = linkage(dist_condensed, method="single")

    # Step 3: Quasi-diagonalisation
    sorted_idx = _quasi_diag(link)

    # Step 4: Covariance matrix
    cov = returns.cov()

    # Ensure positive diagonal (numerical safety)
    for col in cov.columns:
        if cov.loc[col, col] <= 0:
            cov.loc[col, col] = 1e-8

    # Step 5: Recursive bisection
    weights = _recursive_bisection(cov, sorted_idx)

    # Normalise
    total = weights.sum()
    if total > 0:
        weights = weights / total

    # Apply floors and caps
    weights = weights.clip(lower=min_weight, upper=max_weight)
    # Re-normalise after clipping
    total = weights.sum()
    if total > 0:
        weights = weights / total

    result = weights.to_dict()

    logger.info(
        f"HRP: {len(result)} assets, "
        f"max_wt={max(result.values()):.3f}, "
        f"min_wt={min(result.values()):.3f}, "
        f"HHI={sum(v**2 for v in result.values()):.4f}"
    )

    return result


def apply_hrp_to_signals(
    signals: list,
    returns: pd.DataFrame,
    min_weight: float = 0.01,
    max_weight: float = 0.25,
) -> list:
    """Scale signal strengths by HRP portfolio weights.

    This replaces the naive signal-weighted allocation with a
    mathematically optimal risk-parity allocation.

    Parameters
    ----------
    signals : list
        List of Signal objects with .symbol and .strength attributes.
    returns : pd.DataFrame
        Recent returns matrix for the traded symbols.
    min_weight : float
        Minimum HRP weight per symbol.
    max_weight : float
        Maximum HRP weight per symbol.

    Returns
    -------
    list
        Signals with strengths adjusted by HRP weights.
    """
    if not signals or returns.empty:
        return signals

    # Get unique symbols from signals
    signal_symbols = list({s.symbol for s in signals})
    available = [s for s in signal_symbols if s in returns.columns]

    if len(available) < 2:
        return signals  # Need at least 2 assets for HRP

    # Compute HRP weights
    hrp_weights = hierarchical_risk_parity(
        returns[available],
        min_weight=min_weight,
        max_weight=max_weight,
    )

    # Scale signal strengths by HRP weights
    for signal in signals:
        if signal.symbol in hrp_weights:
            hrp_w = hrp_weights[signal.symbol]
            # Blend: 60% HRP weight, 40% original signal strength
            # This preserves signal conviction while adding diversification
            blended = 0.4 * signal.strength + 0.6 * hrp_w * len(available)
            signal.strength = float(np.clip(blended, 0.01, 1.0))
            signal.metadata["hrp_weight"] = hrp_w

    return signals

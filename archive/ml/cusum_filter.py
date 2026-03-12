"""
ml/cusum_filter.py
==================
CUSUM (Cumulative Sum) event-driven sampling filter.

Implements the symmetric CUSUM filter from Lopez de Prado's
*Advances in Financial Machine Learning* (Ch. 2.5.2.1).

Instead of sampling at fixed time intervals (daily bars), the CUSUM filter
detects structurally important events — points where the cumulative deviation
from the mean exceeds a threshold.  These events represent genuine regime
shifts or breakouts and produce higher-quality training samples for ML models.

Usage
-----
    from ml.cusum_filter import cusum_filter

    events = cusum_filter(close_prices, threshold=0.02)
    # events is a DatetimeIndex of structurally important timestamps.

References
----------
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*,
  Chapter 2, Section 2.5.2.1 — The CUSUM Filter.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def cusum_filter(
    close: pd.Series,
    threshold: Optional[float] = None,
    vol_lookback: int = 20,
    vol_multiplier: float = 1.0,
) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter for event-driven sampling.

    Monitors the cumulative sum of positive and negative log-return
    deviations from zero.  When either the positive or negative CUSUM
    exceeds the threshold, the bar is flagged as a structural event
    and both counters are reset to zero.

    Parameters
    ----------
    close : pd.Series
        Close price series with DatetimeIndex.
    threshold : float, optional
        Fixed threshold for the CUSUM trigger.  If None, the threshold
        is set dynamically as ``vol_multiplier × rolling_std(log_returns)``.
    vol_lookback : int
        Rolling window (bars) for the dynamic volatility threshold.
    vol_multiplier : float
        Multiplier applied to the rolling volatility to set the threshold.
        Higher values produce fewer (more significant) events.

    Returns
    -------
    pd.DatetimeIndex
        Timestamps of structural events detected by the filter.
    """
    if close.empty or len(close) < 2:
        return pd.DatetimeIndex([])

    log_ret = np.log(close / close.shift(1)).dropna()

    if log_ret.empty:
        return pd.DatetimeIndex([])

    # Dynamic or fixed threshold
    if threshold is None:
        rolling_vol = log_ret.rolling(vol_lookback, min_periods=max(5, vol_lookback // 4)).std()
        h = rolling_vol * vol_multiplier
        h = h.bfill().fillna(log_ret.std())
    else:
        h = pd.Series(threshold, index=log_ret.index)

    events = []
    s_pos = 0.0
    s_neg = 0.0

    for i, (dt, ret) in enumerate(log_ret.items()):
        if np.isnan(ret):
            continue

        thresh_val = h.iloc[i] if not np.isnan(h.iloc[i]) else abs(ret) * 2.0

        s_pos = max(0.0, s_pos + ret)
        s_neg = min(0.0, s_neg + ret)

        if s_pos > thresh_val:
            events.append(dt)
            s_pos = 0.0
            s_neg = 0.0
        elif s_neg < -thresh_val:
            events.append(dt)
            s_pos = 0.0
            s_neg = 0.0

    logger.info(
        f"CUSUM filter: {len(events)} events from {len(log_ret)} bars "
        f"({len(events) / max(len(log_ret), 1):.1%} sampling rate)."
    )

    return pd.DatetimeIndex(events)


def cusum_sample_weights(
    events: pd.DatetimeIndex,
    close: pd.Series,
    horizon: int = 5,
) -> pd.Series:
    """Compute sample weights based on return magnitude at event points.

    Events with larger absolute returns receive higher weight, focusing
    the model on the most informative samples.

    Parameters
    ----------
    events : pd.DatetimeIndex
        CUSUM-detected event timestamps.
    close : pd.Series
        Close price series.
    horizon : int
        Forward return window (bars) for weight computation.

    Returns
    -------
    pd.Series
        Weights indexed by event timestamp, summing to 1.
    """
    if events.empty:
        return pd.Series(dtype=float)

    weights = {}
    for dt in events:
        loc = close.index.get_loc(dt)
        fwd_loc = min(loc + horizon, len(close) - 1)
        fwd_ret = abs(np.log(close.iloc[fwd_loc] / close.iloc[loc]))
        weights[dt] = fwd_ret

    w = pd.Series(weights)
    total = w.sum()
    if total > 0:
        w = w / total
    return w

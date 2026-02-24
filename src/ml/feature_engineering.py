"""
Feature Engineering for Online ML Learner
==========================================

Builds a normalized feature vector from signal + market data for the
SGDClassifier-based OnlineLearner.

Features (7 dimensions, all normalized 0-1):
    0. iv_rank       — Implied volatility rank (0-100 → 0-1)
    1. vix_level     — VIX level (10-80 → 0-1)
    2. dte           — Days to expiration (0-90 → 0-1)
    3. delta         — Option delta (0-1 already)
    4. rv_iv_ratio   — Realized vol / implied vol ratio (0-3 → 0-1)
    5. hour          — Hour of day ET (9-16 → 0-1)
    6. weekday       — Day of week (0=Mon..4=Fri → 0-1)
"""

from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

__all__ = ["build_features"]


def _clip_normalize(value: float, lo: float, hi: float) -> float:
    """Clip value to [lo, hi] then normalize to [0, 1]."""
    if hi <= lo:
        return 0.5
    clamped = max(lo, min(hi, value))
    return (clamped - lo) / (hi - lo)


def build_features(
    signal_dict: Dict[str, Any],
    market_dict: Dict[str, Any],
) -> np.ndarray:
    """Build a normalized feature vector for the OnlineLearner SGD model.

    Args:
        signal_dict: Signal metadata containing any of:
            - iv_rank (float, 0-100)
            - dte (int)
            - delta (float)
            - confidence (float)
        market_dict: Market context containing any of:
            - vix_level (float)
            - realized_vol (float)
            - implied_vol (float)
            - timestamp (datetime or ISO string)

    Returns:
        np.ndarray of shape (7,) with values in [0, 1].
    """
    # 0. IV Rank (0-100)
    iv_rank = float(signal_dict.get("iv_rank") or 50.0)
    f_iv_rank = _clip_normalize(iv_rank, 0.0, 100.0)

    # 1. VIX level (10-80)
    vix_level = float(market_dict.get("vix_level") or 20.0)
    f_vix = _clip_normalize(vix_level, 10.0, 80.0)

    # 2. DTE (0-90)
    dte = float(signal_dict.get("dte") or 30)
    f_dte = _clip_normalize(dte, 0.0, 90.0)

    # 3. Delta (0-1)
    delta = abs(float(signal_dict.get("delta") or 0.3))
    f_delta = _clip_normalize(delta, 0.0, 1.0)

    # 4. RV/IV ratio (0-3)
    rv = float(market_dict.get("realized_vol") or 0.20)
    iv = float(market_dict.get("implied_vol") or 0.20)
    rv_iv_ratio = rv / iv if iv > 1e-6 else 1.0
    f_rv_iv = _clip_normalize(rv_iv_ratio, 0.0, 3.0)

    # 5. Hour of day (9-16 ET)
    ts = market_dict.get("timestamp")
    if ts is None:
        ts = datetime.now()
    elif isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            ts = datetime.now()
    hour = ts.hour
    f_hour = _clip_normalize(float(hour), 9.0, 16.0)

    # 6. Weekday (0=Mon..4=Fri)
    weekday = ts.weekday()
    f_weekday = _clip_normalize(float(weekday), 0.0, 4.0)

    return np.array([f_iv_rank, f_vix, f_dte, f_delta, f_rv_iv, f_hour, f_weekday],
                    dtype=np.float64)

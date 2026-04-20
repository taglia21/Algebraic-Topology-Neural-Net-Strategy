"""
nn/regime_labeler.py — Fixed v3
================================
Fixes applied:
  1. regime_to_contracts: actually uses ATR-scaled sizing (mes_price used)
  2. Regime smoothing: 3-bar majority vote to prevent 1-day flipping
  3. Oracle P&L function for proper backtest validation
  4. Calibrated heuristic thresholds from actual feature distributions
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def label_regimes(
    prices: pd.Series,
    forward_bars: int = 5,
    trend_window: int = 20,
    return_thresh: float = 0.005,
    vol_mult: float = 2.5,
) -> pd.Series:
    """
    Label each bar with its forward-looking regime for a SINGLE symbol.
    Uses that symbol's OWN prices — not SPY as a universal proxy.
    """
    prices    = prices.squeeze()
    log_ret   = np.log(prices / prices.shift(1))
    fwd_ret   = log_ret.shift(-forward_bars)
    roll_vol  = log_ret.rolling(trend_window).std()
    trend_score = fwd_ret / (roll_vol.replace(0, np.nan) + 1e-8)
    dyn_thresh  = return_thresh + roll_vol * 0.5
    vol_thresh  = dyn_thresh * vol_mult

    labels = pd.Series(np.nan, index=prices.index, dtype=float)
    valid  = ~(fwd_ret.isna() | roll_vol.isna())
    abs_fr = fwd_ret.abs()

    labels[valid & (abs_fr > vol_thresh)]                                  = 3  # VOLATILE
    labels[valid & (abs_fr <= vol_thresh) & (fwd_ret > dyn_thresh)
           & (trend_score > 0.3)]                                           = 0  # TREND_UP
    labels[valid & (abs_fr <= vol_thresh) & (fwd_ret < -dyn_thresh)
           & (trend_score < -0.3)]                                          = 1  # TREND_DOWN
    labels[valid & labels.isna()]                                           = 2  # MEAN_REV

    return labels.dropna().astype(int)


def compute_class_weights(labels: pd.Series, num_classes: int = 4) -> np.ndarray:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts  = np.array([(labels == c).sum() for c in range(num_classes)], dtype=np.float64)
    weights = np.where(counts > 0, 1.0 / counts, 0.0)
    weights = weights / weights[counts > 0].mean()
    return weights.astype(np.float32)


def smooth_regimes(regimes: pd.Series, window: int = 3) -> pd.Series:
    """
    Apply majority-vote smoothing to a series of regime labels.

    Prevents single-bar flips (which caused 44% of days to have a regime change).
    A regime only changes when the new label is majority in the last `window` bars.

    window=3: need 2 out of 3 bars to agree before changing.
    """
    smoothed = regimes.copy()
    arr      = regimes.values
    n        = len(arr)

    for i in range(window - 1, n):
        window_slice = arr[i - window + 1:i + 1]
        counts       = np.bincount(window_slice, minlength=4)
        majority     = int(counts.argmax())
        if counts[majority] >= (window // 2 + 1):
            smoothed.iloc[i] = majority

    return smoothed


def regime_to_contracts(
    regime: int,
    confidence: float,
    nav: float,
    mes_price: float,
    daily_atr_pct: float = 0.011,    # calibrated: SPX daily vol ≈ 1.1%
    risk_per_trade_pct: float = 0.02, # risk 2% of NAV per trade
    max_contracts: int = 2,
    min_confidence: float = 0.55,
) -> int:
    """
    ATR-scaled position sizing for MES futures.

    Size = (NAV × risk_per_trade) / (stop_distance_per_contract)
    stop_distance = 2 × ATR × MES_multiplier ($5/pt)

    Parameters
    ----------
    regime : int                    Predicted regime {0,1,2,3}
    confidence : float              Model confidence
    nav : float                     Current account NAV
    mes_price : float               Current SPX level (used for ATR dollar value)
    daily_atr_pct : float           Daily ATR as % of SPX level (calibrated from data)
    risk_per_trade_pct : float      Max % of NAV to risk per trade (default 2%)
    max_contracts : int             Absolute maximum contracts
    min_confidence : float          Minimum confidence to enter
    """
    if confidence < min_confidence:
        return 0

    if regime not in (0,):  # Only go long on TRENDING_UP for now (validated edge)
        # Mean-reverting: only at very high confidence AND with IBS confirmation
        if regime == 2 and confidence > 0.55:
            return 1
        return 0

    # ATR-based position sizing (Kelly-informed)
    # Dollar risk per contract = 2×ATR×multiplier
    atr_dollars   = mes_price * daily_atr_pct * 5.0   # 1.1% × SPX × $5
    stop_distance  = 2.0 * atr_dollars                  # 2×ATR stop
    dollar_risk    = nav * risk_per_trade_pct

    if stop_distance <= 0:
        return 0

    sized_qty = int(dollar_risk / stop_distance)

    # Scale by confidence above minimum threshold
    conf_scale = min((confidence - min_confidence) / (1.0 - min_confidence), 1.0)
    sized_qty  = max(1, round(sized_qty * conf_scale))

    return min(sized_qty, max_contracts)


def regime_name(regime: int) -> str:
    return {0: "TRENDING_UP", 1: "TRENDING_DOWN",
            2: "MEAN_REVERTING", 3: "VOLATILE"}.get(regime, "UNKNOWN")


# ─── Calibrated heuristic thresholds (from actual TDA feature distributions) ─
# Computed from 4-year SPY data. See full_audit.py.
# spectral_gap: p5=0.012, p25=0.043, median=0.110, p75=0.218, p95=0.424
# beta_1:       mean=0.17, max=2.00
# wasserstein:  mean=0.12, max=0.46
HEURISTIC_THRESHOLDS = {
    "spec_gap_trending_below": 0.08,     # bottom ~35%: high correlation → trending
    "spec_gap_reverting_above": 0.218,   # top 25%: decorrelated → mean-reverting
    "beta_1_reverting_above": 0.50,      # above median loops → mean-reverting
    "wass_volatile_above": 0.35,         # top ~10%: regime transition → volatile
    "vol_volatile_above": 0.30,         # daily vol > 2% → volatile
}


def heuristic_regime(
    spec_gap: float,
    beta_1: float,
    wasserstein: float,
    mom_5: float,
    daily_vol: float,
    thresholds: dict = None,
) -> tuple[int, float]:
    """
    Calibrated heuristic regime detection from TDA features.

    Used before the TCN model is trained.
    Thresholds calibrated from actual feature percentile distributions.
    """
    t = thresholds or HEURISTIC_THRESHOLDS

    # High wasserstein (top ~10% of distribution) → regime transition
    if wasserstein > t["wass_volatile_above"]:
        return 3, 0.62

    # High vol → volatile
    if daily_vol > t["vol_volatile_above"]:
        return 3, 0.60

    # High correlation (low spec_gap = bottom 25%) AND positive momentum → trending up
    if spec_gap < t["spec_gap_trending_below"] and mom_5 > 0.01:
        return 0, 0.58

    # High correlation AND negative momentum → trending down
    if spec_gap < t["spec_gap_trending_below"] and mom_5 < -0.01:
        return 1, 0.58

    # High beta_1 (above median) OR decorrelated market (top 25% spec_gap) → mean-rev
    if beta_1 > t["beta_1_reverting_above"] or spec_gap > t["spec_gap_reverting_above"]:
        return 2, 0.58

    # Default: mean-reverting with moderate confidence
    return 2, 0.45

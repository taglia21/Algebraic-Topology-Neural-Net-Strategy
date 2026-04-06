"""
nn/regime_labeler.py
====================
Creates regime labels and provides the regime→signal bridge.

Regime definitions (5-bar forward window):
  0 = TRENDING_UP
  1 = TRENDING_DOWN
  2 = MEAN_REVERTING
  3 = VOLATILE / CHOPPY

Key design principle: labels are PER-SYMBOL (not derived from SPY alone).
SPY is used only as the market regime anchor. Each symbol gets its own
label from its own forward return, modulated by the SPY regime.
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

    Parameters
    ----------
    prices : pd.Series
        Close prices for ONE symbol (not SPY as a universal proxy).
    forward_bars : int
        Forward horizon in bars (default 5).
    trend_window : int
        Lookback for rolling vol (default 20).
    return_thresh : float
        Minimum return to be considered trending (default 0.5%).
    vol_mult : float
        Multiple of avg vol to define 'volatile' (default 2.5).

    Returns
    -------
    pd.Series of int {0,1,2,3}.
    """
    prices = prices.squeeze()
    log_ret   = np.log(prices / prices.shift(1))
    fwd_ret   = log_ret.shift(-forward_bars)
    roll_vol  = log_ret.rolling(trend_window).std()

    # Trend score: forward Sharpe (look-ahead, training only)
    trend_score = fwd_ret / (roll_vol.replace(0, np.nan) + 1e-8)

    # Dynamic thresholds
    dyn_thresh = return_thresh + roll_vol * 0.5
    vol_thresh  = dyn_thresh * vol_mult

    # Vectorized labeling (much faster than row-by-row)
    labels = pd.Series(np.nan, index=prices.index, dtype=float)

    valid = ~(fwd_ret.isna() | roll_vol.isna())

    abs_fr = fwd_ret.abs()

    # Priority order: volatile > trending > mean-reverting
    labels[valid & (abs_fr > vol_thresh)] = 3
    labels[valid & (abs_fr <= vol_thresh) & (fwd_ret > dyn_thresh) & (trend_score > 0.3)] = 0
    labels[valid & (abs_fr <= vol_thresh) & (fwd_ret < -dyn_thresh) & (trend_score < -0.3)] = 1
    # Everything else = mean-reverting (fill remaining valid bars)
    labels[valid & labels.isna()] = 2

    return labels.dropna().astype(int)


def compute_class_weights(labels: pd.Series, num_classes: int = 4) -> np.ndarray:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    This is critical: without weighting, a model trained on 72% class-2 data
    will learn to always predict class 2 and report "72% accuracy" — meaningless.

    Returns float32 array of shape (num_classes,) for use as:
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights))
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        counts[c] = (labels == c).sum()

    # Inverse frequency, scaled so mean weight = 1
    weights = np.where(counts > 0, 1.0 / counts, 0.0)
    weights = weights / weights[counts > 0].mean()

    return weights.astype(np.float32)


def regime_to_contracts(
    regime: int,
    confidence: float,
    nav: float,
    mes_price: float,
    max_contracts: int = 2,
    min_confidence: float = 0.55,
) -> int:
    """
    Map TCN regime prediction to MES contract count.

    This is the ACTUAL position sizing function — used by live_futures.py.
    Confidence-gated: if model isn't sure, stay flat.

    Parameters
    ----------
    regime : int
        Predicted regime class {0,1,2,3}.
    confidence : float
        Softmax probability of the predicted class.
    nav : float
        Current account NAV.
    mes_price : float
        Current MES price (SPX index level).
    max_contracts : int
        Maximum contracts to hold.
    min_confidence : float
        Minimum confidence to act (default 0.55 — above majority-class threshold).

    Returns
    -------
    int: Number of contracts (positive=long, negative=short, 0=flat).
         Note: short selling on futures requires margin and carries risk.
         For now, returns 0 for short signals (conservative).
    """
    if confidence < min_confidence:
        return 0  # not confident enough

    if regime == 0:  # TRENDING_UP — long
        # Scale from 1 to max_contracts based on confidence
        scale = min((confidence - min_confidence) / (1.0 - min_confidence), 1.0)
        qty = max(1, round(max_contracts * scale))
        return qty

    elif regime == 1:  # TRENDING_DOWN — flat (conservative: no shorting yet)
        return 0

    elif regime == 2:  # MEAN_REVERTING
        # Only enter mean-reversion at very high confidence (model rarely confident here)
        return 1 if confidence > 0.70 else 0

    else:  # VOLATILE — flat
        return 0


def regime_name(regime: int) -> str:
    return {0: "TRENDING_UP", 1: "TRENDING_DOWN",
            2: "MEAN_REVERTING", 3: "VOLATILE"}.get(regime, "UNKNOWN")

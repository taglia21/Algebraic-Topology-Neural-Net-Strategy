"""
nn/regime_labeler.py
====================
Creates regime labels for training the TCN.

Regime definitions (based on 5-bar forward window):
  0 = TRENDING_UP:    forward_return > +thresh AND abs(trend_score) > 0.5
  1 = TRENDING_DOWN:  forward_return < -thresh AND abs(trend_score) > 0.5
  2 = MEAN_REVERTING: abs(forward_return) < thresh AND autocorr < 0
  3 = VOLATILE:       abs(forward_return) > thresh*2 (large erratic move)

These are determined entirely from PAST+FUTURE data during training
and from PAST data only during live trading (predicted by TCN).
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def label_regimes(
    prices: pd.Series,
    forward_bars: int = 5,
    trend_window: int = 20,
    return_thresh: float = 0.005,   # 0.5% threshold
    vol_mult: float = 2.5,
) -> pd.Series:
    """
    Label each bar with its forward-looking regime.

    Parameters
    ----------
    prices : pd.Series
        Daily close prices.
    forward_bars : int
        Horizon for computing forward return (default 5 days).
    trend_window : int
        Lookback for measuring current trend (default 20 days).
    return_thresh : float
        Minimum return to classify as trending (default 0.5%).
    vol_mult : float
        Multiplier on rolling vol to define 'volatile' regime.

    Returns
    -------
    pd.Series of int {0, 1, 2, 3}, NaN for last forward_bars rows.
    """
    log_ret   = np.log(prices / prices.shift(1))
    fwd_ret   = log_ret.shift(-forward_bars)          # forward-looking (training only)
    roll_vol  = log_ret.rolling(trend_window).std()
    autocorr  = log_ret.rolling(trend_window).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 3 else 0, raw=False
    )

    # Trend score: ratio of return to recent volatility (like a forward Sharpe)
    trend_score = fwd_ret / (roll_vol + 1e-8)

    # Dynamic threshold scaled to vol regime
    dyn_thresh = return_thresh + roll_vol * 0.5
    vol_thresh  = dyn_thresh * vol_mult

    labels = pd.Series(np.nan, index=prices.index)

    for idx in prices.index:
        fr  = fwd_ret.get(idx, np.nan)
        ts  = trend_score.get(idx, 0)
        rv  = roll_vol.get(idx, 0.01)
        ac  = autocorr.get(idx, 0)
        dt  = dyn_thresh.get(idx, return_thresh)
        vt  = vol_thresh.get(idx, return_thresh * vol_mult)

        if np.isnan(fr) or np.isnan(rv):
            continue

        abs_fr = abs(fr)

        if abs_fr > vt:
            labels[idx] = 3    # VOLATILE (large erratic move)
        elif fr > dt and ts > 0.3:
            labels[idx] = 0    # TRENDING UP
        elif fr < -dt and ts < -0.3:
            labels[idx] = 1    # TRENDING DOWN
        else:
            labels[idx] = 2    # MEAN REVERTING / SIDEWAYS

    return labels.dropna().astype(int)


def regime_to_strategy_weights(regime: int) -> dict[str, float]:
    """
    Map regime class to strategy sleeve weights.

    Returns weights for: momentum, mean_reversion, stat_arb, cash

    This is the core of how TDA-detected regime drives the ensemble.
    """
    weights = {
        0: {"momentum": 0.70, "mean_reversion": 0.10, "stat_arb": 0.10, "cash": 0.10},  # trend up
        1: {"momentum": 0.00, "mean_reversion": 0.20, "stat_arb": 0.20, "cash": 0.60},  # trend down → defensive
        2: {"momentum": 0.10, "mean_reversion": 0.60, "stat_arb": 0.30, "cash": 0.00},  # mean rev
        3: {"momentum": 0.00, "mean_reversion": 0.00, "stat_arb": 0.00, "cash": 1.00},  # volatile → flat
    }
    return weights.get(regime, weights[2])

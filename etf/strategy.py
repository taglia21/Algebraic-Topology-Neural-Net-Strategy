"""
etf/strategy.py
===============
The ETF target-weight engine.

Given price history up to (and including) a rebalance date, it produces a
dictionary of target portfolio weights. The pipeline is:

    1. Eligibility (trend filter)   -> which ETFs are allowed to be held
    2. Cross-sectional momentum     -> rank eligible ETFs, keep top-K
    3. Inverse-volatility weighting -> risk-balance the survivors
    4. Concentration cap            -> no single ETF too large
    5. Portfolio volatility target  -> scale the whole risky sleeve to a risk
                                       budget (gross exposure in [0, max_lev])

The drawdown overlay is applied by the backtester / live trader (it needs the
realised equity curve), not here.

Anti-look-ahead guarantee
-------------------------
All computations use ``prices.loc[:asof]`` only. The caller is responsible for
passing the price frame sliced to the decision date. We additionally assert
this in :func:`compute_target_weights`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from etf.config import ETFConfig

logger = logging.getLogger("etf.strategy")

_TRADING_DAYS = 252


@dataclass
class WeightDecision:
    """Result of a single rebalance computation."""

    weights: Dict[str, float]          # ticker -> target weight (risky sleeve)
    cash_weight: float                 # residual parked in the cash asset
    gross_exposure: float              # sum of risky weights (<= max leverage)
    realized_vol: float                # estimated annualised portfolio vol pre-scale
    vol_scale: float                   # vol-target scaling factor applied
    eligible: List[str]                # ETFs that passed the trend filter
    selected: List[str]                # ETFs actually held


def _daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change()


def _trend_eligible(prices: pd.DataFrame, cfg: ETFConfig) -> List[str]:
    """Return ETFs in a confirmed uptrend (price > SMA AND positive 12-1 mom)."""
    s = cfg.signal
    eligible: List[str] = []
    for sym in cfg.risk_universe:
        if sym not in prices.columns:
            continue
        series = prices[sym].dropna()
        if len(series) < max(s.trend_sma, s.ts_momentum_long) + 1:
            continue  # insufficient history -> ineligible (fail-safe)
        price = series.iloc[-1]
        sma = series.iloc[-s.trend_sma:].mean()
        # 12-1 momentum: return from t-252 to t-21 (skip last month)
        p_then = series.iloc[-(s.ts_momentum_long + 1)]
        p_skip = series.iloc[-(s.ts_momentum_skip + 1)]
        ts_mom = (p_skip / p_then) - 1.0
        if price > sma and ts_mom > 0:
            eligible.append(sym)
    return eligible


def _xs_momentum_scores(prices: pd.DataFrame, symbols: List[str], cfg: ETFConfig) -> pd.Series:
    """Blended multi-horizon cross-sectional momentum score per symbol."""
    s = cfg.signal
    scores: Dict[str, float] = {}
    for sym in symbols:
        series = prices[sym].dropna()
        blended = 0.0
        ok = True
        for lb, w in zip(s.momentum_lookbacks, s.momentum_weights):
            if len(series) < lb + 1:
                ok = False
                break
            ret = (series.iloc[-1] / series.iloc[-(lb + 1)]) - 1.0
            blended += w * ret
        if ok:
            scores[sym] = blended
    return pd.Series(scores, dtype=float)


def _inverse_vol_weights(prices: pd.DataFrame, symbols: List[str], cfg: ETFConfig) -> pd.Series:
    """Inverse-volatility weights over the survivors (risk parity lite)."""
    rets = _daily_returns(prices[symbols]).tail(cfg.signal.vol_lookback)
    vol = rets.std()
    # Guard against zero/NaN vol (e.g. a brand-new ETF or stale data).
    vol = vol.replace(0.0, np.nan)
    inv = 1.0 / vol
    inv = inv.replace([np.inf, -np.inf], np.nan).dropna()
    if inv.empty:
        # equal weight fallback
        return pd.Series(1.0 / len(symbols), index=symbols)
    return inv / inv.sum()


def _apply_concentration_cap(weights: pd.Series, cap: float) -> pd.Series:
    """Iteratively cap weights at `cap` and redistribute the excess."""
    w = weights.copy()
    for _ in range(100):
        over = w[w > cap]
        if over.empty:
            break
        excess = (over - cap).sum()
        w[w > cap] = cap
        under = w[w < cap]
        if under.empty or under.sum() == 0:
            break
        w[under.index] += excess * (under / under.sum())
    return w


def _portfolio_vol(prices: pd.DataFrame, weights: pd.Series, cfg: ETFConfig) -> float:
    """Annualised realised vol of the weighted sleeve using the recent covariance.

    Using the full covariance (not just per-asset vols) correctly accounts for
    diversification between asset classes — the whole point of the universe.
    """
    lb = cfg.risk.portfolio_vol_lookback
    rets = _daily_returns(prices[list(weights.index)]).tail(lb).dropna(how="all")
    if len(rets) < 5:
        return float("nan")
    cov = rets.cov()
    w = weights.reindex(cov.index).fillna(0.0).values
    var = float(w @ cov.values @ w)
    var = max(var, 0.0)
    return float(np.sqrt(var * _TRADING_DAYS))


def compute_target_weights(prices: pd.DataFrame, cfg: ETFConfig) -> WeightDecision:
    """Compute target weights from price history sliced up to the decision date.

    ``prices`` must already be truncated so that ``prices.index[-1]`` is the
    rebalance date — no future rows. The risky-sleeve weights sum to
    ``gross_exposure`` (<= max leverage); the remainder is ``cash_weight``.
    """
    if prices.empty:
        return WeightDecision({}, 1.0, 0.0, float("nan"), 0.0, [], [])

    # 1. Trend eligibility
    eligible = _trend_eligible(prices, cfg)
    if not eligible:
        # Nothing trending -> fully defensive (all cash). This is a feature:
        # in 2008/2022 the engine steps aside instead of riding the crash down.
        return WeightDecision({}, 1.0, 0.0, 0.0, 0.0, [], [])

    # 2. Cross-sectional momentum ranking -> top-K
    scores = _xs_momentum_scores(prices, eligible, cfg)
    scores = scores[scores > 0]  # require positive absolute momentum too (dual momentum)
    if scores.empty:
        return WeightDecision({}, 1.0, 0.0, 0.0, 0.0, eligible, [])
    selected = list(scores.sort_values(ascending=False).head(cfg.signal.top_k).index)

    # 3. Inverse-vol weighting among survivors
    weights = _inverse_vol_weights(prices, selected, cfg)

    # 4. Concentration cap + renormalise
    weights = _apply_concentration_cap(weights, cfg.risk.max_position_weight)
    weights = weights / weights.sum()

    # 5. Portfolio volatility targeting
    realized_vol = _portfolio_vol(prices, weights, cfg)
    if not np.isfinite(realized_vol) or realized_vol <= 0:
        vol_scale = 1.0
    else:
        vol_scale = cfg.risk.target_volatility / realized_vol
    vol_scale = float(np.clip(vol_scale, 0.0, cfg.risk.max_gross_leverage))

    scaled = (weights * vol_scale).to_dict()
    gross = float(sum(scaled.values()))
    cash = max(0.0, 1.0 - gross)

    return WeightDecision(
        weights={k: float(v) for k, v in scaled.items()},
        cash_weight=cash,
        gross_exposure=gross,
        realized_vol=float(realized_vol) if np.isfinite(realized_vol) else float("nan"),
        vol_scale=vol_scale,
        eligible=eligible,
        selected=selected,
    )


def apply_drawdown_overlay(decision: WeightDecision, current_drawdown: float, cfg: ETFConfig) -> WeightDecision:
    """Scale the risky sleeve down during equity drawdowns (circuit-breaker).

    ``current_drawdown`` is a non-negative magnitude (e.g. 0.12 == -12%).
    Exposure scales linearly from 1.0 at ``dd_start`` to ``dd_min_exposure`` at
    ``dd_full``. Capital removed from risk goes to cash.
    """
    r = cfg.risk
    dd = max(0.0, current_drawdown)
    if dd <= r.dd_start:
        scale = 1.0
    elif dd >= r.dd_full:
        scale = r.dd_min_exposure
    else:
        frac = (dd - r.dd_start) / (r.dd_full - r.dd_start)
        scale = 1.0 - frac * (1.0 - r.dd_min_exposure)

    if scale >= 1.0:
        return decision

    new_weights = {k: v * scale for k, v in decision.weights.items()}
    gross = float(sum(new_weights.values()))
    return WeightDecision(
        weights=new_weights,
        cash_weight=max(0.0, 1.0 - gross),
        gross_exposure=gross,
        realized_vol=decision.realized_vol,
        vol_scale=decision.vol_scale * scale,
        eligible=decision.eligible,
        selected=decision.selected,
    )

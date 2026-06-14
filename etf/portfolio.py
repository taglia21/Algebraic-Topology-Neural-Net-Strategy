"""
etf/portfolio.py
================
Phase 3 — cross-sleeve portfolio construction (the "money gate").

This module turns the validated, low-correlation sleeves (A trend, B
mean-reversion, C defensive carry) into a single institutional book by
allocating *capital across sleeves* with **equal risk contribution (ERC)** and
then scaling the whole book to a volatility target.

Why ERC over naive inverse-vol
-------------------------------
Inverse-vol ignores correlation and therefore over-weights a low-vol sleeve even
when it is correlated to the rest (this is exactly what dragged the Sleeve-D
experiment). ERC equalises each sleeve's *contribution to portfolio risk*,
accounting for the full covariance — so a sleeve only earns weight to the extent
it actually diversifies. With low pairwise correlation, ERC concentrates risk
budget into the genuinely orthogonal sources.

No-look-ahead guarantee
-----------------------
Combiner weights at day ``t`` use a trailing covariance window that ends at
``t-1`` (``.shift(1)`` semantics via slicing ``[:t]`` excluding ``t``). The
allocation is recomputed only every ``rebalance_every`` days and held in
between, matching how a live book rebalances.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from etf.config import ETFConfig
from etf.metrics import ETFMetrics, compute_metrics
from etf.sleeves import Sleeve, backtest_sleeve

logger = logging.getLogger("etf.portfolio")

_TRADING_DAYS = 252


# ===========================================================================
# Risk-parity (ERC) solver
# ===========================================================================
def erc_weights(
    cov: np.ndarray,
    *,
    budget: Optional[np.ndarray] = None,
    iters: int = 10_000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Equal-risk-contribution (long-only) weights for covariance ``cov``.

    Solves, via the standard cyclical coordinate-descent on the convex
    log-barrier formulation, for ``w >= 0`` summing to 1 such that each asset's
    risk contribution ``w_i (cov w)_i`` matches its budget share ``budget_i``.
    With the default equal budget this is the ERC / "risk-parity" portfolio.

    Falls back to inverse-vol (then equal weight) if the covariance is
    degenerate (e.g. a zero-variance sleeve), so it never returns NaNs.
    """
    n = cov.shape[0]
    if budget is None:
        budget = np.full(n, 1.0 / n)
    budget = np.asarray(budget, dtype=float)

    diag = np.diag(cov).astype(float)
    if not np.all(np.isfinite(diag)) or np.any(diag <= 0):
        # Degenerate variance -> inverse-vol on whatever is finite, else equal.
        vol = np.sqrt(np.where(diag > 0, diag, np.nan))
        inv = np.where(np.isfinite(vol) & (vol > 0), 1.0 / vol, 0.0)
        if inv.sum() > 0:
            return inv / inv.sum()
        return np.full(n, 1.0 / n)

    # Spinu/Griveau-Billion cyclical coordinate descent on the convex objective
    #   f(y) = 1/2 y' cov y - sum_i budget_i log(y_i),
    # whose stationary point satisfies y_i (cov y)_i = budget_i. We iterate on the
    # *un-normalised* vector y (the fixed point is scale-specific) and normalise
    # only once at the end — normalising every sweep would break convergence.
    y = 1.0 / np.sqrt(diag)  # inverse-vol warm start (un-normalised)

    for _ in range(iters):
        y_prev = y.copy()
        for i in range(n):
            # Hold others fixed; solve a y_i^2 + b y_i - budget_i = 0.
            a = cov[i, i]
            b = float(cov[i, :] @ y) - cov[i, i] * y[i]
            disc = b * b + 4.0 * a * budget[i]
            y[i] = (-b + np.sqrt(disc)) / (2.0 * a)
        if not np.all(np.isfinite(y)):
            return np.full(n, 1.0 / n)
        if np.max(np.abs(y - y_prev)) < tol:
            break

    s = y.sum()
    if s <= 0 or not np.isfinite(s):
        return np.full(n, 1.0 / n)
    return y / s


def _alloc_weights(cov: np.ndarray, vols: np.ndarray, method: str) -> np.ndarray:
    """Dispatch the cross-sleeve allocation by method."""
    n = cov.shape[0]
    if method == "equal":
        return np.full(n, 1.0 / n)
    if method == "inverse_vol":
        inv = np.where(np.isfinite(vols) & (vols > 0), 1.0 / vols, 0.0)
        return inv / inv.sum() if inv.sum() > 0 else np.full(n, 1.0 / n)
    return erc_weights(cov)


def _dd_scale(drawdown: float, dd_start: float, dd_full: float, dd_min: float) -> float:
    """Linear drawdown circuit-breaker (same shape as the sleeve overlay).

    ``drawdown`` is a non-negative magnitude (0.12 == -12%). Exposure scales
    from 1.0 at ``dd_start`` down to ``dd_min`` at ``dd_full``. This is the
    Phase-4 control that keeps a *levered* book's realised drawdown bounded:
    losses shrink gross, which shrinks further losses (negative feedback).
    """
    dd = max(0.0, drawdown)
    if dd <= dd_start:
        return 1.0
    if dd >= dd_full:
        return dd_min
    frac = (dd - dd_start) / (dd_full - dd_start)
    return 1.0 - frac * (1.0 - dd_min)


# ===========================================================================
# Combined backtest
# ===========================================================================
@dataclass
class CombinedResult:
    equity: pd.Series
    returns: pd.Series
    gross_exposure: pd.Series          # total deployed fraction (post vol-target)
    sleeve_weights: pd.DataFrame       # capital allocation per sleeve over time
    metrics: ETFMetrics
    sleeve_returns: pd.DataFrame       # each sleeve's standalone daily returns
    rebalance_dates: List[pd.Timestamp]


def run_combined_backtest(
    prices: pd.DataFrame,
    cfg: ETFConfig,
    sleeves: Sequence[Sleeve],
) -> CombinedResult:
    """Backtest the ERC-combined multi-sleeve book.

    Each sleeve is run standalone (its own internal costs already charged); the
    combiner then allocates capital across the sleeve *return streams* with
    causal ERC weights and a portfolio vol target, charging combiner-level
    turnover cost when the capital allocation changes.
    """
    cfg.validate()
    pcfg = cfg.portfolio

    # 1. Standalone sleeve return streams, aligned on the common active window.
    cols: Dict[str, pd.Series] = {}
    for s in sleeves:
        cols[s.name] = backtest_sleeve(prices, s, cfg).returns
    sret = pd.DataFrame(cols).dropna()
    active = sret.abs().sum(axis=1) > 0
    if active.any():
        sret = sret.loc[active.idxmax():]
    if sret.empty:
        raise ValueError("No overlapping sleeve returns to combine")

    names = list(sret.columns)
    n = len(names)
    dates = sret.index
    ret_mat = sret.values

    cost_rate = (cfg.execution.commission_bps + cfg.execution.slippage_bps) / 1e4
    rf_daily = cfg.backtest.risk_free_rate / _TRADING_DAYS if pcfg.cash_earns_rf else 0.0
    # Phase 4: margin interest on the levered portion = (rf + spread) per day.
    borrow_daily = (cfg.backtest.risk_free_rate + pcfg.margin_spread_annual) / _TRADING_DAYS
    rcfg = cfg.risk

    warmup = pcfg.cov_lookback + 1
    equity = cfg.backtest.initial_capital
    peak = equity
    target_alloc = np.zeros(n)   # strategic target (post vol-target, pre DD overlay)
    held_alloc = np.zeros(n)     # what is actually held today (post DD overlay)
    last_rebalance = -10**9

    eq_curve: List[float] = []
    gross_series: List[float] = []
    alloc_rows: List[Dict[str, float]] = []
    rebalance_dates: List[pd.Timestamp] = []

    for i in range(len(dates)):
        # 1. Earn the day's return on what we HELD entering the day. Idle cash
        #    earns rf; a levered book pays margin interest on (gross - 1).
        if i > 0:
            deployed = float(held_alloc.sum())
            port_ret = float(held_alloc @ ret_mat[i])
            if deployed > 1.0:
                port_ret -= (deployed - 1.0) * borrow_daily
            else:
                port_ret += (1.0 - deployed) * rf_daily
            equity *= (1.0 + port_ret)

        peak = max(peak, equity)
        drawdown = 1.0 - equity / peak if peak > 0 else 0.0

        # 2. Rebalance the strategic target (vol-target + leverage cap).
        if i >= warmup and (i - last_rebalance) >= pcfg.rebalance_every:
            window = ret_mat[max(0, i - pcfg.cov_lookback): i]  # excludes day i
            cov = np.cov(window, rowvar=False)
            cov = np.atleast_2d(cov)
            vols = np.sqrt(np.clip(np.diag(cov), 0.0, None))
            w = _alloc_weights(cov, vols, pcfg.method)

            # Portfolio vol target: scale so annualised book vol ~ target. With
            # max_leverage > 1 this can scale UP toward the risk budget (the
            # Phase-4 return lever); it never exceeds the cap.
            port_var = float(w @ cov @ w)
            port_vol_annual = np.sqrt(max(port_var, 0.0) * _TRADING_DAYS)
            scale = pcfg.target_volatility / port_vol_annual if port_vol_annual > 0 else 1.0
            scale = float(np.clip(scale, pcfg.min_scale, pcfg.max_leverage))
            target_alloc = w * scale
            last_rebalance = i
            rebalance_dates.append(dates[i])

        # 3. Daily drawdown circuit-breaker -> the allocation we actually hold.
        dd_mult = (
            _dd_scale(drawdown, rcfg.dd_start, rcfg.dd_full, rcfg.dd_min_exposure)
            if pcfg.dd_derisk else 1.0
        )
        new_held = target_alloc * dd_mult

        # 4. Charge turnover on the change in ACTUAL holdings (captures both
        #    rebalances and de-risking/re-risking trades — the honest cost of a
        #    dynamic circuit-breaker).
        turnover = float(np.abs(new_held - held_alloc).sum())
        if turnover > 0:
            equity *= (1.0 - turnover * cost_rate)
        held_alloc = new_held

        eq_curve.append(equity)
        gross_series.append(float(held_alloc.sum()))
        alloc_rows.append({names[j]: float(held_alloc[j]) for j in range(n)})

    equity_s = pd.Series(eq_curve, index=dates, name="equity")
    returns_s = equity_s.pct_change().fillna(0.0)
    gross_s = pd.Series(gross_series, index=dates, name="gross_exposure")
    alloc_df = pd.DataFrame(alloc_rows, index=dates)

    metrics = compute_metrics(
        equity_s,
        risk_free_rate=cfg.backtest.risk_free_rate,
        gross_exposure=gross_s,
    )

    return CombinedResult(
        equity=equity_s,
        returns=returns_s,
        gross_exposure=gross_s,
        sleeve_weights=alloc_df,
        metrics=metrics,
        sleeve_returns=sret,
        rebalance_dates=rebalance_dates,
    )


# ===========================================================================
# Live target weights (Phase 5) — the SINGLE source of truth for trading
# ===========================================================================
@dataclass
class LiveAllocation:
    """Today's combined ETF target weights for live/paper execution."""

    as_of: pd.Timestamp
    weights: Dict[str, float]        # combined ETF weights (residual = cash)
    sleeve_alloc: Dict[str, float]   # capital allocation per sleeve
    gross_exposure: float
    vol_scale: float
    cash_weight: float


def live_target_weights(
    prices: pd.DataFrame,
    cfg: ETFConfig,
    sleeves: Sequence[Sleeve],
    *,
    current_drawdown: float = 0.0,
) -> LiveAllocation:
    """Compute **today's** combined ETF target weights with the *same* allocation
    logic as :func:`run_combined_backtest`.

    This eliminates backtest/live divergence: the live book is the combiner
    evaluated at the most recent bar. Each sleeve produces its current target
    weights; the cross-sleeve capital allocation comes from the trailing
    covariance window (ending at the latest complete bar), is scaled to the vol
    target and clipped to the leverage cap, then optionally de-risked by the
    book drawdown circuit-breaker (``current_drawdown`` supplied by the live
    runner from broker-tracked equity).

    No look-ahead: the covariance and sleeve signals use only data up to and
    including the last available row, exactly as the backtest does at a
    rebalance.
    """
    cfg.validate()
    pcfg = cfg.portfolio

    # 1. Each sleeve's current target ETF weights (as of the last price row).
    sleeve_targets = {s.name: s.target_weights(prices) for s in sleeves}

    # 2. Cross-sleeve capital allocation from the trailing covariance of the
    #    sleeve return streams (same estimator the backtest rebalance uses).
    cols: Dict[str, pd.Series] = {}
    for s in sleeves:
        cols[s.name] = backtest_sleeve(prices, s, cfg).returns
    names = [s.name for s in sleeves]
    sret = pd.DataFrame(cols).dropna()

    if sret.shape[0] >= 2:
        sret = sret[names]
        window = sret.values[-pcfg.cov_lookback:]
        cov = np.atleast_2d(np.cov(window, rowvar=False))
        vols = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        w = _alloc_weights(cov, vols, pcfg.method)
        port_var = float(w @ cov @ w)
        port_vol_annual = np.sqrt(max(port_var, 0.0) * _TRADING_DAYS)
        scale = pcfg.target_volatility / port_vol_annual if port_vol_annual > 0 else 1.0
        scale = float(np.clip(scale, pcfg.min_scale, pcfg.max_leverage))
    else:
        w = np.full(len(names), 1.0 / len(names))
        scale = 1.0

    dd_mult = (
        _dd_scale(current_drawdown, cfg.risk.dd_start, cfg.risk.dd_full, cfg.risk.dd_min_exposure)
        if pcfg.dd_derisk else 1.0
    )
    alloc = w * scale * dd_mult

    # 3. Combine into a single ETF target book: combined_w[sym] = sum_i alloc_i * sleeve_w_i[sym].
    combined: Dict[str, float] = {}
    for i, name in enumerate(names):
        for sym, sw in sleeve_targets[name].items():
            combined[sym] = combined.get(sym, 0.0) + float(alloc[i]) * float(sw)
    combined = {k: v for k, v in combined.items() if abs(v) > 1e-6}

    gross = float(sum(abs(v) for v in combined.values()))
    return LiveAllocation(
        as_of=prices.index[-1],
        weights=combined,
        sleeve_alloc={names[i]: float(alloc[i]) for i in range(len(names))},
        gross_exposure=gross,
        vol_scale=float(scale),
        cash_weight=max(0.0, 1.0 - gross),
    )

"""
etf/backtest.py
===============
Event-driven (daily) backtester for the ETF target-weight engine.

Timing convention (no look-ahead)
---------------------------------
Weights are decided using prices through the *close* of the rebalance day and
take effect the **next** trading day. The return earned on day *t* is always
driven by weights chosen on a strictly earlier day. Transaction costs are
charged at the moment weights change. This mirrors how the live trader behaves
(decide on today's close, send orders, fills next session), keeping backtest
and live logic aligned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from etf.config import ETFConfig
from etf.metrics import ETFMetrics, compute_metrics
from etf.strategy import apply_drawdown_overlay, compute_target_weights, enforce_gross_cap

logger = logging.getLogger("etf.backtest")


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    gross_exposure: pd.Series
    turnover: pd.Series
    weights_history: pd.DataFrame
    metrics: ETFMetrics
    benchmark_equity: Optional[pd.Series] = None
    benchmark_metrics: Optional[ETFMetrics] = None
    rebalance_dates: List[pd.Timestamp] = field(default_factory=list)


def _warmup_bars(cfg: ETFConfig) -> int:
    s = cfg.signal
    return max(s.trend_sma, s.ts_momentum_long, max(s.momentum_lookbacks)) + 1


def _apply_min_rebalance_delta(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    *,
    equity: float,
    min_rebalance_delta: float,
) -> Dict[str, float]:
    """Apply live-like min-notional rebalance threshold to target weights.

    For each symbol, if the absolute notional trade implied by the weight change
    is below ``min_rebalance_delta * equity``, retain the current weight.
    """
    threshold = max(0.0, float(min_rebalance_delta)) * max(0.0, float(equity))
    if threshold <= 0.0:
        return {k: float(v) for k, v in target_weights.items() if abs(v) > 1e-12}

    effective: Dict[str, float] = {}
    for sym in set(current_weights) | set(target_weights):
        cur_w = float(current_weights.get(sym, 0.0))
        tgt_w = float(target_weights.get(sym, 0.0))
        trade_notional = abs(tgt_w - cur_w) * equity
        w = cur_w if trade_notional < threshold else tgt_w
        if abs(w) > 1e-12:
            effective[sym] = w
    return effective


def run_backtest(
    prices: pd.DataFrame,
    cfg: ETFConfig,
    *,
    weight_fn: Optional[Callable[[pd.DataFrame], Dict[str, float]]] = None,
    rebalance_every: Optional[int] = None,
    apply_dd: bool = True,
    warmup: Optional[int] = None,
) -> BacktestResult:
    """Run an ETF strategy over ``prices`` (wide adjusted-close frame).

    By default this runs the built-in trend/momentum strategy
    (:func:`compute_target_weights` + drawdown overlay) — identical to the
    original behavior. To backtest an arbitrary sleeve, pass ``weight_fn``: a
    callable that receives the price frame sliced up to (and including) the
    decision date and returns ``{symbol: target_weight}`` for the risky sleeve
    (the residual is parked in cash). ``rebalance_every``/``warmup`` override the
    config defaults (e.g. daily rebalancing for a mean-reversion sleeve), and
    ``apply_dd=False`` disables the portfolio drawdown overlay when measuring a
    single sleeve's raw edge.
    """
    cfg.validate()
    prices = prices.sort_index().dropna(how="all")
    if prices.empty:
        raise ValueError("No price data supplied to backtest")

    dates = prices.index
    rets = prices.pct_change().fillna(0.0)

    cash_asset = cfg.cash_asset
    has_cash = cash_asset in prices.columns
    cash_rets = rets[cash_asset] if has_cash else pd.Series(0.0, index=dates)

    warmup = _warmup_bars(cfg) if warmup is None else warmup
    rebalance_every = cfg.execution.rebalance_every if rebalance_every is None else rebalance_every
    cost_rate = (cfg.execution.commission_bps + cfg.execution.slippage_bps) / 1e4

    equity = cfg.backtest.initial_capital
    peak = equity
    active_weights: Dict[str, float] = {}
    active_cash = 1.0

    eq_curve: List[float] = []
    gross_series: List[float] = []
    turn_series: List[float] = []
    weights_rows: List[Dict[str, float]] = []
    rebalance_dates: List[pd.Timestamp] = []
    last_rebalance = -10**9

    for i, date in enumerate(dates):
        # 1. Earn the day's return on the currently-active weights.
        if i > 0:
            port_ret = active_cash * float(cash_rets.iloc[i])
            for sym, w in active_weights.items():
                if sym in rets.columns:
                    port_ret += w * float(rets[sym].iloc[i])
            equity *= (1.0 + port_ret)

        peak = max(peak, equity)
        drawdown = 1.0 - equity / peak if peak > 0 else 0.0

        # 2. Decide whether to rebalance (effective next day).
        turnover_today = 0.0
        if i >= warmup and (i - last_rebalance) >= rebalance_every:
            sliced = prices.iloc[: i + 1]
            if weight_fn is None:
                decision = compute_target_weights(sliced, cfg)
                if apply_dd:
                    decision = apply_drawdown_overlay(decision, drawdown, cfg)
                new_weights = decision.weights
                new_cash = decision.cash_weight
            else:
                raw = weight_fn(sliced) or {}
                new_weights = {k: float(v) for k, v in raw.items() if abs(v) > 1e-12}
                new_cash = max(0.0, 1.0 - float(sum(new_weights.values())))

            # Mirror live execution: tiny notional drifts are ignored.
            effective_weights = _apply_min_rebalance_delta(
                active_weights,
                new_weights,
                equity=equity,
                min_rebalance_delta=cfg.execution.min_rebalance_delta,
            )
            # Strictly enforce the gross-leverage cap on the actually-held book.
            # The min-delta filter mixes retained-old and adopted-new weights and
            # can push gross above the cap even though both books individually
            # satisfy it; trim proportionally so the held exposure never breaches
            # max_gross_leverage (matches the live-broker enforcement).
            effective_weights = enforce_gross_cap(
                effective_weights, cfg.risk.max_gross_leverage
            )

            # L1 turnover across the union of old & effective new positions.
            syms = set(active_weights) | set(effective_weights)
            turnover_today = sum(
                abs(effective_weights.get(s, 0.0) - active_weights.get(s, 0.0))
                for s in syms
            )
            # Charge cost on the traded notional now (reduces tomorrow's base).
            equity *= (1.0 - turnover_today * cost_rate)

            active_weights = effective_weights
            active_cash = max(0.0, 1.0 - float(sum(active_weights.values())))
            last_rebalance = i
            rebalance_dates.append(date)


        eq_curve.append(equity)
        # Gross exposure = sum of |weights| (correct for long/short sleeves;
        # identical to the net sum when every weight is non-negative).
        gross_series.append(float(sum(abs(w) for w in active_weights.values())))
        turn_series.append(turnover_today)
        weights_rows.append(dict(active_weights))

    equity_s = pd.Series(eq_curve, index=dates, name="equity")
    returns_s = equity_s.pct_change().fillna(0.0)
    gross_s = pd.Series(gross_series, index=dates, name="gross_exposure")
    turn_s = pd.Series(turn_series, index=dates, name="turnover")
    weights_df = pd.DataFrame(weights_rows, index=dates).fillna(0.0)

    # Benchmark (buy & hold) for alpha/beta.
    bench_equity = None
    bench_metrics = None
    bench_rets = None
    if cfg.benchmark in prices.columns:
        bench_px = prices[cfg.benchmark].dropna()
        bench_equity = (cfg.backtest.initial_capital * bench_px / bench_px.iloc[0]).reindex(dates).ffill()
        bench_rets = bench_equity.pct_change()
        bench_metrics = compute_metrics(bench_equity, risk_free_rate=cfg.backtest.risk_free_rate)

    metrics = compute_metrics(
        equity_s,
        risk_free_rate=cfg.backtest.risk_free_rate,
        benchmark_returns=bench_rets,
        gross_exposure=gross_s,
        turnover=turn_s,
    )

    return BacktestResult(
        equity=equity_s,
        returns=returns_s,
        gross_exposure=gross_s,
        turnover=turn_s,
        weights_history=weights_df,
        metrics=metrics,
        benchmark_equity=bench_equity,
        benchmark_metrics=bench_metrics,
        rebalance_dates=rebalance_dates,
    )

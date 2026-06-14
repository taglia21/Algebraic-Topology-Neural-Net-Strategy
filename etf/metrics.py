"""
etf/metrics.py
==============
Self-contained performance metrics for the ETF engine.

Kept independent of the equities backtest metrics module so the ETF engine has
zero coupling and its tests run fully offline. Definitions follow standard
institutional conventions (252 trading days, daily-return aggregation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


@dataclass
class ETFMetrics:
    total_return: float = 0.0
    cagr: float = 0.0
    annual_volatility: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    avg_gross_exposure: float = 0.0
    turnover_annual: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    n_periods: int = 0
    extras: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        d = {k: v for k, v in self.__dict__.items() if k != "extras"}
        d.update(self.extras)
        return d


def max_drawdown(equity: pd.Series) -> float:
    """Return max drawdown as a negative fraction (e.g. -0.23)."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def compute_metrics(
    equity: pd.Series,
    *,
    risk_free_rate: float = 0.03,
    benchmark_returns: Optional[pd.Series] = None,
    gross_exposure: Optional[pd.Series] = None,
    turnover: Optional[pd.Series] = None,
) -> ETFMetrics:
    """Compute performance metrics from a daily equity curve."""
    equity = equity.dropna()
    if len(equity) < 2:
        return ETFMetrics(n_periods=len(equity))

    rets = equity.pct_change().dropna()
    n = len(rets)
    years = n / _TRADING_DAYS

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    ann_vol = float(rets.std() * np.sqrt(_TRADING_DAYS))

    rf_daily = risk_free_rate / _TRADING_DAYS
    excess = rets - rf_daily
    sharpe = float(excess.mean() / rets.std() * np.sqrt(_TRADING_DAYS)) if rets.std() > 0 else 0.0

    downside = rets[rets < 0]
    dstd = downside.std()
    sortino = float(excess.mean() / dstd * np.sqrt(_TRADING_DAYS)) if dstd and dstd > 0 else 0.0

    mdd = max_drawdown(equity)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0

    wins = rets[rets > 0]
    losses = rets[rets < 0]
    win_rate = float(len(wins) / n) if n > 0 else 0.0
    gross_win = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else float("inf")

    var_95 = float(np.percentile(rets, 5))
    tail = rets[rets <= var_95]
    cvar_95 = float(tail.mean()) if len(tail) else var_95

    alpha = beta = 0.0
    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(rets.index).dropna()
        common = rets.reindex(bench.index).dropna()
        bench = bench.reindex(common.index)
        if len(common) > 2 and bench.var() > 0:
            beta = float(np.cov(common, bench)[0, 1] / np.var(bench))
            alpha = float((common.mean() - beta * bench.mean()) * _TRADING_DAYS)

    avg_gross = float(gross_exposure.mean()) if gross_exposure is not None and len(gross_exposure) else 0.0
    turn_ann = 0.0
    if turnover is not None and len(turnover):
        turn_ann = float(turnover.sum() / years) if years > 0 else float(turnover.sum())

    return ETFMetrics(
        total_return=total_return,
        cagr=cagr,
        annual_volatility=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        calmar=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        var_95=var_95,
        cvar_95=cvar_95,
        avg_gross_exposure=avg_gross,
        turnover_annual=turn_ann,
        alpha=alpha,
        beta=beta,
        n_periods=n,
    )

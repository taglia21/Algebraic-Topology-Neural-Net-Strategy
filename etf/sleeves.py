"""
etf/sleeves.py
==============
Sleeve framework for the multi-strategy ETF engine (Phase 2+).

A *sleeve* is a self-contained return source: given price history sliced up to
(and including) a decision date, it returns target portfolio weights for the
ETFs it wants to hold (the residual is cash). Each sleeve manages its own
internal risk; the Phase 3 portfolio combiner allocates capital *across* sleeves
by their risk contribution.

The profit thesis (validated in Phase 0): a single ETF sleeve has only a modest
Sharpe, and tuning it is overfitting (PBO > 50%). The durable path to an
institutional Sharpe is to **stack low-correlation sleeves**. This module
provides:

- ``Sleeve``            : the common interface (name, cadence, warmup, weights).
- ``TrendMomentumSleeve`` (Sleeve A): wraps the existing trend/momentum engine.
- ``MeanReversionSleeve`` (Sleeve B): Connors-style RSI(2) dip-buying restricted
  to broad equity ETFs in a long-term uptrend — structurally low-correlation to
  trend-following.
- ``backtest_sleeve``   : run any sleeve through the shared backtester.

No look-ahead: every sleeve computes signals from ``prices`` whose last row is
the decision date; the backtester applies the resulting weights the next day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from etf.backtest import BacktestResult, run_backtest
from etf.config import ETFConfig
from etf.strategy import apply_drawdown_overlay, compute_target_weights

logger = logging.getLogger("etf.sleeves")

_TRADING_DAYS = 252


# ===========================================================================
# Sleeve interface
# ===========================================================================
@runtime_checkable
class Sleeve(Protocol):
    name: str
    rebalance_every: int
    warmup: int

    def target_weights(self, prices_asof: pd.DataFrame) -> Dict[str, float]:
        """Return {symbol: weight} for the risky sleeve at the last row's date."""
        ...


# ===========================================================================
# Volatility-managed overlay — composable Moreira–Muir conditional vol timing
# ===========================================================================
@dataclass
class VolManagedSleeve:
    """Wrap any sleeve and time its exposure inversely to recent realised vol.

    Implements Moreira & Muir (2017): each decision bar, measure the realised
    volatility of the inner sleeve's *actual held basket* over a short trailing
    window and scale the whole basket toward a fixed annual vol target. High
    recent vol -> cut exposure; calm -> add (capped). Because volatility is
    persistent and forecastable while returns are not, this harvests a timing
    premium and truncates the left tail, on top of the inner sleeve's own slow
    unconditional vol target.

    Fully causal: the basket weights come from the inner sleeve (which only saw
    ``prices_asof``), and realised vol is measured on the trailing returns of
    those same prices — no future data. When the overlay is disabled or the
    window is too short, it returns the inner weights unchanged.
    """

    inner: Sleeve
    cfg: ETFConfig
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.inner.name}_vm"

    @property
    def rebalance_every(self) -> int:
        return self.inner.rebalance_every

    @property
    def warmup(self) -> int:
        return max(self.inner.warmup, self.cfg.vol_managed.realized_window + 1)

    def _scale(self, prices_asof: pd.DataFrame, weights: Dict[str, float]) -> float:
        """Conditional vol-timing multiplier for the held basket (>= 0)."""
        vm = self.cfg.vol_managed
        syms = [s for s in weights if s in prices_asof.columns and abs(weights[s]) > 0]
        if not syms:
            return 1.0
        rets = prices_asof[syms].pct_change().tail(vm.realized_window)
        if rets.shape[0] < 2:
            return 1.0
        wv = np.array([weights[s] for s in syms], dtype=float)
        basket = rets.to_numpy() @ wv                    # daily basket returns
        basket = basket[np.isfinite(basket)]
        if basket.size < 2:
            return 1.0
        realized_daily = float(np.std(basket, ddof=0))
        if not np.isfinite(realized_daily) or realized_daily <= 0:
            return 1.0
        target_daily = vm.target_vol_annual / np.sqrt(_TRADING_DAYS)
        scale = target_daily / realized_daily
        return float(np.clip(scale, vm.min_scale, vm.max_scale))

    def target_weights(self, prices_asof: pd.DataFrame) -> Dict[str, float]:
        weights = self.inner.target_weights(prices_asof)
        if not weights or not self.cfg.vol_managed.enabled:
            return weights
        scale = self._scale(prices_asof, weights)
        return {s: float(w * scale) for s, w in weights.items()}


# ===========================================================================
# Sleeve A — Trend / time-series momentum (wraps the existing engine)
# ===========================================================================
@dataclass
class TrendMomentumSleeve:
    """The Phase 0/1 trend+cross-sectional-momentum strategy as a sleeve.

    ``apply_dd`` keeps the per-sleeve drawdown overlay off by default, because
    drawdown control is a *portfolio-level* concern handled by the combiner;
    measuring the raw sleeve edge should not be distorted by it.
    """

    cfg: ETFConfig
    apply_dd: bool = False
    name: str = "trend_momentum"

    @property
    def rebalance_every(self) -> int:
        return self.cfg.execution.rebalance_every

    @property
    def warmup(self) -> int:
        s = self.cfg.signal
        return max(s.trend_sma, s.ts_momentum_long, max(s.momentum_lookbacks)) + 1

    def target_weights(self, prices_asof: pd.DataFrame) -> Dict[str, float]:
        decision = compute_target_weights(prices_asof, self.cfg)
        if self.apply_dd:
            # Overlay needs an equity drawdown; at sleeve level we don't track
            # it here, so this path is only used when explicitly requested.
            decision = apply_drawdown_overlay(decision, 0.0, self.cfg)
        return decision.weights


# ===========================================================================
# Sleeve B — Short-horizon mean reversion (RSI-2 dip buying)
# ===========================================================================
def rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI computed causally (uses only past/current observations)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing == EWM with alpha = 1/period, no look-ahead.
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # When avg_loss == 0 (no down moves), RSI is 100 by definition.
    out = out.where(avg_loss.ne(0.0), 100.0)
    return out


@dataclass
class MeanReversionSleeve:
    """Connors-style RSI(2) oversold dip-buying on broad equity ETFs.

    Stateless rule (recomputed each day, so it is naturally causal):
      hold symbol  <=>  price > SMA(trend_sma)  AND  RSI(period) < rsi_oversold
    Among qualifying names, keep the most-oversold up to ``max_positions``,
    weight them inverse-vol, cap per-name, and deploy ``deploy_fraction`` of
    capital (rest cash). When nothing qualifies, the sleeve is fully in cash.
    """

    cfg: ETFConfig
    name: str = "mean_reversion"
    rebalance_every: int = 1  # mean reversion must be evaluated daily

    @property
    def warmup(self) -> int:
        return max(self.cfg.mean_reversion.trend_sma, self.cfg.mean_reversion.rsi_period) + 1

    def target_weights(self, prices_asof: pd.DataFrame) -> Dict[str, float]:
        mr = self.cfg.mean_reversion
        candidates: Dict[str, float] = {}  # symbol -> RSI (lower = more oversold)
        for sym in mr.universe:
            if sym not in prices_asof.columns:
                continue
            series = prices_asof[sym].dropna()
            if len(series) < max(mr.trend_sma, mr.rsi_period) + 1:
                continue
            price = series.iloc[-1]
            sma = series.iloc[-mr.trend_sma:].mean()
            if price <= sma:
                continue  # only buy dips inside a long-term uptrend
            rsi_val = rsi(series, mr.rsi_period).iloc[-1]
            if np.isfinite(rsi_val) and rsi_val < mr.rsi_oversold:
                candidates[sym] = float(rsi_val)

        if not candidates:
            return {}

        # Keep the most-oversold up to max_positions.
        ranked = sorted(candidates.items(), key=lambda kv: kv[1])
        selected = [s for s, _ in ranked[: mr.max_positions]]

        # Inverse-vol weighting within the sleeve.
        rets = prices_asof[selected].pct_change().tail(mr.vol_lookback)
        vol = rets.std().replace(0.0, np.nan)
        inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan).dropna()
        if inv.empty:
            weights = pd.Series(1.0 / len(selected), index=selected)
        else:
            weights = inv / inv.sum()

        # Per-name cap + renormalise, then scale by deploy_fraction.
        weights = weights.clip(upper=mr.max_position_weight)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        weights = weights * mr.deploy_fraction
        return {s: float(w) for s, w in weights.items() if w > 1e-12}


# ===========================================================================
# Sleeve C — Defensive carry (absolute momentum on non-equity assets)
# ===========================================================================
@dataclass
class DefensiveCarrySleeve:
    """Antonacci-style absolute (time-series) momentum on defensive assets only.

    Stateless rule (recomputed each rebalance, so naturally causal):
      hold symbol  <=>  price > SMA(trend_sma)  AND  absolute momentum > 0
    where absolute momentum is the skip-month return over ``momentum_lookback``.
    Qualifying names are inverse-vol weighted, per-name capped, and scaled by
    ``deploy_fraction``. When nothing qualifies the sleeve is fully in cash.

    Because the universe is equity-free, this sleeve is structurally
    non-equity-beta: it earns from trending duration/credit/gold during
    flights-to-safety and equity dead zones.
    """

    cfg: ETFConfig
    name: str = "defensive_carry"

    @property
    def rebalance_every(self) -> int:
        # Monthly cadence matches the existing trend sleeve and keeps turnover
        # (and cost) low for these slow-moving defensive trends.
        return self.cfg.execution.rebalance_every

    @property
    def warmup(self) -> int:
        dc = self.cfg.defensive_carry
        return max(dc.trend_sma, dc.momentum_lookback + dc.momentum_skip) + 1

    def target_weights(self, prices_asof: pd.DataFrame) -> Dict[str, float]:
        dc = self.cfg.defensive_carry
        need = max(dc.trend_sma, dc.momentum_lookback + dc.momentum_skip) + 1
        selected: List[str] = []
        for sym in dc.universe:
            if sym not in prices_asof.columns:
                continue
            series = prices_asof[sym].dropna()
            if len(series) < need:
                continue
            price = series.iloc[-1]
            sma = series.iloc[-dc.trend_sma:].mean()
            if price <= sma:
                continue  # trend gate
            # Skip-month absolute momentum: return from t-(LB+skip) to t-skip.
            p_then = series.iloc[-(dc.momentum_lookback + dc.momentum_skip + 1)]
            p_skip = series.iloc[-(dc.momentum_skip + 1)] if dc.momentum_skip > 0 else price
            abs_mom = (p_skip / p_then) - 1.0
            if abs_mom > 0:
                selected.append(sym)

        if not selected:
            return {}
        if len(selected) > dc.max_positions:
            # Keep the strongest by recent (skip-month) momentum.
            mom = {}
            for sym in selected:
                series = prices_asof[sym].dropna()
                p_then = series.iloc[-(dc.momentum_lookback + dc.momentum_skip + 1)]
                p_skip = series.iloc[-(dc.momentum_skip + 1)] if dc.momentum_skip > 0 else series.iloc[-1]
                mom[sym] = (p_skip / p_then) - 1.0
            selected = sorted(mom, key=mom.get, reverse=True)[: dc.max_positions]

        rets = prices_asof[selected].pct_change().tail(dc.vol_lookback)
        vol = rets.std().replace(0.0, np.nan)
        inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan).dropna()
        if inv.empty:
            weights = pd.Series(1.0 / len(selected), index=selected)
        else:
            weights = inv / inv.sum()

        weights = weights.clip(upper=dc.max_position_weight)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        weights = weights * dc.deploy_fraction
        return {s: float(w) for s, w in weights.items() if w > 1e-12}


# ===========================================================================
# Sleeve D — Cross-sectional relative strength (dollar-neutral long/short)
# ===========================================================================
@dataclass
class CrossSectionalSleeve:
    """Dollar-neutral long/short cross-sectional relative-strength sleeve.

    Each rebalance (causal — uses only data up to the decision date):
      1. Score every universe name by risk-adjusted skip-month momentum
         (return over ``momentum_lookback`` skipping the last ``momentum_skip``
         days, divided by trailing vol).
      2. Long the ``top_k`` strongest, short the ``bottom_k`` weakest.
      3. Size each side to half the ``gross_target`` (so longs ≈ +0.5·gross and
         shorts ≈ −0.5·gross → ~zero net dollar exposure), inverse-vol within
         a side, per-name capped.

    The market-neutral construction strips equity beta, isolating the
    relative-strength spread — structurally low-correlation to the long-only
    sleeves. The backtester's cash leg models collateral earning the cash rate.
    """

    cfg: ETFConfig
    name: str = "cross_sectional"

    @property
    def rebalance_every(self) -> int:
        return self.cfg.execution.rebalance_every

    @property
    def warmup(self) -> int:
        cs = self.cfg.cross_sectional
        return cs.momentum_lookback + cs.momentum_skip + 1

    def _scores(self, prices_asof: pd.DataFrame) -> pd.Series:
        cs = self.cfg.cross_sectional
        need = cs.momentum_lookback + cs.momentum_skip + 1
        scores: Dict[str, float] = {}
        for sym in cs.universe:
            if sym not in prices_asof.columns:
                continue
            series = prices_asof[sym].dropna()
            if len(series) < need:
                continue
            p_then = series.iloc[-(cs.momentum_lookback + cs.momentum_skip + 1)]
            p_skip = series.iloc[-(cs.momentum_skip + 1)] if cs.momentum_skip > 0 else series.iloc[-1]
            mom = (p_skip / p_then) - 1.0
            vol = series.pct_change().tail(cs.vol_lookback).std()
            if not np.isfinite(vol) or vol <= 0:
                continue
            scores[sym] = mom / vol  # risk-adjusted momentum
        return pd.Series(scores, dtype=float)

    def _side_weights(self, prices_asof: pd.DataFrame, names: List[str],
                      signed_notional: float) -> Dict[str, float]:
        """Inverse-vol weights for one side, scaled to ``signed_notional``."""
        cs = self.cfg.cross_sectional
        if not names:
            return {}
        rets = prices_asof[names].pct_change().tail(cs.vol_lookback)
        vol = rets.std().replace(0.0, np.nan)
        inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan).dropna()
        if inv.empty:
            w = pd.Series(1.0 / len(names), index=names)
        else:
            w = inv / inv.sum()
        w = w.clip(upper=cs.max_position_weight)
        if w.sum() > 0:
            w = w / w.sum()
        w = w * signed_notional
        return {s: float(v) for s, v in w.items()}

    def target_weights(self, prices_asof: pd.DataFrame) -> Dict[str, float]:
        cs = self.cfg.cross_sectional
        scores = self._scores(prices_asof)
        if len(scores) < cs.top_k + cs.bottom_k:
            return {}  # not enough names to form both legs -> flat (cash)
        ranked = scores.sort_values(ascending=False)
        longs = list(ranked.index[: cs.top_k])
        shorts = list(ranked.index[-cs.bottom_k:])
        side = 0.5 * cs.gross_target
        weights = self._side_weights(prices_asof, longs, +side)
        weights.update(self._side_weights(prices_asof, shorts, -side))
        return {s: w for s, w in weights.items() if abs(w) > 1e-12}


# ===========================================================================
# Sleeve E — Turn-of-month calendar seasonality (long broad equity in-window)
# ===========================================================================
def _tom_in_window(decision_date: pd.Timestamp, tdom_from_start: int,
                   first_trading_days: int, last_calendar_days: int) -> bool:
    """Is the NEXT session in the turn-of-month hold window?

    Fully causal — uses only the decision date and its trading-day-of-month
    counted from the month start (both known at decision time):

    - ``tdom_from_start <= first_trading_days``  captures the early-month leg
      (decision on +1..+N -> hold +2..+(N+1)); a one-day execution phase that
      the multi-day window absorbs.
    - ``day >= days_in_month - last_calendar_days + 1`` captures the month-end
      "-1" leg from the calendar date alone (no future index access).
    """
    import calendar
    if first_trading_days > 0 and tdom_from_start <= first_trading_days:
        return True
    if last_calendar_days > 0:
        days_in_month = calendar.monthrange(decision_date.year, decision_date.month)[1]
        if decision_date.day >= days_in_month - last_calendar_days + 1:
            return True
    return False


@dataclass
class TurnOfMonthSleeve:
    """Long broad-equity ETFs only inside the turn-of-month window, else cash.

    The signal is the CALENDAR (see :class:`etf.config.SeasonalityConfig`), so
    the sleeve is orthogonal-by-construction to the price-driven sleeves. A
    long-term trend gate (price > SMA(trend_sma)) keeps it out of equities in
    sustained downtrends. Among qualifying names it weights inverse-vol, caps per
    name, and deploys ``deploy_fraction`` of capital (rest cash).
    """

    cfg: ETFConfig
    name: str = "seasonality_tom"
    rebalance_every: int = 1  # calendar signal must be evaluated daily

    @property
    def warmup(self) -> int:
        se = self.cfg.seasonality
        return max(se.trend_sma, se.vol_lookback) + 1

    def target_weights(self, prices_asof: pd.DataFrame) -> Dict[str, float]:
        se = self.cfg.seasonality
        if prices_asof.empty:
            return {}
        decision_date = prices_asof.index[-1]
        # Trading-day-of-month counted causally from the slice: how many rows in
        # this (year, month) up to and including the decision date.
        idx = prices_asof.index
        same_month = (idx.year == decision_date.year) & (idx.month == decision_date.month)
        tdom_from_start = int(same_month.sum())

        if not _tom_in_window(decision_date, tdom_from_start,
                              se.first_trading_days, se.last_calendar_days):
            return {}  # outside the ToM window -> fully in cash

        selected: List[str] = []
        for sym in se.universe:
            if sym not in prices_asof.columns:
                continue
            series = prices_asof[sym].dropna()
            if len(series) < se.trend_sma + 1:
                continue
            price = series.iloc[-1]
            sma = series.iloc[-se.trend_sma:].mean()
            if price > sma:  # trend gate: only deploy in long-term uptrends
                selected.append(sym)

        if not selected:
            return {}

        rets = prices_asof[selected].pct_change().tail(se.vol_lookback)
        vol = rets.std().replace(0.0, np.nan)
        inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan).dropna()
        if inv.empty:
            weights = pd.Series(1.0 / len(selected), index=selected)
        else:
            weights = inv / inv.sum()

        weights = weights.clip(upper=se.max_position_weight)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        weights = weights * se.deploy_fraction
        return {s: float(w) for s, w in weights.items() if w > 1e-12}


# ===========================================================================
# Sleeve backtest helper
# ===========================================================================
def backtest_sleeve(prices: pd.DataFrame, sleeve: Sleeve, cfg: ETFConfig) -> BacktestResult:
    """Run a single sleeve through the shared backtester (no DD overlay)."""
    return run_backtest(
        prices,
        cfg,
        weight_fn=sleeve.target_weights,
        rebalance_every=sleeve.rebalance_every,
        apply_dd=False,
        warmup=sleeve.warmup,
    )


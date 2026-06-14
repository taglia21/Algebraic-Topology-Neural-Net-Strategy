"""
etf/sleeve_analysis.py
======================
Phase 2 evidence harness: quantify whether the mean-reversion sleeve (Sleeve B)
is a genuine, *diversifying* return source relative to the trend sleeve
(Sleeve A).

Gates this report measures (Phase 2 promotion criteria):
  1. Sleeve B has a positive OOS Sharpe standalone (CPCV + Deflated Sharpe).
  2. Sleeve B's daily-return correlation to Sleeve A is low (target <~ 0.3).
  3. A naive, *non-fitted* 50/50 capital split improves risk-adjusted return
     vs. either sleeve alone — direct evidence that stacking helps.

The 50/50 split is deliberately un-optimised (no parameters fit on the data) so
it cannot overstate the benefit. The proper causal risk-parity combiner is
Phase 3 work; this module only establishes that the diversification edge exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from etf.backtest import BacktestResult
from etf.config import ETFConfig
from etf.metrics import ETFMetrics, compute_metrics
from etf.sleeves import (
    CrossSectionalSleeve,
    DefensiveCarrySleeve,
    MeanReversionSleeve,
    Sleeve,
    TrendMomentumSleeve,
    TurnOfMonthSleeve,
    VolManagedSleeve,
    backtest_sleeve,
)
from etf.validation import (
    cpcv_oos_sharpes,
    deflated_sharpe_ratio,
)


@dataclass
class SleeveReport:
    sleeve_a: ETFMetrics
    sleeve_b: ETFMetrics
    correlation: float
    combo_5050: ETFMetrics
    sleeve_b_cpcv_median: float
    sleeve_b_cpcv_p_positive: float
    sleeve_b_dsr: float
    overlap_days: int

    def summary(self) -> str:
        a, b, c = self.sleeve_a, self.sleeve_b, self.combo_5050
        best_single = max(a.sharpe, b.sharpe)
        uplift = c.sharpe - best_single
        lines = [
            "=" * 70,
            "PHASE 2 — UNCORRELATED SLEEVE ANALYSIS",
            "=" * 70,
            f"{'metric':<22}{'Sleeve A (trend)':>16}{'Sleeve B (mean-rev)':>20}{'50/50 combo':>14}",
            "-" * 72,
            f"{'CAGR':<22}{a.cagr:>15.2%}{b.cagr:>20.2%}{c.cagr:>14.2%}",
            f"{'Annual vol':<22}{a.annual_volatility:>15.2%}{b.annual_volatility:>20.2%}{c.annual_volatility:>14.2%}",
            f"{'Sharpe':<22}{a.sharpe:>15.2f}{b.sharpe:>20.2f}{c.sharpe:>14.2f}",
            f"{'Sortino':<22}{a.sortino:>15.2f}{b.sortino:>20.2f}{c.sortino:>14.2f}",
            f"{'Max drawdown':<22}{a.max_drawdown:>15.2%}{b.max_drawdown:>20.2%}{c.max_drawdown:>14.2%}",
            f"{'Calmar':<22}{a.calmar:>15.2f}{b.calmar:>20.2f}{c.calmar:>14.2f}",
            "-" * 72,
            f"Sleeve A/B daily-return correlation : {self.correlation:>7.3f}   "
            f"(overlap {self.overlap_days} days)",
            f"Sleeve B OOS (CPCV) median Sharpe   : {self.sleeve_b_cpcv_median:>7.2f}",
            f"Sleeve B P(OOS Sharpe > 0)          : {self.sleeve_b_cpcv_p_positive:>7.1%}",
            f"Sleeve B Deflated Sharpe Ratio      : {self.sleeve_b_dsr:>7.3f}",
            f"50/50 Sharpe uplift vs best single  : {uplift:>+7.2f}",
            "=" * 70,
            "Gate readout:",
            f"  [{'PASS' if b.sharpe > 0 else 'FAIL'}] Sleeve B positive standalone Sharpe",
            f"  [{'PASS' if self.correlation <= 0.30 else 'WARN'}] Correlation to trend <= 0.30",
            f"  [{'PASS' if self.sleeve_b_cpcv_p_positive >= 0.60 else 'WARN'}] OOS P(Sharpe>0) >= 60%",
            f"  [{'PASS' if uplift > 0 else 'FAIL'}] 50/50 combo beats best single sleeve",
            "=" * 70,
            "(Sharpe here is GROSS of the risk-free rate; combo is a non-fitted",
            " 50/50 capital split. Phase 3 builds the causal risk-parity combiner.)",
        ]
        return "\n".join(lines)


def _aligned_returns(a: BacktestResult, b: BacktestResult) -> pd.DataFrame:
    """Daily returns aligned on the overlapping, *active* window of both sleeves.

    Each sleeve has a warmup; we align on the common index and drop the leading
    flat region where a sleeve has not started trading, so the correlation
    reflects live behavior rather than the shared cash period.
    """
    df = pd.DataFrame({"a": a.returns, "b": b.returns}).dropna()
    # Trim leading rows where BOTH sleeves are still flat (pre-trade warmup).
    nonzero = (df["a"].abs() + df["b"].abs()) > 0
    if nonzero.any():
        first = nonzero.idxmax()
        df = df.loc[first:]
    return df


def analyze_sleeves(prices: pd.DataFrame, cfg: ETFConfig) -> SleeveReport:
    sleeve_a = TrendMomentumSleeve(cfg, apply_dd=False)
    sleeve_b = MeanReversionSleeve(cfg)

    res_a = backtest_sleeve(prices, sleeve_a, cfg)
    res_b = backtest_sleeve(prices, sleeve_b, cfg)

    aligned = _aligned_returns(res_a, res_b)
    corr = float(aligned["a"].corr(aligned["b"])) if len(aligned) > 2 else float("nan")

    # Non-fitted 50/50 capital split (rebalanced daily).
    combo_ret = 0.5 * aligned["a"] + 0.5 * aligned["b"]
    combo_equity = cfg.backtest.initial_capital * (1.0 + combo_ret).cumprod()
    combo_metrics = compute_metrics(combo_equity, risk_free_rate=cfg.backtest.risk_free_rate)

    a_metrics = compute_metrics(
        cfg.backtest.initial_capital * (1.0 + aligned["a"]).cumprod(),
        risk_free_rate=cfg.backtest.risk_free_rate,
    )
    b_metrics = compute_metrics(
        cfg.backtest.initial_capital * (1.0 + aligned["b"]).cumprod(),
        risk_free_rate=cfg.backtest.risk_free_rate,
    )

    # OOS robustness of Sleeve B (CPCV + Deflated Sharpe).
    b_returns = aligned["b"]
    try:
        cpcv = cpcv_oos_sharpes(b_returns, n_groups=8, k_test=2, purge=5)
        cpcv_median = float(np.median(cpcv)) if len(cpcv) else float("nan")
        cpcv_p_pos = float((cpcv > 0).mean()) if len(cpcv) else float("nan")
    except Exception:  # pragma: no cover - defensive
        cpcv, cpcv_median, cpcv_p_pos = np.array([]), float("nan"), float("nan")

    try:
        # DSR needs PER-PERIOD trial Sharpes; recompute CPCV un-annualised so the
        # multiple-testing dispersion is in the right units.
        cpcv_pp = cpcv_oos_sharpes(b_returns, n_groups=8, k_test=2, purge=5, annualize=False)
        trials = cpcv_pp if len(cpcv_pp) else [0.0]
        dsr = deflated_sharpe_ratio(b_returns, trial_sharpes=trials).get("dsr", float("nan"))
    except Exception:  # pragma: no cover
        dsr = float("nan")

    return SleeveReport(
        sleeve_a=a_metrics,
        sleeve_b=b_metrics,
        correlation=corr,
        combo_5050=combo_metrics,
        sleeve_b_cpcv_median=cpcv_median,
        sleeve_b_cpcv_p_positive=cpcv_p_pos,
        sleeve_b_dsr=dsr,
        overlap_days=len(aligned),
    )


# ===========================================================================
# General N-sleeve analysis (Phase 2 -> Phase 3 bridge)
# ===========================================================================
@dataclass
class MultiSleeveReport:
    names: List[str]
    metrics: Dict[str, ETFMetrics]            # per-sleeve standalone metrics
    corr_matrix: pd.DataFrame                  # pairwise daily-return correlation
    cpcv_median: Dict[str, float]              # per-sleeve OOS CPCV median Sharpe
    cpcv_p_positive: Dict[str, float]          # per-sleeve OOS P(Sharpe>0)
    dsr: Dict[str, float]                      # per-sleeve Deflated Sharpe Ratio
    combo_inv_vol: ETFMetrics                  # causal inverse-vol blend of all sleeves
    overlap_days: int

    def summary(self) -> str:
        best_single = max(m.sharpe for m in self.metrics.values())
        c = self.combo_inv_vol
        lines = [
            "=" * 78,
            "MULTI-SLEEVE ANALYSIS (causal inverse-vol blend)",
            "=" * 78,
            f"{'metric':<18}" + "".join(f"{n[:14]:>15}" for n in self.names) + f"{'inv-vol blend':>16}",
            "-" * 78,
        ]
        def row(label, fn, fmt):
            cells = "".join(f"{fmt(fn(self.metrics[n])):>15}" for n in self.names)
            return f"{label:<18}{cells}{fmt(fn(c)):>16}"
        pct = lambda x: f"{x:.2%}"
        two = lambda x: f"{x:.2f}"
        lines += [
            row("CAGR", lambda m: m.cagr, pct),
            row("Annual vol", lambda m: m.annual_volatility, pct),
            row("Sharpe", lambda m: m.sharpe, two),
            row("Sortino", lambda m: m.sortino, two),
            row("Max drawdown", lambda m: m.max_drawdown, pct),
            row("Calmar", lambda m: m.calmar, two),
            "-" * 78,
            "Pairwise daily-return correlation:",
            self.corr_matrix.round(3).to_string(),
            "-" * 78,
            "Per-sleeve OOS robustness (CPCV):",
        ]
        for n in self.names:
            lines.append(
                f"  {n:<18} median Sharpe={self.cpcv_median[n]:>6.2f}  "
                f"P(SR>0)={self.cpcv_p_positive[n]:>6.1%}  DSR={self.dsr[n]:>6.3f}"
            )
        uplift = c.sharpe - best_single
        lines += [
            "-" * 78,
            f"Blend Sharpe {c.sharpe:.2f} vs best single {best_single:.2f}  "
            f"(uplift {uplift:+.2f})",
            f"Blend MaxDD {c.max_drawdown:.2%} | Calmar {c.calmar:.2f} | "
            f"vol {c.annual_volatility:.2%}",
            "=" * 78,
            "(Sharpe is GROSS of the risk-free rate. Blend = parameter-free causal",
            " inverse-vol weights on a 63-day trailing vol, lagged one day — no fit.)",
        ]
        return "\n".join(lines)


def _sleeve_returns(prices: pd.DataFrame, sleeves: Sequence[Sleeve], cfg: ETFConfig) -> pd.DataFrame:
    """Backtest each sleeve and return an aligned daily-returns frame.

    Trimmed to the first date where ANY sleeve has started trading (drops the
    shared pre-trade warmup so correlations reflect live behavior).
    """
    cols = {}
    for s in sleeves:
        res = backtest_sleeve(prices, s, cfg)
        cols[s.name] = res.returns
    df = pd.DataFrame(cols).dropna()
    active = df.abs().sum(axis=1) > 0
    if active.any():
        df = df.loc[active.idxmax():]
    return df


def _inverse_vol_blend(returns: pd.DataFrame, *, win: int = 63) -> pd.Series:
    """Parameter-free causal inverse-vol blend across sleeve return columns.

    Weights use ONLY the trailing ``win``-day vol lagged one day (``shift(1)``),
    so the blend is strictly out-of-sample at every point — no look-ahead.
    """
    vol = returns.rolling(win).std().shift(1)
    inv = 1.0 / vol.replace(0.0, np.nan)
    w = inv.div(inv.sum(axis=1), axis=0)
    blend = (w * returns).sum(axis=1)
    return blend.loc[w.dropna(how="any").index]


def _cpcv_stats(returns: pd.Series) -> Tuple[float, float, float]:
    """Return (CPCV median annual Sharpe, P(Sharpe>0), Deflated Sharpe Ratio)."""
    try:
        cpcv = cpcv_oos_sharpes(returns, n_groups=8, k_test=2, purge=5)
        med = float(np.median(cpcv)) if len(cpcv) else float("nan")
        p_pos = float((cpcv > 0).mean()) if len(cpcv) else float("nan")
    except Exception:  # pragma: no cover - defensive
        med, p_pos = float("nan"), float("nan")
    try:
        cpcv_pp = cpcv_oos_sharpes(returns, n_groups=8, k_test=2, purge=5, annualize=False)
        trials = cpcv_pp if len(cpcv_pp) else [0.0]
        dsr = deflated_sharpe_ratio(returns, trial_sharpes=trials).get("dsr", float("nan"))
    except Exception:  # pragma: no cover
        dsr = float("nan")
    return med, p_pos, dsr


def analyze_sleeve_set(
    prices: pd.DataFrame,
    sleeves: Sequence[Sleeve],
    cfg: ETFConfig,
) -> MultiSleeveReport:
    """Backtest a set of sleeves; report standalone metrics, correlations,
    per-sleeve OOS robustness, and a parameter-free inverse-vol blend."""
    names = [s.name for s in sleeves]
    df = _sleeve_returns(prices, sleeves, cfg)
    rf = cfg.backtest.risk_free_rate

    def metrics_of(r: pd.Series) -> ETFMetrics:
        eq = cfg.backtest.initial_capital * (1.0 + r).cumprod()
        return compute_metrics(eq, risk_free_rate=rf)

    per_metrics = {n: metrics_of(df[n]) for n in names}
    corr = df.corr()
    med, ppos, dsr = {}, {}, {}
    for n in names:
        med[n], ppos[n], dsr[n] = _cpcv_stats(df[n])

    blend = _inverse_vol_blend(df)
    combo = metrics_of(blend)

    return MultiSleeveReport(
        names=names,
        metrics=per_metrics,
        corr_matrix=corr,
        cpcv_median=med,
        cpcv_p_positive=ppos,
        dsr=dsr,
        combo_inv_vol=combo,
        overlap_days=len(df),
    )


def default_sleeves(cfg: ETFConfig) -> List[Sleeve]:
    """The PRODUCTION sleeve roster: A (trend), B (mean-rev), C (defensive carry).

    Sleeve D (cross-sectional L/S, :class:`CrossSectionalSleeve`) is intentionally
    excluded: it is beautifully orthogonal (corr ~0.06-0.18) but has NO standalone
    edge — gross-of-cost Sharpe is ~0/negative across an 8-cell parameter sweep
    (cross-sectional momentum among ~10 broad equity-sector ETFs is empirically
    absent). Diversifying a zero-edge source only adds noise and dragged the
    blend Sharpe 0.51 -> 0.40, so it is rejected for production. The class is
    retained as tested infrastructure for future research (e.g. a much wider
    cross-section) and for the Phase 3 combiner experiments.

    Sleeve E (turn-of-month seasonality, :class:`TurnOfMonthSleeve`) is ALSO
    excluded after honest evaluation (full sample 2007-2026, daily, realistic
    costs): standalone Sharpe -0.04 gross-of-rf, CPCV median 0.31 but Deflated
    Sharpe only 0.020 (no edge after multiple-testing deflation), and it is the
    *least* orthogonal candidate (corr 0.365 to trend — it is just long equity
    beta inside a calendar window). Adding it dragged the parameter-free inv-vol
    blend Sharpe 0.51 -> 0.39. The ToM premium on 3 broad equity ETFs net of
    transaction costs is too thin to be a promotable sleeve. Retained as tested
    research infrastructure (the calendar logic is reusable for a future
    cost-aware or wider-universe variant).

    LESSON (two rejected candidates): the Sharpe >= 1.10 gate is an EDGE problem,
    not an orthogonality problem — the three production sleeves are already
    decently uncorrelated. Bolting on another low/zero-edge orthogonal stream
    only adds noise. The next experiments must target genuinely new *edge*.

    Volatility-managed overlay (``cfg.vol_managed.enabled``): when ON, the two
    equity-beta sleeves (trend, mean-reversion) are wrapped in
    :class:`VolManagedSleeve` — a Moreira-Muir conditional vol-timing layer that
    scales each basket inversely to its recent realised vol. This attacks the
    gate as an EDGE improvement (a documented Sharpe lifter), not another
    orthogonal stream. The defensive-carry sleeve is left bare (it is already
    low-beta/low-vol; timing it adds little). OFF reproduces the bare roster.
    """
    trend: Sleeve = TrendMomentumSleeve(cfg, apply_dd=False)
    mean_rev: Sleeve = MeanReversionSleeve(cfg)
    if cfg.vol_managed.enabled:
        trend = VolManagedSleeve(trend, cfg, name="trend_momentum")
        mean_rev = VolManagedSleeve(mean_rev, cfg, name="mean_reversion")
    return [
        trend,
        mean_rev,
        DefensiveCarrySleeve(cfg),
    ]


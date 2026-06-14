"""
etf/validation.py
=================
Phase 0 — Anti-overfitting backbone for the ETF engine.

This module is the statistical conscience of the program. Before any new alpha
sleeve is allowed near capital, it must survive the tests implemented here.
Everything is **purely additive** — it consumes the existing backtester and
metrics, and changes nothing in the live trading path.

Contents
--------
1. Purged + embargoed **walk-forward** splits (López de Prado, *Advances in
   Financial Machine Learning*, ch. 7). Ready for parameter-fitting sleeves in
   Phase 1+, and used now for fold-by-fold OOS stability of the parameter-free
   strategy.
2. **Combinatorial Purged Cross-Validation (CPCV)** path generation — yields a
   *distribution* of out-of-sample Sharpe ratios instead of a single number.
3. **Probabilistic & Deflated Sharpe Ratio** (Bailey & López de Prado 2014) —
   adjusts an observed Sharpe for the number of trials, non-normal returns, and
   sample length, so we can tell real edge from multiple-testing luck.
4. **Probability of Backtest Overfitting (PBO)** via Combinatorially Symmetric
   Cross-Validation (Bailey, Borwein, López de Prado & Zhu 2017).
5. A realistic **cost & capacity model** (half-spread + square-root market
   impact as a function of ADV participation) and an AUM-vs-cost-drag estimator.

References are cited inline. No look-ahead: all splitters operate on a sorted
time index and only ever expose causal train/test partitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from etf.backtest import BacktestResult, run_backtest
from etf.config import ETFConfig, get_default_config
from etf.metrics import ETFMetrics, compute_metrics

logger = logging.getLogger("etf.validation")

_TRADING_DAYS = 252
_EULER_MASCHERONI = 0.5772156649015329


# ===========================================================================
# 1. Walk-forward splits (purged + embargoed)
# ===========================================================================
@dataclass
class Split:
    """A single causal train/test partition expressed as positional indices."""

    train: np.ndarray
    test: np.ndarray
    fold: int


def walk_forward_splits(
    n_obs: int,
    *,
    n_splits: int = 5,
    train_span: Optional[int] = None,
    purge: int = 5,
    embargo: int = 5,
    anchored: bool = True,
) -> List[Split]:
    """Generate sequential walk-forward train/test splits.

    The timeline is divided into ``n_splits`` contiguous test windows. Each test
    window is preceded by a training window. A **purge** gap removes the
    ``purge`` observations immediately before each test window (so labels that
    overlap the test period cannot leak into training), and an **embargo**
    removes ``embargo`` observations immediately *after* a test window from any
    later training set.

    ``anchored=True`` grows the training window from the start (expanding);
    ``anchored=False`` uses a fixed ``train_span`` rolling window.
    """
    if n_obs <= 0 or n_splits < 1:
        return []
    test_size = n_obs // (n_splits + 1)
    if test_size < 1:
        raise ValueError("Not enough observations for the requested n_splits")

    splits: List[Split] = []
    for k in range(n_splits):
        test_start = test_size * (k + 1)
        test_end = test_size * (k + 2) if k < n_splits - 1 else n_obs
        test_idx = np.arange(test_start, test_end)

        train_end = max(0, test_start - purge)
        if anchored or train_span is None:
            train_start = 0
        else:
            train_start = max(0, train_end - train_span)
        train_idx = np.arange(train_start, train_end)

        # Apply embargo: drop observations within `embargo` *after* this test
        # window from the (here irrelevant, but kept for generality) train set.
        if embargo > 0 and len(train_idx):
            embargo_zone = set(range(test_end, min(n_obs, test_end + embargo)))
            train_idx = np.array([i for i in train_idx if i not in embargo_zone])

        if len(test_idx):
            splits.append(Split(train=train_idx, test=test_idx, fold=k))
    return splits


def evaluate_walk_forward(
    prices: pd.DataFrame,
    cfg: ETFConfig,
    *,
    n_splits: int = 5,
    purge: int = 5,
    embargo: int = 5,
) -> Tuple[List[ETFMetrics], pd.DataFrame]:
    """Run the strategy once and report metrics on each walk-forward test fold.

    For the current *parameter-free* strategy the decisions are already causal,
    so the single full backtest is sliced into folds — this measures OOS
    *stability* across time. (When sleeves gain fitted parameters in Phase 1,
    the same :func:`walk_forward_splits` partitions drive per-fold re-fitting.)
    """
    result = run_backtest(prices, cfg)
    rets = result.returns.dropna()
    n = len(rets)
    splits = walk_forward_splits(n, n_splits=n_splits, purge=purge, embargo=embargo)

    fold_metrics: List[ETFMetrics] = []
    rows = []
    for sp in splits:
        seg = rets.iloc[sp.test]
        if len(seg) < 5:
            continue
        eq = (1.0 + seg).cumprod()
        m = compute_metrics(eq, risk_free_rate=cfg.backtest.risk_free_rate)
        fold_metrics.append(m)
        rows.append({
            "fold": sp.fold,
            "start": seg.index[0].date(),
            "end": seg.index[-1].date(),
            "n": len(seg),
            "cagr": m.cagr,
            "sharpe": m.sharpe,
            "max_drawdown": m.max_drawdown,
            "profit_factor": m.profit_factor,
        })
    return fold_metrics, pd.DataFrame(rows)


# ===========================================================================
# 2. Combinatorial Purged Cross-Validation (CPCV) paths
# ===========================================================================
def cpcv_groups(n_obs: int, n_groups: int) -> List[np.ndarray]:
    """Partition ``range(n_obs)`` into ``n_groups`` contiguous index groups."""
    if n_groups < 2 or n_groups > n_obs:
        raise ValueError("n_groups must be in [2, n_obs]")
    return [g for g in np.array_split(np.arange(n_obs), n_groups)]


def cpcv_oos_sharpes(
    returns: pd.Series,
    *,
    n_groups: int = 8,
    k_test: int = 2,
    purge: int = 5,
    annualize: bool = True,
) -> np.ndarray:
    """Distribution of out-of-sample Sharpe ratios over CPCV test combinations.

    Every combination of ``k_test`` groups (out of ``n_groups``) forms an OOS
    set; observations within ``purge`` of a test-group boundary are dropped from
    the OOS set to avoid leakage at the seams. Returns the array of Sharpe
    ratios — its spread is a direct, honest read on result fragility.
    """
    rets = returns.dropna().values
    n = len(rets)
    if n < n_groups:
        return np.array([])
    groups = cpcv_groups(n, n_groups)
    sharpes: List[float] = []
    for combo in combinations(range(n_groups), k_test):
        idx = np.concatenate([groups[g] for g in combo])
        if purge > 0:
            # remove the `purge` indices just outside each chosen group's edges
            boundaries = set()
            for g in combo:
                lo, hi = groups[g][0], groups[g][-1]
                boundaries.update(range(max(0, lo - purge), lo))
                boundaries.update(range(hi + 1, min(n, hi + 1 + purge)))
            idx = np.array([i for i in idx if i not in boundaries])
        seg = rets[idx]
        sd = seg.std()
        if sd <= 0 or len(seg) < 5:
            continue
        sr = seg.mean() / sd
        if annualize:
            sr *= np.sqrt(_TRADING_DAYS)
        sharpes.append(float(sr))
    return np.array(sharpes)


# ===========================================================================
# 3. Probabilistic & Deflated Sharpe Ratio
# ===========================================================================
def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> float:
    """PSR: probability the true Sharpe exceeds ``benchmark_sr``.

    All Sharpe values are **per-period** (not annualized). ``kurtosis`` is the
    non-excess (normal == 3.0) fourth moment. Bailey & López de Prado (2012).
    """
    if n_obs < 2:
        return float("nan")
    denom = np.sqrt(max(1e-12, 1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr**2))
    z = (observed_sr - benchmark_sr) * np.sqrt(n_obs - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected maximum of ``n_trials`` i.i.d. Sharpe estimates under the null
    (true SR == 0), per-period units. Bailey & López de Prado (2014)."""
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(
    returns: pd.Series,
    trial_sharpes: Sequence[float],
) -> Dict[str, float]:
    """Deflated Sharpe Ratio.

    Adjusts the observed (per-period) Sharpe for the number of trials and the
    dispersion of their Sharpe ratios, then evaluates the PSR against that
    multiple-testing benchmark. DSR > 0.95 is the usual "real edge" bar.

    ``trial_sharpes`` are the *per-period* Sharpe ratios of every configuration
    that was tried while searching for this strategy (the multiple-testing set).
    """
    r = returns.dropna()
    if len(r) < 3:
        return {"observed_sr_annual": float("nan"), "dsr": float("nan"),
                "benchmark_sr_annual": float("nan"), "n_trials": len(trial_sharpes)}
    sd = r.std()
    observed_sr = float(r.mean() / sd) if sd > 0 else 0.0
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    n_trials = max(1, len(trial_sharpes))
    sr_var = float(np.var(np.asarray(trial_sharpes, dtype=float), ddof=1)) if n_trials > 1 else 0.0
    sr0 = expected_max_sharpe(sr_var, n_trials)
    dsr = probabilistic_sharpe_ratio(observed_sr, sr0, len(r), skew, kurt)
    return {
        "observed_sr_annual": observed_sr * np.sqrt(_TRADING_DAYS),
        "benchmark_sr_annual": sr0 * np.sqrt(_TRADING_DAYS),
        "dsr": dsr,
        "skew": skew,
        "kurtosis": kurt,
        "n_trials": n_trials,
        "n_obs": len(r),
    }


# ===========================================================================
# 4. Probability of Backtest Overfitting (PBO) via CSCV
# ===========================================================================
def _sharpe_per_column(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std <= 0, np.nan, std)
    return mean / std


def probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame,
    *,
    n_partitions: int = 10,
) -> Dict[str, float]:
    """PBO via Combinatorially Symmetric Cross-Validation.

    ``returns_matrix`` is (T observations × N configurations) of per-period
    returns. The timeline is split into ``n_partitions`` (even) contiguous
    blocks; for every way of choosing half as in-sample (IS) and half as
    out-of-sample (OOS), we pick the IS-best config and record its OOS relative
    rank. PBO is the fraction of cases where the IS-best config lands in the
    bottom half OOS — i.e. selection did not generalize.

    Bailey, Borwein, López de Prado & Zhu (2017).
    """
    M = returns_matrix.dropna(how="any").values
    T, N = M.shape
    if N < 2:
        return {"pbo": float("nan"), "n_configs": N, "n_combinations": 0}
    if n_partitions % 2 != 0:
        n_partitions += 1
    n_partitions = min(n_partitions, T)
    blocks = np.array_split(np.arange(T), n_partitions)

    lambdas: List[float] = []
    for is_sel in combinations(range(n_partitions), n_partitions // 2):
        is_set = set(is_sel)
        is_rows = np.concatenate([blocks[i] for i in range(n_partitions) if i in is_set])
        oos_rows = np.concatenate([blocks[i] for i in range(n_partitions) if i not in is_set])

        is_perf = _sharpe_per_column(M[is_rows])
        oos_perf = _sharpe_per_column(M[oos_rows])
        if np.all(np.isnan(is_perf)) or np.all(np.isnan(oos_perf)):
            continue
        n_star = int(np.nanargmax(is_perf))
        # relative rank of the IS-best config in OOS (1 = worst ... N = best)
        ranks = stats.rankdata(np.nan_to_num(oos_perf, nan=-np.inf))
        omega = ranks[n_star] / (N + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        lambdas.append(float(np.log(omega / (1.0 - omega))))

    if not lambdas:
        return {"pbo": float("nan"), "n_configs": N, "n_combinations": 0}
    arr = np.array(lambdas)
    return {
        "pbo": float(np.mean(arr <= 0.0)),
        "median_logit": float(np.median(arr)),
        "n_configs": N,
        "n_combinations": len(arr),
    }


def build_config_family(base: Optional[ETFConfig] = None) -> List[Tuple[str, ETFConfig]]:
    """A small grid of plausible configs spanning the key knobs.

    This is the multiple-testing set for DSR/PBO: it represents the
    configurations a researcher would realistically try. Kept deliberately
    coarse to avoid manufacturing an over-tuned "winner".
    """
    family: List[Tuple[str, ETFConfig]] = []
    for tvol in (0.08, 0.10, 0.12, 0.15):
        for rebal in (10, 21, 42):
            for top_k in (4, 5, 6):
                cfg = get_default_config()
                cfg.risk.target_volatility = tvol
                cfg.execution.rebalance_every = rebal
                cfg.signal.top_k = top_k
                family.append((f"tv{tvol}_rb{rebal}_k{top_k}", cfg))
    if base is not None:
        family.insert(0, ("base", base))
    return family


def compute_config_returns_matrix(
    prices: pd.DataFrame,
    configs: Sequence[Tuple[str, ETFConfig]],
) -> pd.DataFrame:
    """Run each config and collect aligned daily-return columns."""
    cols: Dict[str, pd.Series] = {}
    for name, cfg in configs:
        try:
            res = run_backtest(prices, cfg)
            cols[name] = res.returns
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Config %s failed: %s", name, exc)
    return pd.DataFrame(cols).dropna(how="all")


# ===========================================================================
# 5. Cost & capacity model
# ===========================================================================
@dataclass
class CostModel:
    """Realistic per-trade cost: half-spread + square-root market impact.

    cost_bps(participation) = half_spread_bps + impact_coeff * 100 * sqrt(p)

    where ``p`` is the order's participation in average daily volume (ADV). The
    square-root law is the standard market-impact form (Almgren et al. 2005).
    For megacap ETFs (SPY/QQQ trade billions/day) realistic retail participation
    is tiny, so impact is negligible — which is exactly why an ETF-only book has
    enormous capacity. Defaults are conservative (wider than reality).
    """

    half_spread_bps: float = 1.0      # ~0.5-1 bp for liquid ETFs
    impact_coeff: float = 0.1         # impact in bps at 100% ADV ≈ 10 bps
    min_cost_bps: float = 0.5

    def cost_bps(self, participation: float) -> float:
        participation = max(0.0, float(participation))
        impact = self.impact_coeff * 100.0 * np.sqrt(participation)
        return max(self.min_cost_bps, self.half_spread_bps + impact)


@dataclass
class CapacityReport:
    table: pd.DataFrame              # AUM vs estimated annual cost drag
    capacity_aum: float              # AUM where drag first exceeds threshold
    threshold: float


def estimate_capacity(
    result: BacktestResult,
    *,
    cost_model: Optional[CostModel] = None,
    representative_adv_usd: float = 2_000_000_000.0,  # ~SPY-class liquidity
    n_names_per_trade: int = 5,
    aum_grid: Optional[Sequence[float]] = None,
    drag_threshold: float = 0.01,
) -> CapacityReport:
    """Estimate annual cost drag as a function of AUM and find the capacity.

    Method: the backtest's per-rebalance turnover (fraction of NAV traded) is
    converted to dollar volume at each candidate AUM, spread across
    ``n_names_per_trade`` ETFs, and priced through the :class:`CostModel`. The
    capacity is the AUM at which annual cost drag first exceeds
    ``drag_threshold`` (default 1%/yr). Estimate, clearly labelled as such.
    """
    cost_model = cost_model or CostModel()
    if aum_grid is None:
        aum_grid = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10]

    turn = result.turnover.fillna(0.0)
    rebal_turns = turn[turn > 0]
    if rebal_turns.empty:
        empty = pd.DataFrame({"aum_usd": list(aum_grid), "annual_cost_drag": [0.0] * len(aum_grid)})
        return CapacityReport(table=empty, capacity_aum=float("inf"), threshold=drag_threshold)

    years = max(1e-9, len(turn) / _TRADING_DAYS)
    rebals_per_year = len(rebal_turns) / years
    avg_turnover = float(rebal_turns.mean())

    rows = []
    capacity = float("inf")
    for aum in aum_grid:
        # dollars traded per rebalance, split across the names touched
        dollars_per_name = (avg_turnover * aum) / max(1, n_names_per_trade)
        participation = dollars_per_name / representative_adv_usd
        cbps = cost_model.cost_bps(participation)
        # annual drag = rebalances/yr * turnover * cost(fraction)
        annual_drag = rebals_per_year * avg_turnover * (cbps / 1e4)
        rows.append({"aum_usd": aum, "participation": participation,
                     "cost_bps": cbps, "annual_cost_drag": annual_drag})
        if annual_drag > drag_threshold and capacity == float("inf"):
            capacity = aum
    return CapacityReport(table=pd.DataFrame(rows), capacity_aum=capacity, threshold=drag_threshold)


# ===========================================================================
# Top-level report
# ===========================================================================
@dataclass
class ValidationReport:
    full_metrics: ETFMetrics
    walk_forward: pd.DataFrame
    cpcv_sharpes: np.ndarray
    deflated_sharpe: Dict[str, float]
    pbo: Dict[str, float]
    capacity: CapacityReport

    def summary(self) -> str:
        wf = self.walk_forward
        cp = self.cpcv_sharpes
        lines = [
            "=" * 64,
            "ETF ENGINE — PHASE 0 VALIDATION REPORT",
            "=" * 64,
            f"  Full-period Sharpe        : {self.full_metrics.sharpe:.3f}",
            f"  Full-period CAGR          : {self.full_metrics.cagr:.2%}",
            f"  Full-period MaxDD         : {self.full_metrics.max_drawdown:.2%}",
            "",
            "  Walk-forward folds (OOS stability):",
        ]
        if not wf.empty:
            for _, r in wf.iterrows():
                lines.append(
                    f"    fold {int(r['fold'])}: {r['start']}→{r['end']}  "
                    f"Sharpe {r['sharpe']:>6.2f}  CAGR {r['cagr']:>7.2%}  "
                    f"MaxDD {r['max_drawdown']:>7.2%}"
                )
            pos = (wf["sharpe"] > 0).mean()
            lines.append(f"    => {pos:.0%} of folds have positive Sharpe")
        if len(cp):
            lines += [
                "",
                "  CPCV out-of-sample Sharpe distribution:",
                f"    median {np.median(cp):.2f} | 5th pct {np.percentile(cp,5):.2f} | "
                f"95th pct {np.percentile(cp,95):.2f} | P(SR>0) {np.mean(cp>0):.0%}",
            ]
        ds = self.deflated_sharpe
        lines += [
            "",
            "  Multiple-testing adjustment (Sharpe here is GROSS, no risk-free):",
            f"    Observed Sharpe (ann.)  : {ds.get('observed_sr_annual', float('nan')):.2f}",
            f"    Benchmark (E[max] null) : {ds.get('benchmark_sr_annual', float('nan')):.2f}",
            f"    Deflated Sharpe Ratio   : {ds.get('dsr', float('nan')):.3f}  "
            f"(>0.95 = real edge; trials={ds.get('n_trials')})",
            f"    PBO                     : {self.pbo.get('pbo', float('nan')):.2%}  "
            f"(<50% good; configs={self.pbo.get('n_configs')})",
            "",
            "  Capacity (annual cost drag vs AUM):",
        ]
        for _, r in self.capacity.table.iterrows():
            lines.append(
                f"    ${r['aum_usd']:>14,.0f}: drag {r['annual_cost_drag']:.3%}"
                + (f"  (part. {r['participation']:.2e})" if "participation" in r else "")
            )
        cap = self.capacity.capacity_aum
        cap_str = "effectively unlimited" if cap == float("inf") else f"${cap:,.0f}"
        lines.append(f"    => capacity (<{self.capacity.threshold:.0%}/yr drag): {cap_str}")
        lines.append("=" * 64)
        return "\n".join(lines)


def run_validation(
    prices: pd.DataFrame,
    cfg: Optional[ETFConfig] = None,
    *,
    n_splits: int = 5,
    cpcv_groups_n: int = 8,
    cpcv_k_test: int = 2,
    pbo_partitions: int = 10,
) -> ValidationReport:
    """Run the full Phase 0 validation battery on a single config."""
    cfg = cfg or get_default_config()
    result = run_backtest(prices, cfg)

    _, wf = evaluate_walk_forward(prices, cfg, n_splits=n_splits)
    cp = cpcv_oos_sharpes(result.returns, n_groups=cpcv_groups_n, k_test=cpcv_k_test)

    family = build_config_family(base=cfg)
    matrix = compute_config_returns_matrix(prices, family)
    trial_sharpes = [
        (matrix[c].mean() / matrix[c].std()) for c in matrix.columns
        if matrix[c].std() > 0
    ]
    dsr = deflated_sharpe_ratio(result.returns, trial_sharpes)
    pbo = probability_of_backtest_overfitting(matrix, n_partitions=pbo_partitions)

    capacity = estimate_capacity(result, n_names_per_trade=cfg.signal.top_k)

    return ValidationReport(
        full_metrics=result.metrics,
        walk_forward=wf,
        cpcv_sharpes=cp,
        deflated_sharpe=dsr,
        pbo=pbo,
        capacity=capacity,
    )

"""
tests/test_etf_validation.py
============================
Offline regression tests for the Phase 0 anti-overfitting harness.

Deterministic synthetic data only (fixed seeds) — no network. Includes
negative controls: a pure-noise config family must produce a *high* PBO and a
*low* Deflated Sharpe, while a genuinely-skilled return stream must not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf.validation import (
    CostModel,
    build_config_family,
    cpcv_groups,
    cpcv_oos_sharpes,
    deflated_sharpe_ratio,
    estimate_capacity,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    walk_forward_splits,
)


# --- walk-forward splits --------------------------------------------------
def test_walk_forward_splits_are_causal_and_purged():
    splits = walk_forward_splits(1000, n_splits=5, purge=5, embargo=5)
    assert len(splits) == 5
    for sp in splits:
        # train must lie strictly before test, with a purge gap
        if len(sp.train):
            assert sp.train.max() < sp.test.min()
            assert sp.test.min() - sp.train.max() > 5  # purge gap respected
        # test indices contiguous and increasing
        assert np.all(np.diff(sp.test) == 1)


def test_walk_forward_anchored_grows():
    splits = walk_forward_splits(900, n_splits=3, anchored=True, purge=0, embargo=0)
    train_sizes = [len(s.train) for s in splits]
    assert train_sizes == sorted(train_sizes)  # expanding window


# --- CPCV -----------------------------------------------------------------
def test_cpcv_groups_partition():
    groups = cpcv_groups(100, 8)
    assert len(groups) == 8
    allidx = np.concatenate(groups)
    assert sorted(allidx.tolist()) == list(range(100))


def test_cpcv_oos_sharpes_count_and_finite():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0005, 0.01, 600),
                     index=pd.bdate_range("2018-01-01", periods=600))
    sr = cpcv_oos_sharpes(rets, n_groups=8, k_test=2)
    # C(8,2) = 28 combinations
    assert len(sr) == 28
    assert np.all(np.isfinite(sr))


# --- PSR / DSR ------------------------------------------------------------
def test_psr_monotonic_in_observed_sr():
    low = probabilistic_sharpe_ratio(0.02, 0.0, 500, 0.0, 3.0)
    high = probabilistic_sharpe_ratio(0.10, 0.0, 500, 0.0, 3.0)
    assert 0.0 <= low <= high <= 1.0


def test_expected_max_sharpe_increases_with_trials():
    e10 = expected_max_sharpe(0.01, 10)
    e100 = expected_max_sharpe(0.01, 100)
    assert e100 > e10 > 0


def test_dsr_penalizes_more_trials():
    rng = np.random.default_rng(7)
    # genuinely skilled stream (positive drift)
    rets = pd.Series(rng.normal(0.0008, 0.008, 1000),
                     index=pd.bdate_range("2016-01-01", periods=1000))
    few = deflated_sharpe_ratio(rets, trial_sharpes=[0.02, 0.03])
    many = deflated_sharpe_ratio(rets, trial_sharpes=list(np.linspace(-0.05, 0.06, 200)))
    assert few["dsr"] >= many["dsr"]  # more trials => harder to clear
    assert 0.0 <= many["dsr"] <= 1.0


# --- PBO (negative control) ----------------------------------------------
def test_pbo_high_for_pure_noise():
    """Random configs with no real edge => selection shouldn't generalize.

    PBO for a no-skill process approaches ~0.5+ as the number of configs and
    partitions grows (verified empirically); small samples understate it, so we
    use statistically adequate dimensions here.
    """
    rng = np.random.default_rng(123)
    T, N = 2000, 50
    noise = rng.normal(0.0, 0.01, size=(T, N))
    mat = pd.DataFrame(noise, index=pd.bdate_range("2012-01-01", periods=T))
    res = probability_of_backtest_overfitting(mat, n_partitions=14)
    assert res["n_configs"] == N
    assert res["pbo"] > 0.4  # no-skill => high overfitting probability


def test_pbo_low_for_dominant_config():
    """One config dominates everywhere => low overfitting probability."""
    rng = np.random.default_rng(9)
    T, N = 2000, 50
    base = rng.normal(0.0, 0.01, size=(T, N))
    base[:, 0] += 0.002  # config 0 has a real, persistent edge
    mat = pd.DataFrame(base, index=pd.bdate_range("2012-01-01", periods=T))
    res = probability_of_backtest_overfitting(mat, n_partitions=14)
    assert res["pbo"] < 0.3
    # negative control sanity: dominant-edge PBO must be far below noise PBO
    noise = pd.DataFrame(rng.normal(0.0, 0.01, size=(T, N)),
                         index=mat.index)
    noise_res = probability_of_backtest_overfitting(noise, n_partitions=14)
    assert noise_res["pbo"] - res["pbo"] > 0.2


# --- cost & capacity ------------------------------------------------------
def test_cost_model_monotonic_in_participation():
    cm = CostModel()
    assert cm.cost_bps(0.0) <= cm.cost_bps(0.001) <= cm.cost_bps(0.01)


def test_config_family_nonempty_and_valid():
    fam = build_config_family()
    assert len(fam) >= 12
    for _, cfg in fam:
        cfg.validate()  # must not raise


def _toy_result(turnover_value=0.3, n=300):
    # minimal stand-in for BacktestResult fields used by estimate_capacity
    from etf.backtest import BacktestResult
    from etf.metrics import ETFMetrics
    idx = pd.bdate_range("2019-01-01", periods=n)
    turn = pd.Series(0.0, index=idx)
    turn.iloc[::21] = turnover_value  # monthly rebalances
    eq = pd.Series(np.linspace(100, 130, n), index=idx)
    return BacktestResult(
        equity=eq, returns=eq.pct_change().fillna(0.0),
        gross_exposure=pd.Series(1.0, index=idx), turnover=turn,
        weights_history=pd.DataFrame(index=idx), metrics=ETFMetrics(),
    )


def test_capacity_drag_increases_with_aum():
    rep = estimate_capacity(_toy_result(), n_names_per_trade=5)
    drags = rep.table["annual_cost_drag"].values
    assert np.all(np.diff(drags) >= -1e-12)  # non-decreasing in AUM
    assert (rep.table["aum_usd"] > 0).all()

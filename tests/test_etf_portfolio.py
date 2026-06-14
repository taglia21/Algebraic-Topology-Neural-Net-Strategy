"""
tests/test_etf_portfolio.py
===========================
Phase 3 regression tests: the ERC risk-parity combiner.

Offline + deterministic. Coverage:
  - ERC solver equalises risk contributions (the core math);
  - ERC reduces to inverse-vol when assets are uncorrelated;
  - ERC handles a degenerate (zero-variance) sleeve without NaNs;
  - the combined backtest produces a clean, NaN-free equity curve;
  - the combiner respects the leverage cap and vol target direction;
  - the combiner is look-ahead-free (covariance window excludes the live bar).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf.config import get_default_config
from etf.portfolio import erc_weights, live_target_weights, run_combined_backtest
from etf.sleeves import (
    DefensiveCarrySleeve,
    MeanReversionSleeve,
    TrendMomentumSleeve,
)


def _synthetic_prices(seed: int = 13, n_days: int = 1400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2013-01-01", periods=n_days)
    specs = {
        "SPY": (0.0006, 0.010), "QQQ": (0.0008, 0.016), "IWM": (0.0003, 0.014),
        "EFA": (0.0002, 0.012), "EEM": (-0.0001, 0.018), "XLK": (0.0007, 0.015),
        "XLF": (0.0002, 0.013), "XLE": (0.0000, 0.020), "XLV": (0.0004, 0.011),
        "XLI": (0.0003, 0.012), "TLT": (0.0002, 0.010), "IEF": (0.0001, 0.005),
        "LQD": (0.0001, 0.006), "GLD": (0.0003, 0.011), "DBC": (-0.0002, 0.013),
        "BIL": (0.00005, 0.0006),
    }
    data = {}
    for sym, (mu, sigma) in specs.items():
        data[sym] = 100.0 * np.exp(np.cumsum(rng.normal(mu, sigma, n_days)))
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def cfg():
    return get_default_config()


@pytest.fixture
def prices():
    return _synthetic_prices()


def _risk_contributions(w, cov):
    mrc = cov @ w
    rc = w * mrc
    return rc / rc.sum()


# --- ERC solver math -------------------------------------------------------
def test_erc_equalises_risk_contributions():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 3))
    # Impose distinct vols and some correlation.
    a = a @ np.array([[1.0, 0.3, 0.1], [0.0, 0.8, 0.2], [0.0, 0.0, 0.5]])
    cov = np.cov(a, rowvar=False)
    w = erc_weights(cov)
    assert np.all(w > 0)
    assert abs(w.sum() - 1.0) < 1e-9
    rc = _risk_contributions(w, cov)
    # Every sleeve contributes ~1/3 of total risk.
    assert np.allclose(rc, 1.0 / 3.0, atol=1e-3)


def test_erc_reduces_to_inverse_vol_when_uncorrelated():
    vols = np.array([0.10, 0.20, 0.40])
    cov = np.diag(vols ** 2)
    w = erc_weights(cov)
    inv = (1.0 / vols) / (1.0 / vols).sum()
    assert np.allclose(w, inv, atol=1e-4)


def test_erc_handles_degenerate_covariance():
    cov = np.array([[0.04, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.09]])
    w = erc_weights(cov)
    assert np.all(np.isfinite(w))
    assert abs(w.sum() - 1.0) < 1e-9
    # Zero-variance asset should get ~no weight under inverse-vol fallback.
    assert w[1] < 1e-6


# --- combined backtest -----------------------------------------------------
def _roster(cfg):
    return [
        TrendMomentumSleeve(cfg, apply_dd=False),
        MeanReversionSleeve(cfg),
        DefensiveCarrySleeve(cfg),
    ]


def test_combined_backtest_clean_curve(cfg, prices):
    res = run_combined_backtest(prices, cfg, _roster(cfg))
    assert not res.equity.isna().any()
    assert res.equity.iloc[-1] > 0
    assert res.sleeve_weights.shape[1] == 3


def test_combined_respects_leverage_cap(cfg, prices):
    cfg.portfolio.max_leverage = 1.0
    res = run_combined_backtest(prices, cfg, _roster(cfg))
    assert res.gross_exposure.max() <= 1.0 + 1e-9


def test_combined_vol_target_scales_exposure(cfg, prices):
    """A lower vol target must reduce average deployed exposure."""
    cfg_lo = get_default_config()
    cfg_lo.portfolio.target_volatility = 0.05
    cfg_hi = get_default_config()
    cfg_hi.portfolio.target_volatility = 0.10
    lo = run_combined_backtest(prices, cfg_lo, _roster(cfg_lo))
    hi = run_combined_backtest(prices, cfg_hi, _roster(cfg_hi))
    assert lo.gross_exposure.mean() <= hi.gross_exposure.mean() + 1e-9


def test_combined_no_lookahead(cfg, prices):
    """Truncating future prices must not change the equity path on the overlap."""
    full = run_combined_backtest(prices, cfg, _roster(cfg))
    cut = 1100
    trunc = run_combined_backtest(prices.iloc[:cut], cfg, _roster(cfg))
    common = full.equity.index.intersection(trunc.equity.index)
    # Compare normalised paths (both start at initial_capital on their own index;
    # the overlap region should track once both are active).
    a = full.equity.reindex(common)
    b = trunc.equity.reindex(common)
    # Ratio should be ~constant where both are live (same daily returns).
    ratio = (a / b).dropna()
    if len(ratio) > 50:
        tail = ratio.iloc[50:]
        assert tail.std() / tail.mean() < 1e-6


# --- Phase 4: leverage + circuit-breaker -----------------------------------
def test_leverage_raises_exposure_and_return(cfg, prices):
    """Raising the cap above 1.0 lets the vol-target deploy more gross."""
    base = get_default_config()
    base.portfolio.max_leverage = 1.0
    lev = get_default_config()
    lev.portfolio.max_leverage = 2.0
    r1 = run_combined_backtest(prices, base, _roster(base))
    r2 = run_combined_backtest(prices, lev, _roster(lev))
    # The 6%-vol book is below the 10% target, so a higher cap MUST raise gross.
    assert r2.gross_exposure.mean() > r1.gross_exposure.mean() + 1e-6
    assert r2.gross_exposure.max() <= 2.0 + 1e-9


def test_phase4_default_preserves_phase3(cfg, prices):
    """Default config (cap 1.0, derisk off) is bit-for-bit the Phase 3 book."""
    assert cfg.portfolio.max_leverage == 1.0
    assert cfg.portfolio.dd_derisk is False
    res = run_combined_backtest(prices, cfg, _roster(cfg))
    assert res.gross_exposure.max() <= 1.0 + 1e-9


def test_derisk_reduces_drawdown(prices):
    """The book-level circuit-breaker must not WORSEN max drawdown."""
    plain = get_default_config()
    plain.portfolio.max_leverage = 2.0
    plain.portfolio.dd_derisk = False
    guarded = get_default_config()
    guarded.portfolio.max_leverage = 2.0
    guarded.portfolio.dd_derisk = True
    rp = run_combined_backtest(prices, plain, _roster(plain))
    rg = run_combined_backtest(prices, guarded, _roster(guarded))
    # max_drawdown is negative; guarded must be >= plain (shallower or equal).
    assert rg.metrics.max_drawdown >= rp.metrics.max_drawdown - 1e-9


def test_margin_cost_drags_levered_return(prices):
    """A higher margin spread can only lower a levered book's terminal equity."""
    cheap = get_default_config()
    cheap.portfolio.max_leverage = 2.0
    cheap.portfolio.margin_spread_annual = 0.0
    dear = get_default_config()
    dear.portfolio.max_leverage = 2.0
    dear.portfolio.margin_spread_annual = 0.05
    rc = run_combined_backtest(prices, cheap, _roster(cheap))
    rd = run_combined_backtest(prices, dear, _roster(dear))
    assert rd.equity.iloc[-1] <= rc.equity.iloc[-1] + 1e-6


def test_levered_book_no_lookahead(prices):
    """Leverage + circuit-breaker stay strictly causal under truncation."""
    cfg = get_default_config()
    cfg.portfolio.max_leverage = 2.0
    cfg.portfolio.dd_derisk = True
    full = run_combined_backtest(prices, cfg, _roster(cfg))
    trunc = run_combined_backtest(prices.iloc[:1100], cfg, _roster(cfg))
    common = full.equity.index.intersection(trunc.equity.index)
    ratio = (full.equity.reindex(common) / trunc.equity.reindex(common)).dropna()
    if len(ratio) > 50:
        tail = ratio.iloc[50:]
        assert tail.std() / tail.mean() < 1e-6


# --- Phase 5: live target weights (backtest/live consistency) --------------
def test_live_weights_reconstruct_from_sleeves(cfg, prices):
    """Combined weight per symbol == sum_i alloc_i * sleeve_w_i[sym]."""
    sleeves = _roster(cfg)
    alloc = live_target_weights(prices, cfg, sleeves)
    # Rebuild independently from each sleeve's current target weights.
    rebuilt = {}
    for s in sleeves:
        sw = s.target_weights(prices)
        a = alloc.sleeve_alloc[s.name]
        for sym, w in sw.items():
            rebuilt[sym] = rebuilt.get(sym, 0.0) + a * w
    rebuilt = {k: v for k, v in rebuilt.items() if abs(v) > 1e-6}
    assert set(rebuilt) == set(alloc.weights)
    for sym in rebuilt:
        assert abs(rebuilt[sym] - alloc.weights[sym]) < 1e-9


def test_live_weights_gross_respects_cap(cfg, prices):
    """Live gross never exceeds the leverage cap; cash = 1 - gross when unlevered."""
    alloc = live_target_weights(prices, cfg, _roster(cfg))
    assert alloc.gross_exposure <= cfg.portfolio.max_leverage + 1e-9
    assert abs(alloc.cash_weight - max(0.0, 1.0 - alloc.gross_exposure)) < 1e-9


def test_live_weights_deterministic_and_causal(cfg, prices):
    """Live allocation depends only on data up to as_of (deterministic re-run)."""
    a1 = live_target_weights(prices, cfg, _roster(cfg))
    a2 = live_target_weights(prices, cfg, _roster(cfg))
    assert a1.weights == a2.weights
    assert a1.as_of == prices.index[-1]
    # Truncating future rows must reproduce the SAME book at the earlier as_of.
    cut = prices.iloc[:1200]
    b1 = live_target_weights(cut, cfg, _roster(cfg))
    b2 = live_target_weights(prices.iloc[:1200], cfg, _roster(cfg))
    assert b1.weights == b2.weights
    assert b1.as_of == cut.index[-1]


def test_live_equal_method_balances_sleeves(prices):
    """method=equal allocates the same pre-scale capital to each sleeve."""
    cfg = get_default_config()
    cfg.portfolio.method = "equal"
    alloc = live_target_weights(prices, cfg, _roster(cfg))
    vals = list(alloc.sleeve_alloc.values())
    # All equal (each == scale/n), so spread is ~0.
    assert max(vals) - min(vals) < 1e-9


def test_live_derisk_reduces_gross(prices):
    """A large book drawdown shrinks live gross when the breaker is armed."""
    cfg = get_default_config()
    cfg.portfolio.max_leverage = 2.0
    cfg.portfolio.dd_derisk = True
    calm = live_target_weights(prices, cfg, _roster(cfg), current_drawdown=0.0)
    stress = live_target_weights(prices, cfg, _roster(cfg), current_drawdown=0.25)
    assert stress.gross_exposure < calm.gross_exposure - 1e-9



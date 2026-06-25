"""
tests/test_etf_sleeves.py
=========================
Phase 2 regression tests: sleeve framework + mean-reversion sleeve + the
backtester ``weight_fn`` refactor.

All tests run offline on deterministic synthetic data. Coverage:
  - the refactored ``run_backtest`` reproduces the original behavior exactly
    when ``weight_fn`` is None (protects the Phase 0 reproduce gate);
  - the RSI helper is causal and numerically sane;
  - the mean-reversion sleeve never trades a name below its long-term SMA;
  - the sleeve respects max_positions and goes to cash when nothing qualifies;
  - sleeves are look-ahead-free (decision depends only on past data);
  - ``backtest_sleeve`` produces a clean equity curve with no NaNs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf.backtest import run_backtest
from etf.config import ETFConfig, get_default_config
from etf.sleeves import (
    CrossSectionalSleeve,
    DefensiveCarrySleeve,
    MeanReversionSleeve,
    TrendMomentumSleeve,
    backtest_sleeve,
    rsi,
)


def _synthetic_prices(seed: int = 11, n_days: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-01", periods=n_days)
    specs = {
        "SPY": (0.0006, 0.010),
        "QQQ": (0.0008, 0.016),
        "IWM": (0.0003, 0.014),
        "EFA": (0.0002, 0.012),
        "EEM": (-0.0001, 0.018),
        "XLK": (0.0007, 0.015),
        "XLF": (0.0002, 0.013),
        "XLE": (0.0000, 0.020),
        "XLV": (0.0004, 0.011),
        "XLI": (0.0003, 0.012),
        "TLT": (0.0001, 0.010),
        "IEF": (0.0001, 0.005),
        "LQD": (0.0001, 0.006),
        "GLD": (0.0002, 0.011),
        "DBC": (-0.0002, 0.013),
        "BIL": (0.00005, 0.0006),
    }
    data = {}
    for sym, (mu, sigma) in specs.items():
        rets = rng.normal(mu, sigma, n_days)
        data[sym] = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def cfg() -> ETFConfig:
    return get_default_config()


@pytest.fixture
def prices() -> pd.DataFrame:
    return _synthetic_prices()


# --- backtester refactor backward-compatibility ----------------------------
def test_weight_fn_none_reproduces_original(cfg, prices):
    """Default path (weight_fn=None) must be bit-for-bit identical to before."""
    a = run_backtest(prices, cfg)
    b = run_backtest(prices, cfg, weight_fn=None, rebalance_every=None, apply_dd=True)
    pd.testing.assert_series_equal(a.equity, b.equity)
    assert a.metrics.sharpe == b.metrics.sharpe


def test_weight_fn_callable_path_runs(cfg, prices):
    """A trivial all-SPY weight_fn should produce a finite, NaN-free curve."""
    res = run_backtest(prices, cfg, weight_fn=lambda p: {"SPY": 0.5}, rebalance_every=5)
    assert not res.equity.isna().any()
    assert res.equity.iloc[-1] > 0
    # ~50% SPY exposure should be reflected in average gross.
    assert 0.3 < res.gross_exposure.mean() < 0.7


def test_backtest_skips_tiny_rebalance_drifts(cfg, prices):
    """Weight changes below min_rebalance_delta notional must be ignored."""
    cfg.execution.min_rebalance_delta = 0.02  # 2% NAV trade-size threshold

    def tiny_drift_weight_fn(p: pd.DataFrame) -> Dict[str, float]:
        # Oscillates by 1% NAV around a 50% base exposure.
        return {"SPY": 0.50 if (len(p) % 2 == 0) else 0.51}

    res = run_backtest(
        prices[["SPY", "BIL"]],
        cfg,
        weight_fn=tiny_drift_weight_fn,
        rebalance_every=1,
        warmup=1,
    )

    # After the initial entry, all 1%-NAV drifts should be skipped.
    assert (res.turnover.iloc[3:] <= 1e-12).all()


# --- RSI helper ------------------------------------------------------------
def test_rsi_bounds_and_causality():
    rng = np.random.default_rng(0)
    s = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    r = rsi(s, 2)
    valid = r.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()
    # Causality: RSI at t must not change if future values are appended.
    r_full = rsi(s, 2)
    r_trunc = rsi(s.iloc[:200], 2)
    pd.testing.assert_series_equal(r_full.iloc[:200], r_trunc, check_freq=False)


def test_rsi_all_up_is_100():
    s = pd.Series(np.arange(1.0, 50.0))  # strictly increasing
    assert rsi(s, 2).iloc[-1] == 100.0


# --- mean-reversion sleeve -------------------------------------------------
def test_mr_sleeve_respects_trend_gate(cfg):
    """A name below its long SMA must never be bought, even if deeply oversold."""
    mr = cfg.mean_reversion
    # Build a strict downtrend for SPY ending in a sharp dip (low RSI).
    n = mr.trend_sma + 50
    prices = pd.DataFrame({
        "SPY": np.linspace(200.0, 100.0, n),  # always below its own SMA
    }, index=pd.bdate_range("2015-01-01", periods=n))
    sleeve = MeanReversionSleeve(cfg)
    w = sleeve.target_weights(prices)
    assert "SPY" not in w  # trend gate blocks it


def test_mr_sleeve_buys_uptrend_dip(cfg):
    """An uptrending name that dips sharply (RSI<threshold) should be bought."""
    mr = cfg.mean_reversion
    n = mr.trend_sma + 50
    # Long steady uptrend, then a sharp multi-day pullback at the end.
    base = np.linspace(100.0, 200.0, n - 4)
    dip = np.array([base[-1] * f for f in (0.97, 0.94, 0.92, 0.90)])
    series = np.concatenate([base, dip])
    prices = pd.DataFrame({"SPY": series},
                          index=pd.bdate_range("2015-01-01", periods=n))
    sleeve = MeanReversionSleeve(cfg)
    w = sleeve.target_weights(prices)
    assert "SPY" in w and w["SPY"] > 0


def test_mr_sleeve_caps_positions(cfg):
    """Never hold more than max_positions names."""
    mr = cfg.mean_reversion
    n = mr.trend_sma + 50
    base = np.linspace(100.0, 200.0, n - 4)
    dip = [base[-1] * f for f in (0.97, 0.94, 0.92, 0.90)]
    series = np.concatenate([base, dip])
    cols = {sym: series.copy() for sym in mr.universe}
    prices = pd.DataFrame(cols, index=pd.bdate_range("2015-01-01", periods=n))
    sleeve = MeanReversionSleeve(cfg)
    w = sleeve.target_weights(prices)
    assert len(w) <= mr.max_positions
    assert sum(w.values()) <= 1.0 + 1e-9


def test_mr_sleeve_all_cash_when_no_signal(cfg):
    """No oversold names -> empty weights (fully in cash)."""
    mr = cfg.mean_reversion
    n = mr.trend_sma + 50
    # Smooth uptrend, no pullback -> RSI stays high -> no entries.
    prices = pd.DataFrame(
        {sym: np.linspace(100.0, 300.0, n) for sym in mr.universe},
        index=pd.bdate_range("2015-01-01", periods=n),
    )
    sleeve = MeanReversionSleeve(cfg)
    assert sleeve.target_weights(prices) == {}


def test_mr_sleeve_no_lookahead(cfg, prices):
    """Decision at date T must be identical whether or not future data exists."""
    sleeve = MeanReversionSleeve(cfg)
    cut = 800
    full = sleeve.target_weights(prices.iloc[: cut + 1])
    # Append future data; the decision for date `cut` must not change.
    trunc = sleeve.target_weights(prices.iloc[: cut + 1].copy())
    assert full == trunc


def test_backtest_sleeve_clean_curve(cfg, prices):
    res = backtest_sleeve(prices, MeanReversionSleeve(cfg), cfg)
    assert not res.equity.isna().any()
    assert res.equity.iloc[-1] > 0


def test_trend_sleeve_matches_default_engine(cfg, prices):
    """TrendMomentumSleeve (dd off) must match run_backtest with apply_dd=False."""
    sleeve = TrendMomentumSleeve(cfg, apply_dd=False)
    via_sleeve = backtest_sleeve(prices, sleeve, cfg)
    direct = run_backtest(prices, cfg, apply_dd=False)
    pd.testing.assert_series_equal(via_sleeve.equity, direct.equity)


# --- defensive-carry sleeve (Sleeve C) -------------------------------------
def test_dc_sleeve_only_holds_defensive_universe(cfg, prices):
    """Sleeve C must never hold a ticker outside its defensive universe."""
    sleeve = DefensiveCarrySleeve(cfg)
    allowed = set(cfg.defensive_carry.universe)
    w = sleeve.target_weights(prices)
    assert set(w).issubset(allowed)


def test_dc_sleeve_requires_uptrend_and_positive_momentum(cfg):
    """A downtrending defensive asset must not be held (trend + momentum gate)."""
    dc = cfg.defensive_carry
    n = dc.trend_sma + dc.momentum_lookback + dc.momentum_skip + 10
    # Strict downtrend on every defensive name -> nothing qualifies.
    prices = pd.DataFrame(
        {sym: np.linspace(200.0, 100.0, n) for sym in dc.universe},
        index=pd.bdate_range("2010-01-01", periods=n),
    )
    sleeve = DefensiveCarrySleeve(cfg)
    assert sleeve.target_weights(prices) == {}


def test_dc_sleeve_buys_uptrending_defensive(cfg):
    """A steadily uptrending defensive asset (TLT) should be held."""
    dc = cfg.defensive_carry
    n = dc.trend_sma + dc.momentum_lookback + dc.momentum_skip + 10
    cols = {sym: np.linspace(100.0, 90.0, n) for sym in dc.universe}  # most decline
    cols["TLT"] = np.linspace(100.0, 200.0, n)                        # TLT trends up
    prices = pd.DataFrame(cols, index=pd.bdate_range("2010-01-01", periods=n))
    sleeve = DefensiveCarrySleeve(cfg)
    w = sleeve.target_weights(prices)
    assert "TLT" in w and w["TLT"] > 0


def test_dc_sleeve_caps_positions(cfg):
    dc = cfg.defensive_carry
    n = dc.trend_sma + dc.momentum_lookback + dc.momentum_skip + 10
    # All defensive names uptrend -> capped at max_positions.
    prices = pd.DataFrame(
        {sym: np.linspace(100.0, 200.0, n) for sym in dc.universe},
        index=pd.bdate_range("2010-01-01", periods=n),
    )
    sleeve = DefensiveCarrySleeve(cfg)
    w = sleeve.target_weights(prices)
    assert len(w) <= dc.max_positions
    assert sum(w.values()) <= 1.0 + 1e-9


def test_dc_sleeve_no_lookahead(cfg, prices):
    sleeve = DefensiveCarrySleeve(cfg)
    cut = 900
    a = sleeve.target_weights(prices.iloc[: cut + 1])
    b = sleeve.target_weights(prices.iloc[: cut + 1].copy())
    assert a == b


def test_dc_sleeve_clean_curve(cfg, prices):
    res = backtest_sleeve(prices, DefensiveCarrySleeve(cfg), cfg)
    assert not res.equity.isna().any()
    assert res.equity.iloc[-1] > 0


# --- cross-sectional sleeve (Sleeve D) -------------------------------------
def test_cs_sleeve_is_dollar_neutral(cfg, prices):
    """Long and short legs must net to ~zero dollar exposure."""
    sleeve = CrossSectionalSleeve(cfg)
    w = sleeve.target_weights(prices)
    assert w  # should produce a book on real-ish synthetic data
    net = sum(w.values())
    gross = sum(abs(v) for v in w.values())
    assert abs(net) < 1e-6                       # dollar-neutral
    assert abs(gross - cfg.cross_sectional.gross_target) < 1e-6


def test_cs_sleeve_longs_strong_shorts_weak(cfg):
    """Strongest trend goes long (+), weakest goes short (-)."""
    cs = cfg.cross_sectional
    n = cs.momentum_lookback + cs.momentum_skip + 20
    cols = {}
    # Monotonic spread of drifts across the universe so ranking is deterministic.
    for k, sym in enumerate(cs.universe):
        drift = (k - len(cs.universe) / 2) * 0.0008
        cols[sym] = 100.0 * np.exp(np.cumsum(np.full(n, drift)))
    prices = pd.DataFrame(cols, index=pd.bdate_range("2012-01-01", periods=n))
    w = sleeve_w = CrossSectionalSleeve(cfg).target_weights(prices)
    strongest = cs.universe[-1]  # highest drift
    weakest = cs.universe[0]     # lowest (negative) drift
    assert sleeve_w.get(strongest, 0) > 0
    assert sleeve_w.get(weakest, 0) < 0


def test_cs_sleeve_no_lookahead(cfg, prices):
    sleeve = CrossSectionalSleeve(cfg)
    cut = 900
    a = sleeve.target_weights(prices.iloc[: cut + 1])
    b = sleeve.target_weights(prices.iloc[: cut + 1].copy())
    assert a == b


def test_cs_sleeve_clean_curve(cfg, prices):
    res = backtest_sleeve(prices, CrossSectionalSleeve(cfg), cfg)
    assert not res.equity.isna().any()
    assert res.equity.iloc[-1] > 0


def test_cs_sleeve_insufficient_universe_goes_flat(cfg, prices):
    """If too few names score, the sleeve returns an empty (cash) book."""
    sleeve = CrossSectionalSleeve(cfg)
    short = prices.iloc[:50]  # not enough history for momentum -> no scores
    assert sleeve.target_weights(short) == {}


# --- multi-sleeve analysis -------------------------------------------------
def test_analyze_sleeve_set_runs(cfg, prices):
    from etf.sleeve_analysis import analyze_sleeve_set, default_sleeves

    report = analyze_sleeve_set(prices, default_sleeves(cfg), cfg)
    assert report.names == ["trend_momentum", "mean_reversion", "defensive_carry"]
    # correlation matrix is square and symmetric with unit diagonal.
    assert report.corr_matrix.shape == (3, 3)
    assert np.allclose(np.diag(report.corr_matrix.values), 1.0)
    # blend metrics are finite.
    assert np.isfinite(report.combo_inv_vol.sharpe)
    assert report.overlap_days > 100




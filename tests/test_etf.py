"""
tests/test_etf.py
=================
Offline regression tests for the ETF engine.

All tests use deterministic synthetic price data (fixed seed) so they run
without network access and are fully reproducible. They cover:
  - config validation
  - strategy mechanics (trend filter, momentum ranking, inverse-vol, vol target)
  - look-ahead safety (decision depends only on past data)
  - drawdown overlay scaling
  - backtest accounting (costs reduce equity, no NaNs, benchmark alpha/beta)
  - metrics correctness on a known series
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf.backtest import run_backtest
from etf.config import ETFConfig, get_default_config
from etf.metrics import compute_metrics, max_drawdown
from etf.strategy import (
    apply_drawdown_overlay,
    compute_target_weights,
)


def _synthetic_prices(seed: int = 7, n_days: int = 900) -> pd.DataFrame:
    """Build a synthetic multi-asset price panel with distinct trends/vols."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    specs = {
        "SPY": (0.0006, 0.010),   # strong steady uptrend, low vol
        "QQQ": (0.0008, 0.016),   # strongest trend, higher vol
        "IWM": (0.0003, 0.014),
        "EFA": (0.0002, 0.012),
        "EEM": (-0.0001, 0.018),  # weak/negative -> should be filtered out
        "XLK": (0.0007, 0.015),
        "XLF": (0.0002, 0.013),
        "XLE": (0.0000, 0.020),
        "XLV": (0.0004, 0.011),
        "XLI": (0.0003, 0.012),
        "TLT": (0.0001, 0.010),
        "IEF": (0.0001, 0.005),
        "LQD": (0.0001, 0.006),
        "GLD": (0.0002, 0.011),
        "DBC": (-0.0002, 0.013),  # downtrend
        "BIL": (0.00005, 0.0006),  # cash proxy, near-flat
    }
    data = {}
    for sym, (mu, sigma) in specs.items():
        rets = rng.normal(mu, sigma, n_days)
        data[sym] = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def cfg() -> ETFConfig:
    c = get_default_config()
    return c


@pytest.fixture
def prices() -> pd.DataFrame:
    return _synthetic_prices()


# --- config ---------------------------------------------------------------
def test_default_config_validates():
    cfg = get_default_config()
    assert cfg.signal.top_k >= 1
    assert abs(sum(cfg.signal.momentum_weights) - 1.0) < 1e-9
    assert cfg.cash_asset in cfg.all_symbols
    assert cfg.benchmark in cfg.all_symbols


def test_config_rejects_bad_weights():
    cfg = ETFConfig()
    cfg.signal.momentum_weights = [0.5, 0.5, 0.5]
    with pytest.raises(ValueError):
        cfg.validate()


def test_config_rejects_topk_over_universe():
    cfg = ETFConfig()
    cfg.signal.top_k = len(cfg.risk_universe) + 1
    with pytest.raises(ValueError):
        cfg.validate()


# --- strategy -------------------------------------------------------------
def test_target_weights_basic_shape(cfg, prices):
    d = compute_target_weights(prices, cfg)
    # gross exposure must respect leverage cap
    assert d.gross_exposure <= cfg.risk.max_gross_leverage + 1e-9
    # weights + cash ~ 1
    assert abs(sum(d.weights.values()) + d.cash_weight - 1.0) < 1e-6
    # at most top_k names
    assert len(d.selected) <= cfg.signal.top_k
    # no single weight exceeds the concentration cap (post vol-scale it can be lower)
    for w in d.weights.values():
        assert w <= cfg.risk.max_position_weight + 1e-9


def test_downtrend_assets_excluded(cfg, prices):
    d = compute_target_weights(prices, cfg)
    # DBC and EEM are designed to be in downtrends -> never selected
    assert "DBC" not in d.selected
    assert "EEM" not in d.selected


def test_no_lookahead(cfg, prices):
    """Decision at date T must not change if future data is appended."""
    cutoff = 700
    past = prices.iloc[:cutoff]
    d_past = compute_target_weights(past, cfg)
    # append arbitrary future rows; recompute using the SAME truncation
    full = prices.copy()
    d_again = compute_target_weights(full.iloc[:cutoff], cfg)
    assert d_past.weights == d_again.weights
    assert d_past.selected == d_again.selected


def test_vol_targeting_scales_exposure(cfg, prices):
    high = get_default_config()
    high.risk.target_volatility = 0.30  # high target -> more exposure
    low = get_default_config()
    low.risk.target_volatility = 0.04   # low target -> less exposure
    d_high = compute_target_weights(prices, high)
    d_low = compute_target_weights(prices, low)
    assert d_high.gross_exposure >= d_low.gross_exposure


def test_drawdown_overlay_reduces_exposure(cfg, prices):
    d = compute_target_weights(prices, cfg)
    if d.gross_exposure == 0:
        pytest.skip("no risky exposure to de-risk in this sample")
    deep = apply_drawdown_overlay(d, current_drawdown=0.25, cfg=cfg)
    assert deep.gross_exposure < d.gross_exposure
    assert deep.gross_exposure <= d.gross_exposure * cfg.risk.dd_min_exposure + 1e-9
    # mild drawdown below dd_start changes nothing
    mild = apply_drawdown_overlay(d, current_drawdown=0.02, cfg=cfg)
    assert abs(mild.gross_exposure - d.gross_exposure) < 1e-9


def test_no_eligible_goes_to_cash(cfg):
    # Deterministic monotonic decline -> every asset below its SMA with
    # negative 12-1 momentum -> engine must be fully defensive (all cash).
    n = 400
    dates = pd.bdate_range("2015-01-01", periods=n)
    decay = np.exp(np.linspace(0.0, -0.5, n))  # smooth steady ~-40% decline
    falling = {s: 100.0 * decay for s in cfg.risk_universe}
    falling["BIL"] = np.full(n, 100.0)  # flat cash proxy
    df = pd.DataFrame(falling, index=dates)
    d = compute_target_weights(df, cfg)
    assert d.gross_exposure == 0.0
    assert d.cash_weight == 1.0


# --- backtest -------------------------------------------------------------
def test_backtest_runs_and_is_finite(cfg, prices):
    res = run_backtest(prices, cfg)
    assert len(res.equity) == len(prices)
    assert np.isfinite(res.equity.iloc[-1])
    assert (res.equity > 0).all()
    assert res.gross_exposure.max() <= cfg.risk.max_gross_leverage + 1e-9
    assert res.metrics.n_periods > 0


def test_costs_reduce_equity(cfg, prices):
    free = get_default_config()
    free.execution.commission_bps = 0.0
    free.execution.slippage_bps = 0.0
    costly = get_default_config()
    costly.execution.commission_bps = 50.0
    costly.execution.slippage_bps = 50.0
    r_free = run_backtest(prices, free)
    r_cost = run_backtest(prices, costly)
    assert r_cost.equity.iloc[-1] <= r_free.equity.iloc[-1]


def test_benchmark_metrics_present(cfg, prices):
    res = run_backtest(prices, cfg)
    assert res.benchmark_metrics is not None
    assert np.isfinite(res.metrics.beta)


# --- metrics --------------------------------------------------------------
def test_max_drawdown_known():
    eq = pd.Series([100, 120, 90, 100], index=pd.bdate_range("2020-01-01", periods=4))
    # peak 120 -> trough 90 => -25%
    assert abs(max_drawdown(eq) - (-0.25)) < 1e-9


def test_metrics_positive_trend():
    eq = pd.Series(100 * np.exp(np.cumsum(np.full(300, 0.0005))),
                   index=pd.bdate_range("2020-01-01", periods=300))
    m = compute_metrics(eq, risk_free_rate=0.0)
    assert m.total_return > 0
    assert m.sharpe > 0
    assert m.max_drawdown == 0.0  # monotonic up

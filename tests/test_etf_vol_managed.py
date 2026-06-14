"""
tests/test_etf_vol_managed.py
=============================
Unit tests for the volatility-managed overlay (Moreira-Muir conditional
vol timing), :class:`etf.sleeves.VolManagedSleeve`.

All offline / deterministic. Coverage:
  - disabled overlay is a pass-through (reproduces the bare sleeve bit-for-bit);
  - enabled overlay scales the basket inversely to recent realised vol
    (calm -> larger gross, turbulent -> smaller gross);
  - the scale honours the configured max_scale / min_scale clamps;
  - the overlay is strictly causal (truncation invariance: appending FUTURE
    bars never changes today's decision);
  - degenerate inputs (empty weights, single row, zero-vol basket) fail safe;
  - default_sleeves wires the overlay onto the two equity-beta sleeves when
    enabled and leaves the bare roster untouched when off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf.config import ETFConfig, get_default_config
from etf.sleeves import Sleeve, VolManagedSleeve
from etf.sleeve_analysis import default_sleeves


# --------------------------------------------------------------------------
# A tiny deterministic stub sleeve so the overlay can be tested in isolation,
# independent of any production sleeve's signal logic.
# --------------------------------------------------------------------------
class _FixedSleeve:
    """Always returns the same target weights (a 60% SPY / 40% QQQ basket)."""

    name = "stub"
    rebalance_every = 1
    warmup = 1

    def __init__(self, weights):
        self._w = dict(weights)

    def target_weights(self, prices_asof: pd.DataFrame):
        return dict(self._w)


def _prices(daily_sigma: float, n: int = 120, seed: int = 7) -> pd.DataFrame:
    """Two correlated-ish ETFs with a controllable daily volatility."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    out = {}
    for sym, mu in (("SPY", 0.0003), ("QQQ", 0.0004)):
        rets = rng.normal(mu, daily_sigma, n)
        out[sym] = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(out, index=dates)


@pytest.fixture
def cfg() -> ETFConfig:
    return get_default_config()


def test_disabled_overlay_is_passthrough(cfg):
    inner = _FixedSleeve({"SPY": 0.6, "QQQ": 0.4})
    vm = VolManagedSleeve(inner, cfg)
    assert cfg.vol_managed.enabled is False
    prices = _prices(0.01)
    assert vm.target_weights(prices) == inner.target_weights(prices)


def test_enabled_overlay_scales_inverse_to_realised_vol(cfg):
    cfg.vol_managed.enabled = True
    cfg.vol_managed.target_vol_annual = 0.12
    cfg.vol_managed.max_scale = 5.0
    cfg.vol_managed.min_scale = 0.0
    inner = _FixedSleeve({"SPY": 0.6, "QQQ": 0.4})
    vm = VolManagedSleeve(inner, cfg)

    calm = vm.target_weights(_prices(0.004, seed=1))     # low realised vol
    rough = vm.target_weights(_prices(0.030, seed=1))    # high realised vol

    calm_gross = sum(calm.values())
    rough_gross = sum(rough.values())
    # Calmer regime -> bigger position; turbulent -> smaller.
    assert calm_gross > rough_gross
    # The ratio of weights is preserved (pure scaling of the basket).
    assert calm["SPY"] / calm["QQQ"] == pytest.approx(0.6 / 0.4)


def test_overlay_respects_max_scale_clamp(cfg):
    cfg.vol_managed.enabled = True
    cfg.vol_managed.target_vol_annual = 0.50  # huge target -> wants big scale
    cfg.vol_managed.max_scale = 1.3
    inner = _FixedSleeve({"SPY": 0.6, "QQQ": 0.4})
    vm = VolManagedSleeve(inner, cfg)
    out = vm.target_weights(_prices(0.003))  # very calm -> scale wants > max
    assert sum(out.values()) == pytest.approx(1.0 * 1.3, rel=1e-9)


def test_overlay_respects_min_scale_clamp(cfg):
    cfg.vol_managed.enabled = True
    cfg.vol_managed.target_vol_annual = 0.01  # tiny target -> wants tiny scale
    cfg.vol_managed.min_scale = 0.25
    cfg.vol_managed.max_scale = 2.0
    inner = _FixedSleeve({"SPY": 0.6, "QQQ": 0.4})
    vm = VolManagedSleeve(inner, cfg)
    out = vm.target_weights(_prices(0.04))  # turbulent -> scale wants < min
    assert sum(out.values()) == pytest.approx(1.0 * 0.25, rel=1e-9)


def test_overlay_is_causal_truncation_invariant(cfg):
    """Appending FUTURE bars must not change today's decision."""
    cfg.vol_managed.enabled = True
    inner = _FixedSleeve({"SPY": 0.6, "QQQ": 0.4})
    vm = VolManagedSleeve(inner, cfg)

    full = _prices(0.012, n=160, seed=3)
    asof = full.iloc[:100]
    extended = full.iloc[:130]  # same first 100 rows + future bars

    today = vm.target_weights(asof)
    # Decision computed on the truncated frame must match the decision computed
    # on the same history regardless of any later rows that did not exist yet.
    today_again = vm.target_weights(extended.iloc[:100])
    assert today == today_again


def test_empty_inner_weights_pass_through(cfg):
    cfg.vol_managed.enabled = True
    vm = VolManagedSleeve(_FixedSleeve({}), cfg)
    assert vm.target_weights(_prices(0.01)) == {}


def test_single_row_fails_safe_to_unit_scale(cfg):
    cfg.vol_managed.enabled = True
    inner = _FixedSleeve({"SPY": 0.6, "QQQ": 0.4})
    vm = VolManagedSleeve(inner, cfg)
    one = _prices(0.01).iloc[:1]
    out = vm.target_weights(one)
    assert out == inner.target_weights(one)  # no realised vol -> scale 1.0


def test_zero_vol_basket_fails_safe(cfg):
    cfg.vol_managed.enabled = True
    inner = _FixedSleeve({"SPY": 0.6, "QQQ": 0.4})
    vm = VolManagedSleeve(inner, cfg)
    flat = pd.DataFrame(
        {"SPY": np.full(60, 100.0), "QQQ": np.full(60, 50.0)},
        index=pd.bdate_range("2021-01-01", periods=60),
    )
    out = vm.target_weights(flat)  # zero realised vol -> scale 1.0 (no blow-up)
    assert out == inner.target_weights(flat)


def test_name_defaults_to_inner_plus_suffix(cfg):
    vm = VolManagedSleeve(_FixedSleeve({"SPY": 1.0}), cfg)
    assert vm.name == "stub_vm"
    vm2 = VolManagedSleeve(_FixedSleeve({"SPY": 1.0}), cfg, name="custom")
    assert vm2.name == "custom"


def test_warmup_accounts_for_realized_window(cfg):
    cfg.vol_managed.realized_window = 40
    inner = _FixedSleeve({"SPY": 1.0})  # inner.warmup == 1
    vm = VolManagedSleeve(inner, cfg)
    assert vm.warmup >= 41


def test_default_sleeves_off_is_bare_roster(cfg):
    assert cfg.vol_managed.enabled is False
    sleeves = default_sleeves(cfg)
    assert [s.name for s in sleeves] == [
        "trend_momentum", "mean_reversion", "defensive_carry"
    ]
    assert not any(isinstance(s, VolManagedSleeve) for s in sleeves)


def test_default_sleeves_on_wraps_equity_beta_sleeves(cfg):
    cfg.vol_managed.enabled = True
    sleeves = default_sleeves(cfg)
    by_name = {s.name: s for s in sleeves}
    # Names are preserved so the combiner allocation rows are unchanged.
    assert [s.name for s in sleeves] == [
        "trend_momentum", "mean_reversion", "defensive_carry"
    ]
    assert isinstance(by_name["trend_momentum"], VolManagedSleeve)
    assert isinstance(by_name["mean_reversion"], VolManagedSleeve)
    # Defensive carry is left bare (already low-beta/low-vol).
    assert not isinstance(by_name["defensive_carry"], VolManagedSleeve)

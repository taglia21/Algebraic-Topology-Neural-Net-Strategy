"""
tests/test_etf_safety.py
========================
Unit tests for the ETF-native live-safety layer (etf/safety.py): the pre-trade
kill-switch and slippage telemetry. All pure-function tests — no live IBKR.
"""

import json
from dataclasses import dataclass

import pytest

from etf.config import ETFConfig
from etf.safety import (
    SafetyDecision,
    SlippageReport,
    pretrade_safety_check,
    compute_slippage,
    log_slippage,
    _adverse_slippage_bps,
)


@dataclass
class _Order:
    """Minimal duck-typed stand-in for ibkr_broker.PlannedOrder."""
    symbol: str
    action: str
    quantity: int
    est_price: float


@pytest.fixture
def cfg():
    return ETFConfig()


# ---------------------------------------------------------------------------
# pretrade_safety_check — happy path
# ---------------------------------------------------------------------------
def test_safety_allows_normal_conditions(cfg):
    d = pretrade_safety_check(
        cfg, current_drawdown=0.05, daily_pnl_pct=-0.01,
        gross_exposure=0.8, reconciliation_ok=True, data_is_fresh=True,
    )
    assert isinstance(d, SafetyDecision)
    assert d.allowed
    assert not d.halt
    assert d.reasons == []


# ---------------------------------------------------------------------------
# Catastrophic halts (kill-switch)
# ---------------------------------------------------------------------------
def test_safety_halts_on_hard_drawdown(cfg):
    # Default hard_halt_drawdown = 0.25.
    d = pretrade_safety_check(
        cfg, current_drawdown=0.25, daily_pnl_pct=0.0,
        gross_exposure=0.5, reconciliation_ok=True, data_is_fresh=True,
    )
    assert not d.allowed
    assert d.halt
    assert any("drawdown" in r.lower() for r in d.reasons)


def test_safety_halts_on_daily_loss(cfg):
    # Default max_daily_loss = 0.08.
    d = pretrade_safety_check(
        cfg, current_drawdown=0.02, daily_pnl_pct=-0.09,
        gross_exposure=0.5, reconciliation_ok=True, data_is_fresh=True,
    )
    assert not d.allowed
    assert d.halt
    assert any("daily loss" in r.lower() for r in d.reasons)


def test_drawdown_just_below_halt_is_allowed(cfg):
    d = pretrade_safety_check(
        cfg, current_drawdown=0.2499, daily_pnl_pct=0.0,
        gross_exposure=0.5, reconciliation_ok=True, data_is_fresh=True,
    )
    assert d.allowed
    assert not d.halt


# ---------------------------------------------------------------------------
# Recoverable blocks (no halt, skip cycle)
# ---------------------------------------------------------------------------
def test_safety_blocks_on_stale_data(cfg):
    d = pretrade_safety_check(
        cfg, current_drawdown=0.0, daily_pnl_pct=0.0,
        gross_exposure=0.5, reconciliation_ok=True, data_is_fresh=False,
    )
    assert not d.allowed
    assert not d.halt  # recoverable, not catastrophic
    assert any("stale" in r.lower() or "missing" in r.lower() for r in d.reasons)


def test_safety_blocks_on_reconciliation_mismatch(cfg):
    d = pretrade_safety_check(
        cfg, current_drawdown=0.0, daily_pnl_pct=0.0,
        gross_exposure=0.5, reconciliation_ok=False, data_is_fresh=True,
    )
    assert not d.allowed
    assert not d.halt
    assert any("reconciliation" in r.lower() for r in d.reasons)


def test_safety_blocks_on_gross_over_cap(cfg):
    # Default max_gross_leverage = 1.0; 1.5 is over the cap.
    d = pretrade_safety_check(
        cfg, current_drawdown=0.0, daily_pnl_pct=0.0,
        gross_exposure=1.5, reconciliation_ok=True, data_is_fresh=True,
    )
    assert not d.allowed
    assert not d.halt
    assert any("leverage" in r.lower() or "gross" in r.lower() for r in d.reasons)


def test_gross_at_cap_is_allowed(cfg):
    # Exactly at the cap (within float tolerance) must pass.
    d = pretrade_safety_check(
        cfg, current_drawdown=0.0, daily_pnl_pct=0.0,
        gross_exposure=1.0, reconciliation_ok=True, data_is_fresh=True,
    )
    assert d.allowed


def test_halt_takes_priority_and_accumulates_reasons(cfg):
    # Both a halt AND a block condition -> not allowed, halt True, two reasons.
    d = pretrade_safety_check(
        cfg, current_drawdown=0.30, daily_pnl_pct=0.0,
        gross_exposure=2.0, reconciliation_ok=True, data_is_fresh=True,
    )
    assert not d.allowed
    assert d.halt
    assert len(d.reasons) >= 2


# ---------------------------------------------------------------------------
# _adverse_slippage_bps
# ---------------------------------------------------------------------------
def test_buy_above_expected_is_adverse():
    # Paid 100.10 vs expected 100.00 -> +10 bps adverse.
    assert _adverse_slippage_bps("BUY", 100.0, 100.10) == pytest.approx(10.0, abs=1e-6)


def test_buy_below_expected_is_improvement():
    # Paid less than expected -> negative (price improvement).
    assert _adverse_slippage_bps("BUY", 100.0, 99.95) == pytest.approx(-5.0, abs=1e-6)


def test_sell_below_expected_is_adverse():
    # Received 99.90 vs expected 100.00 -> +10 bps adverse.
    assert _adverse_slippage_bps("SELL", 100.0, 99.90) == pytest.approx(10.0, abs=1e-6)


def test_adverse_slippage_nonpositive_expected_raises():
    with pytest.raises(ValueError):
        _adverse_slippage_bps("BUY", 0.0, 100.0)


# ---------------------------------------------------------------------------
# compute_slippage
# ---------------------------------------------------------------------------
def test_compute_slippage_aggregates(cfg):
    orders = [
        _Order("SPY", "BUY", 10, 100.0),   # fill 100.10 -> +10 bps, notional 1000
        _Order("QQQ", "SELL", 5, 200.0),   # fill 199.80 -> +10 bps, notional 1000
    ]
    fills = {"SPY": 100.10, "QQQ": 199.80}
    rep = compute_slippage(orders, fills, cfg)
    assert isinstance(rep, SlippageReport)
    assert len(rep.records) == 2
    assert rep.total_notional == pytest.approx(2000.0)
    # Both +10 bps, equal notional -> weighted avg 10 bps.
    assert rep.avg_slippage_bps == pytest.approx(10.0, abs=1e-6)
    assert rep.worst_slippage_bps == pytest.approx(10.0, abs=1e-6)
    # Cost = 10bps * 2000 notional = $2.00.
    assert rep.total_cost_usd == pytest.approx(2.0, abs=1e-6)
    # Modeled budget 2.0bps*1.2 = 2.4bps; realised 10bps -> OVER tolerance.
    assert not rep.within_tolerance


def test_compute_slippage_within_tolerance(cfg):
    orders = [_Order("SPY", "BUY", 10, 100.0)]
    fills = {"SPY": 100.02}  # +2 bps, budget 2.4 bps -> within
    rep = compute_slippage(orders, fills, cfg)
    assert rep.avg_slippage_bps == pytest.approx(2.0, abs=1e-6)
    assert rep.within_tolerance


def test_compute_slippage_skips_unfilled(cfg):
    orders = [
        _Order("SPY", "BUY", 10, 100.0),
        _Order("QQQ", "BUY", 5, 200.0),   # no fill -> skipped
    ]
    fills = {"SPY": 100.05}
    rep = compute_slippage(orders, fills, cfg)
    assert len(rep.records) == 1
    assert rep.records[0].symbol == "SPY"


def test_compute_slippage_empty_when_no_fills(cfg):
    orders = [_Order("SPY", "BUY", 10, 100.0)]
    rep = compute_slippage(orders, {}, cfg)
    assert rep.records == []
    assert rep.total_notional == 0.0
    assert rep.avg_slippage_bps == 0.0
    assert rep.within_tolerance  # nothing traded -> trivially within budget


# ---------------------------------------------------------------------------
# log_slippage
# ---------------------------------------------------------------------------
def test_log_slippage_writes_jsonl(cfg, tmp_path):
    orders = [_Order("SPY", "BUY", 10, 100.0)]
    rep = compute_slippage(orders, {"SPY": 100.05}, cfg)
    path = tmp_path / "telemetry" / "slip.jsonl"
    log_slippage(rep, path, as_of="2026-06-14T00:00:00Z")
    log_slippage(rep, path, as_of="2026-06-15T00:00:00Z")
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2  # appended, not overwritten
    row = json.loads(lines[0])
    assert row["as_of"] == "2026-06-14T00:00:00Z"
    assert row["n_fills"] == 1
    assert row["fills"][0]["symbol"] == "SPY"


def test_log_slippage_failsafe_on_bad_path(cfg):
    # A path under a file (not a dir) would raise on mkdir; log_slippage must swallow it.
    orders = [_Order("SPY", "BUY", 10, 100.0)]
    rep = compute_slippage(orders, {"SPY": 100.05}, cfg)
    # Should not raise even with an obviously invalid path component.
    log_slippage(rep, "/proc/cannot/write/here/slip.jsonl")

"""
tests/test_etf_state.py
=======================
Unit tests for the persistent equity-state tracker (etf/state.py): the pure
``update_state`` drawdown / daily-P&L logic and atomic load/save round-trips.
"""

from datetime import datetime, timezone

import pytest

from etf.state import (
    EquityState,
    ReconciliationState,
    load_reconciliation_state,
    load_state,
    save_reconciliation_state,
    save_state,
    update_state,
)


def _dt(y, m, d, h=16):
    return datetime(y, m, d, h, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# update_state — first observation
# ---------------------------------------------------------------------------
def test_first_observation_has_no_drawdown_or_pnl():
    state, dd, pnl = update_state(None, 100_000.0, now=_dt(2026, 6, 14))
    assert dd == 0.0
    assert pnl == 0.0
    assert state.peak_equity == 100_000.0
    assert state.sod_equity == 100_000.0
    assert state.last_equity == 100_000.0
    assert state.sod_date == "2026-06-14"


def test_nonpositive_equity_raises():
    with pytest.raises(ValueError):
        update_state(None, 0.0)
    with pytest.raises(ValueError):
        update_state(None, -5.0)


# ---------------------------------------------------------------------------
# update_state — intraday (same day) updates
# ---------------------------------------------------------------------------
def test_intraday_gain_updates_peak_and_daily_pnl():
    s0, _, _ = update_state(None, 100_000.0, now=_dt(2026, 6, 14))
    s1, dd, pnl = update_state(s0, 102_000.0, now=_dt(2026, 6, 14, 17))
    # Same day -> sod baseline unchanged at 100k; +2% on the day.
    assert pnl == pytest.approx(0.02, abs=1e-9)
    # New high -> no drawdown.
    assert dd == 0.0
    assert s1.peak_equity == 102_000.0
    assert s1.sod_equity == 100_000.0
    assert s1.sod_date == "2026-06-14"


def test_intraday_loss_produces_drawdown_and_negative_pnl():
    s0, _, _ = update_state(None, 100_000.0, now=_dt(2026, 6, 14))
    s1, dd, pnl = update_state(s0, 95_000.0, now=_dt(2026, 6, 14, 17))
    assert pnl == pytest.approx(-0.05, abs=1e-9)
    # Peak stays at 100k -> 5% drawdown.
    assert dd == pytest.approx(0.05, abs=1e-9)
    assert s1.peak_equity == 100_000.0


# ---------------------------------------------------------------------------
# update_state — day rollover
# ---------------------------------------------------------------------------
def test_day_rollover_resets_sod_baseline_to_prior_close():
    # Day 1: open at 100k, close at 110k (a +10% day).
    s0, _, _ = update_state(None, 100_000.0, now=_dt(2026, 6, 14))
    s1, _, _ = update_state(s0, 110_000.0, now=_dt(2026, 6, 14, 20))
    # Day 2: equity now 104.5k. Baseline should roll to prior close (110k).
    s2, dd, pnl = update_state(s1, 104_500.0, now=_dt(2026, 6, 15, 16))
    assert s2.sod_date == "2026-06-15"
    assert s2.sod_equity == pytest.approx(110_000.0)
    # Daily P&L = 104.5k / 110k - 1 = -5%.
    assert pnl == pytest.approx(-0.05, abs=1e-9)
    # Drawdown from peak 110k -> 5%.
    assert dd == pytest.approx(0.05, abs=1e-9)


def test_peak_is_high_water_mark_across_days():
    s, _, _ = update_state(None, 100_000.0, now=_dt(2026, 6, 14))
    s, _, _ = update_state(s, 120_000.0, now=_dt(2026, 6, 14, 20))  # peak 120k
    s, _, _ = update_state(s, 90_000.0, now=_dt(2026, 6, 15))       # drop
    s, dd, _ = update_state(s, 96_000.0, now=_dt(2026, 6, 16))      # partial recover
    # Peak held at 120k -> drawdown = 1 - 96/120 = 20%.
    assert s.peak_equity == pytest.approx(120_000.0)
    assert dd == pytest.approx(0.20, abs=1e-9)


# ---------------------------------------------------------------------------
# Persistence — atomic save / load round-trip
# ---------------------------------------------------------------------------
def test_save_load_roundtrip(tmp_path):
    state, _, _ = update_state(None, 100_000.0, now=_dt(2026, 6, 14))
    path = tmp_path / "telemetry" / "equity_state.json"
    save_state(state, path)
    loaded = load_state(path)
    assert loaded is not None
    assert loaded.peak_equity == state.peak_equity
    assert loaded.sod_equity == state.sod_equity
    assert loaded.sod_date == state.sod_date
    assert loaded.last_equity == state.last_equity


def test_load_missing_file_returns_none(tmp_path):
    assert load_state(tmp_path / "does_not_exist.json") is None


def test_load_corrupt_file_returns_none(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ this is not valid json")
    assert load_state(p) is None


def test_save_failsafe_on_bad_path():
    # Saving under an unwritable path must not raise (best-effort I/O).
    state, _, _ = update_state(None, 100_000.0)
    save_state(state, "/proc/cannot/write/here/state.json")


def test_persisted_state_survives_restart_simulation(tmp_path):
    # Simulate: run 1 records a peak, process restarts, run 2 loads and sees DD.
    path = tmp_path / "equity_state.json"
    s0, _, _ = update_state(None, 100_000.0, now=_dt(2026, 6, 14))
    s1, _, _ = update_state(s0, 120_000.0, now=_dt(2026, 6, 14, 20))
    save_state(s1, path)
    # ---- restart ----
    reloaded = load_state(path)
    s2, dd, _ = update_state(reloaded, 102_000.0, now=_dt(2026, 6, 15))
    # Peak 120k preserved across the "restart" -> 15% drawdown detected.
    assert dd == pytest.approx(0.15, abs=1e-9)


# ---------------------------------------------------------------------------
# Reconciliation state — cross-cycle safety memory
# ---------------------------------------------------------------------------
def test_load_missing_reconciliation_state_returns_none(tmp_path):
    # No prior cycle => None (caller treats as reconciled, free to trade).
    assert load_reconciliation_state(tmp_path / "missing.json") is None


def test_reconciliation_ok_roundtrip(tmp_path):
    path = tmp_path / "recon.json"
    save_reconciliation_state(True, {}, path, now=_dt(2026, 6, 14))
    loaded = load_reconciliation_state(path)
    assert isinstance(loaded, ReconciliationState)
    assert loaded.ok is True
    assert loaded.mismatches == {}


def test_reconciliation_mismatch_roundtrip(tmp_path):
    path = tmp_path / "recon.json"
    save_reconciliation_state(False, {"SPY": 0.05, "TLT": 0.03}, path, now=_dt(2026, 6, 14))
    loaded = load_reconciliation_state(path)
    assert loaded.ok is False
    # Mismatches are preserved (as floats) for the runbook / audit trail.
    assert loaded.mismatches["SPY"] == pytest.approx(0.05)
    assert loaded.mismatches["TLT"] == pytest.approx(0.03)


def test_reconciliation_mismatch_blocks_next_cycle_simulation(tmp_path):
    # A mismatch persisted on cycle N must be visible (ok=False) on cycle N+1.
    path = tmp_path / "recon.json"
    save_reconciliation_state(False, {"QQQ": 0.10}, path, now=_dt(2026, 6, 14))
    # ---- next cycle (after restart) ----
    prior = load_reconciliation_state(path)
    assert prior is not None and prior.ok is False
    # ...then a human resolves and the next OK reconciliation clears the block.
    save_reconciliation_state(True, {}, path, now=_dt(2026, 6, 15))
    assert load_reconciliation_state(path).ok is True


def test_save_reconciliation_failsafe_on_bad_path():
    # Best-effort I/O: an unwritable path must not raise.
    save_reconciliation_state(False, {"SPY": 0.2}, "/proc/cannot/write/here/recon.json")


def test_load_corrupt_reconciliation_returns_none(tmp_path):
    p = tmp_path / "bad_recon.json"
    p.write_text("{ not valid json")
    assert load_reconciliation_state(p) is None

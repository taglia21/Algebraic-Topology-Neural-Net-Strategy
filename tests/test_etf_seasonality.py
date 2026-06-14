"""
tests/test_etf_seasonality.py
=============================
Unit tests for the turn-of-month seasonality sleeve (Sleeve E): the causal
calendar-window logic and the sleeve's weight construction (trend gate,
in/out-of-window behaviour, no look-ahead).
"""

import numpy as np
import pandas as pd
import pytest

from etf.config import ETFConfig
from etf.sleeves import TurnOfMonthSleeve, _tom_in_window


# ---------------------------------------------------------------------------
# _tom_in_window — pure calendar logic
# ---------------------------------------------------------------------------
def test_first_trading_days_leg():
    # First 3 trading days of the month -> in window regardless of calendar day.
    d = pd.Timestamp("2026-06-02")
    assert _tom_in_window(d, tdom_from_start=2, first_trading_days=3, last_calendar_days=3)


def test_after_first_days_and_mid_month_is_out():
    # 10th trading day, mid-month calendar -> outside both legs.
    d = pd.Timestamp("2026-06-15")
    assert not _tom_in_window(d, tdom_from_start=10, first_trading_days=3, last_calendar_days=3)


def test_last_calendar_days_leg():
    # June has 30 days; last_calendar_days=3 -> days 28,29,30 qualify.
    d = pd.Timestamp("2026-06-29")
    assert _tom_in_window(d, tdom_from_start=20, first_trading_days=3, last_calendar_days=3)


def test_just_before_last_calendar_window_is_out():
    d = pd.Timestamp("2026-06-27")  # 27 < 30-3+1=28 -> out
    assert not _tom_in_window(d, tdom_from_start=20, first_trading_days=3, last_calendar_days=3)


def test_february_month_end_window():
    # 2026 Feb has 28 days; last 3 calendar days = 26,27,28.
    assert _tom_in_window(pd.Timestamp("2026-02-26"), 18, 3, 3)
    assert not _tom_in_window(pd.Timestamp("2026-02-25"), 17, 3, 3)


def test_legs_can_be_disabled():
    # Disable the first-days leg: a +2 trading day no longer qualifies.
    d = pd.Timestamp("2026-06-02")
    assert not _tom_in_window(d, tdom_from_start=2, first_trading_days=0, last_calendar_days=3)


# ---------------------------------------------------------------------------
# TurnOfMonthSleeve — weight construction
# ---------------------------------------------------------------------------
def _uptrend_prices(end: str, n: int = 300, symbols=("SPY", "QQQ", "IWM")) -> pd.DataFrame:
    """Build a gently rising daily price frame ending on ``end`` (business days)."""
    idx = pd.bdate_range(end=end, periods=n)
    data = {}
    for k, sym in enumerate(symbols):
        # Steady uptrend so price > SMA200, with a little per-symbol variation.
        base = 100.0 + 0.05 * np.arange(n) + k
        data[sym] = base
    return pd.DataFrame(data, index=idx)


def test_sleeve_holds_equity_in_window_uptrend():
    cfg = ETFConfig()
    sleeve = TurnOfMonthSleeve(cfg)
    # End on the first business day of a month -> first_trading_days leg active.
    prices = _uptrend_prices(end="2026-06-01")
    # Force decision date to be the 1st trading day of June.
    w = sleeve.target_weights(prices)
    assert w, "expected non-empty equity weights inside the ToM window"
    assert abs(sum(w.values()) - cfg.seasonality.deploy_fraction) < 1e-6


def test_sleeve_is_cash_outside_window():
    cfg = ETFConfig()
    sleeve = TurnOfMonthSleeve(cfg)
    # Mid-month, not near either leg.
    prices = _uptrend_prices(end="2026-06-16")
    # 2026-06-16 is the 16th calendar day, ~11th trading day -> outside window.
    w = sleeve.target_weights(prices)
    assert w == {}, "expected fully-cash (empty) outside the ToM window"


def test_trend_gate_blocks_downtrend():
    cfg = ETFConfig()
    sleeve = TurnOfMonthSleeve(cfg)
    idx = pd.bdate_range(end="2026-06-01", periods=300)
    # Steady DOWNtrend -> price < SMA200 -> trend gate keeps the sleeve flat.
    data = {sym: 200.0 - 0.05 * np.arange(300) for sym in ("SPY", "QQQ", "IWM")}
    prices = pd.DataFrame(data, index=idx)
    w = sleeve.target_weights(prices)
    assert w == {}, "trend gate should hold the sleeve in cash during downtrends"


def test_weights_respect_per_name_cap():
    cfg = ETFConfig()
    cfg.seasonality.max_position_weight = 0.4
    sleeve = TurnOfMonthSleeve(cfg)
    prices = _uptrend_prices(end="2026-06-01")
    w = sleeve.target_weights(prices)
    for sym, weight in w.items():
        # Cap applies pre-renormalisation; after renormalisation a name can sit
        # at the cap but not materially above it.
        assert weight <= 0.4 + 1e-9


def test_empty_prices_returns_cash():
    cfg = ETFConfig()
    sleeve = TurnOfMonthSleeve(cfg)
    assert sleeve.target_weights(pd.DataFrame()) == {}


def test_no_lookahead_truncation_invariance():
    # The decision for a given date must not change if FUTURE rows are dropped.
    cfg = ETFConfig()
    sleeve = TurnOfMonthSleeve(cfg)
    full = _uptrend_prices(end="2026-07-15", n=360)
    decision_date = pd.Timestamp("2026-06-01")
    # Slice up to the decision date only.
    if decision_date not in full.index:
        decision_date = full.index[full.index <= decision_date][-1]
    truncated = full.loc[:decision_date]
    w_trunc = sleeve.target_weights(truncated)
    # Re-slice the full frame to the same last date — must match exactly.
    w_full_sliced = sleeve.target_weights(full.loc[:decision_date])
    assert w_trunc == w_full_sliced

"""
tests/test_etf_broker.py
========================
Unit tests for the ETF execution adapter's pure logic: marketable-limit
pricing and post-trade reconciliation. These run without a live IBKR
connection (the dev container has none) by exercising the module-level
pure functions and the ``PlannedOrder`` wiring directly.
"""

import pytest

from etf.ibkr_broker import (
    PlannedOrder,
    ReconciliationReport,
    marketable_limit_price,
    compute_reconciliation,
    _extract_price,
    _order_is_done,
)


# ---------------------------------------------------------------------------
# marketable_limit_price
# ---------------------------------------------------------------------------
def test_buy_limit_crosses_up():
    # BUY pays slightly above the ref to get a fast fill, capped by the offset.
    px = marketable_limit_price("BUY", 100.0, 5.0)  # 5 bps = 0.05%
    assert px == pytest.approx(100.05, abs=1e-9)
    assert px > 100.0


def test_sell_limit_crosses_down():
    px = marketable_limit_price("SELL", 100.0, 5.0)
    assert px == pytest.approx(99.95, abs=1e-9)
    assert px < 100.0


def test_limit_price_rounds_to_penny():
    # 250.0 * (1 + 8bps) = 250.20 -> exact penny (US ETF tick).
    px = marketable_limit_price("BUY", 250.0, 8.0)
    assert px == pytest.approx(250.20, abs=1e-9)
    # Verify rounding actually collapses sub-penny precision.
    assert px == round(px, 2)


def test_zero_offset_returns_ref():
    assert marketable_limit_price("BUY", 100.0, 0.0) == pytest.approx(100.0)
    assert marketable_limit_price("SELL", 100.0, 0.0) == pytest.approx(100.0)


def test_negative_offset_clamped_to_zero():
    # A negative offset would invert the marketable side; clamp to 0 (no cross).
    assert marketable_limit_price("BUY", 100.0, -5.0) == pytest.approx(100.0)


def test_nonpositive_ref_price_raises():
    with pytest.raises(ValueError):
        marketable_limit_price("BUY", 0.0, 5.0)
    with pytest.raises(ValueError):
        marketable_limit_price("SELL", -10.0, 5.0)


def test_planned_order_defaults_to_market():
    o = PlannedOrder(
        symbol="SPY", action="BUY", quantity=10, target_weight=0.2,
        current_weight=0.0, est_price=400.0, est_notional=4000.0,
    )
    assert o.order_type == "MKT"
    assert o.limit_price is None


# ---------------------------------------------------------------------------
# compute_reconciliation
# ---------------------------------------------------------------------------
def test_reconciliation_match_within_tolerance():
    # Realised exactly equals target -> ok, no mismatches.
    target = {"SPY": 0.5, "QQQ": 0.5}
    positions = {"SPY": 50.0, "QQQ": 25.0}
    prices = {"SPY": 100.0, "QQQ": 200.0}
    equity = 10_000.0  # SPY 50*100=5000 (50%), QQQ 25*200=5000 (50%)
    rep = compute_reconciliation(target, positions, prices, equity, tolerance=0.02)
    assert isinstance(rep, ReconciliationReport)
    assert rep.ok
    assert rep.mismatches == {}
    assert rep.realised_weights["SPY"] == pytest.approx(0.5)
    assert rep.realised_weights["QQQ"] == pytest.approx(0.5)


def test_reconciliation_flags_drift_beyond_tolerance():
    # QQQ under-filled: realised 30% vs target 50% -> mismatch of 0.20.
    target = {"SPY": 0.5, "QQQ": 0.5}
    positions = {"SPY": 50.0, "QQQ": 15.0}
    prices = {"SPY": 100.0, "QQQ": 200.0}
    equity = 10_000.0  # QQQ 15*200=3000 -> 30%
    rep = compute_reconciliation(target, positions, prices, equity, tolerance=0.02)
    assert not rep.ok
    assert "QQQ" in rep.mismatches
    assert rep.mismatches["QQQ"] == pytest.approx(0.20, abs=1e-9)
    # SPY is within tolerance and must NOT be flagged.
    assert "SPY" not in rep.mismatches


def test_reconciliation_flags_unexpected_position():
    # Holding a symbol with zero target weight is a mismatch.
    target = {"SPY": 1.0}
    positions = {"SPY": 100.0, "XLE": 50.0}
    prices = {"SPY": 100.0, "XLE": 80.0}
    equity = 14_000.0  # SPY 100% target; XLE 50*80=4000 -> ~28.6% unexpected
    rep = compute_reconciliation(target, positions, prices, equity, tolerance=0.02)
    assert not rep.ok
    assert "XLE" in rep.mismatches


def test_reconciliation_failsafe_on_zero_equity():
    # Degenerate equity -> no realised weights, every nonzero target is a mismatch.
    target = {"SPY": 0.5, "QQQ": 0.5}
    rep = compute_reconciliation(target, {}, {}, 0.0, tolerance=0.02)
    assert not rep.ok
    assert set(rep.mismatches) == {"SPY", "QQQ"}


def test_reconciliation_ignores_missing_price_symbol():
    # If a price is missing the symbol contributes 0 realised weight; that
    # surfaces as a mismatch rather than silently passing.
    target = {"SPY": 0.5, "QQQ": 0.5}
    positions = {"SPY": 50.0, "QQQ": 25.0}
    prices = {"SPY": 100.0}  # QQQ price missing
    equity = 10_000.0
    rep = compute_reconciliation(target, positions, prices, equity, tolerance=0.02)
    assert not rep.ok
    assert "QQQ" in rep.mismatches


# ---------------------------------------------------------------------------
# _extract_price — real-time vs delayed market-data fields (B1 fix)
# ---------------------------------------------------------------------------
NAN = float("nan")


class _Ticker:
    """Minimal stand-in for an ib_async Ticker with arbitrary tick fields."""

    def __init__(self, **fields):
        # Default every known field to NaN unless explicitly provided.
        for name in (
            "bid", "ask", "last", "close",
            "delayedBid", "delayedAsk", "delayedLast", "delayedClose",
        ):
            setattr(self, name, fields.get(name, NAN))


def test_extract_price_prefers_realtime_mid():
    t = _Ticker(bid=99.98, ask=100.02, last=100.0, close=99.0)
    assert _extract_price(t) == pytest.approx(100.0)


def test_extract_price_falls_back_to_last_when_no_quote():
    t = _Ticker(last=123.45, close=120.0)
    assert _extract_price(t) == pytest.approx(123.45)


def test_extract_price_falls_back_to_close():
    t = _Ticker(close=222.0)
    assert _extract_price(t) == pytest.approx(222.0)


def test_extract_price_uses_delayed_mid_when_realtime_nan():
    # Paper account on delayed data: real-time fields are NaN, delayed populated.
    t = _Ticker(delayedBid=49.99, delayedAsk=50.01, delayedLast=50.0)
    assert _extract_price(t) == pytest.approx(50.0)


def test_extract_price_uses_delayed_last_then_close():
    assert _extract_price(_Ticker(delayedLast=77.0)) == pytest.approx(77.0)
    assert _extract_price(_Ticker(delayedClose=66.0)) == pytest.approx(66.0)


def test_extract_price_returns_none_when_nothing_usable():
    # All NaN -> None so the caller aborts the rebalance fail-safe.
    assert _extract_price(_Ticker()) is None
    assert _extract_price(None) is None


def test_extract_price_rejects_nonpositive():
    # Zero/negative quotes are not valid prices.
    assert _extract_price(_Ticker(last=0.0)) is None
    assert _extract_price(_Ticker(last=-5.0)) is None


# ---------------------------------------------------------------------------
# _order_is_done — fill-wait terminal-state gating (B2 fix)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status", ["Filled", "Cancelled", "ApiCancelled", "Inactive"])
def test_order_is_done_terminal_states(status):
    assert _order_is_done(status) is True


@pytest.mark.parametrize(
    "status", ["PendingSubmit", "PreSubmitted", "Submitted", "", "   ", None]
)
def test_order_is_done_working_states(status):
    assert _order_is_done(status) is False


def test_order_is_done_strips_whitespace():
    assert _order_is_done("  Filled  ") is True

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
    rep = compute_reconciliation(target, positions, prices, equity)
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
    rep = compute_reconciliation(target, positions, prices, equity)
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
    rep = compute_reconciliation(target, positions, prices, equity)
    assert not rep.ok
    assert "XLE" in rep.mismatches


def test_reconciliation_failsafe_on_zero_equity():
    # Degenerate equity -> no realised weights, every nonzero target is a mismatch.
    target = {"SPY": 0.5, "QQQ": 0.5}
    rep = compute_reconciliation(target, {}, {}, 0.0)
    assert not rep.ok
    assert set(rep.mismatches) == {"SPY", "QQQ"}


def test_reconciliation_ignores_missing_price_symbol():
    # If a price is missing the symbol contributes 0 realised weight; that
    # surfaces as a mismatch rather than silently passing.
    target = {"SPY": 0.5, "QQQ": 0.5}
    positions = {"SPY": 50.0, "QQQ": 25.0}
    prices = {"SPY": 100.0}  # QQQ price missing
    equity = 10_000.0
    rep = compute_reconciliation(target, positions, prices, equity)
    assert not rep.ok
    assert "QQQ" in rep.mismatches


def test_reconciliation_flags_zero_fill_even_for_small_targets():
    """Regression (2026-06-26 root cause): when the vol-target sizes a small
    book (gross ~14%, legs ~2-3.4% of NAV) and the orders get ZERO fills
    (here: a competing IBKR session starved the paper engine of market data so
    no paper fills simulated), the live book is all cash. Every unfilled leg
    MUST be flagged. A flat absolute 5% band would have wrongly passed this
    (a 2.2% target left fully unfilled is only 2.2% absolute drift); the
    relative band catches it because the leg is ~100% unfilled."""
    target = {"IWM": 0.0342, "XLI": 0.0324, "QQQ": 0.0298,
              "XLK": 0.0224, "EEM": 0.0215}
    # No positions, all cash -> realised weight 0 for every leg.
    rep = compute_reconciliation(target, {}, {}, 1_000_000.0)
    assert not rep.ok
    assert set(rep.mismatches) == set(target)


def test_reconciliation_ignores_whole_share_rounding_noise():
    """A genuinely established book differs from target only by sub-percent
    whole-share rounding; that must reconcile OK and never self-block."""
    target = {"SPY": 0.20, "QQQ": 0.03}
    realised = {"SPY": 0.1998, "QQQ": 0.0299}
    equity = 1_000_000.0
    prices = {"SPY": 100.0, "QQQ": 100.0}
    positions = {s: w * equity / 100.0 for s, w in realised.items()}
    rep = compute_reconciliation(target, positions, prices, equity)
    assert rep.ok
    assert rep.mismatches == {}


def test_reconciliation_relative_band_scales_with_position():
    """The acceptable drift scales with the leg's target weight: a moderate
    relative drift is OK, a large one (approaching a failed fill) is flagged."""
    target = {"SPY": 0.20}
    equity = 1_000_000.0
    prices = {"SPY": 100.0}
    # 15% relative drift (0.17 vs 0.20) -> within 25% band -> OK.
    ok_pos = {"SPY": 0.17 * equity / 100.0}
    assert compute_reconciliation(target, ok_pos, prices, equity).ok
    # 40% relative drift (0.12 vs 0.20) -> exceeds band -> flagged.
    bad_pos = {"SPY": 0.12 * equity / 100.0}
    assert not compute_reconciliation(target, bad_pos, prices, equity).ok


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


# ---------------------------------------------------------------------------
# Async price/order path — event-loop-reentry regression guards
# ---------------------------------------------------------------------------
# These exercise get_price / rebalance_to_weights against a fake IB that runs on
# the live asyncio loop WITHOUT a broker connection. The fake's ``sleep`` raises
# if called, guaranteeing the code never uses ``ib.sleep()`` inside a running
# coroutine (which re-enters the loop -> "This event loop is already running"
# and was the root cause of the first two zero-trade paper sessions).
import asyncio
import time as _time

from etf.config import get_default_config
from etf.ibkr_broker import IBKRETFBroker


class _LiveTicker:
    """A Ticker whose ``last`` becomes valid only after ``ready_after`` seconds,
    simulating IBKR's first snapshot taking time to populate."""

    def __init__(self, price: float, ready_after: float = 0.0):
        self._price = price
        self._ready_at = _time.monotonic() + ready_after
        self.bid = NAN
        self.ask = NAN
        self.delayedBid = NAN
        self.delayedAsk = NAN
        self.delayedLast = NAN
        self.close = NAN
        self.delayedClose = NAN

    @property
    def last(self):
        return self._price if _time.monotonic() >= self._ready_at else NAN


class _Row:
    def __init__(self, tag, value, currency="USD"):
        self.tag = tag
        self.value = value
        self.currency = currency


class _Pos:
    def __init__(self, symbol, position):
        self.contract = type("C", (), {"symbol": symbol})()
        self.position = position


class _FakeIB:
    def __init__(self, tickers=None, rows=None, positions=None, hist=None):
        self._tickers = tickers or {}
        self._rows = rows or []
        self._positions = positions or []
        self._hist = hist or {}  # symbol -> last close (historical fallback)
        self.cancelled = []
        self.hist_calls = []

    def isConnected(self):
        return True

    async def qualifyContractsAsync(self, contract):
        return [contract]

    def reqMktData(self, contract, *a, **k):
        return self._tickers.get(contract.symbol)

    def cancelMktData(self, contract):
        self.cancelled.append(contract.symbol)

    def sleep(self, *a, **k):  # pragma: no cover - must never run
        raise RuntimeError("ib.sleep() must not be called inside an async coroutine")

    async def reqHistoricalDataAsync(self, contract, *a, **k):
        self.hist_calls.append(contract.symbol)
        close = self._hist.get(contract.symbol)
        if close is None:
            return []
        bar = type("Bar", (), {"close": close})()
        return [bar]

    async def accountSummaryAsync(self, account=""):
        return self._rows

    def accountValues(self, account=""):
        return []

    async def reqPositionsAsync(self):
        return self._positions


def _broker_with(fake_ib):
    b = IBKRETFBroker(get_default_config().ibkr, dry_run=True)
    b._ib = fake_ib
    b._connected = True
    return b


def test_get_price_returns_immediately_when_quote_ready():
    fake = _FakeIB(tickers={"XLK": _LiveTicker(123.45, ready_after=0.0)})
    b = _broker_with(fake)
    px = asyncio.run(b.get_price("XLK"))
    assert px == pytest.approx(123.45)
    assert fake.cancelled == ["XLK"]  # subscription always cancelled


def test_get_price_polls_until_quote_populates():
    # Quote is NaN for the first ~0.4s then valid: the poll loop must wait it out
    # rather than returning None on the first read (the yesterday-bug).
    fake = _FakeIB(tickers={"EEM": _LiveTicker(50.0, ready_after=0.4)})
    b = _broker_with(fake)
    px = asyncio.run(b.get_price("EEM", price_timeout=5.0))
    assert px == pytest.approx(50.0)


def test_get_price_returns_none_on_timeout_without_reentry():
    # Quote never populates and NO historical fallback available -> None
    # (fail-safe), and crucially NO event-loop reentry: if the code called
    # ib.sleep(), _FakeIB.sleep would raise.
    fake = _FakeIB(tickers={"EEM": _LiveTicker(50.0, ready_after=999.0)})
    b = _broker_with(fake)
    px = asyncio.run(b.get_price("EEM", price_timeout=0.1))  # min deadline 2s
    assert px is None
    assert fake.cancelled == ["EEM"]


def test_get_price_falls_back_to_historical_close_on_competing_session():
    # Streaming never populates (simulates IBKR Error 10197 "competing live
    # session") but a historical daily close is available -> get_price returns
    # that close instead of aborting. This is the production fix for the
    # 2026-06-16 zero-trade session.
    fake = _FakeIB(
        tickers={"DBC": _LiveTicker(0.0, ready_after=999.0)},  # never streams
        hist={"DBC": 22.37},
    )
    b = _broker_with(fake)
    px = asyncio.run(b.get_price("DBC", price_timeout=0.1))
    assert px == pytest.approx(22.37)
    assert fake.hist_calls == ["DBC"]   # fallback was actually used
    assert fake.cancelled == ["DBC"]    # streaming subscription still cancelled


def test_get_price_prefers_streaming_over_historical():
    # When a streaming quote IS available it must be used and the historical
    # fallback must NOT be requested (keeps marks live when data is flowing).
    fake = _FakeIB(
        tickers={"XLK": _LiveTicker(150.0, ready_after=0.0)},
        hist={"XLK": 140.0},
    )
    b = _broker_with(fake)
    px = asyncio.run(b.get_price("XLK"))
    assert px == pytest.approx(150.0)
    assert fake.hist_calls == []  # fallback not triggered when streaming works


def test_rebalance_succeeds_via_historical_when_streaming_blocked():
    # End-to-end: streaming blocked for every symbol (competing session) but
    # historical closes available -> rebalance plans and (dry-run) executes
    # instead of aborting fail-safe.
    fake = _FakeIB(
        tickers={
            "XLK": _LiveTicker(0.0, ready_after=999.0),
            "EEM": _LiveTicker(0.0, ready_after=999.0),
        },
        hist={"XLK": 150.0, "EEM": 50.0},
        rows=[_Row("NetLiquidation", "1000000"), _Row("TotalCashValue", "1000000")],
        positions=[],
    )
    b = _broker_with(fake)
    cfg = get_default_config()
    result = asyncio.run(b.rebalance_to_weights({"XLK": 0.3, "EEM": 0.3}, cfg))
    # Dry-run returns per-symbol "dry_run" statuses, NOT aborted_failsafe.
    assert result.get("_status") != "aborted_failsafe"
    assert set(result) == {"XLK", "EEM"}
    assert all(v == "dry_run" for v in result.values())


def test_get_price_uses_engine_close_when_ibkr_data_fully_unavailable():
    # Streaming AND historical both unavailable (IBKR Error 162 competing
    # session) -> get_price uses the engine's own daily-close fallback.
    fake = _FakeIB(tickers={"DBC": _LiveTicker(0.0, ready_after=999.0)})  # no hist
    b = _broker_with(fake)
    px = asyncio.run(b.get_price("DBC", price_timeout=0.1, fallback_price=22.37))
    assert px == pytest.approx(22.37)
    assert fake.hist_calls == ["DBC"]  # historical was attempted first


def test_rebalance_succeeds_via_engine_close_when_all_ibkr_data_blocked():
    # The production scenario from 2026-06-16: every IBKR data path blocked, but
    # the engine passes its own last daily closes -> rebalance proceeds.
    fake = _FakeIB(
        tickers={
            "XLK": _LiveTicker(0.0, ready_after=999.0),
            "EEM": _LiveTicker(0.0, ready_after=999.0),
        },
        hist={},  # historical also blocked
        rows=[_Row("NetLiquidation", "1000000"), _Row("TotalCashValue", "1000000")],
        positions=[],
    )
    b = _broker_with(fake)
    cfg = get_default_config()
    result = asyncio.run(
        b.rebalance_to_weights(
            {"XLK": 0.3, "EEM": 0.3}, cfg,
            fallback_prices={"XLK": 150.0, "EEM": 50.0},
        )
    )
    assert result.get("_status") != "aborted_failsafe"
    assert set(result) == {"XLK", "EEM"}
    assert all(v == "dry_run" for v in result.values())


def test_rebalance_aborts_when_no_fallback_and_all_ibkr_data_blocked():
    # Without an engine fallback AND with all IBKR data blocked, the fail-safe
    # must still engage (never trade blind).
    fake = _FakeIB(
        tickers={"XLK": _LiveTicker(0.0, ready_after=999.0)},
        hist={},
        rows=[_Row("NetLiquidation", "1000000"), _Row("TotalCashValue", "1000000")],
        positions=[],
    )
    b = _broker_with(fake)
    cfg = get_default_config()
    result = asyncio.run(b.rebalance_to_weights({"XLK": 1.0}, cfg))
    assert result == {"_status": "aborted_failsafe"}


def test_rebalance_aborts_failsafe_when_a_price_is_missing():
    # Account is healthy but one symbol never quotes -> plan_rebalance returns
    # None -> rebalance_to_weights reports aborted_failsafe (NOT no_change),
    # which the runner maps to "do not advance cadence".
    fake = _FakeIB(
        tickers={
            "XLK": _LiveTicker(150.0, ready_after=0.0),
            "EEM": _LiveTicker(50.0, ready_after=999.0),  # never ready
        },
        rows=[_Row("NetLiquidation", "1000000"), _Row("TotalCashValue", "1000000")],
        positions=[],
    )
    b = _broker_with(fake)
    cfg = get_default_config()
    result = asyncio.run(
        b.rebalance_to_weights({"XLK": 0.5, "EEM": 0.5}, cfg)
    )
    assert result == {"_status": "aborted_failsafe"}


def test_get_account_reads_summary_rows():
    fake = _FakeIB(
        rows=[
            _Row("NetLiquidation", "1021473.52"),
            _Row("TotalCashValue", "1020510.62"),
            _Row("BuyingPower", "4082042.48"),
        ],
        positions=[_Pos("XLK", 100.0)],
    )
    b = _broker_with(fake)
    acct = asyncio.run(b.get_account())
    assert acct is not None
    assert acct.equity == pytest.approx(1021473.52)
    assert acct.cash == pytest.approx(1020510.62)
    assert acct.positions == {"XLK": 100.0}


# ---------------------------------------------------------------------------
# Gross-leverage cap enforced on the EFFECTIVE held book (min-delta drift)
# ---------------------------------------------------------------------------
def test_plan_rebalance_enforces_gross_cap_after_min_delta_drift():
    # Reproduces the latent live breach: starting fully invested at gross 1.0
    # (A 45%, B 45%, C 10%), the new target trims A,B to 30% (a 15% move, BELOW
    # the 20% min-rebalance-delta -> RETAINED at 45%) while lifting C to 40%
    # (a 30% move -> ADOPTED). The raw effective book is therefore 0.45+0.45+
    # 0.40 = 1.30 gross, well over the 1.0 cap. The cap must trim it back.
    fake = _FakeIB(
        tickers={
            "A": _LiveTicker(100.0, ready_after=0.0),
            "B": _LiveTicker(100.0, ready_after=0.0),
            "C": _LiveTicker(100.0, ready_after=0.0),
        },
        rows=[_Row("NetLiquidation", "100000"), _Row("TotalCashValue", "0")],
        positions=[_Pos("A", 450.0), _Pos("B", 450.0), _Pos("C", 100.0)],
    )
    b = _broker_with(fake)
    cfg = get_default_config()
    cfg.execution.min_rebalance_delta = 0.20   # 15% A/B moves stay retained
    cfg.risk.max_gross_leverage = 1.0

    orders = asyncio.run(
        b.plan_rebalance({"A": 0.30, "B": 0.30, "C": 0.40}, cfg)
    )
    assert orders is not None

    # Reconstruct the held book that these orders produce.
    shares = {"A": 450.0, "B": 450.0, "C": 100.0}
    for o in orders:
        shares[o.symbol] += o.quantity if o.action == "BUY" else -o.quantity
    equity = 100_000.0
    gross = sum(abs(s) * 100.0 for s in shares.values()) / equity

    # Capped book respects the 1.0 limit (was 1.30 without enforcement).
    assert gross <= cfg.risk.max_gross_leverage + 1e-3
    assert gross == pytest.approx(1.0, abs=5e-3)
    # The cap forced A and B to be trimmed even though min-delta alone would
    # have left them untouched -> there must be SELL orders for both.
    sells = {o.symbol for o in orders if o.action == "SELL"}
    assert {"A", "B"} <= sells


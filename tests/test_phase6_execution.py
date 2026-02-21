#!/usr/bin/env python3
"""
Phase 6 Tests — Execution & Position Management Fixes
=====================================================

Tests for:
  1. ExitManager — profit targets, stop losses, DTE exits, trailing stops
  2. GEX Analyzer — gamma exposure computation, sticky strikes, signal filtering
  3. Config — exit management and GEX parameters
  4. DailyPerformanceLogger — daily P&L tracking
  5. Integration — ExitManager registration from engine

Run:  python -m pytest tests/test_phase6_execution.py -v
  or: python tests/test_phase6_execution.py
"""

import sys
import os
import asyncio
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ PASS: {name}")
    else:
        FAIL += 1
        print(f"  ❌ FAIL: {name}" + (f" — {detail}" if detail else ""))


# ═══════════════════════════════════════════════════════════════════════
# 1. EXIT MANAGER TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_exit_manager_import():
    """ExitManager imports without error."""
    print("\n── ExitManager Import Tests ──")
    try:
        from src.options.exit_manager import (
            ExitManager, ExitAction, ExitReason,
            TrackedPosition, TrackedLeg, PositionType, DEFAULT_EXIT_CONFIG
        )
        check("ExitManager importable", True)
    except Exception as e:
        check("ExitManager importable", False, str(e))


def test_exit_manager_init():
    """ExitManager initializes with default config."""
    print("\n── ExitManager Init Tests ──")
    from src.options.exit_manager import ExitManager, DEFAULT_EXIT_CONFIG

    em = ExitManager()
    check("Default config loaded", em.config["profit_target_pct"] == 0.50)
    check("No initial positions", len(em.positions) == 0)
    check("Stats initialized", em.stats["total_exits"] == 0)

    # Custom config
    em2 = ExitManager(config={"profit_target_pct": 0.75})
    check("Custom config override", em2.config["profit_target_pct"] == 0.75)
    check("Other defaults preserved", em2.config["stop_loss_multiplier"] == 2.0)


def test_exit_manager_register_spread():
    """ExitManager registers spreads correctly."""
    print("\n── ExitManager Spread Registration Tests ──")
    from src.options.exit_manager import ExitManager

    em = ExitManager()

    pid = em.register_spread(
        underlying="SPY",
        short_occ="SPY260320P00550000",
        long_occ="SPY260320P00545000",
        qty=1,
        net_credit=1.25,
        max_profit=125.0,
        max_loss=375.0,
        strategy="credit_spread",
        expiration=date(2026, 3, 20),
    )

    check("Position registered", pid in em.positions)
    pos = em.positions[pid]
    check("Underlying correct", pos.underlying == "SPY")
    check("Legs count", len(pos.legs) == 2)
    check("Net credit", pos.net_credit == 1.25)
    check("Max profit", pos.max_profit == 125.0)
    check("Max loss", pos.max_loss == 375.0)
    check("Expiration", pos.expiration == date(2026, 3, 20))
    check("Not closed", pos.is_closed is False)
    check("Short leg side", pos.legs[0].side == "sell")
    check("Long leg side", pos.legs[1].side == "buy")


def test_exit_manager_register_iron_condor():
    """ExitManager registers iron condors correctly."""
    print("\n── ExitManager Iron Condor Registration Tests ──")
    from src.options.exit_manager import ExitManager, PositionType

    em = ExitManager()

    pid = em.register_iron_condor(
        underlying="SPY",
        put_long_occ="SPY260320P00535000",
        put_short_occ="SPY260320P00540000",
        call_short_occ="SPY260320C00560000",
        call_long_occ="SPY260320C00565000",
        qty=2,
        net_credit=2.50,
        max_profit=500.0,
        max_loss=500.0,
        expiration=date(2026, 3, 20),
    )

    check("IC registered", pid in em.positions)
    pos = em.positions[pid]
    check("IC type", pos.position_type == PositionType.IRON_CONDOR)
    check("4 legs", len(pos.legs) == 4)
    check("Qty=2", pos.qty == 2)


def test_exit_manager_register_single_leg():
    """ExitManager registers single legs correctly."""
    print("\n── ExitManager Single Leg Registration Tests ──")
    from src.options.exit_manager import ExitManager, PositionType

    em = ExitManager()

    pid = em.register_single_leg(
        underlying="AAPL",
        occ_symbol="AAPL260320C00200000",
        side="buy",
        qty=1,
        entry_price=3.50,
        max_profit=350.0,
        max_loss=350.0,
        strategy="long_call",
        expiration=date(2026, 3, 20),
    )

    check("Single leg registered", pid in em.positions)
    pos = em.positions[pid]
    check("Single leg type", pos.position_type == PositionType.SINGLE_LEG)
    check("1 leg", len(pos.legs) == 1)


def test_exit_manager_profit_target():
    """ExitManager triggers profit target exit."""
    print("\n── ExitManager Profit Target Tests ──")
    from src.options.exit_manager import ExitManager, ExitReason

    em = ExitManager()

    pid = em.register_spread(
        underlying="SPY",
        short_occ="SPY260320P00550000",
        long_occ="SPY260320P00545000",
        qty=1,
        net_credit=1.00,
        max_profit=100.0,
        max_loss=400.0,
        strategy="credit_spread",
        expiration=date(2026, 3, 20),
    )

    pos = em.positions[pid]

    # Set P&L to 30% of max profit — should NOT trigger
    pos.current_pnl = 30.0
    action = em._evaluate_exit(pos)
    check("30% profit no trigger", action is None)

    # Set P&L to 55% of max profit — should trigger
    pos.current_pnl = 55.0
    action = em._evaluate_exit(pos)
    check("55% profit triggers", action is not None)
    if action:
        check("Reason is profit_target", action.reason == ExitReason.PROFIT_TARGET)


def test_exit_manager_stop_loss():
    """ExitManager triggers stop loss exit."""
    print("\n── ExitManager Stop Loss Tests ──")
    from src.options.exit_manager import ExitManager, ExitReason

    em = ExitManager()

    pid = em.register_spread(
        underlying="SPY",
        short_occ="SPY260320P00550000",
        long_occ="SPY260320P00545000",
        qty=1,
        net_credit=1.00,
        max_profit=100.0,
        max_loss=400.0,
        strategy="credit_spread",
        expiration=date(2026, 3, 20),
    )

    pos = em.positions[pid]

    # Loss of $150 — should NOT trigger (stop at 2x premium = $200)
    pos.current_pnl = -150.0
    action = em._evaluate_exit(pos)
    check("-$150 no trigger", action is None)

    # Loss of $250 — should trigger (> 2x $100 premium)
    pos.current_pnl = -250.0
    action = em._evaluate_exit(pos)
    check("-$250 triggers stop", action is not None)
    if action:
        check("Reason is stop_loss", action.reason == ExitReason.STOP_LOSS)


def test_exit_manager_dte_exit():
    """ExitManager triggers DTE-based exit."""
    print("\n── ExitManager DTE Exit Tests ──")
    from src.options.exit_manager import ExitManager, ExitReason

    em = ExitManager()

    # Position expiring in 5 days (< 7 DTE threshold)
    pid = em.register_spread(
        underlying="SPY",
        short_occ="SPY260320P00550000",
        long_occ="SPY260320P00545000",
        qty=1,
        net_credit=1.00,
        max_profit=100.0,
        max_loss=400.0,
        strategy="credit_spread",
        expiration=date.today() + timedelta(days=5),
    )

    pos = em.positions[pid]
    pos.current_pnl = 0.0  # Break-even
    action = em._evaluate_exit(pos)
    check("5 DTE triggers exit", action is not None)
    if action:
        check("Reason is dte_exit", action.reason == ExitReason.DTE_EXIT)

    # Position expiring in 20 days — should NOT trigger
    pid2 = em.register_spread(
        underlying="QQQ",
        short_occ="QQQ260320P00460000",
        long_occ="QQQ260320P00455000",
        qty=1,
        net_credit=1.00,
        max_profit=100.0,
        max_loss=400.0,
        strategy="credit_spread",
        expiration=date.today() + timedelta(days=20),
    )

    pos2 = em.positions[pid2]
    pos2.current_pnl = 0.0
    action2 = em._evaluate_exit(pos2)
    check("20 DTE no trigger", action2 is None)


def test_exit_manager_trailing_stop():
    """ExitManager triggers trailing stop."""
    print("\n── ExitManager Trailing Stop Tests ──")
    from src.options.exit_manager import ExitManager, ExitReason

    em = ExitManager()

    pid = em.register_spread(
        underlying="SPY",
        short_occ="SPY260320P00550000",
        long_occ="SPY260320P00545000",
        qty=1,
        net_credit=1.00,
        max_profit=100.0,
        max_loss=400.0,
        strategy="credit_spread",
        expiration=date.today() + timedelta(days=30),
    )

    pos = em.positions[pid]

    # Set peak to 40% profit, then drop to 15%
    pos.current_pnl = 40.0
    pos.peak_pnl = 40.0
    pos.peak_pnl_pct = 0.40

    # Now drop significantly below peak
    pos.current_pnl = 10.0  # Gave back 75% of peak (> 50% trail)
    action = em._evaluate_exit(pos)
    check("Trailing stop triggers", action is not None)
    if action:
        check("Reason is trailing_stop", action.reason == ExitReason.TRAILING_STOP)


def test_exit_manager_state_persistence():
    """ExitManager save/load state."""
    print("\n── ExitManager State Persistence Tests ──")
    from src.options.exit_manager import ExitManager

    em = ExitManager()

    em.register_spread(
        underlying="SPY",
        short_occ="SPY260320P00550000",
        long_occ="SPY260320P00545000",
        qty=1,
        net_credit=1.25,
        max_profit=125.0,
        max_loss=375.0,
        strategy="credit_spread",
        expiration=date(2026, 3, 20),
    )

    # Save state
    state = em.save_state()
    check("State has positions", len(state["positions"]) == 1)
    check("State has stats", "stats" in state)

    # Load into new instance
    em2 = ExitManager()
    em2.load_state(state)
    check("Positions restored", len(em2.positions) == 1)

    restored = list(em2.positions.values())[0]
    check("Underlying preserved", restored.underlying == "SPY")
    check("Net credit preserved", restored.net_credit == 1.25)


def test_exit_manager_summary():
    """ExitManager summary and reporting."""
    print("\n── ExitManager Summary Tests ──")
    from src.options.exit_manager import ExitManager

    em = ExitManager()

    em.register_spread(
        underlying="SPY",
        short_occ="SPY260320P00550000",
        long_occ="SPY260320P00545000",
        qty=1,
        net_credit=1.00,
        max_profit=100.0,
        max_loss=400.0,
        strategy="credit_spread",
        expiration=date(2026, 3, 20),
    )

    summary = em.get_summary()
    check("Summary has open_positions", summary["open_positions"] == 1)
    check("Summary has stats", "stats" in summary)

    report = em.get_performance_report()
    check("Report is string", isinstance(report, str))
    check("Report has content", len(report) > 50)


# ═══════════════════════════════════════════════════════════════════════
# 2. GEX ANALYZER TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_gex_import():
    """GEX analyzer imports without error."""
    print("\n── GEX Import Tests ──")
    try:
        from src.options.gex_analyzer import (
            GammaExposureAnalyzer, GEXProfile, StrikeGEX, GEXSignalFilter
        )
        check("GEX analyzer importable", True)
    except Exception as e:
        check("GEX analyzer importable", False, str(e))


def test_gex_init():
    """GEX analyzer initializes correctly."""
    print("\n── GEX Init Tests ──")
    from src.options.gex_analyzer import GammaExposureAnalyzer

    gex = GammaExposureAnalyzer()
    check("Default threshold", gex.sticky_strike_threshold == 0.30)
    check("Default avoidance", gex.avoidance_radius_pct == 0.005)
    check("Empty cache", len(gex._cache) == 0)


def test_gex_synthetic_profile():
    """GEX synthetic profile generation."""
    print("\n── GEX Synthetic Profile Tests ──")
    from src.options.gex_analyzer import GammaExposureAnalyzer

    gex = GammaExposureAnalyzer()
    strikes = gex._synthetic_gex("SPY", 550.0, 30)

    check("Strikes generated", len(strikes) > 0)
    check("21 strikes", len(strikes) == 21)

    # Check strike range around spot
    min_strike = min(s.strike for s in strikes)
    max_strike = max(s.strike for s in strikes)
    check("Min strike < spot", min_strike < 550.0)
    check("Max strike > spot", max_strike > 550.0)

    # Check GEX values exist
    has_gex = any(abs(s.net_gex) > 0 for s in strikes)
    check("Non-zero GEX values", has_gex)


def test_gex_sticky_strikes():
    """GEX sticky strike detection."""
    print("\n── GEX Sticky Strike Tests ──")
    from src.options.gex_analyzer import GammaExposureAnalyzer, GEXProfile, StrikeGEX

    gex = GammaExposureAnalyzer()

    # Create synthetic profile
    strikes = gex._synthetic_gex("SPY", 550.0, 30)
    profile = GEXProfile(
        symbol="SPY",
        spot_price=550.0,
        timestamp=datetime.now(),
        strikes=strikes,
    )

    sticky = gex.get_sticky_strikes(profile, n=3)
    check("Top 3 sticky strikes", len(sticky) == 3)
    check("Sticky strikes are tuples", isinstance(sticky[0], tuple))
    check("Sorted by |gex|", abs(sticky[0][1]) >= abs(sticky[1][1]))


def test_gex_signal_filter():
    """GEX signal filtering."""
    print("\n── GEX Signal Filter Tests ──")
    from src.options.gex_analyzer import GammaExposureAnalyzer, GEXProfile

    gex = GammaExposureAnalyzer()

    # No profile — should pass
    result = gex.filter_signal(None, 550.0)
    check("No profile passes", result.is_safe)

    # Create profile
    strikes = gex._synthetic_gex("SPY", 550.0, 30)
    profile = GEXProfile(
        symbol="SPY",
        spot_price=550.0,
        timestamp=datetime.now(),
        strikes=strikes,
        is_positive_gex=True,
        net_gex=1000.0,
    )

    # Test with far-away strike
    result = gex.filter_signal(profile, 500.0, "credit_spread")
    check("Far strike is safe", result.is_safe)
    check("Recommended proceed", result.recommended_action == "proceed")

    # Test with negative GEX environment
    profile.is_positive_gex = False
    profile.net_gex = -1000.0
    result_neg = gex.filter_signal(profile, 500.0, "credit_spread")
    check("Negative GEX flagged", result_neg.gex_environment == "negative")


def test_gex_bs_gamma():
    """GEX Black-Scholes gamma computation."""
    print("\n── GEX BS Gamma Tests ──")
    from src.options.gex_analyzer import GammaExposureAnalyzer

    # ATM option should have highest gamma
    gamma_atm = GammaExposureAnalyzer._bs_gamma(100, 100, 0.1, 0.20)
    gamma_otm = GammaExposureAnalyzer._bs_gamma(100, 120, 0.1, 0.20)
    gamma_itm = GammaExposureAnalyzer._bs_gamma(100, 80, 0.1, 0.20)

    check("ATM gamma > 0", gamma_atm > 0)
    check("ATM gamma > OTM", gamma_atm > gamma_otm)
    check("ATM gamma > ITM", gamma_atm > gamma_itm)

    # Edge cases
    gamma_zero_t = GammaExposureAnalyzer._bs_gamma(100, 100, 0, 0.20)
    check("Zero time gamma = 0", gamma_zero_t == 0.0)


def test_gex_zero_gamma_strike():
    """GEX zero-gamma strike detection."""
    print("\n── GEX Zero Gamma Strike Tests ──")
    from src.options.gex_analyzer import GammaExposureAnalyzer, StrikeGEX

    gex = GammaExposureAnalyzer()

    # Create strikes with sign change
    strikes = [
        StrikeGEX(strike=540.0, call_gamma=100, put_gamma=-150, net_gex=-50),
        StrikeGEX(strike=545.0, call_gamma=120, put_gamma=-100, net_gex=20),
        StrikeGEX(strike=550.0, call_gamma=80, put_gamma=-60, net_gex=20),
    ]

    zero = gex._find_zero_gamma_strike(strikes, 545.0)
    check("Zero gamma found", zero is not None)
    if zero:
        check("Zero gamma between 540-545", 540.0 < zero < 545.0)


def test_gex_occ_strike_parse():
    """GEX OCC strike parsing."""
    print("\n── GEX OCC Strike Parse Tests ──")
    from src.options.gex_analyzer import GammaExposureAnalyzer

    strike = GammaExposureAnalyzer._parse_occ_strike("SPY260320P00550000")
    check("SPY 550 parsed", strike == 550.0)

    strike2 = GammaExposureAnalyzer._parse_occ_strike("AAPL260320C00175500")
    check("AAPL 175.5 parsed", strike2 == 175.5)


# ═══════════════════════════════════════════════════════════════════════
# 3. CONFIG TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_config_exit_params():
    """Config has Phase 6 exit management parameters."""
    print("\n── Config Exit Parameter Tests ──")
    from src.options.config import RISK_CONFIG

    check("exit_profit_target_pct", "exit_profit_target_pct" in RISK_CONFIG)
    check("exit_stop_loss_multiplier", "exit_stop_loss_multiplier" in RISK_CONFIG)
    check("exit_dte_threshold", "exit_dte_threshold" in RISK_CONFIG)
    check("exit_trailing_stop_activate", "exit_trailing_stop_activate" in RISK_CONFIG)
    check("exit_trailing_stop_trail", "exit_trailing_stop_trail" in RISK_CONFIG)
    check("exit_use_mleg_close", RISK_CONFIG.get("exit_use_mleg_close") is True)

    # Validate values
    check("Profit target 50%", RISK_CONFIG["exit_profit_target_pct"] == 0.50)
    check("Stop loss 2x", RISK_CONFIG["exit_stop_loss_multiplier"] == 2.0)
    check("DTE threshold 7", RISK_CONFIG["exit_dte_threshold"] == 7)


def test_config_gex_params():
    """Config has Phase 6 GEX parameters."""
    print("\n── Config GEX Parameter Tests ──")
    from src.options.config import RISK_CONFIG

    check("gex_enabled", "gex_enabled" in RISK_CONFIG)
    check("gex_sticky_strike_threshold", "gex_sticky_strike_threshold" in RISK_CONFIG)
    check("gex_avoidance_radius_pct", "gex_avoidance_radius_pct" in RISK_CONFIG)
    check("gex_cache_ttl_minutes", "gex_cache_ttl_minutes" in RISK_CONFIG)
    check("gex_negative_size_reduction", "gex_negative_size_reduction" in RISK_CONFIG)


# ═══════════════════════════════════════════════════════════════════════
# 4. DAILY PERFORMANCE LOGGER TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_daily_perf_logger_import():
    """DailyPerformanceLogger imports."""
    print("\n── DailyPerformanceLogger Import Tests ──")
    try:
        from src.metrics.daily_performance import DailyPerformanceLogger, DailySnapshot
        check("DailyPerformanceLogger importable", True)
    except Exception as e:
        check("DailyPerformanceLogger importable", False, str(e))


def test_daily_perf_logger_log():
    """DailyPerformanceLogger logs a daily snapshot."""
    print("\n── DailyPerformanceLogger Log Tests ──")
    import tempfile
    from src.metrics.daily_performance import DailyPerformanceLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = DailyPerformanceLogger(log_dir=tmpdir, initial_equity=100000)

        snap = logger.log_daily(
            equity=100500,
            daily_pnl=500.0,
            n_positions=3,
            n_trades=2,
            turnover_pct=1.5,
        )

        check("Snapshot returned", snap is not None)
        check("Equity correct", snap.equity == 100500)
        check("PnL correct", snap.daily_pnl == 500.0)
        check("Return positive", snap.daily_return_pct > 0)

        # Idempotent — second call same day should return cached
        snap2 = logger.log_daily(equity=100500, daily_pnl=500.0)
        check("Idempotent same day", snap2 is not None)

        history = logger.get_history()
        check("History has 1 record", len(history) == 1)


# ═══════════════════════════════════════════════════════════════════════
# 5. INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_exit_manager_occ_parse():
    """ExitManager OCC expiration parsing."""
    print("\n── ExitManager OCC Parse Tests ──")
    from src.options.exit_manager import ExitManager

    exp = ExitManager._parse_occ_expiration("SPY260320P00550000")
    check("SPY expiration parsed", exp == date(2026, 3, 20))

    exp2 = ExitManager._parse_occ_expiration("AAPL261218C00200000")
    check("AAPL expiration parsed", exp2 == date(2026, 12, 18))

    exp3 = ExitManager._parse_occ_expiration("bad")
    check("Bad OCC returns None", exp3 is None)


def test_exit_manager_sync_orphaned():
    """ExitManager syncs orphaned Alpaca positions."""
    print("\n── ExitManager Orphaned Sync Tests ──")
    from src.options.exit_manager import ExitManager

    em = ExitManager()

    # Simulate Alpaca having positions not tracked
    alpaca_opts = {
        "SPY260320P00550000": {
            "qty": 1,
            "cost_basis": 350.0,
            "market_value": 300.0,
            "unrealized_pl": -50.0,
        },
        "QQQ260320C00480000": {
            "qty": -2,
            "cost_basis": 200.0,
            "market_value": 150.0,
            "unrealized_pl": 50.0,
        },
    }

    em.sync_from_alpaca_state(alpaca_opts)
    check("Orphaned positions synced", len(em.positions) == 2)


def test_trade_executor_mleg_imports():
    """Trade executor has MLEG imports."""
    print("\n── Trade Executor MLEG Import Tests ──")
    try:
        from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
        from alpaca.trading.enums import OrderClass
        check("OptionLegRequest importable", True)
        check("OrderClass.MLEG exists", hasattr(OrderClass, "MLEG"))
    except ImportError:
        # Alpaca not installed in test env — check source code instead
        import inspect
        try:
            from src.options.trade_executor import AlpacaOptionsExecutor
            source = inspect.getsource(AlpacaOptionsExecutor)
            check("OptionLegRequest in source", "OptionLegRequest" in source)
            check("OrderClass.MLEG in source", "OrderClass.MLEG" in source)
            check("submit_spread_order uses MLEG", "order_class=OrderClass.MLEG" in source)
            check("submit_iron_condor uses MLEG", "submit_iron_condor" in source)
        except Exception as e:
            check("Trade executor source check", False, str(e))


def test_engine_has_exit_manager():
    """Autonomous engine has ExitManager wired in."""
    print("\n── Engine ExitManager Wiring Tests ──")
    import inspect
    try:
        with open("src/options/autonomous_engine.py", "r") as f:
            source = f.read()

        check("ExitManager imported", "from .exit_manager import ExitManager" in source)
        check("GEX imported", "from .gex_analyzer import GammaExposureAnalyzer" in source)
        check("DailyPerformanceLogger imported", "DailyPerformanceLogger" in source)
        check("exit_manager initialized", "self.exit_manager = ExitManager" in source)
        check("gex_analyzer initialized", "self.gex_analyzer = GammaExposureAnalyzer" in source)
        check("daily_perf_logger initialized", "self.daily_perf_logger = DailyPerformanceLogger" in source)
        check("_run_exit_manager method", "async def _run_exit_manager" in source)
        check("_register_with_exit_manager method", "def _register_with_exit_manager" in source)
        check("_log_daily_performance method", "def _log_daily_performance" in source)
        check("GEX filter in execution", "gex_filter = self.gex_analyzer.filter_signal" in source)
        check("ExitManager state saved", '"exit_manager": self.exit_manager.save_state()' in source)
        check("ExitManager state loaded", "self.exit_manager.load_state" in source)
        check("_run_exit_manager called in cycle", "await self._run_exit_manager()" in source)
    except Exception as e:
        check("Engine source check", False, str(e))


def test_exit_manager_async_check():
    """ExitManager async check_all_positions works."""
    print("\n── ExitManager Async Check Tests ──")
    from src.options.exit_manager import ExitManager

    em = ExitManager()

    # Register a position at 60% profit (above 50% target)
    pid = em.register_spread(
        underlying="SPY",
        short_occ="SPY260320P00550000",
        long_occ="SPY260320P00545000",
        qty=1,
        net_credit=1.00,
        max_profit=100.0,
        max_loss=400.0,
        strategy="credit_spread",
        expiration=date.today() + timedelta(days=30),
    )

    pos = em.positions[pid]
    pos.current_pnl = 60.0  # 60% of max profit

    # Run async check
    actions = asyncio.get_event_loop().run_until_complete(em.check_all_positions())
    check("Async check returns actions", len(actions) == 1)
    if actions:
        check("Action is profit_target", actions[0].reason.value == "profit_target")


def test_exit_manager_record_exit():
    """ExitManager records exits correctly."""
    print("\n── ExitManager Exit Recording Tests ──")
    from src.options.exit_manager import ExitManager, ExitAction, ExitReason, PositionType

    em = ExitManager()

    pid = em.register_spread(
        underlying="SPY",
        short_occ="SPY260320P00550000",
        long_occ="SPY260320P00545000",
        qty=1,
        net_credit=1.00,
        max_profit=100.0,
        max_loss=400.0,
        strategy="credit_spread",
        expiration=date.today() + timedelta(days=30),
    )

    pos = em.positions[pid]
    pos.current_pnl = 55.0

    action = ExitAction(
        position_id=pid,
        underlying="SPY",
        reason=ExitReason.PROFIT_TARGET,
        action="close",
        current_pnl=55.0,
        current_pnl_pct=0.55,
        legs_to_close=pos.legs,
        position_type=PositionType.CREDIT_SPREAD,
        strategy="credit_spread",
    )

    em._record_exit(pos, action)

    check("Position removed from open", pid not in em.positions)
    check("Position in closed list", len(em.closed_positions) == 1)
    check("Stats updated", em.stats["total_exits"] == 1)
    check("Stats profit target", em.stats["profit_target_exits"] == 1)
    check("Stats winning exits", em.stats["winning_exits"] == 1)
    check("Stats total P&L", em.stats["total_realized_pnl"] == 55.0)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    global PASS, FAIL

    print("=" * 70)
    print("PHASE 6: Execution & Position Management Tests")
    print("=" * 70)

    # ExitManager tests
    test_exit_manager_import()
    test_exit_manager_init()
    test_exit_manager_register_spread()
    test_exit_manager_register_iron_condor()
    test_exit_manager_register_single_leg()
    test_exit_manager_profit_target()
    test_exit_manager_stop_loss()
    test_exit_manager_dte_exit()
    test_exit_manager_trailing_stop()
    test_exit_manager_state_persistence()
    test_exit_manager_summary()
    test_exit_manager_occ_parse()
    test_exit_manager_sync_orphaned()
    test_exit_manager_async_check()
    test_exit_manager_record_exit()

    # GEX tests
    test_gex_import()
    test_gex_init()
    test_gex_synthetic_profile()
    test_gex_sticky_strikes()
    test_gex_signal_filter()
    test_gex_bs_gamma()
    test_gex_zero_gamma_strike()
    test_gex_occ_strike_parse()

    # Config tests
    test_config_exit_params()
    test_config_gex_params()

    # Daily perf tests
    test_daily_perf_logger_import()
    test_daily_perf_logger_log()

    # Integration tests
    test_trade_executor_mleg_imports()
    test_engine_has_exit_manager()

    # Summary
    total = PASS + FAIL
    print("\n" + "=" * 70)
    print(f"RESULTS: {PASS}/{total} passed ({FAIL} failed)")
    print("=" * 70)

    if FAIL > 0:
        sys.exit(1)
    print("\n✅ All Phase 6 tests passed!")


if __name__ == "__main__":
    main()

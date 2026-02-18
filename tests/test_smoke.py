#!/usr/bin/env python3
"""
Smoke tests for run_v28_production.py — validates that all wired
protections (RiskGuardian, StrategyEngine, anti-churn, universe filter,
regime sizing, bracket stops) are correctly configured.

Run:  python -m pytest tests/test_smoke.py -v
  or: python tests/test_smoke.py
"""

import sys
import os
import numpy as np

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


def test_imports():
    """All critical modules import without error."""
    print("\n── Import Tests ──")
    try:
        from run_v28_production import EquityEngine, EQUITY_UNIVERSE
        check("EquityEngine importable", True)
    except Exception as e:
        check("EquityEngine importable", False, str(e))

    try:
        from config.universe import BANNED_SYMBOLS, MAX_BETA
        check("BANNED_SYMBOLS importable", True)
    except Exception as e:
        check("BANNED_SYMBOLS importable", False, str(e))

    try:
        from risk_guardian import RiskGuardian
        check("RiskGuardian importable", True)
    except Exception as e:
        check("RiskGuardian importable", False, str(e))

    try:
        from strategy_engine import StrategyEngine, EngineConfig
        check("StrategyEngine importable", True)
    except Exception as e:
        check("StrategyEngine importable", False, str(e))


def test_banned_symbols():
    """BANNED_SYMBOLS contains the required set."""
    print("\n── BANNED_SYMBOLS Tests ──")
    from config.universe import BANNED_SYMBOLS
    required = {"BBBY", "SIVB", "FRC", "COIN"}
    check("BANNED_SYMBOLS == {BBBY, SIVB, FRC, COIN}",
          BANNED_SYMBOLS == required,
          f"got {BANNED_SYMBOLS}")


def test_equity_universe_clean():
    """EQUITY_UNIVERSE has no banned symbols."""
    print("\n── Universe Tests ──")
    from run_v28_production import EQUITY_UNIVERSE
    from config.universe import BANNED_SYMBOLS
    overlap = [s for s in EQUITY_UNIVERSE if s in BANNED_SYMBOLS]
    check("No banned symbols in EQUITY_UNIVERSE",
          len(overlap) == 0,
          f"found: {overlap}")
    check("EQUITY_UNIVERSE has ≥ 40 symbols",
          len(EQUITY_UNIVERSE) >= 40,
          f"got {len(EQUITY_UNIVERSE)}")


def test_equity_engine_constants():
    """EquityEngine class-level constants are correct."""
    print("\n── EquityEngine Constants ──")
    from run_v28_production import EquityEngine

    check("MAX_DAILY_TURNOVER_PCT == 0.15",
          EquityEngine.MAX_DAILY_TURNOVER_PCT == 0.15,
          f"got {EquityEngine.MAX_DAILY_TURNOVER_PCT}")

    check("MIN_HOLD_BARS == 6",
          EquityEngine.MIN_HOLD_BARS == 6,
          f"got {EquityEngine.MIN_HOLD_BARS}")

    check("EQUITY_STOP_LOSS_PCT == -0.05",
          EquityEngine.EQUITY_STOP_LOSS_PCT == -0.05,
          f"got {EquityEngine.EQUITY_STOP_LOSS_PCT}")

    check("EQUITY_TAKE_PROFIT_PCT == 0.10",
          EquityEngine.EQUITY_TAKE_PROFIT_PCT == 0.10,
          f"got {EquityEngine.EQUITY_TAKE_PROFIT_PCT}")

    check("TRAILING_STOP_ACTIVATE_PCT == 0.04",
          EquityEngine.TRAILING_STOP_ACTIVATE_PCT == 0.04,
          f"got {EquityEngine.TRAILING_STOP_ACTIVATE_PCT}")

    check("TRAILING_STOP_TRAIL_PCT == 0.40",
          EquityEngine.TRAILING_STOP_TRAIL_PCT == 0.40,
          f"got {EquityEngine.TRAILING_STOP_TRAIL_PCT}")


def test_equity_engine_init():
    """EquityEngine constructor sets anti-churn and risk attributes."""
    print("\n── EquityEngine Init ──")
    from run_v28_production import EquityEngine
    e = EquityEngine("paper")

    check("risk_guardian attr exists",
          hasattr(e, "risk_guardian"))
    check("strategy_engine attr exists",
          hasattr(e, "strategy_engine"))
    check("_daily_turnover_used starts at 0",
          e._daily_turnover_used == 0.0,
          f"got {e._daily_turnover_used}")
    check("_bar_count starts at 0",
          e._bar_count == 0,
          f"got {e._bar_count}")
    check("_position_entry_bar is empty dict",
          isinstance(e._position_entry_bar, dict) and len(e._position_entry_bar) == 0)
    check("_regime_size_scale default == 0.70",
          e._regime_size_scale == 0.70,
          f"got {e._regime_size_scale}")
    check("_max_positions_regime default == 6",
          e._max_positions_regime == 6,
          f"got {e._max_positions_regime}")


def test_risk_guardian_init():
    """RiskGuardian initializes with correct regime parameters."""
    print("\n── RiskGuardian Init ──")
    from risk_guardian import RiskGuardian
    rg = RiskGuardian(
        initial_equity=100_000,
        max_drawdown_pct=0.15,
        daily_loss_limit_pct=0.03,
    )
    check("RiskGuardian instantiates", rg is not None)
    check("initial_equity stored",
          hasattr(rg, "initial_equity") or hasattr(rg, "_initial_equity"))

    # compute_safe_position_size exists and accepts regime_scale
    import inspect
    sig = inspect.signature(rg.compute_safe_position_size)
    params = list(sig.parameters.keys())
    check("compute_safe_position_size has regime_scale param",
          "regime_scale" in params,
          f"params: {params}")


def test_strategy_engine_init():
    """StrategyEngine initializes with correct config."""
    print("\n── StrategyEngine Init ──")
    from strategy_engine import StrategyEngine, EngineConfig
    cfg = EngineConfig()
    se = StrategyEngine(cfg)
    check("StrategyEngine instantiates", se is not None)
    check("EngineConfig instantiates", cfg is not None)


def test_turnover_gate():
    """Anti-churn turnover gate math is correct."""
    print("\n── Turnover Gate ──")
    from run_v28_production import EquityEngine
    e = EquityEngine("paper")

    # Equity = 100k, turnover cap = 15% = $15k
    check("$10k trade allowed on fresh day",
          e._turnover_allows_trade(10_000, 100_000))

    e._daily_turnover_used = 14_000
    check("$2k trade blocked when $14k already used (>15%)",
          not e._turnover_allows_trade(2_000, 100_000))

    e._daily_turnover_used = 14_000
    check("$1k trade allowed when $14k used (exactly 15%)",
          e._turnover_allows_trade(1_000, 100_000))


def test_min_hold():
    """Min-hold gate enforces 6-bar minimum."""
    print("\n── Min Hold ──")
    from run_v28_production import EquityEngine
    e = EquityEngine("paper")

    e._position_entry_bar["AAPL"] = 10
    e._bar_count = 14
    check("4 bars held → exit blocked",
          not e._min_hold_allows_exit("AAPL"))

    e._bar_count = 16
    check("6 bars held → exit allowed",
          e._min_hold_allows_exit("AAPL"))

    check("Unknown symbol → exit allowed",
          e._min_hold_allows_exit("ZZZZ"))


def test_universe_filter():
    """Universe filter blocks banned / freefall / death-cross symbols."""
    print("\n── Universe Filter ──")
    from run_v28_production import EquityEngine
    e = EquityEngine("paper")

    check("BBBY blocked by BANNED_SYMBOLS",
          not e._passes_universe_filter("BBBY"))
    check("SIVB blocked by BANNED_SYMBOLS",
          not e._passes_universe_filter("SIVB"))
    check("AAPL passes (not banned, no price data)",
          e._passes_universe_filter("AAPL"))

    # Freefall: -10% in 5 bars
    prices = np.array([100.0] * 200 + [90.0])
    # Need exactly index[-6] → index[-1] drop > 8%
    freefall_prices = np.array([100.0] * 195 + [100, 100, 100, 100, 100, 88])
    check("Freefall (-12% in 5 bars) blocked",
          not e._passes_universe_filter("TEST", freefall_prices))

    # Death cross: SMA50 < SMA200
    death_cross_prices = np.concatenate([
        np.full(150, 120.0),  # old high prices (push SMA200 up)
        np.full(50, 80.0),    # recent low prices (SMA50 low)
    ])
    check("Death-cross (SMA50 < SMA200) blocked",
          not e._passes_universe_filter("TEST", death_cross_prices))


def test_volume_check():
    """Volume filter: avg > 500K and today > 0.3x avg (liquidity gate)."""
    print("\n── Volume Check ──")
    from run_v28_production import EquityEngine
    e = EquityEngine("paper")

    # Illiquid stock: avg vol = 100K (< 500K minimum)
    illiquid = np.array([100_000.0] * 20 + [100_000.0])
    check("Illiquid stock (avg 100K) blocked",
          not e._passes_volume_check(illiquid))

    # Dead day: today vol = 0.1x avg (< 0.3x threshold)
    dead_day = np.array([1_000_000.0] * 20 + [100_000.0])
    check("Dead day (0.1x avg) blocked",
          not e._passes_volume_check(dead_day))

    # Normal day: today vol = 0.5x avg (>= 0.3x threshold)
    normal = np.array([1_000_000.0] * 20 + [500_000.0])
    check("Normal volume (0.5x avg) passes",
          e._passes_volume_check(normal))

    # None → allow (no data to check)
    check("None volumes → passes (no data)",
          e._passes_volume_check(None))


def test_bracket_order_structure():
    """Bracket order dict has required fields."""
    print("\n── Bracket Order Structure ──")
    # Verify the bracket order construction pattern exists in source
    import inspect
    from run_v28_production import EquityEngine
    source = inspect.getsource(EquityEngine._execute_equity_trade)

    check("Uses order_class='bracket'",
          "'bracket'" in source or '"bracket"' in source)
    check("Includes stop_loss in order_data",
          "stop_loss" in source)
    check("Includes take_profit in order_data",
          "take_profit" in source)
    check("Uses type='limit' (not market)",
          "'limit'" in source or '"limit"' in source)
    check("NEVER uses 'market'",
          "'market'" not in source and '"market"' not in source,
          "found 'market' order type in _execute_equity_trade")


def test_metrics_module():
    """Prometheus metrics module imports and has expected exports."""
    print("\n── Metrics Module ──")
    try:
        from src.metrics import MetricsServer
        check("MetricsServer importable", True)
    except Exception as e:
        check("MetricsServer importable", False, str(e))
        return

    try:
        from src.metrics import (
            PORTFOLIO_VALUE, POSITIONS_COUNT, CYCLE_DURATION,
            SIGNALS_TOTAL, ORDERS_TOTAL, FILTERS_BLOCKED,
        )
        check("Prometheus gauges/counters importable", True)
    except Exception as e:
        check("Prometheus gauges/counters importable", False, str(e))


# ── Run all tests ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  SMOKE TESTS — run_v28_production.py")
    print("=" * 60)

    test_imports()
    test_banned_symbols()
    test_equity_universe_clean()
    test_equity_engine_constants()
    test_equity_engine_init()
    test_risk_guardian_init()
    test_strategy_engine_init()
    test_turnover_gate()
    test_min_hold()
    test_universe_filter()
    test_volume_check()
    test_bracket_order_structure()
    test_metrics_module()

    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
    print("=" * 60)

    sys.exit(1 if FAIL > 0 else 0)

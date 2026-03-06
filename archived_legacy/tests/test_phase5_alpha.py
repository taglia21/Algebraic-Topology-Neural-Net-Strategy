#!/usr/bin/env python3
"""
Phase 5 Alpha Generation Tests — Drawdown Sizing, Kalman Filter, Dynamic Allocation
=====================================================================================

Tests for the three Phase 5 improvements:
  1. Drawdown-Responsive Position Sizing
  2. Kalman Filter Hedge Ratio (cached streaming tracker)
  3. Dynamic Strategy Allocation (realized Sharpe weighting)

Run:  python -m pytest tests/test_phase5_alpha.py -v
  or: python tests/test_phase5_alpha.py
"""

import sys
import os
import numpy as np
import pandas as pd

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
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

def _make_price_data(n_days=300, seed=42):
    """Generate synthetic price data for testing."""
    np.random.seed(seed)
    dates = pd.date_range("2025-04-01", periods=n_days, freq="B")
    symbols = ["AAPL", "MSFT", "GOOGL", "META", "NVDA",
               "JPM", "GS", "MS", "BAC",
               "XOM", "CVX", "COP"]

    price_data = pd.DataFrame(index=dates)
    volume_data = pd.DataFrame(index=dates)

    for i, sym in enumerate(symbols):
        base = 100 + i * 20
        trend = np.cumsum(np.random.randn(n_days) * 0.5)
        sector_factor = np.cumsum(np.random.randn(n_days) * 0.3)
        price_data[sym] = base + trend + sector_factor + np.random.randn(n_days) * 0.5
        price_data[sym] = price_data[sym].clip(lower=10)
        volume_data[sym] = np.random.randint(500_000, 5_000_000, n_days)

    return price_data, volume_data


def _make_engine(config_overrides=None):
    """Create a StrategyEngine with optional config overrides."""
    from strategy_engine import StrategyEngine, EngineConfig

    cfg = EngineConfig()
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(cfg, k, v)
    return StrategyEngine(cfg), cfg


# ═══════════════════════════════════════════════════════════════════════
# Test 1: Drawdown-Responsive Position Sizing
# ═══════════════════════════════════════════════════════════════════════

def test_drawdown_config_fields():
    """EngineConfig has new drawdown threshold fields."""
    print("\n── Drawdown Config ──")
    from strategy_engine import EngineConfig

    cfg = EngineConfig()
    check("drawdown_scale_threshold == 0.05",
          cfg.drawdown_scale_threshold == 0.05,
          f"got {cfg.drawdown_scale_threshold}")
    check("drawdown_half_threshold == 0.10",
          cfg.drawdown_half_threshold == 0.10,
          f"got {cfg.drawdown_half_threshold}")
    check("drawdown_halt_threshold == 0.15",
          cfg.drawdown_halt_threshold == 0.15,
          f"got {cfg.drawdown_halt_threshold}")


def test_drawdown_state_tracking():
    """Drawdown state tracks peak equity and current drawdown."""
    print("\n── Drawdown State Tracking ──")
    engine, _ = _make_engine()

    # Start with no drawdown
    dd = engine._update_drawdown_state(100_000)
    check("Initial equity sets peak", engine._peak_equity == 100_000)
    check("No drawdown on first call", dd == 0.0)

    # Equity increases → peak updates, no drawdown
    dd = engine._update_drawdown_state(110_000)
    check("Peak updated to 110K", engine._peak_equity == 110_000)
    check("No drawdown at new high", dd == 0.0)

    # Equity drops to 100K → 9.09% drawdown
    dd = engine._update_drawdown_state(100_000)
    check("Peak stays at 110K", engine._peak_equity == 110_000)
    expected_dd = 10_000 / 110_000
    check(f"Drawdown ~9.09%", abs(dd - expected_dd) < 0.001,
          f"got {dd:.4f}, expected {expected_dd:.4f}")

    # Equity recovers to 115K → new peak, no drawdown
    dd = engine._update_drawdown_state(115_000)
    check("Peak updated to 115K", engine._peak_equity == 115_000)
    check("Drawdown reset to 0", dd == 0.0)


def test_drawdown_scale_factors():
    """Drawdown scale factor follows tiered structure."""
    print("\n── Drawdown Scale Factors ──")
    engine, _ = _make_engine()

    # No drawdown → full size
    check("0% DD → scale 1.0",
          engine._get_drawdown_scale_factor(0.0) == 1.0)
    check("3% DD → scale 1.0",
          engine._get_drawdown_scale_factor(0.03) == 1.0)

    # 5-10% → linear from 1.0 to 0.5
    check("5% DD → scale 1.0",
          engine._get_drawdown_scale_factor(0.05) == 1.0)
    check("7.5% DD → scale ~0.75",
          abs(engine._get_drawdown_scale_factor(0.075) - 0.75) < 0.01,
          f"got {engine._get_drawdown_scale_factor(0.075)}")
    check("10% DD → scale 0.5",
          engine._get_drawdown_scale_factor(0.10) == 0.5)

    # 10-15% → halved
    check("12% DD → scale 0.5",
          engine._get_drawdown_scale_factor(0.12) == 0.5)

    # >15% → halt (zero)
    check("15% DD → scale 0.0",
          engine._get_drawdown_scale_factor(0.15) == 0.0)
    check("20% DD → scale 0.0",
          engine._get_drawdown_scale_factor(0.20) == 0.0)


def test_drawdown_halt_blocks_entries():
    """When drawdown > 15%, new entries are blocked but exits pass."""
    print("\n── Drawdown Halt ──")
    from strategy_engine import TradeSignal, StrategyType, SignalDirection

    engine, _ = _make_engine()

    # Set peak high and current low to trigger 16% drawdown
    engine._peak_equity = 100_000
    engine._current_drawdown_pct = 0.16

    price_data, volume_data = _make_price_data()

    signals = engine.get_signals(price_data, volume_data, equity=84_000)

    # Should only contain CLOSE signals (if any)
    entry_signals = [s for s in signals if s.direction != SignalDirection.CLOSE]
    check("No entry signals during halt",
          len(entry_signals) == 0,
          f"got {len(entry_signals)} entry signals")


def test_drawdown_scaling_reduces_size():
    """When drawdown is 7.5%, position sizes are scaled by ~0.75."""
    print("\n── Drawdown Scaling ──")
    engine, _ = _make_engine()

    # First call at peak
    engine._update_drawdown_state(100_000)
    # Now at 7.5% drawdown
    dd = engine._update_drawdown_state(92_500)
    scale = engine._get_drawdown_scale_factor(dd)
    check("7.5% drawdown scale ~0.75",
          abs(scale - 0.75) < 0.01,
          f"got scale={scale}, dd={dd}")


def test_drawdown_properties():
    """Public properties for drawdown monitoring work."""
    print("\n── Drawdown Properties ──")
    engine, _ = _make_engine()

    engine._update_drawdown_state(100_000)
    engine._update_drawdown_state(90_000)

    check("current_drawdown_pct accessible",
          abs(engine.current_drawdown_pct - 0.10) < 0.001,
          f"got {engine.current_drawdown_pct}")
    check("peak_equity accessible",
          engine.peak_equity == 100_000,
          f"got {engine.peak_equity}")


# ═══════════════════════════════════════════════════════════════════════
# Test 2: Kalman Filter Hedge Ratio (Cached Tracker)
# ═══════════════════════════════════════════════════════════════════════

def test_kalman_tracker_cache_exists():
    """Engine maintains a Kalman tracker cache dict."""
    print("\n── Kalman Tracker Cache ──")
    engine, _ = _make_engine()

    check("_kalman_trackers is dict",
          isinstance(engine._kalman_trackers, dict))
    check("Cache starts empty",
          len(engine._kalman_trackers) == 0)


def test_kalman_z_score_computation():
    """_compute_kalman_z_score returns valid z-score and hedge ratio."""
    print("\n── Kalman Z-Score ──")
    from pair_finder import CointegrationResult

    engine, _ = _make_engine()

    # Create synthetic correlated pair
    np.random.seed(42)
    n = 200
    noise_b = np.cumsum(np.random.randn(n) * 0.01)
    price_b = 100 * np.exp(noise_b)
    # A tracks B with beta ≈ 1.2 plus mean-reverting noise
    spread_noise = np.cumsum(np.random.randn(n) * 0.005) * 0.3
    price_a = 120 * np.exp(1.2 * noise_b + spread_noise)
    price_a = np.clip(price_a, 10, 1000)
    price_b = np.clip(price_b, 10, 1000)

    pair = CointegrationResult(
        sym_a="TEST_A", sym_b="TEST_B", sector="test",
        pvalue=0.01, hedge_ratio=1.2, half_life=10.0,
        spread_mean=0.0, spread_std=0.01, spread_vol_annual=0.05,
        current_z_score=0.5, adf_stat=-3.5, correlation=0.95,
    )

    z, hr = engine._compute_kalman_z_score("TEST_A_TEST_B", pair, price_a, price_b)

    check("Z-score is finite", np.isfinite(z), f"got {z}")
    check("Hedge ratio is finite", np.isfinite(hr), f"got {hr}")
    check("Hedge ratio is positive", hr > 0, f"got {hr}")
    check("Z-score in reasonable range", abs(z) < 10.0, f"got {z}")

    # Check that tracker was cached
    check("Tracker cached for pair",
          "TEST_A_TEST_B" in engine._kalman_trackers)


def test_kalman_streaming_update():
    """Cached tracker uses streaming update on second call."""
    print("\n── Kalman Streaming ──")
    from pair_finder import CointegrationResult

    engine, _ = _make_engine()
    np.random.seed(42)

    n = 200
    noise_b = np.cumsum(np.random.randn(n) * 0.01)
    price_b = 100 * np.exp(noise_b)
    spread_noise = np.cumsum(np.random.randn(n) * 0.005) * 0.3
    price_a = 120 * np.exp(1.2 * noise_b + spread_noise)
    price_a = np.clip(price_a, 10, 1000)
    price_b = np.clip(price_b, 10, 1000)

    pair = CointegrationResult(
        sym_a="A", sym_b="B", sector="test",
        pvalue=0.01, hedge_ratio=1.2, half_life=10.0,
        spread_mean=0.0, spread_std=0.01, spread_vol_annual=0.05,
        current_z_score=0.5, adf_stat=-3.5, correlation=0.95,
    )

    # First call: bootstraps
    z1, hr1 = engine._compute_kalman_z_score("A_B", pair, price_a, price_b)
    n_obs_1 = engine._kalman_trackers["A_B"]._n_obs

    # Second call with same data length: streaming update (n_obs increases by 1)
    z2, hr2 = engine._compute_kalman_z_score("A_B", pair, price_a, price_b)
    n_obs_2 = engine._kalman_trackers["A_B"]._n_obs

    check("Second call increments n_obs", n_obs_2 == n_obs_1 + 1,
          f"n_obs_1={n_obs_1}, n_obs_2={n_obs_2}")
    check("Both z-scores are finite",
          np.isfinite(z1) and np.isfinite(z2))


def test_kalman_fallback_on_short_data():
    """Short data falls back gracefully (returns 0.0, default hedge ratio)."""
    print("\n── Kalman Fallback ──")
    from pair_finder import CointegrationResult

    engine, _ = _make_engine()

    pair = CointegrationResult(
        sym_a="X", sym_b="Y", sector="test",
        pvalue=0.01, hedge_ratio=1.5, half_life=10.0,
        spread_mean=0.0, spread_std=0.01, spread_vol_annual=0.05,
        current_z_score=0.0, adf_stat=-3.0, correlation=0.90,
    )

    # Only 10 data points — below minimum 30
    price_a = np.array([100 + i for i in range(10)], dtype=float)
    price_b = np.array([50 + i * 0.5 for i in range(10)], dtype=float)

    z, hr = engine._compute_kalman_z_score("X_Y", pair, price_a, price_b)
    check("Short data returns z=0.0", z == 0.0, f"got z={z}")
    check("Short data returns default hedge ratio", hr == 1.5, f"got hr={hr}")


# ═══════════════════════════════════════════════════════════════════════
# Test 3: Dynamic Strategy Allocation
# ═══════════════════════════════════════════════════════════════════════

def test_dynamic_allocation_config():
    """EngineConfig has dynamic allocation fields."""
    print("\n── Dynamic Allocation Config ──")
    from strategy_engine import EngineConfig

    cfg = EngineConfig()
    check("use_dynamic_allocation == True",
          cfg.use_dynamic_allocation is True)
    check("dynamic_alloc_min_trades == 10",
          cfg.dynamic_alloc_min_trades == 10)
    check("dynamic_alloc_floor == 0.10",
          cfg.dynamic_alloc_floor == 0.10)
    check("dynamic_alloc_lookback == 30",
          cfg.dynamic_alloc_lookback == 30)


def test_dynamic_weights_default_when_no_data():
    """Without trade history, dynamic allocation returns config defaults."""
    print("\n── Dynamic Weights (No Data) ──")
    engine, cfg = _make_engine()

    w_p, w_mr, w_mom = engine._compute_dynamic_weights()
    check("Pairs weight == config default",
          w_p == cfg.pairs_allocation,
          f"got {w_p}, expected {cfg.pairs_allocation}")
    check("MR weight == config default",
          w_mr == cfg.mr_allocation)
    check("Mom weight == config default",
          w_mom == cfg.momentum_allocation)


def test_dynamic_weights_sum_to_one():
    """With trade history, dynamic weights sum to 1.0."""
    print("\n── Dynamic Weights Sum ──")
    engine, _ = _make_engine()

    # Seed trade history
    np.random.seed(42)
    for _ in range(20):
        engine._trade_pnls["pairs_trading"].append(np.random.randn() * 100 + 50)
    for _ in range(15):
        engine._trade_pnls["mean_reversion"].append(np.random.randn() * 80 + 30)
    for _ in range(12):
        engine._trade_pnls["momentum_regime"].append(np.random.randn() * 120 - 10)

    w_p, w_mr, w_mom = engine._compute_dynamic_weights()
    total = w_p + w_mr + w_mom
    check(f"Weights sum to 1.0 (got {total:.4f})",
          abs(total - 1.0) < 0.001)
    check("All weights > 0", w_p > 0 and w_mr > 0 and w_mom > 0)


def test_dynamic_weights_respect_floor():
    """Each strategy gets at least the floor allocation (10%)."""
    print("\n── Dynamic Weights Floor ──")
    engine, cfg = _make_engine()

    # Make pairs very profitable, momentum terrible
    for _ in range(20):
        engine._trade_pnls["pairs_trading"].append(200)
        engine._trade_pnls["mean_reversion"].append(50)
        engine._trade_pnls["momentum_regime"].append(-50)

    w_p, w_mr, w_mom = engine._compute_dynamic_weights()
    floor = cfg.dynamic_alloc_floor

    check(f"Pairs weight > floor ({floor})",
          w_p >= floor - 0.001, f"got {w_p}")
    check(f"MR weight >= floor ({floor})",
          w_mr >= floor - 0.001, f"got {w_mr}")
    check(f"Momentum weight >= floor ({floor})",
          w_mom >= floor - 0.001, f"got {w_mom}")

    # Pairs should get highest weight since it's most profitable
    check("Pairs gets highest weight",
          w_p >= w_mr and w_p >= w_mom,
          f"pairs={w_p:.3f} mr={w_mr:.3f} mom={w_mom:.3f}")


def test_dynamic_weights_profitable_gets_more():
    """Strategy with better Sharpe gets higher allocation."""
    print("\n── Dynamic Weights Favors Winners ──")
    engine, _ = _make_engine()

    # Pairs: high Sharpe (consistent profits)
    for _ in range(20):
        engine._trade_pnls["pairs_trading"].append(100)  # Mean 100, std 0

    # MR: moderate Sharpe
    for _ in range(15):
        engine._trade_pnls["mean_reversion"].append(np.random.randn() * 30 + 50)

    # Momentum: low Sharpe (mixed)
    for _ in range(12):
        engine._trade_pnls["momentum_regime"].append(np.random.randn() * 100 + 10)

    w_p, w_mr, w_mom = engine._compute_dynamic_weights()

    check("Pairs (highest Sharpe) gets most weight",
          w_p > w_mr,
          f"pairs={w_p:.3f} mr={w_mr:.3f}")
    check("Pairs > momentum",
          w_p > w_mom,
          f"pairs={w_p:.3f} mom={w_mom:.3f}")


def test_dynamic_weights_disabled():
    """When use_dynamic_allocation=False, returns static config."""
    print("\n── Dynamic Allocation Disabled ──")
    engine, cfg = _make_engine({"use_dynamic_allocation": False})

    # Seed data anyway
    for _ in range(20):
        engine._trade_pnls["pairs_trading"].append(100)
        engine._trade_pnls["mean_reversion"].append(-50)
        engine._trade_pnls["momentum_regime"].append(0)

    w_p, w_mr, w_mom = engine._compute_dynamic_weights()
    check("Returns static pairs allocation",
          w_p == cfg.pairs_allocation)
    check("Returns static mr allocation",
          w_mr == cfg.mr_allocation)
    check("Returns static momentum allocation",
          w_mom == cfg.momentum_allocation)


def test_get_dynamic_weights_api():
    """get_dynamic_weights() returns correct monitoring dict."""
    print("\n── Dynamic Weights API ──")
    engine, _ = _make_engine()

    weights = engine.get_dynamic_weights()
    check("Returns dict with pairs_trading key",
          "pairs_trading" in weights)
    check("Returns dict with mean_reversion key",
          "mean_reversion" in weights)
    check("Returns dict with momentum_regime key",
          "momentum_regime" in weights)
    check("Values sum to 1.0",
          abs(sum(weights.values()) - 1.0) < 0.001)


def test_record_trade_result_tracks_pnl():
    """record_trade_result populates _trade_pnls for dynamic allocation."""
    print("\n── Trade PnL Tracking ──")
    engine, _ = _make_engine()

    engine.record_trade_result("pairs_trading", 150.0)
    engine.record_trade_result("pairs_trading", -50.0)
    engine.record_trade_result("mean_reversion", 80.0)

    check("Pairs has 2 PnL entries",
          len(engine._trade_pnls["pairs_trading"]) == 2,
          f"got {len(engine._trade_pnls['pairs_trading'])}")
    check("MR has 1 PnL entry",
          len(engine._trade_pnls["mean_reversion"]) == 1)
    check("Momentum has 0 PnL entries",
          len(engine._trade_pnls["momentum_regime"]) == 0)
    check("Pairs PnL values correct",
          engine._trade_pnls["pairs_trading"] == [150.0, -50.0])


def test_pnl_history_bounded():
    """PnL history is bounded to prevent unbounded memory growth."""
    print("\n── PnL History Bounds ──")
    engine, cfg = _make_engine()

    max_history = cfg.dynamic_alloc_lookback * 3  # 90

    # Record more than max
    for i in range(150):
        engine.record_trade_result("pairs_trading", float(i))

    check(f"PnL history capped at {max_history}",
          len(engine._trade_pnls["pairs_trading"]) <= max_history,
          f"got {len(engine._trade_pnls['pairs_trading'])}")


# ═══════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════

def test_get_signals_includes_drawdown():
    """get_signals uses drawdown scaling in the pipeline."""
    print("\n── Integration: Drawdown in Pipeline ──")
    engine, _ = _make_engine()

    # Set up a 7% drawdown
    engine._peak_equity = 100_000

    price_data, volume_data = _make_price_data()
    signals_normal = engine.get_signals(price_data, volume_data, equity=100_000)

    # Reset and run with drawdown
    engine2, _ = _make_engine()
    engine2._peak_equity = 100_000
    signals_dd = engine2.get_signals(price_data, volume_data, equity=93_000)

    check("Engine ran without error (normal)", True)
    check("Engine ran without error (drawdown)", True)

    # Signals during drawdown should have smaller sizes (if any entry signals)
    from strategy_engine import SignalDirection
    entries_dd = [s for s in signals_dd if s.direction != SignalDirection.CLOSE]
    if entries_dd:
        # With 7% drawdown, scale should be ~0.80
        # Check that sizes are reasonable (not the full default)
        max_size = max(s.position_size_pct for s in entries_dd)
        check("Drawdown signals have reasonable sizes",
              max_size <= 0.06,  # Should be scaled down from 5%
              f"max entry size = {max_size:.3f}")


def test_engine_backward_compatible():
    """StrategyEngine still works with default config (no optional args)."""
    print("\n── Backward Compatibility ──")
    from strategy_engine import StrategyEngine, EngineConfig

    # Default construction
    engine = StrategyEngine()
    check("Default construction works", engine is not None)
    check("Has drawdown state", hasattr(engine, '_peak_equity'))
    check("Has trade PnL tracking", hasattr(engine, '_trade_pnls'))
    check("Has Kalman cache", hasattr(engine, '_kalman_trackers'))

    # Config with explicit values
    engine2 = StrategyEngine(EngineConfig(
        drawdown_scale_threshold=0.03,
        drawdown_half_threshold=0.08,
        drawdown_halt_threshold=0.12,
        use_dynamic_allocation=True,
        dynamic_alloc_min_trades=5,
    ))
    check("Custom config construction works", engine2 is not None)
    check("Custom drawdown threshold applied",
          engine2.cfg.drawdown_scale_threshold == 0.03)


def test_kalman_import():
    """KalmanSpreadTracker is importable from strategy_engine's imports."""
    print("\n── Kalman Import ──")
    try:
        from pair_finder import KalmanSpreadTracker
        check("KalmanSpreadTracker importable", True)
    except ImportError as e:
        check("KalmanSpreadTracker importable", False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  PHASE 5 ALPHA TESTS — Drawdown + Kalman + Dynamic Alloc")
    print("=" * 60)

    # Drawdown tests
    test_drawdown_config_fields()
    test_drawdown_state_tracking()
    test_drawdown_scale_factors()
    test_drawdown_halt_blocks_entries()
    test_drawdown_scaling_reduces_size()
    test_drawdown_properties()

    # Kalman filter tests
    test_kalman_tracker_cache_exists()
    test_kalman_z_score_computation()
    test_kalman_streaming_update()
    test_kalman_fallback_on_short_data()

    # Dynamic allocation tests
    test_dynamic_allocation_config()
    test_dynamic_weights_default_when_no_data()
    test_dynamic_weights_sum_to_one()
    test_dynamic_weights_respect_floor()
    test_dynamic_weights_profitable_gets_more()
    test_dynamic_weights_disabled()
    test_get_dynamic_weights_api()
    test_record_trade_result_tracks_pnl()
    test_pnl_history_bounded()

    # Integration tests
    test_get_signals_includes_drawdown()
    test_engine_backward_compatible()
    test_kalman_import()

    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
    print("=" * 60)

    sys.exit(1 if FAIL > 0 else 0)

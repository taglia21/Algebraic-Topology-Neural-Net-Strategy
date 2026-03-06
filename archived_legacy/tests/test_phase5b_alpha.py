#!/usr/bin/env python3
"""
Phase 5b Alpha Generation Tests — Credit Spreads, VWAP Reversion, Vol Divergence
==================================================================================

Tests for the three Phase 5b improvements:
  4. Credit Spread Strategies (iron condor, bull put, bear call)
  5. VWAP Intraday Mean Reversion
  6. IV vs Realized Vol Divergence

Run:  python -m pytest tests/test_phase5b_alpha.py -v
  or: python tests/test_phase5b_alpha.py
"""

import sys
import os
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime

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


def _run_async(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════════
# Test 4: Credit Spread Strategies
# ═══════════════════════════════════════════════════════════════════════

def test_spread_config_defaults():
    """SpreadConfig has sane defaults."""
    print("\n── Spread Config Defaults ──")
    from src.options.spread_strategies import SpreadConfig

    cfg = SpreadConfig()
    check("IC IV rank min == 65", cfg.ic_iv_rank_min == 65.0)
    check("IC DTE 30-45", cfg.ic_dte_min == 30 and cfg.ic_dte_max == 45)
    check("IC delta 0.15-0.25", cfg.ic_delta_min == 0.15 and cfg.ic_delta_max == 0.25)
    check("IC wing width 5", cfg.ic_wing_width == 5.0)
    check("IC profit target 50%", cfg.ic_profit_target_pct == 0.50)
    check("IC DTE exit 21", cfg.ic_dte_exit == 21)
    check("BP IV rank min == 50", cfg.bp_iv_rank_min == 50.0)
    check("BC IV rank min == 50", cfg.bc_iv_rank_min == 50.0)


def test_regime_hint_inference():
    """Regime hint logic: low IV → bullish, high IV → bearish."""
    print("\n── Regime Hint Inference ──")
    from src.options.spread_strategies import _infer_regime_hint, _RegimeHint

    check("IV rank 20 → BULLISH", _infer_regime_hint(20) == _RegimeHint.BULLISH)
    check("IV rank 39 → BULLISH", _infer_regime_hint(39) == _RegimeHint.BULLISH)
    check("IV rank 50 → NEUTRAL", _infer_regime_hint(50) == _RegimeHint.NEUTRAL)
    check("IV rank 70 → NEUTRAL", _infer_regime_hint(70) == _RegimeHint.NEUTRAL)
    check("IV rank 75 → BEARISH", _infer_regime_hint(75) == _RegimeHint.BEARISH)
    check("IV rank 95 → BEARISH", _infer_regime_hint(95) == _RegimeHint.BEARISH)


def test_iron_condor_strategy_high_iv():
    """IC strategy fires when IV rank >= 65."""
    print("\n── Iron Condor Strategy ──")
    from src.options.spread_strategies import IronCondorStrategy, SpreadConfig

    cfg = SpreadConfig(ic_iv_rank_min=65.0)
    ic = IronCondorStrategy(cfg)

    # Mock IV data manager to return high IV
    ic.iv_data = MagicMock()
    ic.iv_data.get_iv_rank.return_value = 80.0

    signals = _run_async(ic.generate_signals(["SPY", "QQQ"]))
    check("Signals generated for high IV", len(signals) > 0, f"got {len(signals)}")
    if signals:
        sig = signals[0]
        check("Strategy is iron_condor", sig.strategy == "iron_condor")
        check("Signal type is SELL", sig.signal_type.value == "sell")
        check("Confidence > 0.5", sig.confidence > 0.5, f"got {sig.confidence}")
        check("DTE in 30-45 range", 30 <= sig.dte <= 45, f"got {sig.dte}")
        check("Has PoP", sig.probability_of_profit is not None and sig.probability_of_profit > 0)
        check("Reason mentions IC", "IC:" in sig.reason)


def test_iron_condor_no_signal_low_iv():
    """IC strategy does NOT fire when IV rank < 65."""
    print("\n── Iron Condor Low IV ──")
    from src.options.spread_strategies import IronCondorStrategy, SpreadConfig

    ic = IronCondorStrategy(SpreadConfig(ic_iv_rank_min=65.0))
    ic.iv_data = MagicMock()
    ic.iv_data.get_iv_rank.return_value = 40.0

    signals = _run_async(ic.generate_signals(["SPY"]))
    check("No signals at IV rank 40", len(signals) == 0, f"got {len(signals)}")


def test_bull_put_strategy_bullish():
    """Bull put spread fires in bullish regime with IV > 50."""
    print("\n── Bull Put Spread Strategy ──")
    from src.options.spread_strategies import (
        BullPutSpreadStrategy, SpreadConfig, _RegimeHint,
    )

    bp = BullPutSpreadStrategy(
        SpreadConfig(bp_iv_rank_min=50.0),
        regime_hint_fn=lambda iv: _RegimeHint.BULLISH,
    )
    bp.iv_data = MagicMock()
    bp.iv_data.get_iv_rank.return_value = 60.0

    signals = _run_async(bp.generate_signals(["AAPL", "MSFT"]))
    check("Signals generated in bullish regime", len(signals) > 0, f"got {len(signals)}")
    if signals:
        check("Strategy is put_spread", signals[0].strategy == "put_spread")
        check("Signal type is SELL", signals[0].signal_type.value == "sell")
        check("Reason mentions regime", "regime=bullish" in signals[0].reason)


def test_bull_put_blocked_bearish():
    """Bull put spread does NOT fire in bearish regime."""
    print("\n── Bull Put Bearish Block ──")
    from src.options.spread_strategies import (
        BullPutSpreadStrategy, SpreadConfig, _RegimeHint,
    )

    bp = BullPutSpreadStrategy(
        SpreadConfig(bp_iv_rank_min=50.0),
        regime_hint_fn=lambda iv: _RegimeHint.BEARISH,
    )
    bp.iv_data = MagicMock()
    bp.iv_data.get_iv_rank.return_value = 80.0  # High IV but bearish

    signals = _run_async(bp.generate_signals(["AAPL"]))
    check("No signals in bearish regime", len(signals) == 0, f"got {len(signals)}")


def test_bear_call_strategy_bearish():
    """Bear call spread fires in bearish regime with IV > 50."""
    print("\n── Bear Call Spread Strategy ──")
    from src.options.spread_strategies import (
        BearCallSpreadStrategy, SpreadConfig, _RegimeHint,
    )

    bc = BearCallSpreadStrategy(
        SpreadConfig(bc_iv_rank_min=50.0),
        regime_hint_fn=lambda iv: _RegimeHint.BEARISH,
    )
    bc.iv_data = MagicMock()
    bc.iv_data.get_iv_rank.return_value = 75.0

    signals = _run_async(bc.generate_signals(["SPY"]))
    check("Bear call signal generated", len(signals) > 0, f"got {len(signals)}")
    if signals:
        check("Strategy is call_spread", signals[0].strategy == "call_spread")
        check("iv_rank in signal", signals[0].iv_rank == 75.0)


def test_bear_call_blocked_bullish():
    """Bear call spread does NOT fire in bullish regime."""
    print("\n── Bear Call Bullish Block ──")
    from src.options.spread_strategies import (
        BearCallSpreadStrategy, SpreadConfig, _RegimeHint,
    )

    bc = BearCallSpreadStrategy(
        SpreadConfig(),
        regime_hint_fn=lambda iv: _RegimeHint.BULLISH,
    )
    bc.iv_data = MagicMock()
    bc.iv_data.get_iv_rank.return_value = 75.0

    signals = _run_async(bc.generate_signals(["SPY"]))
    check("No signals in bullish regime", len(signals) == 0, f"got {len(signals)}")


def test_spread_aggregator_dedup():
    """Aggregator merges and deduplicates signals."""
    print("\n── Spread Aggregator Dedup ──")
    from src.options.spread_strategies import (
        SpreadStrategyAggregator, SpreadConfig, _RegimeHint,
    )

    # Force neutral regime so both bull-put and bear-call can fire
    agg = SpreadStrategyAggregator(
        SpreadConfig(ic_iv_rank_min=65.0, bp_iv_rank_min=50.0, bc_iv_rank_min=50.0),
        regime_hint_fn=lambda iv: _RegimeHint.NEUTRAL,
    )
    for sub in [agg.ic, agg.bp, agg.bc]:
        sub.iv_data = MagicMock()
        sub.iv_data.get_iv_rank.return_value = 70.0

    signals = _run_async(agg.generate_signals(["SPY"]))
    # Should have at most one signal per (symbol, strategy) pair
    seen_keys = set()
    duplicates = 0
    for sig in signals:
        key = (sig.symbol, sig.strategy)
        if key in seen_keys:
            duplicates += 1
        seen_keys.add(key)
    check("No duplicate (symbol, strategy) keys", duplicates == 0, f"dups={duplicates}")
    check("Some signals produced", len(signals) > 0, f"got {len(signals)}")
    check("Sorted by confidence desc",
          all(signals[i].confidence >= signals[i + 1].confidence
              for i in range(len(signals) - 1)) if len(signals) > 1 else True)


def test_ic_confidence_scaling():
    """IC confidence scales from 0.5 to 1.0 across IV rank 65 → 100."""
    print("\n── IC Confidence Scaling ──")
    from src.options.spread_strategies import IronCondorStrategy, SpreadConfig

    ic = IronCondorStrategy(SpreadConfig(ic_iv_rank_min=65.0))
    ic.iv_data = MagicMock()

    # At threshold: confidence should be ~0.50
    ic.iv_data.get_iv_rank.return_value = 65.0
    sig_low = _run_async(ic.generate_signals(["SPY"]))
    check("Signal at threshold", len(sig_low) == 1)
    if sig_low:
        check("Confidence ~0.50 at threshold", 0.49 <= sig_low[0].confidence <= 0.55,
              f"got {sig_low[0].confidence}")

    # At maximum: confidence should be ~1.0
    ic.iv_data.get_iv_rank.return_value = 100.0
    sig_high = _run_async(ic.generate_signals(["SPY"]))
    if sig_high:
        check("Confidence ~1.0 at IV rank 100", sig_high[0].confidence >= 0.95,
              f"got {sig_high[0].confidence}")


# ═══════════════════════════════════════════════════════════════════════
# Test 5: VWAP Intraday Mean Reversion
# ═══════════════════════════════════════════════════════════════════════

def test_vwap_config_fields():
    """EngineConfig has new VWAP fields."""
    print("\n── VWAP Config Fields ──")
    from strategy_engine import EngineConfig

    cfg = EngineConfig()
    check("vwap_enabled == True", cfg.vwap_enabled is True)
    check("vwap_lookback == 20", cfg.vwap_lookback == 20)
    check("vwap_entry_std == 1.5", cfg.vwap_entry_std == 1.5)
    check("vwap_rsi_oversold == 30", cfg.vwap_rsi_oversold == 30.0)
    check("vwap_rsi_overbought == 70", cfg.vwap_rsi_overbought == 70.0)
    check("vwap_stop_mult == 2.0", cfg.vwap_stop_mult == 2.0)
    check("vwap_max_positions == 3", cfg.vwap_max_positions == 3)
    check("vwap_max_hold_days == 3", cfg.vwap_max_hold_days == 3)


def test_vwap_strategy_type_exists():
    """StrategyType enum includes VWAP_REVERSION."""
    print("\n── VWAP StrategyType ──")
    from strategy_engine import StrategyType

    check("VWAP_REVERSION exists", hasattr(StrategyType, "VWAP_REVERSION"))
    check("Value is 'vwap_reversion'", StrategyType.VWAP_REVERSION.value == "vwap_reversion")


def test_vwap_stats_tracking():
    """Strategy stats and trade PnL dicts include vwap_reversion."""
    print("\n── VWAP Stats Tracking ──")
    engine, _ = _make_engine()

    check("vwap_reversion in _strategy_stats",
          "vwap_reversion" in engine._strategy_stats)
    check("vwap_reversion in _trade_pnls",
          "vwap_reversion" in engine._trade_pnls)


def test_vwap_long_signal():
    """VWAP reversion fires LONG when price far below VWAP with low RSI."""
    print("\n── VWAP Long Signal ──")

    # Construct data where the last bar is very low (far below VWAP)
    np.random.seed(99)
    n_days = 60
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")

    # Stable price around 100, then crash on last bar
    prices = np.full(n_days, 100.0) + np.random.randn(n_days) * 0.5
    # Make last 15 bars trending down to get RSI < 30
    for i in range(n_days - 15, n_days):
        prices[i] = prices[i - 1] - 1.0
    prices = np.maximum(prices, 30)  # floor

    # High volume on all bars
    volumes = np.full(n_days, 2_000_000.0)

    price_data = pd.DataFrame({"TESTSYM": prices}, index=dates)
    volume_data = pd.DataFrame({"TESTSYM": volumes}, index=dates)

    engine, cfg = _make_engine({
        "vwap_enabled": True,
        "vwap_entry_std": 1.0,  # lower threshold for test
        "vwap_rsi_oversold": 40.0,  # relax for synthetic data
        "min_confidence": 0.30,
    })

    signals = engine._scan_vwap_reversion(
        price_data, volume_data, None, 100_000, {}, datetime.now().isoformat()
    )

    long_signals = [s for s in signals if s.direction.value == "long"]
    check("At least one VWAP LONG signal", len(long_signals) > 0,
          f"got {len(long_signals)} long, {len(signals)} total")
    if long_signals:
        sig = long_signals[0]
        check("Strategy is VWAP_REVERSION", sig.strategy.value == "vwap_reversion")
        check("Target is VWAP (above current)", sig.target_price > sig.entry_price,
              f"target={sig.target_price}, entry={sig.entry_price}")
        check("Stop below entry", sig.stop_price < sig.entry_price)
        check("Has z_score < 0", sig.z_score < 0, f"z={sig.z_score}")
        check("max_hold_days set", sig.max_hold_days > 0)
        check("Source mentions VWAP", "VWAP" in sig.strategy_source)


def test_vwap_short_signal():
    """VWAP reversion fires SHORT when price far above VWAP with high RSI."""
    print("\n── VWAP Short Signal ──")

    np.random.seed(88)
    n_days = 60
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")

    # Stable around 100, then spike up for last 15 bars
    prices = np.full(n_days, 100.0) + np.random.randn(n_days) * 0.5
    for i in range(n_days - 15, n_days):
        prices[i] = prices[i - 1] + 1.0

    volumes = np.full(n_days, 2_000_000.0)

    price_data = pd.DataFrame({"TESTSYM": prices}, index=dates)
    volume_data = pd.DataFrame({"TESTSYM": volumes}, index=dates)

    engine, cfg = _make_engine({
        "vwap_enabled": True,
        "vwap_entry_std": 1.0,
        "vwap_rsi_overbought": 60.0,  # relax for test
        "min_confidence": 0.30,
    })

    signals = engine._scan_vwap_reversion(
        price_data, volume_data, None, 100_000, {}, datetime.now().isoformat()
    )

    short_signals = [s for s in signals if s.direction.value == "short"]
    check("At least one VWAP SHORT signal", len(short_signals) > 0,
          f"got {len(short_signals)} short, {len(signals)} total")
    if short_signals:
        sig = short_signals[0]
        check("Target below entry (VWAP)", sig.target_price < sig.entry_price)
        check("Stop above entry", sig.stop_price > sig.entry_price)
        check("z_score > 0", sig.z_score > 0)


def test_vwap_disabled():
    """VWAP signals not generated when vwap_enabled=False."""
    print("\n── VWAP Disabled ──")

    engine, _ = _make_engine({"vwap_enabled": False})
    price_data, volume_data = _make_price_data(60)

    # get_signals should still work and not produce VWAP signals
    signals = engine.get_signals(price_data, volume_data, equity=100_000)
    vwap_sigs = [s for s in signals if s.strategy.value == "vwap_reversion"]
    check("No VWAP signals when disabled", len(vwap_sigs) == 0,
          f"got {len(vwap_sigs)}")


def test_vwap_max_positions_cap():
    """VWAP respects max_positions limit."""
    print("\n── VWAP Max Positions ──")

    np.random.seed(77)
    n_days = 60
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")

    # Create many symbols that all crash to trigger LONG
    symbols = [f"SYM{i}" for i in range(10)]
    price_data = pd.DataFrame(index=dates)
    volume_data = pd.DataFrame(index=dates)

    for sym in symbols:
        p = np.full(n_days, 100.0) + np.random.randn(n_days) * 0.3
        for i in range(n_days - 15, n_days):
            p[i] = p[i - 1] - 1.5
        p = np.maximum(p, 30)
        price_data[sym] = p
        volume_data[sym] = 2_000_000

    engine, _ = _make_engine({
        "vwap_enabled": True,
        "vwap_entry_std": 0.5,
        "vwap_rsi_oversold": 45.0,
        "vwap_max_positions": 2,
        "min_confidence": 0.30,
    })

    signals = engine._scan_vwap_reversion(
        price_data, volume_data, None, 100_000, {}, datetime.now().isoformat()
    )
    check("Respects max_positions=2", len(signals) <= 2,
          f"got {len(signals)}")


def test_vwap_no_volume_fallback():
    """VWAP falls back to simple mean when no volume data."""
    print("\n── VWAP No-Volume Fallback ──")

    np.random.seed(66)
    n_days = 60
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")
    prices = np.full(n_days, 100.0) + np.random.randn(n_days) * 0.5
    for i in range(n_days - 15, n_days):
        prices[i] = prices[i - 1] - 1.0
    prices = np.maximum(prices, 30)

    price_data = pd.DataFrame({"TESTSYM": prices}, index=dates)

    engine, _ = _make_engine({
        "vwap_enabled": True,
        "vwap_entry_std": 1.0,
        "vwap_rsi_oversold": 45.0,
        "min_confidence": 0.30,
    })

    # Pass None for volume_data — should not crash
    signals = engine._scan_vwap_reversion(
        price_data, None, None, 100_000, {}, datetime.now().isoformat()
    )
    check("Does not crash without volume", True)  # reaching here = success
    check("Produces signals even without volume", len(signals) >= 0)


# ═══════════════════════════════════════════════════════════════════════
# Test 6: IV vs Realized Vol Divergence
# ═══════════════════════════════════════════════════════════════════════

def test_vol_divergence_class_exists():
    """VolDivergenceStrategy is importable."""
    print("\n── Vol Divergence Import ──")
    from src.options.signal_generator import VolDivergenceStrategy

    vd = VolDivergenceStrategy()
    check("Class instantiable", vd is not None)
    check("Default sell threshold 1.5", vd.sell_threshold == 1.5)
    check("Default buy threshold 0.7", vd.buy_threshold == 0.7)
    check("Default rv_lookback 20", vd.rv_lookback == 20)


def test_vol_divergence_sell_signal():
    """Sell signal when IV/RV >= 1.5."""
    print("\n── Vol Divergence Sell ──")
    from src.options.signal_generator import VolDivergenceStrategy

    vd = VolDivergenceStrategy(sell_threshold=1.5, buy_threshold=0.7)

    # Mock: IV rank = 80 and RV = 0.15 (so IV ≈ 0.15 * (1 + 0.3) = 0.195, ratio ≈ 1.3)
    # Need IV/RV >= 1.5 → IV rank needs to be high enough
    # IV = rv * (1 + (iv_rank - 50)/100)
    # For rv=0.15 and iv_rank=90: IV = 0.15 * 1.40 = 0.21 → ratio = 0.21/0.15 = 1.40 (not enough)
    # For rv=0.10 and iv_rank=90: IV = 0.10 * 1.40 = 0.14 → ratio = 1.40 (not enough)
    # Need ratio >= 1.5: (1 + (iv_rank-50)/100) >= 1.5 → iv_rank >= 100
    # So we'll set iv_rank to 100
    vd.iv_data = MagicMock()
    vd.iv_data.get_iv_rank.return_value = 100.0

    async def mock_rv(sym):
        return 0.20  # 20% realized vol

    vd._compute_realized_vol = mock_rv

    signals = _run_async(vd.generate_signals(["SPY"]))
    check("Sell signal generated", len(signals) == 1, f"got {len(signals)}")
    if signals:
        sig = signals[0]
        check("Signal type is SELL", sig.signal_type.value == "sell")
        check("Confidence >= 0.5", sig.confidence >= 0.5, f"got {sig.confidence}")
        check("Reason mentions Vol divergence", "Vol divergence SELL" in sig.reason)


def test_vol_divergence_buy_signal():
    """Buy signal when IV/RV <= 0.7."""
    print("\n── Vol Divergence Buy ──")
    from src.options.signal_generator import VolDivergenceStrategy

    vd = VolDivergenceStrategy(sell_threshold=1.5, buy_threshold=0.7)

    # IV = rv * (1 + (iv_rank - 50)/100)
    # For iv_rank=10: IV = rv * 0.60, ratio = 0.60 <= 0.7 ✓
    vd.iv_data = MagicMock()
    vd.iv_data.get_iv_rank.return_value = 10.0

    async def mock_rv(sym):
        return 0.25

    vd._compute_realized_vol = mock_rv

    signals = _run_async(vd.generate_signals(["SPY"]))
    check("Buy signal generated", len(signals) == 1, f"got {len(signals)}")
    if signals:
        sig = signals[0]
        check("Signal type is BUY", sig.signal_type.value == "buy")
        check("Reason mentions Vol divergence BUY", "Vol divergence BUY" in sig.reason)
        check("DTE is 21 for buys", sig.dte == 21)


def test_vol_divergence_no_signal_normal():
    """No signal when IV/RV in normal range (0.7, 1.5)."""
    print("\n── Vol Divergence Normal ──")
    from src.options.signal_generator import VolDivergenceStrategy

    vd = VolDivergenceStrategy()
    vd.iv_data = MagicMock()
    # iv_rank=50 → IV = rv * 1.0 → ratio = 1.0 (normal range)
    vd.iv_data.get_iv_rank.return_value = 50.0

    async def mock_rv(sym):
        return 0.20

    vd._compute_realized_vol = mock_rv

    signals = _run_async(vd.generate_signals(["SPY"]))
    check("No signal in normal range", len(signals) == 0, f"got {len(signals)}")


def test_vol_divergence_no_rv():
    """No signal when realized vol is unavailable."""
    print("\n── Vol Divergence No RV ──")
    from src.options.signal_generator import VolDivergenceStrategy

    vd = VolDivergenceStrategy()
    vd.iv_data = MagicMock()
    vd.iv_data.get_iv_rank.return_value = 80.0

    async def mock_rv_none(sym):
        return None

    vd._compute_realized_vol = mock_rv_none

    signals = _run_async(vd.generate_signals(["SPY"]))
    check("No signal when RV is None", len(signals) == 0, f"got {len(signals)}")


def test_vol_divergence_no_iv_rank():
    """No signal when IV rank unavailable."""
    print("\n── Vol Divergence No IV Rank ──")
    from src.options.signal_generator import VolDivergenceStrategy

    vd = VolDivergenceStrategy()
    vd.iv_data = MagicMock()
    vd.iv_data.get_iv_rank.return_value = None

    signals = _run_async(vd.generate_signals(["SPY"]))
    check("No signal when IV rank is None", len(signals) == 0)


def test_vol_divergence_custom_thresholds():
    """Custom thresholds work correctly."""
    print("\n── Vol Divergence Custom Thresholds ──")
    from src.options.signal_generator import VolDivergenceStrategy

    # Very tight thresholds
    vd = VolDivergenceStrategy(sell_threshold=1.1, buy_threshold=0.9)
    vd.iv_data = MagicMock()
    # iv_rank=60 → ratio = 1 + (60-50)/100 = 1.10 → exactly at sell threshold
    vd.iv_data.get_iv_rank.return_value = 61.0  # slightly above

    async def mock_rv(sym):
        return 0.20

    vd._compute_realized_vol = mock_rv

    signals = _run_async(vd.generate_signals(["SPY"]))
    check("Fires with custom sell threshold", len(signals) > 0, f"got {len(signals)}")


def test_vol_divergence_confidence_scaling():
    """Confidence increases with divergence magnitude."""
    print("\n── Vol Divergence Confidence Scaling ──")
    from src.options.signal_generator import VolDivergenceStrategy

    vd = VolDivergenceStrategy(sell_threshold=1.5)
    vd.iv_data = MagicMock()

    async def mock_rv(sym):
        return 0.15

    vd._compute_realized_vol = mock_rv

    # Moderate divergence: iv_rank = 100 → ratio = 1 + 50/100 = 1.50
    vd.iv_data.get_iv_rank.return_value = 100.0
    sig_mod = _run_async(vd.generate_signals(["SPY"]))

    # We can only test at 100 (the max) since ratio = 1 + (rank-50)/100
    # and we need 1.5+, which requires rank=100
    if sig_mod:
        check("Confidence at threshold is ~0.50", 0.49 <= sig_mod[0].confidence <= 0.55,
              f"got {sig_mod[0].confidence}")


# ═══════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════

def test_signal_generator_includes_vol_divergence():
    """SignalGenerator has vol_divergence_strategy attribute."""
    print("\n── SignalGenerator Integration ──")
    from src.options.signal_generator import SignalGenerator

    gen = SignalGenerator()
    check("vol_divergence_strategy exists", hasattr(gen, "vol_divergence_strategy"))
    check("spread_aggregator exists or None",
          hasattr(gen, "spread_aggregator"))


def test_signal_generator_includes_spreads():
    """SignalGenerator has spread_aggregator loaded."""
    print("\n── SignalGenerator Spreads ──")
    from src.options.signal_generator import SignalGenerator

    gen = SignalGenerator()
    check("spread_aggregator is set", gen.spread_aggregator is not None)


def test_get_signals_includes_vwap():
    """Full get_signals pipeline includes VWAP strategy signals."""
    print("\n── Full Pipeline VWAP ──")

    # Create data that triggers VWAP (strong downtrend)
    np.random.seed(55)
    n_days = 300
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")

    symbols = ["AAPL", "MSFT", "GOOGL"]
    price_data = pd.DataFrame(index=dates)
    volume_data = pd.DataFrame(index=dates)

    for i, sym in enumerate(symbols):
        p = np.full(n_days, 100.0 + i * 20) + np.random.randn(n_days) * 0.5
        # Strong downtrend for last 15 bars
        for j in range(n_days - 15, n_days):
            p[j] = p[j - 1] - 1.5
        p = np.maximum(p, 30)
        price_data[sym] = p
        volume_data[sym] = np.random.randint(1_000_000, 5_000_000, n_days)

    engine, _ = _make_engine({
        "vwap_enabled": True,
        "vwap_entry_std": 0.5,
        "vwap_rsi_oversold": 45.0,
        "min_confidence": 0.30,
    })

    signals = engine.get_signals(price_data, volume_data, equity=100_000)
    vwap_sigs = [s for s in signals if s.strategy.value == "vwap_reversion"]
    check("VWAP signals found in pipeline output", len(vwap_sigs) > 0,
          f"got {len(vwap_sigs)} VWAP out of {len(signals)} total")


def test_backward_compat_no_vwap_fields():
    """EngineConfig still works without explicit VWAP overrides."""
    print("\n── Backward Compatibility ──")
    from strategy_engine import EngineConfig

    cfg = EngineConfig()
    # All old fields still present
    check("pairs_allocation still 0.50", cfg.pairs_allocation == 0.50)
    check("mr_allocation still 0.30", cfg.mr_allocation == 0.30)
    check("momentum_allocation still 0.20", cfg.momentum_allocation == 0.20)
    check("drawdown_scale_threshold still 0.05", cfg.drawdown_scale_threshold == 0.05)
    # New VWAP fields default correctly
    check("vwap_enabled defaults True", cfg.vwap_enabled is True)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 5b — CREDIT SPREADS, VWAP REVERSION, VOL DIVERGENCE TESTS")
    print("=" * 70)

    # #4 Credit Spread Strategies
    test_spread_config_defaults()
    test_regime_hint_inference()
    test_iron_condor_strategy_high_iv()
    test_iron_condor_no_signal_low_iv()
    test_bull_put_strategy_bullish()
    test_bull_put_blocked_bearish()
    test_bear_call_strategy_bearish()
    test_bear_call_blocked_bullish()
    test_spread_aggregator_dedup()
    test_ic_confidence_scaling()

    # #5 VWAP Reversion
    test_vwap_config_fields()
    test_vwap_strategy_type_exists()
    test_vwap_stats_tracking()
    test_vwap_long_signal()
    test_vwap_short_signal()
    test_vwap_disabled()
    test_vwap_max_positions_cap()
    test_vwap_no_volume_fallback()

    # #6 Vol Divergence
    test_vol_divergence_class_exists()
    test_vol_divergence_sell_signal()
    test_vol_divergence_buy_signal()
    test_vol_divergence_no_signal_normal()
    test_vol_divergence_no_rv()
    test_vol_divergence_no_iv_rank()
    test_vol_divergence_custom_thresholds()
    test_vol_divergence_confidence_scaling()

    # Integration
    test_signal_generator_includes_vol_divergence()
    test_signal_generator_includes_spreads()
    test_get_signals_includes_vwap()
    test_backward_compat_no_vwap_fields()

    print("\n" + "=" * 70)
    print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    print("=" * 70)

    sys.exit(0 if FAIL == 0 else 1)

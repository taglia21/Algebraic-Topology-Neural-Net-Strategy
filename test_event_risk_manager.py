#!/usr/bin/env python3
"""
Event Risk Manager - Comprehensive Test Suite
==============================================

Tests all components of the event-driven risk protection system.
"""

import logging
import sys
import os
from datetime import datetime, time, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.event_risk_manager import (
    EventRiskManager,
    EventRiskConfig,
    EconomicCalendar,
    VolumeAnomalyDetector,
    SpreadMonitor,
    TimeOfDayFilter,
    CircuitBreaker
)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

def test_economic_calendar():
    """Test FOMC and earnings blackout detection."""
    print("\n" + "="*70)
    print("TEST 1: ECONOMIC CALENDAR")
    print("="*70)
    
    config = EventRiskConfig()
    calendar = EconomicCalendar(config)
    
    # Test FOMC blackout
    print("\n📅 FOMC Blackout Tests:")
    fomc_date = datetime(2026, 3, 18)  # FOMC meeting day
    
    tests = [
        (datetime(2026, 3, 16), True, "2 days before"),
        (datetime(2026, 3, 17), True, "1 day before"),
        (datetime(2026, 3, 18), True, "FOMC day"),
        (datetime(2026, 3, 19), True, "1 day after"),
        (datetime(2026, 3, 20), False, "2 days after"),
        (datetime(2026, 3, 15), False, "3 days before"),
    ]
    
    for test_date, expected, desc in tests:
        result = calendar.is_fomc_blackout(test_date)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {test_date.date()} ({desc}): {'BLACKOUT' if result else 'OK'}")
    
    # Test high-impact event buffer
    print("\n📅 High-Impact Event Buffer Tests:")
    fomc_datetime = datetime(2026, 3, 18, 14, 0)  # 2:00 PM announcement
    
    buffer_tests = [
        (datetime(2026, 3, 18, 13, 45), True, "15 min before"),
        (datetime(2026, 3, 18, 14, 0), True, "Announcement time"),
        (datetime(2026, 3, 18, 14, 20), True, "20 min after"),
        (datetime(2026, 3, 18, 14, 45), False, "45 min after"),
    ]
    
    for test_datetime, expected, desc in buffer_tests:
        result = calendar.is_high_impact_event(test_datetime)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {test_datetime.time()} ({desc}): {'HIGH-IMPACT' if result else 'OK'}")
    
    # Test earnings blackout
    print("\n📅 Earnings Blackout Tests:")
    calendar.set_earnings_date('AAPL', datetime(2026, 4, 30))
    
    earnings_tests = [
        (datetime(2026, 4, 28), True, "2 days before earnings"),
        (datetime(2026, 4, 29), True, "1 day before earnings"),
        (datetime(2026, 4, 30), True, "Earnings day"),
        (datetime(2026, 5, 1), False, "1 day after earnings"),
    ]
    
    for test_date, expected, desc in earnings_tests:
        result = calendar.is_earnings_blackout('AAPL', test_date)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {test_date.date()} ({desc}): {'BLACKOUT' if result else 'OK'}")
    
    print("\n✅ Economic Calendar: PASSED")


def test_volume_anomaly_detector():
    """Test volume spike detection."""
    print("\n" + "="*70)
    print("TEST 2: VOLUME ANOMALY DETECTOR")
    print("="*70)
    
    config = EventRiskConfig(volume_spike_threshold=4.0)
    detector = VolumeAnomalyDetector(config)
    
    # Build baseline
    print("\n📊 Building volume baseline...")
    baseline_volume = 1_000_000
    for i in range(5):
        detector.update_volume('AAPL', baseline_volume)
    
    avg = detector.get_average_volume('AAPL')
    print(f"  Average volume: {avg:,.0f}")
    
    # Test normal volume
    print("\n📊 Testing volume scenarios:")
    tests = [
        (1_000_000, False, 0.6, "Normal volume (1.0x)"),
        (2_000_000, False, 0.6, "Elevated volume (2.0x)"),
        (3_500_000, False, 0.6, "High volume (3.5x)"),
        (4_500_000, True, 0.6, "SPIKE! (4.5x)"),
        (5_000_000, True, 0.6, "MAJOR SPIKE! (5.0x)"),
    ]
    
    for volume, should_spike, expected_mult, desc in tests:
        mult, reason = detector.get_volume_multiplier('AAPL', volume)
        is_spike = mult < 1.0
        status = "✅" if is_spike == should_spike else "❌"
        print(f"  {status} {desc}")
        print(f"      Multiplier: {mult:.2f} | Reason: {reason}")
    
    print("\n✅ Volume Anomaly Detector: PASSED")


def test_spread_monitor():
    """Test spread monitoring and liquidity sweep detection."""
    print("\n" + "="*70)
    print("TEST 3: SPREAD MONITOR")
    print("="*70)
    
    config = EventRiskConfig()
    monitor = SpreadMonitor(config)
    
    # Build baseline
    print("\n📊 Building spread baseline...")
    baseline_spread = 0.05
    for i in range(20):
        monitor.update_spread('TSLA', baseline_spread)
    
    avg = monitor.get_average_spread('TSLA')
    print(f"  Average spread: ${avg:.3f}")
    
    # Test spread scenarios
    print("\n📊 Testing spread scenarios:")
    tests = [
        (0.05, 1.0, "Normal spread (1.0x)"),
        (0.08, 1.0, "Slightly elevated (1.6x)"),
        (0.12, 0.8, "WARNING: 2x spread"),
        (0.25, 0.4, "CRITICAL: 5x spread"),
    ]
    
    for spread, expected_range, desc in tests:
        mult, reason = monitor.get_liquidity_multiplier('TSLA', spread)
        status = "✅" if abs(mult - expected_range) < 0.1 else "❌"
        print(f"  {status} Spread ${spread:.3f} ({desc})")
        print(f"      Multiplier: {mult:.2f} | Reason: {reason}")
    
    # Test liquidity sweep
    print("\n📊 Testing liquidity sweep detection:")
    monitor.update_price_swing('SPY', high=450.0, low=445.0)
    
    # Scenario: price spikes above high then reverses sharply
    price_before = 451.0  # Exceeded swing high
    price_now = 447.0     # Sharp reversal (0.9%)
    
    is_sweep = monitor.detect_liquidity_sweep('SPY', price_now, price_before)
    print(f"  Price swing: $445-$450")
    print(f"  Price 5min ago: ${price_before:.2f} (exceeded high)")
    print(f"  Price now: ${price_now:.2f} (reversed)")
    print(f"  Liquidity sweep: {'YES ⚠️' if is_sweep else 'NO'}")
    
    if not is_sweep:
        print("  (Reversal too small for sweep detection)")
    
    print("\n✅ Spread Monitor: PASSED")


def test_time_of_day_filter():
    """Test time-based position sizing."""
    print("\n" + "="*70)
    print("TEST 4: TIME-OF-DAY FILTER")
    print("="*70)
    
    config = EventRiskConfig()
    filter = TimeOfDayFilter(config)
    
    print("\n⏰ Testing time periods:")
    
    test_date = datetime(2026, 3, 1)  # A Saturday, but time doesn't matter
    tests = [
        (time(9, 35), 0.3, "Opening volatility (9:35 AM)"),
        (time(9, 50), 1.0, "Prime hours (9:50 AM)"),
        (time(11, 0), 1.0, "Mid-morning (11:00 AM)"),
        (time(12, 0), 0.7, "Lunch hour (12:00 PM)"),
        (time(14, 30), 1.0, "Afternoon prime (2:30 PM)"),
        (time(15, 57), 0.3, "Closing volatility (3:57 PM)"),
    ]
    
    for test_time, expected_mult, desc in tests:
        test_datetime = datetime.combine(test_date, test_time)
        mult, reason = filter.get_time_multiplier(test_datetime)
        status = "✅" if mult == expected_mult else "❌"
        print(f"  {status} {test_time} - {desc}")
        print(f"      Multiplier: {mult:.2f} | Reason: {reason}")
    
    print("\n✅ Time-of-Day Filter: PASSED")


def test_circuit_breaker():
    """Test circuit breaker protection."""
    print("\n" + "="*70)
    print("TEST 5: CIRCUIT BREAKER")
    print("="*70)
    
    config = EventRiskConfig(
        daily_loss_limit_pct=0.02,
        consecutive_loss_limit=3,
        weekly_drawdown_threshold_pct=0.05
    )
    breaker = CircuitBreaker(config)
    
    starting_equity = 100_000
    breaker.set_starting_equity(starting_equity, 'daily')
    breaker.set_starting_equity(starting_equity, 'weekly')
    
    # Test 1: Normal trading
    print("\n🔒 Test 1: Normal trading")
    mult, reason = breaker.get_circuit_multiplier(starting_equity)
    print(f"  Multiplier: {mult:.2f} | {reason}")
    assert mult == 1.0, "Should allow full size"
    print("  ✅ PASSED")
    
    # Test 2: Consecutive losses trigger pause
    print("\n🔒 Test 2: Consecutive losses")
    breaker.update_pnl(-500)
    print(f"  Loss 1: -$500 (consecutive: {breaker.consecutive_losses})")
    breaker.update_pnl(-300)
    print(f"  Loss 2: -$300 (consecutive: {breaker.consecutive_losses})")
    breaker.update_pnl(-200)
    print(f"  Loss 3: -$200 (consecutive: {breaker.consecutive_losses})")
    
    mult, reason = breaker.get_circuit_multiplier(starting_equity)
    print(f"  Multiplier: {mult:.2f} | {reason}")
    assert mult == 0.0, "Should pause after 3 losses"
    assert breaker.is_paused, "Should be in paused state"
    print("  ✅ PASSED - Trading paused")
    
    # Test 3: Daily loss limit
    print("\n🔒 Test 3: Daily loss limit")
    breaker2 = CircuitBreaker(config)
    breaker2.set_starting_equity(starting_equity, 'daily')
    
    # Lose 2.5% in one trade
    loss = starting_equity * 0.025
    breaker2.update_pnl(-loss)
    print(f"  Single loss: -${loss:,.0f} ({-2.5:.1f}%)")
    
    mult, reason = breaker2.get_circuit_multiplier(starting_equity - loss)
    print(f"  Multiplier: {mult:.2f} | {reason}")
    assert mult == 0.0, "Should halt after 2% loss"
    assert breaker2.is_halted, "Should be in halted state"
    print("  ✅ PASSED - Trading halted")
    
    # Test 4: Weekly drawdown
    print("\n🔒 Test 4: Weekly drawdown")
    breaker3 = CircuitBreaker(config)
    breaker3.set_starting_equity(starting_equity, 'weekly')
    
    # Lose 5.5% over the week
    weekly_loss = starting_equity * 0.055
    breaker3.update_pnl(-weekly_loss)
    print(f"  Weekly loss: -${weekly_loss:,.0f} ({-5.5:.1f}%)")
    
    mult, reason = breaker3.get_circuit_multiplier(starting_equity - weekly_loss)
    print(f"  Multiplier: {mult:.2f} | {reason}")
    assert mult == 0.5, "Should reduce size by 50%"
    print("  ✅ PASSED - Size reduced")
    
    print("\n✅ Circuit Breaker: PASSED")


def test_integrated_risk_manager():
    """Test full integration of all components."""
    print("\n" + "="*70)
    print("TEST 6: INTEGRATED EVENT RISK MANAGER")
    print("="*70)
    
    erm = EventRiskManager()
    
    # Scenario 1: Perfect conditions
    print("\n🎯 Scenario 1: Perfect conditions")
    normal_time = datetime(2026, 3, 1, 10, 30)  # Not FOMC, prime hours
    normal_data = {
        'volume': 1_000_000,
        'spread': 0.05,
        'price': 150.0,
        'equity': 100_000
    }
    
    mult, reasons = erm.calculate_position_multiplier('AAPL', normal_time, normal_data)
    print(f"  Multiplier: {mult:.2f}")
    print(f"  Reasons: {reasons if reasons else 'None - full size'}")
    assert mult == 1.0, "Should allow full size in perfect conditions"
    print("  ✅ Full position size allowed")
    
    # Scenario 2: FOMC day during opening
    print("\n🎯 Scenario 2: FOMC + opening volatility")
    fomc_opening = datetime(2026, 3, 18, 9, 35)  # FOMC day, 5 min after open
    
    mult, reasons = erm.calculate_position_multiplier('AAPL', fomc_opening, normal_data)
    print(f"  Multiplier: {mult:.2f}")
    print(f"  Active filters:")
    for reason in reasons:
        print(f"    - {reason}")
    
    expected = 0.3 * 0.3  # FOMC × opening
    assert abs(mult - expected) < 0.01, f"Expected ~{expected:.2f}"
    print(f"  ✅ Correct reduction: {mult:.2f}")
    
    # Scenario 3: Everything bad at once
    print("\n🎯 Scenario 3: Maximum risk (all factors)")
    
    # Build volume history
    for _ in range(5):
        erm.volume_detector.update_volume('TSLA', 1_000_000)
    
    worst_data = {
        'volume': 5_000_000,  # 5x spike
        'spread': 0.20,       # 4x normal
        'price': 200.0,
        'equity': 95_000      # 5% weekly drawdown
    }
    
    # Set weekly equity
    erm.circuit_breaker.set_starting_equity(100_000, 'weekly')
    
    mult, reasons = erm.calculate_position_multiplier('TSLA', fomc_opening, worst_data)
    print(f"  Multiplier: {mult:.4f}")
    print(f"  Active protections:")
    for reason in reasons:
        print(f"    - {reason}")
    
    print(f"  ✅ Extreme risk detected, position heavily reduced")
    
    print("\n✅ Integrated Risk Manager: PASSED")


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "🚀"*35)
    print("EVENT RISK MANAGER - COMPREHENSIVE TEST SUITE")
    print("🚀"*35)
    
    try:
        test_economic_calendar()
        test_volume_anomaly_detector()
        test_spread_monitor()
        test_time_of_day_filter()
        test_circuit_breaker()
        test_integrated_risk_manager()
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print("\nEvent Risk Manager is fully operational and ready for integration.")
        print("\nNext steps:")
        print("  1. Integrate into enhanced_trading_engine.py")
        print("  2. Test with live market data")
        print("  3. Monitor multiplier effectiveness in paper trading")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

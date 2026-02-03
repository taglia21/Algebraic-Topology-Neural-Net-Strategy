#!/bin/bash
################################################################################
# Offline Test Suite - Run While Market is Closed
# Tests all CRITICAL and HIGH-severity fixes without requiring live data
################################################################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASSED=0
FAILED=0

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   Trading System Offline Test Suite${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Set Python path
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

run_test() {
    local test_name="$1"
    local test_code="$2"
    
    echo -n "Testing: $test_name... "
    
    if python3 -c "$test_code" > /tmp/test_output.log 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        echo -e "${YELLOW}Error output:${NC}"
        cat /tmp/test_output.log
        ((FAILED++))
        return 1
    fi
}

echo -e "${YELLOW}[1/7] Module Import Tests${NC}"
echo "Testing all modules can be imported without errors..."
echo ""

run_test "risk_manager import" "
from src.risk_manager import RiskManager, RiskConfig, Position
print('✓ risk_manager imported')
"

run_test "position_sizer import" "
from src.position_sizer import PositionSizer, SizingConfig, PerformanceMetrics
print('✓ position_sizer imported')
"

run_test "multi_timeframe_analyzer import" "
from src.multi_timeframe_analyzer import MultiTimeframeAnalyzer, AnalyzerConfig
print('✓ multi_timeframe_analyzer imported')
"

run_test "sentiment_analyzer import" "
from src.sentiment_analyzer import SentimentAnalyzer, SentimentConfig
print('✓ sentiment_analyzer imported')
"

run_test "enhanced_trading_engine import" "
from src.enhanced_trading_engine import EnhancedTradingEngine, EngineConfig
print('✓ enhanced_trading_engine imported')
"

echo ""
echo -e "${YELLOW}[2/7] CRITICAL FIX #1: Division by Zero Protection${NC}"
echo "Testing Kelly Criterion calculation with edge cases..."
echo ""

run_test "Kelly with avg_win=0" "
from src.position_sizer import PositionSizer
sizer = PositionSizer()
result = sizer.calculate_kelly_fraction(win_rate=0.6, avg_win=0, avg_loss=50)
assert result == sizer.config.min_kelly_fraction, f'Expected {sizer.config.min_kelly_fraction}, got {result}'
print(f'✓ Handled avg_win=0: returned {result}')
"

run_test "Kelly with avg_loss=0" "
from src.position_sizer import PositionSizer
sizer = PositionSizer()
result = sizer.calculate_kelly_fraction(win_rate=0.6, avg_win=100, avg_loss=0)
assert result == sizer.config.min_kelly_fraction, f'Expected {sizer.config.min_kelly_fraction}, got {result}'
print(f'✓ Handled avg_loss=0: returned {result}')
"

run_test "Kelly with invalid win_rate" "
from src.position_sizer import PositionSizer
sizer = PositionSizer()
result = sizer.calculate_kelly_fraction(win_rate=1.5, avg_win=100, avg_loss=50)
assert result == sizer.config.min_kelly_fraction, f'Expected {sizer.config.min_kelly_fraction}, got {result}'
print(f'✓ Handled invalid win_rate: returned {result}')
"

run_test "Kelly with valid inputs" "
from src.position_sizer import PositionSizer
sizer = PositionSizer()
result = sizer.calculate_kelly_fraction(win_rate=0.55, avg_win=110, avg_loss=90)
assert result > 0 and result <= sizer.config.max_kelly_fraction, f'Kelly result out of bounds: {result}'
print(f'✓ Valid Kelly calculation: {result:.4f}')
"

echo ""
echo -e "${YELLOW}[3/7] CRITICAL FIX #2 & #3: Cache Management (Memory Leak Prevention)${NC}"
echo "Testing LRU cache with size limits..."
echo ""

run_test "MTF Analyzer cache limit" "
from src.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from datetime import datetime

analyzer = MultiTimeframeAnalyzer()

# Manually add cache entries beyond limit
for i in range(150):
    analyzer.cache[f'SYMBOL{i}'] = (None, datetime.now())

# Should have evicted to max_cache_size
assert len(analyzer.cache) <= analyzer.max_cache_size, \
    f'Cache exceeded limit: {len(analyzer.cache)} > {analyzer.max_cache_size}'
print(f'✓ Cache limited to {len(analyzer.cache)} entries (max: {analyzer.max_cache_size})')
"

run_test "Sentiment Analyzer cache limit" "
from src.sentiment_analyzer import SentimentAnalyzer
from datetime import datetime

analyzer = SentimentAnalyzer()

# Manually add cache entries beyond limit
for i in range(150):
    analyzer.cache[f'SYMBOL{i}'] = (None, datetime.now())

# Should have evicted to max_cache_size
assert len(analyzer.cache) <= analyzer.max_cache_size, \
    f'Cache exceeded limit: {len(analyzer.cache)} > {analyzer.max_cache_size}'
print(f'✓ Cache limited to {len(analyzer.cache)} entries (max: {analyzer.max_cache_size})')
"

echo ""
echo -e "${YELLOW}[4/7] CRITICAL FIX #4: NaN Validation${NC}"
echo "Testing ATR calculation returns valid numbers..."
echo ""

run_test "ATR returns valid number" "
import numpy as np
from src.enhanced_trading_engine import EnhancedTradingEngine

engine = EnhancedTradingEngine()

# Test with empty/invalid data should return 0.0, not NaN
# We can't test actual calculation without live data, but we can test the logic
import pandas as pd

# Mock a scenario that would produce NaN
class MockTicker:
    def history(self, *args, **kwargs):
        return pd.DataFrame()  # Empty dataframe

# Verify handling of empty data
result = engine._calculate_atr('INVALID_SYMBOL')
assert np.isfinite(result), f'ATR returned non-finite value: {result}'
assert result == 0.0, f'Empty data should return 0.0, got {result}'
print(f'✓ ATR handles invalid data: returns {result}')
"

echo ""
echo -e "${YELLOW}[5/7] HIGH-SEVERITY FIX: Volatility Array Validation${NC}"
echo "Testing volatility calculation with invalid data..."
echo ""

run_test "Volatility with NaN values" "
import numpy as np
from src.position_sizer import PositionSizer

sizer = PositionSizer()

# Array with NaN and inf values
bad_array = np.array([1.0, 2.0, np.nan, 3.0, np.inf, -1.0, 4.0])

result = sizer.calculate_volatility_scalar(current_volatility=2.5, historical_volatilities=bad_array)

# Should handle gracefully, not crash
assert np.isfinite(result), f'Volatility scalar returned non-finite: {result}'
assert 0.5 <= result <= 1.5, f'Volatility scalar out of bounds: {result}'
print(f'✓ Handled invalid volatility data: scalar={result:.2f}')
"

run_test "Volatility with all invalid values" "
import numpy as np
from src.position_sizer import PositionSizer

sizer = PositionSizer()

# Array with only invalid values
bad_array = np.array([np.nan, np.inf, -1.0, 0.0])

result = sizer.calculate_volatility_scalar(current_volatility=2.5, historical_volatilities=bad_array)

# Should return 1.0 (no scaling) when all data invalid
assert result == 1.0, f'Expected 1.0 for all-invalid data, got {result}'
print(f'✓ All-invalid data returns neutral scalar: {result}')
"

echo ""
echo -e "${YELLOW}[6/7] HIGH-SEVERITY FIX: Risk Manager Validation${NC}"
echo "Testing stop loss and take-profit calculations..."
echo ""

run_test "Stop loss validation (long)" "
from src.risk_manager import RiskManager

mgr = RiskManager()

# Test with normal values
stop = mgr.calculate_stop_loss(entry_price=100.0, atr=2.0, is_long=True)
assert stop < 100.0, f'Long stop must be below entry: {stop} vs 100.0'
assert stop > 0, f'Stop must be positive: {stop}'
print(f'✓ Valid long stop: entry=100.0, stop={stop:.2f}')
"

run_test "Stop loss validation (short)" "
from src.risk_manager import RiskManager

mgr = RiskManager()

# Test with normal values
stop = mgr.calculate_stop_loss(entry_price=100.0, atr=2.0, is_long=False)
assert stop > 100.0, f'Short stop must be above entry: {stop} vs 100.0'
print(f'✓ Valid short stop: entry=100.0, stop={stop:.2f}')
"

run_test "Take-profit validation" "
from src.risk_manager import RiskManager

mgr = RiskManager()

# Test with normal values
tps = mgr.calculate_take_profits(entry_price=100.0, stop_loss=95.0, is_long=True)

assert len(tps) > 0, 'No take-profits generated'
for i, tp in enumerate(tps):
    assert tp > 100.0, f'Long TP[{i}] must be above entry: {tp} vs 100.0'
print(f'✓ Valid take-profits: {[f\"{tp:.2f}\" for tp in tps]}')
"

run_test "Take-profit with zero risk" "
from src.risk_manager import RiskManager

mgr = RiskManager()

# Edge case: entry equals stop
tps = mgr.calculate_take_profits(entry_price=100.0, stop_loss=100.0, is_long=True)

assert len(tps) > 0, 'Should generate fallback TPs'
for tp in tps:
    assert tp > 100.0, f'TP must be above entry even with zero risk: {tp}'
print(f'✓ Zero-risk fallback TPs: {[f\"{tp:.2f}\" for tp in tps]}')
"

echo ""
echo -e "${YELLOW}[7/7] Integration Tests${NC}"
echo "Testing module integration..."
echo ""

run_test "Full system instantiation" "
from src.risk_manager import RiskManager
from src.position_sizer import PositionSizer, PerformanceMetrics
from src.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from src.sentiment_analyzer import SentimentAnalyzer
from src.enhanced_trading_engine import EnhancedTradingEngine

# Instantiate all modules
risk_mgr = RiskManager()
sizer = PositionSizer()
mtf = MultiTimeframeAnalyzer()
sentiment = SentimentAnalyzer()
engine = EnhancedTradingEngine()

print('✓ All modules instantiated')
print(f'  Risk Manager: max_positions={risk_mgr.config.max_concurrent_positions}')
print(f'  Position Sizer: mode={sizer.config.sizing_mode.value}')
print(f'  MTF Analyzer: timeframes={len(mtf.config.timeframes)}')
print(f'  Sentiment: cache_ttl={sentiment.config.cache_ttl_seconds}s')
print(f'  Engine: min_mtf_score={engine.config.min_mtf_score}')
"

run_test "Position sizing workflow" "
from src.position_sizer import PositionSizer, PerformanceMetrics

sizer = PositionSizer()

metrics = PerformanceMetrics(
    total_trades=100,
    winning_trades=55,
    losing_trades=45,
    total_profit=11000.0,
    total_loss=-9000.0
)

result = sizer.calculate_position_size(
    portfolio_value=100000.0,
    confidence=0.75,
    performance_metrics=metrics
)

assert result.is_valid, f'Position sizing failed: {result.rejection_reason}'
assert result.position_value > 0, f'Invalid position value: {result.position_value}'

print(f'✓ Position sizing: ${result.position_value:.2f} ({result.position_pct:.2%})')
print(f'  Kelly: {result.kelly_fraction:.3f}')
print(f'  Confidence adjusted: {result.confidence_adjusted:.3f}')
"

echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   Test Results Summary${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

TOTAL=$((PASSED + FAILED))
PASS_PCT=$((PASSED * 100 / TOTAL))

echo -e "Total Tests: $TOTAL"
echo -e "${GREEN}Passed: $PASSED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
else
    echo -e "${GREEN}Failed: $FAILED${NC}"
fi
echo -e "Success Rate: $PASS_PCT%"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    echo -e "${GREEN}System is ready for deployment${NC}"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo -e "${RED}Review errors above before deployment${NC}"
    exit 1
fi

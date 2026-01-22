# V6.0 Leveraged Dual Momentum Strategy - Production Report

## Executive Summary

| Decision | ❌ **NO-GO** |
|----------|--------------|
| Sharpe Ratio | 0.36 (Target: ≥1.8) |
| Ann Return | 6.0% (Target: ≥20%) |
| Max Drawdown | -26.5% (Target: ≤-20%) |
| p-value | 0.7164 (Target: <0.01) |

**V6.0 significantly underperforms all targets and previous versions. Do not deploy.**

---

## Strategy Overview

V6.0 implements a **Leveraged Dual Momentum** strategy using 3x ETFs:

### Asset Universe
| Asset | Description | Leverage |
|-------|-------------|----------|
| UPRO | 3x S&P 500 | 3.0x |
| TQQQ | 3x Nasdaq 100 | 3.0x |
| URTY | 3x Russell 2000 | 3.0x |
| TMF | 3x 20-Year Treasury | 3.0x |
| CASH | Money Market | 1.0x |

### Dual Momentum Framework
**Absolute Momentum (Trend Filter):**
- Accelerating average: 0.5×12m + 0.3×6m + 0.2×3m returns
- Risk-off when all equity assets have negative absolute momentum

**Relative Momentum (Ranking):**
- Select top-ranked equity asset by accelerating momentum
- Concentrated position (70%) in winner

**Kelly Position Sizing:**
- Rolling 6-month Sharpe → Kelly fraction
- Position range: 50-90% based on Kelly optimal

---

## Performance Results

### Full Period Metrics (266 trading days)

| Metric | V6.0 | V3.0 | UPRO B&H | SPY B&H | Target |
|--------|------|------|----------|---------|--------|
| **Sharpe** | 0.36 | 1.29 | 0.69 | 0.70 | ≥1.8 ❌ |
| **Ann Return** | 6.0% | 12.2% | 26.5% | 12.4% | ≥20% ❌ |
| **Total Return** | 6.3% | 12.2% | 28.0% | 13.1% | - |
| **Max DD** | -26.5% | -8.3% | -48.7% | - | ≤-20% ❌ |
| **Win Rate** | 53.6% | - | - | - | ≥55% ❌ |
| **Calmar** | 0.23 | 1.47 | - | - | ≥1.0 ❌ |
| **Sortino** | 0.50 | - | - | - | - |
| **p-value** | 0.716 | 0.05 | - | - | <0.01 ❌ |

### Risk Metrics
| Metric | Value |
|--------|-------|
| Alpha (vs SPY) | +10.0% annually |
| Beta (vs SPY) | 0.19 |
| Up Capture | 134% |
| Down Capture | 142% |

---

## Target Evaluation

| Target | Result | Status |
|--------|--------|--------|
| Sharpe ≥ 1.8 | 0.36 | ❌ FAIL |
| Sharpe ≥ 1.5 (conditional) | 0.36 | ❌ FAIL |
| Ann Return ≥ 20% | 6.0% | ❌ FAIL |
| Max DD ≤ -20% | -26.5% | ❌ FAIL |
| Win Rate ≥ 55% | 53.6% | ❌ FAIL |
| p-value < 0.01 | 0.716 | ❌ FAIL |
| Beats V3.0 Sharpe | 0.36 vs 1.29 | ❌ FAIL |
| Better DD than UPRO B&H | -26.5% vs -48.7% | ✅ PASS |

---

## Regime Performance

| Regime | Days | Sharpe | Return | SPY Excess |
|--------|------|--------|--------|------------|
| **BULL_LOW_VOL** | 148 | **1.64** | **+32.6%** | **+18.6%** |
| BEAR_NORMAL | 17 | 2.39 | +5.3% | +3.8% |
| BEAR_HIGH_VOL | 25 | 0.06 | -1.4% | -1.5% |
| BULL_NORMAL | 27 | -2.59 | -10.2% | -15.9% |
| UNDEFINED | 48 | -2.66 | -14.8% | -7.2% |

### Key Insight
V6.0 **excels in BULL_LOW_VOL** (Sharpe 1.64, +32.6%) but **fails catastrophically** in transition periods (UNDEFINED: -14.8%) and BULL_NORMAL (-10.2%).

---

## Critical Analysis

### What Worked ✅
1. **BULL_LOW_VOL regime**: Sharpe 1.64, +32.6% return, +18.6% excess over SPY
2. **TMF hedge in BEAR_HIGH_VOL**: Beat SPY by 1.5% during volatility
3. **BEAR_NORMAL performance**: Sharpe 2.39, +5.3% return
4. **DD control vs UPRO**: -26.5% vs -48.7% for buy-and-hold

### What Didn't Work ❌
1. **UNDEFINED regime (early period)**: -14.8% loss, Sharpe -2.66
2. **BULL_NORMAL regime**: -10.2% loss despite bull market
3. **Negative selection alpha**: TQQQ selection lost -45.5% vs B&H
4. **Monthly rebalance too slow**: Missed rallies after TMF switch
5. **TMF timing wrong**: Bonds fell when we held them

### Root Cause Analysis
| Issue | Impact | Fix |
|-------|--------|-----|
| 12-month lookback too slow | Missed recovery signals | Use 6-3-1 month |
| TMF hedge in 2025 | Bonds fell, lost -14.8% | Use cash instead |
| 70% equity cap | Missed TQQQ +76% run | Increase to 90% |
| Monthly rebalance | Missed intra-month rotations | Weekly rebalance |

---

## Asset Selection Analysis

| Asset | Days Selected | V6.0 Return | B&H Return | Selection Alpha |
|-------|--------------|-------------|------------|-----------------|
| TQQQ | 182 | +31.0% | +76.5% | **-45.5%** |
| UPRO | 52 | -12.6% | -4.3% | -8.3% |
| URTY | 6 | -7.4% | +3.1% | -10.5% |

**Problem**: Momentum timing added **negative alpha** to all selections.

---

## Worst Periods

| Metric | Value |
|--------|-------|
| Worst 3-month return | -26.4% (ending 2025-05-20) |
| Max consecutive losing months | 3 |
| Days with >5% loss | 4 (1.5%) |

---

## Strategy Comparison: All Versions

| Version | Strategy | Sharpe | Return | Max DD | Decision |
|---------|----------|--------|--------|--------|----------|
| **V3.0** | Unleveraged Momentum | **1.29** | 12.2% | -8.3% | ✅ Current Best |
| V4.0 | Pairs Trading | 0.37 | 0.6% | -0.8% | ❌ NO-GO |
| V5.0 | Regime Adaptive | 0.98 | 14.9% | -6.9% | ❌ NO-GO |
| **V6.0** | Leveraged Dual Momentum | 0.36 | 6.3% | -26.5% | ❌ NO-GO |

---

## Why V6.0 Failed

### 1. Leverage Amplified Losses, Not Just Gains
The absolute momentum filter correctly identified risk-off periods, but the TMF hedge (3x bonds) also lost money in early 2025.

### 2. Momentum Lookback Mismatch
12-month accelerating momentum was too slow for the fast-moving 2024-2025 market. By the time signals flipped, the best gains were already captured.

### 3. Monthly Rebalance Lag
Monthly rebalancing meant we stayed in TMF for weeks after equities recovered, missing critical rally days.

### 4. The 70% Cap Problem
Even when correctly positioned in TQQQ, the 70% cap (with 30% in TMF/Cash) meant we captured only ~50% of TQQQ's +76% gain during selection periods.

---

## Recommendations

### Primary Recommendation: Keep V3.0
V3.0 remains the best production strategy:
- Sharpe 1.29 (3.6x better than V6.0)
- Max DD -8.3% (much safer than V6.0's -26.5%)
- Simpler implementation, lower costs

### If Attempting V6.0 Refinement
1. **Shorten lookback**: Use 6-3-1 month weighted instead of 12-6-3
2. **Weekly rebalance**: Faster response to regime changes
3. **Remove TMF**: Use cash only for risk-off (bonds are unreliable hedge)
4. **Increase position**: 90% equity in risk-on periods
5. **Add stop-loss**: Exit if monthly return < -10%

### V6.0 as Satellite Position
Could consider 10-20% allocation to modified V6.0 as satellite position alongside V3.0 core.

---

## Files Generated

| File | Description |
|------|-------------|
| [leveraged_etf_prices.csv](leveraged_etf_prices.csv) | 518 days of 3x ETF price data |
| [leveraged_etf_returns.csv](leveraged_etf_returns.csv) | Daily returns with fee drag |
| [dual_momentum_signals.csv](dual_momentum_signals.csv) | 266 days of momentum signals |
| [v60_equity_curve.csv](v60_equity_curve.csv) | Daily portfolio equity |
| [v60_full_simulation.json](v60_full_simulation.json) | Complete simulation results |
| [regime_analysis.json](regime_analysis.json) | Regime breakdown |
| [V60_VALIDATION_REPORT.json](V60_VALIDATION_REPORT.json) | Validation results |

---

## Final Decision

| Criterion | V6.0 | Requirement | Result |
|-----------|------|-------------|--------|
| Sharpe ≥1.8, CAGR ≥20%, DD ≤-20%, p<0.01, beats V3.0 | All FAIL | FULL GO | ❌ |
| Sharpe 1.5-1.79, CAGR ≥18%, DD ≤-22%, p<0.05 | All FAIL | CONDITIONAL GO | ❌ |
| Any minimum threshold failed | Multiple | NO-GO | ✅ |

## ❌ NO-GO

**Do not deploy V6.0. Continue with V3.0 as primary production strategy.**

Leveraged momentum with dual momentum framework underperformed due to:
- Poor timing of risk-off periods
- Slow momentum lookback
- TMF hedge backfired
- Monthly rebalance too slow

---

*Report generated: 2025-01-22*
*Validation period: Dec 2024 - Jan 2026 (266 trading days post-warmup)*
*Universe: UPRO, TQQQ, URTY, TMF, CASH*

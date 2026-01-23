# V18.0 Diagnostic Report

**Generated:** 2026-01-22 18:45:07
**Analyzing:** V17.0 Performance (-12.8% CAGR)

---

## Executive Summary

### Top 3 Issues Identified

1. **BAD SIGNALS**: 95.7% of loss from signal quality (not costs)

2. **NEGATIVE IC FACTORS**: 29/50 factors have NEGATIVE IC (hurting performance)

3. **HMM OK**: Regime detection validated against VIX


---

## 1. Regime Attribution Analysis

**Question:** What % of loss comes from each regime?

| Regime | Name | Days | Total Return | % of Total P&L |
|--------|------|------|--------------|----------------|
| 0 | LowVolTrend | 176 | -0.2261 | 134.9% |
| 1 | HighVolTrend | 56 | 0.0614 | -36.7% |
| 2 | LowVolMeanRevert | 69 | 0.0120 | -7.2% |
| 3 | Crisis | 20 | -0.0149 | 8.9% |


---

## 2. Transaction Cost vs Signal Quality

**Question:** What % of loss comes from costs vs bad signals?


| Metric | Value |
|--------|-------|
| Total Return | -15.95% |
| Gross Return (before costs) | -15.26% |
| Transaction Costs | $6,851 |
| Cost Drag | 0.69% of capital |
| **% Loss from Costs** | 4.3% |
| **% Loss from Signals** | 95.7% |

**Conclusion:** Signals are the main problem


---

## 3. Factor Information Coefficient (IC) Analysis

**Question:** Which factors have NEGATIVE IC?

A negative IC means the factor is **inversely** correlated with forward returns - 
we're using it backwards or it's spurious.

**Summary:** 29 negative IC factors, 21 positive IC factors

### Factors with NEGATIVE IC (REMOVE OR FLIP)

| Factor | Mean IC | Std IC | T-Stat | Action |
|--------|---------|--------|--------|--------|
| distance_from_high | -0.1909 | 0.1136 | -16.81 | FLIP |
| price_vs_ma200 | -0.1652 | 0.1038 | -15.91 | FLIP |
| distance_from_low | -0.1408 | 0.1289 | -10.93 | FLIP |
| momentum_12m | -0.1278 | 0.1084 | -11.79 | FLIP |
| momentum_6m | -0.1216 | 0.0904 | -13.44 | FLIP |
| ma_cross_50_200 | -0.1151 | 0.1041 | -11.06 | FLIP |
| momentum_3m | -0.1050 | 0.0867 | -12.11 | FLIP |
| risk_adjusted_momentum | -0.1028 | 0.0893 | -11.51 | FLIP |
| momentum_consistency | -0.0929 | 0.0851 | -10.92 | FLIP |
| channel_position | -0.0860 | 0.0839 | -10.25 | FLIP |
| zscore_50d | -0.0849 | 0.0794 | -10.69 | FLIP |
| momentum_1m | -0.0801 | 0.0763 | -10.51 | FLIP |
| new_high_count | -0.0795 | 0.0941 | -8.44 | FLIP |
| breakout_20d | -0.0769 | 0.0785 | -9.80 | FLIP |
| relative_strength | -0.0744 | 0.0754 | -9.87 | FLIP |
| zscore_20d | -0.0742 | 0.0751 | -9.89 | FLIP |
| betti_0_estimate | -0.0544 | 0.0894 | -6.09 | FLIP |
| macd_histogram | -0.0543 | 0.0799 | -6.79 | FLIP |
| ma_cross_20_50 | -0.0533 | 0.0883 | -6.04 | FLIP |
| momentum_6_1 | -0.0532 | 0.0857 | -6.21 | FLIP |
| momentum_12_1 | -0.0445 | 0.1122 | -3.96 | FLIP |
| obv_momentum | -0.0410 | 0.0846 | -4.84 | FLIP |
| tda_complexity | -0.0360 | 0.0953 | -3.77 | FLIP |
| vwap_distance | -0.0306 | 0.0349 | -8.75 | FLIP |
| betti_1_estimate | -0.0220 | 0.0804 | -2.73 | FLIP |
| dollar_volume | -0.0198 | 0.0755 | -2.63 | REMOVE |
| persistence_entropy | -0.0152 | 0.1003 | -1.51 | REMOVE |
| price_gap | -0.0115 | 0.0432 | -2.67 | REMOVE |
| trend_strength_adx | -0.0008 | 0.0901 | -0.09 | REMOVE |

### Factors with POSITIVE IC (KEEP)

| Factor | Mean IC | Std IC | T-Stat | Significant |
|--------|---------|--------|--------|-------------|
| downside_vol | 0.0884 | 0.0996 | 8.87 |  |
| amihud_illiquidity | 0.0877 | 0.0758 | 11.56 |  |
| atr_ratio | 0.0794 | 0.0868 | 9.15 |  |
| overbought_oversold | 0.0707 | 0.0726 | 9.74 |  |
| volatility_60d | 0.0701 | 0.1033 | 6.79 |  |
| landscape_distance | 0.0700 | 0.0820 | 8.53 |  |
| reversal_5d | 0.0610 | 0.0651 | 9.37 |  |
| idio_vol | 0.0523 | 0.0875 | 5.97 |  |
| volatility_20d | 0.0494 | 0.0822 | 6.00 |  |
| volume_price_trend | 0.0398 | 0.0738 | 5.39 |  |
| volume_momentum | 0.0287 | 0.0743 | 3.86 |  |
| volatility_ratio | 0.0268 | 0.0946 | 2.83 |  |
| upside_vol | 0.0239 | 0.0985 | 2.42 |  |
| bollinger_width | 0.0236 | 0.0870 | 2.71 |  |
| momentum_acceleration | 0.0200 | 0.0903 | 2.21 |  |
| mean_reversion_speed | 0.0197 | 0.0946 | 2.08 |  |
| volume_volatility | 0.0185 | 0.1011 | 1.83 |  |
| wasserstein_distance | 0.0165 | 0.0831 | 1.99 |  |
| kurtosis_60d | 0.0106 | 0.0958 | 1.10 |  |
| relative_volume | 0.0092 | 0.0807 | 1.13 |  |

*...and 1 more positive IC factors*


---

## 4. HMM Regime Validation

**Question:** Is the HMM detecting regimes correctly?

We compare HMM regime assignments to VIX levels:
- **Crisis** should have HIGH VIX (>25)
- **LowVolTrend** should have LOW VIX (<15)

| Regime | Name | Days | Avg VIX | Median VIX | Max VIX |
|--------|------|------|---------|------------|--------|
| 0 | LowVolTrend | 298 | 15.4 | 15.1 | 22.6 |
| 1 | HighVolTrend | 86 | 17.6 | 16.5 | 25.1 |
| 2 | LowVolMeanRevert | 104 | 18.8 | 18.3 | 27.6 |
| 3 | Crisis | 28 | 30.3 | 29.1 | 52.3 |

### ✅ Validation Passed

HMM regime detection aligns with VIX behavior.


---

## Recommendations

Based on the diagnostic findings:

1. **Remove/Flip 29 negative IC factors** - These are hurting performance

2. **Focus on signal quality** - Costs are not the main issue

3. **Review LowVolTrend strategy** - This regime caused 135% of losses


---

## Next Steps

1. Run `v18_factor_calibration.py` to:
   - Remove factors with IC < -0.01
   - Flip factors with IC < -0.02 (strong negative signal is still signal)
   - Weight remaining factors by IC magnitude

2. Rerun `run_v17_full.py` with calibrated factors

3. Target: Move from -12.8% CAGR to **positive CAGR**

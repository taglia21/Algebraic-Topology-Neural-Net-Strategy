# V17.0 Trading System Report

**Generated:** 2026-01-22T18:28:59.795676

## System Configuration
- Initial Capital: $1,000,000
- Universe Size: 100 symbols
- Factors: 50 factors
- Current Regime: LowVolMeanRevert
- Active Strategy: v17_stat_arb

## Performance Summary

| Metric | Value | Target Range |
|--------|-------|--------------|
| Sharpe Ratio | -1.35 | 1.5 - 3.0 |
| CAGR | -12.8% | 25% - 50% |
| Max Drawdown | -21.1% | -15% to -25% |
| Annual Vol | 9.8% | 10% - 20% |
| Sortino Ratio | -1.69 | >2.0 |
| Calmar Ratio | -0.61 | >1.0 |

## Trading Statistics
- Total Return: -15.9%
- Trading Days: 320
- Total Trades: 2856
- Win Rate: 36.2%
- Annual Turnover: 275%

## Transaction Costs
- Total Commission: $3,411
- Total Slippage: $3,440
- Cost Drag: 0.69%

## Quality Assessment

✅ **REALISTIC**: Metrics within expected bounds

### Red Flags Checked:
- Sharpe > 5.0: ✅ OK
- CAGR > 100%: ✅ OK
- Max DD < -50%: ✅ OK

## Notes
- Walk-forward validation with 9-month train, 3-month test
- Transaction costs: 5bps commission + 5-20bps slippage
- Position limits: 4% max per position
- Vol target: 15% annual

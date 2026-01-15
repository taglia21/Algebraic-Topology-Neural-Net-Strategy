# Production Readiness Notes

## V1.3 Optimized Strategy - January 2026

### ✅ Backtest Validation Complete

| Metric | Target | Achieved |
|--------|--------|----------|
| Portfolio Sharpe (net) | > 1.3 | **1.35** ✅ |
| Max Drawdown | < 8% | **2.08%** ✅ |
| Assets with positive Sharpe | ≥ 2/5 | **2/5** ✅ |

---

## Pre-Deployment Checklist

### Phase 1: Paper Trading (3-6 months)

- [ ] Set up paper trading account (Interactive Brokers / Alpaca)
- [ ] Configure real-time data feed (Polygon OTREP recommended)
- [ ] Implement order execution module
- [ ] Log all signals, trades, and portfolio state
- [ ] Compare paper performance to backtest expectations

### Phase 2: Monitoring Setup

- [ ] Create dashboard for:
  - Daily P&L tracking
  - Per-asset allocation weights
  - Regime detection status
  - Signal distribution histogram
  - Drawdown meter

- [ ] Alert thresholds:
  - Drawdown > 3%: Yellow alert
  - Drawdown > 5%: Red alert, reduce position sizes
  - Drawdown > 8%: Circuit breaker, halt trading

### Phase 3: Error Handling

Implement robust error handling for:

```python
# Data quality checks
- Missing data handling
- Outlier detection (>5 std moves)
- Data staleness (>1hr without update)

# Execution errors
- Order rejection handling
- Partial fill management
- Connection retry logic

# Model errors
- NaN prediction handling
- Feature computation failures
- Memory management for TDA calculations
```

### Phase 4: Position Management

```python
# Current optimal parameters
KELLY_FRACTION = 0.50      # Half-Kelly
MAX_POSITION_PCT = 0.15    # 15% max per asset
RISK_SCALE = 0.5           # Conservative overlay

# Recommended for live trading
LIVE_KELLY_FRACTION = 0.25  # Quarter-Kelly initially
LIVE_MAX_POSITION = 0.10    # 10% max initially
```

---

## Known Risks & Mitigations

### 1. Model Degradation

**Risk**: Strategy performance may degrade over time as market conditions change.

**Mitigation**:
- Monitor rolling 30-day Sharpe ratio
- Retrain model quarterly with fresh data
- If 60-day Sharpe < 0.5, reduce position sizes by 50%

### 2. Regime Shifts

**Risk**: Major market regime changes may invalidate model assumptions.

**Mitigation**:
- Regime detector already filters UNFAVORABLE conditions
- Add macro indicators (VIX, yield curve) as secondary filters
- Paper trade through at least one market correction before live

### 3. Data Quality

**Risk**: Real-time data may differ from historical backtest data.

**Mitigation**:
- Use Polygon (same source as backtest) for live trading
- Implement data validation checks
- Log discrepancies for investigation

### 4. Execution Slippage

**Risk**: Actual slippage may exceed 5bp assumption.

**Mitigation**:
- Use limit orders (not market orders)
- Trade during high-liquidity hours (10am-3pm ET)
- Monitor actual vs expected execution costs
- If slippage > 10bp average, adjust COST_BP_PER_SIDE

---

## Asset-Specific Notes

### IWM (Russell 2000)
- **Current allocation**: 60.3%
- **Performance**: Sharpe 1.21 (best performer)
- **Liquidity**: Excellent (high volume ETF)
- **Notes**: Works well in current regime

### XLF (Financials)
- **Current allocation**: 39.7%
- **Performance**: Sharpe 0.78
- **Liquidity**: Excellent
- **Notes**: Sector-concentrated; monitor Fed policy sensitivity

### QQQ (Nasdaq 100)
- **Current allocation**: 0% (underperforming)
- **Performance**: Sharpe -1.59
- **Future**: May re-enable when regime becomes FAVORABLE
- **Notes**: High correlation with SPY (0.96) limits diversification

### SPY / XLK
- **Current allocation**: 0%
- **Notes**: Risk-weighting naturally reduced allocation

---

## Recommended Deployment Timeline

| Week | Activity |
|------|----------|
| 1-2 | Set up paper trading, data feeds |
| 3-4 | Run parallel: live signals vs paper execution |
| 5-8 | Full paper trading with monitoring |
| 9-12 | Validate paper results match expectations |
| 13+ | Consider small live allocation (10% of intended size) |

---

## Performance Monitoring KPIs

Track these metrics daily/weekly:

| Metric | Frequency | Target |
|--------|-----------|--------|
| Daily P&L | Daily | Log all trades |
| Rolling Sharpe (30d) | Weekly | > 0.8 |
| Max Drawdown | Daily | < 5% |
| Win Rate | Weekly | > 50% |
| Average Trade | Weekly | > 0.3% |
| Regime Distribution | Weekly | Monitor FAVORABLE % |

---

## Escalation Procedures

### Yellow Alert (Drawdown 3-5%)
1. Reduce new position sizes by 50%
2. Review recent trades for anomalies
3. Check if regime detector is functioning

### Red Alert (Drawdown 5-8%)
1. Stop new trades
2. Consider reducing existing positions
3. Investigate root cause
4. Manual review before resuming

### Circuit Breaker (Drawdown > 8%)
1. Close all positions
2. Halt automated trading
3. Full system review
4. Do not resume without manual approval

---

## Contact & Support

For strategy questions or issues:
- Review logs in `results/` directory
- Check `results/full_3year_analysis.txt` for baseline expectations
- Compare current signals to historical in `results/signal_diagnostics.csv`

---

*Last Updated: January 2026*

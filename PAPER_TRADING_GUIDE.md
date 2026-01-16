# Paper Trading Deployment Guide

## Phase 14: Production Paper Trading

This guide covers the deployment of the Phase 12 v3 All-Weather Regime-Switching Strategy to Alpaca paper trading.

---

## Quick Start

```bash
# Test connection
python scripts/deploy_paper_trading.py test

# Check status
python scripts/deploy_paper_trading.py status

# Single rebalance (when market is open)
python scripts/deploy_paper_trading.py rebalance

# Start continuous trading
python scripts/deploy_paper_trading.py start

# Monitor (separate terminal)
python scripts/monitor_paper_trading.py

# Emergency exit
python scripts/deploy_paper_trading.py exit
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   PAPER TRADING SYSTEM                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Market     │    │   Regime     │    │  Portfolio   │  │
│  │    Data      │ -> │  Detector    │ -> │ Constructor  │  │
│  │  (yfinance)  │    │  (SMA-based) │    │  (allocator) │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                    │         │
│                                                    v         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Circuit    │ <- │   Alpaca     │ <- │   Target     │  │
│  │   Breaker    │    │   Client     │    │  Positions   │  │
│  │  (controls)  │    │    (API)     │    │ (rebalance)  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration

All configuration is via `.env` file:

```bash
# Alpaca Credentials
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_ACCOUNT_ID=your_account_id
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Risk Management
PAPER_TRADING=true
MAX_DAILY_LOSS_PCT=0.03      # 3% daily loss limit
MAX_SINGLE_POSITION_PCT=0.08  # 8% max single position
MAX_LEVERAGED_ETF_PCT=0.25    # 25% max per leveraged ETF

# Strategy
REGIME_CONFIRMATION_DAYS=5    # Days to confirm regime change
STARTING_CAPITAL=99543.22     # For P&L tracking
REBALANCE_TIME=15:50          # Daily rebalance time (before close)

# Logging
LOG_FILE=logs/paper_trading.log
```

---

## Strategy: Phase 12 v3

### Regime Classification (SPY-based)

| Regime | Condition | Allocation |
|--------|-----------|------------|
| **BULL** | Price > SMA20 > SMA50 > SMA200 + positive momentum | Long 3x ETFs |
| **BEAR** | Price < SMA20 < SMA50 < SMA200 + negative momentum | Inverse 3x ETFs |
| **NEUTRAL** | Mixed signals | 100% Cash |

### ETF Allocations

**Bull Regime (Long 3x ETFs):**
- TQQQ: 50% (Nasdaq 3x)
- SPXL: 30% (S&P 500 3x)
- SOXL: 20% (Semiconductors 3x)

**Bear Regime (Inverse 3x ETFs):**
- SQQQ: 50% (Nasdaq -3x)
- SPXU: 30% (S&P 500 -3x)
- SOXS: 20% (Semiconductors -3x)

### Risk Adjustments

Position sizing is reduced based on:
- **Volatility**: >35% annualized → 50% reduction
- **Drawdown**: >10% → 50% reduction
- **Daily loss**: >3% → 50% reduction, >5% → full exit

---

## Production Controls

### Circuit Breakers

| Trigger | Action |
|---------|--------|
| Daily loss ≥ 3% | Reduce positions to 50% |
| Daily loss ≥ 5% | Exit all positions |
| Max DD ≥ 15% | Reduce to 30% allocation |
| Max DD ≥ 20% | Exit all positions |

### Position Limits

- Max single position: 8%
- Max leveraged ETF: 25%
- Base allocation: 70% (never 100% invested)

---

## Backtest Performance (Phase 12 v3)

| Metric | 3-Year Backtest | Monthly Target |
|--------|-----------------|----------------|
| CAGR | 64.7% | ~5-7% |
| Max Drawdown | 10.8% | <11% |
| Sharpe Ratio | 2.49 | >2.0 |
| Total Return | 289% | ~7%/month |

---

## 30-Day Validation Protocol

### Week 1: Observation
- [ ] Verify regime detection accuracy
- [ ] Confirm positions match targets
- [ ] Monitor daily P&L vs expectations
- [ ] Check no API errors

### Week 2: Small Scale
- [ ] Allow rebalancing with position limits
- [ ] Verify circuit breakers function
- [ ] Compare live vs backtest regime changes

### Week 3: Full Scale
- [ ] Enable full position sizing
- [ ] Test emergency exit procedure
- [ ] Verify logging completeness

### Week 4: Production Ready
- [ ] Cumulative return within 1σ of backtest
- [ ] Max DD below 11%
- [ ] No system errors
- [ ] Ready for live trading consideration

---

## Files

| File | Purpose |
|------|---------|
| `src/trading/alpaca_client.py` | Alpaca API wrapper |
| `src/trading/paper_trading_engine.py` | Core trading engine |
| `scripts/deploy_paper_trading.py` | Deployment CLI |
| `scripts/monitor_paper_trading.py` | Monitoring dashboard |
| `.env` | Configuration (not in git) |
| `logs/paper_trading.log` | Trading log |
| `logs/trading_state.json` | State persistence |

---

## Logging

All trades and events logged to `logs/paper_trading.log`:

```
2025-01-16 15:50:00 - INFO - Executing rebalance
2025-01-16 15:50:01 - INFO - Regime: BULL | Confidence: 82% | Days: 12
2025-01-16 15:50:02 - INFO - BUY TQQQ: $24,500.00
2025-01-16 15:50:03 - INFO - BUY SPXL: $14,700.00
2025-01-16 15:50:04 - INFO - BUY SOXL: $9,800.00
2025-01-16 15:50:05 - INFO - Rebalance complete: 3 trades
```

---

## Emergency Procedures

### Manual Emergency Exit
```bash
python scripts/deploy_paper_trading.py exit
# Type 'EXIT' to confirm
```

### Automatic Triggers
- 5% daily loss → Auto-exit
- 20% drawdown → Auto-exit
- API errors → Halt trading, alert

### Recovery
1. Identify cause
2. Fix issue
3. Run `test` to verify connection
4. Run `status` to confirm positions
5. Restart with `start`

---

## Next Steps After 30-Day Validation

1. **Review Performance**: Compare live vs backtest
2. **Adjust Parameters**: If needed based on live behavior
3. **Consider Live Trading**: After successful validation
4. **Scale Cautiously**: Start with fraction of intended capital

---

## Support

- Logs: `logs/paper_trading.log`
- State: `logs/trading_state.json`
- Monitor: `scripts/monitor_paper_trading.py`

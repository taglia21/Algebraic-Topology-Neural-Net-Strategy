# V23 Production Execution & Monitoring System
## Complete Architecture Documentation & Operations Runbook

**Version:** 23.0  
**Status:** Production Ready  
**Last Updated:** January 23, 2026

---

## Executive Summary

The V23 Production Execution & Monitoring System provides enterprise-grade infrastructure for deploying the validated V21 mean reversion strategy. This system transforms a backtested strategy into a production-ready trading operation with comprehensive risk controls, real-time monitoring, and paper trading validation.

### V21 Strategy Performance (Validated)
| Metric | Value |
|--------|-------|
| CAGR | 55.2% |
| Sharpe Ratio | 1.54 |
| Max Drawdown | -22.3% |
| Win Rate | 55.1% |
| Holding Period | 5 days |
| Position Count | 30 |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        V23 PRODUCTION TRADING SYSTEM                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐            │
│  │  V21 Strategy  │───>│  Position      │───>│   Execution    │            │
│  │   Signals      │    │    Sizer       │    │    Engine      │            │
│  │                │    │  (Kelly)       │    │  (Orders)      │            │
│  └────────────────┘    └────────────────┘    └───────┬────────┘            │
│                                                       │                     │
│                          ┌───────────────────────────┘                     │
│                          ▼                                                  │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐            │
│  │   Monitoring   │<───│   Circuit      │<───│   Broker       │            │
│  │   Dashboard    │    │   Breakers     │    │     API        │            │
│  │   (Alerts)     │    │  (Risk)        │    │  (Alpaca)      │            │
│  └────────────────┘    └────────────────┘    └────────────────┘            │
│                                                       │                     │
│                          ┌───────────────────────────┘                     │
│                          ▼                                                  │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐            │
│  │   Paper        │<───│   Kill         │    │    State       │            │
│  │  Validator     │    │   Switch       │    │  Persistence   │            │
│  │  (Go-Live)     │    │ (Emergency)    │    │   (Disk)       │            │
│  └────────────────┘    └────────────────┘    └────────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Overview

### 1. Execution Engine (`v23_execution_engine.py`)

**Purpose:** Production-grade order management with intelligent routing and slippage tracking.

**Key Features:**
- **Order Types:** Market, Limit, Market-with-Limit, TWAP
- **Entry Optimization:** Automatic order type selection based on spread and size
- **Slippage Tracking:** Expected vs actual fill price logging
- **Broker Integration:** Alpaca API with paper/live mode support

**Order Type Selection Logic:**
```python
if order_value > $10,000:
    order_type = TWAP  # Split into 3 tranches over 15 min
elif spread > 30bps:
    order_type = LIMIT  # Use midpoint limit order
else:
    order_type = MARKET  # Direct market order
```

**Classes:**
- `ExecutionEngine` - Core order management
- `ExecutionManager` - High-level rebalance interface
- `AlpacaAPI` - Broker API implementation
- `Order`, `Quote` - Data classes

---

### 2. Position Sizer (`v23_position_sizer.py`)

**Purpose:** Dynamic position sizing using fractional Kelly criterion with multi-factor adjustments.

**Key Features:**
- **Kelly Calculator:** Optimal position sizing from win rate and win/loss ratio
- **Drawdown Scaling:** Automatic size reduction during losses
- **Regime Adjustment:** VIX-based position scaling
- **Volatility Targeting:** Scale positions to target portfolio volatility

**Position Size Formula:**
```python
base_size = kelly_fraction * kelly_optimal  # Half-Kelly default
drawdown_mult = get_drawdown_multiplier()   # 0.25 - 1.0
regime_mult = get_regime_multiplier()       # 0.6 - 1.1
vol_mult = target_vol / realized_vol        # 0.5 - 2.0

final_size = base_size * drawdown_mult * regime_mult * vol_mult
final_size = min(final_size, max_position_pct)  # Hard cap at 10%
```

**Sizing Adjustments:**

| Condition | Multiplier | Effect |
|-----------|------------|--------|
| VIX < 18 (Bull) | 1.10x | +10% larger positions |
| VIX 18-30 (Neutral) | 1.00x | Normal sizing |
| VIX > 30 (Bear) | 0.60x | -40% smaller positions |
| Drawdown < -10% | 0.50x | -50% smaller positions |
| Drawdown < -15% | 0.00x | No new positions |
| Drawdown < -20% | Emergency | Close all positions |

---

### 3. Circuit Breakers (`v23_circuit_breakers.py`)

**Purpose:** Pre-trade validation and real-time risk controls with emergency kill switch.

**Key Features:**
- **Pre-Trade Validation:** 9 checks before any order
- **Circuit Breaker States:** NORMAL → WARNING → REDUCED → HALTED → EMERGENCY
- **Kill Switch:** Immediate position liquidation
- **Alert Integration:** Priority-based notifications

**Pre-Trade Validation Checks:**
1. Circuit breaker state
2. Position size limit (< 10% per position)
3. Daily loss limit (< -5%)
4. Daily trade count (< 50)
5. Consecutive losses (< 5)
6. Spread check (< 50bps)
7. Sector concentration (< 25%)
8. Market hours (9:35 - 15:55)
9. Error limits (< 3 execution, < 5 API)

**Circuit Breaker Thresholds:**

| Trigger | Action |
|---------|--------|
| Daily loss > -5% | HALT trading for day |
| Weekly loss > -10% | HALT trading, manual review |
| Drawdown > -10% | REDUCE position sizes 50% |
| Drawdown > -15% | HALT new entries |
| Drawdown > -20% | EMERGENCY: Close all positions |
| 5 consecutive losses | PAUSE new entries 24hr |
| 3 execution errors | HALT trading |

**Kill Switch:**
- Manual activation via `KillSwitch.activate(reason)`
- Immediately closes all positions at market
- Sends critical alerts (Email + SMS)
- Requires confirmation code to deactivate

---

### 4. Monitoring Dashboard (`v23_monitoring_dashboard.py`)

**Purpose:** Real-time metrics tracking, performance visualization, and alert system.

**Key Features:**
- **Real-Time Metrics:** P&L, positions, drawdown, exposure
- **Performance Tracking:** Daily, 7-day, 30-day summaries
- **Alert System:** Multi-channel with priority routing
- **System Health:** Component health checks, uptime monitoring

**Alert Priority Levels:**

| Priority | Channels | Examples |
|----------|----------|----------|
| CRITICAL | SMS + Email + Log | Kill switch, API down, daily loss limit |
| HIGH | Email + Push + Log | Drawdown > 10%, position limit warning |
| MEDIUM | Email + Log | Trade executed, daily summary |
| LOW | Log only | Heartbeat, routine checks |

**Dashboard Metrics:**
```python
DASHBOARD_METRICS = {
    # Performance
    'daily_pnl': float,
    'daily_pnl_pct': float,
    'mtd_pnl': float,
    'ytd_pnl': float,
    'current_drawdown': float,
    
    # Portfolio State
    'open_positions': int,
    'total_exposure': float,
    'cash_available': float,
    'buying_power': float,
    
    # Risk Metrics
    'portfolio_beta': float,
    'sector_concentration': dict,
    'largest_position_pct': float,
    
    # Execution Quality
    'avg_slippage_bps': float,
    'fill_rate_pct': float,
    'pending_orders': int,
    
    # System Health
    'api_latency_ms': float,
    'last_heartbeat': datetime,
    'error_count_today': int
}
```

---

### 5. Paper Trading Validator (`v23_paper_validator.py`)

**Purpose:** Validates paper trading performance against backtest expectations before go-live.

**Key Features:**
- **Paper vs Backtest Comparison:** Win rate, Sharpe, slippage alignment
- **Go-Live Checklist:** Automated readiness assessment
- **Signal Alignment Tracking:** Compare generated signals to backtest
- **Execution Quality Analysis:** Fill rate, slippage distribution

**Go-Live Requirements:**

| Check | Requirement | Notes |
|-------|-------------|-------|
| Paper trading period | >= 14 days | Minimum validation period |
| Trade count | >= 20 trades | Statistical significance |
| Fill rate | >= 95% | Execution reliability |
| Slippage | <= 15bps avg | Within assumptions |
| Win rate | Within ±5% of backtest | Performance alignment |
| Sharpe ratio | >= 70% of backtest | Performance alignment |
| Critical errors | 0 | System reliability |
| Circuit breakers tested | Verified | Manual confirmation |
| Kill switch tested | Verified | Manual confirmation |
| Alerts working | Verified | All channels tested |

---

## Configuration Guide

### Environment Variables

Create a `.env` file with the following:

```bash
# Alpaca API Credentials
ALPACA_API_KEY=your_api_key
ALPACA_API_SECRET=your_api_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper trading

# Alert Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
ALERT_EMAIL_SENDER=your_email@gmail.com
ALERT_EMAIL_PASSWORD=your_app_password
ALERT_EMAIL_RECIPIENT=alerts@yourdomain.com

# Optional: Twilio for SMS
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_PHONE_FROM=+1234567890
TWILIO_PHONE_TO=+0987654321
```

### Strategy Parameters

```python
# Position Sizing
KELLY_FRACTION = 0.5        # Half-Kelly for safety
MAX_POSITION_PCT = 0.10     # 10% max per position
MAX_POSITIONS = 30          # Maximum positions
TARGET_VOLATILITY = 0.25    # 25% annual volatility target

# Risk Controls
DAILY_LOSS_LIMIT_PCT = -5.0
WEEKLY_LOSS_LIMIT_PCT = -10.0
DRAWDOWN_REDUCE_PCT = -10.0
DRAWDOWN_HALT_PCT = -15.0
DRAWDOWN_EMERGENCY_PCT = -20.0
MAX_CONSECUTIVE_LOSSES = 5

# Execution
SPREAD_THRESHOLD_BPS = 30   # Use limit order above this
LARGE_ORDER_THRESHOLD = 10000  # Use TWAP above this ($)
TWAP_TRANCHES = 3
TWAP_INTERVAL_SECONDS = 300

# Monitoring
REBALANCE_TIME = "15:30"    # Daily rebalance
HEARTBEAT_INTERVAL = 60     # Seconds
ALERT_RATE_LIMIT = 60       # Seconds between same alerts
```

---

## Operations Runbook

### Daily Operations

**Market Open Checklist:**
```bash
# 1. Check system health
python -c "from v23_monitoring_dashboard import MonitoringDashboard; d = MonitoringDashboard(); print(d.check_health())"

# 2. Verify API connection
python -c "from v23_execution_engine import AlpacaAPI; a = AlpacaAPI(); print(a.is_market_open())"

# 3. Check circuit breaker state
python -c "from v23_circuit_breakers import CircuitBreakerManager; m = CircuitBreakerManager(); print(m.get_status())"

# 4. Load previous state
python -c "from v23_execution_engine import ExecutionManager; m = ExecutionManager(); m.engine.load_state()"
```

**Market Close Checklist:**
```bash
# 1. Save all state
python -c "
from v23_execution_engine import ExecutionManager
from v23_position_sizer import PositionSizer
from v23_circuit_breakers import CircuitBreakerManager
from v23_monitoring_dashboard import MonitoringDashboard

m = ExecutionManager()
m.engine.save_state()
PositionSizer().save_state()
CircuitBreakerManager().save_state()
MonitoringDashboard().save_state()
print('State saved')
"

# 2. Send daily summary
python -c "from v23_monitoring_dashboard import MonitoringDashboard; d = MonitoringDashboard(); d.send_daily_summary()"

# 3. Review slippage stats
python -c "from v23_execution_engine import ExecutionManager; m = ExecutionManager(); print(m.engine.get_slippage_stats())"
```

### Emergency Procedures

**Activate Kill Switch:**
```python
from v23_circuit_breakers import KillSwitch
from v23_execution_engine import ExecutionManager

manager = ExecutionManager()
kill = KillSwitch(execution_engine=manager.engine)

# EMERGENCY ACTIVATION
kill.activate(reason="Manual activation - describe situation")
```

**Deactivate Kill Switch:**
```python
from datetime import datetime
from v23_circuit_breakers import KillSwitch

kill = KillSwitch()
confirmation_code = f"CONFIRM-{datetime.now().strftime('%Y%m%d')}"
kill.deactivate(confirmation_code)
```

**Cancel All Orders:**
```python
from v23_execution_engine import ExecutionManager

manager = ExecutionManager()
cancelled = manager.engine.cancel_all_pending()
print(f"Cancelled {cancelled} orders")
```

### Troubleshooting

| Issue | Diagnosis | Resolution |
|-------|-----------|------------|
| API connection failed | Check `ALPACA_API_KEY` env var | Verify credentials, check Alpaca status |
| High slippage | Review `get_slippage_stats()` | Adjust spread thresholds, use more limit orders |
| Orders not filling | Check spread, liquidity | Increase limit order aggressiveness |
| Circuit breaker stuck | Check `get_status()` | Reset daily counters, investigate trigger |
| Missing alerts | Check email config | Verify SMTP credentials, test manually |

---

## Paper Trading Validation Process

### Phase 1: Initial Deployment (Days 1-3)

1. Deploy system in paper mode
2. Verify all components start correctly
3. Confirm signal generation matches backtest
4. Check order submission and fills
5. Monitor slippage vs assumptions

### Phase 2: Burn-In Period (Days 4-10)

1. Run strategy through full rebalance cycles
2. Track fill rate and execution quality
3. Test circuit breaker triggers (manually)
4. Verify alert delivery on all channels
5. Test kill switch activation/deactivation

### Phase 3: Validation (Days 11-14+)

1. Run full validation checks:
```python
from v23_paper_validator import PaperTradingValidator

validator = PaperTradingValidator()
validator.load_state()

passed, results = validator.run_validation()
checklist = validator.generate_go_live_checklist()
print(checklist['recommendation'])
```

2. Generate comparison report
3. Review go-live checklist
4. Address any failed checks

### Go-Live Decision Matrix

| Scenario | Recommendation |
|----------|----------------|
| All checks pass | ✅ Go live with 50% position sizing |
| Minor warnings only | ⚠️ Go live with 25% position sizing |
| Performance misaligned | ❌ Investigate strategy degradation |
| Critical errors present | ❌ Fix issues, restart validation |
| Kill switch not tested | ❌ Complete testing before go-live |

---

## File Structure

```
/workspaces/Algebraic-Topology-Neural-Net-Strategy/
├── v23_execution_engine.py      # Order management & broker API
├── v23_position_sizer.py        # Kelly-based position sizing
├── v23_circuit_breakers.py      # Risk controls & kill switch
├── v23_monitoring_dashboard.py  # Metrics & alerts
├── v23_paper_validator.py       # Paper trading validation
├── state/
│   ├── execution/
│   │   └── execution_state.json
│   ├── sizing/
│   │   └── sizing_state.json
│   ├── circuit_breakers/
│   │   ├── circuit_breaker_state.json
│   │   └── kill_switch_state.json
│   ├── monitoring/
│   │   ├── dashboard_state.json
│   │   ├── metrics_state.json
│   │   └── alert_history.json
│   └── paper_validation/
│       ├── validator_state.json
│       └── validation_report.json
└── results/
    └── v23/
        └── V23_PRODUCTION_REPORT.md
```

---

## Performance Targets

### Production KPIs

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Fill Rate | > 98% | < 95% | < 90% |
| Avg Slippage | < 10bps | > 15bps | > 25bps |
| API Uptime | > 99.9% | < 99.5% | < 99% |
| Error Rate | < 1/week | > 3/week | > 1/day |
| Sharpe (30d) | > 1.3 | < 1.0 | < 0.5 |

### System Health Thresholds

| Component | Check | Frequency |
|-----------|-------|-----------|
| API Connection | Heartbeat | 60 seconds |
| Order Status | Poll pending | 30 seconds |
| Market Data | Quote freshness | 5 seconds |
| Positions | Reconciliation | 5 minutes |
| State Persistence | Auto-save | 15 minutes |

---

## Appendix: Component Test Commands

```bash
# Test Execution Engine
python v23_execution_engine.py

# Test Position Sizer
python v23_position_sizer.py

# Test Circuit Breakers
python v23_circuit_breakers.py

# Test Monitoring Dashboard
python v23_monitoring_dashboard.py

# Test Paper Validator
python v23_paper_validator.py
```

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 23.0 | 2026-01-23 | Initial production release |
| 22.0 | 2026-01-22 | Walk-forward validation complete |
| 21.0 | 2026-01-20 | Strategy optimization finalized |

---

*Document maintained by V23 Trading System*  
*For questions, contact: trading-ops@example.com*

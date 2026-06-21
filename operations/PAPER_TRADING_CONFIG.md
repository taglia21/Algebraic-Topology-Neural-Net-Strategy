# Paper Trading Configuration Guide

This guide explains all configuration options needed to start paper trading safely.

## Quick Start

```bash
# 1. Copy credentials template
cp .env.example .env

# 2. Edit .env with your Alpaca paper keys
#    ALPACA_API_KEY=...
#    ALPACA_API_SECRET=...

# 3. Verify credentials work
python -c "from equities.alpaca_broker import AlpacaBroker; \
  b = AlpacaBroker(); \
  print(f'Account equity: ${b.get_account().equity:,.2f}')"

# 4. Start paper trading
python main.py --mode paper
```

---

## Environment Variables (.env)

### Alpaca Credentials

```env
# Paper trading API key (from https://app.alpaca.markets/account/info)
ALPACA_API_KEY=PK_XXXXXXXXXXXXXXXX

# Paper trading API secret
ALPACA_API_SECRET=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Paper trading endpoint (do NOT use live URL)
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Mode: paper, backtest, or live
SYSTEM_MODE=paper
```

**Where to get credentials:**
1. Go to https://app.alpaca.markets
2. Click "Account" → "API Keys"
3. Use the "Paper Trading" key set (not live)
4. Copy both API key and secret

**⚠️ CRITICAL:** Use `https://paper-api.alpaca.markets` NOT the live endpoint (`https://api.alpaca.markets`)

### Portfolio Configuration

```env
# Initial capital for paper trading (USD)
INITIAL_CAPITAL=100000

# Target capital (same as initial for paper)
TARGET_CAPITAL=100000

# Maximum position size as % of equity
MAX_POSITION_PCT=20

# Maximum sector concentration as % of equity
MAX_SECTOR_PCT=35

# Maximum drawdown before trading halts
MAX_DRAWDOWN_PCT=30

# Daily loss limit (% of equity)
DAILY_LOSS_LIMIT_PCT=3
```

### Strategy Configuration

```env
# Enable ML meta-learner (optional, slower)
USE_ML=false

# Regime detection (HMM for market regimes)
REGIME_DETECTION=true

# Live trade logging (to file)
LIVE_LOG_DIR=./operations
```

### Data Provider

```env
# Primary data source (yfinance or alpaca)
DATA_PROVIDER=yfinance

# Alternative data provider for fallback
DATA_FALLBACK=alpaca
```

---

## Configuration File (core/config.py)

The main configuration is defined in `core/config.py`. Key sections for paper trading:

### SignalConfig (Strategy Parameters)

```python
# From core/config.py

@dataclass
class SignalConfig:
    """Strategy signal generation parameters."""
    
    # Statistical arbitrage (pairs trading)
    stat_arb_enabled: bool = True
    stat_arb_min_correlation: float = 0.70
    stat_arb_max_zscore: float = 3.5
    stat_arb_stop_zscore: float = 2.0
    
    # Momentum
    momentum_enabled: bool = True
    momentum_lookback: int = 252
    momentum_skip: int = 21
    
    # Factor model (quality, value, low-vol, momentum)
    factor_model_enabled: bool = True
    factor_model_min_market_cap: float = 1e9  # $1B minimum
    
    # Minimum universe size
    universe_size: int = 15
```

Modify these for paper trading if you want to:
- Focus on fewer strategies (set others to False)
- Tighten entry signals (increase thresholds)
- Test specific regime conditions

### ExecutionConfig (Order Settings)

```python
@dataclass
class ExecutionConfig:
    """Execution and broker settings."""
    
    slippage_bps: float = 7.0  # Basis points for market orders
    commission_per_share: float = 0.005  # Alpaca typical
    order_type: str = "market"  # "market" or "limit"
    max_position_value_usd: float = 5000.0  # Per-name cap
```

**Important:** The `slippage_bps` here is the *modeled* slippage. Actual realized slippage will be tracked and compared.

### RiskConfig (Risk Management)

```python
@dataclass
class RiskConfig:
    """Risk management thresholds."""
    
    # Position sizing
    max_position_pct: float = 20.0  # % of equity per name
    max_sector_pct: float = 35.0    # % of equity per sector
    max_correlation: float = 0.85   # Position correlation cap
    
    # Drawdown gates
    max_drawdown_pct: float = 30.0  # Hard halt
    max_drawdown_reduce_pct: float = 20.0  # Reduce to 60%
    
    # Daily loss
    daily_loss_limit_pct: float = 3.0
    
    # Gross exposure cap
    max_gross_exposure_pct: float = 150.0
    max_net_exposure_pct: float = 100.0
```

**For paper trading:** Keep these conservative (as shown). Don't relax them.

### MLConfig (Machine Learning)

```python
@dataclass
class MLConfig:
    """ML overlay configuration."""
    
    use_ml: bool = False  # Disable for paper baseline
    ood_action: str = "neutral"  # "auto", "skip", "neutral", "block"
    ood_quantile_threshold: float = 0.95
    ood_outlier_threshold: float = 3.0
    retrain_frequency_days: int = 5
```

For paper trading baseline, keep `use_ml=False`. Add ML overlay later after validating base strategies.

---

## Command-Line Options

```bash
# Paper trading (reads SYSTEM_MODE=paper from .env)
python main.py --mode paper

# Override mode without editing .env
python main.py --mode paper --start 2026-06-01 --end 2026-06-30

# Run with specific symbols
python main.py --mode paper --symbols AAPL,MSFT,GOOG

# Enable debug logging
python main.py --mode paper --log-level DEBUG

# Dry-run (no actual orders)
python main.py --mode paper --dry-run
```

---

## Pre-Trading Checklist

### 1. Verify .env is Correct

```bash
cat .env | grep ALPACA
# Output should show:
# ALPACA_API_KEY=PK_...
# ALPACA_API_SECRET=...
# ALPACA_BASE_URL=https://paper-api.alpaca.markets
# SYSTEM_MODE=paper
```

### 2. Test Broker Connection

```bash
python -c "
from equities.alpaca_broker import AlpacaBroker

try:
    broker = AlpacaBroker()
    account = broker.get_account()
    print(f'✓ Connected to Alpaca')
    print(f'  Account ID: {account.account_id}')
    print(f'  Equity: \${account.equity:,.2f}')
    print(f'  Buying Power: \${account.buying_power:,.2f}')
except Exception as e:
    print(f'✗ Connection failed: {e}')
"
```

### 3. Verify Data Access

```bash
python -c "
from data.data_manager import DataManager

try:
    dm = DataManager()
    prices = dm.get_bars('AAPL', '2026-06-01', '2026-06-21')
    print(f'✓ Data access working')
    print(f'  AAPL bars: {len(prices)} days')
except Exception as e:
    print(f'✗ Data access failed: {e}')
"
```

### 4. Run Core Tests

```bash
# Quick sanity tests
python -m pytest tests/test_core_modules.py -v -k "not live"

# Should pass with minimal warnings
```

### 5. Run Backtest Smoke Test

```bash
# Quick backtest to verify logic
python run_backtest.py --small

# Should complete without errors
```

---

## Common Issues & Fixes

### "Cannot access repository. Check permissions."

**Problem:** Branch protection script fails.

**Solution:** This is expected in dev environment. See [BRANCH_PROTECTION_SETUP.md](../BRANCH_PROTECTION_SETUP.md) for manual GitHub UI method.

### "401 Unauthorized - Invalid credentials"

**Problem:** Alpaca API key or secret is wrong.

**Solution:**
1. Verify `.env` has correct keys
2. Check keys are from **Paper Trading** set (not Live)
3. Regenerate keys in Alpaca web UI if needed
4. Test: `python -c "from equities.alpaca_broker import AlpacaBroker; AlpacaBroker()"`

### "No bars for symbol AAPL"

**Problem:** Data provider failed or symbol delisted.

**Solution:**
1. Check if symbol is still tradeable (verify on Alpaca)
2. Check data provider is online (yfinance or Alpaca)
3. Try fallback provider: edit `DATA_PROVIDER` in `.env`
4. Check date range is valid (not weekends/holidays)

### "Daily loss limit hit - halting trading"

**Problem:** System hit `-3%` daily loss limit.

**Solution:**
1. This is **normal** for some days
2. System resumes next trading day
3. Not a gate failure unless excessive
4. Review strategy or adjust position sizing if persistent

### "Position reconciliation mismatch"

**Problem:** System position != Alpaca position.

**Solution:**
1. Wait; system auto-reconciles each cycle
2. Check order fill times in Alpaca web UI
3. Verify no stuck/cancelled orders
4. Review logs: `grep "Discrepancy" operations/paper_trading_*.log`
5. Manual adjustment if persistent (document for gate evidence)

---

## Graceful Startup & Shutdown

### Start (Long-running)

Use `screen` or `tmux` for detached session:

```bash
# Start in background session
screen -S paper_trading -dm bash -c 'cd /workspaces/Algebraic-Topology-Neural-Net-Strategy && python main.py --mode paper 2>&1 | tee -a operations/paper_trading_$(date +%Y%m%d).log'

# Check status
screen -list

# View logs
tail -f operations/paper_trading_$(date +%Y%m%d).log
```

### Stop (Graceful)

```bash
# Reattach and exit
screen -r paper_trading
# Type: Ctrl+C

# Or kill from outside
screen -S paper_trading -X send-keys C-c
screen -S paper_trading -X kill
```

### Restart After Fix

```bash
# Apply fix, run tests
python -m pytest tests/test_core_modules.py -v

# Run smoke backtest
python run_backtest.py --small

# Restart paper trading
screen -S paper_trading -dm bash -c 'cd /workspaces/Algebraic-Topology-Neural-Net-Strategy && python main.py --mode paper 2>&1 | tee -a operations/paper_trading_$(date +%Y%m%d).log'
```

---

## Next Steps

1. **Set up credentials** in `.env`
2. **Test connection** to Alpaca
3. **Review strategy config** in `core/config.py`
4. **Run tests** to verify setup
5. **Start paper trading** for 20+ days
6. **Collect telemetry** and validate gate
7. **Open PR** with gate evidence for review

See: [PAPER_TRADING_CHECKLIST.md](PAPER_TRADING_CHECKLIST.md) for day-by-day plan.

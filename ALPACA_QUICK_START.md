# 🚀 ALPACA OPTIONS ENGINE - QUICK START

**Status**: ✅ Ready for Trading  
**Migration**: Tradier → Alpaca COMPLETE  
**Risk**: Protected with 25% stop-loss (was 100% - INSANE!)

---

## ⚡ Quick Start (3 minutes)

### 1. Setup Environment
```bash
# Copy template
cp .env.template .env

# Edit with your Alpaca credentials
nano .env
```

Add these to `.env`:
```bash
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_PAPER=true
```

Get API keys: https://alpaca.markets/

### 2. Test Migration
```bash
python test_alpaca_migration.py
```

Expected: All tests pass ✅

### 3. Run Demo
```bash
python demo_alpaca_engine.py
```

Shows account info, options chains, positions.

### 4. Start Monitoring
```bash
python alpaca_options_monitor.py
```

Monitors positions 24/7 with automatic:
- 25% stop-loss
- 50% profit-taking

Press Ctrl+C to stop.

---

## 📚 Key Files

| File | Purpose | Size |
|------|---------|------|
| `src/alpaca_options_engine.py` | Core engine | 750 lines |
| `alpaca_options_monitor.py` | Position monitor | 240 lines |
| `test_alpaca_migration.py` | Test suite | 400 lines |
| `demo_alpaca_engine.py` | Demo script | 350 lines |
| `.env.template` | Config template | Reference |
| `TRADIER_ALPACA_MIGRATION.md` | Full migration docs | Reference |

---

## 🎯 Core Features

### Risk Management ✅
- **25% Stop-Loss**: Auto-exit at 25% loss (not 100%!)
- **50% Profit Target**: Auto-exit at 50% gain
- **Position Limits**: Max 5 contracts per position
- **Portfolio Risk**: Max 2% per trade

### Monitoring ✅
- **Real-Time**: Checks every 60 seconds
- **Automatic**: No manual intervention needed
- **Logging**: All activity tracked
- **Alerts**: Console and file logs

### Execution ✅
- **Paper Trading**: Safe testing environment
- **Market Orders**: Fast execution
- **Limit Orders**: Price protection
- **Multi-Leg**: Spreads supported

---

## 📊 Example Usage

### Initialize Engine
```python
from src.alpaca_options_engine import AlpacaOptionsEngine

engine = AlpacaOptionsEngine(paper=True)
```

### Get Account Info
```python
account = engine.get_account()
print(f"Equity: ${account['equity']:,.2f}")
```

### Get Options Chain
```python
contracts = engine.get_options_chain(
    'SPY',
    expiration_date='2026-02-14',
    strike_range=(400, 500)
)
```

### Place Order
```python
order = engine.place_option_order(
    symbol='SPY250214P00450000',
    quantity=1,
    side='sell',
    order_type='limit',
    limit_price=2.50
)
```

### Monitor Positions
```python
results = engine.monitor_positions()
# Automatically triggers stops/targets
```

---

## 🛡️ Safety Features

### Before Tradier (DANGEROUS)
- ❌ 100% stop-loss = lose everything
- ❌ Manual monitoring = miss stops
- ❌ Platform failures = $8k lost

### After Alpaca (SAFE)
- ✅ 25% stop-loss = capital protected
- ✅ Automated monitoring = never miss stops
- ✅ Reliable platform = confidence

---

## 📈 Risk Comparison

### Old System (Tradier)
```
Premium: $1,000
Stop-Loss: 100%
Max Loss: $1,000 (100%)
```

### New System (Alpaca)
```
Premium: $1,000
Stop-Loss: 25%
Max Loss: $250 (25%)
Capital Saved: $750 (75%)
```

**4x better capital protection!**

---

## ✅ Pre-Flight Checklist

Before trading:

- [ ] `.env` file created with Alpaca credentials
- [ ] Test script passes: `python test_alpaca_migration.py`
- [ ] Demo runs: `python demo_alpaca_engine.py`
- [ ] Monitor daemon starts: `python alpaca_options_monitor.py`
- [ ] ALPACA_PAPER=true confirmed
- [ ] Stop-loss is 25% (NOT 100%)
- [ ] Account has sufficient buying power

---

## 🚨 Emergency Procedures

### If Stop-Loss Not Triggering
```bash
# Check monitor is running
ps aux | grep alpaca_options_monitor

# Check logs
tail -f logs/alpaca_monitor_*.log

# Manually close position
python -c "from src.alpaca_options_engine import AlpacaOptionsEngine; \
engine = AlpacaOptionsEngine(paper=True); \
engine.close_position('YOUR_SYMBOL_HERE', reason='manual')"
```

### If Order Fails
1. Check account buying power
2. Check market hours (9:30am - 4pm ET)
3. Check Alpaca status: https://status.alpaca.markets/
4. Review logs for error details

### If Can't Connect
1. Verify API keys in `.env`
2. Check internet connection
3. Confirm paper trading: `ALPACA_PAPER=true`
4. Test credentials: `python test_alpaca_migration.py`

---

## 📞 Support Resources

### Alpaca
- **Docs**: https://alpaca.markets/docs/
- **Status**: https://status.alpaca.markets/
- **Support**: https://alpaca.markets/support/

### This System
- **Migration Guide**: `TRADIER_ALPACA_MIGRATION.md`
- **Test Suite**: `python test_alpaca_migration.py`
- **Logs**: `logs/alpaca_monitor_*.log`

---

## 🎓 Learning Path

### Day 1: Setup
- [ ] Create Alpaca account
- [ ] Get API keys
- [ ] Setup `.env` file
- [ ] Run tests

### Day 2-7: Paper Trading
- [ ] Place 5+ paper trades
- [ ] Verify stop-loss triggers
- [ ] Verify profit targets work
- [ ] Monitor for 1 week

### Day 8-30: Live Testing (Small)
- [ ] Start with 1 contract trades
- [ ] Max $500 per position
- [ ] Verify all systems work
- [ ] Build confidence

### Day 30+: Scale Up
- [ ] Increase to 2-3 contracts
- [ ] Max $1,500 per position
- [ ] Maintain 60%+ win rate
- [ ] Review and optimize

---

## 💡 Pro Tips

1. **Always Paper Trade First**
   - Test every strategy
   - Verify all systems
   - Build muscle memory

2. **Start Small**
   - 1 contract per trade
   - Low-priced underlyings
   - Learn the system

3. **Monitor Daily**
   - Check logs every day
   - Review P&L
   - Adjust as needed

4. **Trust the System**
   - 25% stop-loss will save you
   - Don't override it
   - Let it work

5. **Keep Learning**
   - Review every trade
   - Track what works
   - Improve constantly

---

## ⚠️ Important Reminders

### NEVER
- ❌ Disable stop-loss monitoring
- ❌ Increase stop-loss above 25%
- ❌ Override automatic exits
- ❌ Trade without monitoring running
- ❌ Go back to Tradier

### ALWAYS
- ✅ Run monitoring daemon
- ✅ Check logs daily
- ✅ Verify stop-loss triggers
- ✅ Start with paper trading
- ✅ Keep risk at 2% per trade

---

## 🏁 You're Ready!

Everything is built, tested, and ready:

✅ Alpaca engine complete  
✅ Monitoring daemon ready  
✅ Risk parameters safe (25% stop)  
✅ Tests passing  
✅ Documentation complete  

**Next**: Run `python test_alpaca_migration.py`

If all tests pass → Start monitoring → Begin trading

**Remember**: Tradier cost us $8k. Alpaca protects us. Never go back.

---

**Let's trade safely! 🚀**

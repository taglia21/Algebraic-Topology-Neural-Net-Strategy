# 🚀 Tonight's Deployment Guide - Ready for Tomorrow's Trading

**Created**: February 3, 2026 (Market Closed)  
**Deploy Target**: DigitalOcean Droplet  
**Go-Live**: February 4, 2026 @ 9:30 AM EST

---

## 📋 Quick Start

### Step 1: Deploy to Droplet (15 minutes)

```bash
# Set your droplet IP
export DROPLET_IP="your.droplet.ip.address"

# Run deployment script
./deploy_to_droplet.sh
```

The script will:
- ✅ Create deployment package with all fixed modules
- ✅ Upload to droplet
- ✅ Backup existing installation
- ✅ Install dependencies
- ✅ Run validation tests

### Step 2: Configure API Keys (5 minutes)

```bash
# SSH into droplet
ssh root@$DROPLET_IP

# Navigate to system
cd /opt/trading-system

# Edit environment file
nano .env.production

# Update these CRITICAL values:
# - FINNHUB_API_KEY (get from https://finnhub.io)
# - TRADIER_API_KEY (get from https://developer.tradier.com)
# - TRADIER_ACCOUNT_ID

# Save and set permissions
chmod 600 .env.production
```

### Step 3: Run Offline Tests (10 minutes)

```bash
# Still on droplet
./run_tests_offline.sh
```

Expected output: **All tests should PASS**

If any failures:
- Check Python version (need 3.8+)
- Verify dependencies installed
- Review error messages

### Step 4: Tomorrow Morning Pre-Market (90 minutes before open)

```bash
# Follow the comprehensive checklist
cat PRE_MARKET_CHECKLIST.md

# Start at 8:00 AM EST
# Complete all validation by 9:30 AM EST
```

---

## 📦 What Gets Deployed

### Fixed Modules (All CRITICAL/HIGH Issues Resolved)

1. **src/risk_manager.py** (520 lines)
   - Thread-safe position tracking
   - Validated stop-loss calculations
   - LRU cache for correlations

2. **src/position_sizer.py**
   - Kelly div-by-zero protection ✅
   - Volatility array validation ✅
   - Safe position sizing

3. **src/multi_timeframe_analyzer.py**
   - Memory leak fixed (LRU cache) ✅
   - Exponential backoff retry ✅
   - Data validation

4. **src/sentiment_analyzer.py**
   - Memory leak fixed (LRU cache) ✅
   - API key security (headers) ✅
   - HTTP timeouts ✅
   - is_valid field

5. **src/enhanced_trading_engine.py**
   - NaN propagation prevented ✅
   - ATR validation ✅
   - Quantity rounding fix ✅

### Documentation Included

- ✅ PRODUCTION_AUDIT_REPORT.md
- ✅ FIXES_IMPLEMENTATION_COMPLETE.md
- ✅ PRE_MARKET_CHECKLIST.md
- ✅ run_tests_offline.sh
- ✅ .env.production.example

---

## ⚙️ Required Environment Variables

### MUST SET Before Trading

```bash
# API Keys
FINNHUB_API_KEY=your_real_key_here
TRADIER_API_KEY=your_real_key_here
TRADIER_ACCOUNT_ID=your_account_id_here

# Trading Mode (IMPORTANT!)
TRADING_MODE=paper  # Start with paper trading!
```

### Recommended Settings

```bash
# Portfolio
INITIAL_CAPITAL=100000.0
MAX_POSITION_SIZE_PCT=0.10

# Risk
MAX_DAILY_LOSS_PCT=0.02
MAX_CONCURRENT_POSITIONS=5

# Strategy
MIN_MTF_SCORE=60.0
MIN_COMBINED_SCORE=0.6
```

See [.env.production.example](.env.production.example) for all options.

---

## ✅ Pre-Deployment Checklist

### Tonight (Before Deployment)

- [ ] Droplet IP address confirmed
- [ ] SSH access verified
- [ ] API keys obtained (Finnhub, Tradier)
- [ ] Backup plan reviewed
- [ ] Emergency contacts notified

### During Deployment

- [ ] Run `./deploy_to_droplet.sh`
- [ ] All modules deployed successfully
- [ ] Dependencies installed
- [ ] Import tests passed
- [ ] `.env.production` configured
- [ ] Offline tests passed

### Tomorrow Morning (8:00 AM - 9:30 AM)

- [ ] Complete PRE_MARKET_CHECKLIST.md
- [ ] All API connectivity tests passed
- [ ] Live data validation successful
- [ ] Integration test passed
- [ ] Logs configured
- [ ] Monitoring ready

---

## 🧪 Testing Strategy

### Tonight: Offline Tests

Run while market is closed:

```bash
./run_tests_offline.sh
```

Tests:
- ✅ Module imports
- ✅ Division-by-zero protection
- ✅ Cache management (memory leaks)
- ✅ NaN validation
- ✅ Volatility calculation
- ✅ Risk calculations
- ✅ Integration

### Tomorrow: Pre-Market Tests

Run before market open (8:15 AM - 9:25 AM):

- ✅ API connectivity (yfinance, Finnhub)
- ✅ Live data fetching
- ✅ Multi-timeframe analysis
- ✅ Sentiment analysis
- ✅ Full integration test with AMD

### First Trading Day: Live Monitoring

Monitor from 9:30 AM - 10:30 AM:

- Memory usage (should be stable)
- API rate limits (no 429 errors)
- Error logs (no uncaught exceptions)
- Cache performance (hit rates)
- Trade execution (if signals generated)

---

## 🚨 Emergency Procedures

### If Deployment Fails

```bash
# Rollback to previous version
ssh root@$DROPLET_IP
cd /opt/trading-system-backups
LATEST=$(ls -t backup_*.tar.gz | head -1)
cd /opt/trading-system
tar -xzf ../trading-system-backups/$LATEST
```

### If Tests Fail Tomorrow

1. **API Issues**: Check keys, network, rate limits
2. **Import Issues**: Reinstall dependencies
3. **Data Issues**: Verify yfinance/Finnhub responding

**STOP TRADING if**:
- Any CRITICAL errors occur
- Memory usage spikes
- Uncaught exceptions
- Invalid position sizes
- API failures

### Contact Info

- **System Admin**: [Your contact]
- **Backup**: [Backup contact]
- **Rollback**: Use backup procedure above

---

## 📊 Success Metrics

### Deployment Success

- ✅ All modules uploaded
- ✅ All dependencies installed
- ✅ All offline tests passing
- ✅ Environment configured

### Pre-Market Success

- ✅ All API connections working
- ✅ Live data flowing
- ✅ No import errors
- ✅ Integration test passing

### First-Hour Success

- ✅ No crashes
- ✅ Memory stable
- ✅ No API errors
- ✅ Proper validation functioning

---

## 🎯 AMD Strategy Parameters

### Target Symbol: AMD

**Why AMD?**
- High liquidity
- Good volatility (tradeable moves)
- Strong sentiment data
- Options availability

### Entry Criteria (All Must Be True)

1. **Multi-Timeframe Alignment** ≥ 60/100
   - Timeframes: 5m, 15m, 1h, 4h, 1d
   - Majority bullish/bearish consensus

2. **Sentiment Score** ≥ 0.0 (neutral or better for long)
   - Minimum 3 articles
   - 48-hour lookback
   - Valid sentiment reading

3. **Combined Score** ≥ 0.6
   - 60% MTF weight
   - 40% sentiment weight

4. **Risk Validation**
   - Position size within limits
   - Stop-loss calculated
   - Take-profits set

### Position Sizing

- **Portfolio**: $100,000 (initial)
- **Max Position**: 10% = $10,000
- **Kelly-based sizing** with 0.5 multiplier
- **Volatility-adjusted**
- **Confidence-weighted**

### Risk Management

- **Stop-Loss**: 2x ATR from entry
- **Take-Profits**: 3 levels (1.5R, 3R, 5R)
- **Max Daily Loss**: 2% of portfolio
- **Max Positions**: 5 concurrent

---

## 📈 Expected First-Day Scenarios

### Scenario 1: Strong Signal Generated

- MTF Score: 75/100
- Sentiment: +0.4
- Combined: 0.68
- **Action**: Enter position per sizing rules
- **Monitor**: Every 5 minutes first hour

### Scenario 2: Weak Signal

- MTF Score: 45/100
- Sentiment: -0.1
- Combined: 0.20
- **Action**: NO TRADE (below threshold)
- **Monitor**: Wait for better setup

### Scenario 3: System Issues

- API errors
- Data validation failures
- Unexpected exceptions
- **Action**: STOP TRADING immediately
- **Review**: Check logs, troubleshoot

---

## 🔐 Security Reminders

### File Permissions

```bash
# On droplet
chmod 600 .env.production
chmod 700 /opt/trading-system
```

### API Key Security

- ✅ Never commit keys to Git
- ✅ Use headers (not query params)
- ✅ Rotate keys regularly
- ✅ Monitor for unauthorized use
- ✅ Different keys for paper/live

### Data Protection

- ✅ Backups encrypted
- ✅ Logs sanitized (no API keys)
- ✅ Access limited to authorized users

---

## 📝 Post-Deployment Verification

### After Deploy (Tonight)

```bash
# Verify files exist
ls -la /opt/trading-system/src/*.py

# Check permissions
ls -la /opt/trading-system/.env.production

# Test imports
python3 -c "from src.enhanced_trading_engine import EnhancedTradingEngine; print('OK')"
```

### After Pre-Market Tests (Tomorrow 9:25 AM)

```bash
# Check logs
tail -100 /opt/trading-system/logs/$(date +%Y%m%d)/*.log

# Memory check
free -h

# Process check
ps aux | grep python
```

---

## 🎉 Ready to Deploy!

### Final Checklist

- [ ] Read this document completely
- [ ] Have droplet IP ready
- [ ] Have API keys ready
- [ ] Understand rollback procedure
- [ ] Understand emergency stops
- [ ] Calendar blocked for tomorrow 8:00-10:30 AM

### Deploy Command

```bash
export DROPLET_IP="your.droplet.ip"
./deploy_to_droplet.sh
```

### Timeline

- **Tonight**: Deploy (15 min) + Configure (5 min) + Test (10 min) = **30 minutes**
- **Tomorrow 8:00 AM**: Pre-market validation = **90 minutes**
- **Tomorrow 9:30 AM**: Market open, monitoring begins

---

## 📞 Support

- Documentation: Check FIXES_IMPLEMENTATION_COMPLETE.md
- Test Results: run_tests_offline.sh output
- Pre-Market: Follow PRE_MARKET_CHECKLIST.md
- Issues: Review PRODUCTION_AUDIT_REPORT.md

---

**Remember**: This is your first live trading day with these fixes. Start conservative, monitor closely, and stop immediately if anything unexpected happens.

**Good luck! 🚀📈**

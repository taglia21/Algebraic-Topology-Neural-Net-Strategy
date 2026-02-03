# 📋 Pre-Market Checklist - February 4, 2026

**Market Open**: 9:30 AM EST  
**Recommended Start Time**: 8:00 AM EST (90 minutes before market)  
**Status**: 🔴 NOT STARTED

---

## ⏰ Timeline

| Time | Duration | Task | Status |
|------|----------|------|--------|
| 8:00 AM | 15 min | System Startup & Health Check | ⬜ |
| 8:15 AM | 20 min | API Connectivity Tests | ⬜ |
| 8:35 AM | 15 min | Run Offline Test Suite | ⬜ |
| 8:50 AM | 20 min | Pre-Market Data Validation | ⬜ |
| 9:10 AM | 20 min | Final System Validation | ⬜ |
| 9:30 AM | - | **MARKET OPEN** | ⬜ |

---

## 1. System Startup & Health Check (8:00 AM - 8:15 AM)

### SSH Connection
```bash
ssh root@YOUR_DROPLET_IP
cd /opt/trading-system
```
- [ ] SSH connection successful
- [ ] Trading system directory accessible

### Environment Check
```bash
# Check Python version (should be 3.8+)
python3 --version

# Verify virtual environment (if using)
source venv/bin/activate 2>/dev/null || echo "No venv"

# Check disk space (need >1GB free)
df -h /opt/trading-system

# Check memory (need >500MB free)
free -h
```
- [ ] Python 3.8+ installed
- [ ] Disk space adequate (>1GB)
- [ ] Memory adequate (>500MB free)
- [ ] CPU usage normal (<50%)

### Process Check
```bash
# Check if any trading processes already running
ps aux | grep -E "trading|enhanced"

# Kill any stale processes if needed
# pkill -f trading_system
```
- [ ] No stale processes running
- [ ] Ports available (if needed)

---

## 2. API Connectivity Tests (8:15 AM - 8:35 AM)

### Environment Variables
```bash
# Verify .env.production exists and has required keys
cat .env.production | grep -v "^#" | grep -v "^$"
```
- [ ] `.env.production` exists
- [ ] FINNHUB_API_KEY set (not "your_key_here")
- [ ] TRADIER_API_KEY set (if using Tradier)
- [ ] Other required keys present

### yfinance API Test
```bash
python3 -c "
import yfinance as yf
ticker = yf.Ticker('SPY')
data = ticker.history(period='1d', interval='1m')
print(f'✓ yfinance OK: {len(data)} bars fetched')
assert len(data) > 0, 'No data returned'
"
```
- [ ] yfinance API responding
- [ ] Data retrieval successful
- [ ] No rate limit errors

### Finnhub API Test
```bash
python3 -c "
import os
import requests
from dotenv import load_dotenv
load_dotenv('.env.production')

api_key = os.getenv('FINNHUB_API_KEY')
assert api_key and api_key != 'your_finnhub_api_key_here', 'Invalid API key'

url = 'https://finnhub.io/api/v1/quote?symbol=AAPL'
headers = {'X-Finnhub-Token': api_key}
response = requests.get(url, headers=headers, timeout=10)
response.raise_for_status()
data = response.json()
print(f\"✓ Finnhub OK: AAPL price = \${data.get('c', 0):.2f}\")
"
```
- [ ] Finnhub API responding
- [ ] API key valid
- [ ] Quote data retrieved

### Network Latency Test
```bash
# Test latency to key endpoints
ping -c 3 finance.yahoo.com
ping -c 3 finnhub.io
```
- [ ] Latency to yahoo.com <100ms
- [ ] Latency to finnhub.io <100ms
- [ ] No packet loss

---

## 3. Run Offline Test Suite (8:35 AM - 8:50 AM)

### Execute Test Script
```bash
chmod +x run_tests_offline.sh
./run_tests_offline.sh
```
- [ ] All import tests passed
- [ ] Kelly calculation tests passed
- [ ] Cache management tests passed
- [ ] Validation tests passed
- [ ] No critical errors in output

### Module Import Verification
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '/opt/trading-system')

from src.risk_manager import RiskManager, RiskConfig
from src.position_sizer import PositionSizer, SizingConfig, PerformanceMetrics
from src.multi_timeframe_analyzer import MultiTimeframeAnalyzer, AnalyzerConfig
from src.sentiment_analyzer import SentimentAnalyzer, SentimentConfig
from src.enhanced_trading_engine import EnhancedTradingEngine, EngineConfig

print("✓ All modules imported successfully")

# Quick instantiation test
risk_mgr = RiskManager()
sizer = PositionSizer()
mtf = MultiTimeframeAnalyzer()
sentiment = SentimentAnalyzer()
engine = EnhancedTradingEngine()

print("✓ All modules instantiated successfully")
EOF
```
- [ ] All modules import without errors
- [ ] All modules instantiate without errors
- [ ] No warnings about missing dependencies

---

## 4. Pre-Market Data Validation (8:50 AM - 9:10 AM)

### Test Live Data Fetching
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '/opt/trading-system')
from src.enhanced_trading_engine import EnhancedTradingEngine

# Initialize engine
engine = EnhancedTradingEngine()

# Test symbols (liquid pre-market stocks)
test_symbols = ['SPY', 'AAPL', 'MSFT']

print("Testing live data fetching...")
for symbol in test_symbols:
    try:
        # Test current price
        price = engine._get_current_price(symbol)
        assert price is not None and price > 0, f"Invalid price for {symbol}"
        print(f"✓ {symbol}: ${price:.2f}")
        
        # Test ATR calculation
        atr = engine._calculate_atr(symbol)
        assert atr >= 0, f"Invalid ATR for {symbol}"
        print(f"  ATR: ${atr:.2f}")
    except Exception as e:
        print(f"✗ {symbol} FAILED: {e}")
        
print("\nData validation complete")
EOF
```
- [ ] SPY price fetched successfully
- [ ] AAPL price fetched successfully
- [ ] MSFT price fetched successfully
- [ ] ATR calculations successful
- [ ] No timeout errors

### Test Multi-Timeframe Analysis
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '/opt/trading-system')
from src.multi_timeframe_analyzer import MultiTimeframeAnalyzer

analyzer = MultiTimeframeAnalyzer()

# Test with SPY
print("Testing multi-timeframe analysis on SPY...")
analysis = analyzer.analyze('SPY')

print(f"  Alignment Score: {analysis.alignment_score:.1f}/100")
print(f"  Dominant Trend: {analysis.dominant_trend.name}")
print(f"  Bullish TFs: {analysis.bullish_timeframes}")
print(f"  Bearish TFs: {analysis.bearish_timeframes}")
print(f"  ✓ Multi-timeframe analysis working")
EOF
```
- [ ] MTF analysis completes without errors
- [ ] Alignment score calculated
- [ ] Timeframe data available
- [ ] Cache functioning

### Test Sentiment Analysis
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '/opt/trading-system')
from src.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()

print("Testing sentiment analysis on AAPL...")
result = analyzer.get_sentiment('AAPL')

print(f"  Score: {result.score:+.2f}")
print(f"  Level: {result.level.value}")
print(f"  Articles: {result.article_count}")
print(f"  Valid: {result.is_valid}")
print(f"  ✓ Sentiment analysis working")
EOF
```
- [ ] Sentiment analysis completes
- [ ] Articles fetched
- [ ] Score calculated
- [ ] No API errors

---

## 5. Final System Validation (9:10 AM - 9:30 AM)

### Complete Integration Test
```bash
python3 << 'EOF'
import sys
import os
sys.path.insert(0, '/opt/trading-system')
from dotenv import load_dotenv
load_dotenv('.env.production')

from src.enhanced_trading_engine import EnhancedTradingEngine
from src.position_sizer import PerformanceMetrics

print("="*60)
print("FULL INTEGRATION TEST - AMD Options Strategy")
print("="*60)

# Initialize engine
engine = EnhancedTradingEngine()

# Test symbol (AMD - your primary target)
symbol = 'AMD'
portfolio_value = 10000.0  # Start small for testing

# Sample performance metrics (will be replaced with real data)
metrics = PerformanceMetrics(
    total_trades=100,
    winning_trades=55,
    losing_trades=45,
    total_profit=11000.0,
    total_loss=-9000.0
)

print(f"\nAnalyzing {symbol} with ${portfolio_value:,.0f} portfolio...")
print("-"*60)

try:
    decision = engine.analyze_opportunity(
        symbol=symbol,
        portfolio_value=portfolio_value,
        performance_metrics=metrics
    )
    
    print(f"\nDECISION SUMMARY:")
    print(f"  Signal: {decision.signal.value.upper()}")
    print(f"  Tradeable: {'YES' if decision.is_tradeable else 'NO'}")
    print(f"  MTF Score: {decision.mtf_score:.1f}/100")
    print(f"  Sentiment: {decision.sentiment_score:+.2f}")
    print(f"  Combined Score: {decision.combined_score:.2f}")
    print(f"  Confidence: {decision.confidence:.2%}")
    print(f"\nPOSITION SIZING:")
    print(f"  Entry Price: ${decision.entry_price:.2f}")
    print(f"  Quantity: {decision.recommended_quantity}")
    print(f"  Position Value: ${decision.recommended_position_value:.2f}")
    print(f"  Stop Loss: ${decision.stop_loss:.2f}")
    print(f"  Take Profits: {[f'${tp:.2f}' for tp in decision.take_profits]}")
    
    if decision.rejection_reasons:
        print(f"\nREJECTION REASONS:")
        for reason in decision.rejection_reasons:
            print(f"  - {reason}")
    
    print("\n" + "="*60)
    print("✓ INTEGRATION TEST PASSED")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ INTEGRATION TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF
```
- [ ] Integration test completed successfully
- [ ] No exceptions thrown
- [ ] Trade decision generated
- [ ] Position sizing calculated
- [ ] Risk parameters set

### Log File Setup
```bash
# Create logs directory
mkdir -p /opt/trading-system/logs

# Set up log rotation
cat > /etc/logrotate.d/trading-system << 'EOF'
/opt/trading-system/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
EOF

# Verify log directory writable
touch /opt/trading-system/logs/test.log && rm /opt/trading-system/logs/test.log
```
- [ ] Logs directory created
- [ ] Log rotation configured
- [ ] Permissions correct

### System Resource Baseline
```bash
# Record baseline metrics
echo "=== System Baseline at $(date) ===" > /opt/trading-system/logs/baseline.log
free -h >> /opt/trading-system/logs/baseline.log
df -h >> /opt/trading-system/logs/baseline.log
ps aux --sort=-%mem | head -10 >> /opt/trading-system/logs/baseline.log
```
- [ ] Baseline metrics recorded
- [ ] Memory usage normal
- [ ] CPU usage normal

---

## 6. Market Open Readiness (9:30 AM)

### Pre-Open Final Checks (9:25 AM - 9:30 AM)
```bash
# Quick health check
python3 -c "
from src.enhanced_trading_engine import EnhancedTradingEngine
engine = EnhancedTradingEngine()
price = engine._get_current_price('SPY')
print(f'✓ System ready - SPY: \${price:.2f}')
"
```
- [ ] System responding
- [ ] Live data flowing
- [ ] No errors in last 5 minutes

### Trading Session Initialization
```bash
# Start logging
mkdir -p /opt/trading-system/logs/$(date +%Y%m%d)

# Set log file for today
export LOG_FILE="/opt/trading-system/logs/$(date +%Y%m%d)/trading_$(date +%H%M%S).log"

# Test logging
echo "Trading session started at $(date)" >> $LOG_FILE
```
- [ ] Log file created
- [ ] Logging working
- [ ] Timestamp correct

---

## ⚠️ Emergency Procedures

### If Tests Fail

1. **Import Errors**:
   ```bash
   pip3 install -r requirements.txt --force-reinstall
   ```

2. **API Errors**:
   - Check API key validity
   - Verify network connectivity
   - Check rate limits

3. **Data Errors**:
   - Use cached data if available
   - Fall back to alternative data source
   - Consider manual override

### Emergency Contacts
- **System Admin**: [Your contact]
- **Backup Contact**: [Backup contact]

### Rollback Procedure
```bash
# If deployment has critical issues, rollback:
cd /opt/trading-system-backups
LATEST_BACKUP=$(ls -t backup_*.tar.gz | head -1)
cd /opt/trading-system
tar -xzf ../trading-system-backups/$LATEST_BACKUP
```

---

## 📊 First-Hour Monitoring

### Monitor These Metrics (9:30 AM - 10:30 AM)

```bash
# Monitor log for errors
tail -f /opt/trading-system/logs/$(date +%Y%m%d)/*.log | grep -E "ERROR|CRITICAL|EXCEPTION"

# Monitor memory usage
watch -n 60 'free -h'

# Monitor API rate limits
# Check logs for "rate limit" or "429" errors
```

**Watch for**:
- [ ] No memory leaks (stable RAM usage)
- [ ] No API rate limit errors
- [ ] No uncaught exceptions
- [ ] Cache functioning properly
- [ ] Proper data validation

---

## ✅ Final Sign-Off

**Time**: ____________  
**Completed By**: ____________  

- [ ] All checklist items completed
- [ ] System validated and ready
- [ ] Logs configured
- [ ] Emergency procedures reviewed
- [ ] Backup contact notified

**Notes**:
```
[Add any special notes, warnings, or observations here]
```

---

## 🚀 GO/NO-GO Decision

**Decision**: ⬜ GO  ⬜ NO-GO

**Reasoning**:
```
[Document why system is/isn't ready for live trading]
```

**Authorized By**: ____________  
**Timestamp**: ____________

---

**Remember**: 
- Start with SMALL position sizes
- Monitor EVERY trade closely
- Stop trading if ANYTHING unusual happens
- This is first live trading day with new fixes - BE CAUTIOUS!

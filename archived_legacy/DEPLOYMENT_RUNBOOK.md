# V2.1 Production Deployment Runbook

## Executive Summary

This runbook provides step-by-step procedures for deploying, operating, and maintaining the V2.1 TDA+Neural Net Trading System in production.

**System Overview:**
- **Version**: V2.1 Production
- **Target Sharpe**: > 1.55 (V1.3 baseline 1.35 + proven enhancements)
- **Risk Limits**: 5% circuit breaker, 8% emergency halt
- **Universe**: 700+ stocks with TDA features

---

## Table of Contents

1. [Pre-Deployment Checklist](#1-pre-deployment-checklist)
2. [Environment Setup](#2-environment-setup)
3. [Startup Sequence](#3-startup-sequence)
4. [Health Checks](#4-health-checks)
5. [Monitoring Operations](#5-monitoring-operations)
6. [Failure Recovery](#6-failure-recovery)
7. [Scaling Procedures](#7-scaling-procedures)
8. [Emergency Procedures](#8-emergency-procedures)
9. [Maintenance Tasks](#9-maintenance-tasks)
10. [Appendix](#10-appendix)

---

## 1. Pre-Deployment Checklist

### 1.1 Infrastructure Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8 cores |
| RAM | 8 GB | 16 GB |
| Storage | 50 GB SSD | 100 GB SSD |
| Network | 100 Mbps | 1 Gbps |
| OS | Ubuntu 22.04+ | Ubuntu 24.04 LTS |

### 1.2 Required Credentials

```bash
# Create .env file with credentials
cat > /opt/tda-trading/.env << 'EOF'
# Alpaca Trading API
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
PAPER_TRADING=true

# Data Provider
POLYGON_API_KEY_OTREP=your_polygon_key_here

# Notifications
DISCORD_WEBHOOK=https://discord.com/api/webhooks/your_webhook_here

# Optional: Monitoring
SENTRY_DSN=optional_sentry_dsn
EOF

chmod 600 /opt/tda-trading/.env
```

### 1.3 Pre-Deployment Verification

```bash
# Run verification script
cd /opt/tda-trading
python -c "
from dotenv import load_dotenv
import os
load_dotenv()

required = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'POLYGON_API_KEY_OTREP']
missing = [k for k in required if not os.getenv(k)]
if missing:
    print(f'❌ Missing credentials: {missing}')
    exit(1)
print('✅ All required credentials configured')
"

# Verify Alpaca connection
python -c "
from src.trading.alpaca_client import AlpacaClient
client = AlpacaClient()
account = client.get_account()
print(f'✅ Alpaca connected - Equity: \${float(account.equity):,.2f}')
"

# Run test suite
python tests/test_deployment.py
```

---

## 2. Environment Setup

### 2.1 System Installation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.11 python3.11-venv python3-pip git tmux htop

# Clone repository
cd /opt
git clone https://github.com/taglia21/Algebraic-Topology-Neural-Net-Strategy.git tda-trading
cd tda-trading

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install additional V2.1 dependencies
pip install hmmlearn torch scikit-learn
```

### 2.2 Directory Structure

```bash
# Create required directories
mkdir -p /opt/tda-trading/{logs,cache,data,results}

# Set permissions
chmod 755 /opt/tda-trading
chmod 777 /opt/tda-trading/logs
chmod 777 /opt/tda-trading/cache
```

### 2.3 Systemd Service Setup

```bash
# Create systemd service file
sudo cat > /etc/systemd/system/tda-trading.service << 'EOF'
[Unit]
Description=V2.1 TDA Trading Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/tda-trading
Environment="PATH=/opt/tda-trading/venv/bin"
EnvironmentFile=/opt/tda-trading/.env
ExecStart=/opt/tda-trading/venv/bin/python production_launcher.py
Restart=always
RestartSec=30
StandardOutput=append:/opt/tda-trading/logs/stdout.log
StandardError=append:/opt/tda-trading/logs/stderr.log

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable tda-trading
sudo systemctl start tda-trading
```

---

## 3. Startup Sequence

### 3.1 Manual Startup (Development/Testing)

```bash
# Navigate to project
cd /opt/tda-trading
source venv/bin/activate

# Start in paper trading mode (default)
python production_launcher.py

# Start with specific options
python production_launcher.py --dry-run      # No orders submitted
python production_launcher.py --once         # Single cycle then exit
python production_launcher.py --no-dashboard # Disable web dashboard
python production_launcher.py --debug        # Enable debug logging
```

### 3.2 Service Startup

```bash
# Start service
sudo systemctl start tda-trading

# Check status
sudo systemctl status tda-trading

# View logs
journalctl -u tda-trading -f
```

### 3.3 Startup Verification

```bash
# Check process is running
ps aux | grep production_launcher

# Verify dashboard is accessible
curl http://localhost:8080/health

# Check log for successful startup
tail -20 /opt/tda-trading/logs/production.log | grep -i "initialization complete"
```

### 3.4 Post-Startup Checklist

- [ ] Dashboard accessible at http://localhost:8080
- [ ] Discord notification received for startup
- [ ] All components showing ✅ in dashboard
- [ ] Alpaca account connected
- [ ] No error alerts in dashboard

---

## 4. Health Checks

### 4.1 Automated Health Checks

Create `/opt/tda-trading/scripts/health_check.sh`:

```bash
#!/bin/bash
set -e

echo "=== V2.1 Health Check ==="
echo "Time: $(date)"

# Check 1: Process running
if pgrep -f "production_launcher" > /dev/null; then
    echo "✅ Process: Running"
else
    echo "❌ Process: Not running"
    exit 1
fi

# Check 2: Dashboard responsive
if curl -s http://localhost:8080/health | grep -q "ok"; then
    echo "✅ Dashboard: Healthy"
else
    echo "❌ Dashboard: Not responding"
    exit 1
fi

# Check 3: Recent log activity
last_log=$(stat -c %Y /opt/tda-trading/logs/production.log 2>/dev/null || echo 0)
now=$(date +%s)
age=$((now - last_log))
if [ $age -lt 300 ]; then
    echo "✅ Logs: Active (${age}s ago)"
else
    echo "⚠️ Logs: Stale (${age}s ago)"
fi

# Check 4: Disk space
disk_pct=$(df /opt/tda-trading | tail -1 | awk '{print $5}' | tr -d '%')
if [ $disk_pct -lt 90 ]; then
    echo "✅ Disk: ${disk_pct}% used"
else
    echo "⚠️ Disk: ${disk_pct}% used - Consider cleanup"
fi

# Check 5: Memory
mem_pct=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100}')
if [ $mem_pct -lt 90 ]; then
    echo "✅ Memory: ${mem_pct}% used"
else
    echo "⚠️ Memory: ${mem_pct}% used - High usage"
fi

echo "=== Health Check Complete ==="
```

### 4.2 Cron-based Health Monitoring

```bash
# Add to crontab
crontab -e

# Add these lines:
*/5 * * * * /opt/tda-trading/scripts/health_check.sh >> /opt/tda-trading/logs/health.log 2>&1
0 16 * * 1-5 /opt/tda-trading/venv/bin/python /opt/tda-trading/scripts/daily_summary.py
```

### 4.3 Manual Health Check Commands

```bash
# API endpoint check
curl http://localhost:8080/api/metrics | python -m json.tool

# Check component status
python -c "
from src.trading.v21_production_engine import V21ProductionEngine
engine = V21ProductionEngine()
print(engine.get_component_status())
"

# Check Alpaca connection
python -c "
from src.trading.alpaca_client import AlpacaClient
client = AlpacaClient()
account = client.get_account()
print(f'Equity: \${float(account.equity):,.2f}')
print(f'Buying Power: \${float(account.buying_power):,.2f}')
"
```

---

## 5. Monitoring Operations

### 5.1 Dashboard Access

- **URL**: http://YOUR_SERVER_IP:8080
- **API Endpoint**: http://YOUR_SERVER_IP:8080/api/metrics
- **Health Check**: http://YOUR_SERVER_IP:8080/health

### 5.2 Key Metrics to Monitor

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Sharpe (30d) | > 1.0 | < 0.8 | < 0.5 |
| Drawdown | < 3% | > 3% | > 5% |
| Win Rate | > 50% | < 45% | < 40% |
| Avg Slippage | < 5bp | > 10bp | > 20bp |
| Daily Trades | 10-50 | < 5 or > 100 | 0 |

### 5.3 Log Monitoring

```bash
# Real-time production logs
tail -f /opt/tda-trading/logs/production.log

# Filter for errors
grep -i error /opt/tda-trading/logs/production.log | tail -20

# Filter for trades
grep -i "executed" /opt/tda-trading/logs/production.log | tail -20

# View metrics log (JSON lines)
tail -f /opt/tda-trading/logs/metrics.jsonl | python -m json.tool

# View validation reports
cat /opt/tda-trading/logs/validation_$(date +%Y%m%d).json | python -m json.tool
```

### 5.4 Discord Notification Events

The system sends Discord notifications for:
- System startup/shutdown
- Trade executions
- Regime changes
- Daily summaries
- Circuit breaker warnings
- Emergency halts
- Validation anomalies

---

## 6. Failure Recovery

### 6.1 Common Issues and Solutions

#### Issue: Process Not Starting

```bash
# Check for port conflicts
sudo lsof -i :8080

# Kill stuck process
pkill -9 -f production_launcher

# Clear stale lock files
rm -f /opt/tda-trading/*.lock

# Restart
sudo systemctl restart tda-trading
```

#### Issue: Alpaca Connection Failed

```bash
# Verify credentials
source /opt/tda-trading/.env
echo "API Key: ${ALPACA_API_KEY:0:8}..."

# Test connection manually
python -c "
from src.trading.alpaca_client import AlpacaClient
try:
    client = AlpacaClient()
    print('✅ Connection successful')
except Exception as e:
    print(f'❌ Connection failed: {e}')
"

# Check Alpaca status page
curl -s https://status.alpaca.markets/api/v2/status.json
```

#### Issue: Data Feed Stale

```bash
# Check Polygon API
python -c "
from src.data.data_provider import get_ohlcv_data
df = get_ohlcv_data('SPY', start='2026-01-15', end='2026-01-21')
print(f'Latest data: {df.index[-1]}')
"

# Force cache refresh
rm -rf /opt/tda-trading/cache/ohlcv/*
```

#### Issue: High Memory Usage

```bash
# Check memory usage
ps aux --sort=-%mem | head -10

# Clear Python cache
find /opt/tda-trading -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Restart with memory limit
systemctl edit tda-trading
# Add: MemoryMax=8G
```

### 6.2 Recovery Procedures

#### Full System Recovery

```bash
#!/bin/bash
# Full recovery script

echo "Starting full system recovery..."

# Stop service
sudo systemctl stop tda-trading

# Clear stale state
rm -f /opt/tda-trading/logs/*.lock
rm -f /opt/tda-trading/cache/*.tmp

# Verify database/cache integrity
python -c "
import json
from pathlib import Path
for f in Path('/opt/tda-trading/logs').glob('*.json'):
    try:
        json.load(open(f))
    except:
        print(f'Corrupt file: {f}')
        f.rename(f.with_suffix('.json.corrupt'))
"

# Restart
sudo systemctl start tda-trading

# Verify
sleep 10
curl http://localhost:8080/health
```

#### Trading Halt Reset

```bash
# Reset trading halt manually
python -c "
from src.trading.v21_production_engine import V21ProductionEngine
engine = V21ProductionEngine()
engine.reset_halt()
print('Trading halt reset')
"
```

---

## 7. Scaling Procedures

### 7.1 Horizontal Scaling (Multiple Instances)

Not recommended for trading systems due to order duplication risk. Use vertical scaling.

### 7.2 Vertical Scaling

```bash
# Upgrade droplet/instance
# 1. Stop service
sudo systemctl stop tda-trading

# 2. Snapshot current state
# (via cloud provider console)

# 3. Resize instance
# (via cloud provider console)

# 4. Restart service
sudo systemctl start tda-trading
```

### 7.3 Universe Expansion

```bash
# Edit config to expand universe
python -c "
from src.trading.v21_production_engine import V21Config

config = V21Config(
    universe_mode='mega',  # Options: core, expanded, mega
    max_positions=100,     # Increase from default 50
)
print(config.to_dict())
"
```

---

## 8. Emergency Procedures

### 8.1 Emergency Shutdown

```bash
#!/bin/bash
# EMERGENCY SHUTDOWN - Use only in crisis

echo "🚨 EMERGENCY SHUTDOWN INITIATED"

# Stop trading immediately
sudo systemctl stop tda-trading

# Close all positions (optional - use with extreme caution)
python -c "
from src.trading.alpaca_client import AlpacaClient
client = AlpacaClient()
client.close_all_positions()
print('All positions closed')
"

# Send notification
curl -X POST $DISCORD_WEBHOOK \
    -H "Content-Type: application/json" \
    -d '{"content": "🚨 EMERGENCY SHUTDOWN - Trading halted, positions closed"}'

echo "Emergency shutdown complete"
```

### 8.2 Circuit Breaker Activation

The system automatically activates circuit breakers:

| Trigger | Action |
|---------|--------|
| 3% drawdown | Reduce new positions by 50% |
| 5% drawdown | Halt new trades, alert sent |
| 8% drawdown | Emergency halt, all positions reviewed |
| 3 consecutive losing days | Halt trading |

### 8.3 Manual Position Intervention

```bash
# View all positions
python -c "
from src.trading.alpaca_client import AlpacaClient
client = AlpacaClient()
for pos in client.get_positions():
    print(f'{pos.symbol}: {pos.qty} shares @ \${pos.avg_entry_price:.2f}')
"

# Close specific position
python -c "
from src.trading.alpaca_client import AlpacaClient, OrderSide
client = AlpacaClient()
client.submit_order('AAPL', 100, OrderSide.SELL)  # Adjust symbol/qty
"
```

---

## 9. Maintenance Tasks

### 9.1 Daily Tasks (Automated)

- End-of-day summary (16:30 ET)
- Daily validation report
- Log rotation check

### 9.2 Weekly Tasks

```bash
# Log cleanup (keep 30 days)
find /opt/tda-trading/logs -name "*.log" -mtime +30 -delete
find /opt/tda-trading/logs -name "*.jsonl" -mtime +30 -delete

# Update dependencies (non-breaking)
pip list --outdated
```

### 9.3 Monthly Tasks

```bash
# Full backup
tar -czvf /backup/tda-trading-$(date +%Y%m%d).tar.gz /opt/tda-trading

# System updates
sudo apt update && sudo apt upgrade -y

# Review and optimize
python scripts/performance_monitor.py --report monthly
```

### 9.4 Quarterly Tasks

- Model retraining evaluation
- Strategy performance review
- Risk parameter adjustment
- Infrastructure capacity review

---

## 10. Appendix

### 10.1 Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| ALPACA_API_KEY | Yes | Alpaca API key |
| ALPACA_SECRET_KEY | Yes | Alpaca secret key |
| ALPACA_BASE_URL | No | API endpoint (default: paper) |
| PAPER_TRADING | No | Force paper mode (default: true) |
| POLYGON_API_KEY_OTREP | Yes | Polygon.io API key |
| DISCORD_WEBHOOK | No | Discord notification webhook |

### 10.2 Port Reference

| Port | Service | Notes |
|------|---------|-------|
| 8080 | Dashboard | Main monitoring UI |
| 443 | Alpaca API | Outbound only |
| 443 | Polygon API | Outbound only |

### 10.3 Log File Reference

| File | Contents |
|------|----------|
| production.log | Main application logs |
| metrics.jsonl | Structured metrics (JSON lines) |
| cycles.jsonl | Trading cycle results |
| validation_YYYYMMDD.json | Daily validation reports |
| health.log | Health check results |

### 10.4 Key Configuration Files

| File | Purpose |
|------|---------|
| .env | Environment credentials |
| production_launcher.py | Main entry point |
| src/trading/v21_production_engine.py | Trading engine |
| src/trading/monitoring_dashboard.py | Dashboard |
| src/trading/daily_validator.py | Validation |

### 10.5 Support Contacts

- **Repository Issues**: https://github.com/taglia21/Algebraic-Topology-Neural-Net-Strategy/issues
- **Alpaca Support**: https://alpaca.markets/support
- **Polygon Support**: https://polygon.io/support

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-21 | V2.1.0 | Initial production runbook |

---

*Last Updated: January 21, 2026*
*Document Version: 1.0*

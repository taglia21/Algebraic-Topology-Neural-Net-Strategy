# V2.1 Production Launch Checklist

**Generated:** 2026-01-21  
**System:** TDA Trading Bot V2.1  
**Validated:** Sharpe 1.35, Max DD 2.08%

---

## Pre-Deployment Checklist

### 1. Infrastructure Setup (Digital Ocean)

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Droplet created ($12/mo, 2GB RAM, 1vCPU) | ☐ | Ubuntu 22.04 LTS |
| 2 | SSH key configured | ☐ | Root access enabled |
| 3 | Firewall rules set (22, 8080) | ☐ | UFW configured |
| 4 | DNS A record configured | ☐ | Optional: trading.yourdomain.com |
| 5 | Digital Ocean Spaces bucket created | ☐ | For daily backups |

### 2. API Keys & Secrets

| # | Item | Status | Notes |
|---|------|--------|-------|
| 6 | Polygon.io API key obtained | ☐ | POLYGON_API_KEY_OTREP |
| 7 | Polygon API tested (curl works) | ☐ | Test: `curl "https://api.polygon.io/v2/aggs/ticker/SPY/prev?apiKey=YOUR_KEY"` |
| 8 | Alpaca API key obtained | ☐ | ALPACA_API_KEY + ALPACA_SECRET_KEY |
| 9 | Alpaca API tested | ☐ | Verify live trading enabled |
| 10 | Discord webhook created | ☐ | DISCORD_WEBHOOK_URL |
| 11 | Discord webhook tested | ☐ | Test message received |

### 3. Code Validation

| # | Item | Status | Notes |
|---|------|--------|-------|
| 12 | All tests pass (94/94) | ☐ | `pytest tests/` |
| 13 | Production launcher runs | ☐ | `python production_launcher.py --mode=backtest --days=5` |
| 14 | Health check endpoint works | ☐ | `curl localhost:8080/health` |
| 15 | Graceful shutdown works | ☐ | Send SIGTERM, verify state saved |

### 4. Monitoring & Alerts

| # | Item | Status | Notes |
|---|------|--------|-------|
| 16 | Discord alert on drawdown >3% | ☐ | Test alert sent |
| 17 | Discord alert on API error | ☐ | 5 consecutive fails |
| 18 | Discord alert on position breach | ☐ | Exceeds 3% limit |
| 19 | Discord alert on service restart | ☐ | systemd restart notification |
| 20 | Health check monitoring configured | ☐ | Uptime robot / cron ping |

---

## Deployment Steps

### Step 1: Deploy to Droplet

```bash
# From local machine
python scripts/deploy_to_droplet.py \
    --host YOUR_DROPLET_IP \
    --user root \
    --key ~/.ssh/id_rsa
```

### Step 2: Configure Environment

```bash
# SSH to droplet
ssh root@YOUR_DROPLET_IP

# Edit .env with your API keys
nano /opt/trading-bot/.env

# Verify configuration
cat /opt/trading-bot/.env | grep -E "^[A-Z]" | head -10
```

### Step 3: Start Service

```bash
# Start trading bot
sudo systemctl start trading_bot.service

# Check status
sudo systemctl status trading_bot.service

# Enable auto-start
sudo systemctl enable trading_bot.service
```

### Step 4: Verify Health

```bash
# Local health check
curl http://localhost:8080/health | jq

# Expected output:
# {
#   "status": "healthy",
#   "api_status": {"polygon": "ok", "alpaca": "ok"},
#   "uptime_seconds": 30,
#   "version": "2.1.0"
# }
```

### Step 5: Monitor Logs

```bash
# Live logs
sudo journalctl -u trading_bot -f

# Trading log
tail -f /opt/trading-bot/logs/trading.log
```

---

## Post-Deployment Verification

### Day 1 Monitoring

| Time | Check | Expected |
|------|-------|----------|
| Market Open (09:30 ET) | Service running | `systemctl status` = active |
| 10:00 ET | First signals generated | Logs show "Generating signals" |
| 11:00 ET | Health endpoint | status = "healthy" |
| 15:30 ET | Position summary | Discord notification |
| Market Close (16:00 ET) | Daily summary | Discord report |
| 00:00 UTC | Backup completed | Spaces backup created |

### Week 1 Validation

- [ ] No unexpected restarts
- [ ] Sharpe ratio tracking positive
- [ ] Max drawdown < 3%
- [ ] All API calls successful (>99%)
- [ ] Discord alerts working correctly
- [ ] Log rotation functioning
- [ ] Backup restoration tested

---

## Emergency Procedures

### Service Failure

```bash
# Check status
sudo systemctl status trading_bot.service

# View recent logs
sudo journalctl -u trading_bot -n 100

# Restart service
sudo systemctl restart trading_bot.service
```

### API Key Rotation

```bash
# Stop service
sudo systemctl stop trading_bot.service

# Update keys
nano /opt/trading-bot/.env

# Restart
sudo systemctl start trading_bot.service
```

### Rollback

```bash
# Stop service
sudo systemctl stop trading_bot.service

# Restore from backup
tar -xzf /path/to/backup.tar.gz -C /opt/trading-bot

# Restart
sudo systemctl start trading_bot.service
```

### Emergency Position Exit

```bash
# Cancel all open orders via Alpaca
curl -X DELETE https://api.alpaca.markets/v2/orders \
  -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
  -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY"

# Liquidate all positions (CAUTION)
curl -X DELETE https://api.alpaca.markets/v2/positions \
  -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
  -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY"
```

---

## GO LIVE Authorization

### Final Checks

| Check | Verified By | Date |
|-------|-------------|------|
| All 20 checklist items complete | ______________ | ____/____/____ |
| Paper trading profitable for 5+ days | ______________ | ____/____/____ |
| Emergency procedures tested | ______________ | ____/____/____ |
| Backup restoration verified | ______________ | ____/____/____ |
| Discord alerts tested | ______________ | ____/____/____ |

### Authorization

**I authorize the transition from PAPER to LIVE trading:**

Signature: _______________________________

Date: ____/____/____

Initial Capital: $____________

Maximum Loss Tolerance: $____________

---

## Contact Information

| Role | Contact |
|------|---------|
| Primary Engineer | ______________ |
| Backup Contact | ______________ |
| Broker Support | Alpaca: support@alpaca.markets |
| Data Provider | Polygon: support@polygon.io |

---

*Last Updated: 2026-01-21*
*Version: V2.1 Production*

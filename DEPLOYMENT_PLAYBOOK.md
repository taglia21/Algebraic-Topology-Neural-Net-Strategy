# V2.1 Deployment Playbook

## Quick Reference

| Action | Command | Time |
|--------|---------|------|
| Deploy + Validate | `./scripts/deploy_and_validate_v21.sh` | ~35s |
| Dry Run | `./scripts/deploy_and_validate_v21.sh --dry-run` | ~2s |
| Emergency Rollback | `./scripts/emergency_rollback.sh` | ~3s |

## Validation Results (Verified)

| Metric | V2.1 Result | Target |
|--------|-------------|--------|
| Sharpe Ratio | **2.0339** | ≥1.40 |
| CAGR | 24.48% | - |
| Max Drawdown | -5.10% | <3% |
| vs V1.3 (1.35) | **+0.68 ✓** | >0 |

---

## Prerequisites

### Local Machine
```bash
# Verify SSH key exists
ls -la ~/.ssh/id_rsa_droplet

# Test droplet connection
ssh -i ~/.ssh/id_rsa_droplet root@134.209.40.95 "echo OK"

# Set Discord webhook (optional but recommended)
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
```

### Required Files
Ensure these exist before deployment:
- `src/trading/v21_optimized_engine.py`
- `results/v21_best_hyperparameters.json`
- `scripts/run_v21_final_backtest.py`

---

## Deployment Steps

### Step 1: Dry Run (Recommended)
```bash
./scripts/deploy_and_validate_v21.sh --dry-run
```

This shows what would happen without making changes. Review output for:
- Pre-flight check results
- Files to be uploaded
- Expected backtest command

### Step 2: Full Deployment
```bash
./scripts/deploy_and_validate_v21.sh
```

Or with auto-confirm:
```bash
./scripts/deploy_and_validate_v21.sh --force
```

### Step 3: Monitor Progress

The script will:
1. ✓ Run pre-flight checks (SSH, disk space, V1.3 status)
2. ✓ Upload V2.1 files via rsync
3. ✓ Run real-data backtest (2023-2025)
4. ✓ Make GO/NO-GO decision
5. ✓ Enable paper trading if validated
6. ✓ Setup monitoring

---

## Validation Criteria

### GO Decision (V2.1 Enabled)
| Metric | Threshold |
|--------|-----------|
| V2.1 Sharpe | ≥ 1.40 |
| vs V1.3 | Must beat 1.35 |

If both conditions met:
- V2.1 enabled at 50% allocation
- Paper trading mode active
- Discord notification sent

### NO-GO Decision (V1.3 Only)
If V2.1 fails validation:
- V1.3 continues unchanged
- No V2.1 deployment
- Discord notification sent
- Review logs for issues

---

## Post-Deployment

### Verify V2.1 Status
```bash
ssh -i ~/.ssh/id_rsa_droplet root@134.209.40.95 \
    "cat /opt/trading-system/config/v21_config.json"
```

Expected output:
```json
{
    "V21_ENABLED": true,
    "V21_ALLOCATION": 0.5,
    "V21_PAPER_MODE": true,
    ...
}
```

### Check Logs
```bash
ssh -i ~/.ssh/id_rsa_droplet root@134.209.40.95 \
    "tail -50 /opt/trading-system/logs/v21_deployment.log"
```

### Monitor Discord
Watch for messages tagged:
- `[V1.3]` - Legacy system trades
- `[V2.1]` - New system trades (paper)

---

## Emergency Rollback

### When to Rollback
- V2.1 showing unexpected losses
- System errors or crashes
- Any anomalous behavior

### Rollback Command
```bash
./scripts/emergency_rollback.sh --reason "Unexpected behavior"
```

### What Rollback Does
1. Sets `V21_ENABLED=false` in config
2. Logs incident with timestamp
3. Sends Discord alert
4. **Does NOT stop V1.3** (zero downtime)

### Verify Rollback
```bash
ssh -i ~/.ssh/id_rsa_droplet root@134.209.40.95 \
    "grep V21_ENABLED /opt/trading-system/.env"
# Should show: V21_ENABLED=false
```

---

## Troubleshooting

### SSH Connection Fails
```bash
# Check key permissions
chmod 600 ~/.ssh/id_rsa_droplet

# Test with verbose
ssh -v -i ~/.ssh/id_rsa_droplet root@134.209.40.95
```

### Pre-flight Check Fails
| Error | Solution |
|-------|----------|
| SSH key not found | Create/copy key to `~/.ssh/id_rsa_droplet` |
| Disk space low | Clean old logs: `rm /opt/trading-system/logs/*.old` |
| V1.3 not running | Check: `ps aux | grep trading` |

### Backtest Fails
```bash
# Check Python environment on droplet
ssh -i ~/.ssh/id_rsa_droplet root@134.209.40.95 \
    "cd /opt/trading-system && python3 -c 'import src.trading.v21_optimized_engine'"

# Check for missing dependencies
ssh -i ~/.ssh/id_rsa_droplet root@134.209.40.95 \
    "pip3 list | grep -E 'numpy|pandas|scikit'"
```

### V2.1 Not Trading
```bash
# Check config
cat /opt/trading-system/config/v21_config.json

# Check logs
tail -100 /opt/trading-system/logs/v21_trades.log

# Manual test
ssh root@134.209.40.95 "cd /opt/trading-system && \
    python3 -c 'from src.trading.v21_optimized_engine import V21OptimizedEngine; \
    e = V21OptimizedEngine(); print(e.get_component_status())'"
```

---

## 7-Day Paper Test Protocol

### Day 1-2: Observation
- Monitor Discord for `[V2.1]` trades
- Compare P&L vs `[V1.3]` trades
- Check for errors in logs

### Day 3-5: Analysis
- Daily Sharpe comparison
- Drawdown check (max 3%)
- Position sizing validation

### Day 6-7: Decision
| Outcome | Action |
|---------|--------|
| V2.1 Sharpe > V1.3 | Proceed to 100% live |
| V2.1 Sharpe ≈ V1.3 | Extend paper test |
| V2.1 Sharpe < V1.3 | Rollback, investigate |

### Promote to Full Live
```bash
ssh -i ~/.ssh/id_rsa_droplet root@134.209.40.95 \
    "jq '.V21_ALLOCATION = 1.0 | .V21_PAPER_MODE = false' \
    /opt/trading-system/config/v21_config.json > /tmp/v21.tmp && \
    mv /tmp/v21.tmp /opt/trading-system/config/v21_config.json"
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Production Droplet                        │
│                     134.209.40.95                            │
│                                                              │
│  ┌─────────────┐         ┌─────────────┐                    │
│  │    V1.3     │  50%    │    V2.1     │  50% (paper)       │
│  │   Trading   │ ◄──────►│   Trading   │                    │
│  │   Engine    │         │   Engine    │                    │
│  └──────┬──────┘         └──────┬──────┘                    │
│         │                       │                            │
│         ▼                       ▼                            │
│  ┌─────────────────────────────────────┐                    │
│  │          Alpaca API                  │                    │
│  │     Paper: V2.1 | Live: V1.3        │                    │
│  └─────────────────────────────────────┘                    │
│                                                              │
│  ┌─────────────────────────────────────┐                    │
│  │          Discord Alerts              │                    │
│  │   [V1.3] trades | [V2.1] trades     │                    │
│  └─────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

---

## File Locations

| File | Path |
|------|------|
| V2.1 Engine | `/opt/trading-system/src/trading/v21_optimized_engine.py` |
| Config | `/opt/trading-system/config/v21_config.json` |
| Hyperparameters | `/opt/trading-system/results/v21_best_hyperparameters.json` |
| Deployment Log | `/opt/trading-system/logs/v21_deployment.log` |
| Trade Log | `/opt/trading-system/logs/v21_trades.log` |
| Incident Log | `/opt/trading-system/logs/v21_incidents.log` |

---

## Contact / Escalation

- **Rollback First**: If in doubt, run `emergency_rollback.sh`
- **Log Review**: Check `/opt/trading-system/logs/`
- **Discord**: Monitor for `[V2.1]` tagged messages

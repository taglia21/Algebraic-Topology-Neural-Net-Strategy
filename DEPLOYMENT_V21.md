# V2.1 Deployment Guide

**Version:** V2.1 Production-Ready System  
**Date:** January 2026  
**Author:** Trading System Team

---

## Overview

This guide covers the deployment of V2.1 optimized trading system, which includes only the proven enhancements from ablation analysis:

| Component | Status | Sharpe Impact |
|-----------|--------|---------------|
| Ensemble Regime Detection | ✅ INCLUDED | +0.22 |
| Transformer Predictor | ✅ INCLUDED | +0.09 |
| Standard TDA Features | ✅ INCLUDED | (V1.3 base) |
| SAC Agent | ❌ REMOVED | -0.02 |
| Persistent Laplacian | ❌ REMOVED | -0.11 |
| Risk Parity | ❌ REMOVED | -0.22 |

---

## 1. Prerequisites & Environment Setup

### 1.1 System Requirements
- **OS:** Ubuntu 20.04+ or Debian 11+
- **Python:** 3.10+
- **Memory:** 4GB+ RAM
- **Disk:** 10GB+ free space

### 1.2 Required Packages
```bash
# Install system dependencies
sudo apt update && sudo apt install -y python3-pip git

# Clone repository (if not already present)
git clone https://github.com/your-repo/Algebraic-Topology-Neural-Net-Strategy.git
cd Algebraic-Topology-Neural-Net-Strategy

# Install Python dependencies
pip install -r requirements.txt

# Install V2.1 specific dependencies
pip install optuna scikit-learn hmmlearn
```

### 1.3 Environment Variables
```bash
# Add to ~/.bashrc or ~/.profile
export POLYGON_API_KEY="your_polygon_api_key"
export ALPACA_API_KEY="your_alpaca_api_key"
export ALPACA_SECRET_KEY="your_alpaca_secret_key"
export DISCORD_WEBHOOK_URL="your_discord_webhook_url"
```

---

## 2. V2.1 Installation Steps

### 2.1 Copy V2.1 Files to Droplet
```bash
# From local machine
scp src/trading/v21_optimized_engine.py root@YOUR_DROPLET:/path/to/project/src/trading/
scp results/v21_best_hyperparameters.json root@YOUR_DROPLET:/path/to/project/results/
```

### 2.2 Verify V2.1 Engine
```bash
# On droplet
cd /path/to/project
python -c "
from src.trading.v21_optimized_engine import V21OptimizedEngine
engine = V21OptimizedEngine()
print('V2.1 Status:', engine.get_status())
"
```

Expected output:
```
✅ Ensemble Regime initialized
✅ Transformer Predictor initialized
✅ Standard TDA Features initialized
✅ V2.1 Optimized Engine initialized
V2.1 Status: {'version': 'V2.1', 'components': {...}, 'is_halted': False}
```

### 2.3 Update Trading Script
Modify `deploy_tda_trading.py` to use V2.1:

```python
# Add at top of file
from src.trading.v21_optimized_engine import get_trading_engine

# Replace existing engine initialization with:
engine = get_trading_engine(use_v21=True)
```

Or use the backward-compatible flag:
```python
# V2.1 with fallback to V1.3 if components fail
engine = get_trading_engine(
    use_v21=True,
    fallback_to_v13=True
)
```

---

## 3. Configuration & Testing

### 3.1 Load Optimized Hyperparameters
```python
import json
from src.trading.v21_optimized_engine import V21Config, V21OptimizedEngine

# Load optimized hyperparameters
with open('results/v21_best_hyperparameters.json') as f:
    hp = json.load(f)['best_params']

config = V21Config(**hp)
engine = V21OptimizedEngine(config)
```

### 3.2 Test with Historical Data
```bash
# Run quick backtest to verify
python scripts/run_v21_final_backtest.py --start 2024-01-01 --end 2025-01-01
```

### 3.3 Shadow Mode Testing
Run V2.1 in parallel with V1.3 for 1-2 weeks:
```bash
# Start V2.1 in shadow mode (generates signals but doesn't trade)
python deploy_tda_trading.py --mode shadow --engine v21
```

Compare signals daily:
```bash
# Compare V1.3 vs V2.1 signal differences
tail -f /tmp/rebalance.log | grep -E "(V1.3|V2.1|signal|position)"
```

---

## 4. Go-Live Procedure

### 4.1 Pre-Launch Checklist
- [ ] V2.1 engine passes all tests
- [ ] Hyperparameters loaded from JSON
- [ ] Shadow mode comparison shows acceptable differences
- [ ] Discord alerts configured for V2.1 events
- [ ] Circuit breakers verified (3 losing days, 5% DD halt)
- [ ] Position limits set (15% per asset, 60% max cash)
- [ ] Backup of V1.3 configuration saved

### 4.2 Deployment Commands
```bash
# 1. Stop current V1.3 system
sudo systemctl stop trading-bot

# 2. Backup current config
cp /etc/trading/config.json /etc/trading/config.json.v13.backup

# 3. Update to V2.1
python -c "
from src.trading.v21_optimized_engine import V21OptimizedEngine
engine = V21OptimizedEngine()
print('Engine status:', engine.get_component_status())
"

# 4. Start V2.1 system
sudo systemctl start trading-bot

# 5. Verify running
sudo systemctl status trading-bot
tail -f /tmp/rebalance.log
```

### 4.3 Post-Launch Verification
```bash
# Check component status
ssh root@YOUR_DROPLET "cat /tmp/rebalance.log | grep -E '(V2.1|Ensemble|Transformer|regime)' | tail -20"

# Verify positions
ssh root@YOUR_DROPLET "cat /tmp/rebalance.log | grep -E '(position|allocation|trade)' | tail -10"
```

---

## 5. Monitoring & Alerts

### 5.1 Key Metrics to Monitor
| Metric | Alert Threshold | Action |
|--------|-----------------|--------|
| Daily Return | < -2% | Review positions |
| Consecutive Losses | ≥ 3 days | Circuit breaker auto-halts |
| Max Drawdown | > 5% | Auto-halt + manual review |
| Regime Change | Bull ↔ Bear | Log and verify response |
| Component Failure | Any | Fallback to V1.3 |

### 5.2 Discord Alert Integration
V2.1 automatically sends alerts for:
- Regime changes (bull → bear, etc.)
- Circuit breaker triggers
- Component failures
- Daily performance summary

### 5.3 Log Monitoring
```bash
# Real-time monitoring
tail -f /tmp/rebalance.log | grep -E "(regime|circuit|WARN|ERROR|signal)"

# Daily summary
cat /tmp/rebalance.log | grep -E "$(date +%Y-%m-%d).*(complete|summary|PnL)"
```

---

## 6. Rollback Plan

### 6.1 Quick Rollback (< 5 minutes)
```bash
# Stop V2.1
sudo systemctl stop trading-bot

# Restore V1.3 config
cp /etc/trading/config.json.v13.backup /etc/trading/config.json

# Update engine flag
sed -i 's/use_v21=True/use_v21=False/' deploy_tda_trading.py

# Restart with V1.3
sudo systemctl start trading-bot
```

### 6.2 Rollback Triggers
Rollback immediately if:
- [ ] V2.1 Sharpe drops below 0.5 (vs V1.3 baseline)
- [ ] Max DD exceeds 5% within first week
- [ ] Multiple component failures occur
- [ ] Trading halts due to circuit breakers

### 6.3 Post-Rollback Analysis
```bash
# Save V2.1 logs for analysis
cp /tmp/rebalance.log ~/v21_logs/rebalance_$(date +%Y%m%d).log

# Document failure reason
echo "Rollback reason: [describe issue]" >> ~/v21_logs/rollback_reasons.txt
```

---

## 7. Troubleshooting

### 7.1 Common Issues

**Issue:** Ensemble Regime not initializing
```bash
# Check hmmlearn installation
pip install hmmlearn --upgrade

# Verify
python -c "from src.trading.regime_ensemble import EnsembleRegimeDetector; print('OK')"
```

**Issue:** Transformer predictions all zeros
```bash
# Check PyTorch installation
pip install torch --upgrade

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Issue:** Circuit breaker halting unexpectedly
```python
# Reset circuit breaker
from src.trading.v21_optimized_engine import V21OptimizedEngine
engine = V21OptimizedEngine()
engine.reset_circuit_breakers()
```

### 7.2 Debug Mode
```bash
# Run with verbose logging
export LOG_LEVEL=DEBUG
python deploy_tda_trading.py --verbose
```

### 7.3 Health Check Script
```bash
# Create health check script
cat > /usr/local/bin/v21-health-check.sh << 'EOF'
#!/bin/bash
echo "V2.1 Health Check - $(date)"
echo "========================"

# Check process
pgrep -f "deploy_tda_trading" > /dev/null && echo "✅ Process running" || echo "❌ Process not running"

# Check recent activity
tail -1 /tmp/rebalance.log | grep -q "$(date +%Y-%m-%d)" && echo "✅ Recent activity" || echo "⚠️ No recent activity"

# Check errors
grep -c "ERROR" /tmp/rebalance.log | xargs -I{} echo "Errors today: {}"

# Check regime
grep "regime" /tmp/rebalance.log | tail -1
EOF

chmod +x /usr/local/bin/v21-health-check.sh
```

---

## 8. Performance Expectations

Based on backtesting and ablation analysis:

| Metric | V1.3 Baseline | V2.1 Expected | Notes |
|--------|---------------|---------------|-------|
| Sharpe | 1.35 | 1.55-1.70 | +0.22 from regime + 0.09 from transformer |
| CAGR | 18% | 20-25% | Improved market timing |
| Max DD | -2.08% | < -2.5% | Similar risk profile |
| Win Rate | 55% | 57-60% | Better regime filtering |

---

## 9. Support & Escalation

- **Level 1:** Check logs, restart service
- **Level 2:** Rollback to V1.3, investigate
- **Level 3:** Contact development team

---

*Document Version: 1.0*  
*Last Updated: January 2026*

# V2.0 Enhanced Trading System - Deployment Guide

## Overview

This guide covers deploying the V2.0 Enhanced Trading System to production. The system includes:

- **Transformer Predictor**: Multi-head attention for stock direction prediction
- **SAC Agent with PER**: Soft Actor-Critic for dynamic position sizing
- **Persistent Laplacian TDA**: Enhanced topological features (12 new features)
- **Ensemble Regime Detection**: HMM + GMM + Agglomerative Clustering
- **Order Flow Analyzer**: Real-time microstructure analysis

### Target Metrics
| Metric | V1.3 | V2.0 Target |
|--------|------|-------------|
| Sharpe Ratio | 1.35 | ≥ 1.50 |
| Max Drawdown | 2.08% | ≤ 1.50% |
| CAGR | 16.41% | ≥ 18.00% |
| Training Time | N/A | < 120 seconds |

---

## Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ / Debian 11+ (recommended)
- **Python**: 3.10+ (3.11 recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **CPU**: 4+ cores
- **GPU**: Optional (CUDA 11.7+ for faster training)
- **Storage**: 20GB+ available

### API Keys Required
- **Alpaca Trading API**: Paper/Live account
- **Discord Webhook**: For notifications (optional)
- **Polygon.io**: Alternative data source (optional)

---

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-org/Algebraic-Topology-Neural-Net-Strategy.git
cd Algebraic-Topology-Neural-Net-Strategy
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
# Core dependencies
pip install -r requirements.txt

# V2.0 additional dependencies
pip install torch>=2.0.0 hmmlearn>=0.3.0 ripser persim

# Optional: GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 4. Verify Installation
```bash
python -c "
import torch
from src.ml.transformer_predictor import TransformerPredictor
from src.ml.sac_agent import SACAgent
from src.tda_v2.persistent_laplacian import PersistentLaplacian
from src.trading.regime_ensemble import EnsembleRegimeDetector
print('All V2 components loaded successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

---

## Configuration

### Environment Variables
Create `.env` file in project root:

```bash
# Alpaca API
ALPACA_API_KEY=your_api_key_here
ALPACA_API_SECRET=your_api_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Use https://api.alpaca.markets for live

# Discord Notifications
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# V2 Configuration
V2_USE_TRANSFORMER=true
V2_USE_SAC=true
V2_USE_LAPLACIAN=true
V2_USE_ENSEMBLE_REGIME=true
V2_USE_ORDER_FLOW=true
V2_FALLBACK_V13=true

# Risk Parameters
MAX_PORTFOLIO_HEAT=0.25
MAX_POSITION_SIZE=0.05
STOP_LOSS_ATR_MULT=2.0
TAKE_PROFIT_ATR_MULT=3.0
```

### V2 Config File
Create `config/v2_config.json`:

```json
{
  "version": "2.0",
  "components": {
    "transformer": {
      "hidden_dim": 512,
      "n_heads": 8,
      "n_layers": 3,
      "dropout": 0.1
    },
    "sac": {
      "state_dim": 37,
      "gamma": 0.99,
      "tau": 0.005,
      "alpha_init": 0.2,
      "auto_alpha": true
    },
    "tda": {
      "max_dimension": 1,
      "n_filtrations": 20,
      "n_eigenvalues": 10
    },
    "regime": {
      "n_regimes": 3,
      "hmm_weight": 0.5,
      "gmm_weight": 0.3,
      "cluster_weight": 0.2,
      "consensus_threshold": 2
    }
  },
  "risk": {
    "max_portfolio_heat": 0.25,
    "max_position_size": 0.05,
    "min_position_size": 0.005,
    "stop_loss_atr_mult": 2.0,
    "take_profit_atr_mult": 3.0
  },
  "trading": {
    "rebalance_frequency": "daily",
    "market_open_delay_minutes": 15,
    "market_close_buffer_minutes": 30
  }
}
```

---

## Pre-Deployment Validation

### 1. Run Test Suite
```bash
python -m pytest tests/test_v2_components.py -v

# Expected output: All tests pass
```

### 2. Run Ablation Study
```bash
python scripts/run_v2_backtest_ablation.py \
  --start 2022-01-01 \
  --end 2024-12-31 \
  --output results/v2_ablation_results.json

# Verify V2 Full meets targets
```

### 3. Validate with Paper Trading
```bash
# Start paper trading session
python main.py --mode paper --duration 5d --version v2

# Monitor for:
# - No errors in logs
# - Reasonable trade frequency
# - Risk limits respected
```

---

## Deployment Options

### Option 1: DigitalOcean Droplet (Recommended)

#### Create Droplet
```bash
# Using doctl CLI
doctl compute droplet create tda-trading-v2 \
  --size s-2vcpu-4gb \
  --image ubuntu-22-04-x64 \
  --region nyc1 \
  --ssh-keys YOUR_SSH_KEY_ID
```

#### Setup Droplet
```bash
# SSH into droplet
ssh root@YOUR_DROPLET_IP

# Update system
apt update && apt upgrade -y

# Install Python
apt install -y python3.11 python3.11-venv python3-pip git

# Clone and setup
git clone YOUR_REPO_URL
cd Algebraic-Topology-Neural-Net-Strategy
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch hmmlearn ripser persim

# Setup environment
cp .env.example .env
nano .env  # Edit with your API keys

# Test
python -c "from src.trading.v2_enhanced_engine import V2EnhancedEngine; print('OK')"
```

#### Create Systemd Service
```bash
sudo tee /etc/systemd/system/tda-trading-v2.service << 'EOF'
[Unit]
Description=TDA Trading Bot V2.0
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/Algebraic-Topology-Neural-Net-Strategy
Environment=PATH=/root/Algebraic-Topology-Neural-Net-Strategy/venv/bin
ExecStart=/root/Algebraic-Topology-Neural-Net-Strategy/venv/bin/python main.py --mode live --version v2
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable tda-trading-v2
sudo systemctl start tda-trading-v2
```

### Option 2: Docker Deployment

#### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch hmmlearn ripser persim

# Copy source code
COPY . .

# Environment
ENV PYTHONUNBUFFERED=1

# Run
CMD ["python", "main.py", "--mode", "live", "--version", "v2"]
```

#### Docker Compose
```yaml
version: '3.8'

services:
  trading-bot:
    build: .
    container_name: tda-trading-v2
    restart: always
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
      - ./results:/app/results
      - ./models:/app/models
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f trading-bot
```

### Option 3: AWS EC2

```bash
# Launch t3.medium instance with Ubuntu 22.04

# SSH and setup
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# Follow same setup as DigitalOcean
# Use systemd or screen/tmux for persistence
```

---

## Migration from V1.3

### Step 1: Backup Current State
```bash
# On production server
cd /path/to/v1.3
cp -r . ../v1.3_backup_$(date +%Y%m%d)

# Backup model weights
cp results/*.weights.h5 ../v1.3_backup_$(date +%Y%m%d)/
```

### Step 2: Deploy V2.0 Alongside V1.3
```bash
# Clone V2 to new directory
git clone REPO_URL v2.0
cd v2.0

# Copy V1.3 credentials
cp ../v1.3/.env .
```

### Step 3: Run V2.0 in Shadow Mode
```bash
# V2 runs but doesn't trade, just logs signals
python main.py --mode shadow --version v2 --shadow-duration 7d
```

### Step 4: Gradual Transition
```bash
# Week 1: V1.3 100%, V2.0 shadow
# Week 2: V1.3 75%, V2.0 25% capital
# Week 3: V1.3 50%, V2.0 50% capital
# Week 4: V1.3 25%, V2.0 75% capital
# Week 5: V2.0 100%

# Adjust via config:
echo '{"v2_capital_ratio": 0.25}' > config/transition.json
```

### Step 5: Full Cutover
```bash
# Stop V1.3
sudo systemctl stop tda-trading-v13

# Start V2.0
sudo systemctl start tda-trading-v2

# Monitor
journalctl -u tda-trading-v2 -f
```

---

## Monitoring

### Health Checks
```bash
# Check service status
sudo systemctl status tda-trading-v2

# View recent logs
journalctl -u tda-trading-v2 --since "1 hour ago"

# Check component status
curl http://localhost:8080/health  # If health endpoint enabled
```

### Discord Alerts
The system sends alerts for:
- Trade executions
- Regime changes
- Position limit breaches
- Error conditions
- Daily performance summary

### Log Files
```
logs/
├── trading.log          # Main trading log
├── v2_engine.log        # V2 component logs
├── transformer.log      # Predictor training/inference
├── sac.log              # RL agent updates
├── regime.log           # Regime detection
└── errors.log           # Error-only log
```

### Key Metrics Dashboard
Monitor these metrics:
- **Sharpe Ratio** (rolling 30d): Target ≥ 1.5
- **Max Drawdown** (rolling 30d): Target ≤ 1.5%
- **CAGR** (annualized): Target ≥ 18%
- **Win Rate**: Expected 50-55%
- **Average Trade Duration**: Expected 3-7 days
- **Position Count**: Expected 15-25 positions

---

## Troubleshooting

### Common Issues

#### 1. PyTorch Not Using GPU
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Fix: Reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### 2. HMM Fitting Fails
```bash
# Symptom: "HMM fitting failed: convergence..."

# Fix: Increase iterations or reduce n_regimes
# In config: "hmm_n_iter": 200, "n_regimes": 2
```

#### 3. Memory Issues
```bash
# Symptom: OOM errors

# Fix: Reduce batch sizes
export V2_TRANSFORMER_BATCH=64
export V2_SAC_BATCH=128
```

#### 4. Slow Training
```bash
# Symptom: Training > 120s

# Fix: Enable GPU or reduce epochs
export V2_TRANSFORMER_EPOCHS=3
export V2_USE_GPU=true
```

#### 5. Connection Errors
```bash
# Symptom: Alpaca API timeouts

# Fix: Check network, use retry logic
# Built-in: System retries 3 times with exponential backoff
```

### Recovery Procedures

#### Emergency Stop
```bash
# Stop all trading immediately
sudo systemctl stop tda-trading-v2

# Or send SIGTERM
pkill -f "main.py.*v2"
```

#### Rollback to V1.3
```bash
# Stop V2
sudo systemctl stop tda-trading-v2

# Start V1.3
sudo systemctl start tda-trading-v13

# Notify
curl -X POST $DISCORD_WEBHOOK -H "Content-Type: application/json" \
  -d '{"content": "⚠️ Rollback to V1.3 executed"}'
```

---

## Maintenance

### Daily Tasks
- Review overnight trade log
- Check position limits
- Verify regime detection accuracy

### Weekly Tasks
- Analyze performance vs targets
- Review error logs
- Check disk space

### Monthly Tasks
- Retrain Transformer on new data
- Update regime detector
- Backup model weights
- Performance report generation

### Model Retraining
```bash
# Monthly retraining script
python scripts/retrain_models.py \
  --transformer-epochs 10 \
  --lookback-days 365 \
  --output models/v2_retrained_$(date +%Y%m).pt
```

---

## Support

### Logs to Include in Bug Reports
1. `logs/trading.log` (last 1000 lines)
2. `logs/errors.log` (full file)
3. System info: `uname -a`, Python version, PyTorch version
4. Configuration files (redact API keys)

### Contact
- GitHub Issues: [Repository URL]
- Discord: #trading-bot-support

---

## Changelog

### V2.0.0 (Current)
- Added Transformer predictor (replacing LSTM)
- Added SAC agent with PER (replacing Q-learning)
- Added Persistent Laplacian TDA (12 new features)
- Added Ensemble regime detection (HMM + GMM + Clustering)
- Added Order flow analyzer
- Target improvements: Sharpe +11%, Max DD -28%, CAGR +10%

### V1.3.x (Previous)
- LSTM predictor
- Q-learning agent
- Basic TDA features
- HMM-only regime detection

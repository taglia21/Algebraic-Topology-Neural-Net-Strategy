# ATNN v2 — Algebraic Topology + Neural Network Trading System

Quantitative trading system that combines **Topological Data Analysis (TDA)** with **deep learning** for market regime detection, signal generation, and automated execution via Interactive Brokers.

## Architecture

```
                         +------------------+
                         |   Market Data    |
                         |   (IBKR Feed)    |
                         +--------+---------+
                                  |
                    +-------------+-------------+
                    |                           |
              +-----v------+            +------v------+
              |    TDA      |            |     NN      |
              | Persistent  |            |   LSTM /    |
              | Homology    |            |  Attention  |
              | Betti Curves|            |   LSTM      |
              | Spectral Gap|            |             |
              +-----+------+            +------+------+
                    |                           |
                    +-------------+-------------+
                                  |
                         +--------v---------+
                         |    Ensemble       |
                         | Meta-Allocator    |
                         | Signal Aggregator |
                         +--------+---------+
                                  |
                         +--------v---------+
                         |  Risk Manager     |
                         |  Kelly Sizing     |
                         |  Drawdown Gates   |
                         +--------+---------+
                                  |
                    +-------------+-------------+
                    |                           |
              +-----v------+            +------v------+
              |  Options    |            |  Equities   |
              |  Trader     |            |  Trader     |
              | (vertical,  |            | (market,    |
              |  condors)   |            |  limit)     |
              +-----+------+            +------+------+
                    |                           |
                    +-------------+-------------+
                                  |
                         +--------v---------+
                         |      IBKR        |
                         | TWS / Gateway    |
                         +------------------+
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/taglia21/Algebraic-Topology-Neural-Net-Strategy.git
cd Algebraic-Topology-Neural-Net-Strategy
pip install -r requirements.txt

# Run backtest
python main.py backtest

# Run options backtest
python main.py backtest --options

# Train NN models
python main.py train

# Check system status
python main.py status

# Run live (paper mode — requires IBKR TWS/Gateway)
python main.py live

# Run with custom config
python main.py --config config/custom.yaml live
```

## Modules

| Module | Description |
|--------|-------------|
| `tda/` | Persistent homology, Betti curves, spectral gap, diffusion, regime detection |
| `nn/` | LSTM and Attention-LSTM models, walk-forward training pipeline |
| `ensemble/` | Meta-allocator, signal aggregator, ensemble risk manager |
| `backtest/` | Event-driven backtester, walk-forward optimizer, options backtester, HTML reports |
| `broker/` | IBKR data feed, equity/options traders, portfolio manager, risk monitor |
| `core/` | Config, structured logging, kill switch, market hours, regime detector, risk manager |

## Configuration

All configuration lives in `config/default.yaml`. Key sections:

- **system** — Mode (paper/live/backtest), logging, directories
- **broker** — IBKR connection (host, port, account)
- **universe** — Trading symbols and benchmark
- **tda** — Persistent homology and spectral parameters
- **nn** — LSTM architecture and training hyperparameters
- **ensemble** — TDA/NN weighting and signal aggregation
- **risk** — Position limits, Kelly fraction, drawdown gates, small-account rules
- **options/equities** — Trading engine enable flags (both DORMANT by default)
- **backtest** — Capital, walk-forward windows, commissions
- **schedule** — Signal generation and reconciliation times

Override any setting with environment variables:
```bash
IBKR_HOST=192.168.1.100 IBKR_PORT=4002 python main.py live
```

## Deployment (Docker)

```bash
# Copy and edit environment
cp .env.example .env
# Edit .env with your IBKR credentials

# Build and run
docker compose up -d

# View logs
docker compose logs -f atnn-bot
```

## Safety

- Both options and equities engines start **DORMANT** (`enabled: false`)
- Kill switch halts trading at 5% daily loss or 15% max drawdown
- Small-account rules: $50 max risk per trade, 3 concurrent positions max
- All signals are logged even when engines are dormant (dry-run mode)
- Graceful shutdown on SIGINT/SIGTERM

## Risk Warning

This software is for educational and research purposes. Trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk. The authors are not responsible for any financial losses incurred.

## License

MIT

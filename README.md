# ATNN v2 — Algebraic Topology + Neural Network Ensemble Trading System

Production-grade quantitative trading system combining:
- **Topological Data Analysis (TDA)**: Persistent homology, Betti curves, graph Laplacian diffusion for regime detection and arbitrage signal generation
- **Neural Networks**: LSTM/Attention-LSTM for directional prediction with topology-aware features
- **Meta-Classifier Ensemble**: Dynamic capital allocation between TDA arbitrage and NN directional strategies
- **Options Engine**: TDA+NN informed options trading (credit spreads, verticals, iron condors)
- **Equities Engine**: Long/short equity execution (dormant until authorized)

## Broker
Interactive Brokers (IBKR) — sole broker and data source via `ib_async` API.

## Status
- [x] Phase 0: Repository cleanup and infrastructure
- [ ] Phase 1: TDA module
- [ ] Phase 2: Neural network module
- [ ] Phase 3: Options engine
- [ ] Phase 4: Ensemble + risk management
- [ ] Phase 5: Backtesting engine
- [ ] Phase 6: Equities engine (dormant)
- [ ] Phase 7: Integration + deployment

## Architecture
```
IBKR Data Feed
     │
     ▼
┌─────────────┐     ┌─────────────┐
│  TDA Module  │────▶│  NN Module   │
│ (Topology)   │     │ (LSTM/Attn)  │
└──────┬───────┘     └──────┬───────┘
       │                     │
       ▼                     ▼
┌──────────────────────────────────┐
│      Ensemble Meta-Classifier    │
│   (Capital Allocation + Risk)    │
└──────────────┬───────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌────────────┐   ┌─────────────┐
│  Options    │   │  Equities   │
│  Engine     │   │  Engine     │
│  (ACTIVE)   │   │  (DORMANT)  │
└─────┬──────┘   └─────────────┘
      │
      ▼
   IBKR TWS/Gateway
```

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your IBKR credentials

# Run
python main.py --mode live
```

# ATNN v2 — Architecture

## Overview

The system combines Topological Data Analysis (TDA) with neural networks in a
meta-classifier ensemble. IBKR is the sole broker and data source.

## Module Map

```
core/                  Infrastructure (config, logging, risk, regime detection)
tda/                   Topological Data Analysis (persistent homology, Betti curves, Laplacian diffusion)
nn/                    Neural networks (LSTM, Attention-LSTM, topology-aware features)
nn/models/             Model definitions
ensemble/              Meta-classifier for capital allocation between TDA + NN
broker/                IBKR integration via ib_async
options/               Options trading engine (credit spreads, verticals, iron condors)
equities/              Equities engine (dormant until authorized)
data/                  Market data abstraction (IBKR data provider)
backtest/              Backtesting engine
```

## Data Flow

1. **IBKR Data Feed** → raw OHLCV bars, options chains, account data
2. **TDA Module** → persistent homology on price manifolds → Betti curves, topological features
3. **NN Module** → LSTM/Attention-LSTM with topology-aware features → directional predictions
4. **Ensemble** → meta-classifier combines TDA arbitrage signals + NN directional signals
5. **Risk Manager** → Kelly sizing, drawdown gates, correlation limits, sector caps
6. **Options Engine** (active) / Equities Engine (dormant) → order generation
7. **Broker** → IBKR TWS/Gateway execution

## TDA Approach

- **Persistent Homology**: Compute persistence diagrams from sliding-window point clouds of returns
- **Betti Curves**: Track topological feature evolution (β₀ = connected components, β₁ = loops/cycles)
- **Graph Laplacian Diffusion**: Correlation graph → Laplacian eigenvalues → diffusion process for regime detection
- **Regime Signals**: Topological phase transitions indicate regime changes before traditional indicators

## Neural Network Approach

- **Base Model**: LSTM for sequential return prediction
- **Enhanced Model**: Attention-LSTM hybrid with multi-head self-attention
- **Features**: Standard technical features + TDA-derived topological features
- **Training**: Walk-forward with purged cross-validation

## Ensemble

- **Meta-Classifier**: Learns optimal capital allocation between TDA and NN strategies
- **Dynamic Weighting**: Adjusts based on recent strategy performance and regime state
- **Risk Budget**: Total portfolio risk allocated across strategies via risk parity

## Risk Framework

| Parameter | Default | Description |
|---|---|---|
| Max Position | 20% | Single name cap as % of equity |
| Max Sector | 35% | Sector exposure cap |
| Max Drawdown Halt | -30% | Trading halts at this drawdown |
| Max Drawdown Reduce | -20% | Exposure reduced to 60% |
| Daily Loss Limit | -3% | Intraday loss circuit breaker |
| Vol Target | 20% | Annualized portfolio volatility target |
| Kelly Fraction | 0.5 | Half-Kelly position sizing |

## Broker Integration

- **Library**: `ib_async` (async Python wrapper for IBKR TWS/Gateway API)
- **Account**: U22452226
- **Capabilities**: Equities, options, real-time data, historical data, account management
- **No Alpaca**: Alpaca has been fully removed from the system
- **No yfinance**: yfinance has been fully removed from the system

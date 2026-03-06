# Machine Learning & Training Systems

This document explains how the trading bot trains itself to recognize patterns and generate alpha.

## Training Infrastructure

### 1. Walk-Forward Optimization
- Rolling 252-day training windows with 63-day out-of-sample validation
- Prevents overfitting by simulating real trading conditions

### 2. Prioritized Experience Replay
- Stores 100,000 recent trading experiences
- Prioritizes high-volatility events (>2% moves)
- 70% priority / 30% random sampling
- ~40% improved sample efficiency

### 3. Alpha Decay Monitoring  
- Tracks Information Coefficient (IC) over time
- Fits exponential decay models
- Triggers retraining when IC drops >50%

### 4. Automatic Retraining
**Triggers**: Every 7 days OR 30% performance drop OR alpha decay

## Neural Network
- 4-layer MLP with dropout + batch normalization
- Best Sharpe: 1.58 (backtested on 176 symbols)
- 50+ engineered features

## Data Sources
1. OHLCV Price/Volume
2. 50+ Technical Indicators
3. Cross-Asset Correlations
4. Volatility Regimes (HMM)
5. Cointegration Relationships
6. Order Flow Proxies

## Key Differentiators
1. Walk-forward validation (not curve-fit)
2. Automatic alpha decay detection
3. Self-retraining without human intervention  
4. Prioritized learning from significant events
5. Multi-strategy approach
6. 176-symbol diversification

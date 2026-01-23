# V17.0 Architecture Summary

## Overview

V17.0 is a complete institutional-grade trading system with realistic expectations and proper validation. Unlike V16.0 which achieved unrealistic Sharpe 20.69 with only 6 stocks, V17.0 is designed for production deployment with proper risk management.

## Target Metrics (Realistic)

| Metric | Target Range | V16 (Overfit) |
|--------|--------------|---------------|
| Sharpe | 1.5 - 3.0 | 20.69 ❌ |
| CAGR | 25% - 50% | 212.8% ❌ |
| MaxDD | -15% to -25% | -1.7% ❌ |
| Universe | 1,000+ | 6 ❌ |

**Red Flags for Overfitting:**
- Sharpe > 5.0
- CAGR > 100%
- MaxDD < -5% (too good)

## System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                     V17.0 TRADING SYSTEM                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐ │
│  │   Universe   │ -> │    Data      │ -> │   Factor Zoo         │ │
│  │   Builder    │    │   Pipeline   │    │   (50+ factors)      │ │
│  │   1,087 sym  │    │   yfinance   │    │   TDA, Momentum      │ │
│  └──────────────┘    └──────────────┘    └──────────────────────┘ │
│         │                   │                      │              │
│         v                   v                      v              │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                    HMM REGIME DETECTOR                      │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐   │   │
│  │  │LowVolTrend│ │HighVolTrnd│ │LowVolMean │ │  Crisis   │   │   │
│  │  │   42.1%   │ │   24.4%   │ │   23.0%   │ │   10.5%   │   │   │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘   │   │
│  └────────────────────────────────────────────────────────────┘   │
│                              │                                    │
│                              v                                    │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                   STRATEGY ROUTER                           │   │
│  │  Regime 0 -> Momentum (50 pos, 4% max, 15% vol)            │   │
│  │  Regime 1 -> TrendFollow (30 pos, 5% max, 20% vol)         │   │
│  │  Regime 2 -> StatArb (40 pos, 3% max, 12% vol)             │   │
│  │  Regime 3 -> Defensive (10 pos, 2% max, 8% vol)            │   │
│  └────────────────────────────────────────────────────────────┘   │
│                              │                                    │
│                              v                                    │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │               WALK-FORWARD BACKTEST ENGINE                  │   │
│  │  • 12-month train, 3-month test, rolling monthly           │   │
│  │  • Transaction costs: 5bps commission + 5-20bps slippage   │   │
│  │  • Position limits: 4% max per position                    │   │
│  │  • Vol targeting: 15% annual                               │   │
│  │  • Max drawdown stop: -20%                                 │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Universe Builder (`v17_universe_builder.py`)
- **Input**: Russell 3000 approximation (1,258 candidates)
- **Filters**: 
  - Price > $5
  - Avg daily $ volume > $1M
  - 20-day lookback
- **Smoothing**: 16/21 days to enter, 5/21 days to exit
- **Output**: 1,087 qualified symbols

### 2. Data Pipeline (`v17_data_pipeline.py`)
- **Source**: yfinance (Polygon.io fallback available)
- **Storage**: Parquet with snappy compression
- **Fields**: date, symbol, open, high, low, close, volume, vwap, return
- **Output**: 561,416 rows (1,087 symbols × 517 days) in 25.5 MB

### 3. HMM Regime Detector (`v17_hmm_regime.py`)
- **Model**: 4-state Gaussian HMM
- **Features**: returns_10d, realized_vol_10d, volume_ratio, trend_strength
- **Training**: 5 years of SPY data
- **States**:
  | State | Name | Return | Vol | Freq |
  |-------|------|--------|-----|------|
  | 0 | LowVolTrend | +1.46% | 9.3% | 42.1% |
  | 1 | HighVolTrend | +2.54% | 18.5% | 24.4% |
  | 2 | LowVolMeanRevert | -1.62% | 15.4% | 23.0% |
  | 3 | Crisis | -2.99% | 29.3% | 10.5% |

### 4. Strategy Router (`v17_strategy_router.py`)
Maps regimes to strategies:

| Regime | Strategy | Max Pos | Position Size | Vol Target |
|--------|----------|---------|---------------|------------|
| 0 | Momentum XSection | 50 | 4% | 15% |
| 1 | Trend Follow | 30 | 5% | 20% |
| 2 | Stat Arb | 40 | 3% | 12% |
| 3 | Defensive | 10 | 2% | 8% |

### 5. Factor Zoo (`v17_factor_zoo.py`)
50 factors across 6 categories:

| Category | Count | Key Factors |
|----------|-------|-------------|
| Momentum | 10 | momentum_12_1, risk_adjusted_momentum |
| Volatility | 10 | realized_vol, downside_vol, skewness |
| Volume | 8 | dollar_volume, amihud_illiquidity |
| Mean Reversion | 8 | zscore_20d, reversal_5d |
| Trend | 8 | ma_cross_50_200, breakout_20d, ADX |
| TDA | 6 | betti_0, betti_1, persistence_entropy |

### 6. Walk-Forward Engine (`v17_walkforward.py`)
- **Train/Test**: 12-month train, 3-month test
- **Rolling**: Monthly with 15 folds
- **Transaction Costs**:
  - Commission: 5 bps per side
  - Slippage: 5-20 bps based on participation rate
- **Risk Limits**:
  - Max position: 5%
  - Max gross exposure: 200%
  - Drawdown stop: -20%

## Files Created

```
v17_universe_builder.py   # Universe construction
v17_data_pipeline.py      # Data fetching & storage
v17_hmm_regime.py         # 4-state HMM regime detection
v17_strategy_router.py    # Regime-to-strategy mapping
v17_factor_zoo.py         # 50+ factor library
v17_walkforward.py        # Walk-forward backtest engine
run_v17_full.py           # Integrated system runner
```

## Cache Files

```
cache/universe/universe_latest.json        # 1,087 symbols
cache/v17_prices/v17_prices_latest.parquet # 561K rows (25.5 MB)
cache/v17_hmm_regime.pkl                   # Trained HMM model
cache/v17_regime_history.parquet           # Regime predictions
cache/v17_factor_sample.parquet            # Factor values
```

## Results

```
results/v17/v17_full_results.json     # Backtest metrics
results/v17/v17_equity_curve.parquet  # Daily equity
results/v17/V17_REPORT.md             # Markdown report
```

## Next Steps

1. **Factor Weight Optimization**: Use cross-validation to find optimal factor weights per regime
2. **Dynamic Rebalancing**: Implement daily/weekly signal updates vs static signals
3. **Sector Neutrality**: Add sector constraints to reduce factor exposure
4. **Risk Parity**: Implement risk parity position sizing
5. **Live Trading**: Connect to Interactive Brokers/Alpaca for paper trading

## Usage

```bash
# Build universe
python v17_universe_builder.py

# Fetch data
python v17_data_pipeline.py

# Train HMM
python v17_hmm_regime.py

# Run full system
python run_v17_full.py
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 17.0.0 | 2026-01-22 | Initial V17 architecture |
| 16.0.0 | 2026-01-21 | Aggressive (overfit) system |
| 15.0.0 | 2026-01-20 | Baseline systematic strategy |

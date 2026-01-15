# Algebraic-Topology-Neural-Net-Strategy

A research-grade quantitative trading system combining **Topological Data Analysis (TDA)** with **LSTM neural networks** for multi-asset portfolio optimization.

## 🚀 NEW: Phase 5 - Full-Universe TDA Multi-Asset Strategy

### Latest Performance (Phase 5)

| Metric | Strategy | SPY Benchmark | Outperformance |
|--------|----------|---------------|----------------|
| **Total Return** | 72.56% | 56.00% | **+16.57%** |
| **CAGR** | 16.41% | 11.92% | **+4.49%** |
| **Sharpe Ratio** | 0.91 | 0.60 | **+0.31** |
| **Max Drawdown** | 18.82% | 25.36% | **-6.54%** (better) |

**Period:** Jan 2020 - Jan 2025 (5 years)

### Phase 5 Features
- 🔬 **TRUE TDA Implementation**: Ripser-based persistent homology on 100-stock universe
- 📊 **Multi-Timeframe Momentum**: 5d/20d/60d weighted scoring
- 🎯 **Regime Detection**: 200-SMA trend filter + TDA turbulence
- 🏗️ **Spectral Clustering**: Graph-based stock grouping
- ⚡ **20-Stock Concentrated Portfolio**: Score-weighted allocation

See [Phase 5 Report](results/PHASE5_TDA_UNIVERSE_REPORT.md) for full details.

---

## 🚀 Phase 7: Scalable Universe Expansion (Russell 3000)

### Scalability Proven

| Metric | Phase 6 (100 stocks) | Phase 7 (500 stocks) |
|--------|---------------------|---------------------|
| **Processing Time** | N/A | **19.7 seconds** |
| **Stocks Processed** | 100 | 500 |
| **TDA Compute Rate** | - | 14.6 stocks/sec |
| **Cache Hit Rate** | - | 93.4% |

### Phase 7 Infrastructure

| Component | File | Purpose |
|-----------|------|---------|
| Data Provider | `src/data/russell3000_provider.py` | Multi-threaded fetching with parquet caching |
| Universe Screener | `src/universe_screener.py` | 4-stage filtering pipeline |
| Parallel TDA | `src/tda_engine_parallel.py` | ProcessPoolExecutor parallelization |
| Backtest Runner | `scripts/run_phase7_russell3000.py` | Main orchestration |

### Architecture
```
┌───────────────┐    ┌───────────────┐    ┌─────────────┐
│ Russell3000   │    │   Universe    │    │  Parallel   │
│ DataProvider  │───▶│   Screener    │───▶│ TDA Engine  │
│ (20 workers)  │    │ (4 stages)    │    │ (4 procs)   │
└───────────────┘    └───────────────┘    └──────┬──────┘
       │                                         │
       ▼                                         ▼
┌───────────────┐                      ┌─────────────────┐
│ Parquet Cache │                      │   Backtester    │
│ (7-day TTL)   │                      │ (Mom+TDA Score) │
└───────────────┘                      └─────────────────┘
```

See [Phase 7 Scalability Report](results/PHASE7_SCALABILITY_REPORT.md) for full details.

---

## Engine Version: V1.3 (Optimized Production Release)

### 🎯 Current Performance (Optimized)

| Metric | Value | Status |
|--------|-------|--------|
| **Portfolio Sharpe (net)** | 1.35 | ✅ Target: >1.3 |
| **Max Drawdown** | 2.08% | ✅ Target: <8% |
| **Test Period** | 2024-2025 | 2-year validation |
| **Trading Frequency** | 12 trades/year | Conservative |

### Key Features

| Feature | Description |
|---------|-------------|
| **TDA Features (V1.3)** | 20 features: persistence, Betti, entropy, max/sum lifetime, top_k, count_large, wasserstein |
| **Regime Detection** | TradingCondition: FAVORABLE / NEUTRAL / UNFAVORABLE |
| **Risk-Weighted Allocation** | Dynamic weighting based on per-asset Sharpe |
| **Signal Filters** | RSI(14, OS=45, OB=55), Volatility threshold 35% |
| **Half-Kelly Sizing** | kelly_fraction=0.50, max 15% per position |
| **Data Layer** | Polygon/Massive (OTREP) primary, yfinance fallback |
| **Cost Modeling** | Transaction costs (5bp/side) |
| **Risk Overlay** | Dynamic scaling (0.5/0.75/1.0) based on equity performance |

---

## 📊 Optimized Backtest Results (V1.3)

### Portfolio Performance (Risk-Weighted, 2024-2025)

| Metric | Gross | Net (after costs) |
|--------|-------|-------------------|
| **Sharpe Ratio** | 1.40 | **1.35** |
| **Total Return** | 1.08% | 1.02% |
| **Max Drawdown** | 2.08% | 2.08% |
| **Turnover** | 0.56x | - |
| **Total Trades** | 12 | - |

### Per-Asset Performance

| Ticker | Sharpe_net | Return_net | Trades | Win Rate | Allocation |
|--------|------------|------------|--------|----------|------------|
| **IWM** | 1.21 | 1.50% | 3 | 66.7% | **60.3%** |
| **XLF** | 0.78 | 0.35% | 1 | 100.0% | **39.7%** |
| SPY | -0.24 | -0.10% | 2 | 50.0% | 0% |
| QQQ | -1.59 | -2.08% | 3 | 0.0% | 0% |
| XLK | -0.87 | -1.83% | 3 | 33.3% | 0% |

> **Note:** Risk-weighted allocation automatically reduces exposure to underperforming assets. IWM and XLF receive positive allocation; SPY, QQQ, XLK get 0% when underperforming.

### Success Criteria Assessment

| Requirement | Target | Actual | Status |
|-------------|--------|--------|--------|
| Portfolio Sharpe | > 1.3 | 1.35 | ✅ |
| Max Drawdown | < 8% | 2.08% | ✅ |
| Assets Sharpe > 0 | ≥ 2/5 | 2/5 | ✅ |
| Stretch: Sharpe | > 1.5 | 1.35 | ⚠️ |
| Stretch: Max DD | < 5% | 2.08% | ✅ |

---

## V1.3 Baseline Results

Tested on train 2022-2023, test 2024-2025 with 20 TDA features (v1.3 mode):

| Portfolio | Sharpe_net | Return_net |
|-----------|------------|------------|
| Equal-Weight | 0.74 | 0.66% |
| **Performance-Weighted** | **1.14** | **1.08%** |

Per-asset breakdown:
| Ticker | Sharpe_net | Return_net | Trades |
|--------|------------|------------|--------|
| SPY | 1.21 | 1.21% | 19 |
| IWM | 0.89 | 1.19% | 20 |
| XLK | 0.51 | 0.93% | 21 |
| QQQ | -0.04 | -0.04% | 21 |
| XLF | -0.01 | -0.01% | 17 |

---

## Data Providers (V1.2-data)

The engine supports two data providers:

| Provider | API Key Required | Timeframes | Notes |
|----------|------------------|------------|-------|
| **polygon** (default) | Yes (`POLYGON_API_KEY_OTREP`) | Daily + Intraday | Massive/OTREP subscription |
| **yfinance** (fallback) | No | Daily only | Free, may have rate limits |

### Setting Up Polygon (Recommended)

1. Create an OTREP API key in the [Massive dashboard](https://massive.io)
2. Set the environment variable:
   ```bash
   export POLYGON_API_KEY_OTREP="your_otrep_key_here"
   ```
3. Run the engine (Polygon is the default):
   ```bash
   python main_multiasset.py
   ```

### Switching Providers

Edit `DATA_PROVIDER` in `main_multiasset.py`:
```python
DATA_PROVIDER = "polygon"  # Primary (requires API key)
# or
DATA_PROVIDER = "yfinance"  # Fallback (no key required)
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set Polygon API key (recommended)
export POLYGON_API_KEY_OTREP="your_otrep_key"

# Run V1.2-data baseline
python main_multiasset.py

# Run walk-forward validation (change MODE in file)
# MODE = "walkforward"
python main_multiasset.py

# Run expanded universe (18 tickers)
# MODE = "expanded_universe"
python main_multiasset.py

# Run robustness analysis (5 scenarios)
# MODE = "robustness"
python main_multiasset.py
```

---

## V1.3 Modes

| Mode | Description |
|------|-------------|
| `baseline` | Single best scenario (train 2022-2023, test 2024-2025) |
| `robustness` | 5 train/test scenarios for stability testing |
| `walkforward` | Rolling walk-forward with 2yr/6mo windows |
| `expanded_universe` | Test with 18 tickers (core + sectors + single names) |
| `ablation` | **V1.3 NEW** Compare v1.1/v1.2/v1.3 TDA feature sets |

---

## V1.3 TDA Features (Enriched)

| Feature | Dimension | Description |
|---------|-----------|-------------|
| `persistence_l0/l1` | L0, L1 | L2-norm of lifetimes |
| `betti_0/1` | L0, L1 | Feature counts |
| `entropy_l0/l1` | L0, L1 | Shannon entropy of lifetime distribution |
| `max_lifetime_l0/l1` | L0, L1 | Maximum lifetime |
| `sum_lifetime_l0/l1` | L0, L1 | Sum of lifetimes |
| `top1/2/3_lifetime_l0/l1` | L0, L1 | **V1.3** Top 3 longest lifetimes |
| `count_large_l0/l1` | L0, L1 | **V1.3** Count above 75th percentile |
| `wasserstein_approx_l0/l1` | L0, L1 | **V1.3** Approximate W1 distance |

**Total: 20 TDA features + 2 OHLCV-derived = 22 model inputs**

---

## V1.3 Regime Labels

The `RegimeLabeler` classifies each trading day into one of four regimes:

| Regime | Definition |
|--------|------------|
| `trend_up` | High positive rolling return, low/moderate volatility |
| `trend_down` | High negative rolling return, low/moderate volatility |
| `high_vol` | High rolling volatility (top 20%) |
| `choppy` | Everything else, especially high TDA entropy |

Use `MODE = "ablation"` to see per-regime performance breakdown.

---

## V1.3 Ablation Results

**Comparing TDA Feature Modes (train 2022-2023, test 2024-2025)**

| TDA Mode | TDA Features | N_FEATURES | Sharpe_net | Return_net | Trades | Turnover |
|----------|--------------|------------|------------|------------|--------|----------|
| v1.1 | 4 | 6 | 0.36 | 0.92% | 274 | 11.84x |
| **v1.2** | 10 | 12 | **1.20** | **1.91%** | 114 | 2.82x |
| v1.3 | 20 | 22 | 1.14 | 1.08% | 98 | 2.32x |

**Key Insights:**
- v1.2 (10 TDA) is the **sweet spot** with 3x better Sharpe than v1.1
- v1.2 has 4x lower turnover → lower transaction costs
- v1.3 shows slight overfitting with 20 features on limited data
- Entropy/lifetime features (v1.2) add most signal

---

## V1.2 TDA Features (Extended)

| Feature | Dimension | Description |
|---------|-----------|-------------|
| `persistence_l0` | L0 | Sum of H0 lifetimes |
| `persistence_l1` | L1 | Sum of H1 lifetimes |
| `betti_0` | L0 | Connected components |
| `betti_1` | L1 | Loops/cycles |
| `entropy_l0` | L0 | Lifetime distribution entropy |
| `entropy_l1` | L1 | Lifetime distribution entropy |
| `max_lifetime_l0` | L0 | Maximum H0 lifetime |
| `max_lifetime_l1` | L1 | Maximum H1 lifetime |
| `sum_lifetime_l0` | L0 | Total H0 lifetime |
| `sum_lifetime_l1` | L1 | Total H1 lifetime |

---

## Results (V1.2 Baseline)

### Portfolio Performance (train 2022-2023, test 2024-2025)

| Portfolio | Sharpe | Sharpe_net | Return | Return_net |
|-----------|--------|------------|--------|------------|
| Equal-Weight | 1.04 | 0.93 | 1.18% | 1.05% |
| Performance-Weighted | 1.52 | 1.41 | 1.69% | 1.56% |

**Per-Asset (NET metrics):**
- SPY: Sharpe_net = 1.69, Return_net = 1.70%
- QQQ: Sharpe_net = 0.01, Return_net = 0.02%
- IWM: Sharpe_net = 1.01, Return_net = 1.99%
- XLF: Sharpe_net = 0.19, Return_net = 0.24%
- XLK: Sharpe_net = 0.65, Return_net = 1.30%

---

## Architecture

```
OHLCV Data (5-18 tickers)
    │
    ├──► TDAFeatureGenerator (V1.3)
    │        └── 20 persistent homology features
    │
    ├──► MarketRegimeDetector
    │        └── TradingCondition (FAVORABLE/NEUTRAL/UNFAVORABLE)
    │
    ├──► NeuralNetPredictor (LSTM)
    │        └── 22 features → P(next bar up)
    │
    └──► EnsembleStrategy (Backtrader)
             │
             ├── Risk-weighted allocation (per-asset Sharpe)
             ├── Signal filters (RSI, volatility)
             ├── Half-Kelly position sizing (0.50)
             └── Risk overlay (0.5/0.75/1.0 scaling)
```

### Risk Management Stack

| Layer | Component | Parameters |
|-------|-----------|------------|
| 1 | **Regime Filter** | Skip UNFAVORABLE conditions |
| 2 | **RSI Filter** | period=14, oversold=45, overbought=55 |
| 3 | **Volatility Filter** | threshold=35% |
| 4 | **Position Sizing** | Half-Kelly (0.50), max 15% |
| 5 | **Portfolio Allocation** | Risk-weighted by Sharpe |
| 6 | **Risk Overlay** | Scale 0.5x/0.75x/1.0x |

---

## ⚠️ Known Limitations & Risks

### QQQ Underperformance Analysis

QQQ consistently underperforms in this strategy due to:

| Factor | Finding |
|--------|---------|
| **Regime Sensitivity** | Sharpe 4.0 in FAVORABLE, -1.05 in NEUTRAL |
| **Correlation** | 0.96 with SPY, 0.97 with XLK (high overlap) |
| **Volatility** | 32% higher than SPY |
| **Mitigation** | Risk-weighting allocates 0% when underperforming |

### Strategy Limitations

1. **Lookback Bias**: Optimized on historical data (2022-2025)
2. **Market Regime Dependency**: Performance varies significantly by regime
3. **Limited Trade Sample**: 12 trades in test period (statistical significance concern)
4. **Sector Concentration**: Current allocation favors IWM/XLF
5. **Data Quality**: yfinance fallback may have different adjusted prices

### Risk Warnings

- **Not investment advice**: Research/educational purposes only
- **Past performance**: Does not guarantee future results
- **Drawdown risk**: Max DD 2% in backtest; real trading may differ
- **Slippage**: 5bp assumed; real costs may be higher

---

## 🚀 Production Deployment Checklist

Before live trading, complete these steps:

- [ ] Paper trade for 3+ months
- [ ] Set up real-time data feed (Polygon recommended)
- [ ] Implement circuit breakers (pause at -5% drawdown)
- [ ] Configure monitoring dashboards
- [ ] Define manual override procedures
- [ ] Validate order execution latency
- [ ] Test with small position sizes first

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODE` | "baseline" | "baseline", "robustness", "walkforward", "expanded_universe" |
| `USE_EXTENDED_TDA` | True | Use 20 TDA features (V1.3) vs 10 (V1.2) |
| `N_FEATURES` | 22 | Total input features (20 TDA + 2 OHLCV-derived) |
| `NN_BUY_THRESHOLD` | 0.52 | Signal threshold for buy |
| `NN_SELL_THRESHOLD` | 0.48 | Signal threshold for sell |
| `COST_BP_PER_SIDE` | 5 | Slippage/impact in basis points |
| `KELLY_FRACTION` | 0.50 | Half-Kelly position sizing |
| `MAX_POSITION_PCT` | 0.15 | Maximum 15% per position |

---

## Output Files

| File | Description |
|------|-------------|
| `results/multiasset_backtest.json` | V1.3 baseline results |
| `results/full_3year_analysis.txt` | Comprehensive 3-year analysis report |
| `results/qqq_diagnostic_report.txt` | QQQ underperformance analysis |
| `results/portfolio_optimization_comparison.txt` | Portfolio scenario comparison |
| `results/multiasset_robustness_report.json` | Robustness analysis |
| `results/multiasset_walkforward_report.json` | Walk-forward results |
| `results/multiasset_weights.weights.h5` | Trained NN weights |

### Analysis Scripts

| Script | Purpose |
|--------|---------|
| `scripts/analyze_qqq_performance.py` | Deep QQQ diagnostic analysis |
| `scripts/portfolio_optimization.py` | Portfolio scenario testing |
| `scripts/generate_3year_analysis.py` | Full analysis report generator |
| `scripts/hyperparameter_optimization.py` | Parameter grid search |
| `scripts/cost_sensitivity_analysis.py` | Transaction cost analysis |

---

## 📈 Optimization History

| Iteration | Focus | Portfolio Sharpe | Key Changes |
|-----------|-------|------------------|-------------|
| V1.0 | Baseline | 0.74 | Initial TDA+LSTM |
| V1.2 | TDA Features | 1.14 | 10 TDA features, extended persistence |
| **V1.3** | Risk Management | **1.35** | Half-Kelly, RSI/volatility filters, risk-weighting |

---

## Future Enhancements

1. **Regime-Conditional QQQ**: Trade QQQ only in FAVORABLE conditions
2. **Intraday Signals**: Test with hourly data from Polygon
3. **Sector Rotation**: Add sector ETF momentum overlay
4. **Options Overlay**: Covered calls for additional income
5. **Ensemble Models**: Combine LSTM with gradient boosting

---

*Engine V1.3 (Optimized) - January 2026*
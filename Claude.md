# Claude.md - TDA + Neural Net Trading Bot Progress

## Project Status: ✅ ENGINE V1.3 COMPLETE

### Overview
Multi-asset ensemble trading bot combining:
- **TDA (V1.3 Enriched)**: 20 features (persistence, entropy, top_k, count_large, wasserstein)
- **Regime Labeling (V1.3)**: trend_up, trend_down, high_vol, choppy
- **Feature Ablation (V1.3)**: Compare v1.1/v1.2/v1.3 TDA feature sets
- **Data Layer (V1.2-data)**: Polygon/Massive primary, yfinance fallback, intraday-ready
- **LSTM Neural Network**: Direction prediction using OHLCV + 20 TDA features (22 total)
- **Backtrader**: Strategy execution and backtesting
- **Walk-Forward Validation**: Rolling 2-year train / 6-month test
- **Expanded Universe**: Core 5 ETFs + sector ETFs + large-cap single names (18 tickers)
- **Cost-Aware Metrics**: Net returns and Sharpe after transaction costs
- **Risk Overlay**: Dynamic scaling based on equal-weight portfolio performance

---

## V1.3 Baseline Results

**Train: 2022-2023, Test: 2024-2025, TDA Mode: v1.3 (20 features)**

| Portfolio | Sharpe_net | Return_net | Trades | Turnover |
|-----------|------------|------------|--------|----------|
| Equal-Weight | 0.74 | 0.66% | 98 | 2.33x |
| **Performance-Weighted** | **1.14** | **1.08%** | 98 | 2.32x |

---

## Engine V1.3 Features

| Feature | Description |
|---------|-------------|
| **TDA (Enriched)** | 20 features: persistence, Betti, entropy, max/sum lifetime, top_k, count_large, wasserstein |
| **Regime Labels** | trend_up, trend_down, high_vol, choppy (returns + vol + TDA entropy) |
| **Feature Ablation** | Compare v1.1 (4 TDA), v1.2 (10 TDA), v1.3 (20 TDA) |
| **Data Provider** | Polygon (OTREP key) primary, yfinance fallback |
| **Timeframe Support** | Daily + intraday-ready (60m, 30m, 15m, 5m, 1m) |
| **Multi-Asset** | Core: 5 ETFs | Expanded: 18 tickers |
| **Walk-Forward** | 2yr train, 6mo test, rolling validation |
| **Cost Model** | 5bp/side slippage + 0.1% commission |
| **Risk Overlay** | 0.5/0.75/1.0 scaling with cash allocation |

---

## V1.3 TDA Features (Enriched)

| Feature | Dimension | New in V1.3 |
|---------|-----------|-------------|
| `persistence_l0/l1` | L0, L1 | No |
| `betti_0/1` | L0, L1 | No |
| `entropy_l0/l1` | L0, L1 | No |
| `max_lifetime_l0/l1` | L0, L1 | No |
| `sum_lifetime_l0/l1` | L0, L1 | No |
| `top1/2/3_lifetime_l0/l1` | L0, L1 | ✅ Yes |
| `count_large_l0/l1` | L0, L1 | ✅ Yes |
| `wasserstein_approx_l0/l1` | L0, L1 | ✅ Yes |

**Total: 20 TDA + 2 OHLCV = 22 model input features**

---

## V1.3 Modes

| Mode | Description |
|------|-------------|
| `baseline` | Single best scenario (train 2022-2023, test 2024-2025) |
| `robustness` | 5 train/test scenarios for stability testing |
| `walkforward` | Rolling walk-forward with 2yr/6mo windows |
| `expanded_universe` | Test with 18 tickers |
| `ablation` | **NEW** Compare v1.1/v1.2/v1.3 TDA feature sets |

---

## V1.3 Ablation Results

**Comparing TDA Feature Modes (train 2022-2023, test 2024-2025)**

| TDA Mode | TDA Features | N_FEATURES | Sharpe_net | Return_net | Trades | Turnover |
|----------|--------------|------------|------------|------------|--------|----------|
| v1.1 | 4 | 6 | 0.36 | 0.92% | 274 | 11.84x |
| **v1.2** | 10 | 12 | **1.20** | **1.91%** | 114 | 2.82x |
| v1.3 | 20 | 22 | 1.14 | 1.08% | 98 | 2.32x |

**Key Insights:**
- v1.2 is the **sweet spot** with 3x better Sharpe than v1.1 (0.36 → 1.20)
- v1.2 has 4x lower turnover (11.84x → 2.82x) = lower transaction costs
- v1.3 slight overfitting with 20 features on limited training data
- Entropy/lifetime features (v1.2) add most signal; top_k/wasserstein (v1.3) add noise

---

## Phase Completion Status

| Phase | Module | Status | Notes |
|-------|--------|--------|-------|
| 1 | `/src/tda_features.py` | ✅ V1.3 | Enriched features (top_k, count_large, wasserstein) |
| 2 | `/src/nn_predictor.py` | ✅ V1.2 | Dynamic TDA feature handling |
| 3 | `/src/ensemble_strategy.py` | ✅ V1.1 | Turnover tracking, cost metrics |
| 4 | `/src/regime_labeler.py` | ✅ V1.3 | Regime labels for analysis |
| 5 | `/main.py` | ✅ COMPLETE | Single-asset SPY backtest |
| 6 | `/main_multiasset.py` | ✅ V1.3 | Enriched TDA, ablation, regimes |
| 7 | `/src/data/` | ✅ V1.2-data | Polygon + yfinance unified API |
| 8 | `requirements.txt` | ✅ COMPLETE | Dependencies pinned |

---

## ✅ V1.2 BASELINE RESULTS

### Portfolio Performance (train 2022-2023, test 2024-2025)

| Portfolio | Sharpe | Sharpe_net | Return | Return_net |
|-----------|--------|------------|--------|------------|
| Equal-Weight | 1.04 | 0.93 | 1.18% | 1.05% |
| Performance-Weighted | 1.52 | 1.41 | 1.69% | 1.56% |

### Per-Asset NET Metrics

| Ticker | Sharpe_net | Return_net | Trades |
|--------|------------|------------|--------|
| SPY | 1.69 | 1.70% | 23 |
| QQQ | 0.01 | 0.02% | 22 |
| IWM | 1.01 | 1.99% | 17 |
| XLF | 0.19 | 0.24% | 21 |
| XLK | 0.65 | 1.30% | 21 |

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

## Architecture (V1.2)

```
OHLCV Data (5-18 tickers)
    │
    ├──► TDAFeatureGenerator (V1.2)
    │        │
    │        ├── takens_embedding() → Point cloud
    │        ├── compute_persistence_features() → 10 features
    │        │      └── persistence, betti, entropy, max/sum lifetime
    │        └── generate_features() → Rolling window TDA
    │
    ├──► DataPreprocessor (V1.2)
    │        │
    │        ├── _normalize_features() → Dynamic TDA handling
    │        └── _create_sliding_windows() → (batch, 15, 12) sequences
    │
    └──► NeuralNetPredictor
             │
             ├── LSTM(64) → Dense → Dense(1, sigmoid)
             └── Output: P(next bar up)

Trading Logic (EnsembleStrategy):
    - Turbulence Index = sqrt(L0² + L1²) normalized
    - BUY: NN signal > 0.52 AND no position
    - SELL: NN signal < 0.48 AND has position
```

---

## Testing Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Test individual modules
python src/tda_features.py
python src/nn_predictor.py

# Run V1.2 baseline
python main_multiasset.py

# Run walk-forward (change MODE in file)
# MODE = "walkforward"
python main_multiasset.py

# Run expanded universe
# MODE = "expanded_universe"
python main_multiasset.py
```

---

## Output Metrics

| Metric | Description |
|--------|-------------|
| `sharpe_ratio` | Gross Sharpe (before extra costs) |
| `sharpe_ratio_net` | Net Sharpe (after 5bp/side slippage) |
| `total_return` | Gross return |
| `total_return_net` | Net return after costs |
| `turnover` | Total notional traded / initial cash |
| `risk_scale` | 0.5 (risk-off) / 0.75 (moderate) / 1.0 (risk-on) |
| `cash_weight` | Portion held in cash |

---

## File Structure

```
/workspaces/Algebraic-Topology-Neural-Net-Strategy/
├── main.py                    # Single-asset SPY backtest
├── main_multiasset.py         # V1.2-data multi-asset engine
├── requirements.txt           # Dependencies
├── Claude.md                  # This file
├── README.md                  # Project docs
├── src/
│   ├── tda_features.py        # V1.2 TDA feature generator (10 features)
│   ├── nn_predictor.py        # V1.2 LSTM + dynamic TDA handling
│   ├── ensemble_strategy.py   # Backtrader strategy + cost tracking
│   └── data/                  # V1.2-data unified data layer
│       ├── __init__.py        # Package exports
│       ├── polygon_client.py  # REST client for Polygon.io
│       └── data_provider.py   # Unified get_ohlcv_data() API
├── results/
│   ├── multiasset_backtest.json           # V1.2 baseline
│   ├── multiasset_robustness_report.json  # Robustness analysis
│   ├── multiasset_walkforward_report.json # Walk-forward results
│   ├── expanded_universe_backtest.json    # Expanded universe
│   ├── multiasset_weights.weights.h5      # Trained NN
│   └── backtest_results.json              # Legacy single-asset
```

---

## Version History

| Version | Date | Features |
|---------|------|----------|
| V1.0 | Jan 2026 | Single-asset SPY, basic TDA |
| V1.1 | Jan 2026 | Multi-asset, cost-aware, risk overlay |
| V1.2 | Jan 2026 | Walk-forward, extended TDA, expanded universe |
| V1.2-data | Jan 2026 | Polygon/Massive data layer, intraday-ready |

---

*Engine V1.2-data Completed: January 14, 2026*
*Multi-Provider Data Layer | Walk-Forward | Richer TDA | Expanded Universe*

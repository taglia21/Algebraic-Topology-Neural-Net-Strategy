# Algebraic-Topology-Neural-Net-Strategy

A research-grade quantitative trading system combining **Topological Data Analysis (TDA)** with **LSTM neural networks** for multi-asset portfolio optimization.

## Engine Version: V1.3 (Feature & TDA Enrichment)

### Key Features

| Feature | Description |
|---------|-------------|
| **TDA Features (V1.3)** | 20 features: persistence, Betti, entropy, max/sum lifetime, top_k, count_large, wasserstein |
| **Regime Labeling (V1.3)** | trend_up, trend_down, high_vol, choppy (based on returns, volatility, TDA entropy) |
| **Feature Ablation (V1.3)** | Compare v1.1 (4), v1.2 (10), v1.3 (20) TDA feature sets |
| **Data Layer (V1.2-data)** | Polygon/Massive (OTREP) primary, yfinance fallback, intraday-ready |
| **LSTM Predictor** | 64-unit LSTM with entropy-penalty loss, 22 input features |
| **Multi-Asset** | Core: 5 ETFs, Expanded: 18 tickers (incl. sector ETFs, large-caps) |
| **Walk-Forward (V1.2)** | Rolling 2yr train / 6mo test, non-overlapping validation |
| **Cost Modeling** | Transaction costs (5bp/side slippage + 0.1% commission) |
| **Risk Overlay** | Dynamic scaling (0.5/0.75/1.0) based on EQ portfolio performance |

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
    ├──► TDAFeatureGenerator (V1.2)
    │        └── 10 persistent homology features
    │
    ├──► NeuralNetPredictor (LSTM)
    │        └── 12 features → P(next bar up)
    │
    └──► EnsembleStrategy (Backtrader)
             │
             ├── Turbulence index for regime detection
             ├── Performance-weighted allocation
             └── Risk overlay (0.5/0.75/1.0 scaling)
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODE` | "baseline" | "baseline", "robustness", "walkforward", "expanded_universe" |
| `USE_EXTENDED_TDA` | True | Use 10 TDA features (V1.2) vs 4 (V1.1) |
| `N_FEATURES` | 12 | Total input features (10 TDA + 2 OHLCV-derived) |
| `NN_BUY_THRESHOLD` | 0.52 | Signal threshold for buy |
| `NN_SELL_THRESHOLD` | 0.48 | Signal threshold for sell |
| `COST_BP_PER_SIDE` | 5 | Slippage/impact in basis points |

---

## Output Files

| File | Description |
|------|-------------|
| `results/multiasset_backtest.json` | V1.2 baseline results |
| `results/multiasset_robustness_report.json` | Robustness analysis |
| `results/multiasset_walkforward_report.json` | Walk-forward results |
| `results/expanded_universe_backtest.json` | Expanded universe results |
| `results/multiasset_weights.weights.h5` | Trained NN weights |

---

*Engine V1.2 - January 2026*
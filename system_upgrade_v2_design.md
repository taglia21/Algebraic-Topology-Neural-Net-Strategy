# System Upgrade V2.0 - Architecture Design Document

## Executive Summary

This document outlines the upgrade from V1.3 to V2.0 of the Algebraic Topology Neural Net Trading Strategy. The upgrade introduces 7 critical enhancements targeting improved prediction accuracy, faster learning convergence, enhanced market structure detection, and reduced execution costs.

**Current V1.3 Performance:**
- Sharpe Ratio: 1.35
- Max Drawdown: 2.08%
- CAGR: 16.41%
- Training Time: ~80 seconds

**Target V2.0 Performance:**
- Sharpe Ratio: ≥1.50 (+11%)
- Max Drawdown: ≤1.50% (-28%)
- CAGR: ≥18% (+10%)
- Training Time: ≤120 seconds

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           V2.0 ENHANCED TRADING ENGINE                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        DATA INGESTION LAYER                              │   │
│  │  • Price Data (OHLCV)  • Order Book Data  • VIX Term Structure          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│          ┌───────────────────────────┼───────────────────────────┐              │
│          ▼                           ▼                           ▼              │
│  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────────────┐ │
│  │  TRANSFORMER      │   │  PERSISTENT       │   │  ORDER FLOW               │ │
│  │  PREDICTOR        │   │  LAPLACIAN        │   │  ANALYZER                 │ │
│  │  ────────────────│   │  ────────────────│   │  ─────────────────────────│ │
│  │  • 8-head attn   │   │  • Graph Laplacian│   │  • Bid-Ask Spread         │ │
│  │  • 512 dims      │   │  • PL Eigenvalues │   │  • Order Book Depth       │ │
│  │  • Positional enc│   │  • 12 new features│   │  • Trade Imbalance        │ │
│  │  • 10 input feats│   │  • L0/L1 analysis │   │  • Liquidity Score        │ │
│  └─────────┬─────────┘   └─────────┬─────────┘   └─────────────┬─────────────┘ │
│            │                       │                           │                │
│            └───────────────────────┼───────────────────────────┘                │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     ENSEMBLE REGIME DETECTOR                             │   │
│  │  ────────────────────────────────────────────────────────────────────── │   │
│  │  HMM (0.5 weight) + GMM (0.3 weight) + Agglomerative (0.2 weight)       │   │
│  │  • 2/3 Consensus Required  • VIX Term Structure Signal                   │   │
│  │  • Regime Confidence Score (0-1)  • Reduced False Switches (-35%)        │   │
│  └─────────────────────────────────────────┬───────────────────────────────┘   │
│                                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    SOFT ACTOR-CRITIC (SAC) AGENT                         │   │
│  │  ────────────────────────────────────────────────────────────────────── │   │
│  │  • Twin Q-Networks            • Prioritized Experience Replay (PER)     │   │
│  │  • Entropy Regularization     • P(i) = p_i^α / Σ p_j^α, α=0.6          │   │
│  │  • Auto Temperature Tuning    • Importance Sampling β: 0.4 → 1.0        │   │
│  │  • Continuous Actions [0,2]   • TD-error Priority: |δ| + ε              │   │
│  └─────────────────────────────────────────┬───────────────────────────────┘   │
│                                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    DYNAMIC ACTION SPACE MANAGER                          │   │
│  │  ────────────────────────────────────────────────────────────────────── │   │
│  │  • Volatility-Scaled Bounds: bounds = base * (1 / sqrt(vol_20d))        │   │
│  │  • VIX > 25: max_position = 0.75x  │  VIX < 15: max_position = 1.5x     │   │
│  │  • Real-time adjustment every rebalance cycle                            │   │
│  └─────────────────────────────────────────┬───────────────────────────────┘   │
│                                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                       PORTFOLIO EXECUTION ENGINE                         │   │
│  │  ────────────────────────────────────────────────────────────────────── │   │
│  │  • Risk Parity Allocation     • Liquidity-Aware Entry Timing            │   │
│  │  • Max 15% Single Position    • 5% Intraday DD Stop-Loss                │   │
│  │  • 50% Cash in Risk-Off       • 3-Day Loss Emergency Shutdown           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 Transformer Predictor (`src/ml/transformer_predictor.py`)

**Purpose:** Replace LSTM with attention-based architecture for better long-range dependency capture.

**Architecture:**
```
Input: (batch, seq_len=20, features=10)
    │
    ▼
┌─────────────────────────────────┐
│  Positional Encoding            │  Sinusoidal encoding for time awareness
│  PE(pos, 2i) = sin(pos/10000^(2i/d))
│  PE(pos, 2i+1) = cos(pos/10000^(2i/d))
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Input Embedding (Linear)       │  10 → 512 dims
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Transformer Encoder Block ×3   │
│  ─────────────────────────────  │
│  • Multi-Head Attention (8 heads, 64 dims/head)
│  • Layer Normalization          │
│  • Feed-Forward (512 → 2048 → 512)
│  • Residual Connections         │
│  • Dropout (0.1)               │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Global Average Pooling         │  (batch, seq, 512) → (batch, 512)
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Classification Head            │  512 → 128 → 1 (sigmoid)
└─────────────────────────────────┘
    │
    ▼
Output: Probability of positive return (0-1)
```

**Key Parameters:**
- `d_model`: 512
- `n_heads`: 8
- `n_layers`: 3
- `d_ff`: 2048
- `dropout`: 0.1
- `max_seq_len`: 100

**Expected Improvement:** +5-10% prediction accuracy, better regime transition detection

---

### 2.2 Prioritized Experience Replay (`src/ml/sac_agent.py`)

**Purpose:** Sample high-value experiences more frequently to accelerate learning.

**Algorithm:**
```python
# Priority calculation
p_i = |TD_error_i| + ε  # ε = 1e-6

# Sampling probability
P(i) = p_i^α / Σ_j p_j^α  # α = 0.6

# Importance sampling weight (bias correction)
w_i = (N * P(i))^(-β) / max_j(w_j)  # β anneals 0.4 → 1.0

# Update priority after learning
p_i = |new_TD_error_i| + ε
```

**Data Structure:** Sum-tree for O(log n) sampling and updates

**Expected Improvement:** 40% faster convergence, 15% higher returns

---

### 2.3 Persistent Laplacian TDA (`src/tda_v2/persistent_laplacian.py`)

**Purpose:** Capture geometric information missed by standard persistent homology.

**Mathematical Foundation:**

**Graph Laplacian:**
```
L = D - A

Where:
- A = adjacency matrix (from correlation matrix)
- D = degree matrix (diagonal, D_ii = Σ_j A_ij)
- L = Laplacian matrix
```

**Persistent Laplacian:**
```
L_k^(i,j) = δ_{k}^(i,j) (δ_{k}^(i,j))* + (δ_{k-1}^(i,j))* δ_{k-1}^(i,j)

Where:
- δ_k = k-th boundary operator at filtration level
- (i,j) = birth-death pair in persistence diagram
```

**12 New Features:**
| Feature | Description |
|---------|-------------|
| L0_eig_mean | Mean eigenvalue of L0 (connected components) |
| L0_eig_std | Std dev of L0 eigenvalues |
| L0_eig_max | Max eigenvalue of L0 |
| L0_eig_entropy | Spectral entropy of L0 |
| L1_eig_mean | Mean eigenvalue of L1 (loops) |
| L1_eig_std | Std dev of L1 eigenvalues |
| L1_eig_max | Max eigenvalue of L1 |
| L1_eig_entropy | Spectral entropy of L1 |
| Betti0_curve_area | Area under Betti-0 curve |
| Betti1_curve_area | Area under Betti-1 curve |
| PL_spectral_gap | Gap between 1st and 2nd eigenvalue |
| PL_algebraic_conn | Fiedler eigenvalue (algebraic connectivity) |

**Expected Improvement:** Detect regime shifts 2-3 days earlier

---

### 2.4 Soft Actor-Critic Agent (`src/ml/sac_agent.py`)

**Purpose:** Replace discrete Q-learning with continuous, entropy-regularized policy.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                   SOFT ACTOR-CRITIC                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────┐    ┌───────────────────────────────┐│
│  │   Actor Network   │    │   Twin Critic Networks        ││
│  │   π(a|s)          │    │   Q1(s,a), Q2(s,a)           ││
│  │   ──────────────  │    │   ───────────────────────    ││
│  │   State → 256     │    │   State+Action → 256         ││
│  │   256 → 256       │    │   256 → 256                   ││
│  │   256 → μ, log(σ) │    │   256 → Q-value              ││
│  │   Sample: a ~ N(μ,σ)   │   Use min(Q1, Q2)            ││
│  └───────────────────┘    └───────────────────────────────┘│
│            │                           │                    │
│            ▼                           ▼                    │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Entropy Temperature (α)                    ││
│  │              ────────────────────────                   ││
│  │              Auto-tuned to target: -dim(action_space)   ││
│  │              α = exp(log_α)                             ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  Objective: max_π E[Σ γ^t (r_t + α H(π(·|s_t)))]           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Continuous Action Space:** Position sizing multiplier ∈ [0.0, 2.0]

**Expected Improvement:** 20% better risk-adjusted returns

---

### 2.5 Dynamic Action Space (`src/trading/v2_enhanced_engine.py`)

**Purpose:** Adapt position sizing bounds to current volatility regime.

**Volatility Scaling:**
```python
def compute_action_bounds(vol_20d, vix_level):
    base_max = 1.0
    
    # Scale by inverse sqrt of volatility
    vol_scale = 1.0 / np.sqrt(vol_20d + 1e-6)
    vol_scale = np.clip(vol_scale, 0.5, 2.0)
    
    # VIX regime adjustment
    if vix_level > 25:  # High fear
        regime_scale = 0.75
    elif vix_level < 15:  # Complacency
        regime_scale = 1.5
    else:  # Normal
        regime_scale = 1.0
    
    max_position = base_max * vol_scale * regime_scale
    return np.clip(max_position, 0.25, 2.0)
```

**Expected Improvement:** 25% reduction in max drawdown

---

### 2.6 Ensemble Regime Detector (`src/trading/regime_ensemble.py`)

**Purpose:** Robust regime classification with reduced false switches.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│              ENSEMBLE REGIME DETECTOR                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Returns, Volatility, Correlation, VIX               │
│                    │                                        │
│    ┌───────────────┼───────────────┬───────────────┐       │
│    ▼               ▼               ▼               ▼       │
│  ┌─────┐       ┌─────┐       ┌─────────────┐   ┌───────┐  │
│  │ HMM │       │ GMM │       │ Agglomerative│   │ VIX   │  │
│  │     │       │     │       │ Clustering  │   │ Term  │  │
│  │ 0.5 │       │ 0.3 │       │     0.2     │   │ Struct│  │
│  └──┬──┘       └──┬──┘       └──────┬──────┘   └───┬───┘  │
│     │             │                  │              │       │
│     └─────────────┼──────────────────┘              │       │
│                   ▼                                 │       │
│  ┌─────────────────────────────────────────────────┼───────┐│
│  │           CONSENSUS VOTING (2/3 Required)       │       ││
│  │           ──────────────────────────────────    │       ││
│  │           Regime: {bull, bear, neutral, crisis} │       ││
│  └─────────────────────────────────────────────────┼───────┘│
│                          │                         │        │
│                          ▼                         ▼        │
│  ┌────────────────────────────────────────────────────────┐│
│  │              CONFIDENCE SCORE (0-1)                    ││
│  │              ──────────────────────                    ││
│  │              conf = (agreement_count / 3) * model_conf ││
│  │              + vix_signal_strength * 0.2               ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**VIX Term Structure Signal:**
```python
# Contango (VIX < VIX3M): bullish signal
# Backwardation (VIX > VIX3M): bearish signal
vix_slope = (vix_3m - vix_spot) / vix_spot
```

**Expected Improvement:** 35% fewer false regime switches

---

### 2.7 Order Flow Analyzer (`src/microstructure/order_flow_analyzer.py`)

**Purpose:** Incorporate market microstructure for better execution.

**Metrics:**
```python
# Liquidity Score
L = volume / (spread * 100) * depth_ratio

Where:
- volume = 20-period avg daily volume
- spread = (ask - bid) / mid_price
- depth_ratio = bid_depth / ask_depth (imbalance)

# Trade Size Imbalance
imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)

# Execution Quality
if L < percentile_20(L_history):
    delay_entry = True
    wait_for_liquidity = True
```

**Expected Improvement:** 30 bps slippage reduction

---

## 3. Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                        │
└──────────────────────────────────────────────────────────────────────────────┘

                        ┌─────────────────────┐
                        │   Market Data API   │
                        │   (Alpaca/Yahoo)    │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
            ┌───────────┐  ┌───────────┐  ┌───────────┐
            │   OHLCV   │  │   Quotes  │  │  VIX Data │
            │   1467    │  │   Top 100 │  │  Spot+3M  │
            │   stocks  │  │   bid/ask │  │           │
            └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
                  │              │              │
                  ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FEATURE ENGINEERING                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐  │
│  │  Transformer Features │  │  PL Features (12)    │  │  Order Flow (5)  │  │
│  │  ────────────────────│  │  ────────────────────│  │  ──────────────  │  │
│  │  • Log returns       │  │  • L0_eig_mean       │  │  • Spread        │  │
│  │  • RSI_14            │  │  • L0_eig_std        │  │  • Depth ratio   │  │
│  │  • MACD              │  │  • L0_eig_max        │  │  • Imbalance     │  │
│  │  • BB_position       │  │  • L0_eig_entropy    │  │  • Liquidity     │  │
│  │  • Mom_5/10/20/50    │  │  • L1_* (4 features) │  │  • Volume        │  │
│  │  • HL_range          │  │  • Betti curves (2)  │  │                  │  │
│  │  • Vol_change        │  │  • Spectral gap      │  │                  │  │
│  │                      │  │  • Algebraic conn    │  │                  │  │
│  └──────────┬───────────┘  └──────────┬───────────┘  └────────┬─────────┘  │
│             │                         │                        │            │
└─────────────┼─────────────────────────┼────────────────────────┼────────────┘
              │                         │                        │
              └─────────────────────────┼────────────────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MODEL INFERENCE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    Transformer          Regime Ensemble           SAC Agent                 │
│    Predictions    +     Confidence Score    →    Position Sizing            │
│    (1467 stocks)        (4 regimes)              (continuous 0-2)           │
│                                                                             │
│    Direction prob       Regime weights:          Action bounds scaled       │
│    + confidence         bull/bear/neutral        by volatility + VIX        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PORTFOLIO CONSTRUCTION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    1. Rank stocks by Transformer confidence × predicted return             │
│    2. Apply risk parity weights (inverse volatility)                        │
│    3. Scale by SAC position sizing output                                   │
│    4. Apply dynamic action space bounds                                     │
│    5. Check liquidity constraints (delay if L < 20th percentile)           │
│    6. Enforce: max 15% single position, 50% cash in risk_off               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ORDER EXECUTION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    Alpaca Paper Trading API                                                 │
│    ─────────────────────────                                                │
│    • Market orders (liquid stocks)                                          │
│    • Limit orders (illiquid, spread > 0.5%)                                 │
│    • Smart routing for best execution                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Integration Points

### 4.1 File Structure

```
src/
├── ml/
│   ├── __init__.py
│   ├── transformer_predictor.py    # NEW: Attention-based predictor
│   └── sac_agent.py                # NEW: SAC with PER
├── tda_v2/
│   ├── __init__.py
│   └── persistent_laplacian.py     # NEW: PL features
├── microstructure/
│   ├── __init__.py
│   └── order_flow_analyzer.py      # NEW: Market microstructure
├── trading/
│   ├── adaptive_learning_engine.py # EXISTING: V1.3 (unchanged)
│   ├── regime_ensemble.py          # NEW: Multi-model regime detection
│   ├── v2_enhanced_engine.py       # NEW: V2.0 orchestrator
│   └── tda_paper_trading_engine.py # EXISTING: Paper trading
scripts/
├── migrate_tf_to_pytorch.py        # NEW: Model conversion
├── run_v2_backtest_ablation.py     # NEW: Ablation study
└── deploy_tda_trading.py           # EXISTING: Deployment
tests/
├── test_transformer.py             # NEW
├── test_sac_per.py                 # NEW
├── test_persistent_laplacian.py    # NEW
├── test_regime_ensemble.py         # NEW
├── test_order_flow.py              # NEW
└── test_v2_integration.py          # NEW
```

### 4.2 Backward Compatibility

```python
# In v2_enhanced_engine.py

class V2EnhancedEngine:
    def __init__(self, use_v2: bool = True, **kwargs):
        """
        Initialize enhanced engine with version toggle.
        
        Args:
            use_v2: If True, use V2 components. If False, fall back to V1.3.
        """
        self.use_v2 = use_v2
        
        if use_v2:
            self.predictor = TransformerPredictor(...)
            self.agent = SACAgent(...)
            self.tda = PersistentLaplacian(...)
            self.regime_detector = EnsembleRegimeDetector(...)
            self.order_flow = OrderFlowAnalyzer(...)
        else:
            # Fall back to V1.3 components
            from .adaptive_learning_engine import AdaptiveLearningEngine
            self.legacy_engine = AdaptiveLearningEngine(...)
```

---

## 5. Performance Estimates

### 5.1 Expected Improvements by Component

| Component | Metric | V1.3 | V2.0 Target | Improvement |
|-----------|--------|------|-------------|-------------|
| Transformer | Accuracy | 80.8% | ≥82% | +1.5% |
| PER | Convergence | 200 eps | ≤120 eps | -40% |
| Persistent Laplacian | Early Detection | 0 days | 2-3 days | N/A |
| SAC | Risk-Adj Return | baseline | +20% | +20% |
| Dynamic Actions | Max DD | 2.08% | ≤1.56% | -25% |
| Ensemble Regime | False Switches | baseline | -35% | -35% |
| Order Flow | Slippage | 5 bps | ≤3.5 bps | -30% |

### 5.2 Combined System Performance

| Metric | V1.3 | V2.0 Target | Confidence |
|--------|------|-------------|------------|
| Sharpe Ratio | 1.35 | ≥1.50 | High |
| Max Drawdown | 2.08% | ≤1.50% | Medium |
| CAGR | 16.41% | ≥18% | Medium |
| Training Time | 80s | ≤120s | High |
| Win Rate | 53% | ≥55% | Medium |
| Profit Factor | 1.45 | ≥1.60 | Medium |

---

## 6. Risk Mitigation

### 6.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Transformer overfitting | Medium | High | Dropout, early stopping, validation split |
| SAC instability | Low | Medium | Twin Q-networks, entropy regularization |
| PL computation timeout | Medium | Low | Sampling, caching, timeout limits |
| Memory overflow | Low | High | Batch processing, gradient accumulation |
| Regime misclassification | Medium | Medium | Consensus voting, confidence thresholds |

### 6.2 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Production deployment failure | Low | High | Blue-green deployment, instant rollback |
| Model degradation | Medium | Medium | Continuous monitoring, auto-retraining |
| Data quality issues | Low | Medium | Input validation, anomaly detection |
| API rate limits | Medium | Low | Caching, request throttling |

### 6.3 Emergency Procedures

```python
# Hard stops in v2_enhanced_engine.py

EMERGENCY_SHUTDOWN_CONDITIONS = {
    'max_dd_intraday': 0.05,      # 5% intraday drawdown
    'consecutive_losses': 3,       # 3 consecutive losing days
    'max_single_position': 0.15,   # 15% single position limit
    'min_portfolio_value': 0.85,   # 85% of starting capital
}

def check_emergency_conditions(self) -> bool:
    """
    Check if emergency shutdown is required.
    Returns True if trading should halt.
    """
    ...
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

| Module | Test File | Coverage Target |
|--------|-----------|-----------------|
| Transformer | test_transformer.py | ≥90% |
| SAC + PER | test_sac_per.py | ≥90% |
| Persistent Laplacian | test_persistent_laplacian.py | ≥85% |
| Ensemble Regime | test_regime_ensemble.py | ≥90% |
| Order Flow | test_order_flow.py | ≥85% |
| V2 Integration | test_v2_integration.py | ≥80% |

### 7.2 Backtesting Validation

```
Training Period: 2022-01-01 to 2023-12-31 (2 years)
Testing Period: 2024-01-01 to 2025-01-20 (1 year)

Ablation Study:
1. V1.3 baseline
2. V1.3 + Transformer only
3. V1.3 + SAC+PER only
4. V1.3 + Persistent Laplacian only
5. V1.3 + Ensemble Regime only
6. V1.3 + Order Flow only
7. V1.3 + Dynamic Actions only
8. Full V2.0 (all components)
```

### 7.3 Paper Trading Validation

- 48-hour burn-in period
- 7-day observation period
- All metrics logged to Discord
- Daily comparison vs V1.3

---

## 8. Deployment Plan

### 8.1 Phases

| Phase | Duration | Activities |
|-------|----------|------------|
| 1. Development | 2-3 days | Implement all components |
| 2. Unit Testing | 1 day | Run test suite, fix bugs |
| 3. Backtesting | 1 day | Run ablation study |
| 4. Integration | 0.5 day | Combine components, integration tests |
| 5. Paper Trading | 7 days | Live paper trading validation |
| 6. Production | Ongoing | Monitor, tune, maintain |

### 8.2 Rollback Procedure

```bash
# If V2 fails, instant rollback to V1.3

# 1. Stop V2 service
ssh root@134.209.40.95 "systemctl stop tda-trading.service"

# 2. Edit service to use V1.3
ssh root@134.209.40.95 "sed -i 's/v2_enhanced_engine/tda_paper_trading_engine/g' /etc/systemd/system/tda-trading.service"

# 3. Reload and start
ssh root@134.209.40.95 "systemctl daemon-reload && systemctl start tda-trading.service"

# 4. Verify
ssh root@134.209.40.95 "systemctl status tda-trading.service"
```

---

## 9. Success Criteria

### 9.1 Mandatory (Must Pass All)

- [ ] All unit tests pass with ≥90% coverage
- [ ] Transformer achieves ≥80% accuracy on test set
- [ ] SAC+PER converges in ≤120 episodes
- [ ] V2.0 backtest: Sharpe ≥1.50, Max DD ≤1.50%
- [ ] No overfitting: train-test Sharpe delta <0.15
- [ ] Production deployment: zero errors, full compatibility
- [ ] 48-hour paper trading: zero crashes

### 9.2 Optional (Nice to Have)

- [ ] Persistent Laplacian detects 2022 regime shift early
- [ ] Ensemble regime detector: <5% false positive rate
- [ ] Order flow reduces slippage to <3 bps
- [ ] CAGR ≥18%
- [ ] Win rate ≥55%

---

## 10. Appendix

### A. References

1. Vaswani et al., "Attention Is All You Need" (2017)
2. Haarnoja et al., "Soft Actor-Critic" (2018)
3. Schaul et al., "Prioritized Experience Replay" (2016)
4. Wang et al., "Persistent Laplacian" (2020)
5. Cont & de Larrard, "Order Book Dynamics" (2013)

### B. Hyperparameter Summary

```yaml
transformer:
  d_model: 512
  n_heads: 8
  n_layers: 3
  d_ff: 2048
  dropout: 0.1
  learning_rate: 0.0001

sac:
  hidden_dims: [256, 256]
  gamma: 0.99
  tau: 0.005
  alpha_init: 0.2
  learning_rate: 0.0003

per:
  alpha: 0.6
  beta_start: 0.4
  beta_end: 1.0
  epsilon: 1e-6
  buffer_size: 100000

regime_ensemble:
  hmm_weight: 0.5
  gmm_weight: 0.3
  agg_weight: 0.2
  n_states: 4
  consensus_threshold: 2

dynamic_actions:
  base_max: 1.0
  vix_high: 25
  vix_low: 15
  vol_scale_min: 0.5
  vol_scale_max: 2.0
```

---

*Document Version: 2.0.0*
*Last Updated: January 20, 2026*
*Author: GitHub Copilot*

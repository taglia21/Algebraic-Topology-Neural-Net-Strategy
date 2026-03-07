# ATNN Quant Powerhouse

A production-grade quantitative equity trading system built for retail traders who want institutional-quality alpha generation.

## Performance

| Metric | Value |
|---|---|
| Total Return (2023-2025) | +304.7% |
| Annualized Return | +40.9% |
| Sharpe Ratio | 1.13 |
| Sortino Ratio | 1.28 |
| Max Drawdown | -34.3% |
| Calmar Ratio | 1.19 |
| Alpha (vs SPY) | +22.9% |
| Win Rate | 67.6% |
| Profit Factor | 17.2 |

Backtest: 15-stock diversified universe, $100K initial capital, 7 bps slippage, $0.005/share commission. No look-ahead bias.

## Architecture

```
main.py → SystemOrchestrator
             │
             ├── DataManager (yfinance/Alpaca)
             ├── RegimeDetector (HMM + VIX + ADX)
             ├── SignalGenerator
             │    ├── StatArbStrategy    (Kalman filter pairs, OU model)
             │    ├── MomentumStrategy   (Cross-sectional, sector-neutral)
             │    └── FactorModelStrategy (Quality, Value, Low-Vol, Momentum)
             ├── MLPipeline (optional)
             │    ├── FeatureEngine (60+ features)
             │    ├── GradientBoostModel × 3 horizons (1d, 5d, 20d)
             │    ├── Ridge meta-learner
             │    └── PSI drift detection
             ├── RiskManager
             │    ├── Position sizing (Kelly criterion)
             │    ├── Drawdown gates
             │    ├── Sector exposure caps
             │    └── Correlation limits
             ├── ExecutionManager (150% gross exposure cap)
             │    ├── SimulatedBroker (backtest)
             │    └── AlpacaBroker (live/paper)
             └── Production safety
                  ├── KillSwitch + CircuitBreaker
                  ├── MarketCalendar (NYSE hours/holidays)
                  └── Position Reconciler
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run backtest (no ML, ~6 min)
python run_backtest.py

# Run backtest with ML meta-learner (~15 min)
python run_backtest.py --ml

# Custom date range
python run_backtest.py --start 2020-01-01 --end 2025-12-31

# Full CLI
python main.py --mode backtest --start 2023-01-01 --end 2025-12-31 --ml

# Paper trading (requires Alpaca credentials in .env)
python main.py --mode paper

# Live trading
python main.py --mode live
```

## Project Structure

```
├── core/                   # Infrastructure
│   ├── config.py           # Centralized dataclass configuration
│   ├── logger.py           # Structured JSON trade/event logger
│   ├── regime_detector.py  # HMM regime detection (Bull/Bear/Sideways)
│   ├── risk_manager.py     # Position sizing, drawdown gates, correlation checks
│   ├── kill_switch.py      # Emergency halt + circuit breakers
│   ├── market_hours.py     # NYSE calendar, holidays, early closes
│   └── reconciliation.py   # Position reconciliation engine
│
├── data/                   # Market data
│   ├── data_manager.py     # Unified data abstraction layer
│   ├── market_data.py      # Alpaca + yfinance providers
│   └── cache.py            # Thread-safe TTL in-memory cache
│
├── equities/               # Trading engine
│   ├── models.py           # Signal, Order, Fill, Position, PortfolioState
│   ├── signal_generator.py # Multi-strategy signal orchestrator
│   ├── execution.py        # Broker interface, SimulatedBroker, ExecutionManager
│   ├── alpaca_broker.py    # Production Alpaca broker adapter
│   └── strategies/
│       ├── stat_arb.py     # Kalman-filter pairs trading with OU dynamics
│       ├── momentum.py     # Cross-sectional residual momentum
│       └── factor_model.py # Multi-factor alpha (Quality, Value, Low-Vol, Mom)
│
├── ml/                     # Machine learning
│   ├── feature_engine.py   # 60+ technical/fundamental/cross-sectional features
│   ├── pipeline.py         # Training orchestrator + Ridge meta-learner
│   ├── validation.py       # Walk-forward + CPCV validation
│   └── models/
│       └── gradient_boost.py  # LightGBM multi-horizon return predictor
│
├── backtest/               # Backtesting
│   ├── backtester.py       # Event-driven bar-by-bar replay engine
│   └── metrics.py          # Performance analytics (Sharpe, alpha, etc.)
│
├── tests/                  # Test suite (89 tests)
│   ├── test_core_modules.py
│   └── test_production_modules.py
│
├── main.py                 # CLI entry point (backtest/paper/live)
├── run_backtest.py         # Quick standalone backtest runner
├── Dockerfile              # Multi-stage production build
├── .env.example            # Environment variable template
└── requirements.txt        # Python dependencies
```

## Strategies

### Statistical Arbitrage (`stat_arb.py`)
- Engle-Granger cointegration testing across all symbol pairs
- Kalman filter for adaptive hedge ratio estimation
- Ornstein-Uhlenbeck model for mean-reversion dynamics
- Z-score entry/exit with stop-loss at 3.5σ

### Cross-Sectional Momentum (`momentum.py`)
- 12-1 month lookback with 1-month skip (reversal avoidance)
- Sector-neutral construction (longs and shorts within sectors)
- Inverse-volatility weighting for risk parity
- Vectorized rolling OLS for residual momentum

### Multi-Factor Alpha (`factor_model.py`)
- Quality: ROE, debt-to-equity, earnings stability
- Value: P/E, P/B, EV/EBITDA via yfinance fundamentals
- Low Volatility: realized vol, downside deviation
- Momentum: 12-1 return with trend confirmation

## ML Pipeline

The ML system is designed as an alpha overlay — base strategies work without it, ML adds incremental signal.

1. **Feature Engine** — 60+ features: returns, volatility, RSI, MACD, Bollinger, OBV, Hurst exponent, VIX beta, cross-sectional rank
2. **LightGBM** — Three models predict 1-day, 5-day, and 20-day forward returns (classification mode: up/down)
3. **Meta-Learner** — Ridge regression combines base model scores + regime state into a single composite signal
4. **Drift Detection** — Population Stability Index (PSI) triggers automatic retraining when feature distributions shift

## Risk Management

| Parameter | Default | Description |
|---|---|---|
| Max Position | 20% | Single name cap as % of equity |
| Max Sector | 35% | Sector exposure cap |
| Max Drawdown Halt | -30% | Trading halts at this drawdown |
| Max Drawdown Reduce | -20% | Exposure reduced to 60% at this level |
| Daily Loss Limit | -3% | Intraday loss triggers circuit breaker |
| Max Correlation | 0.85 | Pairwise correlation cap |
| Kelly Fraction | 0.5 | Half-Kelly position sizing |
| Gross Exposure Cap | 150% | Hard cap prevents excessive leverage |

## Deployment

### Docker
```bash
docker build -t atnn-quant .
docker run --env-file .env atnn-quant python main.py --mode paper
```

### Environment Variables
Copy `.env.example` to `.env` and fill in your Alpaca credentials:
```bash
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
SYSTEM_MODE=paper
```

### DigitalOcean / VPS
```bash
# Recommended: 2+ vCPU, 4GB+ RAM (ML requires more memory)
git clone <repo>
cd Algebraic-Topology-Neural-Net-Strategy
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
python main.py --mode paper
```

## Testing

```bash
# Run all tests (89 tests, <2 seconds)
python -m pytest tests/ -v

# Run core module tests only
python -m pytest tests/test_core_modules.py -v

# Run production module tests only
python -m pytest tests/test_production_modules.py -v
```

## Key Design Decisions

- **Same code path**: Backtest, paper, and live all run identical strategy logic. Only the data source and broker differ.
- **No look-ahead**: The backtester enforces strict no-look-ahead — history windows never include future data.
- **Regime-aware**: HMM regime detection adjusts signal weights (bullish amplifies momentum, bearish reduces exposure).
- **150% gross exposure cap**: Prevents the system from becoming a de-facto leveraged fund.
- **Anti-overfitting**: Walk-forward validation, CPCV cross-validation, PSI drift detection, and Ridge regularization.

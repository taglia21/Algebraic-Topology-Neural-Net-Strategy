# ATNN Quant Powerhouse

A production-grade quantitative equity trading system built for retail traders who want institutional-quality alpha generation.

![Promotion Gates Validation](https://github.com/taglia21/Algebraic-Topology-Neural-Net-Strategy/actions/workflows/promotion-gates.yml/badge.svg?branch=main)
[![Governance](https://img.shields.io/badge/governance-promotion--gates-blue)](docs/CI_AND_GOVERNANCE_SETUP.md)

---

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

## ETF Operations Dashboard

Launch the professional monitoring dashboard for readiness, gate status, slippage, and recent fills:

```bash
streamlit run scripts/etf_dashboard.py
```

What you get:
- Promotion gate status and pass/fail breakdown
- Equity state snapshot (last equity, peak, drawdown)
- Reconciliation and scheduler state
- Slippage trend and latest execution cost
- Recent order/fill events from `logs/trades_*.jsonl`
- One-click preflight probe (runs `python -m etf.main --mode preflight`)

### Paper Trading Fallback (When IBKR Is Down)

If IBKR Gateway/TWS is unreachable, you can still run paper cycles using the
local simulated paper broker:

```bash
python -m etf.main --mode paper --execute --allow-gate-bypass --paper-sim-fallback
```

State is persisted to `.etf_telemetry/paper_sim_account.json` and execution
telemetry to `.etf_telemetry/slippage.jsonl`.

### Real IBKR Paper Trading (For Performance Test Period)

If you want trades to appear in your IBKR paper account, run the engine on the
machine/network that can reach TWS/IB Gateway API and use the real-paper
launcher below (this path does **not** use simulated fallback):

```bash
# Required: host where TWS/IB Gateway paper API is listening
export IBKR_HOST=127.0.0.1

# Optional: set explicitly if you know it.
# If omitted, launcher probes 7497 then 4002 and picks the first reachable.
# export IBKR_PORT=7497        # TWS paper
# export IBKR_PORT=4002        # IB Gateway paper

# Optional: custom client id
export ETF_IBKR_CLIENT_ID=7

# Optional: bypass promotion gate during controlled paper collection
export ETF_ALLOW_GATE_BYPASS=1

# Optional: run one cycle immediately (still market-hours constrained)
export ETF_ONCE=1

scripts/start_etf_ibkr_paper.sh
```

Notes:
- You do **not** need a new Alpaca key for ETF IBKR paper routing.
- This launcher fails fast if IBKR API is unreachable, so you do not get false
     "paper trading" progress from simulation mode.
- To run continuously, unset `ETF_ONCE` and leave the script running.

### Cloud Droplet + Local IBKR Bridge

If the bot runs in a cloud container but TWS/Gateway runs on your laptop,
`127.0.0.1` in the cloud does **not** point to your laptop. You must bridge
the two endpoints.

Use an SSH reverse tunnel from your local IBKR machine to the droplet:

```bash
# Run on the local machine where TWS/Gateway is running.
# Maps droplet 127.0.0.1:4002 -> local 127.0.0.1:7497
scripts/open_ibkr_reverse_tunnel.sh ubuntu@YOUR_DROPLET 4002 7497
```

Then, on the droplet/container, run the real-paper launcher:

```bash
export IBKR_HOST=127.0.0.1
export IBKR_PORT=4002
export ETF_ALLOW_GATE_BYPASS=1
scripts/start_etf_ibkr_paper.sh
```

This routes *real* paper orders to your IBKR account while keeping strategy
runtime in the cloud.

## Promotion Gates And PR Workflow

Use explicit gate checks before moving between research, paper, and live:

```bash
# Validate paper->live gate using evidence JSON
python scripts/check_promotion_gates.py \
     --gate paper_to_live \
     --input templates/promotion_gate_evidence.paper_to_live.example.json \
     --report-out artifacts/promotion_gate_report.paper_to_live.json
```

Exit code semantics:
- `0`: gate passed
- `2`: gate failed

Governance documentation:
- `docs/PR_WORKFLOW.md` for PR-first workflow and recommended branch protection settings.

## Key Design Decisions

- **Same code path**: Backtest, paper, and live all run identical strategy logic. Only the data source and broker differ.
- **No look-ahead**: The backtester enforces strict no-look-ahead — history windows never include future data.
- **Regime-aware**: HMM regime detection adjusts signal weights (bullish amplifies momentum, bearish reduces exposure).
- **150% gross exposure cap**: Prevents the system from becoming a de-facto leveraged fund.
- **Anti-overfitting**: Walk-forward validation, CPCV cross-validation, PSI drift detection, and Ridge regularization.

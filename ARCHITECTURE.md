# VRP Alpha Engine — Architecture

## Philosophy
Strip everything to economics-first. One strategy with documented, persistent edge.
No cargo-cult ML. No 4 strategies fighting each other. No 75% cash drag.

## Strategy: Systematic SPX Put Credit Spreads (VRP Harvesting)

### Why This Works
- Implied volatility > realized volatility ~95% of years (CBOE data, 20+ year history)
- CBOE PutWrite Index (PUT) returns ~9.3% annualized vs SPX ~7.8% with LOWER volatility
- Defined risk: max loss = spread width - credit received
- SPX = cash-settled, European-style, Section 1256 tax treatment
- IBKR margin for spreads = spread width - credit (capital efficient)

### Trade Rules (Optimized via 972-Combo Grid Search)
1. **Entry**: Sell SPX put credit spreads, 30-45 DTE
   - Short leg: -0.15 delta (~85% probability of profit, richer premium)
   - Long leg: 15 points below short leg ($1,500 max risk per spread)
   - Minimum credit: 8% of spread width ($120+)
   - Dynamic width: adjusts based on account equity; widens in high-VIX
2. **Exit**:
   - Take profit at 50% of max credit (primary)
   - Take profit at 75% if within 14 DTE
   - Roll or close at 7 DTE if ITM
   - Hard stop: close if spread reaches 3x credit received
   - Force-close 3 days before expiry
3. **Sizing**:
   - Up to 50% of account per trade
   - Max 2 concurrent positions (optimal per grid search)
   - Scale with VIX regime: 0.75x in low vol, 1.0x standard, 1.5x elevated
4. **VIX Regime Filter**:
   - VIX < 12: No trades (premium too thin, gamma > theta)
   - VIX 12-14: Reduced sizing (0.75x)
   - VIX 14-20: Standard sizing (1.0x)
   - VIX 20-35: Elevated sizing (1.5x), wider spreads
   - VIX > 35: No trades (tail risk)

### Backtest Results (2020-01-01 to 2025-12-30)
Starting capital: $10,000

| Metric | VRP Engine | SPX Buy & Hold |
|--------|-----------|----------------|
| Total Return | +6,392% | +112% |
| Annual Return | +101% | ~16% |
| Sharpe Ratio | 1.15 | 0.47 |
| Sortino Ratio | 1.36 | — |
| Max Drawdown | -85% | -34% |
| Calmar Ratio | 1.19 | — |
| Alpha | +32.4% | — |
| Win Rate | 87.3% | — |
| Profit Factor | 1.74 | — |
| Total Trades | 323 | — |

### Performance Notes
- **46x speedup**: Black-Scholes uses pure-math `erfc` instead of scipy.stats.norm (~0.9s per 5-year backtest)
- **Dynamic spread width**: Automatically narrows after drawdowns, widens when account grows or VIX elevated
- **Drawdown recovery**: System survives COVID (-59%) and tariff crash (-81%) by resizing dynamically
- **Grid search**: 972 parameter combinations tested; delta=-0.15, width=15, SL=3.0x found optimal

### For $10,000 Account
- Spread width: 15 points ($1,500 max risk per contract)
- Dynamic sizing: 2-6 contracts depending on equity
- Max concurrent: 2 spreads
- Expected: ~3-6 trades per month

## File Structure
```
config.py          — All configuration (IBKR, strategy params, risk limits)
broker.py          — IBKR broker interface via ib_async + SimulatedBroker
strategy.py        — VRP strategy: VIXRegimeClassifier, StrikeSelector, PositionManager, VRPStrategy
risk.py            — Portfolio risk management (drawdown halt, daily loss, greeks limits)
backtest.py        — Options backtest engine using yfinance SPX/VIX + Black-Scholes
main.py            — CLI entry point for backtest/paper/live modes
utils.py           — Black-Scholes greeks (pure-math, no scipy), IV solver, date helpers
tests/test_vrp.py  — 33 unit tests
```

## What We Removed (and Why)
- 4 equity strategies (stat_arb, momentum, mean_reversion, factor_model) → zero alpha
- ML pipeline (LightGBM, CUSUM, meta-labeler, HRP) → cargo cult on insufficient data
- 50-stock universe → SPX index only
- Alpaca broker → IBKR (better data, options support, lower commissions)
- Regime detector (HMM) → Simple VIX level check (more robust, fewer parameters)
- Signal generator aggregation → One strategy, one signal
- 21,000 lines → ~2,000 lines
- scipy dependency for BS pricing → pure-math erfc/exp (46x faster)

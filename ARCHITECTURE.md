# Architecture — Single Source of Truth

## Production Entry Point

```
run_bot.py  →  run_v28_production.py  (THE canonical entry point)
```

**Everything else is a library or deprecated.**

---

## Signal Flow (ONE path)

```
EQUITY_UNIVERSE (51 symbols, BANNED_SYMBOLS filtered)
        │
        ▼
┌─ EquityEngine.run_cycle() ────────────────────────────┐
│                                                        │
│  1. Circuit Breaker (RiskGuardian)                     │
│     └─ halt / liquidate if drawdown > limit            │
│                                                        │
│  2. Regime Detection (is_bullish_regime)               │
│     └─ bull=1.0x, neutral=0.7x, bear=0.4x (skip)      │
│                                                        │
│  3. Per-symbol loop:                                   │
│     ├─ BANNED_SYMBOLS check                            │
│     ├─ Freefall filter (−8% in 5 bars → block)         │
│     ├─ Death-cross filter (SMA50 < SMA200 → block)     │
│     ├─ Volume confirmation (≥1.5× 20-period avg)       │
│     ├─ Correlation check (≤0.7 vs holdings)            │
│     │                                                  │
│     ├─ EnhancedTradingEngine.analyze(symbol)           │
│     │   ├─ Multi-timeframe (5m, 15m, 1h, 1D)          │
│     │   ├─ Sentiment (NLP news scoring)                │
│     │   ├─ Medallion math (Hurst, O-U)                 │
│     │   ├─ CAPM + GARCH volatility                     │
│     │   ├─ Signal Aggregator + Adaptive Ensemble       │
│     │   └─ → TradeDecision (signal, conf, qty, SL/TP)  │
│     │                                                  │
│     ├─ Sector cap (30% max per sector)                 │
│     ├─ RiskGuardian.compute_safe_position_size()       │
│     │   └─ ATR, Kelly, regime_scale, half-Kelly cap    │
│     ├─ Max 8% single position                          │
│     └─ Turnover gate (15% daily cap)                   │
│                                                        │
│  4. _execute_equity_trade()                            │
│     └─ LIMIT order + bracket (SL + TP)                 │
│     └─ NEVER market orders                             │
│                                                        │
│  5. _monitor_equity_positions()                        │
│     ├─ Hard stop-loss at −5% (always fires)            │
│     ├─ Trailing stop (activate +4%, trail 40%)         │
│     ├─ Take-profit at +10%                             │
│     └─ Min-hold: 6 bars before soft exit               │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## Module Ownership

| Concern | Module | Owner |
|---------|--------|-------|
| **Signal generation** | `src/enhanced_trading_engine.py` | `EnhancedTradingEngine.analyze()` |
| **Position sizing** | `risk_guardian.py` | `RiskGuardian.compute_safe_position_size()` |
| **Order submission** | `src/trading/alpaca_client.py` | `AlpacaClient.submit_order()` |
| **Universe filter** | `config/universe.py` + inline | `BANNED_SYMBOLS`, freefall, death-cross |
| **Regime detection** | `src/risk/regime_filter.py` | `is_bullish_regime()` |
| **Circuit breaker** | `src/risk/trading_gate.py` | `check_trading_allowed()` |
| **Sector caps** | `src/risk/sector_caps.py` | `sector_allows_trade()` |
| **Options** | `src/options/autonomous_engine.py` | `AutonomousTradingEngine` |
| **Improved MR/momentum** | `strategy_engine.py` | `StrategyEngine` (library, not entry) |
| **Pairs trading** | `pair_finder.py` | `PairFinder` (library, not entry) |
| **Factor exposure** | `src/factor_monitor.py` | `FactorMonitor` (library, not entry) |

---

## DEPRECATED — Do NOT run in production

| File | Reason |
|------|--------|
| `unified_trader.py` | Replaced by `run_v28_production.py` |
| `profit_trader.py` | Legacy entry point |
| `smart_trader.py` | Legacy entry point |
| `continuous_trader.py` | Legacy entry point |
| `continuous_tradier.py` | Legacy entry point (wrong broker) |
| `aggressive_trader.py` | Legacy entry point |
| `auto_trader.py` | Legacy entry point |
| `paper_trading_runner.py` | Legacy entry point |
| `live_trader.py` | Legacy entry point |
| `alpaca_trader.py` | Legacy entry point |

---

## Hard Constraints

- **LIMIT orders only** — MARKET orders are NEVER used
- **BANNED_SYMBOLS**: `{BBBY, SIVB, FRC, COIN}`
- **Max portfolio beta**: 2.5
- **Max single position**: 8% of portfolio
- **Max sector exposure**: 30%
- **Daily turnover cap**: 15% of equity
- **Min hold**: 6 scan cycles before soft exit
- **Half-Kelly cap**: position sizing capped at 0.5× Kelly

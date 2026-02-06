
# CRITICAL SYSTEM OVERHAUL - ALL PHASES MANDATORY

## TASK
Wire all orphaned quantitative modules into the live trading pipeline, implement missing quant models, create a unified production runner, and ensure BOTH the equity engine AND options engine are actively generating and executing trades.

This codebase is 102,056 lines of Python. Roughly 90 percent of the advanced quantitative code is ORPHANED - built but never imported or called by any trading engine. Zero options trades have EVER been placed. The equity engine stopped trading 3 days ago. The systemd service references run_v28_production.py which DOES NOT EXIST. Fix everything.

## CODEBASE ARCHITECTURE

### PRODUCTION ENTRY POINT (BROKEN)
- deploy/v28_trading_bot.service ExecStart calls run_v28_production.py --mode=live - THIS FILE DOES NOT EXIST
- _archived_versions/v28_production_system.py exists but is archived
- WorkingDirectory=/opt/trading-bot on DigitalOcean droplet

### EQUITY ENGINE (partially working - last trade Feb 3)
- src/trading/paper_trading_engine.py - PaperTradingEngine class
- src/enhanced_trading_engine.py - imports ONLY RiskManager + PositionSizer, uses basic multi-timeframe + sentiment scoring

### OPTIONS ENGINE (never placed a single trade)
- src/alpaca_options_engine.py (624 lines) - main entry at line 607
- src/options/autonomous_engine.py (924 lines) - run_forever() at line 184
- src/options/signal_generator.py (478 lines)
- src/options/contract_resolver.py (716 lines) - resolves OCC symbols
- src/options/regime_detector.py - GaussianHMM (ONLY regime detector connected)
- src/options/greeks_engine.py, iv_analyzer.py, volatility_surface.py
- src/options/utils/black_scholes.py - BSM pricing, all 5 Greeks, IV

### ORPHANED MODULES (NEVER IMPORTED BY ANY TRADING ENGINE)

1. ManifoldRegimeDetector (src/options/manifold_regime_detector.py, 1242 lines)
   - Riemannian geometry, Christoffel symbols, geodesic ODE solving via scipy.integrate.odeint
   - Classes: VolatilitySurface3D, GeodesicPathTracker, RegimeClassifier, ManifoldRegimeDetector
   - PROBLEM: autonomous_engine.py imports RegimeDetector from .regime_detector NOT ManifoldRegimeDetector

2. ML Module (src/ml/, 5427 lines total) - ZERO imports from any engine
   - sac_agent.py (1074 lines): SumTree, PrioritizedReplayBuffer, Experience, MLP, SAC RL agent
   - transformer_predictor.py (691 lines): TransformerEncoderBlock(d_model=512,n_heads=8), TransformerPredictorModel
   - gradient_boost_ensemble.py (917 lines): EnsembleConfig, BaseModel(ABC), XGBoostModel(BaseModel)
   - stacked_ensemble.py (551 lines): StackedEnsemble with fit(X,y), predict(X), predict_with_confidence(X)
   - continuous_learner.py (716 lines): ContinuousLearnerConfig, DriftDetector (Kolmogorov-Smirnov)
   - pomdp_controller.py, v25_adaptive_allocator.py

3. Optimization (src/optimization/): bayesian_tuner.py, walk_forward_optimizer.py
4. Regime (src/regime/): hierarchical_controller.py

### MODELS NOT IMPLEMENTED (BUILD FROM SCRATCH)
- CAPM, Merton Jump Diffusion, GARCH(1,1), Monte Carlo Option Pricing
- Heston Stochastic Volatility, Cox-Ross-Rubinstein Binomial Tree, Dupire Local Vol

## EXECUTION PLAN - ALL PHASES MANDATORY

### PHASE 1: CREATE run_v28_production.py at project root
- Accept --mode=live|paper arg
- Initialize BOTH equity and options engines concurrently via asyncio
- Market hours check (9:45AM-3:45PM ET, pre-market analysis 9:00AM)
- Graceful shutdown SIGTERM/SIGINT, load .env, log to file and stdout

### PHASE 2: BUILD MISSING QUANT MODELS in src/quant_models/
a) capm.py - beta regression vs SPY, expected_return = rf + beta*(mkt-rf)
b) merton_jump_diffusion.py - BSM + Poisson jumps, calibrate from return kurtosis
c) garch.py - Full GARCH(1,1) with MLE, multi-step vol forecast
d) monte_carlo_pricer.py - GBM paths, antithetic variates, control variates
e) heston_model.py - Stochastic vol, characteristic function pricing
f) crr_binomial.py - American option pricing, 100+ step tree
g) dupire_local_vol.py - Local vol from market prices

### PHASE 3: CREATE src/signal_aggregator.py
- Collect signals from ALL models (ML, manifold, GARCH, CAPM, etc)
- Each model outputs: signal(-1 to 1), confidence(0 to 1), model_name
- Weighted voting ensemble with regime-dependent weights
- ManifoldRegimeDetector determines regime, weights adjust accordingly
- Minimum confidence threshold 0.6 (configurable)

### PHASE 4: WIRE ALL ORPHANED MODULES
4a) Wire ManifoldRegimeDetector into autonomous_engine.py alongside existing HMM RegimeDetector
4b) Wire ML module (StackedEnsemble, TransformerPredictor, XGBoost) into enhanced_trading_engine.py AND autonomous_engine.py
4c) Wire ContinuousLearner - update after each trade, drift detection triggers retraining
4d) Wire BayesianTuner - optimize params during non-market hours
4e) Wire GARCH into vol forecasting - replace static vol with GARCH forecasts
4f) Wire CAPM into equity engine - screen stocks by expected return
4g) Wire Monte Carlo + Heston into options pricing for mispriced option detection

### PHASE 5: OPTIONS ENGINE - MAKE IT TRADE
- Trading loop every 5 minutes during market hours
- LOWER confidence thresholds for paper trading (we want trades flowing)
- Implement: covered calls, cash-secured puts, vertical spreads, iron condors based on regime
- Log EVERY signal with full reasoning, Discord webhook for executions

### PHASE 6: EQUITY ENGINE UPGRADE
- Replace basic multi-timeframe scoring with signal_aggregator output
- CAPM expected returns for stock screening, GARCH vol for position sizing
- ML ensemble confidence for trade gating

### PHASE 7: TESTING AND DEPLOYMENT
- Unit tests for all new quant models
- Integration test: full signal pipeline
- Update requirements-v28.txt
- git add, commit, push

## CRITICAL RULES
1. Do NOT create stub/placeholder implementations. Every model must have REAL math.
2. Every module must be IMPORTED and CALLED in the live trading pipeline.
3. The options engine MUST be capable of placing real paper trades via Alpaca.
4. Test everything. Run pytest before committing.
5. Do NOT break existing functionality.
6. Use async/await consistently for I/O operations.

"""
Elite Options Trading Engine - Quick Start Guide
================================================

This guide shows how to use all 5 enhanced modules in your trading strategy.
"""

import asyncio
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)

# ============================================================================
# EXAMPLE 1: Market Regime Detection
# ============================================================================

async def example_regime_detection():
    """Detect current market regime and get optimal strategy weights."""
    from src.options.regime_detector import RegimeDetector
    
    detector = RegimeDetector()
    
    # Fit HMM on historical data (do this once at startup)
    await detector.fit()
    
    # Detect current regime
    regime_state = await detector.detect_current_regime()
    
    print(f"\nCurrent Regime: {regime_state.current_regime.value}")
    print(f"Confidence: {regime_state.confidence:.1%}")
    print(f"VIX: {regime_state.features['vix']:.2f}")
    
    # Get optimal strategy weights for this regime
    weights = detector.get_strategy_weights(regime_state.current_regime)
    print(f"\nStrategy Weights:")
    for strategy, weight in weights.items():
        print(f"  {strategy}: {weight:.1%}")
    
    return detector, regime_state


# ============================================================================
# EXAMPLE 2: Portfolio Risk Monitoring
# ============================================================================

async def example_risk_monitoring():
    """Monitor portfolio correlation and concentration risk."""
    from src.options.correlation_manager import CorrelationManager, Position
    
    manager = CorrelationManager()
    
    # Define your current positions
    positions = [
        Position(
            symbol="AAPL", quantity=10, entry_price=2.5, current_price=3.0,
            strategy_type="credit_spread", delta=0.30, gamma=0.05,
            theta=-0.10, vega=0.15, sector="Technology"
        ),
        Position(
            symbol="SPY", quantity=-5, entry_price=5.0, current_price=4.5,
            strategy_type="iron_condor", delta=-0.10, gamma=0.02,
            theta=-0.15, vega=0.20, sector="Index"
        ),
    ]
    
    # Check correlation
    corr_matrix = await manager.build_correlation_matrix(positions)
    print(f"\nCorrelation Matrix:\n{corr_matrix}")
    
    # Check portfolio Greeks
    greeks = await manager.calculate_portfolio_greeks(positions)
    print(f"\nPortfolio Greeks:")
    print(f"  Delta: {greeks.total_delta:.2f}")
    print(f"  Theta: {greeks.total_theta:.2f}")
    print(f"  Bias: {greeks.net_directional_bias}")
    
    # Check concentration risk
    alerts = manager.detect_concentration_risk(
        positions=positions,
        portfolio_value=100000.0,
        correlation_matrix=corr_matrix
    )
    
    if alerts:
        print(f"\n⚠ Concentration Alerts:")
        for alert in alerts:
            print(f"  {alert.severity.upper()}: {alert.message}")
    else:
        print("\n✓ No concentration risks")
    
    # Get hedge recommendations
    hedges = await manager.get_hedge_recommendations(positions, greeks, max_delta=5.0)
    if hedges:
        print(f"\nHedge Recommendations:")
        for hedge in hedges:
            print(f"  {hedge.action} {hedge.symbol} - {hedge.reasoning}")
    
    return manager


# ============================================================================
# EXAMPLE 3: Dynamic Strategy Weights
# ============================================================================

async def example_weight_optimization():
    """Optimize strategy weights based on performance."""
    from src.options.regime_detector import RegimeDetector
    from src.options.weight_optimizer import DynamicWeightOptimizer
    
    # Initialize
    detector = RegimeDetector()
    await detector.fit()
    
    optimizer = DynamicWeightOptimizer(
        strategies=["iv_rank", "theta_decay", "mean_reversion", "delta_hedging"],
        regime_detector=detector
    )
    
    # Record some trade returns
    optimizer.apply_bayesian_update("iv_rank", 0.05)  # 5% profit
    optimizer.apply_bayesian_update("theta_decay", 0.03)  # 3% profit
    optimizer.apply_bayesian_update("mean_reversion", -0.02)  # 2% loss
    optimizer.apply_bayesian_update("delta_hedging", 0.01)  # 1% profit
    
    # Calculate Sharpe ratios
    sharpe_ratios = await optimizer.calculate_strategy_sharpe()
    print(f"\nStrategy Sharpe Ratios:")
    for strategy, sharpe in sharpe_ratios.items():
        print(f"  {strategy}: {sharpe:.2f}")
    
    # Get current regime
    regime_state = await detector.detect_current_regime()
    
    # Optimize weights
    weights = optimizer.optimize_weights(regime_state.current_regime, sharpe_ratios)
    print(f"\nOptimized Weights:")
    for strategy, weight in weights.items():
        print(f"  {strategy}: {weight:.1%}")
    
    # Rebalance if needed
    new_weights = await optimizer.rebalance(regime_state.current_regime)
    
    return optimizer


# ============================================================================
# EXAMPLE 4: Volatility Surface Analysis
# ============================================================================

async def example_vol_surface():
    """Analyze IV surface for arbitrage opportunities."""
    from src.options.volatility_surface import VolatilitySurfaceEngine
    
    engine = VolatilitySurfaceEngine(min_dte=7, max_dte=60)
    
    # Build surface (will fetch option chain)
    try:
        surface = await engine.build_iv_surface("SPY")
        print(f"\nIV Surface:")
        print(f"  Strikes: {len(surface.strikes)}")
        print(f"  Expirations: {len(surface.expirations)}")
        print(f"  Spot: ${surface.spot_price:.2f}")
        
        # Fit SVI model
        svi_params = engine.fit_svi_model(surface)
        print(f"\nSVI Parameters:")
        print(f"  a (level): {svi_params.a:.4f}")
        print(f"  ρ (skew): {svi_params.rho:.2f}")
        
        # Detect anomalies
        anomalies = await engine.detect_anomalies(surface)
        print(f"\nIV Anomalies: {len(anomalies)}")
        for anomaly in anomalies[:3]:  # Show top 3
            print(f"  {anomaly.anomaly_type}: {anomaly.description}")
        
        # Generate arbitrage signals
        arb_signals = await engine.generate_arb_signals(anomalies, surface)
        print(f"\nArbitrage Signals: {len(arb_signals)}")
        for signal in arb_signals[:3]:
            print(f"  {signal.signal_type}: edge={signal.expected_edge_vol:.2%}")
    
    except Exception as e:
        print(f"Vol surface analysis failed: {e}")
    
    return engine


# ============================================================================
# EXAMPLE 5: Cointegration Pairs Trading
# ============================================================================

async def example_pairs_trading():
    """Find cointegrated pairs and generate signals."""
    from src.options.cointegration_engine import CointegrationEngine
    
    engine = CointegrationEngine(lookback_days=252)
    
    # Find cointegrated pairs (example with tech stocks)
    symbols = ["AAPL", "MSFT", "GOOGL", "META", "NVDA"]
    
    print(f"\nSearching for cointegrated pairs...")
    pairs = await engine.find_cointegrated_pairs(symbols, max_pairs=5)
    
    print(f"\nFound {len(pairs)} pairs:")
    for pair in pairs:
        print(f"  {pair.symbol1}/{pair.symbol2}:")
        print(f"    Hedge Ratio: {pair.hedge_ratio:.3f}")
        print(f"    Half-life: {pair.half_life_days:.1f} days")
        print(f"    Z-score: {pair.current_z_score:.2f}")
    
    # Generate trading signals
    signals = await engine.generate_pairs_signals(pairs)
    
    print(f"\nPairs Signals: {len(signals)}")
    for signal in signals[:3]:
        print(f"  {signal.signal_type}: {signal.pair.symbol1}/{signal.pair.symbol2}")
        print(f"    {signal.symbol1_action} {signal.symbol1_quantity} {signal.pair.symbol1}")
        print(f"    {signal.symbol2_action} {signal.symbol2_quantity} {signal.pair.symbol2}")
        print(f"    Confidence: {signal.confidence:.1%}")
    
    return engine


# ============================================================================
# EXAMPLE 6: Full System Integration
# ============================================================================

async def example_full_system():
    """Run complete enhanced trading cycle."""
    from src.options.autonomous_engine import AutonomousTradingEngine
    
    # Initialize engine with all enhancements
    engine = AutonomousTradingEngine(
        portfolio_value=100000,
        paper=True
    )
    
    print("\n" + "="*60)
    print("AUTONOMOUS TRADING ENGINE - ENHANCED")
    print("="*60)
    print("\nInitialized with:")
    print("  ✓ RegimeDetector")
    print("  ✓ CorrelationManager")
    print("  ✓ DynamicWeightOptimizer")
    print("  ✓ VolatilitySurfaceEngine")
    print("  ✓ CointegrationEngine")
    
    # Fit regime detector
    print("\nFitting regime detector...")
    await engine.regime_detector.fit()
    engine.regime_fitted = True
    
    # Detect regime
    regime_state = await engine.regime_detector.detect_current_regime()
    engine.current_regime = regime_state.current_regime
    
    print(f"\nCurrent Market Regime: {regime_state.current_regime.value}")
    print(f"Confidence: {regime_state.confidence:.1%}")
    
    # Get optimized weights
    weights = engine.regime_detector.get_strategy_weights(regime_state.current_regime)
    print(f"\nOptimal Strategy Allocation:")
    for strategy, weight in weights.items():
        print(f"  {strategy:18}: {weight:.1%}")
    
    print("\n✓ System ready for autonomous trading")
    print("\nTo start trading: await engine.run()")
    
    return engine


# ============================================================================
# MAIN - Run all examples
# ============================================================================

async def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("ELITE OPTIONS TRADING ENGINE - QUICK START GUIDE")
    print("="*70)
    
    print("\n\n1. REGIME DETECTION")
    print("-" * 70)
    await example_regime_detection()
    
    print("\n\n2. PORTFOLIO RISK MONITORING")
    print("-" * 70)
    await example_risk_monitoring()
    
    print("\n\n3. DYNAMIC WEIGHT OPTIMIZATION")
    print("-" * 70)
    await example_weight_optimization()
    
    print("\n\n4. VOLATILITY SURFACE ANALYSIS")
    print("-" * 70)
    await example_vol_surface()
    
    print("\n\n5. COINTEGRATION PAIRS TRADING")
    print("-" * 70)
    await example_pairs_trading()
    
    print("\n\n6. FULL SYSTEM INTEGRATION")
    print("-" * 70)
    await example_full_system()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETE")
    print("="*70)
    print("\nYou now have a production-ready elite options trading engine!")
    print("\nNext steps:")
    print("  1. Configure your Alpaca API keys")
    print("  2. Set your risk parameters in config.py")
    print("  3. Run: python -m src.options.autonomous_engine")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())

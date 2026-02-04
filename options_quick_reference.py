#!/usr/bin/env python3
"""
Options Engine - Quick Reference
=================================

Fast examples for common tasks.
"""

# ============================================================================
# BASIC IMPORTS
# ============================================================================

from src.options import (
    # Core
    BlackScholes, OptionType, Greeks,
    
    # Analysis
    IVAnalyzer, IVMetrics,
    ThetaDecayEngine, ThetaMetrics, IVRegime, TrendDirection,
    
    # Risk Management
    GreeksManager, PortfolioGreeks,
    PositionManager, Position, PositionStatus,
    DelayAdapter, MarketPeriod,
    
    # Execution
    TradierExecutor, OrderSide, OrderType as OrderTypeEnum, OrderStatus,
    
    # Strategies
    StrategyEngine, StrategyType, SpreadType,
    OptionCandidate, SpreadCandidate, IronCondorCandidate,
)

# ============================================================================
# 1. PRICE AN OPTION
# ============================================================================

def price_option_example():
    """Calculate option price and Greeks."""
    bs = BlackScholes(risk_free_rate=0.05)
    
    # Price ATM put
    put_price = bs.put_price(
        S=450.0,    # Stock price
        K=450.0,    # Strike
        T=30/365,   # 30 days
        sigma=0.20  # 20% IV
    )
    
    # Get all Greeks
    greeks = bs.calculate_all_greeks(
        S=450.0,
        K=450.0,
        T=30/365,
        sigma=0.20,
        option_type=OptionType.PUT
    )
    
    print(f"Put Price: ${put_price:.2f}")
    print(f"Delta: {greeks.delta:.3f}")
    print(f"Gamma: {greeks.gamma:.4f}")
    print(f"Theta: {greeks.theta:.3f}")
    print(f"Vega: {greeks.vega:.3f}")
    
    return put_price, greeks


# ============================================================================
# 2. FIND WHEEL CANDIDATES
# ============================================================================

def find_wheel_trades():
    """Find cash-secured put candidates."""
    strategy = StrategyEngine()
    
    candidates = strategy.find_wheel_candidates(
        symbol="SPY",
        underlying_price=450.0,
        current_iv=0.22,
        historical_vol=0.18,
        trend=TrendDirection.NEUTRAL,
        top_n=5
    )
    
    for i, candidate in enumerate(candidates, 1):
        print(f"\n{i}. {candidate.symbol} ${candidate.strike} PUT")
        print(f"   DTE: {candidate.dte} | Delta: {candidate.delta:.2f}")
        print(f"   Premium: ${candidate.mid:.2f}")
        print(f"   Score: {candidate.score:.0f}/100")
        print(f"   {candidate.reasoning}")
    
    return candidates


# ============================================================================
# 3. SIZE A POSITION
# ============================================================================

def size_position():
    """Calculate optimal position size."""
    pos_mgr = PositionManager(
        account_value=100_000,
        buying_power=50_000
    )
    
    sizing = pos_mgr.calculate_position_size(
        win_rate=0.65,      # 65% historical win rate
        avg_win=150,        # $150 avg win
        avg_loss=100,       # $100 avg loss
        option_price=4.50,  # $4.50 per contract
        kelly_multiplier=0.25  # Quarter-Kelly
    )
    
    print(f"Recommended Contracts: {sizing.num_contracts}")
    print(f"Capital Required: ${sizing.capital_required:,.0f}")
    print(f"Kelly Fraction: {sizing.kelly_fraction:.2%}")
    print(f"Max Loss: ${sizing.max_loss:,.0f}")
    print(f"Reasoning: {sizing.reasoning}")
    
    return sizing


# ============================================================================
# 4. CHECK PORTFOLIO GREEKS
# ============================================================================

def check_greeks():
    """Monitor portfolio Greeks and limits."""
    greeks_mgr = GreeksManager(account_value=100_000)
    
    # Get portfolio Greeks
    portfolio = greeks_mgr.get_portfolio_greeks()
    
    print("\n" + "="*70)
    print("PORTFOLIO GREEKS")
    print("="*70)
    print(f"Delta: {portfolio.total_delta:+.2f} ({portfolio.delta_per_100k:+.1f}/100K)")
    print(f"Gamma: {portfolio.total_gamma:+.3f} ({portfolio.gamma_per_100k:+.2f}/100K)")
    print(f"Theta: {portfolio.total_theta:+.2f} ({portfolio.theta_per_100k:+.1f}/100K)")
    print(f"Vega:  {portfolio.total_vega:+.2f} ({portfolio.vega_per_100k:+.1f}/100K)")
    print(f"\nPositions: {portfolio.num_positions}")
    print("="*70)
    
    # Check for violations
    violations = greeks_mgr.check_limits()
    
    if violations:
        print(f"\n⚠️  VIOLATIONS: {len(violations)}")
        for v in violations:
            print(f"  [{v.severity.upper()}] {v.message}")
    else:
        print("\n✅ All limits OK")
    
    return portfolio, violations


# ============================================================================
# 5. COMPENSATE FOR 15-MIN DELAY
# ============================================================================

def adjust_for_delay():
    """Adjust entry price for delayed data."""
    adapter = DelayAdapter(delay_minutes=15)
    
    # Check if safe to trade
    is_safe, reason = adapter.is_safe_to_trade(vix_level=18.0)
    
    if not is_safe:
        print(f"❌ Not safe to trade: {reason}")
        return None
    
    # Adjust entry price
    adjusted = adapter.adjust_entry_price(
        quoted_price=4.50,
        is_credit=True,  # Selling premium
        atr=0.25,
        underlying_price=450.0,
        symbol="SPY"
    )
    
    print(f"\n{'='*70}")
    print("DELAY ADJUSTMENT")
    print(f"{'='*70}")
    print(f"Quoted Price:   ${adjusted.original_price:.2f}")
    print(f"Adjusted Price: ${adjusted.adjusted_price:.2f}")
    print(f"Adjustment:     {adjusted.adjustment_pct:+.2f}%")
    print(f"Reason: {adjusted.reason}")
    print(f"{'='*70}")
    
    return adjusted


# ============================================================================
# 6. MONITOR PERFORMANCE
# ============================================================================

def check_performance():
    """Get performance metrics."""
    pos_mgr = PositionManager(
        account_value=100_000,
        buying_power=45_000
    )
    
    perf = pos_mgr.get_performance_summary()
    
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    print(f"Account Value:  ${perf['account_value']:,.0f}")
    print(f"Buying Power:   ${perf['buying_power']:,.0f}")
    print(f"Open Positions: {perf['open_positions']}")
    print(f"\nTotal Trades:   {perf['total_trades']}")
    print(f"Win Rate:       {perf['win_rate']:.1%} ({perf['winning_trades']}W / {perf['losing_trades']}L)")
    print(f"\nRealized P&L:   ${perf['total_realized_pnl']:+,.2f}")
    print(f"Unrealized P&L: ${perf['total_unrealized_pnl']:+,.2f}")
    print(f"Total P&L:      ${perf['total_pnl']:+,.2f} ({perf['pnl_percent']:+.2f}%)")
    print("="*70)
    
    return perf


# ============================================================================
# 7. COMPLETE TRADE WORKFLOW
# ============================================================================

def complete_trade_workflow():
    """Execute complete trade from analysis to entry."""
    
    # Initialize components
    strategy = StrategyEngine()
    iv_analyzer = IVAnalyzer()
    greeks_mgr = GreeksManager(account_value=100_000)
    pos_mgr = PositionManager(account_value=100_000, buying_power=50_000)
    delay = DelayAdapter()
    
    # Market data
    symbol = "SPY"
    price = 450.0
    current_iv = 0.22
    historical_vol = 0.18
    
    print("\n" + "="*70)
    print("TRADE WORKFLOW: WHEEL STRATEGY")
    print("="*70)
    
    # Step 1: Find candidates
    print("\n1. Finding candidates...")
    candidates = strategy.find_wheel_candidates(
        symbol=symbol,
        underlying_price=price,
        current_iv=current_iv,
        historical_vol=historical_vol,
        top_n=3
    )
    
    if not candidates:
        print("   ❌ No candidates found")
        return None
    
    best = candidates[0]
    print(f"   ✅ Best: {symbol} ${best.strike} PUT, Score {best.score:.0f}/100")
    
    # Step 2: Check safety
    print("\n2. Checking trading safety...")
    is_safe, reason = delay.is_safe_to_trade()
    
    if not is_safe:
        print(f"   ❌ {reason}")
        return None
    
    print(f"   ✅ {reason}")
    
    # Step 3: Size position
    print("\n3. Sizing position...")
    sizing = pos_mgr.calculate_position_size(
        win_rate=0.65,
        avg_win=150,
        avg_loss=100,
        option_price=best.mid
    )
    
    print(f"   ✅ {sizing.num_contracts} contracts (${sizing.capital_required:,.0f})")
    
    # Step 4: Adjust for delay
    print("\n4. Adjusting for delay...")
    adjusted = delay.adjust_entry_price(
        quoted_price=best.mid,
        is_credit=True,
        atr=0.25,
        underlying_price=price
    )
    
    print(f"   ✅ ${best.mid:.2f} → ${adjusted.adjusted_price:.2f}")
    
    # Step 5: Validate Greeks
    print("\n5. Validating Greeks...")
    can_add, reason = greeks_mgr.can_add_position(
        new_greeks=best.greeks,
        quantity=-sizing.num_contracts
    )
    
    if not can_add:
        print(f"   ❌ {reason}")
        return None
    
    print(f"   ✅ Greeks within limits")
    
    # Step 6: Ready to execute
    print("\n6. Trade ready for execution")
    print(f"   Symbol:   {symbol}")
    print(f"   Strike:   ${best.strike}")
    print(f"   DTE:      {best.dte} days")
    print(f"   Quantity: {sizing.num_contracts} contracts (short)")
    print(f"   Price:    ${adjusted.adjusted_price:.2f} (limit)")
    print(f"   Premium:  ${adjusted.adjusted_price * sizing.num_contracts * 100:,.0f}")
    
    print("\n" + "="*70)
    print("✅ WORKFLOW COMPLETE - READY TO EXECUTE")
    print("="*70)
    
    return {
        'candidate': best,
        'sizing': sizing,
        'adjusted_price': adjusted,
        'ready': True
    }


# ============================================================================
# MAIN - RUN EXAMPLES
# ============================================================================

def main():
    """Run all examples."""
    
    print("\n" + "🎯 OPTIONS ENGINE QUICK REFERENCE")
    print("="*70 + "\n")
    
    # Example 1: Price option
    print("\n📊 Example 1: Price an Option")
    price_option_example()
    
    # Example 2: Find trades
    print("\n\n🔍 Example 2: Find Wheel Candidates")
    find_wheel_trades()
    
    # Example 3: Size position
    print("\n\n💰 Example 3: Size Position")
    size_position()
    
    # Example 4: Check Greeks
    print("\n\n📈 Example 4: Check Portfolio Greeks")
    check_greeks()
    
    # Example 5: Delay adjustment
    print("\n\n⏰ Example 5: Adjust for Delay")
    adjust_for_delay()
    
    # Example 6: Performance
    print("\n\n📊 Example 6: Check Performance")
    check_performance()
    
    # Example 7: Complete workflow
    print("\n\n🚀 Example 7: Complete Trade Workflow")
    complete_trade_workflow()
    
    print("\n" + "="*70)
    print("✅ All examples complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

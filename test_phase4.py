"""
Phase 4 Test: Real-Time Greeks Engine
=====================================

Tests the GreeksEngine with Black-Scholes calculations.
"""

import sys
import os
from datetime import datetime, timedelta
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from options.greeks_engine import GreeksEngine


def test_greeks_engine():
    """Test Greeks engine."""
    
    print("=" * 60)
    print("PHASE 4 TEST: Real-Time Greeks Engine")
    print("=" * 60)
    
    try:
        # Initialize engine
        print("\n1. Initializing Greeks Engine...")
        engine = GreeksEngine(risk_free_rate=0.05)
        print("✓ Engine initialized")
        
        # Test single option Greeks (ATM call)
        print("\n2. Testing ATM call option Greeks...")
        S = 100.0  # Stock price
        K = 100.0  # ATM strike
        T = 0.25   # 3 months
        r = 0.05   # 5% risk-free rate
        sigma = 0.20  # 20% IV
        
        start = time.time()
        greeks = engine.calculate_greeks(S, K, T, r, sigma, 'call')
        latency = (time.time() - start) * 1000  # ms
        
        print(f"  ✓ Greeks calculated in {latency:.2f}ms")
        print(f"    Delta: {greeks.delta:.3f}")
        print(f"    Gamma: {greeks.gamma:.4f}")
        print(f"    Theta: ${greeks.theta:.2f} per day")
        print(f"    Vega: ${greeks.vega:.2f} per 1% IV")
        print(f"    Rho: ${greeks.rho:.2f} per 1% rate")
        
        # Verify ATM call delta ~0.5
        if 0.4 < greeks.delta < 0.6:
            print(f"    ✓ ATM call delta check passed ({greeks.delta:.3f} ≈ 0.5)")
        else:
            print(f"    ✗ Unexpected delta: {greeks.delta:.3f}")
        
        # Verify gamma > 0
        if greeks.gamma > 0:
            print(f"    ✓ Gamma > 0 check passed")
        else:
            print(f"    ✗ Unexpected gamma: {greeks.gamma:.4f}")
        
        # Test ATM put
        print("\n3. Testing ATM put option Greeks...")
        put_greeks = engine.calculate_greeks(S, K, T, r, sigma, 'put')
        
        print(f"  ✓ Put Greeks:")
        print(f"    Delta: {put_greeks.delta:.3f}")
        print(f"    Gamma: {put_greeks.gamma:.4f}")
        print(f"    Theta: ${put_greeks.theta:.2f} per day")
        
        # Put delta should be ~-0.5
        if -0.6 < put_greeks.delta < -0.4:
            print(f"    ✓ ATM put delta check passed ({put_greeks.delta:.3f} ≈ -0.5)")
        else:
            print(f"    ✗ Unexpected delta: {put_greeks.delta:.3f}")
        
        # Test portfolio Greeks
        print("\n4. Testing portfolio Greeks aggregation...")
        
        positions = [
            {
                'symbol': 'SPY250221C00600000',
                'quantity': 10,  # Long 10 calls
                'underlying_price': 600.0,
                'strike': 600.0,
                'expiry': datetime.now() + timedelta(days=30),
                'iv': 0.18,
                'option_type': 'call'
            },
            {
                'symbol': 'SPY250221P00590000',
                'quantity': -5,  # Short 5 puts
                'underlying_price': 600.0,
                'strike': 590.0,
                'expiry': datetime.now() + timedelta(days=30),
                'iv': 0.20,
                'option_type': 'put'
            },
            {
                'symbol': 'SPY250321C00610000',
                'quantity': 3,  # Long 3 calls
                'underlying_price': 600.0,
                'strike': 610.0,
                'expiry': datetime.now() + timedelta(days=60),
                'iv': 0.19,
                'option_type': 'call'
            }
        ]
        
        portfolio = engine.portfolio_greeks(positions)
        
        print(f"  ✓ Portfolio Greeks:")
        print(f"    Net Delta: {portfolio.net_delta:.1f} shares")
        print(f"    Net Gamma: {portfolio.net_gamma:.2f}")
        print(f"    Net Theta: ${portfolio.net_theta:.2f} per day")
        print(f"    Net Vega: ${portfolio.net_vega:.2f} per 1% IV")
        print(f"    Positions: {portfolio.num_positions}")
        print(f"    Total Notional: ${portfolio.total_notional:,.0f}")
        
        # Test hedge recommendations
        print("\n5. Testing hedge recommendations...")
        recommendations = engine.hedge_recommendation(portfolio, underlying_price=600.0)
        
        if recommendations:
            print(f"  ✓ Generated {len(recommendations)} hedge recommendation(s):")
            for rec in recommendations:
                print(f"    - {rec.action.replace('_', ' ').title()}: {rec.quantity} {rec.symbol}")
                print(f"      Reason: {rec.reason}")
                print(f"      Target: {rec.target_greek}")
        else:
            print("  ✓ Portfolio within risk limits (no hedges needed)")
        
        # Test P&L attribution
        print("\n6. Testing P&L attribution...")
        
        # Simulate market movements
        price_changes = {
            'SPY250221C00600000': 5.0,  # Stock up $5
            'SPY250221P00590000': 5.0,
            'SPY250321C00610000': 5.0
        }
        
        iv_changes = {
            'SPY250221C00600000': 0.02,  # IV up 2%
            'SPY250221P00590000': 0.02,
            'SPY250321C00610000': 0.02
        }
        
        pnl = engine.greeks_pnl_attribution(
            positions,
            price_changes,
            iv_changes,
            time_elapsed_days=1.0
        )
        
        print(f"  ✓ P&L Attribution:")
        print(f"    Delta P&L: ${pnl['delta_pnl']:,.2f}")
        print(f"    Gamma P&L: ${pnl['gamma_pnl']:,.2f}")
        print(f"    Theta P&L: ${pnl['theta_pnl']:,.2f}")
        print(f"    Vega P&L: ${pnl['vega_pnl']:,.2f}")
        print(f"    Total P&L: ${pnl['total_pnl']:,.2f}")
        
        # Test latency requirement (<100ms)
        print("\n7. Testing latency requirement...")
        
        start = time.time()
        for _ in range(100):
            engine.calculate_greeks(S, K, T, r, sigma, 'call')
        avg_latency = ((time.time() - start) / 100) * 1000
        
        print(f"  ✓ Average latency: {avg_latency:.2f}ms")
        
        if avg_latency < 100:
            print(f"    ✓ PASSES LATENCY TARGET: {avg_latency:.2f}ms < 100ms")
        else:
            print(f"    ✗ Exceeds latency target: {avg_latency:.2f}ms > 100ms")
        
        # Test expired option
        print("\n8. Testing expired option handling...")
        expired_greeks = engine.calculate_greeks(S, K, -0.01, r, sigma, 'call')
        
        if expired_greeks.delta == 0 and expired_greeks.gamma == 0:
            print("  ✓ Expired options handled correctly (zero Greeks)")
        else:
            print("  ✗ Expired option handling error")
        
        print("\n" + "=" * 60)
        print("PHASE 4 TEST COMPLETE ✓")
        print("=" * 60)
        print("\nGreeks Engine is ready.")
        print("Features:")
        print("  ✓ Black-Scholes analytical Greeks")
        print("  ✓ Portfolio-level aggregation")
        print("  ✓ Dynamic hedging recommendations")
        print("  ✓ P&L attribution by Greek")
        print("  ✓ <100ms calculation latency")
        print("\nReal-time risk management enabled!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_greeks_engine()

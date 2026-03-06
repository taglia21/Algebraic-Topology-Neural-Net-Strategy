#!/usr/bin/env python3
"""
Quick backtest runner — 15-stock diversified universe.

15 stocks → C(15,2) = 105 pairs (vs 435 with 30 stocks) → ~4× faster.
Covers all 11 GICS sectors for proper diversification testing.
"""

import sys
import json
import time
import traceback

# 15-stock diversified universe (liquid mega-caps across sectors)
UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "NVDA",
    # Healthcare
    "JNJ", "UNH",
    # Financials
    "JPM", "GS",
    # Consumer Discretionary
    "AMZN", "TSLA",
    # Energy
    "XOM",
    # Industrials
    "CAT",
    # Consumer Staples
    "PG",
    # Communication
    "GOOGL",
    # Materials
    "LIN",
    # Utilities
    "NEE",
]

START = "2023-01-01"
END   = "2025-12-31"

def main():
    from backtest.backtester import Backtester

    print(f"Universe: {len(UNIVERSE)} stocks → {len(UNIVERSE)*(len(UNIVERSE)-1)//2} stat-arb pairs")
    print(f"Period:   {START} → {END}")
    print(f"Stocks:   {', '.join(UNIVERSE)}")
    print("-" * 70)

    bt = Backtester(verbose=True)
    t0 = time.time()

    try:
        result = bt.run(
            symbols=UNIVERSE,
            start_date=START,
            end_date=END,
            use_ml=False,
        )
    except Exception as exc:
        print(f"\n{'='*70}")
        print(f"BACKTEST FAILED: {exc}")
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"BACKTEST COMPLETED in {elapsed:.1f}s")
    print(f"{'='*70}")

    # Print metrics
    m = result.metrics
    if hasattr(m, '__dict__'):
        for k, v in m.__dict__.items():
            if isinstance(v, float):
                print(f"  {k:30s}: {v:>12.4f}")
            elif isinstance(v, (int, str)):
                print(f"  {k:30s}: {v}")
    elif isinstance(m, dict):
        for k, v in m.items():
            if isinstance(v, float):
                print(f"  {k:30s}: {v:>12.4f}")
            elif isinstance(v, (int, str)):
                print(f"  {k:30s}: {v}")
    else:
        print(f"  Metrics type: {type(m)}")
        print(f"  {m}")

    # Print trade summary
    trades = result.trades if hasattr(result, 'trades') else []
    print(f"\nTotal trades: {len(trades)}")
    if trades:
        # Show first 20 trades
        for i, t in enumerate(trades[:20]):
            if hasattr(t, '__dict__'):
                print(f"  Trade {i+1}: {t.__dict__}")
            elif isinstance(t, dict):
                sym = t.get('symbol', '?')
                direction = t.get('direction', '?')
                pnl = t.get('pnl', t.get('realized_pnl', '?'))
                print(f"  Trade {i+1}: {sym} {direction} | P&L: {pnl}")
            else:
                print(f"  Trade {i+1}: {t}")
        if len(trades) > 20:
            print(f"  ... and {len(trades)-20} more")

    # Print signal summary
    signals = result.signals if hasattr(result, 'signals') else []
    print(f"\nTotal signals logged: {len(signals)}")

    # Print equity curve stats
    equity = result.equity_curve if hasattr(result, 'equity_curve') else None
    if equity is not None:
        if isinstance(equity, dict):
            values = list(equity.values())
        elif hasattr(equity, 'values'):
            values = equity.values.tolist() if hasattr(equity.values, 'tolist') else list(equity.values)
        else:
            values = list(equity) if equity is not None else []
        if values:
            print(f"\nEquity curve: {len(values)} points")
            print(f"  Start: ${values[0]:,.2f}")
            print(f"  End:   ${values[-1]:,.2f}")
            print(f"  Min:   ${min(values):,.2f}")
            print(f"  Max:   ${max(values):,.2f}")
            total_return = (values[-1] / values[0] - 1) * 100
            print(f"  Total Return: {total_return:+.2f}%")

    # Save full results to JSON
    output = {
        "elapsed_seconds": elapsed,
        "universe": UNIVERSE,
        "start_date": START,
        "end_date": END,
        "total_trades": len(trades),
        "total_signals": len(signals),
    }
    if hasattr(m, '__dict__'):
        for k, v in m.__dict__.items():
            if isinstance(v, (int, float, str, bool)):
                output[k] = v
    elif isinstance(m, dict):
        for k, v in m.items():
            if isinstance(v, (int, float, str, bool)):
                output[k] = v

    with open("backtest_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results saved to backtest_results.json")


if __name__ == "__main__":
    main()

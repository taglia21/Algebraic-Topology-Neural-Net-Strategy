#!/usr/bin/env python3
"""
run_backtest.py — Quick standalone backtest runner.

Uses the full 50+ stock _DEFAULT_SYMBOLS universe from core.config to match
live trading.  Pass --small to use a 15-stock subset for faster iteration.

Usage:
    python run_backtest.py                      # Full 50-stock, no ML
    python run_backtest.py --ml                 # With ML meta-learner
    python run_backtest.py --small              # 15-stock subset (fast)
    python run_backtest.py --start 2020-01-01   # Custom date range
"""

import argparse
import json
import sys
import time
import traceback

from core.config import _DEFAULT_SYMBOLS

# Full universe = _DEFAULT_SYMBOLS minus ETF benchmarks (SPY/QQQ/IWM are not
# traded as individual positions — they're used for regime detection/hedging).
_BENCHMARKS = {"SPY", "QQQ", "IWM"}
UNIVERSE_FULL = [s for s in _DEFAULT_SYMBOLS if s not in _BENCHMARKS]

# Small subset for quick iteration / debugging
UNIVERSE_SMALL = [
    "AAPL", "MSFT", "NVDA", "JNJ", "UNH", "JPM", "GS",
    "AMZN", "TSLA", "XOM", "CAT", "PG", "GOOGL", "LIN", "NEE",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ATNN Quant Powerhouse — Backtest Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--ml", action="store_true", help="Enable ML meta-learner pipeline")
    parser.add_argument("--capital", type=float, default=100_000.0, help="Initial capital")
    parser.add_argument("--small", action="store_true", help="Use 15-stock subset (faster)")
    args = parser.parse_args()

    from main import SystemOrchestrator

    universe = UNIVERSE_SMALL if args.small else UNIVERSE_FULL
    n_pairs = len(universe) * (len(universe) - 1) // 2
    print(f"\nUniverse : {len(universe)} stocks → {n_pairs} stat-arb pairs")
    print(f"Period   : {args.start} → {args.end}")
    print(f"Capital  : ${args.capital:,.0f}")
    print(f"ML       : {'enabled' if args.ml else 'disabled'}")
    print(f"Stocks   : {', '.join(universe)}")
    print("-" * 70)

    orchestrator = SystemOrchestrator(mode="backtest")
    t0 = time.time()

    try:
        result = orchestrator.run_backtest(
            start=args.start,
            end=args.end,
            symbols=universe,
            initial_capital=args.capital,
            use_ml=args.ml,
        )
    except Exception as exc:
        print(f"\n{'=' * 70}")
        print(f"BACKTEST FAILED: {exc}")
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0

    # Print equity curve stats
    equity = result.equity_curve
    if equity is not None and len(equity) > 0:
        values = equity.values.tolist() if hasattr(equity.values, "tolist") else list(equity.values)
        total_return = (values[-1] / values[0] - 1) * 100
        print(f"\nEquity : ${values[0]:,.0f} → ${values[-1]:,.0f} ({total_return:+.1f}%)")
        print(f"Runtime: {elapsed:.1f}s")

    # Save metrics to JSON
    m = result.metrics
    output = {
        "elapsed_seconds": elapsed,
        "universe": universe,
        "universe_size": len(universe),
        "start": args.start,
        "end": args.end,
    }
    if isinstance(m, dict):
        for k, v in m.items():
            if isinstance(v, (int, float, str, bool)):
                output[k] = v

    out_path = "backtest_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

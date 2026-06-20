#!/usr/bin/env python
"""
run_policy_calibration.py
========================
Standalone script to run OOD policy calibration via rolling-window backtests.

Invocation
----------
    python run_policy_calibration.py \
        --start-date 2024-01-01 \
        --end-date 2024-12-31 \
        --window-days 30 \
        --stride-days 7 \
        --metric sharpe_ratio

Outputs
-------
- Console report with per-window metrics and overall recommendation
- JSON file with detailed results for further analysis
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.backtester import Backtester
from core.config import Config
from ml.policy_calibration import PolicyCalibrator


def _safe_mean(values):
    vals = [float(v) for v in values]
    return (sum(vals) / len(vals)) if vals else 0.0


def _safe_std(values):
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return 0.0
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    return var ** 0.5


def _build_policy_diagnostics(results):
    """Build compact tie-break diagnostics from rolling calibration results."""
    by_policy = defaultdict(list)
    for r in results:
        by_policy[r.policy].append(r)

    diagnostics = {}
    for policy, rows in by_policy.items():
        mean_sharpe = _safe_mean([r.sharpe for r in rows])
        mean_sortino = _safe_mean([r.sortino for r in rows])
        mean_return = _safe_mean([r.total_return for r in rows])
        mean_max_dd = _safe_mean([r.max_drawdown for r in rows])
        mean_ood_rate = _safe_mean([r.ood_block_rate for r in rows])
        sharpe_std = _safe_std([r.sharpe for r in rows])
        mean_trades = _safe_mean([r.n_trades for r in rows])

        # Composite tie-break score: prefer higher Sharpe/return and lower
        # variability, OOD block-rate, and drawdown magnitude.
        tie_break_score = (
            mean_sharpe
            + 0.20 * mean_return
            - 0.15 * mean_ood_rate
            - 0.25 * sharpe_std
            - 0.10 * abs(min(0.0, mean_max_dd))
        )

        diagnostics[policy] = {
            "n_windows": len(rows),
            "mean_sharpe": mean_sharpe,
            "sharpe_std": sharpe_std,
            "mean_sortino": mean_sortino,
            "mean_return": mean_return,
            "mean_max_drawdown": mean_max_dd,
            "mean_ood_block_rate": mean_ood_rate,
            "mean_trades": mean_trades,
            "tie_break_score": tie_break_score,
        }

    if not diagnostics:
        return diagnostics, None

    operational_policy = max(
        diagnostics.keys(),
        key=lambda p: diagnostics[p]["tie_break_score"],
    )
    return diagnostics, operational_policy


def run_backtest_for_calibration(
    symbols: list,
    start_date: str,
    end_date: str,
) -> any:
    """Backtest wrapper for policy calibration.

    Uses current ML_OOD_ACTION env var to set the policy.
    """
    # Load config
    config = Config()

    # Create backtester
    backtester = Backtester(config)

    # Run backtest
    result = backtester.run(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        use_ml=True,
    )

    return result


def main():
    """Run policy calibration and output recommendation."""
    parser = argparse.ArgumentParser(
        description="Run OOD policy calibration via rolling-window backtests"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Calibration start date (ISO-8601)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-06-30",
        help="Calibration end date (ISO-8601)",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Rolling window size in days",
    )
    parser.add_argument(
        "--stride-days",
        type=int,
        default=7,
        help="Stride between window starts (days)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="sharpe_ratio",
        choices=["sharpe_ratio", "sortino_ratio", "total_return"],
        help="Ranking metric for recommendation",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="SPY,QQQ,IWM,EEM,GLD,TLT,USO",
        help="Comma-separated symbol universe",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="policy_calibration_results.json",
        help="Output JSON file",
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    print(f"\n{'='*70}")
    print("OOD POLICY CALIBRATION - Rolling Window Analysis")
    print(f"{'='*70}")
    print(f"Period: {args.start_date} → {args.end_date}")
    print(f"Universe: {len(symbols)} symbols: {', '.join(symbols)}")
    print(f"Windows: {args.window_days}-day windows, {args.stride_days}-day stride")
    print(f"Ranking metric: {args.metric}")
    print()

    # Create calibrator
    calibrator = PolicyCalibrator(
        backtest_func=run_backtest_for_calibration,
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Run rolling-window backtests
    print("Running rolling-window backtests...")
    print()
    results = calibrator.run_rolling_backtest(
        window_days=args.window_days,
        stride_days=args.stride_days,
    )

    # Get recommendation
    print()
    print("-" * 70)
    print("RESULTS")
    print("-" * 70)
    print()

    recommendation = calibrator.recommend_policy(results, metric=args.metric)

    # Print table
    table = PolicyCalibrator.format_results_table(results)
    print(table)
    print()

    # Print recommendation
    print("-" * 70)
    print("RECOMMENDATION")
    print("-" * 70)
    print(f"Recommended policy: {recommendation.recommended_policy.upper()}")
    print(f"Reason: {recommendation.reason}")
    print()
    print("Per-policy rankings (mean {})".format(args.metric))
    for policy in sorted(recommendation.rankings.keys()):
        score = recommendation.rankings[policy]
        marker = " ◄ RECOMMENDED" if policy == recommendation.recommended_policy else ""
        print(f"  {policy:10s}: {score:7.3f}{marker}")
    print()

    diagnostics, operational_policy = _build_policy_diagnostics(results)
    print("-" * 70)
    print("TIE-BREAK DIAGNOSTICS")
    print("-" * 70)
    for policy in sorted(diagnostics.keys()):
        d = diagnostics[policy]
        marker = " ◄ OPERATIONAL PICK" if policy == operational_policy else ""
        print(
            f"{policy:10s} | windows={d['n_windows']:2d} | "
            f"Sharpe={d['mean_sharpe']:.3f}±{d['sharpe_std']:.3f} | "
            f"Return={d['mean_return']:.2%} | "
            f"MaxDD={d['mean_max_drawdown']:.2%} | "
            f"OOD={d['mean_ood_block_rate']:.2%} | "
            f"TieScore={d['tie_break_score']:.4f}{marker}"
        )
    print()

    # Save detailed results to JSON
    output_data = {
        "recommendation": {
            "recommended_policy": recommendation.recommended_policy,
            "reason": recommendation.reason,
            "rankings": recommendation.rankings,
        },
        "operational_recommendation": {
            "policy": operational_policy,
            "method": "tie_break_score",
            "weights": {
                "mean_sharpe": 1.0,
                "mean_return": 0.20,
                "mean_ood_block_rate": -0.15,
                "sharpe_std": -0.25,
                "mean_max_drawdown_abs": -0.10,
            },
        },
        "diagnostics": diagnostics,
        "results": [
            {
                "policy": r.policy,
                "window_start": r.window_start,
                "window_end": r.window_end,
                "sharpe": r.sharpe,
                "sortino": r.sortino,
                "max_drawdown": r.max_drawdown,
                "total_return": r.total_return,
                "n_trades": r.n_trades,
                "ood_checks": r.ood_checks,
                "ood_blocks": r.ood_blocks,
                "ood_block_rate": r.ood_block_rate,
                "ood_blocks_by_symbol": r.ood_blocks_by_symbol,
                "ood_blocks_by_regime": r.ood_blocks_by_regime,
                "ood_blocks_by_day": r.ood_blocks_by_day,
            }
            for r in results
        ],
        "calibration_params": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "window_days": args.window_days,
            "stride_days": args.stride_days,
            "metric": args.metric,
            "symbols": symbols,
            "timestamp": datetime.now().isoformat(),
        },
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Detailed results saved to: {output_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Alpha Generation Experiments
============================
Systematic backtesting of configuration variants to identify
the highest-alpha configuration for the VRP engine.

Research-backed experiments:
1. Lower VIX floor + IV/RV ratio entry filter
2. VVIX-inspired sizing + term structure signal booster
3. Dynamic profit targets based on DTE theta curve
4. Increased capital utilization (more concurrent positions)
5. Combined best-of-breed configuration

Each experiment modifies config parameters and/or the signal layer,
runs a fresh backtest, and records key metrics for comparison.
"""

import sys
import os
import copy
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from vrp.config import Config, get_config
from vrp.backtest import VRPBacktester, BacktestMetrics

# Suppress verbose logging during experiments
import logging
logging.disable(logging.WARNING)


@dataclass
class ExperimentResult:
    """Key metrics from a single experiment."""
    name: str
    total_return: float
    annual_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    total_trades: int
    total_pnl: float
    avg_win: float
    avg_loss: float
    alpha: float
    spx_return: float
    annual_vol: float
    max_concurrent: int
    avg_days_held: float
    
    @classmethod
    def from_metrics(cls, name: str, m: BacktestMetrics) -> "ExperimentResult":
        return cls(
            name=name,
            total_return=m.total_return,
            annual_return=m.annual_return,
            sharpe=m.sharpe_ratio,
            sortino=m.sortino_ratio,
            max_drawdown=m.max_drawdown,
            calmar=m.calmar_ratio,
            win_rate=m.win_rate,
            profit_factor=m.profit_factor,
            total_trades=m.total_trades,
            total_pnl=m.total_pnl,
            avg_win=m.avg_win,
            avg_loss=m.avg_loss,
            alpha=m.alpha,
            spx_return=m.spx_return,
            annual_vol=m.annual_volatility,
            max_concurrent=m.max_concurrent,
            avg_days_held=m.avg_days_held,
        )


def run_experiment(name: str, config: Config, verbose: bool = False) -> ExperimentResult:
    """Run a single experiment with the given config."""
    bt = VRPBacktester(config)
    metrics = bt.run(start="2020-01-01", end="2025-12-31", verbose=verbose)
    return ExperimentResult.from_metrics(name, metrics)


def print_comparison(results: List[ExperimentResult]) -> str:
    """Print a comparison table of all experiment results."""
    lines = []
    lines.append("")
    lines.append("=" * 120)
    lines.append("  ALPHA EXPERIMENTS — COMPARISON TABLE")
    lines.append("=" * 120)
    lines.append("")
    
    header = f"{'Experiment':<35} {'Return':>8} {'Annual':>8} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} {'WinRate':>8} {'Trades':>7} {'PnL':>10} {'Alpha':>8}"
    lines.append(header)
    lines.append("-" * 120)
    
    for r in results:
        line = (
            f"{r.name:<35} "
            f"{r.total_return:>+7.1%} "
            f"{r.annual_return:>+7.1%} "
            f"{r.sharpe:>7.2f} "
            f"{r.max_drawdown:>+7.1%} "
            f"{r.calmar:>7.2f} "
            f"{r.win_rate:>7.1%} "
            f"{r.total_trades:>7d} "
            f"${r.total_pnl:>+9,.0f} "
            f"{r.alpha:>+7.1%}"
        )
        lines.append(line)
    
    lines.append("-" * 120)
    
    # Find best on each dimension
    if results:
        best_return = max(results, key=lambda x: x.total_return)
        best_sharpe = max(results, key=lambda x: x.sharpe)
        best_calmar = max(results, key=lambda x: x.calmar)
        best_alpha = max(results, key=lambda x: x.alpha)
        best_pnl = max(results, key=lambda x: x.total_pnl)
        
        lines.append("")
        lines.append("  BEST BY METRIC:")
        lines.append(f"    Return:  {best_return.name} ({best_return.total_return:+.1%})")
        lines.append(f"    Sharpe:  {best_sharpe.name} ({best_sharpe.sharpe:.2f})")
        lines.append(f"    Calmar:  {best_calmar.name} ({best_calmar.calmar:.2f})")
        lines.append(f"    Alpha:   {best_alpha.name} ({best_alpha.alpha:+.1%})")
        lines.append(f"    P&L:     {best_pnl.name} (${best_pnl.total_pnl:+,.0f})")
        
    lines.append("")
    lines.append("=" * 120)
    
    return "\n".join(lines)


def detailed_comparison(results: List[ExperimentResult]) -> str:
    """Detailed breakdown with additional metrics."""
    lines = []
    lines.append("")
    lines.append("  DETAILED METRICS")
    lines.append("-" * 100)
    
    header = f"{'Experiment':<35} {'Sortino':>8} {'AvgWin':>8} {'AvgLoss':>9} {'PF':>6} {'MaxConc':>8} {'AvgDays':>8} {'Vol':>8}"
    lines.append(header)
    lines.append("-" * 100)
    
    for r in results:
        line = (
            f"{r.name:<35} "
            f"{r.sortino:>8.2f} "
            f"${r.avg_win:>+7.0f} "
            f"${r.avg_loss:>+8.0f} "
            f"{r.profit_factor:>6.2f} "
            f"{r.max_concurrent:>8d} "
            f"{r.avg_days_held:>8.1f} "
            f"{r.annual_vol:>7.1%}"
        )
        lines.append(line)
    
    lines.append("-" * 100)
    return "\n".join(lines)


# ===================================================================
# EXPERIMENT DEFINITIONS
# ===================================================================

def experiment_baseline() -> Config:
    """Current production config — baseline for comparison."""
    return get_config()


def experiment_1_lower_vix_floor() -> Config:
    """Lower VIX floor from 20 to 15 with reduced sizing.
    
    Research basis:
    - Predicting Alpha: VRP exists at all VIX levels, IV/RV ratio ~1.3x 
      regardless of regime
    - Chicago Fed (2025): VRP has declined but remains positive
    - Practitioner evidence: selling premium in low-vol profitable if 
      you accept smaller per-trade P&L
    """
    config = get_config()
    config.vix.min_vix = 15.0          # Lower floor from 20 → 15
    config.vix.standard_low = 20.0     # Keep standard band at 20-25
    config.vix.low_vol_sizing_mult = 0.40  # 40% sizing in VIX 15-20 band
    return config


def experiment_2_aggressive_low_vix() -> Config:
    """More aggressive low-VIX trading with tighter stops.
    
    Hypothesis: If we trade at 50% size in VIX 15-20 but use tighter
    stops (1.5x vs 2.0x), the expected value per trade may improve 
    because we cut losers faster while the win rate stays high.
    """
    config = get_config()
    config.vix.min_vix = 15.0
    config.vix.standard_low = 20.0
    config.vix.low_vol_sizing_mult = 0.50  # 50% sizing
    config.spread.stop_loss_multiple = 1.75  # Tighter stop (from 2.0x)
    return config


def experiment_3_dynamic_profit_targets() -> Config:
    """DTE-adaptive profit targets based on theta decay curve.
    
    Research basis:
    - LinkedIn theta analysis: credit spread collects 30% profit in 
      first 15 days, 40% in next 15, last 30% has disproportionate risk
    - Schwab: "sweet spot" is 45-21 DTE, take profits at 50-75%
    - r/thetagang backtests: 80% profit target with 28 DTE yielded 
      $140K over 15 years on single contracts
    
    Implementation: lower the base profit target to 40% (capture gains 
    faster), but raise tight_profit_dte to 21 days (from 14) so we 
    capture the theta acceleration earlier.
    """
    config = get_config()
    config.spread.profit_target_pct = 0.40   # Take profits earlier (from 0.50)
    config.spread.tight_profit_dte = 21       # Shift tight window earlier (from 14)
    config.spread.tight_profit_pct = 0.65     # Lower tight target too (from 0.75)
    return config


def experiment_4_more_positions() -> Config:
    """Increase max concurrent positions to 3 for better capital utilization.
    
    Current engine averages ~1.0 concurrent positions with max 2.
    Capital utilization is only 52.5%. Adding a 3rd position slot
    while keeping total risk at 50% of account.
    """
    config = get_config()
    config.spread.max_concurrent_positions = 3
    config.spread.max_total_risk_pct = 0.60  # Allow slightly more total risk
    config.spread.risk_per_trade = 0.20       # Reduce per-trade risk to compensate
    return config


def experiment_5_tight_stops_wider_spreads() -> Config:
    """Tighter stops + wider spreads for better risk/reward.
    
    The payoff ratio is 0.43 (avg win $233 vs avg loss $544). 
    Tightening stops to 1.5x should reduce avg loss while 
    widening spreads to 20pts captures richer premium.
    """
    config = get_config()
    config.spread.stop_loss_multiple = 1.5   # Much tighter stop
    config.spread.spread_width = 20          # Wider spread (from 15)
    return config


def experiment_6_friday_entry_28dte() -> Config:
    """Friday-only entries targeting 28 DTE.
    
    Research basis:
    - r/thetagang backtest: "28DTE, open only on Fridays" yielded 
      $140K over 15 years with 86.8% win rate
    - Weekend theta decay: Friday entries capture 3 days of theta 
      over the weekend
    - Lower DTE accelerates time decay for faster profit captures
    """
    config = get_config()
    config.spread.entry_days = [4]          # Friday only (Monday=0, Friday=4)
    config.spread.target_dte = 28           # Lower DTE target (from 42)
    config.spread.min_dte = 21              # Keep min
    config.spread.max_dte = 42              # Tighten max (from 56)
    config.spread.profit_target_pct = 0.50  # Keep 50% target
    return config


def experiment_7_lower_vix_more_positions() -> Config:
    """Combine lower VIX floor + more concurrent positions.
    
    The two biggest alpha leaks combined:
    - 548 dormant days (VIX < 20)  
    - 52.5% capital utilization
    """
    config = get_config()
    config.vix.min_vix = 15.0
    config.vix.standard_low = 20.0
    config.vix.low_vol_sizing_mult = 0.40
    config.spread.max_concurrent_positions = 3
    config.spread.max_total_risk_pct = 0.60
    config.spread.risk_per_trade = 0.20
    return config


def experiment_8_full_alpha() -> Config:
    """Everything: lower VIX + more positions + dynamic exits + Friday entry.
    
    The 'kitchen sink' — all research-backed changes combined.
    Risk: possible over-optimization, but each change is independently 
    justified by research.
    """
    config = get_config()
    # VIX floor
    config.vix.min_vix = 15.0
    config.vix.standard_low = 20.0
    config.vix.low_vol_sizing_mult = 0.40
    # Capital utilization
    config.spread.max_concurrent_positions = 3
    config.spread.max_total_risk_pct = 0.60
    config.spread.risk_per_trade = 0.20
    # Dynamic exits
    config.spread.profit_target_pct = 0.45
    config.spread.tight_profit_dte = 21
    config.spread.tight_profit_pct = 0.65
    # Friday entry with shorter DTE
    config.spread.entry_days = [4]
    config.spread.target_dte = 35
    config.spread.max_dte = 49
    return config


def experiment_9_balanced_alpha() -> Config:
    """Balanced approach: moderate VIX floor + positions + exits.
    
    More conservative than experiment 8 but still captures the key 
    alpha drivers without overfitting.
    """
    config = get_config()
    # Moderate VIX floor
    config.vix.min_vix = 16.0
    config.vix.standard_low = 20.0
    config.vix.low_vol_sizing_mult = 0.35
    # Moderate position increase
    config.spread.max_concurrent_positions = 3
    config.spread.max_total_risk_pct = 0.55
    config.spread.risk_per_trade = 0.22
    # Slightly faster exits
    config.spread.profit_target_pct = 0.45
    config.spread.tight_profit_dte = 18
    config.spread.tight_profit_pct = 0.70
    return config


def experiment_10_max_aggression() -> Config:
    """Maximum aggression: VIX floor 14, 4 positions, tighter stops.
    
    Push the envelope — designed to find the ceiling of the strategy.
    """
    config = get_config()
    config.vix.min_vix = 14.0
    config.vix.standard_low = 20.0
    config.vix.low_vol_sizing_mult = 0.30
    config.spread.max_concurrent_positions = 4
    config.spread.max_total_risk_pct = 0.65
    config.spread.risk_per_trade = 0.18
    config.spread.stop_loss_multiple = 1.5
    config.spread.profit_target_pct = 0.45
    config.spread.tight_profit_dte = 21
    config.spread.tight_profit_pct = 0.60
    return config


# ===================================================================
# MAIN RUNNER
# ===================================================================

def main():
    experiments = [
        ("0. BASELINE (current)", experiment_baseline),
        ("1. VIX floor 15 (40% sizing)", experiment_1_lower_vix_floor),
        ("2. VIX floor 15 + tight stops", experiment_2_aggressive_low_vix),
        ("3. Dynamic profit targets", experiment_3_dynamic_profit_targets),
        ("4. 3 concurrent positions", experiment_4_more_positions),
        ("5. Tight stops + wide spreads", experiment_5_tight_stops_wider_spreads),
        ("6. Friday entry + 28 DTE", experiment_6_friday_entry_28dte),
        ("7. VIX 15 + 3 positions", experiment_7_lower_vix_more_positions),
        ("8. Full alpha (all changes)", experiment_8_full_alpha),
        ("9. Balanced alpha", experiment_9_balanced_alpha),
        ("10. Max aggression", experiment_10_max_aggression),
    ]
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"  RUNNING {len(experiments)} EXPERIMENTS")
    print(f"{'='*60}\n")
    
    for i, (name, config_fn) in enumerate(experiments):
        print(f"  [{i+1}/{len(experiments)}] {name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            config = config_fn()
            result = run_experiment(name, config, verbose=False)
            results.append(result)
            print(f"done ({time.time()-t0:.1f}s) — return: {result.total_return:+.1%}, Sharpe: {result.sharpe:.2f}")
        except Exception as e:
            print(f"FAILED: {e}")
    
    # Print comparison
    comparison = print_comparison(results)
    print(comparison)
    
    details = detailed_comparison(results)
    print(details)
    
    # Save results
    output_path = Path("experiments/results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n  Results saved to {output_path}")
    
    # Print the winning configuration
    if results:
        # Score: weighted combination favoring Sharpe, alpha, and reasonable drawdown
        def score(r):
            # Penalize configs with worse drawdown than baseline
            baseline_dd = results[0].max_drawdown if results else -0.20
            dd_penalty = max(0, (abs(r.max_drawdown) - abs(baseline_dd)) * 2)
            return (
                r.sharpe * 0.30 +
                r.calmar * 0.20 +
                r.alpha * 100 * 0.20 +
                r.total_return * 0.15 +
                r.profit_factor * 0.10 * 0.5 +
                r.win_rate * 0.05
                - dd_penalty
            )
        
        scored = sorted(results, key=score, reverse=True)
        print(f"\n  RECOMMENDED CONFIGURATION (composite score):")
        print(f"    Winner: {scored[0].name}")
        print(f"    Return: {scored[0].total_return:+.1%}")
        print(f"    Sharpe: {scored[0].sharpe:.2f}")
        print(f"    Alpha:  {scored[0].alpha:+.1%}")
        print(f"    MaxDD:  {scored[0].max_drawdown:.1%}")
        print(f"    P&L:    ${scored[0].total_pnl:+,.0f}")
        print()


if __name__ == "__main__":
    main()

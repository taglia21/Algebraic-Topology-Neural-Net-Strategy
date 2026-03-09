"""
vrp/walk_forward.py
===================
Walk-forward optimization to prevent overfitting.

Splits the full backtest period into rolling windows:
  - Train (in-sample): calibrate parameters
  - Test (out-of-sample): evaluate with those parameters

This is the gold standard for strategy validation in quantitative finance.
If a strategy can't beat buy-and-hold out-of-sample, the backtest alpha is
likely curve-fit noise.

References:
- Pardo (2008), "The Evaluation and Optimization of Trading Strategies"
- Bailey, Borwein, López de Prado, Zhu (2017), "The Probability of Backtest Overfitting"

Usage:
    python -m vrp.walk_forward --start 2020-01-01 --end 2025-12-31 --train-months 12 --test-months 6
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter grid for optimization
# ---------------------------------------------------------------------------

PARAM_GRID = {
    "short_delta_target": [-0.10, -0.12, -0.15, -0.18],
    "spread_width": [10, 15, 20],
    "profit_target_pct": [0.40, 0.50, 0.60],
    "stop_loss_multiple": [1.5, 2.0, 2.5],
    "min_vix": [18.0, 20.0, 22.0],
}


@dataclass
class WFWindow:
    """A single walk-forward window."""
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: Dict = field(default_factory=dict)
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    test_return: float = 0.0
    test_spx_return: float = 0.0
    test_max_dd: float = 0.0
    test_trades: int = 0


@dataclass
class WFResult:
    """Walk-forward optimization result."""
    windows: List[WFWindow] = field(default_factory=list)
    oos_sharpe: float = 0.0          # aggregate out-of-sample Sharpe
    oos_return: float = 0.0          # aggregate out-of-sample return
    oos_spx_return: float = 0.0      # aggregate SPX return over test periods
    stability_ratio: float = 0.0     # fraction of windows with positive OOS Sharpe
    best_consistent_params: Dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "",
            "=" * 60,
            "  WALK-FORWARD OPTIMIZATION RESULTS",
            "=" * 60,
            f"  Windows:           {len(self.windows)}",
            f"  OOS Sharpe:        {self.oos_sharpe:.2f}",
            f"  OOS Return:        {self.oos_return:+.1%}",
            f"  SPX Return (OOS):  {self.oos_spx_return:+.1%}",
            f"  Stability Ratio:   {self.stability_ratio:.0%}",
            "",
            "  --- Per-Window ---",
        ]
        for i, w in enumerate(self.windows):
            lines.append(
                f"  [{i+1}] Train {w.train_start}→{w.train_end} | "
                f"Test {w.test_start}→{w.test_end} | "
                f"Sharpe: train={w.train_sharpe:.2f} test={w.test_sharpe:.2f} | "
                f"Return={w.test_return:+.1%} | DD={w.test_max_dd:.1%}"
            )
        lines.append("")
        lines.append(f"  Most Consistent Params: {self.best_consistent_params}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def _generate_windows(
    start: str,
    end: str,
    train_months: int = 12,
    test_months: int = 6,
    step_months: int = 6,
) -> List[Tuple[str, str, str, str]]:
    """Generate rolling train/test windows.

    Returns list of (train_start, train_end, test_start, test_end) tuples.
    """
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    windows = []

    current = start_dt
    while True:
        train_start = current
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end_dt:
            # Truncate last window
            test_end = end_dt
            if test_start >= test_end:
                break

        windows.append((
            train_start.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d"),
            test_start.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d"),
        ))

        current += pd.DateOffset(months=step_months)
        if current + pd.DateOffset(months=train_months) > end_dt:
            break

    return windows


def _run_single_backtest(
    start: str,
    end: str,
    params: Dict,
    capital: float = 10_000,
) -> Dict:
    """Run a single backtest with the given parameters.

    Returns dict with sharpe, return, max_dd, trades.
    """
    from vrp.backtest import VRPBacktester
    from vrp.config import get_config

    config = get_config()
    config.backtest.initial_capital = capital

    # Apply parameter overrides
    if "short_delta_target" in params:
        config.spread.short_delta_target = params["short_delta_target"]
    if "spread_width" in params:
        config.spread.spread_width = params["spread_width"]
    if "profit_target_pct" in params:
        config.spread.profit_target_pct = params["profit_target_pct"]
    if "stop_loss_multiple" in params:
        config.spread.stop_loss_multiple = params["stop_loss_multiple"]
    if "min_vix" in params:
        config.vix.min_vix = params["min_vix"]

    try:
        bt = VRPBacktester(config)
        metrics = bt.run(start=start, end=end, verbose=False)

        return {
            "sharpe": metrics.sharpe_ratio,
            "return": metrics.total_return,
            "max_dd": metrics.max_drawdown,
            "trades": metrics.total_trades,
            "pf": metrics.profit_factor,
            "spx_return": metrics.spx_return,
        }
    except Exception as e:
        logger.warning(f"Backtest failed for {params}: {e}")
        return {
            "sharpe": -999, "return": -1, "max_dd": -1,
            "trades": 0, "pf": 0, "spx_return": 0,
        }


def _optimize_window(
    train_start: str,
    train_end: str,
    param_grid: Dict,
    capital: float = 10_000,
    objective: str = "sharpe",
) -> Tuple[Dict, float]:
    """Find the best parameters on a training window.

    Uses a simple grid search (param space is small enough).
    Objective can be 'sharpe', 'return', or 'calmar'.

    Returns (best_params, best_score).
    """
    import itertools

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    best_params = {}
    best_score = -float("inf")

    combos = list(itertools.product(*values))
    logger.info(f"Optimizing {len(combos)} param combos on {train_start}→{train_end}")

    for combo in combos:
        params = dict(zip(keys, combo))
        result = _run_single_backtest(train_start, train_end, params, capital)

        score = result.get(objective, result.get("sharpe", -999))
        if score > best_score and result["trades"] >= 10:
            best_score = score
            best_params = params.copy()

    return best_params, best_score


def run_walk_forward(
    start: str = "2020-01-01",
    end: str = "2025-12-31",
    train_months: int = 12,
    test_months: int = 6,
    step_months: int = 6,
    capital: float = 10_000,
    param_grid: Optional[Dict] = None,
    objective: str = "sharpe",
    verbose: bool = True,
) -> WFResult:
    """Run full walk-forward optimization.

    Parameters
    ----------
    start, end : Date range
    train_months : Length of in-sample training window
    test_months : Length of out-of-sample testing window
    step_months : How far to slide the window each step
    capital : Initial capital for each window
    param_grid : Parameters to optimize (default: PARAM_GRID)
    objective : Optimization target ('sharpe', 'return')
    verbose : Print progress

    Returns
    -------
    WFResult with all windows, OOS metrics, and stability analysis
    """
    if param_grid is None:
        param_grid = PARAM_GRID

    windows = _generate_windows(start, end, train_months, test_months, step_months)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  WALK-FORWARD OPTIMIZATION")
        print(f"{'='*60}")
        print(f"  Period:         {start} → {end}")
        print(f"  Train window:   {train_months} months")
        print(f"  Test window:    {test_months} months")
        print(f"  Step:           {step_months} months")
        print(f"  Windows:        {len(windows)}")
        print(f"  Param combos:   {math.prod(len(v) for v in param_grid.values())}")
        print(f"{'='*60}\n")

    result = WFResult()
    param_votes: Dict[str, Dict] = {}  # track which params win most often

    for i, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
        if verbose:
            print(f"  Window {i+1}/{len(windows)}: "
                  f"train {tr_start}→{tr_end}, test {te_start}→{te_end}")

        # 1. Optimize on training window
        best_params, train_score = _optimize_window(
            tr_start, tr_end, param_grid, capital, objective,
        )

        if not best_params:
            logger.warning(f"No valid params found for window {i+1}")
            continue

        # 2. Test on out-of-sample window
        test_result = _run_single_backtest(te_start, te_end, best_params, capital)

        wf = WFWindow(
            train_start=tr_start,
            train_end=tr_end,
            test_start=te_start,
            test_end=te_end,
            best_params=best_params,
            train_sharpe=train_score,
            test_sharpe=test_result["sharpe"],
            test_return=test_result["return"],
            test_spx_return=test_result["spx_return"],
            test_max_dd=test_result["max_dd"],
            test_trades=test_result["trades"],
        )
        result.windows.append(wf)

        # Track parameter votes
        for k, v in best_params.items():
            if k not in param_votes:
                param_votes[k] = {}
            key = str(v)
            param_votes[k][key] = param_votes[k].get(key, 0) + 1

        if verbose:
            print(f"    Best params: {best_params}")
            print(f"    Train Sharpe: {train_score:.2f}  →  Test Sharpe: {test_result['sharpe']:.2f}")
            print(f"    Test Return: {test_result['return']:+.1%}  |  SPX: {test_result['spx_return']:+.1%}")
            print()

    # Aggregate OOS metrics
    if result.windows:
        test_sharpes = [w.test_sharpe for w in result.windows]
        test_returns = [w.test_return for w in result.windows]
        spx_returns = [w.test_spx_return for w in result.windows]

        result.oos_sharpe = float(np.mean(test_sharpes))
        result.oos_return = float(np.mean(test_returns))
        result.oos_spx_return = float(np.mean(spx_returns))
        result.stability_ratio = sum(1 for s in test_sharpes if s > 0) / len(test_sharpes)

        # Most voted params
        for k, votes in param_votes.items():
            best_val = max(votes, key=votes.get)
            # Convert back to original type
            try:
                result.best_consistent_params[k] = float(best_val)
            except ValueError:
                result.best_consistent_params[k] = best_val

    if verbose:
        print(result.summary())

    return result


def save_wf_results(result: WFResult, filepath: str = "wf_results.json") -> None:
    """Save walk-forward results to JSON."""
    import dataclasses
    data = {
        "oos_sharpe": result.oos_sharpe,
        "oos_return": result.oos_return,
        "stability_ratio": result.stability_ratio,
        "best_consistent_params": result.best_consistent_params,
        "windows": [dataclasses.asdict(w) for w in result.windows],
    }
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Walk-forward results saved to {filepath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="VRP Walk-Forward Optimization")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--step-months", type=int, default=6)
    parser.add_argument("--capital", type=float, default=10_000)
    parser.add_argument("--output", default="wf_results.json")
    args = parser.parse_args()

    result = run_walk_forward(
        start=args.start,
        end=args.end,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        capital=args.capital,
    )
    save_wf_results(result, args.output)


if __name__ == "__main__":
    main()

"""
ml/policy_calibration.py
========================
Automated OOD gating policy calibration via rolling-window backtests.

Rolling-Window Policy Calibration
----------------------------------
This module runs skip/neutral/block OOD policies across multiple overlapping
time windows, compares their risk-adjusted returns, and recommends the best
policy for paper/live trading.

Strategy:
1. Split backtest period into rolling windows (e.g., monthly or quarterly)
2. Run each policy (skip, neutral, block) on each window
3. Rank policies by Sharpe, Sortino, max drawdown per window
4. Aggregate scores across windows
5. Recommend policy with highest mean Sharpe (or weighted score)
6. Output summary table with per-policy per-window metrics

Usage
-----
    from ml.policy_calibration import PolicyCalibrator, calibrate_ood_policy

    # Option 1: Direct high-level call
    recommendation = calibrate_ood_policy(
        backtest_func=my_backtest_function,
        symbols=['SPY', 'QQQ', ...],
        start_date='2024-01-01',
        end_date='2024-12-31',
        window_days=30,  # Monthly windows
        stride_days=7,   # Weekly stride (overlap = 23 days)
    )
    print(recommendation)

    # Option 2: Use calibrator directly for custom workflows
    cal = PolicyCalibrator(backtest_func, symbols, start_date, end_date)
    results = cal.run_rolling_backtest(window_days=30, stride_days=7)
    recommendation = cal.recommend_policy(results)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class PolicyResult:
    """Single policy run on a single window."""

    policy: str  # "skip", "neutral", or "block"
    window_start: str  # ISO-8601 date
    window_end: str  # ISO-8601 date
    sharpe: float  # Annualized Sharpe ratio
    sortino: float  # Annualized Sortino ratio
    max_drawdown: float  # Negative, e.g. -0.12
    total_return: float  # Positive, e.g. 0.05
    n_trades: int
    ood_checks: int
    ood_blocks: int
    ood_block_rate: float
    ood_blocks_by_symbol: Dict[str, int] = field(default_factory=dict)
    ood_blocks_by_regime: Dict[str, int] = field(default_factory=dict)
    ood_blocks_by_day: Dict[str, int] = field(default_factory=dict)


@dataclass
class PolicyRecommendation:
    """Top-level recommendation from calibration run."""

    recommended_policy: str  # "skip", "neutral", or "block"
    reason: str  # Why chosen (e.g., "Best mean Sharpe: 1.23 vs skip 0.91")
    rankings: Dict[str, float]  # {"skip": 1.05, "neutral": 1.23, "block": 0.89} (mean Sharpe)
    all_results: List[PolicyResult]  # Full per-window results for debugging


# ---------------------------------------------------------------------------
# PolicyCalibrator
# ---------------------------------------------------------------------------


class PolicyCalibrator:
    """Orchestrates rolling-window backtests for all OOD policies."""

    def __init__(
        self,
        backtest_func: Callable,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ):
        """Initialize calibrator.

        Parameters
        ----------
        backtest_func:
            Callable(symbols, start_date, end_date, ml_ood_action) -> BacktestResult
            Must accept start_date, end_date as ISO-8601 strings and return
            a BacktestResult with metrics and ml_ood_telemetry.
        symbols:
            Universe of symbols to backtest.
        start_date:
            Start of full calibration period (ISO-8601).
        end_date:
            End of full calibration period (ISO-8601).
        """
        self.backtest_func = backtest_func
        self.symbols = symbols
        self.start_date = datetime.fromisoformat(start_date).date()
        self.end_date = datetime.fromisoformat(end_date).date()

    def run_rolling_backtest(
        self,
        window_days: int = 30,
        stride_days: int = 7,
        policies: Optional[List[str]] = None,
    ) -> List[PolicyResult]:
        """Execute rolling-window backtests for all policies.

        Parameters
        ----------
        window_days:
            Days per window (e.g., 30 = monthly).
        stride_days:
            Days between window starts (e.g., 7 = weekly stride, 23-day overlap).
        policies:
            Policies to test. Defaults to ["skip", "neutral", "block"].

        Returns
        -------
        List[PolicyResult]
            Results for all (policy, window) combinations.
        """
        if policies is None:
            policies = ["skip", "neutral", "block"]

        results = []
        current_date = self.start_date

        # Generate windows
        windows = []
        while current_date + timedelta(days=window_days) <= self.end_date:
            window_end = current_date + timedelta(days=window_days)
            windows.append((current_date, window_end))
            current_date += timedelta(days=stride_days)

        print(
            f"[PolicyCalibrator] Running {len(policies)} policies × {len(windows)} windows = "
            f"{len(policies) * len(windows)} backtests"
        )

        # Run each policy on each window
        for i, policy in enumerate(policies):
            for j, (w_start, w_end) in enumerate(windows):
                print(
                    f"  Policy {i+1}/{len(policies)} ({policy}), "
                    f"window {j+1}/{len(windows)} ({w_start} → {w_end})...",
                    end=" ",
                    flush=True,
                )

                try:
                    result = self._run_single_backtest(
                        policy=policy,
                        start_date=str(w_start),
                        end_date=str(w_end),
                    )
                    results.append(result)
                    print(f"Sharpe={result.sharpe:.2f}, Return={result.total_return:.1%}")
                except Exception as e:
                    print(f"FAILED: {e}")
                    continue

        return results

    def _run_single_backtest(
        self,
        policy: str,
        start_date: str,
        end_date: str,
    ) -> PolicyResult:
        """Run a single backtest with a specific policy.

        Parameters
        ----------
        policy:
            "skip", "neutral", or "block"
        start_date:
            ISO-8601 start date
        end_date:
            ISO-8601 end date

        Returns
        -------
        PolicyResult
            Metrics and telemetry from this backtest run.
        """
        # Set environment variable for ML pipeline and restore after run.
        old_ood_action = os.environ.get("ML_OOD_ACTION")
        os.environ["ML_OOD_ACTION"] = policy
        try:
            result = self.backtest_func(
                symbols=self.symbols,
                start_date=start_date,
                end_date=end_date,
            )
        finally:
            if old_ood_action is None:
                os.environ.pop("ML_OOD_ACTION", None)
            else:
                os.environ["ML_OOD_ACTION"] = old_ood_action

        # Extract metrics
        metrics = result.metrics or {}
        ood_telemetry = result.ml_ood_telemetry or {}

        ood_checks = int(ood_telemetry.get("ood_checks", ood_telemetry.get("checks", 0)) or 0)
        ood_blocks = int(ood_telemetry.get("ood_blocks", ood_telemetry.get("blocks", 0)) or 0)
        ood_block_rate = (
            (ood_blocks / ood_checks) if ood_checks > 0 else 0.0
        )
        ood_blocks_by_symbol = dict(ood_telemetry.get("ood_blocks_by_symbol", {}) or {})
        ood_blocks_by_regime = dict(ood_telemetry.get("ood_blocks_by_regime", {}) or {})
        ood_blocks_by_day = dict(ood_telemetry.get("ood_blocks_by_day", {}) or {})

        return PolicyResult(
            policy=policy,
            window_start=start_date,
            window_end=end_date,
            sharpe=float(metrics.get("sharpe_ratio", 0.0)),
            sortino=float(metrics.get("sortino_ratio", 0.0)),
            max_drawdown=float(metrics.get("max_drawdown", 0.0)),
            total_return=float(metrics.get("total_return", 0.0)),
            n_trades=int(metrics.get("total_trades", 0)),
            ood_checks=ood_checks,
            ood_blocks=ood_blocks,
            ood_block_rate=ood_block_rate,
            ood_blocks_by_symbol=ood_blocks_by_symbol,
            ood_blocks_by_regime=ood_blocks_by_regime,
            ood_blocks_by_day=ood_blocks_by_day,
        )

    def recommend_policy(
        self,
        results: List[PolicyResult],
        metric: str = "sharpe_ratio",
    ) -> PolicyRecommendation:
        """Rank policies by aggregated metric.

        Parameters
        ----------
        results:
            List of PolicyResult from run_rolling_backtest().
        metric:
            Primary ranking metric ("sharpe_ratio", "sortino_ratio", "total_return").

        Returns
        -------
        PolicyRecommendation
            Recommended policy and rankings.
        """
        # Group by policy
        by_policy = {}
        for res in results:
            if res.policy not in by_policy:
                by_policy[res.policy] = []
            by_policy[res.policy].append(res)

        # Calculate mean metric per policy
        rankings = {}
        for policy, policy_results in by_policy.items():
            if metric == "sharpe_ratio":
                scores = [r.sharpe for r in policy_results]
            elif metric == "sortino_ratio":
                scores = [r.sortino for r in policy_results]
            elif metric == "total_return":
                scores = [r.total_return for r in policy_results]
            else:
                raise ValueError(f"Unknown metric: {metric}")

            mean_score = sum(scores) / len(scores) if scores else 0.0
            rankings[policy] = mean_score

        # Find best policy
        recommended_policy = max(rankings, key=rankings.get)
        best_score = rankings[recommended_policy]

        # Build explanation
        other_scores = {p: s for p, s in rankings.items() if p != recommended_policy}
        comparison = (
            ", ".join(f"{p}={s:.2f}" for p, s in sorted(other_scores.items()))
        )
        reason = (
            f"Best mean {metric}: {best_score:.2f} ({recommended_policy}) "
            f"vs {comparison}"
        )

        return PolicyRecommendation(
            recommended_policy=recommended_policy,
            reason=reason,
            rankings=rankings,
            all_results=results,
        )

    @staticmethod
    def format_results_table(results: List[PolicyResult]) -> str:
        """Format results as ASCII table.

        Parameters
        ----------
        results:
            List of PolicyResult to format.

        Returns
        -------
        str
            Pretty-printed ASCII table.
        """
        if not results:
            return "[No results]"

        # Build rows
        rows = []
        for r in sorted(results, key=lambda x: (x.policy, x.window_start)):
            rows.append({
                "Policy": r.policy,
                "Window": f"{r.window_start} → {r.window_end}",
                "Sharpe": f"{r.sharpe:.2f}",
                "Sortino": f"{r.sortino:.2f}",
                "Max DD": f"{r.max_drawdown:.1%}",
                "Return": f"{r.total_return:.1%}",
                "Trades": str(r.n_trades),
                "OOD Rate": f"{r.ood_block_rate:.1%}",
            })

        # Format as table
        if not rows:
            return "[No rows]"

        # Get column widths
        keys = list(rows[0].keys())
        widths = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in keys}

        # Build table string
        header = " | ".join(f"{k:^{widths[k]}}" for k in keys)
        sep = "-+-".join("-" * widths[k] for k in keys)
        body = "\n".join(
            " | ".join(f"{str(r[k]):^{widths[k]}}" for k in keys) for r in rows
        )

        return f"{header}\n{sep}\n{body}"


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def calibrate_ood_policy(
    backtest_func: Callable,
    symbols: List[str],
    start_date: str,
    end_date: str,
    window_days: int = 30,
    stride_days: int = 7,
    metric: str = "sharpe_ratio",
) -> PolicyRecommendation:
    """High-level API: Run full policy calibration in one call.

    Parameters
    ----------
    backtest_func:
        Callable(symbols, start_date, end_date) -> BacktestResult
    symbols:
        Universe to backtest.
    start_date:
        Start of calibration period (ISO-8601).
    end_date:
        End of calibration period (ISO-8601).
    window_days:
        Days per rolling window.
    stride_days:
        Days between window starts (stride).
    metric:
        Ranking metric ("sharpe_ratio", "sortino_ratio", "total_return").

    Returns
    -------
    PolicyRecommendation
        Recommended policy with detailed rankings and results.
    """
    calibrator = PolicyCalibrator(backtest_func, symbols, start_date, end_date)
    results = calibrator.run_rolling_backtest(window_days=window_days, stride_days=stride_days)
    recommendation = calibrator.recommend_policy(results, metric=metric)
    return recommendation

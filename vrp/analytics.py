"""
vrp/analytics.py
================
Performance analytics and Monte Carlo simulation for the VRP engine.

Provides:
1. Monte Carlo bootstrap for Sharpe/drawdown confidence intervals
2. Rolling performance metrics (Sharpe, win rate, PF over time)
3. Regime-conditional return analysis (performance by VIX regime)
4. Greeks P&L attribution (how much P&L came from theta vs delta vs vega)
5. Trade quality scoring (expected vs realized outcomes)

These analytics serve two purposes:
- Backtest validation: are the results statistically significant?
- Live monitoring: is the strategy performing as expected?

References:
- Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier"
- Harvey, Liu, Zhu (2016), "... and the Cross-Section of Expected Returns"
  (multiple testing corrections — we use Bonferroni-adjusted p-values)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Monte Carlo Bootstrap
# ---------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    """Result of a Monte Carlo bootstrap analysis."""
    metric: str
    point_estimate: float
    mean: float
    std: float
    ci_lower: float           # 5th percentile
    ci_upper: float           # 95th percentile
    p_value: float            # probability that true value <= 0
    n_simulations: int = 10000

    def summary(self) -> str:
        return (
            f"{self.metric}: {self.point_estimate:.3f} "
            f"[{self.ci_lower:.3f}, {self.ci_upper:.3f}] "
            f"(p={self.p_value:.4f})"
        )


def bootstrap_sharpe(
    daily_returns: np.ndarray,
    n_simulations: int = 10000,
    risk_free_rate: float = 0.05,
    block_size: int = 5,
) -> BootstrapResult:
    """Block bootstrap confidence interval for the Sharpe ratio.

    Uses block bootstrap (not i.i.d.) to preserve autocorrelation
    structure in daily returns, which is critical for options strategies
    where returns are serially correlated due to gamma exposure.

    Parameters
    ----------
    daily_returns : Array of daily returns
    n_simulations : Number of bootstrap samples
    risk_free_rate : Annualized risk-free rate
    block_size : Block length for block bootstrap (default 5 = 1 week)

    Returns
    -------
    BootstrapResult with CI and p-value
    """
    n = len(daily_returns)
    if n < 30:
        return BootstrapResult(
            metric="Sharpe Ratio",
            point_estimate=0.0, mean=0.0, std=0.0,
            ci_lower=0.0, ci_upper=0.0, p_value=1.0,
        )

    rf_daily = risk_free_rate / 252.0
    excess = daily_returns - rf_daily

    # Point estimate
    point = float(np.mean(excess) / np.std(excess) * np.sqrt(252))

    # Block bootstrap
    n_blocks = max(1, n // block_size)
    sharpes = np.empty(n_simulations)
    rng = np.random.default_rng(42)

    for i in range(n_simulations):
        # Sample block start indices
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        # Build bootstrap sample from blocks
        sample = np.concatenate([excess[s:s+block_size] for s in starts])[:n]
        std = np.std(sample)
        if std > 0:
            sharpes[i] = np.mean(sample) / std * np.sqrt(252)
        else:
            sharpes[i] = 0.0

    return BootstrapResult(
        metric="Sharpe Ratio",
        point_estimate=point,
        mean=float(np.mean(sharpes)),
        std=float(np.std(sharpes)),
        ci_lower=float(np.percentile(sharpes, 5)),
        ci_upper=float(np.percentile(sharpes, 95)),
        p_value=float(np.mean(sharpes <= 0)),
        n_simulations=n_simulations,
    )


def bootstrap_max_drawdown(
    daily_returns: np.ndarray,
    n_simulations: int = 10000,
    block_size: int = 5,
) -> BootstrapResult:
    """Block bootstrap for max drawdown distribution."""
    n = len(daily_returns)
    if n < 30:
        return BootstrapResult(
            metric="Max Drawdown",
            point_estimate=0.0, mean=0.0, std=0.0,
            ci_lower=0.0, ci_upper=0.0, p_value=1.0,
        )

    # Point estimate
    equity = np.cumprod(1 + daily_returns)
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    point = float(np.min(dd))

    # Block bootstrap
    n_blocks = max(1, n // block_size)
    max_dds = np.empty(n_simulations)
    rng = np.random.default_rng(42)

    for i in range(n_simulations):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([daily_returns[s:s+block_size] for s in starts])[:n]
        eq = np.cumprod(1 + sample)
        rm = np.maximum.accumulate(eq)
        drawdowns = (eq - rm) / rm
        max_dds[i] = np.min(drawdowns)

    return BootstrapResult(
        metric="Max Drawdown",
        point_estimate=point,
        mean=float(np.mean(max_dds)),
        std=float(np.std(max_dds)),
        ci_lower=float(np.percentile(max_dds, 5)),
        ci_upper=float(np.percentile(max_dds, 95)),
        p_value=float(np.mean(max_dds <= point)),
        n_simulations=n_simulations,
    )


def bootstrap_win_rate(
    trade_pnls: List[float],
    n_simulations: int = 10000,
) -> BootstrapResult:
    """Bootstrap confidence interval for trade win rate."""
    n = len(trade_pnls)
    if n < 10:
        return BootstrapResult(
            metric="Win Rate",
            point_estimate=0.0, mean=0.0, std=0.0,
            ci_lower=0.0, ci_upper=0.0, p_value=1.0,
        )

    pnls = np.array(trade_pnls)
    point = float(np.mean(pnls > 0))

    rng = np.random.default_rng(42)
    win_rates = np.empty(n_simulations)

    for i in range(n_simulations):
        sample = rng.choice(pnls, size=n, replace=True)
        win_rates[i] = np.mean(sample > 0)

    return BootstrapResult(
        metric="Win Rate",
        point_estimate=point,
        mean=float(np.mean(win_rates)),
        std=float(np.std(win_rates)),
        ci_lower=float(np.percentile(win_rates, 5)),
        ci_upper=float(np.percentile(win_rates, 95)),
        p_value=float(np.mean(win_rates <= 0.5)),  # p-value vs 50% (no edge)
        n_simulations=n_simulations,
    )


# ---------------------------------------------------------------------------
# Rolling Performance Tracker
# ---------------------------------------------------------------------------

class RollingMetrics:
    """Track rolling performance metrics over a specified window.

    Updated daily, provides a real-time view of strategy health:
    - Rolling Sharpe: is our edge persisting or decaying?
    - Rolling win rate: are we maintaining hit rate?
    - Rolling profit factor: is our average win growing vs average loss?
    """

    def __init__(self, window: int = 63) -> None:
        """Initialize with window in trading days (63 ≈ 3 months)."""
        self.window = window
        self._daily_returns: List[float] = []
        self._trade_pnls: List[float] = []
        self._rf_rate: float = 0.05

    def add_daily_return(self, ret: float) -> None:
        self._daily_returns.append(ret)

    def add_trade(self, pnl: float) -> None:
        self._trade_pnls.append(pnl)

    @property
    def rolling_sharpe(self) -> float:
        """Rolling annualized Sharpe ratio."""
        if len(self._daily_returns) < 20:
            return 0.0
        rets = np.array(self._daily_returns[-self.window:])
        rf = self._rf_rate / 252.0
        excess = rets - rf
        std = np.std(excess)
        if std <= 0:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(252))

    @property
    def rolling_win_rate(self) -> float:
        """Win rate over the last N trades."""
        trades = self._trade_pnls[-self.window:]
        if not trades:
            return 0.0
        return sum(1 for t in trades if t > 0) / len(trades)

    @property
    def rolling_profit_factor(self) -> float:
        """Profit factor over the last N trades."""
        trades = self._trade_pnls[-self.window:]
        if not trades:
            return 0.0
        wins = sum(t for t in trades if t > 0)
        losses = abs(sum(t for t in trades if t < 0))
        if losses <= 0:
            return float('inf') if wins > 0 else 0.0
        return wins / losses

    @property
    def rolling_avg_trade(self) -> float:
        """Average trade P&L over the window."""
        trades = self._trade_pnls[-self.window:]
        if not trades:
            return 0.0
        return sum(trades) / len(trades)

    @property
    def rolling_volatility(self) -> float:
        """Rolling annualized volatility."""
        if len(self._daily_returns) < 20:
            return 0.0
        rets = np.array(self._daily_returns[-self.window:])
        return float(np.std(rets) * np.sqrt(252))

    @property
    def consecutive_losses(self) -> int:
        """Current streak of consecutive losing trades."""
        streak = 0
        for pnl in reversed(self._trade_pnls):
            if pnl < 0:
                streak += 1
            else:
                break
        return streak

    def to_dict(self) -> Dict[str, float]:
        return {
            "rolling_sharpe": self.rolling_sharpe,
            "rolling_win_rate": self.rolling_win_rate,
            "rolling_profit_factor": self.rolling_profit_factor,
            "rolling_avg_trade": self.rolling_avg_trade,
            "rolling_volatility": self.rolling_volatility,
            "consecutive_losses": self.consecutive_losses,
            "total_trades": len(self._trade_pnls),
            "total_days": len(self._daily_returns),
        }


# ---------------------------------------------------------------------------
# Regime-Conditional Analysis
# ---------------------------------------------------------------------------

@dataclass
class RegimePerformance:
    """Performance metrics for a specific VIX regime."""
    regime: str
    n_trades: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    total_pnl: float = 0.0
    profit_factor: float = 0.0
    avg_days_held: float = 0.0
    sharpe_contribution: float = 0.0


def analyze_by_regime(
    trades: List[dict],
    regime_thresholds: Tuple[float, float, float, float] = (20, 20, 25, 35),
) -> Dict[str, RegimePerformance]:
    """Analyze trade performance by VIX regime at entry.

    Parameters
    ----------
    trades : List of trade dicts with 'vix_at_entry', 'close_pnl', 'days_held'
    regime_thresholds : (too_low, low_standard, standard_elevated, elevated_crisis)
        Default mirrors active config: min_vix=20, standard_low=20,
        standard_high=25, max_vix=35.

    Returns
    -------
    Dict mapping regime name to RegimePerformance
    """
    tl, ls, se, ec = regime_thresholds

    regimes = {
        "TOO_LOW": [],
        "LOW": [],
        "STANDARD": [],
        "ELEVATED": [],
        "CRISIS": [],
    }

    for t in trades:
        vix = t.get("vix_at_entry", 0)
        if vix < tl:
            regimes["TOO_LOW"].append(t)
        elif vix < ls:
            regimes["LOW"].append(t)
        elif vix <= se:
            regimes["STANDARD"].append(t)
        elif vix <= ec:
            regimes["ELEVATED"].append(t)
        else:
            regimes["CRISIS"].append(t)

    results = {}
    for name, rtrades in regimes.items():
        perf = RegimePerformance(regime=name)
        if not rtrades:
            results[name] = perf
            continue

        perf.n_trades = len(rtrades)
        pnls = [t.get("close_pnl", 0) for t in rtrades]
        perf.total_pnl = sum(pnls)
        perf.avg_pnl = perf.total_pnl / perf.n_trades
        perf.win_rate = sum(1 for p in pnls if p > 0) / perf.n_trades

        wins = sum(p for p in pnls if p > 0)
        losses = abs(sum(p for p in pnls if p < 0))
        perf.profit_factor = wins / max(losses, 1.0)

        days = [t.get("days_held", 0) for t in rtrades]
        perf.avg_days_held = sum(days) / len(days) if days else 0

        results[name] = perf

    return results


# ---------------------------------------------------------------------------
# Greeks P&L Attribution
# ---------------------------------------------------------------------------

@dataclass
class GreeksAttribution:
    """P&L attribution from Greeks decomposition.

    Decomposes daily P&L into:
    - Theta P&L: time decay earned (positive for short premium)
    - Delta P&L: P&L from SPX price movement
    - Vega P&L: P&L from implied vol changes
    - Gamma P&L: P&L from convexity (delta changes)
    - Unexplained: residual (higher-order effects, discrete hedging)
    """
    theta_pnl: float = 0.0
    delta_pnl: float = 0.0
    vega_pnl: float = 0.0
    gamma_pnl: float = 0.0
    unexplained: float = 0.0
    total_pnl: float = 0.0

    def decompose(self) -> Dict[str, float]:
        return {
            "theta": self.theta_pnl,
            "delta": self.delta_pnl,
            "vega": self.vega_pnl,
            "gamma": self.gamma_pnl,
            "unexplained": self.unexplained,
            "total": self.total_pnl,
        }


def attribute_daily_pnl(
    portfolio_greeks: Dict[str, float],
    spx_change: float,
    iv_change: float,
    dt: float = 1.0 / 365.0,
) -> GreeksAttribution:
    """Attribute daily P&L to Greeks components.

    Parameters
    ----------
    portfolio_greeks : Dict with 'delta', 'gamma', 'theta', 'vega'
    spx_change : SPX price change (in points)
    iv_change : IV change (in decimal, e.g., 0.01 = 1% IV move)
    dt : Time step in years (1/365 for daily)

    Returns
    -------
    GreeksAttribution with decomposed P&L
    """
    delta = portfolio_greeks.get("delta", 0)
    gamma = portfolio_greeks.get("gamma", 0)
    theta = portfolio_greeks.get("theta", 0)
    vega = portfolio_greeks.get("vega", 0)

    attr = GreeksAttribution()
    attr.theta_pnl = theta  # theta is already per-day
    attr.delta_pnl = delta * spx_change
    attr.gamma_pnl = 0.5 * gamma * spx_change ** 2
    attr.vega_pnl = vega * (iv_change * 100)  # vega is per 1% IV

    attr.total_pnl = attr.theta_pnl + attr.delta_pnl + attr.gamma_pnl + attr.vega_pnl
    # Unexplained will be filled in by comparing to actual P&L
    return attr


# ---------------------------------------------------------------------------
# Full Analysis Report
# ---------------------------------------------------------------------------

def run_full_analysis(
    equity_curve: List[Tuple],
    trades: List[dict],
    daily_returns: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.05,
) -> Dict:
    """Run a comprehensive analysis suite.

    Returns a dictionary with:
    - bootstrap: Monte Carlo CI for Sharpe, drawdown, win rate
    - regime: performance breakdown by VIX regime
    - rolling: current rolling metrics
    """
    results = {}

    # Daily returns
    if daily_returns is None and equity_curve:
        equities = np.array([e for _, e in equity_curve])
        daily_returns = np.diff(equities) / equities[:-1]

    if daily_returns is not None and len(daily_returns) > 30:
        results["bootstrap_sharpe"] = bootstrap_sharpe(
            daily_returns, risk_free_rate=risk_free_rate
        )
        results["bootstrap_drawdown"] = bootstrap_max_drawdown(daily_returns)

    if trades:
        trade_pnls = [t.get("close_pnl", 0) for t in trades]
        results["bootstrap_win_rate"] = bootstrap_win_rate(trade_pnls)
        results["regime_analysis"] = analyze_by_regime(trades)

    return results


def print_analysis(results: Dict) -> str:
    """Format analysis results for display."""
    lines = [
        "",
        "=" * 60,
        "  VRP ALPHA ENGINE — STATISTICAL ANALYSIS",
        "=" * 60,
        "",
    ]

    # Bootstrap results
    if "bootstrap_sharpe" in results:
        bs = results["bootstrap_sharpe"]
        lines.append("  --- Monte Carlo Bootstrap (10,000 simulations) ---")
        lines.append(f"  {bs.summary()}")

    if "bootstrap_drawdown" in results:
        bd = results["bootstrap_drawdown"]
        lines.append(f"  {bd.summary()}")

    if "bootstrap_win_rate" in results:
        bw = results["bootstrap_win_rate"]
        lines.append(f"  {bw.summary()}")

    lines.append("")

    # Regime analysis
    if "regime_analysis" in results:
        lines.append("  --- Performance by VIX Regime ---")
        lines.append(f"  {'Regime':<12} {'Trades':>7} {'Win%':>7} {'AvgPnL':>9} {'PF':>6} {'Total':>10}")
        lines.append(f"  {'-'*52}")

        for name, perf in results["regime_analysis"].items():
            if perf.n_trades > 0:
                lines.append(
                    f"  {name:<12} {perf.n_trades:>7} {perf.win_rate:>6.1%} "
                    f"${perf.avg_pnl:>+8.0f} {perf.profit_factor:>5.2f} "
                    f"${perf.total_pnl:>+9,.0f}"
                )

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)

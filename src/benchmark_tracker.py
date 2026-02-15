"""
Benchmark Tracker Module (TIER 3)
===================================

Compares portfolio performance against SPY/QQQ and computes
alpha, beta, information ratio, tracking error, and relative metrics.

Features:
1. Rolling alpha/beta via OLS regression
2. Information ratio and tracking error
3. Relative drawdown analysis
4. Up/down capture ratios
5. Multi-benchmark comparison (SPY, QQQ, IWM, 60/40)
6. Calendar-year and rolling-window attribution

Usage:
    from src.benchmark_tracker import BenchmarkTracker, BenchmarkConfig

    tracker = BenchmarkTracker()
    tracker.set_portfolio_returns(dates, returns)
    tracker.add_benchmark("SPY", spy_dates, spy_returns)
    report = tracker.full_report()
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark comparison configuration."""
    rolling_window: int = 60          # days for rolling alpha/beta
    annualization_factor: float = 252.0
    risk_free_rate: float = 0.05      # annual risk-free rate
    min_data_points: int = 20
    benchmarks: List[str] = field(default_factory=lambda: ["SPY", "QQQ"])


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class AlphaBetaResult:
    """OLS alpha/beta regression result."""
    alpha: float = 0.0           # annualized Jensen's alpha
    beta: float = 0.0            # market exposure
    r_squared: float = 0.0       # explanatory power
    alpha_t_stat: float = 0.0    # statistical significance
    residual_vol: float = 0.0    # idiosyncratic volatility


@dataclass
class CaptureRatios:
    """Up/down market capture."""
    up_capture: float = 0.0      # % of benchmark upside captured
    down_capture: float = 0.0    # % of benchmark downside captured
    capture_ratio: float = 0.0   # up_capture / down_capture


@dataclass
class BenchmarkComparison:
    """Full comparison against one benchmark."""
    benchmark_name: str = ""
    period_start: str = ""
    period_end: str = ""

    # Portfolio metrics
    portfolio_return: float = 0.0
    portfolio_sharpe: float = 0.0
    portfolio_volatility: float = 0.0
    portfolio_max_dd: float = 0.0

    # Benchmark metrics
    benchmark_return: float = 0.0
    benchmark_sharpe: float = 0.0
    benchmark_volatility: float = 0.0
    benchmark_max_dd: float = 0.0

    # Relative metrics
    excess_return: float = 0.0
    alpha_beta: Optional[AlphaBetaResult] = None
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    capture_ratios: Optional[CaptureRatios] = None
    correlation: float = 0.0
    relative_max_dd: float = 0.0

    # Rolling (optional)
    rolling_alpha: List[float] = field(default_factory=list)
    rolling_beta: List[float] = field(default_factory=list)
    rolling_dates: List[str] = field(default_factory=list)


# =============================================================================
# MATH UTILITIES
# =============================================================================

class _StatUtils:
    """Statistical computation helpers."""

    @staticmethod
    def ols_alpha_beta(
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        risk_free_daily: float = 0.0,
    ) -> AlphaBetaResult:
        """OLS regression: R_p - R_f = alpha + beta * (R_b - R_f) + epsilon."""
        n = len(portfolio_returns)
        if n < 3:
            return AlphaBetaResult()

        y = portfolio_returns - risk_free_daily
        x = benchmark_returns - risk_free_daily

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        cov_xy = np.mean((x - x_mean) * (y - y_mean))
        var_x = np.var(x, ddof=1) if n > 1 else 1e-10

        beta = cov_xy / var_x if var_x > 1e-12 else 0.0
        alpha_daily = y_mean - beta * x_mean
        alpha_annual = alpha_daily * 252

        # R-squared
        y_hat = alpha_daily + beta * x
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        # t-stat for alpha
        residuals = y - y_hat
        res_vol = np.std(residuals, ddof=2) if n > 2 else 1e-10
        se_alpha = res_vol / np.sqrt(n) if n > 0 else 1e-10
        t_stat = alpha_daily / se_alpha if se_alpha > 1e-12 else 0.0

        return AlphaBetaResult(
            alpha=float(alpha_annual),
            beta=float(beta),
            r_squared=float(max(0, min(1, r_sq))),
            alpha_t_stat=float(t_stat),
            residual_vol=float(res_vol * np.sqrt(252)),
        )

    @staticmethod
    def capture_ratios(
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> CaptureRatios:
        """Compute up/down capture ratios."""
        up_mask = benchmark_returns > 0
        down_mask = benchmark_returns < 0

        up_port = np.mean(portfolio_returns[up_mask]) if np.any(up_mask) else 0.0
        up_bench = np.mean(benchmark_returns[up_mask]) if np.any(up_mask) else 1e-10
        down_port = np.mean(portfolio_returns[down_mask]) if np.any(down_mask) else 0.0
        down_bench = np.mean(benchmark_returns[down_mask]) if np.any(down_mask) else 1e-10

        up_cap = (up_port / up_bench * 100) if abs(up_bench) > 1e-12 else 0.0
        down_cap = (down_port / down_bench * 100) if abs(down_bench) > 1e-12 else 0.0
        ratio = up_cap / down_cap if abs(down_cap) > 1e-6 else 0.0

        return CaptureRatios(
            up_capture=float(up_cap),
            down_capture=float(down_cap),
            capture_ratio=float(ratio),
        )

    @staticmethod
    def max_drawdown(values: np.ndarray) -> float:
        """Maximum drawdown percentage."""
        if len(values) < 2:
            return 0.0
        peak = np.maximum.accumulate(values)
        dd = (peak - values) / np.where(peak > 0, peak, 1) * 100
        return float(np.max(dd))

    @staticmethod
    def sharpe(returns: np.ndarray, risk_free_daily: float = 0.0) -> float:
        """Annualized Sharpe ratio."""
        excess = returns - risk_free_daily
        if len(excess) < 2 or np.std(excess) < 1e-12:
            return 0.0
        return float(np.mean(excess) / np.std(excess) * np.sqrt(252))


# =============================================================================
# BENCHMARK TRACKER
# =============================================================================

class BenchmarkTracker:
    """
    Tracks portfolio vs benchmarks with alpha/beta, IR, capture ratios.

    Usage:
        tracker = BenchmarkTracker()
        tracker.set_portfolio_returns(dates, returns)
        tracker.add_benchmark("SPY", spy_dates, spy_returns)
        comparison = tracker.compare("SPY")
        report = tracker.full_report()
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self._portfolio_dates: List[str] = []
        self._portfolio_returns: np.ndarray = np.array([])
        self._benchmarks: Dict[str, Tuple[List[str], np.ndarray]] = {}
        self._risk_free_daily = self.config.risk_free_rate / self.config.annualization_factor
        logger.info("BenchmarkTracker initialized (rf=%.2f%%)", self.config.risk_free_rate * 100)

    # ── Data input ───────────────────────────────────────────────────────

    def set_portfolio_returns(self, dates: List[str], returns: List[float]) -> None:
        """Set portfolio daily returns series."""
        self._portfolio_dates = list(dates)
        self._portfolio_returns = np.array(returns, dtype=float)
        logger.info("Portfolio returns set: %d data points", len(returns))

    def add_benchmark(self, name: str, dates: List[str], returns: List[float]) -> None:
        """Add a benchmark return series."""
        self._benchmarks[name] = (list(dates), np.array(returns, dtype=float))
        logger.info("Benchmark '%s' added: %d data points", name, len(returns))

    def set_portfolio_values(self, dates: List[str], values: List[float]) -> None:
        """Convenience: set portfolio from value series (computes returns)."""
        arr = np.array(values, dtype=float)
        rets = np.diff(arr) / arr[:-1]
        self._portfolio_dates = list(dates[1:])
        self._portfolio_returns = rets

    def add_benchmark_values(self, name: str, dates: List[str], values: List[float]) -> None:
        """Convenience: add benchmark from value series."""
        arr = np.array(values, dtype=float)
        rets = np.diff(arr) / arr[:-1]
        self._benchmarks[name] = (list(dates[1:]), rets)

    # ── Alignment ────────────────────────────────────────────────────────

    def _align(self, benchmark_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Align portfolio and benchmark returns on common dates."""
        b_dates, b_returns = self._benchmarks[benchmark_name]
        p_date_set = set(self._portfolio_dates)
        b_date_set = set(b_dates)
        common = sorted(p_date_set & b_date_set)

        if len(common) < self.config.min_data_points:
            logger.warning(
                "Only %d common dates with %s (need %d)",
                len(common), benchmark_name, self.config.min_data_points,
            )

        p_idx = {d: i for i, d in enumerate(self._portfolio_dates)}
        b_idx = {d: i for i, d in enumerate(b_dates)}
        p_ret = np.array([self._portfolio_returns[p_idx[d]] for d in common])
        b_ret = np.array([b_returns[b_idx[d]] for d in common])

        return p_ret, b_ret, common

    # ── Comparison ───────────────────────────────────────────────────────

    def compare(self, benchmark_name: str) -> BenchmarkComparison:
        """Full comparison against a single benchmark."""
        if benchmark_name not in self._benchmarks:
            logger.error("Benchmark '%s' not found", benchmark_name)
            return BenchmarkComparison(benchmark_name=benchmark_name)

        p_ret, b_ret, dates = self._align(benchmark_name)
        if len(p_ret) < self.config.min_data_points:
            return BenchmarkComparison(benchmark_name=benchmark_name)

        rf = self._risk_free_daily
        comp = BenchmarkComparison(
            benchmark_name=benchmark_name,
            period_start=dates[0] if dates else "",
            period_end=dates[-1] if dates else "",
        )

        # Cumulative returns
        p_cum = np.cumprod(1 + p_ret)
        b_cum = np.cumprod(1 + b_ret)

        comp.portfolio_return = float((p_cum[-1] - 1) * 100)
        comp.benchmark_return = float((b_cum[-1] - 1) * 100)
        comp.excess_return = comp.portfolio_return - comp.benchmark_return

        comp.portfolio_volatility = float(np.std(p_ret) * np.sqrt(252) * 100)
        comp.benchmark_volatility = float(np.std(b_ret) * np.sqrt(252) * 100)

        comp.portfolio_sharpe = _StatUtils.sharpe(p_ret, rf)
        comp.benchmark_sharpe = _StatUtils.sharpe(b_ret, rf)

        comp.portfolio_max_dd = _StatUtils.max_drawdown(p_cum)
        comp.benchmark_max_dd = _StatUtils.max_drawdown(b_cum)

        # Alpha / Beta
        comp.alpha_beta = _StatUtils.ols_alpha_beta(p_ret, b_ret, rf)

        # Information ratio & tracking error
        active_returns = p_ret - b_ret
        te = float(np.std(active_returns) * np.sqrt(252) * 100) if len(active_returns) > 1 else 0.0
        comp.tracking_error = te
        if te > 1e-6:
            comp.information_ratio = float(np.mean(active_returns) * 252 / (np.std(active_returns) * np.sqrt(252))) if np.std(active_returns) > 0 else 0.0

        # Capture ratios
        comp.capture_ratios = _StatUtils.capture_ratios(p_ret, b_ret)

        # Correlation
        if np.std(p_ret) > 0 and np.std(b_ret) > 0:
            comp.correlation = float(np.corrcoef(p_ret, b_ret)[0, 1])

        # Relative drawdown
        rel_cum = p_cum / b_cum
        comp.relative_max_dd = _StatUtils.max_drawdown(rel_cum)

        # Rolling alpha/beta
        w = self.config.rolling_window
        if len(p_ret) >= w:
            for i in range(w, len(p_ret) + 1):
                ab = _StatUtils.ols_alpha_beta(p_ret[i - w:i], b_ret[i - w:i], rf)
                comp.rolling_alpha.append(ab.alpha)
                comp.rolling_beta.append(ab.beta)
                comp.rolling_dates.append(dates[i - 1])

        return comp

    def full_report(self) -> Dict[str, BenchmarkComparison]:
        """Compare against all loaded benchmarks."""
        return {name: self.compare(name) for name in self._benchmarks}

    # ── Summary display ──────────────────────────────────────────────────

    def print_report(self, comparison: Optional[BenchmarkComparison] = None) -> str:
        """Pretty-print a comparison report. Returns the string."""
        comps = [comparison] if comparison else list(self.full_report().values())
        lines = []
        for c in comps:
            ab = c.alpha_beta or AlphaBetaResult()
            cap = c.capture_ratios or CaptureRatios()
            lines.append(f"\n{'='*60}")
            lines.append(f" Portfolio vs {c.benchmark_name}")
            lines.append(f" Period: {c.period_start} → {c.period_end}")
            lines.append(f"{'='*60}")
            lines.append(f" {'Metric':<25} {'Portfolio':>12} {'Benchmark':>12} {'Diff':>10}")
            lines.append(f" {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
            lines.append(f" {'Return':<25} {c.portfolio_return:>11.2f}% {c.benchmark_return:>11.2f}% {c.excess_return:>+9.2f}%")
            lines.append(f" {'Sharpe':<25} {c.portfolio_sharpe:>12.2f} {c.benchmark_sharpe:>12.2f} {c.portfolio_sharpe - c.benchmark_sharpe:>+10.2f}")
            lines.append(f" {'Volatility':<25} {c.portfolio_volatility:>11.2f}% {c.benchmark_volatility:>11.2f}%")
            lines.append(f" {'Max Drawdown':<25} {c.portfolio_max_dd:>11.2f}% {c.benchmark_max_dd:>11.2f}%")
            lines.append(f"")
            lines.append(f" {'Alpha (ann.)':<25} {ab.alpha:>+11.4f}%")
            lines.append(f" {'Beta':<25} {ab.beta:>12.3f}")
            lines.append(f" {'R²':<25} {ab.r_squared:>12.3f}")
            lines.append(f" {'Alpha t-stat':<25} {ab.alpha_t_stat:>12.3f}")
            lines.append(f" {'Information Ratio':<25} {c.information_ratio:>12.3f}")
            lines.append(f" {'Tracking Error':<25} {c.tracking_error:>11.2f}%")
            lines.append(f" {'Correlation':<25} {c.correlation:>12.3f}")
            lines.append(f"")
            lines.append(f" {'Up Capture':<25} {cap.up_capture:>11.1f}%")
            lines.append(f" {'Down Capture':<25} {cap.down_capture:>11.1f}%")
            lines.append(f" {'Capture Ratio':<25} {cap.capture_ratio:>12.2f}")

        text = "\n".join(lines)
        return text

    def to_dict(self) -> Dict[str, Any]:
        """Serialize full report to dict."""
        report = self.full_report()
        return {name: asdict(comp) for name, comp in report.items()}


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import random

    # Generate synthetic data
    n = 252
    dates = [(date.today() - timedelta(days=n - i)).isoformat() for i in range(n)]
    market_returns = [random.gauss(0.0004, 0.012) for _ in range(n)]

    # Portfolio = beta * market + alpha + noise
    alpha_daily = 0.0003
    beta = 1.15
    portfolio_returns = [
        alpha_daily + beta * m + random.gauss(0, 0.005)
        for m in market_returns
    ]

    # QQQ = correlated but different
    qqq_returns = [m * 1.3 + random.gauss(0.0001, 0.008) for m in market_returns]

    tracker = BenchmarkTracker()
    tracker.set_portfolio_returns(dates, portfolio_returns)
    tracker.add_benchmark("SPY", dates, market_returns)
    tracker.add_benchmark("QQQ", dates, qqq_returns)

    report = tracker.full_report()
    text = tracker.print_report()
    print(text)

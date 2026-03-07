"""
backtest/metrics.py
===================
Performance analytics for the ATNN backtesting engine.

Provides :class:`PerformanceMetrics` — a stateless utility class that computes
a comprehensive suite of risk-adjusted return metrics from an equity curve and
trade list — plus :class:`BacktestResult`, the structured container returned by
:class:`~backtest.backtester.Backtester`.

Metrics Computed
----------------
Return metrics:
    total_return, annual_return, daily_returns (via equity curve pct-change)

Risk-adjusted:
    sharpe_ratio, sortino_ratio, calmar_ratio

Drawdown:
    max_drawdown, max_drawdown_duration

Trade statistics:
    win_rate, profit_factor, avg_win, avg_loss, avg_win_loss_ratio,
    total_trades, avg_holding_period, turnover

Distribution:
    volatility, skewness, kurtosis, var_95, cvar_95

Benchmark-relative (optional SPY comparison):
    alpha, beta, information_ratio, tracking_error

Usage
-----
    from backtest.metrics import PerformanceMetrics, BacktestResult
    import pandas as pd

    equity = pd.Series([100_000, 101_500, 99_200, ...], index=dates)
    metrics = PerformanceMetrics.calculate_all(equity, trades, benchmark=spy)
    report  = PerformanceMetrics.generate_report(metrics, equity, benchmark=spy)
    print(report)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRADING_DAYS_PER_YEAR: int = 252
_DEFAULT_RISK_FREE_RATE: float = 0.05   # 5% annual, used for Sharpe/Sortino
_ROLLING_SHARPE_WINDOW: int = 63        # ~1 quarter


# ---------------------------------------------------------------------------
# BacktestResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Structured container for all outputs of a completed backtest run.

    Attributes
    ----------
    equity_curve:
        Daily portfolio equity values indexed by trading date.
    daily_returns:
        Daily percentage returns (pct_change of equity_curve).
    trades:
        List of trade records.  Each record is a dict with keys:
        ``symbol``, ``side``, ``entry_date``, ``exit_date``,
        ``entry_price``, ``exit_price``, ``qty``, ``pnl``,
        ``holding_days``, ``strategy``.
    signals:
        All raw signals generated during the backtest (strategy diagnostics).
    regime_history:
        Regime label at each bar (indexed by date, values e.g. ``"BULL"``).
    metrics:
        Comprehensive metrics dict from
        :meth:`PerformanceMetrics.calculate_all`.
    config:
        Serialised :class:`~core.config.Config` used for this run.
    start_date:
        ISO-8601 start date string (``"YYYY-MM-DD"``).
    end_date:
        ISO-8601 end date string.
    symbols:
        Universe of symbols traded.
    """

    equity_curve: pd.Series
    daily_returns: pd.Series
    trades: List[dict]
    signals: List[dict]
    regime_history: pd.Series
    metrics: dict
    config: dict
    start_date: str
    end_date: str
    symbols: List[str]

    # Convenience properties ------------------------------------------------

    @property
    def n_trades(self) -> int:
        """Total number of closed trades."""
        return len(self.trades)

    @property
    def sharpe_ratio(self) -> float:
        """Annualised Sharpe ratio (shortcut to metrics dict)."""
        return float(self.metrics.get("sharpe_ratio", float("nan")))

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown fraction (negative, e.g. -0.12)."""
        return float(self.metrics.get("max_drawdown", float("nan")))

    @property
    def total_return(self) -> float:
        """Total cumulative return as a fraction (e.g. 0.15 for 15%)."""
        return float(self.metrics.get("total_return", float("nan")))

    def __repr__(self) -> str:
        return (
            f"BacktestResult("
            f"{self.start_date} → {self.end_date}, "
            f"symbols={len(self.symbols)}, "
            f"trades={self.n_trades}, "
            f"sharpe={self.sharpe_ratio:.2f}, "
            f"max_dd={self.max_drawdown:.1%})"
        )


# ---------------------------------------------------------------------------
# PerformanceMetrics
# ---------------------------------------------------------------------------

class PerformanceMetrics:
    """Stateless performance analytics helper.

    All methods are ``@staticmethod`` — no instance is needed.

    Parameters accepted by :meth:`calculate_all`
    ---------------------------------------------
    equity_curve:
        ``pd.Series`` of portfolio equity values, indexed by trading date.
        Must have at least 2 data points.
    trades:
        List of trade dicts.  Expected keys per record:

        ``pnl``           — realised P&L for the trade (float)
        ``holding_days``  — days position was held (int / float)
        ``entry_date``    — entry timestamp (optional, for turnover calc)
        ``exit_date``     — exit timestamp (optional)
        ``qty``           — number of shares (optional, for turnover calc)
        ``entry_price``   — entry price per share (optional)

    benchmark:
        Optional SPY (or other benchmark) equity curve with the same date
        index.  Required for alpha, beta, tracking_error, information_ratio.
    """

    # ------------------------------------------------------------------
    # Master entry-point
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_all(
        equity_curve: pd.Series,
        trades: List[dict],
        benchmark: Optional[pd.Series] = None,
        regime_history: Optional[pd.Series] = None,
    ) -> dict:
        """Calculate the full suite of performance metrics.

        Parameters
        ----------
        equity_curve:
            Portfolio equity indexed by date.
        trades:
            Closed trade records (see class-level docstring for expected keys).
        benchmark:
            Optional benchmark equity curve for alpha/beta calculations.

        Returns
        -------
        dict
            Comprehensive metrics dictionary.  All monetary metrics are
            expressed as fractions (e.g. 0.15 for 15%), not dollar amounts.
        """
        if equity_curve is None or len(equity_curve) < 2:
            return _empty_metrics()

        # Align index type
        equity_curve = equity_curve.sort_index().dropna()

        # Daily returns
        returns = equity_curve.pct_change().dropna()

        # ----------------------------------------------------------------
        # Return metrics
        # ----------------------------------------------------------------
        initial = float(equity_curve.iloc[0])
        final = float(equity_curve.iloc[-1])
        total_return = (final - initial) / max(initial, 1.0)

        n_days = len(returns)
        years = n_days / _TRADING_DAYS_PER_YEAR
        annual_return = (1.0 + total_return) ** (1.0 / max(years, 1.0 / _TRADING_DAYS_PER_YEAR)) - 1.0

        # ----------------------------------------------------------------
        # Risk metrics
        # ----------------------------------------------------------------
        vol = float(returns.std() * math.sqrt(_TRADING_DAYS_PER_YEAR))
        sharpe = PerformanceMetrics.calculate_sharpe(returns)
        sortino = PerformanceMetrics.calculate_sortino(returns)

        # ----------------------------------------------------------------
        # Drawdown
        # ----------------------------------------------------------------
        max_dd, peak_date, trough_date, recovery_date, dd_duration = (
            PerformanceMetrics.calculate_max_drawdown(equity_curve)
        )
        calmar = (annual_return / abs(max_dd)) if max_dd != 0 else float("nan")

        # ----------------------------------------------------------------
        # Distribution
        # ----------------------------------------------------------------
        skew_val = float(returns.skew()) if len(returns) >= 3 else float("nan")
        kurt_val = float(returns.kurtosis()) if len(returns) >= 4 else float("nan")
        var_95 = float(np.percentile(returns, 5)) if len(returns) >= 2 else float("nan")
        cvar_95 = float(returns[returns <= var_95].mean()) if len(returns) >= 2 else float("nan")

        # ----------------------------------------------------------------
        # Trade statistics
        # ----------------------------------------------------------------
        trade_metrics = PerformanceMetrics._calculate_trade_metrics(trades)

        # ----------------------------------------------------------------
        # Turnover
        # ----------------------------------------------------------------
        turnover = PerformanceMetrics._calculate_turnover(trades, equity_curve)

        # ----------------------------------------------------------------
        # Benchmark-relative (optional)
        # ----------------------------------------------------------------
        bench_metrics = PerformanceMetrics._calculate_benchmark_metrics(returns, benchmark)

        # ----------------------------------------------------------------
        # Assemble result
        # ----------------------------------------------------------------
        # ----------------------------------------------------------------
        # Regime-aware breakdown (optional)
        # ----------------------------------------------------------------
        regime_metrics = PerformanceMetrics._calculate_regime_metrics(
            returns, regime_history
        )

        metrics = {
            "total_return":          total_return,
            "annual_return":         annual_return,
            "sharpe_ratio":          sharpe,
            "sortino_ratio":         sortino,
            "max_drawdown":          max_dd,
            "max_drawdown_duration": dd_duration,
            "calmar_ratio":          calmar,
            "volatility":            vol,
            "skewness":              skew_val,
            "kurtosis":              kurt_val,
            "var_95":                var_95,
            "cvar_95":               cvar_95,
            "turnover":              turnover,
            **trade_metrics,
            **bench_metrics,
            **regime_metrics,
        }
        return metrics

    # ------------------------------------------------------------------
    # Sharpe / Sortino
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_sharpe(
        returns: pd.Series,
        risk_free: float = _DEFAULT_RISK_FREE_RATE,
    ) -> float:
        """Annualised Sharpe ratio.

        Parameters
        ----------
        returns:
            Daily return series.
        risk_free:
            Annual risk-free rate (default 5%).

        Returns
        -------
        float
            Annualised Sharpe ratio, or ``nan`` if insufficient data.
        """
        if returns is None or len(returns) < 2:
            return float("nan")

        daily_rf = risk_free / _TRADING_DAYS_PER_YEAR
        excess = returns - daily_rf
        std = float(excess.std())
        if std == 0:
            return float("nan")
        return float((excess.mean() / std) * math.sqrt(_TRADING_DAYS_PER_YEAR))

    @staticmethod
    def calculate_sortino(
        returns: pd.Series,
        risk_free: float = _DEFAULT_RISK_FREE_RATE,
    ) -> float:
        """Annualised Sortino ratio (downside deviation only).

        Parameters
        ----------
        returns:
            Daily return series.
        risk_free:
            Annual risk-free rate.

        Returns
        -------
        float
            Annualised Sortino ratio, or ``nan`` if insufficient data.
        """
        if returns is None or len(returns) < 2:
            return float("nan")

        daily_rf = risk_free / _TRADING_DAYS_PER_YEAR
        excess = returns - daily_rf
        downside = excess[excess < 0]
        if len(downside) < 2:
            return float("nan")

        downside_std = float(np.sqrt((downside ** 2).mean()))
        if downside_std == 0:
            return float("nan")

        return float((excess.mean() / downside_std) * math.sqrt(_TRADING_DAYS_PER_YEAR))

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_max_drawdown(
        equity_curve: pd.Series,
    ) -> Tuple[float, Any, Any, Any, int]:
        """Calculate maximum drawdown and related statistics.

        Parameters
        ----------
        equity_curve:
            Portfolio equity indexed by date.

        Returns
        -------
        Tuple of:
            - max_dd (float)     : maximum drawdown as a fraction (negative).
            - peak_date          : index value at peak.
            - trough_date        : index value at trough.
            - recovery_date      : index value when equity recovered to peak
                                   (``None`` if never recovered).
            - duration (int)     : trading days from peak to trough.
        """
        if equity_curve is None or len(equity_curve) < 2:
            return 0.0, None, None, None, 0

        equity = equity_curve.dropna()
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max.replace(0, np.nan)

        max_dd = float(drawdown.min())
        if max_dd >= 0:
            return 0.0, equity.index[0], equity.index[0], equity.index[0], 0

        trough_idx = int(drawdown.argmin())
        trough_date = equity.index[trough_idx]

        # Find peak (last time equity was at the running max before trough)
        peak_level = float(running_max.iloc[trough_idx])
        peak_candidates = equity.iloc[:trough_idx + 1]
        peak_matches = peak_candidates[peak_candidates >= peak_level * 0.9999]
        peak_date = peak_matches.index[0] if len(peak_matches) > 0 else equity.index[0]

        # Find recovery (first bar after trough where equity >= peak)
        post_trough = equity.iloc[trough_idx + 1:]
        recovery_mask = post_trough >= peak_level
        recovery_date = post_trough[recovery_mask].index[0] if recovery_mask.any() else None

        # Duration in trading days (peak → trough)
        peak_pos = equity.index.get_loc(peak_date) if peak_date in equity.index else 0
        duration = trough_idx - peak_pos

        return max_dd, peak_date, trough_date, recovery_date, int(duration)

    # ------------------------------------------------------------------
    # Rolling metrics
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_rolling_sharpe(
        returns: pd.Series,
        window: int = _ROLLING_SHARPE_WINDOW,
    ) -> pd.Series:
        """Annualised rolling Sharpe ratio.

        Parameters
        ----------
        returns:
            Daily return series.
        window:
            Rolling window length (default 63 = ~1 quarter).

        Returns
        -------
        pd.Series of annualised Sharpe ratios, same index as ``returns``.
        """
        if returns is None or len(returns) < window:
            return pd.Series(dtype=float)

        daily_rf = _DEFAULT_RISK_FREE_RATE / _TRADING_DAYS_PER_YEAR
        excess = returns - daily_rf

        rolling_mean = excess.rolling(window).mean()
        rolling_std  = excess.rolling(window).std()

        sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * math.sqrt(
            _TRADING_DAYS_PER_YEAR
        )
        return sharpe

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_report(
        metrics: dict,
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series] = None,
    ) -> str:
        """Generate a human-readable performance report.

        Parameters
        ----------
        metrics:
            Output from :meth:`calculate_all`.
        equity_curve:
            Portfolio equity curve.
        benchmark:
            Optional benchmark equity curve.

        Returns
        -------
        str
            Multi-line text report suitable for terminal output.
        """
        def _fmt(val: Any, pct: bool = False, decimals: int = 2) -> str:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return "  N/A"
            if pct:
                return f"{val:+.{decimals}%}"
            return f"{val:.{decimals}f}"

        start = equity_curve.index[0] if equity_curve is not None and len(equity_curve) > 0 else "?"
        end   = equity_curve.index[-1] if equity_curve is not None and len(equity_curve) > 0 else "?"

        lines = [
            "=" * 62,
            "  ATNN QUANT POWERHOUSE — BACKTEST PERFORMANCE REPORT",
            "=" * 62,
            f"  Period      : {start}  →  {end}",
            f"  Total bars  : {len(equity_curve) if equity_curve is not None else 0}",
            "",
            "  ── RETURNS ────────────────────────────────────────────",
            f"  Total Return   : {_fmt(metrics.get('total_return'), pct=True)}",
            f"  Annual Return  : {_fmt(metrics.get('annual_return'), pct=True)}",
            f"  Volatility     : {_fmt(metrics.get('volatility'), pct=True)}",
            "",
            "  ── RISK-ADJUSTED ───────────────────────────────────────",
            f"  Sharpe Ratio   : {_fmt(metrics.get('sharpe_ratio'))}",
            f"  Sortino Ratio  : {_fmt(metrics.get('sortino_ratio'))}",
            f"  Calmar Ratio   : {_fmt(metrics.get('calmar_ratio'))}",
            "",
            "  ── DRAWDOWN ────────────────────────────────────────────",
            f"  Max Drawdown   : {_fmt(metrics.get('max_drawdown'), pct=True)}",
            f"  DD Duration    : {metrics.get('max_drawdown_duration', 'N/A')} trading days",
            "",
            "  ── TRADE STATISTICS ────────────────────────────────────",
            f"  Total Trades   : {metrics.get('total_trades', 0)}",
            f"  Win Rate       : {_fmt(metrics.get('win_rate'), pct=True)}",
            f"  Profit Factor  : {_fmt(metrics.get('profit_factor'))}",
            f"  Avg Win ($)    : {_fmt(metrics.get('avg_win'))}",
            f"  Avg Loss ($)   : {_fmt(metrics.get('avg_loss'))}",
            f"  Win/Loss Ratio : {_fmt(metrics.get('avg_win_loss_ratio'))}",
            f"  Avg Hold (days): {_fmt(metrics.get('avg_holding_period'))}",
            f"  Turnover (ann) : {_fmt(metrics.get('turnover'), pct=True)}",
            "",
            "  ── DISTRIBUTION ────────────────────────────────────────",
            f"  Skewness       : {_fmt(metrics.get('skewness'))}",
            f"  Kurtosis       : {_fmt(metrics.get('kurtosis'))}",
            f"  VaR (95%)      : {_fmt(metrics.get('var_95'), pct=True)}",
            f"  CVaR (95%)     : {_fmt(metrics.get('cvar_95'), pct=True)}",
        ]

        # Benchmark section (only if available)
        if benchmark is not None and metrics.get("alpha") is not None:
            lines += [
                "",
                "  ── BENCHMARK-RELATIVE ──────────────────────────────────",
                f"  Alpha          : {_fmt(metrics.get('alpha'), pct=True)}",
                f"  Beta           : {_fmt(metrics.get('beta'))}",
                f"  Info Ratio     : {_fmt(metrics.get('information_ratio'))}",
                f"  Tracking Error : {_fmt(metrics.get('tracking_error'), pct=True)}",
            ]

        lines += ["=" * 62]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_trade_metrics(trades: List[dict]) -> dict:
        """Compute win rate, profit factor, avg win/loss from trade list.

        Parameters
        ----------
        trades:
            List of trade dicts.  Each must contain ``pnl`` (float).

        Returns
        -------
        dict with keys: win_rate, profit_factor, avg_win, avg_loss,
        avg_win_loss_ratio, total_trades, avg_holding_period.
        """
        if not trades:
            return {
                "win_rate": float("nan"),
                "profit_factor": float("nan"),
                "avg_win": float("nan"),
                "avg_loss": float("nan"),
                "avg_win_loss_ratio": float("nan"),
                "total_trades": 0,
                "avg_holding_period": float("nan"),
            }

        pnls = [float(t.get("pnl", 0.0)) for t in trades]
        winning = [p for p in pnls if p > 0]
        losing  = [p for p in pnls if p < 0]

        win_rate = len(winning) / len(pnls) if pnls else float("nan")
        gross_profit = sum(winning) if winning else 0.0
        gross_loss   = abs(sum(losing)) if losing else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("nan")

        avg_win  = float(np.mean(winning)) if winning else float("nan")
        avg_loss = float(np.mean(losing))  if losing  else float("nan")

        if not math.isnan(avg_win) and not math.isnan(avg_loss) and avg_loss != 0:
            avg_win_loss_ratio = abs(avg_win / avg_loss)
        else:
            avg_win_loss_ratio = float("nan")

        holding_periods = [
            float(t.get("holding_days", 0.0))
            for t in trades
            if t.get("holding_days") is not None
        ]
        avg_holding = float(np.mean(holding_periods)) if holding_periods else float("nan")

        return {
            "win_rate":           win_rate,
            "profit_factor":      profit_factor,
            "avg_win":            avg_win,
            "avg_loss":           avg_loss,
            "avg_win_loss_ratio": avg_win_loss_ratio,
            "total_trades":       len(pnls),
            "avg_holding_period": avg_holding,
        }

    @staticmethod
    def _calculate_turnover(
        trades: List[dict],
        equity_curve: pd.Series,
    ) -> float:
        """Estimate annualised portfolio turnover from trade list.

        Turnover = sum(|traded notional|) / avg_equity / years

        Parameters
        ----------
        trades:
            Trade records with ``qty`` and ``entry_price`` (optional).
        equity_curve:
            Portfolio equity indexed by date.

        Returns
        -------
        float
            Annualised turnover as a fraction (e.g. 2.0 = 200%).
        """
        if not trades or equity_curve is None or len(equity_curve) < 2:
            return float("nan")

        total_notional = 0.0
        for t in trades:
            qty   = abs(float(t.get("qty", 0)))
            price = float(t.get("entry_price", 0))
            total_notional += qty * price

        avg_equity = float(equity_curve.mean())
        if avg_equity <= 0:
            return float("nan")

        n_days = len(equity_curve)
        years  = n_days / _TRADING_DAYS_PER_YEAR
        if years <= 0:
            return float("nan")

        return total_notional / avg_equity / years

    @staticmethod
    def _calculate_benchmark_metrics(
        returns: pd.Series,
        benchmark: Optional[pd.Series],
    ) -> dict:
        """Compute alpha, beta, information ratio, and tracking error vs benchmark.

        Parameters
        ----------
        returns:
            Portfolio daily returns.
        benchmark:
            Benchmark equity curve (same date index as ``returns``).

        Returns
        -------
        dict with keys: alpha, beta, information_ratio, tracking_error.
        All values are ``nan`` when benchmark is not provided.
        """
        default = {
            "alpha": float("nan"),
            "beta":  float("nan"),
            "information_ratio": float("nan"),
            "tracking_error":    float("nan"),
        }

        if benchmark is None or len(benchmark) < 2:
            return default

        bench_returns = benchmark.pct_change().dropna()

        # Align on shared dates
        shared = returns.index.intersection(bench_returns.index)
        if len(shared) < 10:
            return default

        p = returns.loc[shared]
        b = bench_returns.loc[shared]

        # Beta via linear regression
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(b, p)
            beta  = float(slope)
            alpha_daily = float(intercept)
            # Annualise alpha
            alpha = alpha_daily * _TRADING_DAYS_PER_YEAR
        except Exception:
            return default

        # Active return series
        active_returns = p - b
        tracking_error = float(active_returns.std() * math.sqrt(_TRADING_DAYS_PER_YEAR))
        if tracking_error > 0:
            information_ratio = float(
                active_returns.mean() / active_returns.std() * math.sqrt(_TRADING_DAYS_PER_YEAR)
            )
        else:
            information_ratio = float("nan")

        return {
            "alpha":              alpha,
            "beta":               beta,
            "information_ratio":  information_ratio,
            "tracking_error":     tracking_error,
        }

    @staticmethod
    def _calculate_regime_metrics(
        returns: pd.Series,
        regime_history: Optional[pd.Series],
    ) -> dict:
        """Compute per-regime Sharpe ratios and return contributions.

        Parameters
        ----------
        returns:
            Portfolio daily returns.
        regime_history:
            Series of regime labels (e.g. 'BULL', 'BEAR', 'SIDEWAYS')
            aligned to the same index as returns.

        Returns
        -------
        dict
            Keys: ``regime_sharpe_<regime>``, ``regime_return_<regime>``,
            ``regime_bars_<regime>`` for each regime observed.
            Empty dict when regime_history is not provided.
        """
        if regime_history is None or len(regime_history) == 0:
            return {}
        if returns is None or len(returns) < 2:
            return {}

        # Align on common index
        common = returns.index.intersection(regime_history.index)
        if len(common) < 10:
            return {}

        ret_aligned = returns.loc[common]
        reg_aligned = regime_history.loc[common]

        result: dict = {}
        for regime in reg_aligned.unique():
            mask = reg_aligned == regime
            regime_rets = ret_aligned.loc[mask]
            n_bars = int(mask.sum())
            regime_key = str(regime).lower()

            if n_bars < 5:
                result[f"regime_sharpe_{regime_key}"] = float("nan")
                result[f"regime_return_{regime_key}"] = float("nan")
                result[f"regime_bars_{regime_key}"] = n_bars
                continue

            # Annualised Sharpe for this regime
            mu = float(regime_rets.mean())
            sigma = float(regime_rets.std())
            daily_rf = _DEFAULT_RISK_FREE_RATE / _TRADING_DAYS_PER_YEAR
            if sigma > 0:
                regime_sharpe = (mu - daily_rf) / sigma * math.sqrt(_TRADING_DAYS_PER_YEAR)
            else:
                regime_sharpe = float("nan")

            # Total return contribution from this regime
            regime_total = float(regime_rets.sum())

            result[f"regime_sharpe_{regime_key}"] = round(regime_sharpe, 4)
            result[f"regime_return_{regime_key}"] = round(regime_total, 6)
            result[f"regime_bars_{regime_key}"] = n_bars

        return result


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _empty_metrics() -> dict:
    """Return a metrics dict with all NaN values (used on empty data)."""
    keys = [
        "total_return", "annual_return", "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "max_drawdown_duration", "calmar_ratio",
        "win_rate", "profit_factor", "avg_win", "avg_loss",
        "avg_win_loss_ratio", "total_trades", "avg_holding_period",
        "turnover", "volatility", "skewness", "kurtosis",
        "var_95", "cvar_95", "alpha", "beta",
        "information_ratio", "tracking_error",
    ]
    return {k: float("nan") for k in keys}

"""
Tests for backtest/report.py — HTML report generation.
"""

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from backtest.report import (
    BacktestReport,
    HAS_MPL,
    _CSS,
)


# ---------------------------------------------------------------------------
# Mock result object
# ---------------------------------------------------------------------------

@dataclass
class MockBacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict
    initial_capital: float = 444.0


def _make_result(n_bars: int = 100, n_trades: int = 10) -> MockBacktestResult:
    """Create a mock backtest result for report generation."""
    dates = pd.bdate_range("2023-01-03", periods=n_bars)
    rng = np.random.RandomState(42)
    equity = 444.0 + np.cumsum(rng.randn(n_bars) * 2)
    equity = np.maximum(equity, 100.0)
    eq_series = pd.Series(equity, index=dates, name="equity")

    trades_data = []
    for i in range(n_trades):
        entry = dates[i * (n_bars // n_trades)]
        exit_ = dates[min(i * (n_bars // n_trades) + 5, n_bars - 1)]
        pnl = rng.randn() * 10
        trades_data.append({
            "symbol": "SPY",
            "side": "LONG",
            "entry_date": entry,
            "exit_date": exit_,
            "entry_price": 50.0 + rng.randn(),
            "exit_price": 50.0 + rng.randn(),
            "qty": 5,
            "pnl": pnl,
            "commission": 1.0,
            "holding_days": 5,
            "strategy": "momentum",
        })
    trades_df = pd.DataFrame(trades_data)

    metrics = {
        "total_return": 0.15,
        "cagr": 0.12,
        "sharpe_ratio": 1.2,
        "sortino_ratio": 1.8,
        "max_drawdown": -0.08,
        "max_drawdown_duration": 15,
        "calmar_ratio": 1.5,
        "volatility": 0.18,
        "win_rate": 0.6,
        "profit_factor": 1.8,
        "avg_win": 15.0,
        "avg_loss": -8.0,
        "total_trades": n_trades,
        "avg_holding_period": 5.0,
        "var_95": -0.02,
        "cvar_95": -0.03,
    }

    return MockBacktestResult(
        equity_curve=eq_series,
        trades=trades_df,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBacktestReport:
    def test_generate_creates_file(self):
        result = _make_result()
        report = BacktestReport(title="Test Report")
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            out = report.generate(result, path)
            assert os.path.exists(out)
            assert out == path
            content = open(path).read()
            assert "Test Report" in content
            assert "<!DOCTYPE html>" in content
        finally:
            os.unlink(path)

    def test_html_contains_metrics(self):
        result = _make_result()
        report = BacktestReport()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            report.generate(result, path)
            content = open(path).read()
            assert "Total Return" in content
            assert "Sharpe Ratio" in content
            assert "Max Drawdown" in content
        finally:
            os.unlink(path)

    def test_html_contains_css(self):
        result = _make_result()
        report = BacktestReport()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            report.generate(result, path)
            content = open(path).read()
            assert "--bg-primary" in content
            assert "monospace" in content
        finally:
            os.unlink(path)

    def test_html_contains_trades_table(self):
        result = _make_result(n_trades=5)
        report = BacktestReport()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            report.generate(result, path)
            content = open(path).read()
            assert "Best Trades" in content or "P&amp;L" in content
        finally:
            os.unlink(path)

    @pytest.mark.skipif(not HAS_MPL, reason="matplotlib not available")
    def test_html_contains_charts(self):
        result = _make_result(n_bars=300)
        report = BacktestReport()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            report.generate(result, path)
            content = open(path).read()
            assert "data:image/png;base64" in content
        finally:
            os.unlink(path)

    def test_benchmark_overlay(self):
        result = _make_result()
        report = BacktestReport()
        rng = np.random.RandomState(99)
        benchmark = pd.Series(
            100 + np.cumsum(rng.randn(len(result.equity_curve)) * 0.5),
            index=result.equity_curve.index,
        )
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            report.generate(result, path, benchmark=benchmark)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_empty_equity(self):
        """Report with empty equity should not crash."""
        result = MockBacktestResult(
            equity_curve=pd.Series(dtype=float),
            trades=pd.DataFrame(),
            metrics={"total_return": float("nan")},
        )
        report = BacktestReport()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            report.generate(result, path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_css_dark_theme(self):
        """CSS should have the dark-themed variables."""
        assert "--bg-primary: #0d1117" in _CSS
        assert "--accent-green" in _CSS

    def test_metrics_section_handles_nan(self):
        """Cards should display N/A for NaN values."""
        html = BacktestReport._metrics_section({"total_return": float("nan")})
        assert "N/A" in html

    def test_metrics_section_positive_negative(self):
        html = BacktestReport._metrics_section({
            "total_return": 0.15,
            "max_drawdown": -0.08,
        })
        assert "positive" in html
        assert "negative" in html

    def test_summary_table(self):
        metrics = {"total_return": 0.10, "sharpe_ratio": 1.5, "win_rate": 0.6}
        html = BacktestReport._summary_table(metrics)
        assert "<table>" in html
        assert "Total Return" in html

    def test_output_dir_creation(self):
        """Should create parent directories if needed."""
        result = _make_result()
        report = BacktestReport()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "report.html")
            report.generate(result, path)
            assert os.path.exists(path)

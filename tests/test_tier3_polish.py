"""
TIER 3 Polish Modules — Integration Test Suite
=================================================

Tests for:
1. LiveDashboard — Flask/Plotly dashboard
2. AutomatedReporter — daily/weekly email reports
3. TaxLossHarvesting — cost basis & harvesting suggestions
4. BenchmarkTracker — alpha/beta vs SPY/QQQ
"""

import os
import sys
import json
import random
import unittest
import tempfile
from datetime import date, timedelta, datetime
from unittest.mock import patch, MagicMock
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.live_dashboard import (
    LiveDashboard,
    DashboardConfig,
    PortfolioSnapshot,
    TradeRecord,
    ChartBuilder,
    FLASK_AVAILABLE,
)
from src.automated_reporting import (
    AutomatedReporter,
    ReportConfig,
    PerformanceCalculator,
    PerformanceMetrics,
    ReportRenderer,
    TradeSummary,
    Report,
)
from src.tax_loss_harvesting import (
    TaxLotTracker,
    HarvestingEngine,
    TaxConfig,
    TaxLot,
    CostBasisMethod,
    GainType,
    WashSaleDetector,
)
from src.benchmark_tracker import (
    BenchmarkTracker,
    BenchmarkConfig,
    BenchmarkComparison,
    AlphaBetaResult,
    CaptureRatios,
    _StatUtils,
)


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _make_dates(n: int, start_days_ago: int = 0) -> list:
    """Generate date strings."""
    base = date.today() - timedelta(days=start_days_ago + n)
    return [(base + timedelta(days=i)).isoformat() for i in range(n)]


def _make_returns(n: int, mu: float = 0.0005, sigma: float = 0.012) -> list:
    """Generate random daily returns."""
    np.random.seed(42)
    return list(np.random.normal(mu, sigma, n))


def _make_values(n: int, start: float = 100_000.0) -> list:
    """Generate random portfolio value series."""
    np.random.seed(42)
    rets = np.random.normal(0.0005, 0.012, n)
    vals = [start]
    for r in rets:
        vals.append(vals[-1] * (1 + r))
    return vals


def _make_trades(n: int) -> list:
    """Generate random TradeSummary list."""
    random.seed(42)
    trades = []
    for i in range(n):
        trades.append(TradeSummary(
            timestamp=(date.today() - timedelta(days=n - i)).isoformat(),
            symbol=random.choice(["SPY", "QQQ", "AAPL", "MSFT", "IWM"]),
            side=random.choice(["BUY", "SELL"]),
            quantity=random.randint(10, 100),
            price=random.uniform(100, 500),
            pnl=random.gauss(30, 200),
            strategy=random.choice(["TDA_momentum", "mean_reversion", "ensemble"]),
        ))
    return trades


# ═════════════════════════════════════════════════════════════════════════════
# 1. LIVE DASHBOARD TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestLiveDashboard(unittest.TestCase):
    """Tests for LiveDashboard."""

    def test_init_default(self):
        dash = LiveDashboard()
        self.assertIsNotNone(dash.config)
        self.assertEqual(dash.config.port, 5050)
        self.assertFalse(dash.is_running)

    def test_init_custom_config(self):
        cfg = DashboardConfig(port=8888, theme="light", title="Test")
        dash = LiveDashboard(cfg)
        self.assertEqual(dash.config.port, 8888)
        self.assertEqual(dash.config.theme, "light")

    def test_update_portfolio(self):
        dash = LiveDashboard()
        snap = PortfolioSnapshot(
            portfolio_value=105_000,
            cash=20_000,
            daily_pnl=500,
            total_return_pct=5.0,
        )
        dash.update_portfolio(snap)
        self.assertIsNotNone(dash.get_snapshot())
        self.assertEqual(dash.get_snapshot().portfolio_value, 105_000)

    def test_equity_history(self):
        dash = LiveDashboard()
        for i in range(10):
            dash.update_portfolio(PortfolioSnapshot(
                portfolio_value=100_000 + i * 1000,
                daily_pnl=1000,
            ))
        ts, vals, dds = dash.get_equity_history()
        self.assertEqual(len(ts), 10)
        self.assertEqual(len(vals), 10)
        self.assertAlmostEqual(vals[-1], 109_000)

    def test_drawdown_tracking(self):
        dash = LiveDashboard()
        dash.update_portfolio(PortfolioSnapshot(portfolio_value=100_000))
        dash.update_portfolio(PortfolioSnapshot(portfolio_value=110_000))
        dash.update_portfolio(PortfolioSnapshot(portfolio_value=100_000))
        _, _, dds = dash.get_equity_history()
        self.assertAlmostEqual(dds[-1], (110_000 - 100_000) / 110_000 * 100, places=2)

    def test_record_trade(self):
        dash = LiveDashboard()
        trade = TradeRecord(symbol="SPY", side="BUY", quantity=100, price=450.0, pnl=150.0)
        dash.record_trade(trade)
        trades = dash.get_trades()
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].symbol, "SPY")

    def test_multiple_trades(self):
        dash = LiveDashboard()
        for i in range(5):
            dash.record_trade(TradeRecord(symbol=f"SYM{i}", pnl=i * 10))
        self.assertEqual(len(dash.get_trades()), 5)

    @unittest.skipUnless(FLASK_AVAILABLE, "Flask not installed")
    def test_chart_equity_curve(self):
        chart_json = ChartBuilder.equity_curve(
            ["2026-01-01", "2026-01-02", "2026-01-03"],
            [100_000, 101_000, 102_500],
            [0, 0, 0.5],
        )
        data = json.loads(chart_json)
        self.assertIn("data", data)
        self.assertEqual(len(data["data"]), 2)  # equity + drawdown

    @unittest.skipUnless(FLASK_AVAILABLE, "Flask not installed")
    def test_chart_position_heatmap(self):
        chart_json = ChartBuilder.position_heatmap({
            "SPY": {"weight_pct": 30, "unrealized_pnl": 500},
            "QQQ": {"weight_pct": 20, "unrealized_pnl": -200},
        })
        data = json.loads(chart_json)
        self.assertIn("data", data)

    @unittest.skipUnless(FLASK_AVAILABLE, "Flask not installed")
    def test_chart_risk_gauges(self):
        chart_json = ChartBuilder.risk_gauges(2.1, 3.5, 2.0, 58.0)
        data = json.loads(chart_json)
        self.assertIn("data", data)
        self.assertEqual(len(data["data"]), 4)  # 4 gauges

    @unittest.skipUnless(FLASK_AVAILABLE, "Flask not installed")
    def test_state_payload(self):
        dash = LiveDashboard()
        for i in range(5):
            dash.update_portfolio(PortfolioSnapshot(
                portfolio_value=100_000 + i * 500,
                sharpe_ratio=1.8,
                max_drawdown_pct=2.0,
                var_95=1.5,
                win_rate=55.0,
            ))
        payload = dash._build_state_payload()
        self.assertIn("snapshot", payload)
        self.assertIn("equity_chart", payload)
        self.assertIn("trades", payload)

    def test_daily_pnl_tracking(self):
        dash = LiveDashboard()
        # Two updates same day -> should aggregate
        dash.update_portfolio(PortfolioSnapshot(
            timestamp="2026-02-15 10:00:00",
            portfolio_value=100_000,
            daily_pnl=200,
        ))
        dash.update_portfolio(PortfolioSnapshot(
            timestamp="2026-02-15 14:00:00",
            portfolio_value=100_500,
            daily_pnl=500,
        ))
        self.assertEqual(len(dash._daily_pnl_dates), 1)
        self.assertEqual(dash._daily_pnl_values[-1], 500)


# ═════════════════════════════════════════════════════════════════════════════
# 2. AUTOMATED REPORTING TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestAutomatedReporting(unittest.TestCase):
    """Tests for AutomatedReporter."""

    def test_performance_calculator_basic(self):
        values = [100_000, 101_000, 102_000, 101_500, 103_000]
        timestamps = _make_dates(5)
        trades = _make_trades(3)
        metrics = PerformanceCalculator.compute(values, timestamps, trades, "daily", "test")
        self.assertEqual(metrics.period, "test")
        self.assertAlmostEqual(metrics.start_value, 100_000)
        self.assertAlmostEqual(metrics.end_value, 103_000)
        self.assertGreater(metrics.return_pct, 0)

    def test_performance_sharpe(self):
        values = _make_values(100)
        timestamps = _make_dates(100)
        trades = _make_trades(20)
        metrics = PerformanceCalculator.compute(values, timestamps, trades)
        # Sharpe should be computed and finite
        self.assertTrue(np.isfinite(metrics.sharpe_ratio))

    def test_performance_drawdown(self):
        values = [100_000, 110_000, 105_000, 108_000]
        timestamps = _make_dates(4)
        metrics = PerformanceCalculator.compute(values, timestamps, [])
        self.assertGreater(metrics.max_drawdown_pct, 0)

    def test_performance_win_rate(self):
        trades = [
            TradeSummary(pnl=100),
            TradeSummary(pnl=200),
            TradeSummary(pnl=-50),
        ]
        metrics = PerformanceCalculator.compute([100, 200, 300, 400], _make_dates(4), trades)
        self.assertAlmostEqual(metrics.win_rate, 66.67, places=1)

    def test_performance_profit_factor(self):
        trades = [
            TradeSummary(pnl=300),
            TradeSummary(pnl=-100),
        ]
        metrics = PerformanceCalculator.compute([100, 200, 300], _make_dates(3), trades)
        self.assertAlmostEqual(metrics.profit_factor, 3.0)

    def test_performance_empty(self):
        metrics = PerformanceCalculator.compute([], [], [])
        self.assertEqual(metrics.sharpe_ratio, 0)

    def test_generate_daily_report(self):
        reporter = AutomatedReporter()
        values = _make_values(30)
        timestamps = _make_dates(31)
        trades = _make_trades(10)
        report = reporter.generate_daily_report(values, timestamps, trades)
        self.assertIn("Daily Report", report.title)
        self.assertIsNotNone(report.metrics)
        self.assertGreater(len(report.html_content), 100)

    def test_generate_weekly_report(self):
        reporter = AutomatedReporter()
        report = reporter.generate_weekly_report(
            _make_values(7), _make_dates(8), _make_trades(5), "2026-W07"
        )
        self.assertIn("Weekly", report.title)

    def test_generate_monthly_report(self):
        reporter = AutomatedReporter()
        report = reporter.generate_monthly_report(
            _make_values(30), _make_dates(31), _make_trades(15), "2026-02"
        )
        self.assertIn("Monthly", report.title)

    def test_save_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ReportConfig(report_dir=tmpdir)
            reporter = AutomatedReporter(cfg)
            report = reporter.generate_daily_report(
                _make_values(10), _make_dates(11), _make_trades(3)
            )
            path = reporter.save_report(report)
            self.assertTrue(os.path.exists(path))
            content = Path(path).read_text()
            self.assertIn("Daily Report", content)

    def test_html_render(self):
        metrics = PerformanceMetrics(
            period="2026-02-15",
            pnl=1500,
            return_pct=1.5,
            sharpe_ratio=2.1,
            win_rate=58.0,
            max_drawdown_pct=3.2,
            top_contributors=[("SPY", 800), ("QQQ", 400)],
            bottom_contributors=[("IWM", -200)],
        )
        html = ReportRenderer.render(metrics, _make_trades(5))
        self.assertIn("Daily Report", html)
        self.assertIn("1,500", html)
        self.assertIn("SPY", html)

    def test_trades_csv(self):
        reporter = AutomatedReporter()
        trades = _make_trades(5)
        csv_str = reporter._trades_to_csv(trades)
        self.assertIn("timestamp", csv_str)
        self.assertIn("symbol", csv_str)
        lines = csv_str.strip().split("\n")
        self.assertEqual(len(lines), 6)  # header + 5 rows

    def test_email_no_smtp(self):
        reporter = AutomatedReporter(ReportConfig())
        report = Report(title="Test", html_content="<p>test</p>")
        result = reporter.send_email(report)
        self.assertFalse(result)  # no smtp configured

    def test_report_history(self):
        reporter = AutomatedReporter()
        reporter.generate_daily_report(_make_values(5), _make_dates(6), [])
        reporter.generate_weekly_report(_make_values(5), _make_dates(6), [])
        self.assertEqual(len(reporter.get_history()), 2)


# ═════════════════════════════════════════════════════════════════════════════
# 3. TAX-LOSS HARVESTING TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestTaxLossHarvesting(unittest.TestCase):
    """Tests for TaxLotTracker and HarvestingEngine."""

    def test_add_lot(self):
        tracker = TaxLotTracker()
        lot = tracker.add_lot("AAPL", 50, 175.0, "2025-03-15")
        self.assertEqual(lot.symbol, "AAPL")
        self.assertEqual(lot.quantity, 50)
        self.assertEqual(lot.remaining_quantity, 50)
        self.assertAlmostEqual(lot.total_cost, 50 * 175.0)

    def test_get_lots(self):
        tracker = TaxLotTracker()
        tracker.add_lot("AAPL", 50, 175.0, "2025-03-15")
        tracker.add_lot("AAPL", 30, 190.0, "2025-08-20")
        tracker.add_lot("SPY", 100, 450.0, "2025-01-10")
        self.assertEqual(len(tracker.get_lots("AAPL")), 2)
        self.assertEqual(len(tracker.get_lots("SPY")), 1)
        self.assertEqual(len(tracker.get_lots()), 3)

    def test_sell_fifo(self):
        tracker = TaxLotTracker(TaxConfig(cost_basis_method=CostBasisMethod.FIFO))
        tracker.add_lot("AAPL", 50, 170.0, "2025-01-01")
        tracker.add_lot("AAPL", 50, 190.0, "2025-06-01")
        records = tracker.sell_shares("AAPL", 60, 180.0, "2026-02-15")
        # FIFO: sells from first lot (170) first
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].quantity, 50)  # all of first lot
        self.assertAlmostEqual(records[0].realized_pnl, 50 * (180 - 170))
        self.assertEqual(records[1].quantity, 10)   # 10 from second lot
        # Remaining: first lot 0, second lot 40
        self.assertEqual(tracker.get_total_shares("AAPL"), 40)

    def test_sell_lifo(self):
        tracker = TaxLotTracker(TaxConfig(cost_basis_method=CostBasisMethod.LIFO))
        tracker.add_lot("AAPL", 50, 170.0, "2025-01-01")
        tracker.add_lot("AAPL", 50, 190.0, "2025-06-01")
        records = tracker.sell_shares("AAPL", 30, 180.0, "2026-02-15")
        # LIFO: sells from second lot (190) first
        self.assertEqual(records[0].quantity, 30)
        self.assertAlmostEqual(records[0].realized_pnl, 30 * (180 - 190))  # loss
        self.assertEqual(tracker.get_total_shares("AAPL"), 70)

    def test_cost_basis_average(self):
        tracker = TaxLotTracker()
        tracker.add_lot("AAPL", 50, 170.0, "2025-01-01")
        tracker.add_lot("AAPL", 50, 190.0, "2025-06-01")
        avg = tracker.get_cost_basis("AAPL")
        self.assertAlmostEqual(avg, 180.0)

    def test_unrealized_pnl(self):
        tracker = TaxLotTracker()
        tracker.add_lot("AAPL", 100, 180.0, "2025-03-15")
        pnl = tracker.unrealized_pnl("AAPL", 170.0)
        self.assertAlmostEqual(pnl, 100 * (170 - 180))
        pnl_pos = tracker.unrealized_pnl("AAPL", 200.0)
        self.assertAlmostEqual(pnl_pos, 100 * (200 - 180))

    def test_unrealized_pnl_by_lot(self):
        tracker = TaxLotTracker()
        tracker.add_lot("AAPL", 50, 170.0, "2025-01-01")
        tracker.add_lot("AAPL", 30, 190.0, "2025-08-01")
        results = tracker.unrealized_pnl_by_lot("AAPL", 180.0)
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0][1], 50 * (180 - 170))
        self.assertAlmostEqual(results[1][1], 30 * (180 - 190))

    def test_gain_type_short_term(self):
        lot = TaxLot(purchase_date=(date.today() - timedelta(days=100)).isoformat())
        self.assertEqual(lot.gain_type, GainType.SHORT_TERM)

    def test_gain_type_long_term(self):
        lot = TaxLot(purchase_date=(date.today() - timedelta(days=400)).isoformat())
        self.assertEqual(lot.gain_type, GainType.LONG_TERM)

    def test_symbols(self):
        tracker = TaxLotTracker()
        tracker.add_lot("AAPL", 50, 170.0, "2025-01-01")
        tracker.add_lot("SPY", 100, 450.0, "2025-01-01")
        self.assertIn("AAPL", tracker.symbols())
        self.assertIn("SPY", tracker.symbols())

    def test_wash_sale_detector_recent_sale(self):
        detector = WashSaleDetector(window_days=30)
        from src.tax_loss_harvesting import SaleRecord
        sales = [SaleRecord(symbol="AAPL", sale_date="2026-02-10", realized_pnl=-500)]
        result = detector.check_wash_sale("AAPL", "2026-02-15", [], sales)
        self.assertTrue(result)

    def test_wash_sale_detector_clear(self):
        detector = WashSaleDetector(window_days=30)
        from src.tax_loss_harvesting import SaleRecord
        sales = [SaleRecord(symbol="AAPL", sale_date="2025-12-01", realized_pnl=-500)]
        result = detector.check_wash_sale("AAPL", "2026-02-15", [], sales)
        self.assertFalse(result)

    def test_harvesting_suggestions(self):
        tracker = TaxLotTracker()
        tracker.add_lot("AAPL", 100, 200.0, "2025-06-01")  # cost $200
        tracker.add_lot("SPY", 50, 480.0, "2025-01-01")     # cost $480

        engine = HarvestingEngine(tracker)
        suggestions = engine.suggest_harvesting({
            "AAPL": 160.0,  # loss of $40/share = $4000
            "SPY": 490.0,   # gain
        })
        # Only AAPL should be suggested (it's at a loss)
        aapl_suggestions = [s for s in suggestions if s.symbol == "AAPL"]
        self.assertGreater(len(aapl_suggestions), 0)
        self.assertLess(aapl_suggestions[0].unrealized_loss, 0)
        self.assertGreater(aapl_suggestions[0].estimated_tax_savings, 0)

    def test_harvesting_min_threshold(self):
        tracker = TaxLotTracker(TaxConfig(harvest_threshold_pct=50.0))  # very high threshold
        tracker.add_lot("AAPL", 100, 200.0, "2025-06-01")
        engine = HarvestingEngine(tracker)
        suggestions = engine.suggest_harvesting({"AAPL": 190.0})  # only 5% loss
        self.assertEqual(len(suggestions), 0)  # below 50% threshold

    def test_harvesting_substitutes(self):
        tracker = TaxLotTracker()
        tracker.add_lot("SPY", 100, 500.0, "2025-03-01")
        engine = HarvestingEngine(tracker)
        suggestions = engine.suggest_harvesting({"SPY": 420.0})
        if suggestions:
            self.assertIn("VOO", suggestions[0].substitute_securities)

    def test_annual_summary(self):
        tracker = TaxLotTracker()
        tracker.add_lot("AAPL", 100, 200.0, "2025-06-01")
        tracker.add_lot("SPY", 50, 480.0, "2025-01-01")
        tracker.sell_shares("SPY", 20, 490.0, "2026-01-15")

        engine = HarvestingEngine(tracker)
        summary = engine.annual_summary({"AAPL": 160.0, "SPY": 490.0}, year=2026)
        self.assertEqual(summary.year, 2026)
        self.assertGreater(summary.realized_gains, 0)  # SPY sale at gain
        self.assertLess(summary.total_unrealized_losses, 0)  # AAPL at loss

    def test_sale_records(self):
        tracker = TaxLotTracker()
        tracker.add_lot("AAPL", 50, 170.0, "2025-01-01")
        tracker.sell_shares("AAPL", 20, 180.0, "2026-02-15")
        sales = tracker.get_sales("AAPL")
        self.assertEqual(len(sales), 1)
        self.assertAlmostEqual(sales[0].realized_pnl, 20 * (180 - 170))


# ═════════════════════════════════════════════════════════════════════════════
# 4. BENCHMARK TRACKER TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestBenchmarkTracker(unittest.TestCase):
    """Tests for BenchmarkTracker."""

    def _setup_tracker(self, n: int = 252):
        """Create a tracker with synthetic data."""
        np.random.seed(42)
        dates = _make_dates(n)
        mkt = list(np.random.normal(0.0004, 0.012, n))
        port = [0.0003 + 1.15 * m + np.random.normal(0, 0.005) for m in mkt]
        qqq = [m * 1.3 + np.random.normal(0, 0.005) for m in mkt]

        tracker = BenchmarkTracker()
        tracker.set_portfolio_returns(dates, port)
        tracker.add_benchmark("SPY", dates, mkt)
        tracker.add_benchmark("QQQ", dates, qqq)
        return tracker

    def test_init(self):
        tracker = BenchmarkTracker()
        self.assertIsNotNone(tracker.config)

    def test_set_portfolio(self):
        tracker = BenchmarkTracker()
        dates = _make_dates(50)
        rets = _make_returns(50)
        tracker.set_portfolio_returns(dates, rets)
        self.assertEqual(len(tracker._portfolio_returns), 50)

    def test_set_portfolio_values(self):
        tracker = BenchmarkTracker()
        dates = _make_dates(51)
        vals = _make_values(50)
        tracker.set_portfolio_values(dates, vals)
        self.assertEqual(len(tracker._portfolio_returns), 50)

    def test_add_benchmark(self):
        tracker = BenchmarkTracker()
        tracker.add_benchmark("SPY", _make_dates(50), _make_returns(50))
        self.assertIn("SPY", tracker._benchmarks)

    def test_compare_spy(self):
        tracker = self._setup_tracker(100)
        comp = tracker.compare("SPY")
        self.assertEqual(comp.benchmark_name, "SPY")
        self.assertNotEqual(comp.portfolio_return, 0)
        self.assertNotEqual(comp.benchmark_return, 0)

    def test_alpha_beta(self):
        tracker = self._setup_tracker(200)
        comp = tracker.compare("SPY")
        ab = comp.alpha_beta
        self.assertIsNotNone(ab)
        # Beta should be close to 1.15 (our synthetic data param)
        self.assertGreater(ab.beta, 0.5)
        self.assertLess(ab.beta, 2.0)
        # R-squared should be meaningful
        self.assertGreater(ab.r_squared, 0)

    def test_information_ratio(self):
        tracker = self._setup_tracker(200)
        comp = tracker.compare("SPY")
        self.assertTrue(np.isfinite(comp.information_ratio))

    def test_tracking_error(self):
        tracker = self._setup_tracker(200)
        comp = tracker.compare("SPY")
        self.assertGreater(comp.tracking_error, 0)

    def test_capture_ratios(self):
        tracker = self._setup_tracker(200)
        comp = tracker.compare("SPY")
        cap = comp.capture_ratios
        self.assertIsNotNone(cap)
        self.assertGreater(cap.up_capture, 0)
        self.assertGreater(cap.down_capture, 0)

    def test_correlation(self):
        tracker = self._setup_tracker(200)
        comp = tracker.compare("SPY")
        self.assertGreater(comp.correlation, 0.3)  # should be correlated

    def test_rolling_alpha_beta(self):
        tracker = self._setup_tracker(200)
        comp = tracker.compare("SPY")
        self.assertGreater(len(comp.rolling_alpha), 0)
        self.assertEqual(len(comp.rolling_alpha), len(comp.rolling_beta))

    def test_full_report(self):
        tracker = self._setup_tracker(200)
        report = tracker.full_report()
        self.assertIn("SPY", report)
        self.assertIn("QQQ", report)

    def test_print_report(self):
        tracker = self._setup_tracker(100)
        text = tracker.print_report()
        self.assertIn("SPY", text)
        self.assertIn("Alpha", text)
        self.assertIn("Beta", text)

    def test_to_dict(self):
        tracker = self._setup_tracker(100)
        d = tracker.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("SPY", d)

    def test_missing_benchmark(self):
        tracker = BenchmarkTracker()
        tracker.set_portfolio_returns(_make_dates(50), _make_returns(50))
        comp = tracker.compare("NONEXIST")
        self.assertEqual(comp.benchmark_name, "NONEXIST")
        self.assertEqual(comp.portfolio_return, 0)

    def test_stat_utils_ols(self):
        np.random.seed(42)
        x = np.random.normal(0, 0.01, 100)
        y = 0.0002 + 1.1 * x + np.random.normal(0, 0.005, 100)
        result = _StatUtils.ols_alpha_beta(y, x)
        self.assertGreater(result.beta, 0.5)
        self.assertGreater(result.r_squared, 0.3)

    def test_stat_utils_capture(self):
        np.random.seed(42)
        bench = np.random.normal(0, 0.01, 100)
        port = 1.2 * bench + np.random.normal(0, 0.003, 100)
        cap = _StatUtils.capture_ratios(port, bench)
        self.assertGreater(cap.up_capture, 80)

    def test_stat_utils_max_dd(self):
        vals = np.array([100, 110, 105, 108, 112, 100])
        dd = _StatUtils.max_drawdown(vals)
        expected = (112 - 100) / 112 * 100
        self.assertAlmostEqual(dd, expected, places=1)

    def test_stat_utils_sharpe(self):
        rets = np.array([0.01, -0.005, 0.008, 0.003, -0.002])
        s = _StatUtils.sharpe(rets)
        self.assertTrue(np.isfinite(s))


# ═════════════════════════════════════════════════════════════════════════════
# 5. INTEGRATION TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestTier3Integration(unittest.TestCase):
    """Cross-module integration tests."""

    def test_dashboard_to_report(self):
        """Dashboard data feeds into report generation."""
        dash = LiveDashboard()
        values = []
        for i in range(30):
            v = 100_000 + i * 300 + random.gauss(0, 500)
            dash.update_portfolio(PortfolioSnapshot(
                portfolio_value=v, daily_pnl=300,
            ))
            values.append(v)

        # Use dashboard equity history for report
        ts, vals, _ = dash.get_equity_history()
        reporter = AutomatedReporter()
        report = reporter.generate_daily_report(vals, ts, [])
        self.assertIsNotNone(report.metrics)
        self.assertGreater(len(report.html_content), 0)

    def test_tax_lot_to_benchmark(self):
        """Tax lot tracking alongside benchmark comparison."""
        tracker = TaxLotTracker()
        tracker.add_lot("SPY", 100, 450.0, "2025-01-01")

        # Benchmark comparison
        dates = _make_dates(100)
        bench = BenchmarkTracker()
        bench.set_portfolio_returns(dates, _make_returns(100))
        bench.add_benchmark("SPY", dates, _make_returns(100))
        comp = bench.compare("SPY")

        self.assertIsNotNone(comp.alpha_beta)
        self.assertEqual(tracker.get_total_shares("SPY"), 100)

    def test_full_pipeline(self):
        """Full pipeline: dashboard -> reporting -> tax -> benchmark."""
        # 1. Dashboard collects data
        dash = LiveDashboard()
        n = 60
        np.random.seed(42)
        sim_dates = _make_dates(n)

        values = [100_000.0]
        for i in range(n):
            ret = np.random.normal(0.0005, 0.012)
            values.append(values[-1] * (1 + ret))
            dash.update_portfolio(PortfolioSnapshot(
                timestamp=sim_dates[i],
                portfolio_value=values[-1],
                daily_pnl=values[-1] * ret,
            ))

        # 2. Reporter generates report
        ts, vals, _ = dash.get_equity_history()
        reporter = AutomatedReporter()
        report = reporter.generate_daily_report(vals, ts, [])
        self.assertGreater(report.metrics.return_pct, -50)  # sanity check

        # 3. Tax tracking
        tax = TaxLotTracker()
        tax.add_lot("SPY", 100, 450.0, "2025-06-01")
        tax.add_lot("QQQ", 50, 400.0, "2025-06-01")
        engine = HarvestingEngine(tax)
        suggestions = engine.suggest_harvesting({"SPY": 430.0, "QQQ": 380.0})
        # Both at a loss, at least one suggestion expected
        self.assertGreater(len(suggestions), 0)

        # 4. Benchmark comparison using same date strings
        arr = np.array(vals, dtype=float)
        rets = list(np.diff(arr) / arr[:-1])
        ret_dates = ts[1:]  # returns are 1 shorter than values
        bench = BenchmarkTracker()
        bench.set_portfolio_returns(ret_dates, rets)
        bench.add_benchmark("SPY", ret_dates, _make_returns(len(ret_dates)))
        comp = bench.compare("SPY")
        self.assertIsNotNone(comp.alpha_beta)


# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)

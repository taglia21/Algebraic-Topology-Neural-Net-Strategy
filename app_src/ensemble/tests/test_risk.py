"""Tests for EnsembleRiskManager."""

from __future__ import annotations

import pytest

from ensemble.risk_manager import EnsembleRiskManager, PositionSize, RiskReport


class TestKellyFraction:
    """Tests for Kelly criterion calculation."""

    def test_positive_edge(self):
        """Positive edge should return positive Kelly fraction."""
        k = EnsembleRiskManager.kelly_fraction(0.60, 0.02, 0.015)
        assert k > 0

    def test_no_edge(self):
        """No edge (50/50 with equal payoff) should return 0."""
        k = EnsembleRiskManager.kelly_fraction(0.50, 0.01, 0.01)
        assert k == pytest.approx(0.0)

    def test_negative_edge(self):
        """Negative edge should return 0."""
        k = EnsembleRiskManager.kelly_fraction(0.30, 0.01, 0.02)
        assert k == 0.0

    def test_invalid_inputs(self):
        """Edge cases: zero/negative inputs should return 0."""
        assert EnsembleRiskManager.kelly_fraction(0, 0.02, 0.01) == 0.0
        assert EnsembleRiskManager.kelly_fraction(1.0, 0.02, 0.01) == 0.0
        assert EnsembleRiskManager.kelly_fraction(0.5, 0, 0.01) == 0.0
        assert EnsembleRiskManager.kelly_fraction(0.5, 0.01, 0) == 0.0


class TestPositionSizing:
    """Tests for position sizing."""

    @pytest.fixture()
    def risk_mgr(self) -> EnsembleRiskManager:
        return EnsembleRiskManager()

    def test_neutral_signal_zero_size(self, risk_mgr):
        """NEUTRAL signals should get zero position."""
        signal = {"ticker": "AAPL", "direction": "NEUTRAL", "strength": 0.5}
        pos = risk_mgr.size_position(
            signal, 10000, {"gross_pct": 0.0}, regime="NORMAL"
        )
        assert pos.position_pct == 0.0
        assert pos.position_value == 0.0

    def test_normal_regime_sizing(self, risk_mgr):
        """Normal regime: position within expected range."""
        signal = {"ticker": "AAPL", "direction": "LONG", "strength": 0.8}
        pos = risk_mgr.size_position(
            signal, 10000, {"gross_pct": 0.0}, regime="NORMAL"
        )
        assert 0 < pos.position_pct <= 5.0
        assert 0 < pos.position_value <= 500

    def test_stressed_regime_caps(self, risk_mgr):
        """STRESSED regime: max 3% per position."""
        signal = {"ticker": "AAPL", "direction": "LONG", "strength": 1.0}
        pos = risk_mgr.size_position(
            signal, 100000, {"gross_pct": 0.0}, regime="STRESSED"
        )
        assert pos.position_pct <= 3.0

    def test_crash_regime_caps(self, risk_mgr):
        """CRASH regime: max 1% per position."""
        signal = {"ticker": "AAPL", "direction": "LONG", "strength": 1.0}
        pos = risk_mgr.size_position(
            signal, 100000, {"gross_pct": 0.0}, regime="CRASH"
        )
        assert pos.position_pct <= 1.0

    def test_exposure_headroom(self, risk_mgr):
        """Position should be limited by remaining exposure headroom."""
        signal = {"ticker": "AAPL", "direction": "LONG", "strength": 1.0}
        # NORMAL: 100% max exposure, already at 99%
        pos = risk_mgr.size_position(
            signal, 10000, {"gross_pct": 99.0}, regime="NORMAL"
        )
        assert pos.position_pct <= 1.0
        assert pos.capped

    def test_small_account_constraints(self):
        """$444 account: max $50 risk per trade, $100 equity position."""
        rm = EnsembleRiskManager(
            max_risk_per_trade=50.0,
            max_equity_position=100.0,
        )
        signal = {"ticker": "AAPL", "direction": "LONG", "strength": 1.0}
        pos = rm.size_position(
            signal, 444, {"gross_pct": 0.0}, regime="NORMAL"
        )
        assert pos.position_value <= 50.0

    def test_position_size_dataclass(self, risk_mgr):
        """PositionSize should have correct fields."""
        signal = {"ticker": "AAPL", "direction": "LONG", "strength": 0.5}
        pos = risk_mgr.size_position(
            signal, 10000, {"gross_pct": 0.0}, regime="NORMAL"
        )
        assert isinstance(pos, PositionSize)
        assert pos.ticker == "AAPL"
        assert pos.direction == "LONG"
        assert pos.regime == "NORMAL"


class TestPortfolioRisk:
    """Tests for portfolio risk checking."""

    @pytest.fixture()
    def risk_mgr(self) -> EnsembleRiskManager:
        return EnsembleRiskManager()

    def test_clean_portfolio(self, risk_mgr):
        """Portfolio within all limits → no violations."""
        positions = [
            {"ticker": "AAPL", "direction": "LONG", "value": 200, "sector": "Tech"},
            {"ticker": "MSFT", "direction": "LONG", "value": 300, "sector": "Tech"},
        ]
        report = risk_mgr.check_portfolio_risk(positions, 10000)
        assert isinstance(report, RiskReport)
        assert len(report.violations) == 0
        assert len(report.warnings) == 0

    def test_position_concentration_violation(self, risk_mgr):
        """Single position > 5% should trigger violation."""
        positions = [
            {"ticker": "AAPL", "direction": "LONG", "value": 600, "sector": "Tech"},
        ]
        report = risk_mgr.check_portfolio_risk(positions, 10000)
        assert any("POSITION_CONCENTRATION" in v for v in report.violations)

    def test_gross_exposure_violation(self, risk_mgr):
        """Gross > 130% should trigger violation."""
        positions = [
            {"ticker": "AAPL", "direction": "LONG", "value": 7000, "sector": "Tech"},
            {"ticker": "MSFT", "direction": "SHORT", "value": 7000, "sector": "Tech"},
        ]
        report = risk_mgr.check_portfolio_risk(positions, 10000)
        assert any("GROSS_EXPOSURE" in v for v in report.violations)

    def test_sector_exposure_violation(self, risk_mgr):
        """Sector > 20% should trigger violation."""
        positions = [
            {"ticker": "AAPL", "direction": "LONG", "value": 2500, "sector": "Tech"},
        ]
        report = risk_mgr.check_portfolio_risk(positions, 10000)
        assert any("SECTOR_EXPOSURE" in v for v in report.violations)

    def test_daily_loss_flatten(self, risk_mgr):
        """5% daily loss → flatten violation."""
        report = risk_mgr.check_portfolio_risk([], 10000, daily_pnl_pct=-5.5)
        assert any("DAILY_LOSS_FLATTEN" in v for v in report.violations)

    def test_daily_loss_reduce(self, risk_mgr):
        """3% daily loss → reduce warning."""
        report = risk_mgr.check_portfolio_risk([], 10000, daily_pnl_pct=-3.5)
        assert any("DAILY_LOSS_REDUCE" in w for w in report.warnings)

    def test_max_drawdown_halt(self, risk_mgr):
        """15% drawdown → halt violation."""
        report = risk_mgr.check_portfolio_risk([], 10000, drawdown_pct=16.0)
        assert any("MAX_DRAWDOWN_HALT" in v for v in report.violations)

    def test_long_exposure_limit(self, risk_mgr):
        """Long exposure > 100% → violation."""
        positions = [
            {"ticker": "AAPL", "direction": "LONG", "value": 11000, "sector": "Tech"},
        ]
        report = risk_mgr.check_portfolio_risk(positions, 10000)
        assert any("LONG_EXPOSURE" in v for v in report.violations)

    def test_short_exposure_limit(self, risk_mgr):
        """Short exposure > 50% → violation."""
        positions = [
            {"ticker": "AAPL", "direction": "SHORT", "value": 6000, "sector": "Tech"},
        ]
        report = risk_mgr.check_portfolio_risk(positions, 10000)
        assert any("SHORT_EXPOSURE" in v for v in report.violations)

    def test_exposure_calculations(self, risk_mgr):
        """Verify long/short/gross/net exposure calculations."""
        positions = [
            {"ticker": "AAPL", "direction": "LONG", "value": 3000, "sector": "Tech"},
            {"ticker": "MSFT", "direction": "SHORT", "value": 1000, "sector": "Tech"},
        ]
        report = risk_mgr.check_portfolio_risk(positions, 10000)
        assert report.total_long_exposure == 3000
        assert report.total_short_exposure == 1000
        assert report.gross_exposure == 4000
        assert report.net_exposure == 2000

    def test_empty_portfolio(self, risk_mgr):
        """Empty portfolio should produce clean report."""
        report = risk_mgr.check_portfolio_risk([], 10000)
        assert report.gross_exposure == 0
        assert len(report.violations) == 0

"""
Grand Overhaul Test Suite
==========================

Tests for:
1. ContinuousLearner / OnlineLearner updates on trade
2. SignalEngine returns valid scores
3. SharpeOptimizer circuit breaker
4. HealthMonitor detects disconnection
5. Docker Compose YAML validity
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ===========================================================================
# 1. OnlineLearner — self-training ML
# ===========================================================================

class TestOnlineLearner:
    """Test the online learner / continuous self-training ML system."""

    def _make_learner(self, tmp_path):
        from src.ml.online_learner import OnlineLearner, OnlineLearnerConfig

        config = OnlineLearnerConfig(
            retrain_every_n_trades=10,
            sharpe_floor=0.5,
            rolling_sharpe_window=20,
            db_path=str(tmp_path / "test_outcomes.db"),
            models_dir=str(tmp_path / "models"),
            metrics_log=str(tmp_path / "metrics.jsonl"),
            consecutive_loss_limit=3,
            loss_pct_threshold=-0.02,
        )
        return OnlineLearner(config)

    def _make_outcome(self, pnl_pct=0.01, symbol="SPY"):
        from src.ml.online_learner import TradeOutcome

        return TradeOutcome(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            side="long",
            entry_price=100.0,
            exit_price=100.0 * (1 + pnl_pct),
            qty=10,
            pnl=pnl_pct * 1000,
            pnl_pct=pnl_pct,
            signal_confidence=0.75,
            features={"momentum": 0.5, "volatility": 0.02, "rsi": 55.0},
        )

    def test_continuous_learner_updates_on_trade(self, tmp_path):
        """OnlineLearner processes trades and updates weights."""
        learner = self._make_learner(tmp_path)

        # Record a winning trade
        result = learner.record_trade(self._make_outcome(pnl_pct=0.02))
        assert result["trade_count"] == 1
        assert result["circuit_breaker_triggered"] is False
        assert "sharpe" in result

        # Weights should have been updated
        weights = learner.get_weights()
        assert len(weights) > 0

    def test_circuit_breaker_triggers_on_consecutive_losses(self, tmp_path):
        """Circuit breaker opens after N consecutive large losses."""
        learner = self._make_learner(tmp_path)

        # 3 consecutive losses > 2%
        for _ in range(3):
            result = learner.record_trade(self._make_outcome(pnl_pct=-0.03))

        assert result["circuit_breaker_triggered"] is True
        allowed, reason = learner.is_trading_allowed()
        assert allowed is False
        assert "Circuit breaker" in reason

    def test_retraining_triggers_on_schedule(self, tmp_path):
        """Retraining is triggered after every N trades."""
        learner = self._make_learner(tmp_path)

        retrained = False
        for i in range(12):
            result = learner.record_trade(self._make_outcome(pnl_pct=0.005))
            if result["retrain_triggered"]:
                retrained = True

        assert retrained is True

    def test_checkpoint_save_and_load(self, tmp_path):
        """Checkpoints save and restore correctly."""
        learner = self._make_learner(tmp_path)

        # Record some trades
        for _ in range(5):
            learner.record_trade(self._make_outcome(pnl_pct=0.01))

        # Save checkpoint
        path = learner.save_checkpoint(tag="test")
        assert os.path.exists(path)

        # Create new learner and load
        learner2 = self._make_learner(tmp_path)
        assert learner2.load_checkpoint(path) is True
        assert learner2._trade_count == 5

    def test_sqlite_db_stores_outcomes(self, tmp_path):
        """Trade outcomes are persisted in SQLite."""
        learner = self._make_learner(tmp_path)

        for i in range(5):
            learner.record_trade(self._make_outcome(pnl_pct=0.01 * (i + 1)))

        assert learner.db.count() == 5
        recent = learner.db.recent(10)
        assert len(recent) == 5


# ===========================================================================
# 2. SignalEngine — alpha signals
# ===========================================================================

class TestSignalEngine:
    """Test the alpha signal engine returns valid scores."""

    def test_signal_engine_returns_valid_scores(self):
        """All signals return scores in [-1, 1] range."""
        from src.options.signal_engine import SignalEngine

        engine = SignalEngine()
        signal = engine.generate_signal(
            symbol="SPY",
            vix=25.0,
            vix3m=28.0,
            put_call_ratio=1.3,
            iv_history=[0.2 + 0.01 * i for i in range(25)],
        )

        assert -1.0 <= signal.overall_score <= 1.0
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.blocked is False
        assert len(signal.signals) > 0

        # Each sub-signal in valid range
        for s in signal.signals:
            assert -1.0 <= s.score <= 1.0
            assert 0.0 <= s.confidence <= 1.0

    def test_vix_backwardation_bearish(self):
        """VIX backwardation produces bearish signal."""
        from src.options.signal_engine import VIXTermStructureSignal

        sig = VIXTermStructureSignal()
        result = sig.compute(vix=30.0, vix3m=35.0)
        assert result.score < 0  # bearish

    def test_put_call_ratio_contrarian(self):
        """High put/call ratio produces contrarian bullish signal."""
        from src.options.signal_engine import PutCallRatioSignal

        sig = PutCallRatioSignal()
        result = sig.compute(put_call_ratio=1.5)
        assert result.score > 0  # contrarian bullish

    def test_earnings_gate_blocks_trade(self):
        """Earnings gate blocks trades within blackout period."""
        from src.options.signal_engine import EarningsGateSignal

        gate = EarningsGateSignal(blackout_days=2)
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        gate.set_earnings_dates({"AAPL": tomorrow})

        result = gate.check("AAPL")
        assert result.metadata["blocked"] is True
        assert result.confidence == 0.0

    def test_iv_momentum_signal(self):
        """IV momentum produces valid signal."""
        from src.options.signal_engine import IVMomentumSignal

        sig = IVMomentumSignal()
        # Rising IV → last 5 values much higher than rest
        iv_hist = [0.15] * 15 + [0.25] * 5
        result = sig.compute(iv_hist)
        assert -1.0 <= result.score <= 1.0


# ===========================================================================
# 3. SharpeOptimizer — circuit breaker and sizing
# ===========================================================================

class TestSharpeOptimizer:
    """Test Sharpe optimizer circuit breaker and dynamic sizing."""

    def test_sharpe_optimizer_circuit_breaker(self):
        """Drawdown circuit breaker halts trading."""
        from src.optimization.sharpe_optimizer import SharpeOptimizer, SharpeOptimizerConfig

        config = SharpeOptimizerConfig(max_daily_drawdown=0.03)
        opt = SharpeOptimizer(config)

        # Set starting equity
        opt.update_daily_equity(100_000)

        # Before drawdown — trading allowed
        ok, msg = opt.check_drawdown(99_000)
        assert ok is True

        # After drawdown > 3%
        ok, msg = opt.check_drawdown(96_000)
        assert ok is False
        assert "drawdown" in msg.lower()

    def test_dynamic_position_sizing(self):
        """Position sizing adjusts based on Sharpe."""
        from src.optimization.sharpe_optimizer import SharpeOptimizer

        opt = SharpeOptimizer()

        # Mix of wins with a bit of variance to create high positive Sharpe
        np.random.seed(42)
        for r in np.random.normal(0.02, 0.005, 30):
            opt.record_trade(float(r))

        high_sharpe_pct = opt.get_position_size_pct()

        # Losing trades → negative or low Sharpe
        opt2 = SharpeOptimizer()
        for r in np.random.normal(-0.02, 0.005, 30):
            opt2.record_trade(float(r))

        low_sharpe_pct = opt2.get_position_size_pct()

        # Higher Sharpe should produce larger sizes
        assert high_sharpe_pct > low_sharpe_pct

    def test_correlation_check(self):
        """Correlation check rejects highly correlated new positions."""
        from src.optimization.sharpe_optimizer import SharpeOptimizer

        opt = SharpeOptimizer()

        # Create perfectly correlated price data
        prices_a = np.cumsum(np.random.randn(100)) + 200
        prices_b = prices_a * 1.1 + np.random.randn(100) * 0.01  # near-perfect corr
        prices_c = np.cumsum(np.random.randn(100)) + 150  # independent

        data = {"AAPL": prices_a, "AAPL2": prices_b, "GOOG": prices_c}

        # AAPL2 should be rejected (corr > 0.7 with AAPL)
        ok, msg = opt.check_correlation("AAPL2", ["AAPL"], data)
        assert ok is False

        # GOOG should be accepted
        ok, msg = opt.check_correlation("GOOG", ["AAPL"], data)
        assert ok is True

    def test_trade_timing_window(self):
        """Trade timing restricts to preferred windows."""
        from src.optimization.sharpe_optimizer import SharpeOptimizer
        from datetime import time as dt_time

        opt = SharpeOptimizer()

        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo

        # 9:35 AM ET — should be in morning window
        morning = datetime(2026, 2, 23, 9, 35, tzinfo=ZoneInfo("America/New_York"))
        ok, _ = opt.in_trading_window(morning)
        assert ok is True

        # 12:00 PM ET — midday, should NOT be in window
        midday = datetime(2026, 2, 23, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        ok, _ = opt.in_trading_window(midday)
        assert ok is False

        # 15:30 ET — afternoon window
        afternoon = datetime(2026, 2, 23, 15, 30, tzinfo=ZoneInfo("America/New_York"))
        ok, _ = opt.in_trading_window(afternoon)
        assert ok is True


# ===========================================================================
# 4. HealthMonitor — disconnection detection
# ===========================================================================

class TestHealthMonitor:
    """Test health monitor detects disconnection."""

    def test_health_monitor_detects_disconnection(self):
        """Health monitor returns unhealthy when IBKR unreachable."""
        from src.monitoring.health_monitor import HealthMonitor, HealthMonitorConfig

        config = HealthMonitorConfig(
            ibkr_host="127.0.0.1",
            ibkr_port=59999,  # unlikely to be open
            discord_webhook="",  # no alerts in tests
        )
        monitor = HealthMonitor(config)

        # Run checks manually (no background thread)
        results = monitor.run_checks()

        # Find ibkr_connection check
        ibkr_check = next((r for r in results if r.check_name == "ibkr_connection"), None)
        assert ibkr_check is not None
        assert ibkr_check.healthy is False

    def test_health_monitor_status_format(self):
        """get_status() returns correct format."""
        from src.monitoring.health_monitor import HealthMonitor, HealthMonitorConfig

        config = HealthMonitorConfig(
            ibkr_host="127.0.0.1",
            ibkr_port=59999,
            discord_webhook="",
        )
        monitor = HealthMonitor(config)
        status = monitor.get_status()

        assert "checks" in status
        assert "all_healthy" in status
        assert isinstance(status["checks"], list)

    def test_health_monitor_pnl_check(self):
        """PnL check detects excessive daily loss."""
        from src.monitoring.health_monitor import HealthMonitor, HealthMonitorConfig

        config = HealthMonitorConfig(
            ibkr_host="127.0.0.1",
            ibkr_port=59999,
            max_daily_loss_pct=0.05,
            discord_webhook="",
        )
        monitor = HealthMonitor(config)
        monitor.set_daily_pnl_fn(lambda: -0.06)  # -6% loss

        results = monitor.run_checks()
        pnl_check = next((r for r in results if r.check_name == "daily_pnl"), None)
        assert pnl_check is not None
        assert pnl_check.healthy is False


# ===========================================================================
# 5. Docker Compose YAML validity
# ===========================================================================

class TestDockerCompose:
    """Validate Docker Compose file structure."""

    def test_docker_compose_valid_yaml(self):
        """docker-compose.ibkr.yml is valid YAML with required services."""
        import yaml

        compose_path = Path(__file__).resolve().parent.parent / "deploy" / "docker-compose.ibkr.yml"
        assert compose_path.exists(), f"Missing: {compose_path}"

        with open(compose_path) as f:
            data = yaml.safe_load(f)

        # Must have services
        assert "services" in data
        services = data["services"]

        # Required services
        assert "ib-gateway" in services
        assert "trading-bot" in services
        assert "prometheus" in services

        # ib-gateway must have health check
        gw = services["ib-gateway"]
        assert "healthcheck" in gw
        assert gw["restart"] == "always"

        # trading-bot must depend on ib-gateway
        bot = services["trading-bot"]
        assert "ib-gateway" in bot.get("depends_on", {})

        # Network must exist
        assert "networks" in data
        assert "ibkr-net" in data["networks"]

    def test_dockerfile_exists_and_has_entrypoint(self):
        """Dockerfile exists and has proper ENTRYPOINT."""
        dockerfile = Path(__file__).resolve().parent.parent / "Dockerfile"
        assert dockerfile.exists()

        content = dockerfile.read_text()
        assert "python:3.11-slim" in content
        assert "PYTHONPATH" in content
        assert "ENTRYPOINT" in content or "CMD" in content


# ===========================================================================
# 6. Integration — imports work correctly
# ===========================================================================

class TestImports:
    """Verify all new modules import cleanly."""

    def test_import_ibkr_client(self):
        from src.brokers.ibkr_client import IBKRBrokerClient
        assert IBKRBrokerClient is not None

    def test_import_online_learner(self):
        from src.ml.online_learner import OnlineLearner, TradeOutcome
        assert OnlineLearner is not None
        assert TradeOutcome is not None

    def test_import_signal_engine(self):
        from src.options.signal_engine import SignalEngine, CompositeSignal
        assert SignalEngine is not None

    def test_import_sharpe_optimizer(self):
        from src.optimization.sharpe_optimizer import SharpeOptimizer
        assert SharpeOptimizer is not None

    def test_import_health_monitor(self):
        from src.monitoring.health_monitor import HealthMonitor
        assert HealthMonitor is not None


# ===========================================================================
# 7. Phase A — New Strategy Classes
# ===========================================================================

class TestGammaScalpingStrategy:
    """Tests for GammaScalpingStrategy in signal_generator.py."""

    def _make_strategy(self):
        from src.options.signal_generator import GammaScalpingStrategy
        return GammaScalpingStrategy()

    def test_instantiation(self):
        strat = self._make_strategy()
        assert strat is not None
        assert hasattr(strat, "generate_signals")

    @pytest.mark.asyncio
    async def test_no_signal_on_empty_symbols(self):
        strat = self._make_strategy()
        sigs = await strat.generate_signals([])
        assert sigs == []

    @pytest.mark.asyncio
    async def test_returns_list(self):
        strat = self._make_strategy()
        sigs = await strat.generate_signals(["AAPL"])
        assert isinstance(sigs, list)


class TestVolatilityArbitrageStrategy:
    """Tests for VolatilityArbitrageStrategy."""

    def _make_strategy(self):
        from src.options.signal_generator import VolatilityArbitrageStrategy
        return VolatilityArbitrageStrategy()

    def test_instantiation(self):
        strat = self._make_strategy()
        assert hasattr(strat, "generate_signals")

    @pytest.mark.asyncio
    async def test_returns_list_without_data(self):
        strat = self._make_strategy()
        sigs = await strat.generate_signals([])
        assert isinstance(sigs, list)


class TestSkewTradeStrategy:
    """Tests for SkewTradeStrategy."""

    def _make_strategy(self):
        from src.options.signal_generator import SkewTradeStrategy
        return SkewTradeStrategy()

    def test_instantiation(self):
        strat = self._make_strategy()
        assert hasattr(strat, "generate_signals")

    @pytest.mark.asyncio
    async def test_returns_list_without_data(self):
        strat = self._make_strategy()
        sigs = await strat.generate_signals([])
        assert isinstance(sigs, list)


# ===========================================================================
# 8. Phase A — Exit Methods
# ===========================================================================

class TestNewExitMethods:
    """Tests for profit_target_exit, time_stop_exit, delta_breach_exit."""

    def _make_manager(self):
        from src.options.exit_manager import ExitManager
        return ExitManager(trading_client=None, data_client=None)

    def _make_pos(self, **overrides):
        """Build a mock TrackedPosition with sensible defaults."""
        from datetime import date as dt_date
        pos = MagicMock()
        pos.position_id = "test-001"
        pos.underlying = "SPY"
        pos.is_closed = False
        pos.max_profit = 500.0
        pos.current_pnl = overrides.get("current_pnl", 100.0)
        pos.current_pnl_pct = overrides.get("current_pnl_pct", 0.20)
        pos.peak_pnl = 150.0
        pos.peak_pnl_pct = 0.30
        pos.legs = []
        pos.position_type = MagicMock()
        pos.strategy = "iron_condor"
        pos.entry_time = datetime(2025, 1, 1, 10, 0)
        pos.expiration = overrides.get(
            "expiration", dt_date.today() + timedelta(days=20)
        )
        pos.dte = (pos.expiration - dt_date.today()).days
        total_days = (pos.expiration - pos.entry_time.date()).days
        elapsed = (dt_date.today() - pos.entry_time.date()).days
        pos.time_elapsed_pct = min(1.0, elapsed / total_days) if total_days > 0 else 1.0
        for k, v in overrides.items():
            setattr(pos, k, v)
        return pos

    def test_profit_target_exit_under_threshold(self):
        mgr = self._make_manager()
        pos = self._make_pos(current_pnl=100.0, max_profit=500.0)
        result = mgr.profit_target_exit(pos)
        assert result is None  # 20% < 50%

    def test_profit_target_exit_above_threshold(self):
        mgr = self._make_manager()
        pos = self._make_pos(current_pnl=300.0, max_profit=500.0)
        result = mgr.profit_target_exit(pos)
        assert result is not None  # 60% >= 50%
        assert result.reason.name == "PROFIT_TARGET_50PCT"

    def test_time_stop_exit_short_dte_ignored(self):
        """Entries < 21 DTE should NOT trigger time stop."""
        from datetime import date as dt_date
        mgr = self._make_manager()
        pos = self._make_pos(
            entry_time=datetime.now() - timedelta(days=5),
            expiration=dt_date.today() + timedelta(days=10),
        )
        # Re-compute entry_to_expiry to be < 21
        result = mgr.time_stop_exit(pos)
        assert result is None

    def test_delta_breach_exit_triggers(self):
        mgr = self._make_manager()
        pos = self._make_pos()
        result = mgr.delta_breach_exit(pos, position_delta=0.35)
        assert result is not None
        assert result.reason.name == "DELTA_BREACH"

    def test_delta_breach_exit_within_limit(self):
        mgr = self._make_manager()
        pos = self._make_pos()
        result = mgr.delta_breach_exit(pos, position_delta=0.15)
        assert result is None


# ===========================================================================
# 9. Phase A — Position Sizer Helpers
# ===========================================================================

class TestPositionSizerHelpers:
    """Tests for portfolio_heat_check and correlation_adjustment."""

    def test_portfolio_heat_check_normal(self):
        from src.options.position_sizer import portfolio_heat_check
        assert portfolio_heat_check(100, 200) is True

    def test_portfolio_heat_check_hot(self):
        from src.options.position_sizer import portfolio_heat_check
        assert portfolio_heat_check(500, 200) is False

    def test_portfolio_heat_check_zero_avg(self):
        from src.options.position_sizer import portfolio_heat_check
        # avg_daily_vega=0 cannot evaluate → allow (True)
        assert portfolio_heat_check(500, 0) is True

    def test_correlation_adjustment_low(self):
        from src.options.position_sizer import correlation_adjustment
        result = correlation_adjustment(10, 0.3)
        assert result == 10

    def test_correlation_adjustment_high(self):
        from src.options.position_sizer import correlation_adjustment
        result = correlation_adjustment(10, 0.9)
        assert result == 7

    def test_correlation_adjustment_edge(self):
        from src.options.position_sizer import correlation_adjustment
        result = correlation_adjustment(10, 0.7)
        assert result == 10


# ===========================================================================
# 10. Phase B — Feature Engineering
# ===========================================================================

class TestFeatureEngineering:
    """Tests for build_features()."""

    def test_build_features_basic(self):
        from src.ml.feature_engineering import build_features
        sig = {"iv_rank": 50, "dte": 30, "delta": -0.20}
        mkt = {"vix_level": 18.0}
        result = build_features(sig, mkt)
        assert isinstance(result, np.ndarray)
        assert result.shape == (7,)

    def test_build_features_all_zeros(self):
        from src.ml.feature_engineering import build_features
        result = build_features({}, {})
        assert isinstance(result, np.ndarray)
        assert result.shape == (7,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_build_features_extreme_values(self):
        from src.ml.feature_engineering import build_features
        sig = {"iv_rank": 200, "dte": -5, "delta": 5.0}
        mkt = {"vix_level": 100}
        result = build_features(sig, mkt)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


# ===========================================================================
# 11. Phase B — Online Learner SGD
# ===========================================================================

class TestOnlineLearnerSGD:
    """Tests for SGD additions in OnlineLearner."""

    def _make_learner(self):
        from src.ml.online_learner import OnlineLearner, OnlineLearnerConfig
        cfg = OnlineLearnerConfig()
        return OnlineLearner(cfg)

    def test_has_sgd_methods(self):
        learner = self._make_learner()
        assert hasattr(learner, "get_signal_confidence")
        assert hasattr(learner, "retrain_on_close")
        assert hasattr(learner, "_init_sgd_model")

    def test_get_signal_confidence_returns_float(self):
        learner = self._make_learner()
        features = np.array([0.5, 0.3, 0.4, 0.2, 0.6, 0.5, 0.3])
        conf = learner.get_signal_confidence(features)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_retrain_on_close_accepts_trade_dict(self):
        learner = self._make_learner()
        trade = {
            "iv_rank": 60, "vix_level": 20, "dte": 25,
            "delta": -0.15, "rv_iv_ratio": 0.8, "hour": 10,
            "weekday": 2, "pnl": 150.0,
        }
        learner.retrain_on_close(trade)


# ===========================================================================
# 12. Phase C — Circuit Breaker attrs exist
# ===========================================================================

class TestCircuitBreakerAttrs:
    """Tests for consecutive loss pause and daily PnL halt attrs."""

    def test_autonomous_engine_has_consecutive_losses(self):
        from src.options.autonomous_engine import AutonomousTradingEngine
        import inspect
        source = inspect.getsource(AutonomousTradingEngine.__init__)
        assert "consecutive_losses" in source
        assert "_loss_pause_until" in source

    def test_ml_retrain_method_exists(self):
        from src.options.autonomous_engine import AutonomousTradingEngine
        assert hasattr(AutonomousTradingEngine, "_retrain_ml_on_exit")


# ===========================================================================
# 13. Phase C — Health Monitor Extensions
# ===========================================================================

class TestHealthMonitorExtensions:
    """Tests for memory and disk checks."""

    def test_health_monitor_has_memory_check(self):
        from src.monitoring.health_monitor import HealthMonitor
        assert hasattr(HealthMonitor, "_check_memory_usage")

    def test_health_monitor_has_disk_check(self):
        from src.monitoring.health_monitor import HealthMonitor
        assert hasattr(HealthMonitor, "_check_disk_usage")

    def test_memory_check_runs(self):
        """Memory check should execute without error."""
        from src.monitoring.health_monitor import HealthMonitor, HealthMonitorConfig
        cfg = HealthMonitorConfig()
        hm = HealthMonitor(config=cfg)
        hm._check_memory_usage()

    def test_disk_check_runs(self):
        from src.monitoring.health_monitor import HealthMonitor, HealthMonitorConfig
        cfg = HealthMonitorConfig()
        hm = HealthMonitor(config=cfg)
        hm._check_disk_usage()


# ===========================================================================
# 14. Phase D — Vol Surface Engineering
# ===========================================================================

class TestVolSurfaceFit:
    """Tests for IVDataManager.vol_surface_fit()."""

    def _make_manager(self):
        from src.options.iv_data_manager import IVDataManager
        return IVDataManager(data_dir="/tmp/test_iv_cache")

    def _make_chain(self, n=20, underlying=500.0):
        chain = []
        for i in range(n):
            strike = underlying - 50 + (i * 5)
            iv = 0.20 + 0.0001 * (strike - underlying) ** 2
            c = MagicMock()
            c.strike = strike
            c.implied_volatility = iv
            c.underlying_price = underlying
            c.delta = -0.5 + i * 0.05
            c.right = "C" if i % 2 == 0 else "P"
            c.option_type = c.right
            c.symbol = "SPY"
            chain.append(c)
        return chain

    def test_empty_chain_returns_empty(self):
        mgr = self._make_manager()
        assert mgr.vol_surface_fit([]) == {}

    def test_svi_fit_basic(self):
        mgr = self._make_manager()
        chain = self._make_chain()
        result = mgr.vol_surface_fit(chain)
        assert "a" in result
        assert "b" in result
        assert "rho" in result
        assert "rmse" in result

    def test_svi_fit_rmse_reasonable(self):
        mgr = self._make_manager()
        chain = self._make_chain()
        result = mgr.vol_surface_fit(chain)
        if result:
            assert result["rmse"] < 1.0


class TestTermStructureSignal:
    """Tests for term_structure_signal."""

    def _make_manager(self):
        from src.options.iv_data_manager import IVDataManager
        return IVDataManager(data_dir="/tmp/test_iv_cache")

    @patch("src.options.iv_data_manager.yf")
    def test_contango_signal(self, mock_yf):
        import pandas as pd
        mock_yf.download.side_effect = [
            pd.DataFrame({"Close": [18.0]}),
            pd.DataFrame({"Close": [22.0]}),
        ]
        mgr = self._make_manager()
        result = mgr.term_structure_signal()
        assert result.get("signal") == "CONTANGO"
        assert result.get("ratio", 999) < 1.0

    @patch("src.options.iv_data_manager.yf")
    def test_backwardation_signal(self, mock_yf):
        import pandas as pd
        mock_yf.download.side_effect = [
            pd.DataFrame({"Close": [32.0]}),
            pd.DataFrame({"Close": [25.0]}),
        ]
        mgr = self._make_manager()
        result = mgr.term_structure_signal()
        assert result.get("signal") == "BACKWARDATION"
        assert result.get("ratio", 0) > 1.0


class TestSkewSignal:
    """Tests for skew_signal."""

    def _make_manager(self):
        from src.options.iv_data_manager import IVDataManager
        return IVDataManager(data_dir="/tmp/test_iv_cache")

    def _make_chain_with_delta(self):
        chain = []
        for delta, right, iv in [
            (-0.25, "P", 0.28),
            (-0.50, "P", 0.22),
            (0.25, "C", 0.18),
            (0.50, "C", 0.20),
        ]:
            c = MagicMock()
            c.delta = delta
            c.right = right
            c.option_type = right
            c.implied_volatility = iv
            c.symbol = "SPY"
            c.strike = 500
            chain.append(c)
        return chain

    def test_empty_chain(self):
        mgr = self._make_manager()
        assert mgr.skew_signal([]) == {}

    def test_skew_basic(self):
        mgr = self._make_manager()
        chain = self._make_chain_with_delta()
        result = mgr.skew_signal(chain)
        assert "put_iv" in result
        assert "call_iv" in result
        assert "skew" in result
        assert result["skew"] == pytest.approx(0.10, abs=0.01)

    def test_skew_zscore_is_float(self):
        mgr = self._make_manager()
        chain = self._make_chain_with_delta()
        result = mgr.skew_signal(chain)
        assert isinstance(result["zscore"], float)

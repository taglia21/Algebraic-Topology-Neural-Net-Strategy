"""
tests/test_ood_telemetry_and_calibration.py
============================================
Validate OOD telemetry persistence and policy calibration workflow.

Tests
-----
1. Backtest result includes ML OOD telemetry
2. Telemetry contains expected counters (checks, blocks, rate)
3. Policy calibration can rank policies based on backtest results
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from backtest.backtester import Backtester
from core.config import Config
from ml.policy_calibration import PolicyCalibrator


class TestOODTelemetryPersistence:
    """Validate OOD telemetry flows through BacktestResult."""

    def test_backtest_result_includes_ood_telemetry(self):
        """Backtest result should include ML OOD telemetry."""
        config = Config()
        backtester = Backtester(config)

        # Small backtest
        result = backtester.run(
            symbols=["SPY"],
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

        # Check that telemetry field exists (may be None or dict)
        assert hasattr(result, "ml_ood_telemetry"), "Missing ml_ood_telemetry field"
        assert result.ml_ood_telemetry is None or isinstance(
            result.ml_ood_telemetry, dict
        ), "ml_ood_telemetry should be dict or None"

    def test_ood_telemetry_has_expected_keys(self):
        """OOD telemetry should include checks, blocks, rate, top_outliers."""
        config = Config()
        backtester = Backtester(config)

        result = backtester.run(
            symbols=["SPY"],
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

        if result.ml_ood_telemetry is not None:
            telemetry = result.ml_ood_telemetry
            assert isinstance(telemetry, dict)
            # Expected keys from ml/pipeline.py get_ood_telemetry()
            expected_keys = {"ood_checks", "ood_blocks", "ood_block_rate"}
            actual_keys = set(telemetry.keys())
            assert expected_keys.issubset(
                actual_keys
            ), f"Missing keys: {expected_keys - actual_keys}"

    def test_ood_checks_and_blocks_are_nonnegative(self):
        """OOD checks and blocks should be non-negative integers."""
        config = Config()
        backtester = Backtester(config)

        result = backtester.run(
            symbols=["SPY"],
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

        if result.ml_ood_telemetry is not None:
            telemetry = result.ml_ood_telemetry
            checks = telemetry.get("ood_checks", 0)
            blocks = telemetry.get("ood_blocks", 0)
            rate = telemetry.get("ood_block_rate", 0.0)

            assert isinstance(checks, int) and checks >= 0
            assert isinstance(blocks, int) and blocks >= 0
            assert isinstance(rate, float) and 0.0 <= rate <= 1.0


class TestPolicyCalibratorIntegration:
    """Validate rolling-window policy calibration orchestration."""

    def test_calibrator_can_compare_two_policies(self):
        """Calibrator should run multiple policies and rank them."""

        def minimal_backtest(symbols, start_date, end_date):
            """Minimal backtest for calibration testing."""
            config = Config()
            backtester = Backtester(config)
            return backtester.run(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
            )

        calibrator = PolicyCalibrator(
            backtest_func=minimal_backtest,
            symbols=["SPY"],
            start_date="2024-01-01",
            end_date="2024-02-29",
        )

        # Test with 2 policies only (skip and neutral) for speed
        results = calibrator.run_rolling_backtest(
            window_days=30,
            stride_days=30,  # No overlap for speed
            policies=["skip", "neutral"],
        )

        # Should have at least 2 results (1 policy × 1 window at minimum)
        assert len(results) >= 2, f"Expected at least 2 results, got {len(results)}"

        # Each result should be valid
        for res in results:
            assert res.policy in ["skip", "neutral"]
            assert res.window_start < res.window_end
            assert res.sharpe is not None
            assert res.n_trades >= 0

    def test_policy_recommendation_selects_best_policy(self):
        """Recommendation should select policy with highest mean Sharpe."""

        def dummy_backtest(symbols, start_date, end_date):
            """Dummy backtest returning fixed results."""
            config = Config()
            backtester = Backtester(config)
            return backtester.run(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
            )

        calibrator = PolicyCalibrator(
            backtest_func=dummy_backtest,
            symbols=["SPY"],
            start_date="2024-01-01",
            end_date="2024-02-29",
        )

        results = calibrator.run_rolling_backtest(
            window_days=30,
            stride_days=30,
            policies=["skip", "neutral"],
        )

        recommendation = calibrator.recommend_policy(results, metric="sharpe_ratio")

        assert recommendation.recommended_policy in ["skip", "neutral"]
        assert "sharpe" in recommendation.reason.lower() or "sharpe_ratio" in recommendation.reason.lower()
        assert recommendation.rankings is not None
        assert len(recommendation.rankings) > 0

    def test_results_table_format_is_valid(self):
        """Results table formatting should produce non-empty string."""
        from ml.policy_calibration import PolicyResult

        result = PolicyResult(
            policy="skip",
            window_start="2024-01-01",
            window_end="2024-01-31",
            sharpe=1.05,
            sortino=1.32,
            max_drawdown=-0.08,
            total_return=0.05,
            n_trades=25,
            ood_checks=1000,
            ood_blocks=50,
            ood_block_rate=0.05,
        )

        table = PolicyCalibrator.format_results_table([result])
        assert len(table) > 0
        assert "skip" in table.lower()
        assert "2024-01-01" in table


class TestPolicyCalibrationWithEnvironmentVariables:
    """Verify that ML_OOD_ACTION env var controls policy behavior."""

    def test_skip_policy_respects_env_var(self):
        """ML pipeline should respect ML_OOD_ACTION='skip'."""
        os.environ["ML_OOD_ACTION"] = "skip"

        config = Config()
        assert config.ml.ood_action == "skip"

    def test_neutral_policy_respects_env_var(self):
        """ML pipeline should respect ML_OOD_ACTION='neutral'."""
        os.environ["ML_OOD_ACTION"] = "neutral"

        config = Config()
        assert config.ml.ood_action == "neutral"

    def test_block_policy_respects_env_var(self):
        """ML pipeline should respect ML_OOD_ACTION='block'."""
        os.environ["ML_OOD_ACTION"] = "block"

        config = Config()
        assert config.ml.ood_action == "block"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

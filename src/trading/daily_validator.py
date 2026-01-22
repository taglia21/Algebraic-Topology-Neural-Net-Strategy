"""
V2.1 Daily Validation System
============================

Automated daily validation comparing paper vs expected results:
- Performance deviation detection (> 2 sigma)
- Anomaly flagging
- Risk metric validation
- Component health checks

Usage:
    validator = DailyValidator()
    report = validator.run_validation(daily_trades, expected_sharpe=1.35)
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).parent.parent.parent / "logs"


@dataclass
class ValidationResult:
    """Single validation check result."""
    check_name: str
    passed: bool
    expected: Any
    actual: Any
    deviation_sigma: float = 0.0
    message: str = ""


@dataclass
class DailyValidationReport:
    """Complete daily validation report."""
    date: str
    timestamp: str
    overall_status: str  # "pass", "warning", "fail"
    checks_passed: int
    checks_total: int
    has_anomalies: bool
    anomalies: List[Dict]
    validation_results: List[Dict]
    recommendations: List[str]
    metrics_summary: Dict[str, Any]


class DailyValidator:
    """
    Daily validation system for production trading.
    
    Compares actual performance to expected baselines and flags anomalies.
    """
    
    def __init__(self, 
                 expected_sharpe: float = 1.35,
                 expected_win_rate: float = 0.52,
                 expected_max_dd: float = 0.05,
                 sigma_threshold: float = 2.0):
        """
        Initialize validator with expected baselines.
        
        Args:
            expected_sharpe: Expected Sharpe ratio (V1.3 baseline: 1.35)
            expected_win_rate: Expected win rate
            expected_max_dd: Expected max drawdown
            sigma_threshold: Sigma threshold for anomaly detection
        """
        self.expected_sharpe = expected_sharpe
        self.expected_win_rate = expected_win_rate
        self.expected_max_dd = expected_max_dd
        self.sigma_threshold = sigma_threshold
        
        # Historical data for baseline computation
        self.historical_returns: List[float] = []
        self.historical_sharpes: List[float] = []
        self.historical_trades: List[Dict] = []
        
        # Load historical baseline if available
        self._load_historical_baseline()
        
    def _load_historical_baseline(self):
        """Load historical baseline data."""
        baseline_file = LOG_DIR / "validation_baseline.json"
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    data = json.load(f)
                    self.historical_returns = data.get("returns", [])
                    self.historical_sharpes = data.get("sharpes", [])
                    logger.info(f"Loaded baseline: {len(self.historical_returns)} returns")
            except Exception as e:
                logger.warning(f"Failed to load baseline: {e}")
                
    def _save_historical_baseline(self):
        """Save updated historical baseline."""
        LOG_DIR.mkdir(exist_ok=True)
        baseline_file = LOG_DIR / "validation_baseline.json"
        try:
            with open(baseline_file, 'w') as f:
                json.dump({
                    "returns": self.historical_returns[-252:],  # Keep 1 year
                    "sharpes": self.historical_sharpes[-90:],  # Keep 90 days
                    "updated": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save baseline: {e}")
            
    def run_validation(self,
                       daily_trades: List[Dict],
                       daily_return: Optional[float] = None,
                       current_dd: Optional[float] = None,
                       current_sharpe: Optional[float] = None,
                       expected_sharpe: Optional[float] = None,
                       sigma_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Run complete daily validation.
        
        Args:
            daily_trades: List of trade records from today
            daily_return: Today's portfolio return
            current_dd: Current drawdown
            current_sharpe: Current rolling Sharpe
            expected_sharpe: Override expected Sharpe
            sigma_threshold: Override sigma threshold
            
        Returns:
            Validation report dict
        """
        expected_sharpe = expected_sharpe or self.expected_sharpe
        sigma_threshold = sigma_threshold or self.sigma_threshold
        
        results: List[ValidationResult] = []
        anomalies: List[Dict] = []
        recommendations: List[str] = []
        
        # 1. Trade count validation
        trade_result = self._validate_trade_count(daily_trades)
        results.append(trade_result)
        if not trade_result.passed:
            anomalies.append({
                "type": "trade_count",
                "message": trade_result.message,
                "deviation": trade_result.deviation_sigma,
            })
            
        # 2. Win rate validation
        if daily_trades:
            win_result = self._validate_win_rate(daily_trades)
            results.append(win_result)
            if not win_result.passed:
                anomalies.append({
                    "type": "win_rate",
                    "message": win_result.message,
                    "deviation": win_result.deviation_sigma,
                })
                
        # 3. Return validation
        if daily_return is not None:
            return_result = self._validate_return(daily_return)
            results.append(return_result)
            if not return_result.passed:
                anomalies.append({
                    "type": "return",
                    "message": return_result.message,
                    "deviation": return_result.deviation_sigma,
                })
                
        # 4. Drawdown validation
        if current_dd is not None:
            dd_result = self._validate_drawdown(current_dd)
            results.append(dd_result)
            if not dd_result.passed:
                anomalies.append({
                    "type": "drawdown",
                    "message": dd_result.message,
                    "deviation": dd_result.deviation_sigma,
                })
                
        # 5. Sharpe validation
        if current_sharpe is not None:
            sharpe_result = self._validate_sharpe(current_sharpe, expected_sharpe)
            results.append(sharpe_result)
            if not sharpe_result.passed:
                anomalies.append({
                    "type": "sharpe",
                    "message": sharpe_result.message,
                    "deviation": sharpe_result.deviation_sigma,
                })
                
        # 6. Execution quality validation
        exec_result = self._validate_execution_quality(daily_trades)
        results.append(exec_result)
        if not exec_result.passed:
            anomalies.append({
                "type": "execution",
                "message": exec_result.message,
                "deviation": exec_result.deviation_sigma,
            })
            
        # 7. Position concentration validation
        if daily_trades:
            conc_result = self._validate_position_concentration(daily_trades)
            results.append(conc_result)
            if not conc_result.passed:
                anomalies.append({
                    "type": "concentration",
                    "message": conc_result.message,
                    "deviation": conc_result.deviation_sigma,
                })
                
        # Generate recommendations
        recommendations = self._generate_recommendations(anomalies, results)
        
        # Determine overall status
        checks_passed = sum(1 for r in results if r.passed)
        checks_total = len(results)
        
        if len(anomalies) == 0:
            overall_status = "pass"
        elif any(a["deviation"] > 3.0 for a in anomalies):
            overall_status = "fail"
        else:
            overall_status = "warning"
            
        # Update historical baseline
        if daily_return is not None:
            self.historical_returns.append(daily_return)
        if current_sharpe is not None:
            self.historical_sharpes.append(current_sharpe)
        self._save_historical_baseline()
        
        # Build report
        report = DailyValidationReport(
            date=datetime.now().strftime("%Y-%m-%d"),
            timestamp=datetime.now().isoformat(),
            overall_status=overall_status,
            checks_passed=checks_passed,
            checks_total=checks_total,
            has_anomalies=len(anomalies) > 0,
            anomalies=anomalies,
            validation_results=[asdict(r) for r in results],
            recommendations=recommendations,
            metrics_summary={
                "trades_today": len(daily_trades),
                "daily_return": daily_return,
                "current_dd": current_dd,
                "current_sharpe": current_sharpe,
            }
        )
        
        # Log report
        self._log_report(report)
        
        return asdict(report)
        
    def _validate_trade_count(self, trades: List[Dict]) -> ValidationResult:
        """Validate trade count is within expected range."""
        trade_count = len(trades)
        
        # Expected range: 5-100 trades per day for active strategy
        expected_min = 5
        expected_max = 100
        
        if expected_min <= trade_count <= expected_max:
            return ValidationResult(
                check_name="trade_count",
                passed=True,
                expected=f"{expected_min}-{expected_max}",
                actual=trade_count,
                message="Trade count within expected range"
            )
        else:
            deviation = 0
            if trade_count < expected_min:
                deviation = (expected_min - trade_count) / (expected_min * 0.5)
            else:
                deviation = (trade_count - expected_max) / (expected_max * 0.5)
                
            return ValidationResult(
                check_name="trade_count",
                passed=False,
                expected=f"{expected_min}-{expected_max}",
                actual=trade_count,
                deviation_sigma=deviation,
                message=f"Trade count {trade_count} outside expected range"
            )
            
    def _validate_win_rate(self, trades: List[Dict]) -> ValidationResult:
        """Validate win rate is within expected range."""
        if not trades:
            return ValidationResult(
                check_name="win_rate",
                passed=True,
                expected=self.expected_win_rate,
                actual=None,
                message="No trades to validate"
            )
            
        # Calculate win rate from trade PnL
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        win_rate = wins / len(trades)
        
        # Expected: 52% +/- 10%
        deviation = abs(win_rate - self.expected_win_rate) / 0.10
        
        if deviation < self.sigma_threshold:
            return ValidationResult(
                check_name="win_rate",
                passed=True,
                expected=self.expected_win_rate,
                actual=win_rate,
                deviation_sigma=deviation,
                message=f"Win rate {win_rate:.1%} within expected range"
            )
        else:
            return ValidationResult(
                check_name="win_rate",
                passed=False,
                expected=self.expected_win_rate,
                actual=win_rate,
                deviation_sigma=deviation,
                message=f"Win rate {win_rate:.1%} deviates {deviation:.1f} sigma from expected"
            )
            
    def _validate_return(self, daily_return: float) -> ValidationResult:
        """Validate daily return is within expected range."""
        # Calculate expected daily return stats from history
        if len(self.historical_returns) >= 20:
            hist_mean = np.mean(self.historical_returns[-60:])
            hist_std = np.std(self.historical_returns[-60:]) + 1e-8
            deviation = abs(daily_return - hist_mean) / hist_std
        else:
            # Use default: expect 0.05% daily with 1% std
            hist_mean = 0.0005
            hist_std = 0.01
            deviation = abs(daily_return - hist_mean) / hist_std
            
        if deviation < self.sigma_threshold:
            return ValidationResult(
                check_name="daily_return",
                passed=True,
                expected=f"{hist_mean:.2%} +/- {hist_std:.2%}",
                actual=daily_return,
                deviation_sigma=deviation,
                message=f"Daily return {daily_return:.2%} within expected range"
            )
        else:
            return ValidationResult(
                check_name="daily_return",
                passed=False,
                expected=f"{hist_mean:.2%} +/- {hist_std:.2%}",
                actual=daily_return,
                deviation_sigma=deviation,
                message=f"Daily return {daily_return:.2%} deviates {deviation:.1f} sigma"
            )
            
    def _validate_drawdown(self, current_dd: float) -> ValidationResult:
        """Validate drawdown is within limits."""
        # Warning at 3%, fail at 5%
        if current_dd < 0.03:
            return ValidationResult(
                check_name="drawdown",
                passed=True,
                expected=f"< {self.expected_max_dd:.0%}",
                actual=current_dd,
                message=f"Drawdown {current_dd:.2%} within limits"
            )
        elif current_dd < self.expected_max_dd:
            return ValidationResult(
                check_name="drawdown",
                passed=True,
                expected=f"< {self.expected_max_dd:.0%}",
                actual=current_dd,
                deviation_sigma=1.5,
                message=f"Drawdown {current_dd:.2%} elevated but within limits"
            )
        else:
            deviation = (current_dd - self.expected_max_dd) / 0.02
            return ValidationResult(
                check_name="drawdown",
                passed=False,
                expected=f"< {self.expected_max_dd:.0%}",
                actual=current_dd,
                deviation_sigma=deviation,
                message=f"Drawdown {current_dd:.2%} exceeds limit"
            )
            
    def _validate_sharpe(self, current_sharpe: float, expected_sharpe: float) -> ValidationResult:
        """Validate Sharpe ratio is within expected range."""
        # Calculate deviation using historical variance
        if len(self.historical_sharpes) >= 10:
            hist_std = np.std(self.historical_sharpes[-30:]) + 0.1
        else:
            hist_std = 0.3  # Default Sharpe std
            
        deviation = (expected_sharpe - current_sharpe) / hist_std
        
        if deviation < self.sigma_threshold:
            return ValidationResult(
                check_name="sharpe_ratio",
                passed=True,
                expected=expected_sharpe,
                actual=current_sharpe,
                deviation_sigma=deviation,
                message=f"Sharpe {current_sharpe:.2f} within expected range"
            )
        else:
            return ValidationResult(
                check_name="sharpe_ratio",
                passed=False,
                expected=expected_sharpe,
                actual=current_sharpe,
                deviation_sigma=deviation,
                message=f"Sharpe {current_sharpe:.2f} below expected by {deviation:.1f} sigma"
            )
            
    def _validate_execution_quality(self, trades: List[Dict]) -> ValidationResult:
        """Validate execution quality (slippage, fill rates)."""
        if not trades:
            return ValidationResult(
                check_name="execution_quality",
                passed=True,
                expected="< 10bp slippage",
                actual="N/A",
                message="No trades to validate"
            )
            
        # Calculate average slippage
        slippages = [t.get("slippage_bp", 0) for t in trades if "slippage_bp" in t]
        if slippages:
            avg_slippage = np.mean(slippages)
            
            if avg_slippage < 10:
                return ValidationResult(
                    check_name="execution_quality",
                    passed=True,
                    expected="< 10bp slippage",
                    actual=f"{avg_slippage:.1f}bp",
                    message=f"Execution quality good: {avg_slippage:.1f}bp avg slippage"
                )
            else:
                deviation = (avg_slippage - 10) / 5
                return ValidationResult(
                    check_name="execution_quality",
                    passed=False,
                    expected="< 10bp slippage",
                    actual=f"{avg_slippage:.1f}bp",
                    deviation_sigma=deviation,
                    message=f"High slippage: {avg_slippage:.1f}bp average"
                )
        else:
            return ValidationResult(
                check_name="execution_quality",
                passed=True,
                expected="< 10bp slippage",
                actual="N/A",
                message="No slippage data available"
            )
            
    def _validate_position_concentration(self, trades: List[Dict]) -> ValidationResult:
        """Validate position concentration is within limits."""
        # Check for any single position > 5% of traded value
        total_value = sum(abs(t.get("value", 0)) for t in trades)
        if total_value == 0:
            return ValidationResult(
                check_name="concentration",
                passed=True,
                expected="< 5% per position",
                actual="N/A",
                message="No value data available"
            )
            
        max_concentration = max(abs(t.get("value", 0)) / total_value for t in trades)
        
        if max_concentration < 0.10:  # 10% max
            return ValidationResult(
                check_name="concentration",
                passed=True,
                expected="< 10% per position",
                actual=f"{max_concentration:.1%}",
                message=f"Position concentration within limits: {max_concentration:.1%} max"
            )
        else:
            deviation = (max_concentration - 0.10) / 0.05
            return ValidationResult(
                check_name="concentration",
                passed=False,
                expected="< 10% per position",
                actual=f"{max_concentration:.1%}",
                deviation_sigma=deviation,
                message=f"High concentration: {max_concentration:.1%} in single position"
            )
            
    def _generate_recommendations(self, 
                                   anomalies: List[Dict],
                                   results: List[ValidationResult]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        for anomaly in anomalies:
            atype = anomaly["type"]
            deviation = anomaly.get("deviation", 0)
            
            if atype == "sharpe" and deviation > 2:
                recommendations.append(
                    "Consider reducing position sizes until Sharpe recovers"
                )
                recommendations.append(
                    "Review regime detection - may need retraining"
                )
                
            if atype == "drawdown":
                recommendations.append(
                    "Activate circuit breaker: reduce new positions by 50%"
                )
                recommendations.append(
                    "Review stop-loss levels for open positions"
                )
                
            if atype == "execution":
                recommendations.append(
                    "Switch to limit orders during high volatility"
                )
                recommendations.append(
                    "Consider trading during high-liquidity hours only (10am-3pm ET)"
                )
                
            if atype == "concentration":
                recommendations.append(
                    "Reduce max position size parameter"
                )
                recommendations.append(
                    "Increase diversification constraints"
                )
                
            if atype == "trade_count" and anomaly.get("deviation", 0) > 0:
                if "low" in anomaly.get("message", "").lower():
                    recommendations.append(
                        "Check data feeds - may be stale or missing"
                    )
                else:
                    recommendations.append(
                        "Review signal thresholds - may be too sensitive"
                    )
                    
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for r in recommendations:
            if r not in seen:
                seen.add(r)
                unique_recommendations.append(r)
                
        return unique_recommendations
        
    def _log_report(self, report: DailyValidationReport):
        """Log validation report to file."""
        LOG_DIR.mkdir(exist_ok=True)
        
        # Daily log file
        log_file = LOG_DIR / f"validation_{report.date}.json"
        with open(log_file, 'w') as f:
            json.dump(asdict(report), f, indent=2)
            
        # Append to history
        history_file = LOG_DIR / "validation_history.jsonl"
        with open(history_file, 'a') as f:
            f.write(json.dumps({
                "date": report.date,
                "status": report.overall_status,
                "anomaly_count": len(report.anomalies),
                "checks_passed": report.checks_passed,
                "checks_total": report.checks_total,
            }) + "\n")
            
        # Log to stdout
        status_emoji = {"pass": "✅", "warning": "⚠️", "fail": "❌"}
        logger.info(f"{status_emoji.get(report.overall_status, '❓')} Daily validation: "
                   f"{report.checks_passed}/{report.checks_total} checks passed")
        
        if report.anomalies:
            logger.warning(f"Anomalies detected: {[a['type'] for a in report.anomalies]}")


# =============================================================================
# BACKTEST COMPARISON
# =============================================================================

class BacktestComparator:
    """Compare live/paper results to backtest expectations."""
    
    def __init__(self, backtest_results_path: str):
        """
        Initialize with backtest results.
        
        Args:
            backtest_results_path: Path to backtest results JSON
        """
        self.backtest_results = {}
        self._load_backtest_results(backtest_results_path)
        
    def _load_backtest_results(self, path: str):
        """Load backtest results from file."""
        try:
            with open(path, 'r') as f:
                self.backtest_results = json.load(f)
            logger.info(f"Loaded backtest results from {path}")
        except Exception as e:
            logger.warning(f"Failed to load backtest results: {e}")
            self.backtest_results = {
                "sharpe": 1.35,
                "cagr": 0.1641,
                "max_dd": 0.0208,
                "win_rate": 0.52,
            }
            
    def compare(self, live_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare live metrics to backtest.
        
        Returns:
            Comparison report with deviations
        """
        comparisons = []
        
        for metric, backtest_value in self.backtest_results.items():
            if metric in live_metrics:
                live_value = live_metrics[metric]
                deviation_pct = (live_value - backtest_value) / (abs(backtest_value) + 1e-8) * 100
                
                comparisons.append({
                    "metric": metric,
                    "backtest": backtest_value,
                    "live": live_value,
                    "deviation_pct": deviation_pct,
                    "status": "ok" if abs(deviation_pct) < 20 else "warning" if abs(deviation_pct) < 50 else "fail",
                })
                
        return {
            "timestamp": datetime.now().isoformat(),
            "comparisons": comparisons,
            "overall_alignment": sum(1 for c in comparisons if c["status"] == "ok") / max(len(comparisons), 1),
        }

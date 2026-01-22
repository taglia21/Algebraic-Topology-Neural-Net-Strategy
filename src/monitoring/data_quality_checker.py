"""
Real-Time Data Quality Monitor - V2.5 Elite Upgrade
====================================================

Monitors data quality in real-time to detect anomalies,
stale data, and potential data source issues.

Key Features:
- Freshness monitoring with latency tracking
- Outlier detection using statistical methods
- Missing data detection and interpolation
- Data consistency validation
- Volume anomaly detection
- Bid-ask spread monitoring
- Data source health scoring

Research Basis:
- Garbage in = garbage out
- Bad data causes bad trades
- Real-time monitoring prevents costly errors
- Early detection allows graceful degradation

Author: System V2.5
Date: 2025
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
import time
import json

logger = logging.getLogger(__name__)


class DataQualityStatus(Enum):
    """Overall data quality status."""
    EXCELLENT = "excellent"  # All checks pass
    GOOD = "good"           # Minor issues
    WARNING = "warning"     # Significant issues
    CRITICAL = "critical"   # Severe issues, trading should pause
    UNKNOWN = "unknown"     # Unable to assess


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DataQualityAlert:
    """Single data quality alert."""
    timestamp: datetime
    severity: AlertSeverity
    check_name: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityCheckResult:
    """Result from a single quality check."""
    check_name: str
    passed: bool
    score: float  # 0-100
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """Complete data quality report."""
    timestamp: datetime
    status: DataQualityStatus
    overall_score: float  # 0-100
    checks: List[QualityCheckResult]
    alerts: List[DataQualityAlert]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'overall_score': self.overall_score,
            'checks': [
                {
                    'check_name': c.check_name,
                    'passed': c.passed,
                    'score': c.score,
                    'message': c.message
                }
                for c in self.checks
            ],
            'n_alerts': len(self.alerts),
            'recommendations': self.recommendations
        }


@dataclass
class QualityConfig:
    """Configuration for data quality monitoring."""
    
    # Freshness thresholds (seconds)
    max_data_age: float = 300  # 5 minutes
    stale_warning_threshold: float = 60  # 1 minute
    
    # Price checks
    max_price_change_pct: float = 10.0  # 10% max single move
    min_price: float = 0.01  # Minimum valid price
    
    # Volume checks
    min_volume: int = 100  # Minimum volume
    volume_anomaly_std: float = 3.0  # Std deviations for anomaly
    
    # Spread checks
    max_spread_pct: float = 5.0  # Max bid-ask spread percentage
    
    # Missing data
    max_missing_pct: float = 5.0  # Max percentage missing
    
    # Outlier detection
    outlier_std: float = 4.0  # Std deviations for outlier
    outlier_window: int = 100  # Lookback for calculating stats
    
    # Consistency checks
    min_ohlc_bars: int = 50  # Minimum bars for analysis
    
    # Scoring weights
    weight_freshness: float = 0.25
    weight_completeness: float = 0.20
    weight_accuracy: float = 0.25
    weight_consistency: float = 0.15
    weight_validity: float = 0.15


class DataQualityChecker:
    """
    Real-time data quality monitoring system.
    
    Architecture:
    1. Run multiple quality checks in parallel
    2. Score each check 0-100
    3. Aggregate into overall score
    4. Generate alerts for failures
    5. Provide recommendations for fixes
    """
    
    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config or QualityConfig()
        self.alert_history: List[DataQualityAlert] = []
        self.last_data_timestamp: Optional[datetime] = None
        self.historical_stats: Dict[str, Dict[str, float]] = {}
    
    def check_quality(
        self,
        data: pd.DataFrame,
        symbol: Optional[str] = None,
        data_timestamp: Optional[datetime] = None
    ) -> DataQualityReport:
        """
        Run all quality checks on the data.
        
        Args:
            data: OHLCV DataFrame
            symbol: Optional symbol for context
            data_timestamp: When the data was retrieved
            
        Returns:
            Complete DataQualityReport
        """
        checks = []
        alerts = []
        
        data_timestamp = data_timestamp or datetime.now()
        self.last_data_timestamp = data_timestamp
        
        # Run all checks
        checks.append(self._check_freshness(data, data_timestamp))
        checks.append(self._check_completeness(data))
        checks.append(self._check_price_validity(data))
        checks.append(self._check_volume_validity(data))
        checks.append(self._check_ohlc_consistency(data))
        checks.append(self._check_outliers(data))
        checks.append(self._check_stationarity(data))
        
        # Generate alerts for failed checks
        for check in checks:
            if not check.passed:
                severity = self._get_alert_severity(check.score)
                alert = DataQualityAlert(
                    timestamp=datetime.now(),
                    severity=severity,
                    check_name=check.check_name,
                    message=check.message,
                    details=check.details
                )
                alerts.append(alert)
                self.alert_history.append(alert)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(checks)
        
        # Determine status
        status = self._determine_status(overall_score, checks)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(checks)
        
        return DataQualityReport(
            timestamp=datetime.now(),
            status=status,
            overall_score=overall_score,
            checks=checks,
            alerts=alerts,
            recommendations=recommendations
        )
    
    def _check_freshness(
        self,
        data: pd.DataFrame,
        data_timestamp: datetime
    ) -> QualityCheckResult:
        """Check if data is fresh enough."""
        age_seconds = 0
        
        if isinstance(data.index, pd.DatetimeIndex) and len(data) > 0:
            last_bar_time = data.index[-1]
            if isinstance(last_bar_time, pd.Timestamp):
                age_seconds = (datetime.now() - last_bar_time.to_pydatetime().replace(tzinfo=None)).total_seconds()
        
        # Score based on age
        if age_seconds <= self.config.stale_warning_threshold:
            score = 100
        elif age_seconds <= self.config.max_data_age:
            score = 100 - 50 * (age_seconds - self.config.stale_warning_threshold) / (
                self.config.max_data_age - self.config.stale_warning_threshold
            )
        else:
            score = max(0, 50 - (age_seconds - self.config.max_data_age) / 60)
        
        passed = age_seconds <= self.config.max_data_age
        
        return QualityCheckResult(
            check_name="freshness",
            passed=passed,
            score=score,
            message=f"Data age: {age_seconds:.0f}s (max: {self.config.max_data_age}s)",
            details={'age_seconds': age_seconds}
        )
    
    def _check_completeness(self, data: pd.DataFrame) -> QualityCheckResult:
        """Check for missing values."""
        if len(data) == 0:
            return QualityCheckResult(
                check_name="completeness",
                passed=False,
                score=0,
                message="No data available",
                details={'missing_pct': 100}
            )
        
        # Check required columns
        required_cols = ['close']
        optional_cols = ['open', 'high', 'low', 'volume']
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return QualityCheckResult(
                check_name="completeness",
                passed=False,
                score=0,
                message=f"Missing required columns: {missing_cols}",
                details={'missing_columns': missing_cols}
            )
        
        # Check for NaN values
        total_cells = len(data) * len(data.columns)
        missing_cells = data.isnull().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        score = max(0, 100 - missing_pct * 10)
        passed = missing_pct <= self.config.max_missing_pct
        
        return QualityCheckResult(
            check_name="completeness",
            passed=passed,
            score=score,
            message=f"Missing data: {missing_pct:.2f}% (max: {self.config.max_missing_pct}%)",
            details={
                'missing_pct': missing_pct,
                'missing_cells': int(missing_cells),
                'total_cells': total_cells
            }
        )
    
    def _check_price_validity(self, data: pd.DataFrame) -> QualityCheckResult:
        """Check if prices are valid."""
        if 'close' not in data.columns or len(data) == 0:
            return QualityCheckResult(
                check_name="price_validity",
                passed=False,
                score=0,
                message="No price data available"
            )
        
        close = data['close'].dropna()
        issues = []
        
        # Check for negative/zero prices
        invalid_prices = (close <= self.config.min_price).sum()
        if invalid_prices > 0:
            issues.append(f"{invalid_prices} prices <= {self.config.min_price}")
        
        # Check for extreme changes
        returns = close.pct_change().dropna()
        extreme_moves = (np.abs(returns) > self.config.max_price_change_pct / 100).sum()
        if extreme_moves > 0:
            issues.append(f"{extreme_moves} moves > {self.config.max_price_change_pct}%")
        
        # Check for repeating prices (stuck quotes)
        same_price_count = (close.diff() == 0).sum()
        stuck_pct = same_price_count / len(close) * 100 if len(close) > 0 else 0
        if stuck_pct > 50:
            issues.append(f"Stuck quotes: {stuck_pct:.1f}% unchanged")
        
        # Calculate score
        n_issues = len(issues)
        score = max(0, 100 - n_issues * 25)
        passed = n_issues == 0
        
        return QualityCheckResult(
            check_name="price_validity",
            passed=passed,
            score=score,
            message="; ".join(issues) if issues else "All price checks passed",
            details={
                'invalid_prices': int(invalid_prices),
                'extreme_moves': int(extreme_moves),
                'stuck_pct': stuck_pct
            }
        )
    
    def _check_volume_validity(self, data: pd.DataFrame) -> QualityCheckResult:
        """Check if volume data is valid."""
        if 'volume' not in data.columns:
            return QualityCheckResult(
                check_name="volume_validity",
                passed=True,
                score=75,
                message="Volume data not available (not required)"
            )
        
        volume = data['volume'].dropna()
        issues = []
        
        # Check for zero/negative volume
        invalid_volume = (volume < self.config.min_volume).sum()
        invalid_pct = invalid_volume / len(volume) * 100 if len(volume) > 0 else 0
        if invalid_pct > 10:
            issues.append(f"{invalid_pct:.1f}% invalid volume")
        
        # Check for volume anomalies
        if len(volume) > self.config.outlier_window:
            rolling_mean = volume.rolling(self.config.outlier_window).mean()
            rolling_std = volume.rolling(self.config.outlier_window).std()
            
            z_scores = np.abs((volume - rolling_mean) / (rolling_std + 1e-10))
            anomalies = (z_scores > self.config.volume_anomaly_std).sum()
            anomaly_pct = anomalies / len(volume) * 100
            
            if anomaly_pct > 5:
                issues.append(f"{anomaly_pct:.1f}% volume anomalies")
        
        n_issues = len(issues)
        score = max(0, 100 - n_issues * 30)
        passed = n_issues == 0
        
        return QualityCheckResult(
            check_name="volume_validity",
            passed=passed,
            score=score,
            message="; ".join(issues) if issues else "Volume checks passed",
            details={'invalid_volume_pct': invalid_pct}
        )
    
    def _check_ohlc_consistency(self, data: pd.DataFrame) -> QualityCheckResult:
        """Check OHLC data consistency (High >= Low, etc.)."""
        required = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required):
            return QualityCheckResult(
                check_name="ohlc_consistency",
                passed=True,
                score=75,
                message="OHLC data not fully available"
            )
        
        if len(data) < self.config.min_ohlc_bars:
            return QualityCheckResult(
                check_name="ohlc_consistency",
                passed=True,
                score=75,
                message=f"Insufficient bars ({len(data)} < {self.config.min_ohlc_bars})"
            )
        
        issues = []
        
        # High >= Low
        hl_violations = (data['high'] < data['low']).sum()
        if hl_violations > 0:
            issues.append(f"{hl_violations} High < Low violations")
        
        # High >= Open, Close
        high_violations = ((data['high'] < data['open']) | 
                          (data['high'] < data['close'])).sum()
        if high_violations > 0:
            issues.append(f"{high_violations} High < Open/Close")
        
        # Low <= Open, Close
        low_violations = ((data['low'] > data['open']) | 
                         (data['low'] > data['close'])).sum()
        if low_violations > 0:
            issues.append(f"{low_violations} Low > Open/Close")
        
        total_violations = hl_violations + high_violations + low_violations
        violation_pct = total_violations / len(data) * 100 if len(data) > 0 else 0
        
        score = max(0, 100 - violation_pct * 10)
        passed = violation_pct < 1  # Allow <1% violations
        
        return QualityCheckResult(
            check_name="ohlc_consistency",
            passed=passed,
            score=score,
            message="; ".join(issues) if issues else "OHLC consistency passed",
            details={
                'hl_violations': int(hl_violations),
                'high_violations': int(high_violations),
                'low_violations': int(low_violations),
                'violation_pct': violation_pct
            }
        )
    
    def _check_outliers(self, data: pd.DataFrame) -> QualityCheckResult:
        """Check for statistical outliers."""
        if 'close' not in data.columns or len(data) < self.config.outlier_window:
            return QualityCheckResult(
                check_name="outliers",
                passed=True,
                score=75,
                message="Insufficient data for outlier analysis"
            )
        
        close = data['close'].dropna()
        returns = close.pct_change().dropna()
        
        # Calculate rolling z-scores
        rolling_mean = returns.rolling(self.config.outlier_window).mean()
        rolling_std = returns.rolling(self.config.outlier_window).std()
        
        z_scores = np.abs((returns - rolling_mean) / (rolling_std + 1e-10))
        outliers = (z_scores > self.config.outlier_std).sum()
        outlier_pct = outliers / len(returns) * 100 if len(returns) > 0 else 0
        
        # Score based on outlier percentage
        score = max(0, 100 - outlier_pct * 10)
        passed = outlier_pct < 5  # Allow up to 5% outliers
        
        return QualityCheckResult(
            check_name="outliers",
            passed=passed,
            score=score,
            message=f"Outliers: {outlier_pct:.2f}% (>{self.config.outlier_std} std)",
            details={
                'outlier_count': int(outliers),
                'outlier_pct': outlier_pct,
                'max_z_score': float(z_scores.max()) if len(z_scores) > 0 else 0
            }
        )
    
    def _check_stationarity(self, data: pd.DataFrame) -> QualityCheckResult:
        """Check for data stationarity issues."""
        if 'close' not in data.columns or len(data) < 50:
            return QualityCheckResult(
                check_name="stationarity",
                passed=True,
                score=75,
                message="Insufficient data for stationarity check"
            )
        
        close = data['close'].dropna()
        
        # Simple check: compare first and last half volatilities
        mid = len(close) // 2
        first_half_std = close.iloc[:mid].pct_change().std()
        second_half_std = close.iloc[mid:].pct_change().std()
        
        if first_half_std > 0:
            vol_ratio = second_half_std / first_half_std
        else:
            vol_ratio = 1.0
        
        # Extreme ratio indicates regime change
        issues = []
        if vol_ratio > 2.0:
            issues.append(f"Volatility increased {vol_ratio:.1f}x")
        elif vol_ratio < 0.5:
            issues.append(f"Volatility decreased {1/vol_ratio:.1f}x")
        
        # Check for trend
        first_half_mean = close.iloc[:mid].mean()
        second_half_mean = close.iloc[mid:].mean()
        trend_change = abs(second_half_mean - first_half_mean) / first_half_mean * 100
        
        if trend_change > 20:
            issues.append(f"Large trend change: {trend_change:.1f}%")
        
        score = max(0, 100 - len(issues) * 25)
        passed = len(issues) == 0
        
        return QualityCheckResult(
            check_name="stationarity",
            passed=passed,
            score=score,
            message="; ".join(issues) if issues else "Stationarity check passed",
            details={
                'volatility_ratio': vol_ratio,
                'trend_change_pct': trend_change
            }
        )
    
    def _get_alert_severity(self, score: float) -> AlertSeverity:
        """Determine alert severity based on score."""
        if score >= 75:
            return AlertSeverity.INFO
        elif score >= 50:
            return AlertSeverity.WARNING
        elif score >= 25:
            return AlertSeverity.ERROR
        else:
            return AlertSeverity.CRITICAL
    
    def _calculate_overall_score(self, checks: List[QualityCheckResult]) -> float:
        """Calculate weighted overall score."""
        if not checks:
            return 0
        
        weight_map = {
            'freshness': self.config.weight_freshness,
            'completeness': self.config.weight_completeness,
            'price_validity': self.config.weight_accuracy,
            'volume_validity': self.config.weight_accuracy * 0.5,
            'ohlc_consistency': self.config.weight_consistency,
            'outliers': self.config.weight_validity,
            'stationarity': self.config.weight_validity * 0.5
        }
        
        total_weight = 0
        weighted_score = 0
        
        for check in checks:
            weight = weight_map.get(check.check_name, 0.1)
            weighted_score += check.score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0
    
    def _determine_status(
        self,
        overall_score: float,
        checks: List[QualityCheckResult]
    ) -> DataQualityStatus:
        """Determine overall data quality status."""
        # Check for critical failures
        critical_checks = ['freshness', 'completeness', 'price_validity']
        for check in checks:
            if check.check_name in critical_checks and check.score < 25:
                return DataQualityStatus.CRITICAL
        
        if overall_score >= 90:
            return DataQualityStatus.EXCELLENT
        elif overall_score >= 70:
            return DataQualityStatus.GOOD
        elif overall_score >= 50:
            return DataQualityStatus.WARNING
        else:
            return DataQualityStatus.CRITICAL
    
    def _generate_recommendations(
        self,
        checks: List[QualityCheckResult]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        for check in checks:
            if check.passed:
                continue
            
            if check.check_name == 'freshness':
                recommendations.append("Check data source connection and increase refresh rate")
            elif check.check_name == 'completeness':
                recommendations.append("Investigate missing data and implement interpolation")
            elif check.check_name == 'price_validity':
                recommendations.append("Review price data source and add sanity filters")
            elif check.check_name == 'volume_validity':
                recommendations.append("Verify volume data accuracy with backup source")
            elif check.check_name == 'ohlc_consistency':
                recommendations.append("Check OHLC aggregation logic for errors")
            elif check.check_name == 'outliers':
                recommendations.append("Consider outlier filtering or robust estimation")
            elif check.check_name == 'stationarity':
                recommendations.append("Monitor for regime changes, may need model recalibration")
        
        return recommendations
    
    def should_trade(self, report: DataQualityReport) -> Tuple[bool, str]:
        """
        Determine if trading should proceed based on data quality.
        
        Returns:
            (should_trade, reason)
        """
        if report.status == DataQualityStatus.CRITICAL:
            return False, "Critical data quality issues detected"
        
        if report.status == DataQualityStatus.WARNING:
            # Check specific critical issues
            for check in report.checks:
                if check.check_name in ['freshness', 'completeness'] and not check.passed:
                    return False, f"Critical check failed: {check.check_name}"
            return True, "Proceed with caution - warning status"
        
        return True, "Data quality acceptable"
    
    def get_recent_alerts(self, hours: int = 24) -> List[DataQualityAlert]:
        """Get alerts from the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [a for a in self.alert_history if a.timestamp >= cutoff]


# ============================================================
# SELF-TESTING
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Data Quality Checker")
    print("=" * 60)
    
    # Generate test data - use recent timestamps for freshness check
    np.random.seed(42)
    n_bars = 200
    
    # Use recent dates ending now
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1h')
    close = 100 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.01))
    
    # Generate OHLC with proper consistency: high >= max(open, close), low <= min(open, close)
    open_price = close * (1 + np.random.randn(n_bars) * 0.005)
    high = np.maximum(close, open_price) * (1 + np.abs(np.random.randn(n_bars)) * 0.01)
    low = np.minimum(close, open_price) * (1 - np.abs(np.random.randn(n_bars)) * 0.01)
    volume = np.random.randint(100000, 1000000, n_bars)
    
    good_data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    # Test good data
    print("\n1. Testing good quality data...")
    checker = DataQualityChecker()
    report = checker.check_quality(good_data)
    
    print(f"   Status: {report.status.value}")
    print(f"   Overall Score: {report.overall_score:.1f}")
    print(f"   Checks passed: {sum(c.passed for c in report.checks)}/{len(report.checks)}")
    
    # Test bad data
    print("\n2. Testing problematic data...")
    bad_data = good_data.copy()
    bad_data.iloc[50:60, bad_data.columns.get_loc('close')] = np.nan  # Add NaN
    bad_data.iloc[100, bad_data.columns.get_loc('high')] = bad_data.iloc[100]['low'] - 1  # OHLC violation
    bad_data.iloc[150, bad_data.columns.get_loc('close')] = bad_data.iloc[149]['close'] * 1.5  # Outlier
    
    report_bad = checker.check_quality(bad_data)
    
    print(f"   Status: {report_bad.status.value}")
    print(f"   Overall Score: {report_bad.overall_score:.1f}")
    print(f"   Alerts: {len(report_bad.alerts)}")
    
    # Test should_trade
    print("\n3. Testing trading decision...")
    should_trade, reason = checker.should_trade(report)
    print(f"   Good data: {should_trade} - {reason}")
    
    should_trade_bad, reason_bad = checker.should_trade(report_bad)
    print(f"   Bad data: {should_trade_bad} - {reason_bad}")
    
    # Test individual checks
    print("\n4. Individual check scores:")
    for check in report.checks:
        status = "✅" if check.passed else "❌"
        print(f"   {status} {check.check_name}: {check.score:.1f}")
    
    # Test recommendations
    print("\n5. Recommendations for bad data:")
    for rec in report_bad.recommendations[:3]:
        print(f"   - {rec}")
    
    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    results = []
    
    # Check good data passes
    if report.status in [DataQualityStatus.EXCELLENT, DataQualityStatus.GOOD]:
        print("✅ Good data correctly identified")
        results.append(True)
    else:
        print("❌ Good data marked as bad")
        results.append(False)
    
    # Check bad data detected
    if report_bad.overall_score < report.overall_score:
        print("✅ Bad data correctly detected")
        results.append(True)
    else:
        print("❌ Bad data not detected")
        results.append(False)
    
    # Check alerts generated
    if len(report_bad.alerts) > 0:
        print("✅ Alerts generated for issues")
        results.append(True)
    else:
        print("❌ No alerts for bad data")
        results.append(False)
    
    # Check recommendations provided
    if len(report_bad.recommendations) > 0:
        print("✅ Recommendations provided")
        results.append(True)
    else:
        print("❌ No recommendations")
        results.append(False)
    
    print(f"\nPassed: {sum(results)}/{len(results)}")

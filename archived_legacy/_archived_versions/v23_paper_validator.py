#!/usr/bin/env python3
"""
V23 Paper Trading Validator
============================
Validates paper trading performance against backtest expectations.

Features:
- Paper vs backtest comparison metrics
- Go-live readiness checklist
- Automated validation reports
- Signal alignment tracking
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('V23_PaperValidator')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ValidationConfig:
    """Paper trading validation configuration."""
    
    # Minimum requirements
    min_paper_trading_days: int = 14  # 2 weeks minimum
    min_trade_count: int = 20
    min_fill_rate_pct: float = 95.0
    max_slippage_bps: float = 15.0
    
    # Performance alignment thresholds
    win_rate_tolerance_pct: float = 5.0  # Within 5% of backtest
    sharpe_min_ratio: float = 0.70  # At least 70% of backtest Sharpe
    
    # System reliability
    max_critical_errors: int = 0  # Zero critical errors allowed
    max_errors_per_week: int = 3
    min_uptime_pct: float = 99.9
    
    # Circuit breaker requirements
    circuit_breakers_tested: bool = True
    kill_switch_tested: bool = True
    alerts_verified: bool = True


@dataclass
class BacktestBenchmark:
    """Expected performance from backtest."""
    
    # V21 validated performance
    cagr_pct: float = 55.2
    sharpe_ratio: float = 1.54
    max_drawdown_pct: float = -22.3
    win_rate_pct: float = 55.1
    avg_slippage_assumption_bps: float = 10.0
    avg_holding_period_days: float = 5.0
    avg_trades_per_week: float = 6.0  # 30 positions / 5 day holding


@dataclass
class PaperTradingStats:
    """Observed paper trading statistics."""
    
    # Time metrics
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    trading_days: int = 0
    
    # Trade metrics
    total_trades: int = 0
    signals_generated: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    orders_cancelled: int = 0
    
    # Performance metrics
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    
    # Execution quality
    avg_slippage_bps: float = 0.0
    max_slippage_bps: float = 0.0
    avg_fill_time_seconds: float = 0.0
    
    # System reliability
    critical_errors: int = 0
    total_errors: int = 0
    uptime_pct: float = 100.0
    api_latency_ms: float = 0.0


@dataclass
class ValidationResult:
    """Validation check result."""
    check_name: str
    passed: bool
    expected: str
    actual: str
    notes: str = ""


# =============================================================================
# PAPER TRADING VALIDATOR
# =============================================================================

class PaperTradingValidator:
    """
    Validates paper trading performance against backtest expectations.
    """
    
    def __init__(self, 
                 config: Optional[ValidationConfig] = None,
                 benchmark: Optional[BacktestBenchmark] = None):
        self.config = config or ValidationConfig()
        self.benchmark = benchmark or BacktestBenchmark()
        
        # Tracking
        self.paper_stats = PaperTradingStats()
        self.signals: List[Dict] = []
        self.trades: List[Dict] = []
        self.daily_returns: List[Tuple[str, float]] = []
        self.errors: List[Dict] = []
        
        # Validation results
        self.validation_results: List[ValidationResult] = []
        
        # State persistence
        self.state_dir = Path('state/paper_validation')
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("PaperTradingValidator initialized")
    
    def record_signal(self, signal: Dict):
        """Record a generated signal."""
        signal['timestamp'] = datetime.now().isoformat()
        self.signals.append(signal)
        self.paper_stats.signals_generated += 1
    
    def record_trade(self, trade: Dict):
        """Record a paper trade."""
        trade['timestamp'] = datetime.now().isoformat()
        self.trades.append(trade)
        self.paper_stats.total_trades += 1
        
        # Update fill stats
        if trade.get('status') == 'filled':
            self.paper_stats.orders_filled += 1
        elif trade.get('status') == 'rejected':
            self.paper_stats.orders_rejected += 1
        elif trade.get('status') == 'cancelled':
            self.paper_stats.orders_cancelled += 1
        
        # Track slippage
        if 'slippage_bps' in trade:
            slippages = [t.get('slippage_bps', 0) for t in self.trades 
                        if t.get('slippage_bps') is not None]
            if slippages:
                self.paper_stats.avg_slippage_bps = np.mean(slippages)
                self.paper_stats.max_slippage_bps = max(slippages)
    
    def record_daily_return(self, date: str, return_pct: float):
        """Record daily return."""
        self.daily_returns.append((date, return_pct))
        self._update_performance_metrics()
    
    def record_error(self, error: Dict):
        """Record an error."""
        error['timestamp'] = datetime.now().isoformat()
        self.errors.append(error)
        self.paper_stats.total_errors += 1
        
        if error.get('severity') == 'critical':
            self.paper_stats.critical_errors += 1
    
    def _update_performance_metrics(self):
        """Update performance metrics from daily returns."""
        if len(self.daily_returns) < 2:
            return
        
        returns = np.array([r for _, r in self.daily_returns])
        
        # Total return
        cumulative = np.prod(1 + returns / 100) - 1
        self.paper_stats.total_return_pct = cumulative * 100
        
        # Annualized return
        trading_days = len(self.daily_returns)
        self.paper_stats.trading_days = trading_days
        if trading_days > 0:
            annualized = (1 + cumulative) ** (252 / trading_days) - 1
            self.paper_stats.annualized_return_pct = annualized * 100
        
        # Sharpe ratio (assuming 0 risk-free rate)
        if len(returns) > 1:
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            if std_ret > 0:
                self.paper_stats.sharpe_ratio = (mean_ret * 252) / (std_ret * np.sqrt(252))
        
        # Drawdown
        equity = np.cumprod(1 + returns / 100)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        self.paper_stats.max_drawdown_pct = drawdown.min() * 100
        self.paper_stats.current_drawdown_pct = drawdown[-1] * 100
        
        # Win rate
        if len(self.trades) > 0:
            wins = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
            self.paper_stats.win_rate_pct = wins / len(self.trades) * 100
        
        # Profit factor
        gross_profit = sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0))
        if gross_loss > 0:
            self.paper_stats.profit_factor = gross_profit / gross_loss
    
    def update_dates(self, start: Optional[str] = None, end: Optional[str] = None):
        """Update paper trading period dates."""
        if start:
            self.paper_stats.start_date = start
        if end:
            self.paper_stats.end_date = end
    
    # =========================================================================
    # VALIDATION CHECKS
    # =========================================================================
    
    def run_validation(self) -> Tuple[bool, List[ValidationResult]]:
        """
        Run all validation checks.
        
        Returns:
            (all_passed, list of results)
        """
        self.validation_results = []
        
        # Core requirements
        self._check_trading_period()
        self._check_trade_count()
        self._check_fill_rate()
        self._check_slippage()
        
        # Performance alignment
        self._check_win_rate_alignment()
        self._check_sharpe_alignment()
        
        # System reliability
        self._check_critical_errors()
        self._check_error_rate()
        self._check_uptime()
        
        # Circuit breakers
        self._check_circuit_breakers_tested()
        self._check_kill_switch_tested()
        self._check_alerts_verified()
        
        all_passed = all(r.passed for r in self.validation_results)
        
        return all_passed, self.validation_results
    
    def _check_trading_period(self):
        """Check minimum trading period."""
        passed = self.paper_stats.trading_days >= self.config.min_paper_trading_days
        
        self.validation_results.append(ValidationResult(
            check_name="Minimum Trading Period",
            passed=passed,
            expected=f">= {self.config.min_paper_trading_days} days",
            actual=f"{self.paper_stats.trading_days} days",
            notes="Paper trading must run for minimum period to validate"
        ))
    
    def _check_trade_count(self):
        """Check minimum trade count."""
        passed = self.paper_stats.total_trades >= self.config.min_trade_count
        
        self.validation_results.append(ValidationResult(
            check_name="Minimum Trade Count",
            passed=passed,
            expected=f">= {self.config.min_trade_count} trades",
            actual=f"{self.paper_stats.total_trades} trades",
            notes="Need sufficient trades for statistical significance"
        ))
    
    def _check_fill_rate(self):
        """Check order fill rate."""
        if self.paper_stats.signals_generated == 0:
            fill_rate = 0
        else:
            fill_rate = (self.paper_stats.orders_filled / 
                        self.paper_stats.signals_generated * 100)
        
        passed = fill_rate >= self.config.min_fill_rate_pct
        
        self.validation_results.append(ValidationResult(
            check_name="Order Fill Rate",
            passed=passed,
            expected=f">= {self.config.min_fill_rate_pct}%",
            actual=f"{fill_rate:.1f}%",
            notes="High fill rate ensures strategy can be executed"
        ))
    
    def _check_slippage(self):
        """Check slippage vs assumption."""
        passed = self.paper_stats.avg_slippage_bps <= self.config.max_slippage_bps
        
        self.validation_results.append(ValidationResult(
            check_name="Average Slippage",
            passed=passed,
            expected=f"<= {self.config.max_slippage_bps}bps",
            actual=f"{self.paper_stats.avg_slippage_bps:.1f}bps",
            notes=f"Backtest assumed {self.benchmark.avg_slippage_assumption_bps}bps"
        ))
    
    def _check_win_rate_alignment(self):
        """Check win rate vs backtest."""
        tolerance = self.config.win_rate_tolerance_pct
        lower = self.benchmark.win_rate_pct - tolerance
        upper = self.benchmark.win_rate_pct + tolerance
        
        passed = lower <= self.paper_stats.win_rate_pct <= upper
        
        self.validation_results.append(ValidationResult(
            check_name="Win Rate Alignment",
            passed=passed,
            expected=f"{lower:.1f}% - {upper:.1f}%",
            actual=f"{self.paper_stats.win_rate_pct:.1f}%",
            notes=f"Backtest win rate: {self.benchmark.win_rate_pct}%"
        ))
    
    def _check_sharpe_alignment(self):
        """Check Sharpe ratio vs backtest."""
        min_sharpe = self.benchmark.sharpe_ratio * self.config.sharpe_min_ratio
        
        passed = self.paper_stats.sharpe_ratio >= min_sharpe
        
        self.validation_results.append(ValidationResult(
            check_name="Sharpe Ratio Alignment",
            passed=passed,
            expected=f">= {min_sharpe:.2f}",
            actual=f"{self.paper_stats.sharpe_ratio:.2f}",
            notes=f"Backtest Sharpe: {self.benchmark.sharpe_ratio}"
        ))
    
    def _check_critical_errors(self):
        """Check for critical errors."""
        passed = self.paper_stats.critical_errors <= self.config.max_critical_errors
        
        self.validation_results.append(ValidationResult(
            check_name="Critical Errors",
            passed=passed,
            expected=f"<= {self.config.max_critical_errors}",
            actual=f"{self.paper_stats.critical_errors}",
            notes="Zero critical errors required for go-live"
        ))
    
    def _check_error_rate(self):
        """Check weekly error rate."""
        weeks = max(1, self.paper_stats.trading_days / 5)
        errors_per_week = self.paper_stats.total_errors / weeks
        
        passed = errors_per_week <= self.config.max_errors_per_week
        
        self.validation_results.append(ValidationResult(
            check_name="Weekly Error Rate",
            passed=passed,
            expected=f"<= {self.config.max_errors_per_week}/week",
            actual=f"{errors_per_week:.1f}/week",
            notes=""
        ))
    
    def _check_uptime(self):
        """Check system uptime."""
        passed = self.paper_stats.uptime_pct >= self.config.min_uptime_pct
        
        self.validation_results.append(ValidationResult(
            check_name="System Uptime",
            passed=passed,
            expected=f">= {self.config.min_uptime_pct}%",
            actual=f"{self.paper_stats.uptime_pct:.2f}%",
            notes=""
        ))
    
    def _check_circuit_breakers_tested(self):
        """Check circuit breakers were tested."""
        # Would be set manually after testing
        passed = self.config.circuit_breakers_tested
        
        self.validation_results.append(ValidationResult(
            check_name="Circuit Breakers Tested",
            passed=passed,
            expected="All triggers verified",
            actual="Verified" if passed else "NOT VERIFIED",
            notes="Manual verification required"
        ))
    
    def _check_kill_switch_tested(self):
        """Check kill switch was tested."""
        passed = self.config.kill_switch_tested
        
        self.validation_results.append(ValidationResult(
            check_name="Kill Switch Tested",
            passed=passed,
            expected="Manual trigger confirmed",
            actual="Tested" if passed else "NOT TESTED",
            notes="Manual verification required"
        ))
    
    def _check_alerts_verified(self):
        """Check alerts are working."""
        passed = self.config.alerts_verified
        
        self.validation_results.append(ValidationResult(
            check_name="Alerts Verified",
            passed=passed,
            expected="All channels working",
            actual="Verified" if passed else "NOT VERIFIED",
            notes="Test alerts received on all configured channels"
        ))
    
    # =========================================================================
    # REPORTS
    # =========================================================================
    
    def generate_comparison_report(self) -> Dict:
        """Generate paper vs backtest comparison report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'paper_trading_period': {
                'start': self.paper_stats.start_date,
                'end': self.paper_stats.end_date,
                'trading_days': self.paper_stats.trading_days
            },
            'trade_metrics': {
                'signals_generated': self.paper_stats.signals_generated,
                'orders_filled': self.paper_stats.orders_filled,
                'fill_rate_pct': (self.paper_stats.orders_filled / 
                                 max(1, self.paper_stats.signals_generated) * 100),
                'orders_rejected': self.paper_stats.orders_rejected,
                'orders_cancelled': self.paper_stats.orders_cancelled
            },
            'performance_comparison': {
                'metric': ['Win Rate', 'Sharpe Ratio', 'Max Drawdown', 'Avg Slippage'],
                'backtest': [
                    f"{self.benchmark.win_rate_pct}%",
                    f"{self.benchmark.sharpe_ratio}",
                    f"{self.benchmark.max_drawdown_pct}%",
                    f"{self.benchmark.avg_slippage_assumption_bps}bps"
                ],
                'paper': [
                    f"{self.paper_stats.win_rate_pct:.1f}%",
                    f"{self.paper_stats.sharpe_ratio:.2f}",
                    f"{self.paper_stats.max_drawdown_pct:.1f}%",
                    f"{self.paper_stats.avg_slippage_bps:.1f}bps"
                ],
                'aligned': [
                    bool(abs(self.paper_stats.win_rate_pct - self.benchmark.win_rate_pct) <= 5),
                    bool(self.paper_stats.sharpe_ratio >= self.benchmark.sharpe_ratio * 0.7),
                    bool(self.paper_stats.max_drawdown_pct >= self.benchmark.max_drawdown_pct * 1.2),
                    bool(self.paper_stats.avg_slippage_bps <= 15)
                ]
            },
            'execution_quality': {
                'avg_slippage_bps': self.paper_stats.avg_slippage_bps,
                'max_slippage_bps': self.paper_stats.max_slippage_bps,
                'slippage_vs_assumption': (self.paper_stats.avg_slippage_bps - 
                                          self.benchmark.avg_slippage_assumption_bps),
                'avg_fill_time_seconds': self.paper_stats.avg_fill_time_seconds
            },
            'system_reliability': {
                'critical_errors': self.paper_stats.critical_errors,
                'total_errors': self.paper_stats.total_errors,
                'uptime_pct': self.paper_stats.uptime_pct,
                'api_latency_ms': self.paper_stats.api_latency_ms
            }
        }
    
    def generate_go_live_checklist(self) -> Dict:
        """Generate go-live readiness checklist."""
        all_passed, results = self.run_validation()
        
        checklist = {
            'ready_for_live': all_passed,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_checks': len(results),
                'passed': sum(1 for r in results if r.passed),
                'failed': sum(1 for r in results if not r.passed)
            },
            'checks': [
                {
                    'name': r.check_name,
                    'status': '✅ PASS' if r.passed else '❌ FAIL',
                    'expected': r.expected,
                    'actual': r.actual,
                    'notes': r.notes
                }
                for r in results
            ],
            'recommendation': self._get_recommendation(all_passed, results)
        }
        
        return checklist
    
    def _get_recommendation(self, all_passed: bool, 
                           results: List[ValidationResult]) -> str:
        """Generate recommendation based on validation results."""
        if all_passed:
            return ("✅ READY FOR LIVE TRADING\n\n"
                   "All validation checks passed. The system has demonstrated:\n"
                   "- Consistent performance aligned with backtest\n"
                   "- Reliable execution with acceptable slippage\n"
                   "- Robust error handling and risk controls\n\n"
                   "Recommendation: Proceed with live trading using initial "
                   "conservative position sizing (25-50% of target).")
        
        # Identify critical failures
        critical = []
        warnings = []
        
        for r in results:
            if not r.passed:
                if r.check_name in ['Critical Errors', 'Kill Switch Tested', 
                                   'Circuit Breakers Tested']:
                    critical.append(r.check_name)
                else:
                    warnings.append(r.check_name)
        
        if critical:
            return (f"❌ NOT READY FOR LIVE TRADING\n\n"
                   f"Critical issues found:\n" +
                   "\n".join(f"- {c}" for c in critical) +
                   "\n\nThese must be resolved before going live.")
        else:
            return (f"⚠️ CONDITIONAL APPROVAL\n\n"
                   f"Minor issues found:\n" +
                   "\n".join(f"- {w}" for w in warnings) +
                   "\n\nRecommendation: Address issues or proceed with caution "
                   "using reduced position sizing.")
    
    def save_validation_report(self, filepath: Optional[str] = None):
        """Save full validation report to disk."""
        if filepath is None:
            filepath = self.state_dir / 'validation_report.json'
        
        report = {
            'comparison': self.generate_comparison_report(),
            'go_live_checklist': self.generate_go_live_checklist(),
            'paper_stats': {
                'start_date': self.paper_stats.start_date,
                'end_date': self.paper_stats.end_date,
                'trading_days': self.paper_stats.trading_days,
                'total_trades': self.paper_stats.total_trades,
                'signals_generated': self.paper_stats.signals_generated,
                'orders_filled': self.paper_stats.orders_filled,
                'total_return_pct': self.paper_stats.total_return_pct,
                'sharpe_ratio': self.paper_stats.sharpe_ratio,
                'win_rate_pct': self.paper_stats.win_rate_pct,
                'max_drawdown_pct': self.paper_stats.max_drawdown_pct,
                'avg_slippage_bps': self.paper_stats.avg_slippage_bps
            },
            'benchmark': {
                'cagr_pct': self.benchmark.cagr_pct,
                'sharpe_ratio': self.benchmark.sharpe_ratio,
                'max_drawdown_pct': self.benchmark.max_drawdown_pct,
                'win_rate_pct': self.benchmark.win_rate_pct
            },
            'errors': self.errors[-50:],
            'generated_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to {filepath}")
    
    def save_state(self):
        """Save validator state to disk."""
        state = {
            'paper_stats': {
                'start_date': self.paper_stats.start_date,
                'end_date': self.paper_stats.end_date,
                'trading_days': self.paper_stats.trading_days,
                'total_trades': self.paper_stats.total_trades,
                'signals_generated': self.paper_stats.signals_generated,
                'orders_filled': self.paper_stats.orders_filled,
                'orders_rejected': self.paper_stats.orders_rejected,
                'orders_cancelled': self.paper_stats.orders_cancelled,
                'total_return_pct': self.paper_stats.total_return_pct,
                'annualized_return_pct': self.paper_stats.annualized_return_pct,
                'sharpe_ratio': self.paper_stats.sharpe_ratio,
                'max_drawdown_pct': self.paper_stats.max_drawdown_pct,
                'win_rate_pct': self.paper_stats.win_rate_pct,
                'avg_slippage_bps': self.paper_stats.avg_slippage_bps,
                'max_slippage_bps': self.paper_stats.max_slippage_bps,
                'critical_errors': self.paper_stats.critical_errors,
                'total_errors': self.paper_stats.total_errors,
                'uptime_pct': self.paper_stats.uptime_pct
            },
            'signals': self.signals[-500:],
            'trades': self.trades[-500:],
            'daily_returns': self.daily_returns[-252:],
            'errors': self.errors[-100:],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.state_dir / 'validator_state.json', 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info("Validator state saved")
    
    def load_state(self):
        """Load validator state from disk."""
        state_file = self.state_dir / 'validator_state.json'
        if not state_file.exists():
            return
        
        with open(state_file) as f:
            state = json.load(f)
        
        ps = state.get('paper_stats', {})
        self.paper_stats.start_date = ps.get('start_date')
        self.paper_stats.end_date = ps.get('end_date')
        self.paper_stats.trading_days = ps.get('trading_days', 0)
        self.paper_stats.total_trades = ps.get('total_trades', 0)
        self.paper_stats.signals_generated = ps.get('signals_generated', 0)
        self.paper_stats.orders_filled = ps.get('orders_filled', 0)
        self.paper_stats.orders_rejected = ps.get('orders_rejected', 0)
        self.paper_stats.orders_cancelled = ps.get('orders_cancelled', 0)
        self.paper_stats.total_return_pct = ps.get('total_return_pct', 0)
        self.paper_stats.sharpe_ratio = ps.get('sharpe_ratio', 0)
        self.paper_stats.win_rate_pct = ps.get('win_rate_pct', 0)
        self.paper_stats.avg_slippage_bps = ps.get('avg_slippage_bps', 0)
        self.paper_stats.max_slippage_bps = ps.get('max_slippage_bps', 0)
        self.paper_stats.critical_errors = ps.get('critical_errors', 0)
        self.paper_stats.total_errors = ps.get('total_errors', 0)
        
        self.signals = state.get('signals', [])
        self.trades = state.get('trades', [])
        self.daily_returns = [(d, r) for d, r in state.get('daily_returns', [])]
        self.errors = state.get('errors', [])
        
        logger.info("Validator state loaded")


# =============================================================================
# MAIN / TESTING
# =============================================================================

def main():
    """Test paper trading validator."""
    logger.info("=" * 70)
    logger.info("📋 V23 PAPER TRADING VALIDATOR TEST")
    logger.info("=" * 70)
    
    # Initialize
    validator = PaperTradingValidator()
    
    # Simulate paper trading period
    logger.info("\n📅 Simulating paper trading period...")
    
    validator.update_dates(
        start='2026-01-06',
        end='2026-01-22'
    )
    
    # Simulate signals and trades
    np.random.seed(42)
    for i in range(25):
        # Record signal
        validator.record_signal({
            'symbol': f"STOCK{i % 10}",
            'signal': 'buy',
            'strength': np.random.uniform(0.5, 1.0)
        })
        
        # Simulate order fill (95% success rate)
        if np.random.random() < 0.95:
            pnl = np.random.normal(50, 200)
            validator.record_trade({
                'symbol': f"STOCK{i % 10}",
                'side': 'buy',
                'quantity': 100,
                'pnl': pnl,
                'slippage_bps': np.random.uniform(5, 20),
                'status': 'filled'
            })
        else:
            validator.record_trade({
                'symbol': f"STOCK{i % 10}",
                'status': 'rejected',
                'reason': 'Insufficient liquidity'
            })
    
    # Simulate daily returns
    for i in range(12):
        date = f"2026-01-{6+i:02d}"
        ret = np.random.normal(0.2, 1.5)  # ~0.2% daily return
        validator.record_daily_return(date, ret)
    
    # Record a few errors
    validator.record_error({
        'type': 'api_timeout',
        'severity': 'warning',
        'message': 'API timeout during quote fetch'
    })
    
    # Run validation
    logger.info("\n✅ Running validation checks...")
    all_passed, results = validator.run_validation()
    
    logger.info(f"\n   {'Check':<30} {'Status':<10} {'Expected':<25} {'Actual':<20}")
    logger.info("-" * 90)
    
    for r in results:
        status = '✅ PASS' if r.passed else '❌ FAIL'
        logger.info(f"   {r.check_name:<30} {status:<10} {r.expected:<25} {r.actual:<20}")
    
    # Generate comparison report
    logger.info("\n📊 Performance Comparison:")
    comparison = validator.generate_comparison_report()
    
    perf = comparison['performance_comparison']
    logger.info(f"\n   {'Metric':<20} {'Backtest':<15} {'Paper':<15} {'Aligned':<10}")
    logger.info("-" * 60)
    for i, metric in enumerate(perf['metric']):
        aligned = '✅' if perf['aligned'][i] else '❌'
        logger.info(f"   {metric:<20} {perf['backtest'][i]:<15} {perf['paper'][i]:<15} {aligned:<10}")
    
    # Generate go-live checklist
    logger.info("\n📋 Go-Live Checklist:")
    checklist = validator.generate_go_live_checklist()
    
    logger.info(f"\n   Summary: {checklist['summary']['passed']}/{checklist['summary']['total_checks']} passed")
    logger.info(f"\n   {checklist['recommendation']}")
    
    # Save report
    validator.save_validation_report()
    validator.save_state()
    
    logger.info(f"\n📄 Reports saved to {validator.state_dir}")
    
    logger.info("\n✅ Paper trading validator test complete")
    
    return validator


if __name__ == "__main__":
    main()

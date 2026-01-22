#!/usr/bin/env python3
"""
V2.5 Production Validation Script
===================================

Validates V2.5 Elite components on REAL market data from Polygon.io.

Test Universe:
- ETFs: SPY, QQQ, IWM
- High-Volume Stocks: AAPL, MSFT, NVDA, TSLA, GOOGL, AMZN, META, AMD, NFLX, COST, 
                      JPM, BAC, WMT, HD, UNH, PG, V, MA, CRM, ORCL

Date Range: 2 years (504 trading days) for robust walk-forward validation

Metrics Tracked:
- Sharpe Ratio (target > 2.0)
- Sortino Ratio (target > 2.5)
- Win Rate (target > 52%)
- Profit Factor (target > 1.5)
- Max Drawdown (target < 15%)
- Feature generation time (target < 500ms)
- Ensemble prediction time (target < 200ms)
- Memory usage (target < 6GB)

Output:
- results/v25_validation/performance_report.json
- results/v25_validation/equity_curve.png
- results/v25_validation/feature_importance_top20.csv
- results/v25_validation/ensemble_model_weights.json
- results/v25_validation/trade_log.csv
"""

import os
import sys
import json
import time
import logging
import traceback
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ValidationConfig:
    """Configuration for V2.5 production validation."""
    
    # Universe
    etfs: List[str] = field(default_factory=lambda: ['SPY', 'QQQ', 'IWM'])
    stocks: List[str] = field(default_factory=lambda: [
        'AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'AMZN', 'META', 'AMD', 'NFLX', 'COST',
        'JPM', 'BAC', 'WMT', 'HD', 'UNH', 'PG', 'V', 'MA', 'CRM', 'ORCL'
    ])
    
    # Date range (2 years of data)
    lookback_days: int = 504
    
    # Walk-forward parameters
    train_pct: float = 0.7
    validation_pct: float = 0.15
    test_pct: float = 0.15
    
    # Trading parameters
    initial_capital: float = 1_000_000
    position_size_pct: float = 0.05  # 5% per position
    max_positions: int = 10
    slippage_bps: float = 5  # 5 basis points
    commission_bps: float = 1  # 1 basis point
    
    # Signal parameters
    signal_threshold: float = 0.6  # Minimum confidence for trade
    min_holding_days: int = 3
    max_holding_days: int = 20
    
    # Targets
    target_sharpe: float = 2.0
    target_sortino: float = 2.5
    target_win_rate: float = 0.52
    target_profit_factor: float = 1.5
    target_max_dd: float = 0.15
    target_feature_time_ms: float = 500
    target_prediction_time_ms: float = 200
    target_memory_gb: float = 6.0
    
    @property
    def universe(self) -> List[str]:
        return self.etfs + self.stocks


# =============================================================================
# METRICS TRACKING
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics for validation."""
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    avg_trade_return: float = 0.0
    total_trades: int = 0
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    volatility_pct: float = 0.0
    calmar_ratio: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LatencyMetrics:
    """Latency metrics for components."""
    feature_gen_mean_ms: float = 0.0
    feature_gen_max_ms: float = 0.0
    ensemble_pred_mean_ms: float = 0.0
    ensemble_pred_max_ms: float = 0.0
    signal_validation_mean_ms: float = 0.0
    total_pipeline_mean_ms: float = 0.0
    memory_peak_gb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    """Complete validation result."""
    status: str = "pending"  # pending, passed, conditional, failed
    recommendation: str = ""
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    data_quality: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    ensemble_weights: Dict[str, float] = field(default_factory=dict)
    trade_summary: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: str = ""
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status,
            'recommendation': self.recommendation,
            'performance': self.performance.to_dict(),
            'latency': self.latency.to_dict(),
            'data_quality': self.data_quality,
            'feature_importance': self.feature_importance,
            'ensemble_weights': self.ensemble_weights,
            'trade_summary': self.trade_summary,
            'errors': self.errors,
            'warnings': self.warnings,
            'timestamp': self.timestamp,
            'duration_seconds': self.duration_seconds,
        }


# =============================================================================
# V2.5 COMPONENT LOADER
# =============================================================================

class V25ComponentLoader:
    """Load and initialize V2.5 components."""
    
    def __init__(self):
        self.feature_engineer = None
        self.ensemble = None
        self.signal_validator = None
        self.quality_checker = None
        self.components_available = {}
    
    def load_all(self) -> Dict[str, bool]:
        """Load all V2.5 components."""
        
        # 1. Elite Feature Engineer
        try:
            from src.features.elite_feature_engineer import (
                EliteFeatureEngineer, FeatureConfig
            )
            self.feature_engineer = EliteFeatureEngineer(FeatureConfig())
            self.components_available['feature_engineer'] = True
            logger.info("✅ Elite Feature Engineer loaded")
        except Exception as e:
            self.components_available['feature_engineer'] = False
            logger.warning(f"⚠️ Elite Feature Engineer failed: {e}")
        
        # 2. Gradient Boost Ensemble
        try:
            from src.ml.gradient_boost_ensemble import (
                GradientBoostEnsemble, EnsembleConfig
            )
            self.ensemble = GradientBoostEnsemble(EnsembleConfig())
            self.components_available['ensemble'] = True
            logger.info("✅ Gradient Boost Ensemble loaded")
        except Exception as e:
            self.components_available['ensemble'] = False
            logger.warning(f"⚠️ Gradient Boost Ensemble failed: {e}")
        
        # 3. Multi-Indicator Validator
        try:
            from src.validation.multi_indicator_validator import (
                MultiIndicatorValidator, ValidatorConfig
            )
            self.signal_validator = MultiIndicatorValidator(ValidatorConfig())
            self.components_available['signal_validator'] = True
            logger.info("✅ Multi-Indicator Validator loaded")
        except Exception as e:
            self.components_available['signal_validator'] = False
            logger.warning(f"⚠️ Multi-Indicator Validator failed: {e}")
        
        # 4. Data Quality Checker
        try:
            from src.monitoring.data_quality_checker import (
                DataQualityChecker, QualityConfig
            )
            self.quality_checker = DataQualityChecker(QualityConfig())
            self.components_available['quality_checker'] = True
            logger.info("✅ Data Quality Checker loaded")
        except Exception as e:
            self.components_available['quality_checker'] = False
            logger.warning(f"⚠️ Data Quality Checker failed: {e}")
        
        return self.components_available


# =============================================================================
# DATA FETCHER
# =============================================================================

class MarketDataFetcher:
    """Fetch real market data from Polygon."""
    
    def __init__(self):
        self.provider = None
        self.provider_name = "none"
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize data provider."""
        try:
            from src.data.hybrid_provider import HybridDataProvider
            self.provider = HybridDataProvider()
            self.provider_name = self.provider.active_provider
            logger.info(f"✅ Data provider initialized: {self.provider_name}")
        except Exception as e:
            logger.warning(f"⚠️ HybridDataProvider failed: {e}")
            # Try yfinance directly
            try:
                import yfinance as yf
                self.provider_name = "yfinance_direct"
                logger.info("✅ Using yfinance directly")
            except ImportError:
                logger.error("❌ No data provider available!")
    
    def fetch_universe(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        """Fetch OHLCV data for all tickers."""
        
        data = {}
        quality_stats = {
            'total_tickers': len(tickers),
            'successful': 0,
            'failed': [],
            'provider': self.provider_name,
            'date_range': {'start': start_date, 'end': end_date},
            'bars_per_ticker': {},
        }
        
        logger.info(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")
        
        if self.provider is not None:
            try:
                data = self.provider.fetch_batch_parallel(
                    tickers, start_date, end_date
                )
                for ticker, df in data.items():
                    if df is not None and len(df) > 0:
                        quality_stats['successful'] += 1
                        quality_stats['bars_per_ticker'][ticker] = len(df)
                    else:
                        quality_stats['failed'].append(ticker)
            except Exception as e:
                logger.error(f"Batch fetch failed: {e}")
        elif self.provider_name == "yfinance_direct":
            import yfinance as yf
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    df = stock.history(start=start_date, end=end_date)
                    if df is not None and len(df) > 0:
                        df.columns = df.columns.str.lower().str.replace(' ', '_')
                        data[ticker] = df
                        quality_stats['successful'] += 1
                        quality_stats['bars_per_ticker'][ticker] = len(df)
                    else:
                        quality_stats['failed'].append(ticker)
                except Exception as e:
                    quality_stats['failed'].append(ticker)
                    logger.debug(f"Failed to fetch {ticker}: {e}")
        
        logger.info(f"Fetched {quality_stats['successful']}/{len(tickers)} tickers")
        return data, quality_stats


# =============================================================================
# BACKTESTER
# =============================================================================

class V25Backtester:
    """Run backtest using V2.5 components."""
    
    def __init__(
        self,
        config: ValidationConfig,
        components: V25ComponentLoader,
    ):
        self.config = config
        self.components = components
        
        # State tracking
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.latencies: Dict[str, List[float]] = {
            'feature_gen': [],
            'ensemble_pred': [],
            'signal_validation': [],
            'total_pipeline': [],
        }
    
    def _calculate_returns(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate returns matrix from OHLCV data."""
        close_data = {}
        for ticker, df in data.items():
            if df is not None and 'close' in df.columns:
                close_data[ticker] = df['close']
        
        if not close_data:
            return pd.DataFrame()
        
        close_df = pd.DataFrame(close_data)
        returns = close_df.pct_change().dropna()
        return returns
    
    def _generate_features(self, ticker: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate features for a single ticker."""
        if self.components.feature_engineer is None:
            return None
        
        start_time = time.perf_counter()
        try:
            features = self.components.feature_engineer.generate_features(df)
            latency = (time.perf_counter() - start_time) * 1000
            self.latencies['feature_gen'].append(latency)
            return features
        except Exception as e:
            logger.debug(f"Feature generation failed for {ticker}: {e}")
            return None
    
    def _get_ensemble_prediction(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
    ) -> Tuple[float, float]:
        """Get ensemble prediction and confidence."""
        if self.components.ensemble is None:
            return 0.0, 0.0
        
        start_time = time.perf_counter()
        try:
            # Prepare data for ensemble
            X = features.iloc[:-1].values
            y = returns.iloc[1:len(X)+1].values
            
            if len(X) < 100:
                return 0.0, 0.0
            
            # Train on historical, predict next
            train_size = int(len(X) * 0.8)
            X_train, y_train = X[:train_size], y[:train_size]
            X_test = X[train_size:]
            
            self.components.ensemble.fit(X_train, y_train)
            predictions = self.components.ensemble.predict(X_test)
            
            latency = (time.perf_counter() - start_time) * 1000
            self.latencies['ensemble_pred'].append(latency)
            
            # Return latest prediction and confidence
            if len(predictions) > 0:
                pred = predictions[-1]
                confidence = min(abs(pred) / 0.01, 1.0)  # Normalize to 0-1
                return pred, confidence
            return 0.0, 0.0
        except Exception as e:
            logger.debug(f"Ensemble prediction failed: {e}")
            return 0.0, 0.0
    
    def _validate_signal(
        self,
        df: pd.DataFrame,
        direction: str,
    ) -> Tuple[bool, int]:
        """Validate signal using multi-indicator validator."""
        if self.components.signal_validator is None:
            return True, 5  # Default to valid if validator not available
        
        start_time = time.perf_counter()
        try:
            result = self.components.signal_validator.validate(df, direction)
            latency = (time.perf_counter() - start_time) * 1000
            self.latencies['signal_validation'].append(latency)
            return result.is_valid, result.confirmed_count
        except Exception as e:
            logger.debug(f"Signal validation failed: {e}")
            return True, 5
    
    def run_walkforward(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> Tuple[PerformanceMetrics, List[Dict]]:
        """Run walk-forward backtest."""
        
        returns_df = self._calculate_returns(data)
        if returns_df.empty:
            logger.error("No returns data available")
            return PerformanceMetrics(), []
        
        # Align all data to same dates
        common_dates = returns_df.index
        n_days = len(common_dates)
        
        # Walk-forward splits
        train_end = int(n_days * self.config.train_pct)
        val_end = int(n_days * (self.config.train_pct + self.config.validation_pct))
        
        logger.info(f"Walk-forward: Train {train_end} days, Val {val_end - train_end} days, Test {n_days - val_end} days")
        
        # Initialize portfolio
        capital = self.config.initial_capital
        self.equity_curve = [capital]
        
        # Run through test period
        test_dates = common_dates[val_end:]
        daily_returns = []
        
        for i, date in enumerate(test_dates):
            pipeline_start = time.perf_counter()
            date_pnl = 0.0
            
            # Check existing positions for exits
            for ticker in list(self.positions.keys()):
                pos = self.positions[ticker]
                holding_days = i - pos['entry_idx']
                
                # Exit conditions
                should_exit = (
                    holding_days >= self.config.max_holding_days or
                    (holding_days >= self.config.min_holding_days and i == len(test_dates) - 1)
                )
                
                if should_exit and ticker in returns_df.columns:
                    current_return = returns_df.loc[date, ticker]
                    trade_return = pos['pnl_pct'] + current_return
                    
                    self.trades.append({
                        'ticker': ticker,
                        'entry_date': pos['entry_date'],
                        'exit_date': str(date.date()) if hasattr(date, 'date') else str(date),
                        'direction': pos['direction'],
                        'size': pos['size'],
                        'entry_price': pos['entry_price'],
                        'pnl_pct': trade_return,
                        'pnl_usd': trade_return * pos['size'],
                        'holding_days': holding_days,
                    })
                    
                    date_pnl += trade_return * pos['size']
                    del self.positions[ticker]
            
            # Update existing positions
            for ticker in self.positions:
                if ticker in returns_df.columns:
                    self.positions[ticker]['pnl_pct'] += returns_df.loc[date, ticker]
            
            # Generate new signals
            if len(self.positions) < self.config.max_positions:
                for ticker in self.config.universe:
                    if ticker in self.positions or ticker not in data:
                        continue
                    
                    df = data[ticker]
                    if df is None or len(df) < 100:
                        continue
                    
                    # Get data up to current date
                    mask = df.index <= date
                    df_slice = df[mask].tail(200)
                    
                    if len(df_slice) < 50:
                        continue
                    
                    # Generate features
                    features = self._generate_features(ticker, df_slice)
                    if features is None:
                        continue
                    
                    # Get ensemble prediction
                    if ticker in returns_df.columns:
                        returns_slice = returns_df.loc[:date, ticker].tail(len(features))
                        pred, confidence = self._get_ensemble_prediction(features, returns_slice)
                        
                        if confidence >= self.config.signal_threshold:
                            direction = 'long' if pred > 0 else 'short'
                            
                            # Validate signal
                            is_valid, confirmed_count = self._validate_signal(df_slice, direction)
                            
                            if is_valid and confirmed_count >= 5:
                                position_size = capital * self.config.position_size_pct
                                
                                # Apply slippage
                                slippage = self.config.slippage_bps / 10000
                                
                                self.positions[ticker] = {
                                    'entry_date': str(date.date()) if hasattr(date, 'date') else str(date),
                                    'entry_idx': i,
                                    'direction': direction,
                                    'size': position_size,
                                    'entry_price': df_slice['close'].iloc[-1],
                                    'pnl_pct': -slippage,  # Initial slippage cost
                                }
                                
                                if len(self.positions) >= self.config.max_positions:
                                    break
            
            # Calculate daily return
            position_value = sum(
                pos['size'] * (1 + pos['pnl_pct']) for pos in self.positions.values()
            )
            cash = capital - sum(pos['size'] for pos in self.positions.values())
            total_value = position_value + cash + date_pnl
            
            daily_return = (total_value - capital) / capital
            daily_returns.append(daily_return)
            capital = total_value
            self.equity_curve.append(capital)
            
            pipeline_latency = (time.perf_counter() - pipeline_start) * 1000
            self.latencies['total_pipeline'].append(pipeline_latency)
        
        # Calculate metrics
        metrics = self._calculate_metrics(daily_returns)
        return metrics, self.trades
    
    def _calculate_metrics(self, daily_returns: List[float]) -> PerformanceMetrics:
        """Calculate performance metrics from returns."""
        if not daily_returns:
            return PerformanceMetrics()
        
        returns = np.array(daily_returns)
        
        # Basic stats
        total_return = (self.equity_curve[-1] / self.equity_curve[0]) - 1
        n_days = len(returns)
        annual_factor = 252 / n_days if n_days > 0 else 1
        annual_return = (1 + total_return) ** annual_factor - 1
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe (assuming 5% risk-free rate)
        rf_daily = 0.05 / 252
        excess_returns = returns - rf_daily
        sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Sortino (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.01
        sortino = np.mean(excess_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Max drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = abs(np.min(drawdown))
        
        # Trade stats
        if self.trades:
            trade_returns = [t['pnl_pct'] for t in self.trades]
            wins = [r for r in trade_returns if r > 0]
            losses = [r for r in trade_returns if r <= 0]
            
            win_rate = len(wins) / len(trade_returns) if trade_returns else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 0.01
            profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return PerformanceMetrics(
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            win_rate=round(win_rate, 3),
            profit_factor=round(profit_factor, 3),
            max_drawdown=round(max_dd, 4),
            avg_trade_return=round(np.mean([t['pnl_pct'] for t in self.trades]) if self.trades else 0, 5),
            total_trades=len(self.trades),
            total_return_pct=round(total_return * 100, 2),
            annual_return_pct=round(annual_return * 100, 2),
            volatility_pct=round(volatility * 100, 2),
            calmar_ratio=round(annual_return / max_dd if max_dd > 0 else 0, 3),
            winning_trades=len([t for t in self.trades if t['pnl_pct'] > 0]),
            losing_trades=len([t for t in self.trades if t['pnl_pct'] <= 0]),
            avg_win=round(avg_win * 100, 3),
            avg_loss=round(avg_loss * 100, 3),
            largest_win=round(max([t['pnl_pct'] for t in self.trades]) * 100 if self.trades else 0, 3),
            largest_loss=round(min([t['pnl_pct'] for t in self.trades]) * 100 if self.trades else 0, 3),
        )
    
    def get_latency_metrics(self) -> LatencyMetrics:
        """Get latency metrics."""
        import psutil  # type: ignore[import-not-found]
        
        return LatencyMetrics(
            feature_gen_mean_ms=round(np.mean(self.latencies['feature_gen']) if self.latencies['feature_gen'] else 0, 2),
            feature_gen_max_ms=round(max(self.latencies['feature_gen']) if self.latencies['feature_gen'] else 0, 2),
            ensemble_pred_mean_ms=round(np.mean(self.latencies['ensemble_pred']) if self.latencies['ensemble_pred'] else 0, 2),
            ensemble_pred_max_ms=round(max(self.latencies['ensemble_pred']) if self.latencies['ensemble_pred'] else 0, 2),
            signal_validation_mean_ms=round(np.mean(self.latencies['signal_validation']) if self.latencies['signal_validation'] else 0, 2),
            total_pipeline_mean_ms=round(np.mean(self.latencies['total_pipeline']) if self.latencies['total_pipeline'] else 0, 2),
            memory_peak_gb=round(psutil.Process().memory_info().rss / 1e9, 2),
        )


# =============================================================================
# DECISION ENGINE
# =============================================================================

def determine_recommendation(
    config: ValidationConfig,
    metrics: PerformanceMetrics,
    latency: LatencyMetrics,
) -> Tuple[str, str]:
    """
    Determine GO/NO-GO recommendation based on metrics.
    
    Returns:
        status: 'passed', 'conditional', 'failed'
        recommendation: Human-readable recommendation
    """
    
    # Check core metrics
    sharpe_ok = metrics.sharpe_ratio >= config.target_sharpe
    win_rate_ok = metrics.win_rate >= config.target_win_rate
    max_dd_ok = metrics.max_drawdown <= config.target_max_dd
    
    # Check latency
    feature_time_ok = latency.feature_gen_mean_ms <= config.target_feature_time_ms
    memory_ok = latency.memory_peak_gb <= config.target_memory_gb
    
    # Conditional thresholds
    sharpe_conditional = metrics.sharpe_ratio >= 1.5
    win_rate_conditional = metrics.win_rate >= 0.48
    
    if sharpe_ok and win_rate_ok and max_dd_ok:
        status = "passed"
        recommendation = (
            f"GO - Proceed to Production Integration\n"
            f"• Sharpe {metrics.sharpe_ratio:.2f} exceeds target {config.target_sharpe}\n"
            f"• Win Rate {metrics.win_rate:.1%} exceeds target {config.target_win_rate:.1%}\n"
            f"• Max DD {metrics.max_drawdown:.1%} within target {config.target_max_dd:.1%}\n"
            f"• Ready for paper trading validation"
        )
    elif sharpe_conditional and win_rate_conditional:
        status = "conditional"
        issues = []
        if not sharpe_ok:
            issues.append(f"Sharpe {metrics.sharpe_ratio:.2f} below target {config.target_sharpe}")
        if not win_rate_ok:
            issues.append(f"Win Rate {metrics.win_rate:.1%} below target {config.target_win_rate:.1%}")
        if not max_dd_ok:
            issues.append(f"Max DD {metrics.max_drawdown:.1%} exceeds limit {config.target_max_dd:.1%}")
        
        recommendation = (
            f"CONDITIONAL GO - Tune hyperparameters and retest\n"
            f"• Issues: {'; '.join(issues)}\n"
            f"• Suggestion: Run Bayesian optimization on ensemble weights\n"
            f"• Consider increasing signal threshold or min holding period"
        )
    else:
        status = "failed"
        recommendation = (
            f"NO-GO - Debug feature quality and ensemble\n"
            f"• Sharpe: {metrics.sharpe_ratio:.2f} (target: {config.target_sharpe})\n"
            f"• Win Rate: {metrics.win_rate:.1%} (target: {config.target_win_rate:.1%})\n"
            f"• Max DD: {metrics.max_drawdown:.1%} (target: <{config.target_max_dd:.1%})\n"
            f"• Action: Analyze feature importance and check data quality\n"
            f"• Action: Review ensemble model correlations"
        )
    
    # Add latency warnings
    if not feature_time_ok:
        recommendation += f"\n⚠️ Feature gen time {latency.feature_gen_mean_ms:.0f}ms exceeds {config.target_feature_time_ms}ms target"
    if not memory_ok:
        recommendation += f"\n⚠️ Memory usage {latency.memory_peak_gb:.1f}GB exceeds {config.target_memory_gb}GB limit"
    
    return status, recommendation


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def save_outputs(
    result: ValidationResult,
    trades: List[Dict],
    equity_curve: List[float],
    output_dir: Path,
):
    """Save all validation outputs."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Performance report JSON
    report_path = output_dir / "performance_report.json"
    with open(report_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"Saved performance report: {report_path}")
    
    # 2. Trade log CSV
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_path = output_dir / "trade_log.csv"
        trades_df.to_csv(trades_path, index=False)
        logger.info(f"Saved trade log: {trades_path} ({len(trades)} trades)")
    
    # 3. Equity curve plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(equity_curve, linewidth=1.5, color='blue')
        ax.set_title('V2.5 Validation - Equity Curve', fontsize=14)
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=equity_curve[0], color='gray', linestyle='--', alpha=0.5)
        
        # Add metrics annotation
        metrics = result.performance
        annotation = (
            f"Sharpe: {metrics.sharpe_ratio:.2f}\n"
            f"Return: {metrics.total_return_pct:.1f}%\n"
            f"Max DD: {metrics.max_drawdown:.1%}\n"
            f"Win Rate: {metrics.win_rate:.1%}"
        )
        ax.annotate(annotation, xy=(0.02, 0.98), xycoords='axes fraction',
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        equity_path = output_dir / "equity_curve.png"
        plt.tight_layout()
        plt.savefig(equity_path, dpi=150)
        plt.close()
        logger.info(f"Saved equity curve: {equity_path}")
    except Exception as e:
        logger.warning(f"Failed to generate equity curve plot: {e}")
    
    # 4. Feature importance CSV
    if result.feature_importance:
        fi_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in sorted(result.feature_importance.items(), 
                              key=lambda x: x[1], reverse=True)[:20]
        ])
        fi_path = output_dir / "feature_importance_top20.csv"
        fi_df.to_csv(fi_path, index=False)
        logger.info(f"Saved feature importance: {fi_path}")
    
    # 5. Ensemble model weights JSON
    if result.ensemble_weights:
        weights_path = output_dir / "ensemble_model_weights.json"
        with open(weights_path, 'w') as f:
            json.dump(result.ensemble_weights, f, indent=2)
        logger.info(f"Saved ensemble weights: {weights_path}")


# =============================================================================
# MAIN VALIDATION RUNNER
# =============================================================================

def run_validation(config: Optional[ValidationConfig] = None) -> ValidationResult:
    """Run complete V2.5 production validation."""
    
    start_time = time.time()
    result = ValidationResult(timestamp=datetime.now().isoformat())
    
    if config is None:
        config = ValidationConfig()
    
    print("=" * 70)
    print("V2.5 PRODUCTION VALIDATION")
    print("=" * 70)
    print(f"Universe: {len(config.universe)} assets ({len(config.etfs)} ETFs, {len(config.stocks)} stocks)")
    print(f"Lookback: {config.lookback_days} trading days")
    print(f"Targets: Sharpe>{config.target_sharpe}, WinRate>{config.target_win_rate:.0%}, MaxDD<{config.target_max_dd:.0%}")
    print("=" * 70)
    
    # Step 1: Load V2.5 components
    print("\n📦 Step 1: Loading V2.5 Components...")
    components = V25ComponentLoader()
    available = components.load_all()
    
    if not any(available.values()):
        result.errors.append("No V2.5 components available")
        result.status = "failed"
        result.recommendation = "NO-GO - V2.5 components failed to load"
        return result
    
    print(f"   Components: {sum(available.values())}/{len(available)} loaded")
    
    # Step 2: Fetch market data
    print("\n📊 Step 2: Fetching Market Data...")
    fetcher = MarketDataFetcher()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(config.lookback_days * 1.5))  # Extra buffer for weekends
    
    data, data_quality = fetcher.fetch_universe(
        config.universe,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
    )
    
    result.data_quality = data_quality
    
    if data_quality['successful'] < len(config.universe) * 0.5:
        result.errors.append(f"Insufficient data: only {data_quality['successful']}/{len(config.universe)} tickers")
        result.status = "failed"
        result.recommendation = "NO-GO - Data quality issues"
        return result
    
    print(f"   Fetched: {data_quality['successful']}/{len(config.universe)} tickers")
    print(f"   Provider: {data_quality['provider']}")
    
    # Step 3: Run walk-forward backtest
    print("\n🔄 Step 3: Running Walk-Forward Backtest...")
    backtester = V25Backtester(config, components)
    
    try:
        metrics, trades = backtester.run_walkforward(data)
        result.performance = metrics
        result.trade_summary = {
            'total_trades': len(trades),
            'by_direction': {
                'long': len([t for t in trades if t['direction'] == 'long']),
                'short': len([t for t in trades if t['direction'] == 'short']),
            },
            'avg_holding_days': np.mean([t['holding_days'] for t in trades]) if trades else 0,
        }
        print(f"   Trades: {len(trades)}")
        print(f"   Sharpe: {metrics.sharpe_ratio:.2f}")
        print(f"   Win Rate: {metrics.win_rate:.1%}")
        print(f"   Max DD: {metrics.max_drawdown:.1%}")
    except Exception as e:
        result.errors.append(f"Backtest failed: {str(e)}")
        logger.error(f"Backtest error: {traceback.format_exc()}")
    
    # Step 4: Collect latency metrics
    print("\n⏱️ Step 4: Collecting Performance Metrics...")
    result.latency = backtester.get_latency_metrics()
    print(f"   Feature Gen: {result.latency.feature_gen_mean_ms:.0f}ms (avg)")
    print(f"   Ensemble Pred: {result.latency.ensemble_pred_mean_ms:.0f}ms (avg)")
    print(f"   Memory Peak: {result.latency.memory_peak_gb:.2f}GB")
    
    # Step 5: Get feature importance (if available)
    if components.ensemble and hasattr(components.ensemble, 'feature_importance_'):
        result.feature_importance = dict(zip(
            [f"feature_{i}" for i in range(len(components.ensemble.feature_importance_))],
            components.ensemble.feature_importance_.tolist()
        ))
    
    # Step 6: Get ensemble weights (if available)
    if components.ensemble and hasattr(components.ensemble, 'get_model_weights'):
        try:
            result.ensemble_weights = components.ensemble.get_model_weights()
        except:
            pass
    
    # Step 7: Determine recommendation
    print("\n📋 Step 5: Generating Recommendation...")
    result.status, result.recommendation = determine_recommendation(
        config, result.performance, result.latency
    )
    
    result.duration_seconds = round(time.time() - start_time, 2)
    
    # Print recommendation
    print("\n" + "=" * 70)
    print(f"STATUS: {result.status.upper()}")
    print("=" * 70)
    print(result.recommendation)
    print("=" * 70)
    
    # Save outputs
    output_dir = PROJECT_ROOT / "results" / "v25_validation"
    save_outputs(result, trades, backtester.equity_curve, output_dir)
    
    print(f"\n✅ Validation complete in {result.duration_seconds:.1f}s")
    print(f"📁 Results saved to: {output_dir}")
    
    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='V2.5 Production Validation')
    parser.add_argument('--lookback', type=int, default=504, help='Lookback days')
    parser.add_argument('--capital', type=float, default=1_000_000, help='Initial capital')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer assets')
    args = parser.parse_args()
    
    config = ValidationConfig(
        lookback_days=args.lookback,
        initial_capital=args.capital,
    )
    
    if args.quick:
        config.stocks = config.stocks[:5]  # Only first 5 stocks
    
    result = run_validation(config)
    
    # Exit with appropriate code
    sys.exit(0 if result.status in ['passed', 'conditional'] else 1)

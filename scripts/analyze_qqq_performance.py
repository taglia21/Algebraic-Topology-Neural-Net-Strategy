#!/usr/bin/env python3
"""QQQ Performance Deep Dive Analysis.

Analyzes QQQ underperformance across multiple dimensions:
1. Regime-level performance breakdown
2. Trade quality analysis  
3. Signal accuracy comparison
4. Structural factor analysis

Outputs comprehensive diagnostic report with actionable recommendations.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, '/workspaces/Algebraic-Topology-Neural-Net-Strategy')

from src.regime_detector import MarketRegimeDetector, Regime, VolatilityState, TradingCondition

RESULTS_DIR = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results'
TICKERS = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLK']


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download historical data for a ticker using yfinance."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end, auto_adjust=True)
    if not df.empty:
        df.columns = [c.lower() for c in df.columns]
    return df


def analyze_regime_performance(
    ticker: str,
    df: pd.DataFrame,
    regime_detector: MarketRegimeDetector
) -> Dict:
    """Analyze performance by market regime."""
    
    # Add regime classification to each day
    regimes = []
    volatilities = []
    trading_conditions = []
    
    close = df['close'] if 'close' in df.columns else df['Close']
    
    for i in range(len(df)):
        if i < 210:
            regimes.append('INSUFFICIENT_DATA')
            volatilities.append('INSUFFICIENT_DATA')
            trading_conditions.append('INSUFFICIENT_DATA')
            continue
        
        # Get data up to this point
        price_series = close.iloc[:i+1]
        sub_df = df.iloc[:i+1]
        
        result = regime_detector.detect_regime(price_series, df=sub_df)
        regimes.append(result.regime.value)
        volatilities.append(result.volatility.value)
        trading_conditions.append(result.trading_condition.value)
    
    df = df.copy()
    df['regime'] = regimes
    df['volatility'] = volatilities
    df['trading_condition'] = trading_conditions
    df['daily_return'] = close.pct_change()
    
    # Analyze by trading condition
    condition_stats = {}
    for condition in ['favorable', 'neutral', 'unfavorable', 'INSUFFICIENT_DATA']:
        mask = df['trading_condition'] == condition
        subset = df[mask]
        
        if len(subset) > 0:
            returns = subset['daily_return'].dropna()
            condition_stats[condition] = {
                'count': len(subset),
                'pct_of_days': len(subset) / len(df) * 100,
                'mean_return': returns.mean() * 100 if len(returns) > 0 else 0,
                'std_return': returns.std() * 100 if len(returns) > 0 else 0,
                'sharpe': (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0,
                'total_return': (1 + returns).prod() - 1 if len(returns) > 0 else 0
            }
    
    # Analyze by regime
    regime_stats = {}
    for regime in ['bull', 'bear', 'sideways', 'INSUFFICIENT_DATA']:
        mask = df['regime'] == regime
        subset = df[mask]
        
        if len(subset) > 0:
            returns = subset['daily_return'].dropna()
            regime_stats[regime] = {
                'count': len(subset),
                'pct_of_days': len(subset) / len(df) * 100,
                'mean_return': returns.mean() * 100 if len(returns) > 0 else 0,
                'sharpe': (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0,
            }
    
    return {
        'ticker': ticker,
        'total_days': len(df),
        'condition_stats': condition_stats,
        'regime_stats': regime_stats,
        'df_with_regimes': df
    }


def analyze_trade_quality(ticker: str) -> Dict:
    """Analyze trade-by-trade quality from trade journals."""
    
    journal_path = os.path.join(RESULTS_DIR, f'trade_journal_{ticker}.csv')
    
    if not os.path.exists(journal_path):
        return {'error': f'No trade journal found for {ticker}'}
    
    df = pd.read_csv(journal_path)
    
    if len(df) == 0:
        return {'error': f'Empty trade journal for {ticker}'}
    
    # Calculate trade metrics
    trades = []
    for _, row in df.iterrows():
        entry = row.get('entry_price', 0)
        exit_price = row.get('exit_price', 0)
        size = row.get('size', 0)
        
        if entry > 0 and exit_price > 0:
            pnl_pct = (exit_price - entry) / entry * 100
            trades.append({
                'entry_date': row.get('entry_date', ''),
                'exit_date': row.get('exit_date', ''),
                'entry_price': entry,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'size': size,
                'exit_reason': row.get('exit_reason', 'unknown')
            })
    
    if len(trades) == 0:
        return {'error': f'No valid trades for {ticker}'}
    
    trades_df = pd.DataFrame(trades)
    
    # Calculate metrics
    wins = trades_df[trades_df['pnl_pct'] > 0]
    losses = trades_df[trades_df['pnl_pct'] <= 0]
    
    return {
        'ticker': ticker,
        'total_trades': len(trades_df),
        'win_rate': len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        'avg_win_pct': wins['pnl_pct'].mean() if len(wins) > 0 else 0,
        'avg_loss_pct': losses['pnl_pct'].mean() if len(losses) > 0 else 0,
        'max_win_pct': wins['pnl_pct'].max() if len(wins) > 0 else 0,
        'max_loss_pct': losses['pnl_pct'].min() if len(losses) > 0 else 0,
        'profit_factor': abs(wins['pnl_pct'].sum() / losses['pnl_pct'].sum()) if len(losses) > 0 and losses['pnl_pct'].sum() != 0 else float('inf'),
        'expectancy': trades_df['pnl_pct'].mean(),
        'trades': trades
    }


def analyze_signal_quality(ticker: str) -> Dict:
    """Analyze signal diagnostics for a ticker."""
    
    diagnostics_path = os.path.join(RESULTS_DIR, f'diagnostics_{ticker}.csv')
    
    if not os.path.exists(diagnostics_path):
        return {'error': f'No diagnostics found for {ticker}'}
    
    df = pd.read_csv(diagnostics_path)
    
    if len(df) == 0:
        return {'error': f'Empty diagnostics for {ticker}'}
    
    # Analyze signal distribution
    signal_col = 'nn_signal' if 'nn_signal' in df.columns else None
    
    if signal_col is None:
        return {'error': f'No nn_signal column found for {ticker}'}
    
    signals = df[signal_col].dropna()
    
    # Buy/sell signal counts
    buy_signals = (signals > 0.52).sum()
    sell_signals = (signals < 0.48).sum()
    neutral_signals = len(signals) - buy_signals - sell_signals
    
    # Blocked reasons
    blocked_reasons = df['blocked_reason'].value_counts().to_dict() if 'blocked_reason' in df.columns else {}
    
    return {
        'ticker': ticker,
        'total_bars': len(df),
        'signal_mean': signals.mean(),
        'signal_std': signals.std(),
        'signal_min': signals.min(),
        'signal_max': signals.max(),
        'buy_signals': buy_signals,
        'buy_pct': buy_signals / len(signals) * 100,
        'sell_signals': sell_signals,
        'sell_pct': sell_signals / len(signals) * 100,
        'neutral_signals': neutral_signals,
        'neutral_pct': neutral_signals / len(signals) * 100,
        'blocked_reasons': blocked_reasons
    }


def analyze_volatility_structure(ticker: str, df: pd.DataFrame) -> Dict:
    """Analyze volatility patterns."""
    
    close = df['close'] if 'close' in df.columns else df['Close']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    
    # Daily returns volatility
    returns = close.pct_change().dropna()
    
    # ATR-based volatility
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = true_range.rolling(14).mean()
    atr_pct = (atr_14 / close * 100).dropna()
    
    # Volatility clustering
    rolling_vol = returns.rolling(20).std() * np.sqrt(252)
    
    return {
        'ticker': ticker,
        'annualized_vol': returns.std() * np.sqrt(252) * 100,
        'avg_atr_pct': atr_pct.mean(),
        'max_atr_pct': atr_pct.max(),
        'min_atr_pct': atr_pct.min(),
        'vol_of_vol': rolling_vol.std() / rolling_vol.mean() if rolling_vol.mean() > 0 else 0,
        'max_daily_move': abs(returns).max() * 100,
        'days_over_2pct': (abs(returns) > 0.02).sum(),
        'days_over_3pct': (abs(returns) > 0.03).sum(),
    }


def compare_correlations(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare return correlations between assets."""
    
    returns_dict = {}
    for ticker, df in data_dict.items():
        close = df['close'] if 'close' in df.columns else df['Close']
        returns_dict[ticker] = close.pct_change().dropna()
    
    # Align all returns
    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.dropna()
    
    correlation_matrix = returns_df.corr()
    
    return correlation_matrix


def generate_recommendation(
    regime_results: Dict,
    trade_results: Dict,
    signal_results: Dict,
    vol_results: Dict,
    correlations: pd.DataFrame
) -> Dict:
    """Generate actionable recommendation based on analysis."""
    
    recommendation = {
        'decision': 'UNDETERMINED',
        'confidence': 0,
        'rationale': [],
        'specific_actions': []
    }
    
    qqq_regime = regime_results.get('QQQ', {})
    spy_regime = regime_results.get('SPY', {})
    
    qqq_trades = trade_results.get('QQQ', {})
    spy_trades = trade_results.get('SPY', {})
    
    qqq_signals = signal_results.get('QQQ', {})
    
    qqq_vol = vol_results.get('QQQ', {})
    spy_vol = vol_results.get('SPY', {})
    
    issues_found = []
    positives_found = []
    
    # Check regime performance
    if qqq_regime and 'condition_stats' in qqq_regime:
        for condition, stats in qqq_regime['condition_stats'].items():
            if condition in ['favorable', 'neutral', 'unfavorable']:
                if stats.get('sharpe', 0) < 0:
                    issues_found.append(f"QQQ has negative Sharpe ({stats['sharpe']:.2f}) in {condition} conditions")
                elif stats.get('sharpe', 0) > 0.5:
                    positives_found.append(f"QQQ shows promise in {condition} conditions (Sharpe={stats['sharpe']:.2f})")
    
    # Check trade quality
    if qqq_trades and not qqq_trades.get('error'):
        win_rate = qqq_trades.get('win_rate', 0)
        spy_win_rate = spy_trades.get('win_rate', 0) if spy_trades else 0
        
        if win_rate < 50:
            issues_found.append(f"QQQ win rate ({win_rate:.1f}%) is below 50%")
        if spy_win_rate > 0 and win_rate < spy_win_rate - 10:
            issues_found.append(f"QQQ win rate ({win_rate:.1f}%) significantly lower than SPY ({spy_win_rate:.1f}%)")
        
        expectancy = qqq_trades.get('expectancy', 0)
        if expectancy < 0:
            issues_found.append(f"QQQ has negative expectancy ({expectancy:.2f}%)")
    
    # Check volatility structure
    if qqq_vol and spy_vol:
        qqq_ann_vol = qqq_vol.get('annualized_vol', 0)
        spy_ann_vol = spy_vol.get('annualized_vol', 0)
        
        if qqq_ann_vol > spy_ann_vol * 1.3:
            issues_found.append(f"QQQ volatility ({qqq_ann_vol:.1f}%) is >30% higher than SPY ({spy_ann_vol:.1f}%)")
        
        vol_of_vol = qqq_vol.get('vol_of_vol', 0)
        if vol_of_vol > 0.5:
            issues_found.append(f"QQQ has high volatility clustering (vol-of-vol={vol_of_vol:.2f})")
    
    # Check correlations
    if correlations is not None and 'QQQ' in correlations.columns and 'SPY' in correlations.columns:
        qqq_spy_corr = correlations.loc['QQQ', 'SPY']
        if qqq_spy_corr > 0.9:
            issues_found.append(f"QQQ highly correlated with SPY ({qqq_spy_corr:.2f}) - limited diversification benefit")
    
    # Make recommendation
    if len(issues_found) >= 4:
        recommendation['decision'] = 'REMOVE'
        recommendation['confidence'] = 0.8
        recommendation['rationale'] = issues_found
        recommendation['specific_actions'] = [
            "Remove QQQ from portfolio entirely",
            "Redistribute capital to SPY/IWM/XLF/XLK",
            "Consider DIA or MDY as replacement for tech exposure"
        ]
    elif len(issues_found) >= 2 and len(positives_found) >= 1:
        recommendation['decision'] = 'MODIFY'
        recommendation['confidence'] = 0.6
        recommendation['rationale'] = issues_found + positives_found
        recommendation['specific_actions'] = [
            "Trade QQQ only in FAVORABLE conditions",
            "Increase RSI filter stringency for QQQ (oversold=40, overbought=60)",
            "Reduce QQQ position size multiplier to 0.5x",
            "If still underperforms after 6 months, REMOVE"
        ]
    elif len(positives_found) >= 2:
        recommendation['decision'] = 'KEEP'
        recommendation['confidence'] = 0.7
        recommendation['rationale'] = positives_found
        recommendation['specific_actions'] = [
            "Continue current approach",
            "Monitor regime-specific performance",
            "Consider increasing QQQ allocation if Sharpe improves"
        ]
    else:
        recommendation['decision'] = 'TEST_REPLACEMENT'
        recommendation['confidence'] = 0.5
        recommendation['rationale'] = issues_found if issues_found else ["Inconclusive analysis"]
        recommendation['specific_actions'] = [
            "Run parallel test with DIA as QQQ replacement",
            "Test 4-asset portfolio (no QQQ)",
            "Compare 6-month forward performance"
        ]
    
    return recommendation


def generate_report(
    regime_results: Dict,
    trade_results: Dict,
    signal_results: Dict,
    vol_results: Dict,
    correlations: pd.DataFrame,
    recommendation: Dict
) -> str:
    """Generate comprehensive markdown report."""
    
    report = []
    report.append("=" * 80)
    report.append("QQQ PERFORMANCE DIAGNOSTIC REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    
    # Executive Summary
    report.append("\n" + "=" * 80)
    report.append("EXECUTIVE SUMMARY")
    report.append("=" * 80)
    report.append(f"\nRECOMMENDATION: {recommendation['decision']}")
    report.append(f"Confidence: {recommendation['confidence']*100:.0f}%")
    report.append("\nRationale:")
    for r in recommendation['rationale']:
        report.append(f"  • {r}")
    report.append("\nSpecific Actions:")
    for a in recommendation['specific_actions']:
        report.append(f"  → {a}")
    
    # Regime Analysis
    report.append("\n" + "=" * 80)
    report.append("SECTION 1: REGIME-LEVEL PERFORMANCE ANALYSIS")
    report.append("=" * 80)
    
    report.append("\n1.1 Trading Condition Breakdown (QQQ vs SPY vs IWM)")
    report.append("-" * 60)
    report.append(f"{'Condition':<15} {'Ticker':<8} {'Days':<8} {'%Days':<8} {'MeanRet%':<10} {'Sharpe':<8}")
    report.append("-" * 60)
    
    for condition in ['favorable', 'neutral', 'unfavorable']:
        for ticker in ['QQQ', 'SPY', 'IWM']:
            if ticker in regime_results and 'condition_stats' in regime_results[ticker]:
                stats = regime_results[ticker]['condition_stats'].get(condition, {})
                report.append(
                    f"{condition:<15} {ticker:<8} {stats.get('count', 0):<8} "
                    f"{stats.get('pct_of_days', 0):<8.1f} {stats.get('mean_return', 0):<10.4f} "
                    f"{stats.get('sharpe', 0):<8.2f}"
                )
        report.append("")
    
    report.append("\n1.2 Market Regime Breakdown")
    report.append("-" * 60)
    report.append(f"{'Regime':<12} {'Ticker':<8} {'Days':<8} {'%Days':<8} {'Sharpe':<8}")
    report.append("-" * 60)
    
    for regime in ['bull', 'bear', 'sideways']:
        for ticker in ['QQQ', 'SPY', 'IWM']:
            if ticker in regime_results and 'regime_stats' in regime_results[ticker]:
                stats = regime_results[ticker]['regime_stats'].get(regime, {})
                report.append(
                    f"{regime:<12} {ticker:<8} {stats.get('count', 0):<8} "
                    f"{stats.get('pct_of_days', 0):<8.1f} {stats.get('sharpe', 0):<8.2f}"
                )
        report.append("")
    
    # Trade Quality Analysis
    report.append("\n" + "=" * 80)
    report.append("SECTION 2: TRADE QUALITY ANALYSIS")
    report.append("=" * 80)
    
    report.append("\n2.1 Trade Metrics Comparison")
    report.append("-" * 60)
    report.append(f"{'Metric':<20} {'QQQ':<12} {'SPY':<12} {'IWM':<12}")
    report.append("-" * 60)
    
    metrics = ['total_trades', 'win_rate', 'avg_win_pct', 'avg_loss_pct', 'profit_factor', 'expectancy']
    metric_labels = ['Total Trades', 'Win Rate %', 'Avg Win %', 'Avg Loss %', 'Profit Factor', 'Expectancy %']
    
    for metric, label in zip(metrics, metric_labels):
        qqq_val = trade_results.get('QQQ', {}).get(metric, 'N/A')
        spy_val = trade_results.get('SPY', {}).get(metric, 'N/A')
        iwm_val = trade_results.get('IWM', {}).get(metric, 'N/A')
        
        if isinstance(qqq_val, float):
            qqq_str = f"{qqq_val:.2f}"
        else:
            qqq_str = str(qqq_val)
        if isinstance(spy_val, float):
            spy_str = f"{spy_val:.2f}"
        else:
            spy_str = str(spy_val)
        if isinstance(iwm_val, float):
            iwm_str = f"{iwm_val:.2f}"
        else:
            iwm_str = str(iwm_val)
        
        report.append(f"{label:<20} {qqq_str:<12} {spy_str:<12} {iwm_str:<12}")
    
    # Individual QQQ trades
    if 'QQQ' in trade_results and 'trades' in trade_results['QQQ']:
        report.append("\n2.2 QQQ Trade-by-Trade Analysis")
        report.append("-" * 60)
        trades = trade_results['QQQ']['trades']
        for i, trade in enumerate(trades, 1):
            pnl_pct = trade.get('pnl_pct', 0)
            status = "WIN" if pnl_pct > 0 else "LOSS"
            report.append(
                f"  Trade {i}: {trade.get('entry_date', 'N/A')} → {trade.get('exit_date', 'N/A')}, "
                f"Entry: ${trade.get('entry_price', 0):.2f}, Exit: ${trade.get('exit_price', 0):.2f}, "
                f"PnL: {pnl_pct:+.2f}% [{status}]"
            )
    
    # Signal Quality Analysis
    report.append("\n" + "=" * 80)
    report.append("SECTION 3: SIGNAL QUALITY ANALYSIS")
    report.append("=" * 80)
    
    report.append("\n3.1 NN Signal Distribution")
    report.append("-" * 60)
    report.append(f"{'Metric':<20} {'QQQ':<12} {'SPY':<12}")
    report.append("-" * 60)
    
    signal_metrics = ['signal_mean', 'signal_std', 'buy_pct', 'sell_pct', 'neutral_pct']
    signal_labels = ['Mean Signal', 'Std Signal', 'Buy Signals %', 'Sell Signals %', 'Neutral %']
    
    for metric, label in zip(signal_metrics, signal_labels):
        qqq_val = signal_results.get('QQQ', {}).get(metric, 'N/A')
        spy_val = signal_results.get('SPY', {}).get(metric, 'N/A')
        
        qqq_str = f"{qqq_val:.2f}" if isinstance(qqq_val, float) else str(qqq_val)
        spy_str = f"{spy_val:.2f}" if isinstance(spy_val, float) else str(spy_val)
        
        report.append(f"{label:<20} {qqq_str:<12} {spy_str:<12}")
    
    # Volatility Analysis
    report.append("\n" + "=" * 80)
    report.append("SECTION 4: VOLATILITY STRUCTURE ANALYSIS")
    report.append("=" * 80)
    
    report.append("\n4.1 Volatility Metrics")
    report.append("-" * 60)
    report.append(f"{'Metric':<25} {'QQQ':<12} {'SPY':<12} {'IWM':<12}")
    report.append("-" * 60)
    
    vol_metrics = ['annualized_vol', 'avg_atr_pct', 'vol_of_vol', 'max_daily_move', 'days_over_2pct']
    vol_labels = ['Annualized Vol %', 'Avg ATR %', 'Vol of Vol', 'Max Daily Move %', 'Days >2% Move']
    
    for metric, label in zip(vol_metrics, vol_labels):
        qqq_val = vol_results.get('QQQ', {}).get(metric, 'N/A')
        spy_val = vol_results.get('SPY', {}).get(metric, 'N/A')
        iwm_val = vol_results.get('IWM', {}).get(metric, 'N/A')
        
        qqq_str = f"{qqq_val:.2f}" if isinstance(qqq_val, float) else str(qqq_val)
        spy_str = f"{spy_val:.2f}" if isinstance(spy_val, float) else str(spy_val)
        iwm_str = f"{iwm_val:.2f}" if isinstance(iwm_val, float) else str(iwm_val)
        
        report.append(f"{label:<25} {qqq_str:<12} {spy_str:<12} {iwm_str:<12}")
    
    # Correlation Analysis
    report.append("\n" + "=" * 80)
    report.append("SECTION 5: CORRELATION ANALYSIS")
    report.append("=" * 80)
    
    if correlations is not None:
        report.append("\n5.1 Return Correlation Matrix")
        report.append("-" * 60)
        
        # Header
        header = f"{'':8}"
        for ticker in correlations.columns:
            header += f"{ticker:>8}"
        report.append(header)
        
        # Data rows
        for ticker in correlations.index:
            row = f"{ticker:8}"
            for col in correlations.columns:
                row += f"{correlations.loc[ticker, col]:>8.2f}"
            report.append(row)
        
        report.append("\nCorrelation Insights:")
        if 'QQQ' in correlations.columns and 'SPY' in correlations.columns:
            qqq_spy = correlations.loc['QQQ', 'SPY']
            report.append(f"  • QQQ-SPY correlation: {qqq_spy:.2f} (>0.9 = high overlap)")
        if 'QQQ' in correlations.columns and 'XLK' in correlations.columns:
            qqq_xlk = correlations.loc['QQQ', 'XLK']
            report.append(f"  • QQQ-XLK correlation: {qqq_xlk:.2f} (tech sector overlap)")
    
    # Conclusion
    report.append("\n" + "=" * 80)
    report.append("CONCLUSION & NEXT STEPS")
    report.append("=" * 80)
    report.append(f"\nFinal Recommendation: {recommendation['decision']}")
    report.append("\nNext Steps:")
    for i, action in enumerate(recommendation['specific_actions'], 1):
        report.append(f"  {i}. {action}")
    
    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    """Run comprehensive QQQ analysis."""
    print("=" * 60)
    print("QQQ PERFORMANCE DEEP DIVE ANALYSIS")
    print("=" * 60)
    
    # Initialize regime detector
    regime_detector = MarketRegimeDetector()
    
    # Download data for all tickers
    print("\n[1/6] Downloading historical data...")
    data_dict = {}
    for ticker in TICKERS:
        print(f"  Downloading {ticker}...")
        df = download_data(ticker, '2022-01-01', '2025-12-31')
        data_dict[ticker] = df
        print(f"    → {len(df)} bars")
    
    # Regime analysis
    print("\n[2/6] Analyzing regime performance...")
    regime_results = {}
    for ticker in TICKERS:
        print(f"  Analyzing {ticker}...")
        result = analyze_regime_performance(ticker, data_dict[ticker], regime_detector)
        regime_results[ticker] = result
    
    # Trade quality analysis
    print("\n[3/6] Analyzing trade quality...")
    trade_results = {}
    for ticker in TICKERS:
        result = analyze_trade_quality(ticker)
        trade_results[ticker] = result
        if 'error' not in result:
            print(f"  {ticker}: {result['total_trades']} trades, {result['win_rate']:.1f}% win rate")
        else:
            print(f"  {ticker}: {result['error']}")
    
    # Signal quality analysis
    print("\n[4/6] Analyzing signal quality...")
    signal_results = {}
    for ticker in TICKERS:
        result = analyze_signal_quality(ticker)
        signal_results[ticker] = result
        if 'error' not in result:
            print(f"  {ticker}: mean={result['signal_mean']:.3f}, buy_pct={result['buy_pct']:.1f}%")
        else:
            print(f"  {ticker}: {result['error']}")
    
    # Volatility analysis
    print("\n[5/6] Analyzing volatility structure...")
    vol_results = {}
    for ticker in TICKERS:
        result = analyze_volatility_structure(ticker, data_dict[ticker])
        vol_results[ticker] = result
        print(f"  {ticker}: ann_vol={result['annualized_vol']:.1f}%, vol_of_vol={result['vol_of_vol']:.2f}")
    
    # Correlation analysis
    print("\n[6/6] Analyzing correlations...")
    correlations = compare_correlations(data_dict)
    print(f"  QQQ-SPY correlation: {correlations.loc['QQQ', 'SPY']:.2f}")
    print(f"  QQQ-XLK correlation: {correlations.loc['QQQ', 'XLK']:.2f}")
    
    # Generate recommendation
    print("\nGenerating recommendation...")
    recommendation = generate_recommendation(
        regime_results, trade_results, signal_results, vol_results, correlations
    )
    print(f"  Decision: {recommendation['decision']} (confidence: {recommendation['confidence']*100:.0f}%)")
    
    # Generate report
    print("\nGenerating report...")
    report = generate_report(
        regime_results, trade_results, signal_results, vol_results, correlations, recommendation
    )
    
    # Save report
    report_path = os.path.join(RESULTS_DIR, 'qqq_diagnostic_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Report saved to: {report_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nRECOMMENDATION: {recommendation['decision']}")
    print(f"Confidence: {recommendation['confidence']*100:.0f}%")
    print("\nKey Findings:")
    for r in recommendation['rationale'][:3]:
        print(f"  • {r}")
    
    return recommendation


if __name__ == '__main__':
    main()

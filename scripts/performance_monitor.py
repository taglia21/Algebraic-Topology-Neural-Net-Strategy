#!/usr/bin/env python3
"""
Performance Monitoring Dashboard
=================================

Tracks and logs:
- Daily P&L
- Position changes
- Regime changes
- Trade history
- Performance metrics (Sharpe, drawdown, etc.)

Saves to JSON for dashboard display.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.alpaca_client import AlpacaClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Performance log file
PERF_LOG = Path('/opt/Algebraic-Topology-Neural-Net-Strategy/logs/performance_log.json')
DAILY_LOG = Path('/opt/Algebraic-Topology-Neural-Net-Strategy/logs/daily_metrics.json')


def load_performance_log():
    """Load existing performance log."""
    if PERF_LOG.exists():
        with open(PERF_LOG) as f:
            return json.load(f)
    return {
        'start_date': datetime.now().isoformat(),
        'starting_capital': 100000.0,
        'daily_snapshots': [],
        'trades': [],
        'regime_history': [],
    }


def save_performance_log(data):
    """Save performance log."""
    PERF_LOG.parent.mkdir(exist_ok=True)
    with open(PERF_LOG, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def calculate_metrics(snapshots):
    """Calculate performance metrics from daily snapshots."""
    if len(snapshots) < 2:
        return {}
    
    equities = [s['equity'] for s in snapshots]
    returns = np.diff(equities) / equities[:-1]
    
    # Total return
    total_return = (equities[-1] / equities[0] - 1) * 100
    
    # Annualized return (assuming 252 trading days)
    days = len(returns)
    if days > 0:
        ann_return = ((1 + total_return/100) ** (252/days) - 1) * 100
    else:
        ann_return = 0
    
    # Volatility
    if len(returns) > 1:
        daily_vol = np.std(returns)
        ann_vol = daily_vol * np.sqrt(252) * 100
    else:
        ann_vol = 0
    
    # Sharpe Ratio (assuming 5% risk-free rate)
    rf_daily = 0.05 / 252
    if ann_vol > 0:
        sharpe = (np.mean(returns) - rf_daily) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0
    
    # Max Drawdown
    peak = equities[0]
    max_dd = 0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd
    
    # Win rate (positive return days)
    if len(returns) > 0:
        win_rate = sum(1 for r in returns if r > 0) / len(returns) * 100
    else:
        win_rate = 0
    
    return {
        'total_return_pct': round(total_return, 2),
        'annualized_return_pct': round(ann_return, 2),
        'annualized_volatility_pct': round(ann_vol, 2),
        'sharpe_ratio': round(sharpe, 2),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'win_rate_pct': round(win_rate, 1),
        'trading_days': days,
    }


def get_current_snapshot(client):
    """Get current portfolio snapshot."""
    account = client.get_account()
    positions = client.get_positions()
    
    position_data = []
    for pos in positions:
        position_data.append({
            'symbol': pos.symbol,
            'qty': float(pos.qty),
            'market_value': float(pos.market_value),
            'unrealized_pl': float(pos.unrealized_pl),
            'unrealized_plpc': float(pos.unrealized_plpc) * 100,
        })
    
    # Sort by market value
    position_data.sort(key=lambda x: x['market_value'], reverse=True)
    
    return {
        'timestamp': datetime.now().isoformat(),
        'equity': float(account.equity),
        'cash': float(account.cash),
        'buying_power': float(account.buying_power),
        'position_count': len(positions),
        'positions': position_data,
        'total_unrealized_pl': sum(p['unrealized_pl'] for p in position_data),
    }


def print_dashboard(snapshot, metrics, log_data):
    """Print formatted dashboard."""
    print("\n" + "=" * 70)
    print("📊 TDA HEDGE FUND PERFORMANCE DASHBOARD")
    print("=" * 70)
    print(f"  📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  🏦 Started: {log_data['start_date'][:10]}")
    print("-" * 70)
    
    # Account Summary
    print("\n💰 ACCOUNT SUMMARY")
    print(f"  Equity:        ${snapshot['equity']:>12,.2f}")
    print(f"  Cash:          ${snapshot['cash']:>12,.2f}")
    print(f"  Invested:      ${snapshot['equity'] - snapshot['cash']:>12,.2f}")
    print(f"  Unrealized PL: ${snapshot['total_unrealized_pl']:>12,.2f}")
    
    # Performance Metrics
    if metrics:
        print("\n📈 PERFORMANCE METRICS")
        print(f"  Total Return:     {metrics.get('total_return_pct', 0):>8.2f}%")
        print(f"  Ann. Return:      {metrics.get('annualized_return_pct', 0):>8.2f}%")
        print(f"  Ann. Volatility:  {metrics.get('annualized_volatility_pct', 0):>8.2f}%")
        print(f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):>8.2f}")
        print(f"  Max Drawdown:     {metrics.get('max_drawdown_pct', 0):>8.2f}%")
        print(f"  Win Rate:         {metrics.get('win_rate_pct', 0):>8.1f}%")
        print(f"  Trading Days:     {metrics.get('trading_days', 0):>8d}")
    
    # Top Positions
    print(f"\n📋 POSITIONS ({snapshot['position_count']} total)")
    print(f"  {'Symbol':<8} {'Value':>12} {'P&L':>10} {'%':>8}")
    print("  " + "-" * 42)
    for pos in snapshot['positions'][:15]:  # Top 15
        pl_str = f"+${pos['unrealized_pl']:.2f}" if pos['unrealized_pl'] >= 0 else f"-${abs(pos['unrealized_pl']):.2f}"
        pct_str = f"+{pos['unrealized_plpc']:.1f}%" if pos['unrealized_plpc'] >= 0 else f"{pos['unrealized_plpc']:.1f}%"
        print(f"  {pos['symbol']:<8} ${pos['market_value']:>10,.2f} {pl_str:>10} {pct_str:>8}")
    
    if snapshot['position_count'] > 15:
        print(f"  ... and {snapshot['position_count'] - 15} more positions")
    
    print("\n" + "=" * 70)


def main():
    """Run performance monitoring."""
    # Initialize client
    client = AlpacaClient()
    
    # Load existing log
    log_data = load_performance_log()
    
    # Get current snapshot
    snapshot = get_current_snapshot(client)
    
    # Check if we should add a new daily snapshot
    today = datetime.now().date().isoformat()
    existing_dates = [s['timestamp'][:10] for s in log_data['daily_snapshots']]
    
    if today not in existing_dates:
        # Add new daily snapshot (simplified version)
        log_data['daily_snapshots'].append({
            'timestamp': snapshot['timestamp'],
            'equity': snapshot['equity'],
            'cash': snapshot['cash'],
            'position_count': snapshot['position_count'],
        })
        logger.info(f"Added daily snapshot for {today}")
    else:
        # Update today's snapshot
        for i, s in enumerate(log_data['daily_snapshots']):
            if s['timestamp'][:10] == today:
                log_data['daily_snapshots'][i] = {
                    'timestamp': snapshot['timestamp'],
                    'equity': snapshot['equity'],
                    'cash': snapshot['cash'],
                    'position_count': snapshot['position_count'],
                }
                break
    
    # Calculate metrics
    metrics = calculate_metrics(log_data['daily_snapshots'])
    
    # Print dashboard
    print_dashboard(snapshot, metrics, log_data)
    
    # Save updated log
    save_performance_log(log_data)
    
    # Also save latest metrics to separate file for easy access
    latest_metrics = {
        'timestamp': snapshot['timestamp'],
        'snapshot': snapshot,
        'metrics': metrics,
    }
    with open(DAILY_LOG, 'w') as f:
        json.dump(latest_metrics, f, indent=2, default=str)
    
    logger.info(f"Performance log saved to {PERF_LOG}")


if __name__ == '__main__':
    main()

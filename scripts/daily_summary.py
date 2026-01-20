#!/usr/bin/env python3
"""
Daily Performance Summary
=========================
Runs at market close to send end-of-day report via Discord.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set Discord webhook before importing notifications
os.environ['DISCORD_WEBHOOK'] = os.getenv('DISCORD_WEBHOOK', 
    'https://discord.com/api/webhooks/1463214027164352562/XwJ_IyLZPnXCoAJcEuk20PTKl92EOJhN4IESDwds0w1tdJUUIk-6w0LDQTVeVy7ymlIY')

from src.trading.alpaca_client import AlpacaClient
from src.trading.notifications import notify_daily_summary

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PERF_LOG = Path('/opt/Algebraic-Topology-Neural-Net-Strategy/logs/performance_log.json')


def get_previous_equity():
    """Get previous day's equity from log."""
    try:
        if PERF_LOG.exists():
            with open(PERF_LOG) as f:
                data = json.load(f)
            snapshots = data.get('daily_snapshots', [])
            if len(snapshots) >= 2:
                return snapshots[-2]['equity']
            elif len(snapshots) == 1:
                return data.get('starting_capital', 100000)
        return 100000
    except:
        return 100000


def main():
    """Generate and send daily summary."""
    logger.info("Generating daily performance summary...")
    
    # Initialize client
    client = AlpacaClient()
    
    # Get account info
    account = client.get_account()
    positions = client.get_positions()
    
    equity = float(account.equity)
    cash = float(account.cash)
    
    # Calculate daily P&L
    prev_equity = get_previous_equity()
    daily_pnl = equity - prev_equity
    daily_return_pct = (daily_pnl / prev_equity) * 100 if prev_equity > 0 else 0
    
    # Get top gainers and losers
    position_data = []
    for pos in positions:
        position_data.append({
            'symbol': pos.symbol,
            'pct': float(pos.unrealized_plpc) * 100,
            'pl': float(pos.unrealized_pl),
        })
    
    # Sort by percentage
    sorted_by_pct = sorted(position_data, key=lambda x: x['pct'], reverse=True)
    
    top_gainers = [(p['symbol'], p['pct']) for p in sorted_by_pct if p['pct'] > 0][:5]
    top_losers = [(p['symbol'], p['pct']) for p in sorted_by_pct if p['pct'] < 0][-5:][::-1]
    
    # Determine regime (simplified - just check from recent status)
    regime = "NEUTRAL"  # Default
    try:
        daily_log = Path('/opt/Algebraic-Topology-Neural-Net-Strategy/logs/daily_metrics.json')
        if daily_log.exists():
            with open(daily_log) as f:
                data = json.load(f)
                # Could extract regime from here if stored
    except:
        pass
    
    # Send notification
    notify_daily_summary(
        equity=equity,
        daily_pnl=daily_pnl,
        daily_return_pct=daily_return_pct,
        positions=len(positions),
        top_gainers=top_gainers,
        top_losers=top_losers,
        regime=regime,
        cash=cash
    )
    
    # Also update performance log
    if PERF_LOG.exists():
        with open(PERF_LOG) as f:
            data = json.load(f)
    else:
        data = {
            'start_date': datetime.now().isoformat(),
            'starting_capital': 100000.0,
            'daily_snapshots': [],
        }
    
    today = datetime.now().date().isoformat()
    existing_dates = [s['timestamp'][:10] for s in data['daily_snapshots']]
    
    snapshot = {
        'timestamp': datetime.now().isoformat(),
        'equity': equity,
        'cash': cash,
        'position_count': len(positions),
        'daily_pnl': daily_pnl,
        'daily_return_pct': daily_return_pct,
    }
    
    if today not in existing_dates:
        data['daily_snapshots'].append(snapshot)
    else:
        for i, s in enumerate(data['daily_snapshots']):
            if s['timestamp'][:10] == today:
                data['daily_snapshots'][i] = snapshot
                break
    
    with open(PERF_LOG, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Daily summary sent: Equity=${equity:,.2f}, P&L=${daily_pnl:,.2f} ({daily_return_pct:+.2f}%)")
    
    # Print summary
    print(f"\nDaily Summary for {today}")
    print("=" * 50)
    print(f"Portfolio Value:  ${equity:,.2f}")
    print(f"Daily P&L:        ${daily_pnl:,.2f} ({daily_return_pct:+.2f}%)")
    print(f"Positions:        {len(positions)}")
    print(f"Top Gainers:      {top_gainers}")
    print(f"Top Losers:       {top_losers}")


if __name__ == '__main__':
    main()

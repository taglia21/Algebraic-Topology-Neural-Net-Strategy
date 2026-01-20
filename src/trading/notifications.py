#!/usr/bin/env python3
"""
Trade Notification System
=========================
Professional notifications via Discord webhook
"""

import os
import json
import logging
from datetime import datetime
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

DISCORD_WEBHOOK = os.getenv('DISCORD_WEBHOOK', '')


def send_discord(title: str, message: str, color: int = 0x2F3136):
    """Send Discord notification via webhook."""
    if not DISCORD_WEBHOOK:
        logger.warning("Discord webhook not configured")
        return False
    
    payload = {
        "embeds": [{
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "footer": {"text": "TDA Quantitative Trading System"}
        }]
    }
    
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            DISCORD_WEBHOOK,
            data=data,
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'TDA-Trading-Bot/1.0',
            },
            method='POST'
        )
        response = urllib.request.urlopen(req, timeout=15)
        logger.info("Discord notification sent successfully")
        return True
    except urllib.error.HTTPError as e:
        logger.error(f"Discord HTTP error {e.code}: {e.reason}")
        # Try simple content format as fallback
        try:
            simple_payload = {"content": f"**{title}**\n{message[:1900]}"}
            data = json.dumps(simple_payload).encode('utf-8')
            req = urllib.request.Request(
                DISCORD_WEBHOOK,
                data=data,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'TDA-Trading-Bot/1.0',
                },
                method='POST'
            )
            urllib.request.urlopen(req, timeout=15)
            logger.info("Discord notification sent (simple format)")
            return True
        except Exception as e2:
            logger.error(f"Discord fallback also failed: {e2}")
            return False
    except Exception as e:
        logger.error(f"Discord notification failed: {e}")
        return False


def notify_trade_executed(symbol: str, side: str, qty: float, price: float, value: float):
    """Notify about a single trade execution."""
    title = f"Trade Executed: {side.upper()} {symbol}"
    message = (
        f"```\n"
        f"Symbol:      {symbol}\n"
        f"Action:      {side.upper()}\n"
        f"Quantity:    {qty:,.0f} shares\n"
        f"Price:       ${price:,.2f}\n"
        f"Value:       ${value:,.2f}\n"
        f"Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        f"```"
    )
    color = 0x00AA00 if side.lower() == 'buy' else 0xAA0000
    send_discord(title, message, color)


def notify_rebalance_summary(trades: list, regime: str, equity: float, cash: float, positions: int):
    """Notify about rebalance completion with trade summary."""
    if not trades:
        return
    
    buys = [t for t in trades if t.get('side', '').lower() == 'buy']
    sells = [t for t in trades if t.get('side', '').lower() == 'sell']
    
    buy_value = sum(t.get('value', 0) for t in buys)
    sell_value = sum(t.get('value', 0) for t in sells)
    
    title = "Portfolio Rebalance Complete"
    
    trade_lines = []
    for t in trades[:15]:
        side = t.get('side', 'N/A').upper()
        symbol = t.get('symbol', 'N/A')
        qty = t.get('qty', 0)
        value = t.get('value', 0)
        trade_lines.append(f"  {side:<4} {symbol:<6} {qty:>6,.0f} shares  ${value:>10,.2f}")
    
    if len(trades) > 15:
        trade_lines.append(f"  ... and {len(trades) - 15} additional trades")
    
    trades_str = "\n".join(trade_lines) if trade_lines else "  None"
    
    message = (
        f"```\n"
        f"REBALANCE SUMMARY\n"
        f"{'='*50}\n"
        f"Date:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        f"Market Regime:  {regime}\n"
        f"\n"
        f"PORTFOLIO STATUS\n"
        f"{'-'*50}\n"
        f"Total Equity:   ${equity:>14,.2f}\n"
        f"Cash:           ${cash:>14,.2f} ({cash/equity*100:.1f}%)\n"
        f"Invested:       ${equity-cash:>14,.2f} ({(equity-cash)/equity*100:.1f}%)\n"
        f"Positions:      {positions:>14,d}\n"
        f"\n"
        f"TRADE ACTIVITY\n"
        f"{'-'*50}\n"
        f"Total Trades:   {len(trades):>14,d}\n"
        f"Buy Orders:     {len(buys):>14,d}  (${buy_value:>12,.2f})\n"
        f"Sell Orders:    {len(sells):>14,d}  (${sell_value:>12,.2f})\n"
        f"\n"
        f"EXECUTED TRADES\n"
        f"{'-'*50}\n"
        f"{trades_str}\n"
        f"```"
    )
    
    send_discord(title, message, 0x2F3136)


def notify_regime_change(old_regime: str, new_regime: str, turbulence: float, vix: float):
    """Notify about market regime change."""
    title = f"Market Regime Change: {old_regime} to {new_regime}"
    message = (
        f"```\n"
        f"REGIME TRANSITION ALERT\n"
        f"{'='*50}\n"
        f"Previous Regime:  {old_regime}\n"
        f"New Regime:       {new_regime}\n"
        f"\n"
        f"MARKET INDICATORS\n"
        f"{'-'*50}\n"
        f"Turbulence Index: {turbulence:>10.2f}\n"
        f"VIX Level:        {vix:>10.2f}\n"
        f"\n"
        f"Action: Portfolio allocation will be adjusted\n"
        f"```"
    )
    send_discord(title, message, 0xFFAA00)


def notify_daily_summary(equity: float, daily_pnl: float, daily_return_pct: float, 
                         positions: int, top_gainers: list, top_losers: list,
                         regime: str, cash: float):
    """Send end-of-day performance summary."""
    
    gainers_str = "\n".join([f"  {s:<6} {r:>+7.2f}%" for s, r in top_gainers[:5]]) or "  None"
    losers_str = "\n".join([f"  {s:<6} {r:>+7.2f}%" for s, r in top_losers[:5]]) or "  None"
    
    title = f"Daily Performance Report - {datetime.now().strftime('%Y-%m-%d')}"
    message = (
        f"```\n"
        f"END OF DAY SUMMARY\n"
        f"{'='*50}\n"
        f"\n"
        f"PORTFOLIO PERFORMANCE\n"
        f"{'-'*50}\n"
        f"Portfolio Value:  ${equity:>14,.2f}\n"
        f"Daily P&L:        ${daily_pnl:>14,.2f}\n"
        f"Daily Return:     {daily_return_pct:>14.2f}%\n"
        f"\n"
        f"ALLOCATION\n"
        f"{'-'*50}\n"
        f"Cash:             ${cash:>14,.2f} ({cash/equity*100:.1f}%)\n"
        f"Invested:         ${equity-cash:>14,.2f} ({(equity-cash)/equity*100:.1f}%)\n"
        f"Positions:        {positions:>14,d}\n"
        f"Market Regime:    {regime:>14s}\n"
        f"\n"
        f"TOP GAINERS\n"
        f"{'-'*50}\n"
        f"{gainers_str}\n"
        f"\n"
        f"TOP LOSERS\n"
        f"{'-'*50}\n"
        f"{losers_str}\n"
        f"```"
    )
    
    color = 0x00AA00 if daily_pnl >= 0 else 0xAA0000
    send_discord(title, message, color)


def notify_error(error_type: str, error_message: str):
    """Notify about system errors."""
    title = f"System Alert: {error_type}"
    message = (
        f"```\n"
        f"ERROR NOTIFICATION\n"
        f"{'='*50}\n"
        f"Type:    {error_type}\n"
        f"Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        f"\n"
        f"Details:\n"
        f"{error_message}\n"
        f"```"
    )
    send_discord(title, message, 0xFF0000)


if __name__ == '__main__':
    import sys
    os.environ['DISCORD_WEBHOOK'] = 'https://discord.com/api/webhooks/1463214027164352562/XwJ_IyLZPnXCoAJcEuk20PTKl92EOJhN4IESDwds0w1tdJUUIk-6w0LDQTVeVy7ymlIY'
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        send_discord("System Status", "Trading system operational. All services running.", 0x00AA00)
        print("Test notification sent")

#!/usr/bin/env python3
"""
Trade Notification System
=========================
Sends notifications via Discord webhook, Slack, or email
"""

import os
import json
import logging
from datetime import datetime
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# Notification settings - set these environment variables
DISCORD_WEBHOOK = os.getenv('DISCORD_WEBHOOK', '')
SLACK_WEBHOOK = os.getenv('SLACK_WEBHOOK', '')


def send_discord(title: str, message: str, color: int = 0x00d4ff):
    """Send Discord notification via webhook."""
    if not DISCORD_WEBHOOK:
        return False
    
    payload = {
        "embeds": [{
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "TDA Hedge Fund Bot"}
        }]
    }
    
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            DISCORD_WEBHOOK,
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        urllib.request.urlopen(req, timeout=10)
        logger.info("Discord notification sent")
        return True
    except Exception as e:
        logger.error(f"Discord notification failed: {e}")
        return False


def send_slack(title: str, message: str):
    """Send Slack notification via webhook."""
    if not SLACK_WEBHOOK:
        return False
    
    payload = {
        "text": f"*{title}*\n{message}",
        "username": "TDA Hedge Fund Bot",
        "icon_emoji": ":chart_with_upwards_trend:"
    }
    
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            SLACK_WEBHOOK,
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        urllib.request.urlopen(req, timeout=10)
        logger.info("Slack notification sent")
        return True
    except Exception as e:
        logger.error(f"Slack notification failed: {e}")
        return False


def notify_trade(symbol: str, side: str, qty: float, price: float, order_type: str = 'market'):
    """Notify about a trade execution."""
    color = 0x00ff88 if side == 'buy' else 0xff4444
    emoji = "🟢" if side == 'buy' else "🔴"
    
    title = f"{emoji} {side.upper()} {symbol}"
    message = f"**Qty:** {qty}\n**Price:** ${price:.2f}\n**Value:** ${qty * price:,.2f}\n**Type:** {order_type}"
    
    send_discord(title, message, color)
    send_slack(title, message)


def notify_rebalance(trades_executed: int, regime: str, total_value: float, cash: float):
    """Notify about rebalance completion."""
    title = "📊 Rebalance Complete"
    message = (
        f"**Trades Executed:** {trades_executed}\n"
        f"**Current Regime:** {regime}\n"
        f"**Portfolio Value:** ${total_value:,.2f}\n"
        f"**Cash:** ${cash:,.2f} ({cash/total_value*100:.1f}%)\n"
        f"**Invested:** ${total_value-cash:,.2f} ({(total_value-cash)/total_value*100:.1f}%)"
    )
    
    send_discord(title, message, 0x00d4ff)
    send_slack(title, message)


def notify_regime_change(old_regime: str, new_regime: str, turbulence: float, vix: float):
    """Notify about regime change."""
    colors = {
        'RISK_ON': 0x00ff88,
        'BULL': 0x88ff00,
        'NEUTRAL': 0xffff00,
        'BEAR': 0xff8800,
        'RISK_OFF': 0xff4444,
    }
    
    title = f"⚠️ Regime Change: {old_regime} → {new_regime}"
    message = (
        f"**Turbulence:** {turbulence:.1f}\n"
        f"**VIX:** {vix:.1f}\n"
        f"**Action:** Portfolio will be rebalanced accordingly"
    )
    
    send_discord(title, message, colors.get(new_regime, 0x888888))
    send_slack(title, message)


def notify_error(error_type: str, error_message: str):
    """Notify about errors."""
    title = f"❌ Error: {error_type}"
    
    send_discord(title, error_message, 0xff0000)
    send_slack(title, error_message)


def notify_daily_summary(equity: float, daily_return: float, positions: int, top_gainers: list, top_losers: list):
    """Send daily performance summary."""
    emoji = "📈" if daily_return >= 0 else "📉"
    color = 0x00ff88 if daily_return >= 0 else 0xff4444
    
    gainers_str = "\n".join([f"  {s}: +{r:.1f}%" for s, r in top_gainers[:3]]) or "None"
    losers_str = "\n".join([f"  {s}: {r:.1f}%" for s, r in top_losers[:3]]) or "None"
    
    title = f"{emoji} Daily Summary"
    message = (
        f"**Portfolio Value:** ${equity:,.2f}\n"
        f"**Daily Return:** {daily_return:+.2f}%\n"
        f"**Positions:** {positions}\n\n"
        f"**Top Gainers:**\n{gainers_str}\n\n"
        f"**Top Losers:**\n{losers_str}"
    )
    
    send_discord(title, message, color)
    send_slack(title, message)


# Test function
def test_notifications():
    """Test notification system."""
    print("Testing notifications...")
    
    if DISCORD_WEBHOOK:
        print(f"Discord webhook configured: {DISCORD_WEBHOOK[:50]}...")
        send_discord("🧪 Test Notification", "TDA Hedge Fund Bot is connected and ready!", 0x00ff88)
    else:
        print("Discord webhook not configured (set DISCORD_WEBHOOK env var)")
    
    if SLACK_WEBHOOK:
        print(f"Slack webhook configured: {SLACK_WEBHOOK[:50]}...")
        send_slack("🧪 Test Notification", "TDA Hedge Fund Bot is connected and ready!")
    else:
        print("Slack webhook not configured (set SLACK_WEBHOOK env var)")


if __name__ == '__main__':
    test_notifications()

#!/usr/bin/env python3
"""
Discord Integration for Team of Rivals Trading System
Posts daily standup meetings and alerts to Discord.
"""

import os
import json
import requests
from datetime import datetime, time
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import pytz

load_dotenv()

class DiscordMeetingBot:
    """
    Posts Team of Rivals meeting reports to Discord.
    Uses webhooks for simple integration.
    """
    
    def __init__(self):
        self.webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        self.timezone = pytz.timezone("America/New_York")  # EST
        
        if not self.webhook_url:
            print("[!] DISCORD_WEBHOOK_URL not set in .env")
            print("    To set up Discord integration:")
            print("    1. Go to your Discord server settings")
            print("    2. Click Integrations -> Webhooks -> New Webhook")
            print("    3. Copy the webhook URL")
            print("    4. Add to .env: DISCORD_WEBHOOK_URL=your_url_here")
    
    def format_standup_report(self, reports: Dict[str, Any], alerts: list) -> Dict:
        """Format morning standup as Discord embed"""
        now = datetime.now(self.timezone)
        
        # Build status fields
        fields = []
        for dept, report in reports.items():
            status = report.get("status", "UNKNOWN")
            emoji = "🟢" if status == "GREEN" else "🟡" if status == "YELLOW" else "🔴"
            fields.append({
                "name": f"{emoji} {dept.upper()}",
                "value": f"Status: {status}",
                "inline": True
            })
        
        # Build embed
        embed = {
            "title": "🌅 MORNING STANDUP",
            "description": f"**Date:** {now.strftime('%A, %B %d, %Y')}\n**Time:** {now.strftime('%I:%M %p')} EST",
            "color": 0x00ff00 if not alerts else 0xffff00,  # Green or yellow
            "fields": fields,
            "footer": {"text": "Team of Rivals Trading System"}
        }
        
        # Add alerts if any
        if alerts:
            embed["fields"].append({
                "name": "⚠️ ALERTS",
                "value": "\n".join([f"• {alert}" for alert in alerts]),
                "inline": False
            })
        else:
            embed["fields"].append({
                "name": "✅ STATUS",
                "value": "No critical alerts",
                "inline": False
            })
        
        return {"embeds": [embed]}
    
    def format_trade_veto(self, trade_info: Dict, veto_reasons: list) -> Dict:
        """Format trade veto alert"""
        embed = {
            "title": "🚫 TRADE VETOED",
            "color": 0xff0000,  # Red
            "fields": [
                {"name": "Symbol", "value": trade_info.get("symbol", "N/A"), "inline": True},
                {"name": "Side", "value": trade_info.get("side", "N/A").upper(), "inline": True},
                {"name": "Strategy", "value": trade_info.get("strategy", "N/A"), "inline": True},
                {"name": "Position %", "value": f"{trade_info.get('position_pct', 0):.1%}", "inline": True},
                {"name": "Vetoed By", "value": "Risk Team", "inline": True},
                {"name": "Reasons", "value": "\n".join([f"• {r}" for r in veto_reasons]), "inline": False}
            ],
            "footer": {"text": f"Vetoed at {datetime.now(self.timezone).strftime('%I:%M %p')} EST"}
        }
        return {"embeds": [embed]}
    
    def format_trade_executed(self, execution: Dict) -> Dict:
        """Format successful trade execution"""
        embed = {
            "title": "✅ TRADE EXECUTED",
            "color": 0x00ff00,  # Green
            "fields": [
                {"name": "Symbol", "value": execution.get("symbol", "N/A"), "inline": True},
                {"name": "Side", "value": execution.get("side", "N/A").upper(), "inline": True},
                {"name": "Quantity", "value": str(execution.get("quantity", "N/A")), "inline": True},
                {"name": "Status", "value": execution.get("status", "N/A"), "inline": True}
            ],
            "footer": {"text": f"Executed at {datetime.now(self.timezone).strftime('%I:%M %p')} EST"}
        }
        return {"embeds": [embed]}
    
    def format_eod_report(self, trading_log: Dict, vetoes: int) -> Dict:
        """Format end-of-day report"""
        pnl = trading_log.get("pnl", 0)
        trades = len(trading_log.get("trades", []))
        
        pnl_emoji = "📈" if pnl >= 0 else "📉"
        pnl_color = 0x00ff00 if pnl >= 0 else 0xff0000
        
        embed = {
            "title": "🌆 END OF DAY REVIEW",
            "color": pnl_color,
            "fields": [
                {"name": f"{pnl_emoji} P&L", "value": f"${pnl:,.2f}", "inline": True},
                {"name": "📊 Trades", "value": str(trades), "inline": True},
                {"name": "🚫 Vetoes", "value": str(vetoes), "inline": True}
            ],
            "footer": {"text": f"Report generated at {datetime.now(self.timezone).strftime('%I:%M %p')} EST"}
        }
        return {"embeds": [embed]}
    
    def send_message(self, payload: Dict) -> bool:
        """Send message to Discord webhook"""
        if not self.webhook_url:
            print("[!] Cannot send - DISCORD_WEBHOOK_URL not configured")
            return False
        
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 204:
                print("[OK] Message sent to Discord")
                return True
            else:
                print(f"[!] Discord error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"[!] Failed to send Discord message: {e}")
            return False
    
    def send_standup(self, reports: Dict, alerts: list):
        """Send morning standup to Discord"""
        payload = self.format_standup_report(reports, alerts)
        return self.send_message(payload)
    
    def send_veto_alert(self, trade_info: Dict, veto_reasons: list):
        """Send trade veto alert to Discord"""
        payload = self.format_trade_veto(trade_info, veto_reasons)
        return self.send_message(payload)
    
    def send_trade_executed(self, execution: Dict):
        """Send trade execution notification"""
        payload = self.format_trade_executed(execution)
        return self.send_message(payload)
    
    def send_eod_report(self, trading_log: Dict, vetoes: int):
        """Send end-of-day report to Discord"""
        payload = self.format_eod_report(trading_log, vetoes)
        return self.send_message(payload)


if __name__ == "__main__":
    bot = DiscordMeetingBot()
    print("Discord Meeting Bot initialized")
    
    # Test message (only works if webhook is configured)
    if bot.webhook_url:
        test_reports = {
            "strategy": {"status": "GREEN"},
            "data": {"status": "GREEN"},
            "risk": {"status": "GREEN"},
            "execution": {"status": "GREEN"},
            "research": {"status": "GREEN"}
        }
        bot.send_standup(test_reports, [])

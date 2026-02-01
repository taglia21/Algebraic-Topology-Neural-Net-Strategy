#!/usr/bin/env python3
"""
Discord Integration for Team of Rivals Trading Bot
Posts daily standup meetings and trade alerts to Discord
Uses webhooks for simple integration without bot token
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import pytz

load_dotenv()

class DiscordMeetingBot:
    """
    Posts Team of Rivals meeting updates to Discord
    Uses webhooks for simple integration
    Each agent has their own webhook and persona
    """

    # Agent personas with unique characteristics
    AGENTS = {
        "marcus_chen": {
            "name": "Marcus Chen - Chief Orchestrator",
            "webhook_env": "DISCORD_WEBHOOK_MARCUS",
            "role": "Chief Orchestrator",
            "avatar_url": "https://i.imgur.com/4M34hi2.png",
            "personality": "Strategic, decisive, results-oriented",
            "specialty": "Overall coordination and final decisions"
        },
        "victoria_sterling": {
            "name": "Victoria Sterling - Chief Risk Officer",
            "webhook_env": "DISCORD_WEBHOOK_VICTORIA",
            "role": "Chief Risk Officer",
            "avatar_url": "https://i.imgur.com/RyVqvp9.png",
            "personality": "Cautious, analytical, protective",
            "specialty": "Risk management and position sizing"
        },
        "james_thornton": {
            "name": "James Thornton - Strategy Team Lead",
            "webhook_env": "DISCORD_WEBHOOK_JAMES",
            "role": "Strategy Team Lead",
            "avatar_url": "https://i.imgur.com/3wU9h8l.png",
            "personality": "Creative, adaptive, data-driven",
            "specialty": "Strategy development and optimization"
        },
        "elena_rodriguez": {
            "name": "Elena Rodriguez - Data Team Lead",
            "webhook_env": "DISCORD_WEBHOOK_ELENA",
            "role": "Data Team Lead",
            "avatar_url": "https://i.imgur.com/7JzKZ2x.png",
            "personality": "Meticulous, thorough, insight-driven",
            "specialty": "Data quality and market analysis"
        },
        "derek_washington": {
            "name": "Derek Washington - Execution Team Lead",
            "webhook_env": "DISCORD_WEBHOOK_DEREK",
            "role": "Execution Team Lead",
            "avatar_url": "https://i.imgur.com/WJz4x2y.png",
            "personality": "Precise, efficient, detail-oriented",
            "specialty": "Trade execution and operational excellence"
        },
        "sophia_nakamura": {
            "name": "Dr. Sophia Nakamura - Research Team Lead",
            "webhook_env": "DISCORD_WEBHOOK_SOPHIA",
            "role": "Research Team Lead",
            "avatar_url": "https://i.imgur.com/9Hx3k2y.png",
            "personality": "Innovative, rigorous, theory-driven",
            "specialty": "Research and model development"
        }
    }

    def __init__(self):
        # Load webhook URLs from environment
        self.webhooks = {}
        for agent_id, agent_info in self.AGENTS.items():
            webhook_url = os.getenv(agent_info["webhook_env"])
            if webhook_url:
                self.webhooks[agent_id] = webhook_url
            else:
                print(f"Warning: No webhook URL found for {agent_info['name']}")

    def send_message(self, agent_id: str, content: str, embed: Optional[Dict] = None) -> bool:
        """
        Send a message from a specific agent to Discord
        
        Args:
            agent_id: Agent identifier (e.g., 'marcus_chen')
            content: Message text
            embed: Optional embed object for rich formatting
        
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.webhooks:
            print(f"Error: No webhook configured for agent {agent_id}")
            return False

        agent = self.AGENTS[agent_id]
        webhook_url = self.webhooks[agent_id]

        payload = {
            "username": agent["name"],
            "content": content
        }

        if embed:
            payload["embeds"] = [embed]

        try:
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error sending message from {agent['name']}: {e}")
            return False

    def send_standup(self, reports: Dict[str, Any], meeting_notes: Optional[List[str]] = None) -> bool:
        """
        Send daily standup meeting to #morning-standup channel
        
        Args:
            reports: Dictionary of team reports
            meeting_notes: Optional list of meeting discussion points
        """
        # Marcus Chen opens the meeting
        marcus_msg = f"""Good morning team! Let's start our daily standup for {datetime.now().strftime('%B %d, %Y')}.

Each team lead, please share your updates."""
        self.send_message("marcus_chen", marcus_msg)

        # Each team reports
        team_agents = [
            ("victoria_sterling", "Risk Team"),
            ("james_thornton", "Strategy Team"),
            ("elena_rodriguez", "Data Team"),
            ("derek_washington", "Execution Team"),
            ("sophia_nakamura", "Research Team")
        ]

        for agent_id, team_name in team_agents:
            if team_name.lower().replace(" ", "_") in reports:
                report = reports[team_name.lower().replace(" ", "_")]
                self.send_message(agent_id, report)

        # Meeting notes if any
        if meeting_notes:
            notes_text = "\n".join([f"• {note}" for note in meeting_notes])
            self.send_message("marcus_chen", f"Key discussion points:\n{notes_text}")

        return True

    def send_trade_alert(self, trade_info: Dict[str, Any]) -> bool:
        """
        Send trade alert to #trade-alerts channel
        Elena Rodriguez posts data-driven trade signals
        """
        alert_msg = f"""**TRADE SIGNAL DETECTED**

Symbol: {trade_info.get('symbol', 'N/A')}
Action: {trade_info.get('action', 'N/A')}
Confidence: {trade_info.get('confidence', 0):.1%}
Signal Strength: {trade_info.get('signal_strength', 'N/A')}

Analysis: {trade_info.get('analysis', 'No analysis provided')}"""
        
        return self.send_message("elena_rodriguez", alert_msg)

    def send_risk_veto(self, veto_info: Dict[str, Any]) -> bool:
        """
        Send risk veto to #risk-vetoes channel
        Victoria Sterling posts risk concerns
        """
        veto_msg = f"""🛑 **RISK VETO**

Trade: {veto_info.get('symbol', 'N/A')} - {veto_info.get('action', 'N/A')}
Reason: {veto_info.get('reason', 'Risk threshold exceeded')}
Position Size: {veto_info.get('position_size', 'N/A')}
Max Allowed: {veto_info.get('max_allowed', 'N/A')}

This trade has been blocked for risk management purposes."""
        
        return self.send_message("victoria_sterling", veto_msg)

    def send_eod_review(self, performance: Dict[str, Any]) -> bool:
        """
        Send end-of-day review to #eod-review channel
        Derek Washington posts execution summary
        """
        eod_msg = f"""**END OF DAY REVIEW**

Date: {datetime.now().strftime('%B %d, %Y')}
Trades Executed: {performance.get('trades_executed', 0)}
P&L: ${performance.get('pnl', 0):,.2f}
Win Rate: {performance.get('win_rate', 0):.1%}
Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}

Execution Notes: {performance.get('notes', 'No issues to report')}"""
        
        return self.send_message("derek_washington", eod_msg)

if __name__ == "__main__":
    # Test the Discord integration
    bot = DiscordMeetingBot()
    
    # Test standup
    test_reports = {
        "risk_team": "Risk status GREEN. All positions within limits. Monitoring market volatility.",
        "strategy_team": "Testing new mean reversion strategy. Backtests showing promise.",
        "data_team": "Data quality check complete. All feeds operational.",
        "execution_team": "Ready for today's trading. No execution issues yesterday.",
        "research_team": "Working on TDA enhancement. Preliminary results positive."
    }
    
    bot.send_standup(test_reports, [])

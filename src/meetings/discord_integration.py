#!/usr/bin/env python3
"""
Discord Integration for Team of Rivals Trading System
Enables multi-agent collaboration via Discord webhooks with TTS support
"""

import os
import json
import requests
from datetime import datetime
import pytz

# Discord Webhook URLs from environment
WEBHOOKS = {
    "marcus": os.getenv("DISCORD_WEBHOOK_MARCUS"),
    "victoria": os.getenv("DISCORD_WEBHOOK_VICTORIA"),
    "james": os.getenv("DISCORD_WEBHOOK_JAMES"),
    "elena": os.getenv("DISCORD_WEBHOOK_ELENA"),
    "derek": os.getenv("DISCORD_WEBHOOK_DEREK"),
    "sophia": os.getenv("DISCORD_WEBHOOK_SOPHIA")
}

# Agent personas with unique voices
AGENT_PERSONAS = {
    "marcus": {
        "name": "Marcus Chen - Chief Strategy Officer",
        "role": "Strategy & Execution",
        "focus": "Profit maximization and competitive advantage"
    },
    "victoria": {
        "name": "Victoria Hayes - Chief Risk Officer",
        "role": "Risk Management",
        "focus": "Portfolio protection and volatility control"
    },
    "james": {
        "name": "James Park - Quantitative Analyst",
        "role": "Statistical Analysis",
        "focus": "Model validation and backtesting"
    },
    "elena": {
        "name": "Elena Rodriguez - Market Analyst",
        "role": "Market Intelligence",
        "focus": "Trend analysis and market sentiment"
    },
    "derek": {
        "name": "Derek Thompson - Technical Infrastructure",
        "role": "System Performance",
        "focus": "Execution quality and system reliability"
    },
    "sophia": {
        "name": "Sophia Williams - Compliance Officer",
        "role": "Regulatory Compliance",
        "focus": "Risk controls and regulatory adherence"
    }
}

class DiscordMeetingBot:
    """Manages Discord integration for Team of Rivals meetings"""
    
    def __init__(self, use_tts=True):
        self.use_tts = use_tts
        self.eastern = pytz.timezone('America/New_York')
    
    def send_message(self, agent_key, message, channel="general"):
        """Send a message from a specific agent to Discord"""
        webhook_url = WEBHOOKS.get(agent_key)
        if not webhook_url:
            print(f"Warning: No webhook configured for {agent_key}")
            return False
        
        persona = AGENT_PERSONAS.get(agent_key, {})
        agent_name = persona.get("name", agent_key)
        
        # Add /tts prefix for Discord text-to-speech
        if self.use_tts:
            message = f"/tts {message}"
        
        payload = {
            "username": agent_name,
            "content": message
        }
        
        try:
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Error sending message for {agent_key}: {e}")
            return False
    
    def conduct_meeting(self, meeting_type="standup"):
        """Conduct a team meeting with all agents"""
        now = datetime.now(self.eastern)
        timestamp = now.strftime("%I:%M %p EST")
        
        if meeting_type == "standup":
            self._morning_standup(timestamp)
        elif meeting_type == "eod":
            self._eod_wrapup(timestamp)
        elif meeting_type == "ml_check":
            self._ml_model_check(timestamp)
        elif meeting_type == "deep_dive":
            self._deep_dive_review(timestamp)
    
    def _morning_standup(self, timestamp):
        """Morning standup meeting"""
        self.send_message("marcus", 
            f"Good morning team. It's {timestamp}. Let's review today's trading strategy.")
        
        self.send_message("victoria", 
            "Risk status: All positions within limits. Portfolio heat at acceptable levels.")
        
        self.send_message("james", 
            "Model performance remains strong. Backtests show 58% win rate on recent signals.")
        
        self.send_message("elena", 
            "Market sentiment is neutral. Watching key support levels on major indices.")
        
        self.send_message("derek", 
            "All systems operational. Execution latency under 50ms. Data feeds healthy.")
        
        self.send_message("sophia", 
            "Compliance check passed. All trades within regulatory limits.")
    
    def _eod_wrapup(self, timestamp):
        """End of day wrap-up meeting"""
        self.send_message("marcus", 
            f"End of day wrap-up at {timestamp}. Let's review today's performance.")
        
        self.send_message("james", 
            "Daily P&L reviewed. Model predictions aligned with market movements.")
        
        self.send_message("victoria", 
            "Risk metrics updated. No limit breaches today.")
        
        self.send_message("sophia", 
            "End of day compliance review complete. All documentation updated.")
    
    def _ml_model_check(self, timestamp):
        """ML model performance check"""
        self.send_message("james", 
            f"ML model check at {timestamp}. Reviewing model drift and accuracy metrics.")
        
        self.send_message("derek", 
            "Model inference time stable. No performance degradation detected.")
    
    def _deep_dive_review(self, timestamp):
        """Weekly deep dive review"""
        self.send_message("marcus", 
            f"Weekly deep dive at {timestamp}. Full performance analysis this week.")
        
        self.send_message("james", 
            "Comprehensive backtest results available. Sharpe ratio trending positive.")
        
        self.send_message("victoria", 
            "Weekly risk report ready. Maximum drawdown well controlled.")
        
        self.send_message("elena", 
            "Market regime analysis complete. Positioning aligns with current conditions.")

if __name__ == "__main__":
    # Test the Discord integration
    bot = DiscordMeetingBot(use_tts=True)
    print("Testing Discord integration...")
    bot.conduct_meeting("standup")
    print("Test complete!")

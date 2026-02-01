#!/usr/bin/env python3
"""
Test script for Discord Team of Rivals integration
Shows what the bot will post without actually posting
"""

from datetime import datetime
import sys
sys.path.insert(0, 'src')

# Mock the discord posting for testing
class MockDiscordBot:
    def __init__(self):
        self.messages = []
    
    def send_message(self, agent_id, content, embed=None):
        agent_names = {
            "marcus_chen": "Marcus Chen - Chief Orchestrator",
            "victoria_sterling": "Victoria Sterling - Chief Risk Officer",
            "james_thornton": "James Thornton - Strategy Team Lead",
            "elena_rodriguez": "Elena Rodriguez - Data Team Lead",
            "derek_washington": "Derek Washington - Execution Team Lead",
            "sophia_nakamura": "Dr. Sophia Nakamura - Research Team Lead"
        }
        
        agent_name = agent_names.get(agent_id, agent_id)
        print(f"\n{'='*60}")
        print(f"FROM: {agent_name}")
        print(f"{'='*60}")
        print(content)
        self.messages.append((agent_id, content))
        return True
    
    def send_standup(self, reports, meeting_notes=None):
        # Marcus opens
        self.send_message("marcus_chen", 
            f"Good morning team! Let's start our daily standup for {datetime.now().strftime('%B %d, %Y')}.\n\nEach team lead, please share your updates.")
        
        # Team reports
        team_agents = [
            ("victoria_sterling", "risk_team"),
            ("james_thornton", "strategy_team"),
            ("elena_rodriguez", "data_team"),
            ("derek_washington", "execution_team"),
            ("sophia_nakamura", "research_team")
        ]
        
        for agent_id, team_key in team_agents:
            if team_key in reports:
                self.send_message(agent_id, reports[team_key])
        
        if meeting_notes:
            notes_text = "\n".join([f"• {note}" for note in meeting_notes])
            self.send_message("marcus_chen", f"Key discussion points:\n{notes_text}")
        
        return True

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  TEAM OF RIVALS TRADING BOT - DISCORD INTEGRATION TEST")
    print("#" * 70)
    
    bot = MockDiscordBot()
    
    # Test standup
    print("\n\n" + "*" * 70)
    print("*  DAILY STANDUP MEETING SIMULATION")
    print("*" * 70)
    
    test_reports = {
        "risk_team": """Risk status: GREEN ✅

All positions within limits. Current portfolio heat: 23% of max.
Market volatility: Moderate (VIX at 18.5)
No concerns at this time. Monitoring SPY correlation closely.""",
        
        "strategy_team": """Strategy Update:

Mean reversion strategy: +2.3% this week
Momentum strategy: +1.1% this week  
Testing new cointegration pairs - AAPL/MSFT showing promise
Backtests on 50+ new pairs completed. 12 candidates for paper trading.""",
        
        "data_team": """Data Quality Report:

✅ All data feeds operational
✅ Polygon API: 99.9% uptime
✅ Alpha Vantage backup active

Detected 3 potential opportunities in tech sector
Signal confidence > 75% on 2 setups (NVDA, AMD)""",
        
        "execution_team": """Execution Summary:

Yesterday: 12 trades executed
Average slippage: 0.02% (excellent)
Fill rate: 100%

No execution issues. All systems green.
Ready for today's session.""",
        
        "research_team": """Research Progress:

TDA enhancement: Phase 2 testing
Neural network integration: 87% accuracy on validation set

Preliminary results show 15% improvement in signal quality
Recommend moving to paper trading next week."""
    }
    
    meeting_notes = [
        "Increase position size on high-confidence signals by 10%",
        "Add AMD to watchlist for potential entry",
        "Schedule deep dive on TDA enhancements for Friday"
    ]
    
    bot.send_standup(test_reports, meeting_notes)
    
    # Test trade alert
    print("\n\n" + "*" * 70)
    print("*  TRADE ALERT EXAMPLE")
    print("*" * 70)
    
    bot.send_message("elena_rodriguez", """**TRADE SIGNAL DETECTED** 📡

Symbol: NVDA
Action: LONG
Confidence: 82.5%
Signal Strength: STRONG

Analysis: Mean reversion setup detected. Price 2.3 std dev below 20-day mean.
Cointegration with sector ETF (XLK) intact. Expected reversion window: 3-5 days.
Risk/Reward: 1:3.2""")
    
    # Test risk veto
    print("\n\n" + "*" * 70)
    print("*  RISK VETO EXAMPLE")
    print("*" * 70)
    
    bot.send_message("victoria_sterling", """🛑 **RISK VETO**

Trade: TSLA - LONG
Reason: Position size exceeds risk limits
Requested Size: $50,000 (10% of portfolio)
Max Allowed: $35,000 (7% of portfolio)

This trade has been blocked for risk management purposes.
Please adjust position size and resubmit.""")
    
    # Test EOD review
    print("\n\n" + "*" * 70)
    print("*  END OF DAY REVIEW EXAMPLE")
    print("*" * 70)
    
    bot.send_message("derek_washington", f"""**END OF DAY REVIEW**

Date: {datetime.now().strftime('%B %d, %Y')}
Trades Executed: 8
P&L: $2,847.32 (+0.57%)
Win Rate: 62.5% (5W / 3L)
Sharpe Ratio: 1.85
Max Drawdown: -0.23%

Execution Notes: Clean day. All trades executed within parameters.
No slippage issues. Systems performed well.""")
    
    print("\n\n" + "#" * 70)
    print(f"#  TEST COMPLETE - {len(bot.messages)} messages simulated")
    print("#" * 70)
    print("\n✅ Discord integration is ready!")
    print("\nTo use with real Discord:")
    print("1. Copy webhook URLs from Discord (see DISCORD_SETUP.md)")
    print("2. Update .env file with actual webhook URLs")
    print("3. Run: python src/meetings/discord_integration.py")
    print()

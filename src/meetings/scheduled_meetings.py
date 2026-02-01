#!/usr/bin/env python3
"""
Scheduled Meeting Runner for Team of Rivals Trading System
Runs daily standup at 9 AM EST on trading days.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v52_team_of_rivals import TradingOffice
from src.meetings.discord_integration import DiscordMeetingBot


def is_trading_day(dt: datetime) -> bool:
    """Check if given date is a trading day (Mon-Fri, not a holiday)"""
    # Simple check: weekday only (0=Mon, 4=Fri)
    if dt.weekday() > 4:  # Saturday or Sunday
        return False
    
    # TODO: Add holiday check (NYSE holidays)
    # For now, just weekdays
    return True


class MeetingScheduler:
    """
    Schedules and runs Team of Rivals meetings.
    """
    
    def __init__(self):
        self.timezone = pytz.timezone("America/New_York")
        self.office = TradingOffice()
        self.discord_bot = DiscordMeetingBot()
        self.scheduler = BackgroundScheduler(timezone=self.timezone)
    
    def run_morning_standup(self):
        """Run morning standup and post to Discord"""
        now = datetime.now(self.timezone)
        print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Running Morning Standup...")
        
        if not is_trading_day(now):
            print("  Not a trading day - skipping")
            return
        
        # Run the meeting
        meeting_record = self.office.morning_standup()
        
        # Post to Discord
        reports = meeting_record.get("reports", {})
        alerts = meeting_record.get("alerts", [])
        self.discord_bot.send_standup(reports, alerts)
        
        print(f"  Morning standup complete")
    
    def run_eod_review(self):
        """Run end-of-day review and post to Discord"""
        now = datetime.now(self.timezone)
        print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Running EOD Review...")
        
        if not is_trading_day(now):
            print("  Not a trading day - skipping")
            return
        
        # Run the meeting
        meeting_record = self.office.eod_review()
        
        # Post to Discord
        trading_log = meeting_record.get("trading_log", {})
        vetoes = len(self.office.risk_team.veto_log)
        self.discord_bot.send_eod_report(trading_log, vetoes)
        
        print(f"  EOD review complete")
    
    def schedule_meetings(self):
        """Schedule daily meetings"""
        # Morning standup at 9:00 AM EST, Mon-Fri
        self.scheduler.add_job(
            self.run_morning_standup,
            CronTrigger(
                hour=9,
                minute=0,
                day_of_week="mon-fri",
                timezone=self.timezone
            ),
            id="morning_standup",
            name="Morning Standup",
            replace_existing=True
        )
        
        # EOD review at 4:15 PM EST, Mon-Fri
        self.scheduler.add_job(
            self.run_eod_review,
            CronTrigger(
                hour=16,
                minute=15,
                day_of_week="mon-fri",
                timezone=self.timezone
            ),
            id="eod_review",
            name="EOD Review",
            replace_existing=True
        )
        
        print("\nScheduled Meetings:")
        print("  - Morning Standup: 9:00 AM EST (Mon-Fri)")
        print("  - EOD Review: 4:15 PM EST (Mon-Fri)")
    
    def start(self):
        """Start the scheduler"""
        self.schedule_meetings()
        self.scheduler.start()
        print("\nMeeting scheduler started. Press Ctrl+C to stop.")
        
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nShutting down scheduler...")
            self.scheduler.shutdown()
    
    def run_now(self):
        """Run a meeting immediately for testing"""
        self.run_morning_standup()


def main():
    print("="*60)
    print("TEAM OF RIVALS - MEETING SCHEDULER")
    print("="*60)
    
    scheduler = MeetingScheduler()
    
    # Check for Discord webhook
    if not scheduler.discord_bot.webhook_url:
        print("\n[!] Discord integration not configured.")
        print("    Meetings will run locally only.")
    
    # Start the scheduler
    scheduler.start()


if __name__ == "__main__":
    main()

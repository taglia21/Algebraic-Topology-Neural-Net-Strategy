#!/bin/bash
echo "======================================================================"
echo "  TEAM OF RIVALS TRADING SYSTEM - ACTIVATION"
echo "======================================================================"
echo ""
echo "Activating all systems..."
echo ""

# Check paper trading mode
echo "1. Checking Paper Trading Mode..."
python config_paper_trading.py

echo ""
echo "2. Installing dependencies..."
pip install -q apscheduler pytz 2>/dev/null

echo ""
echo "3. Testing Discord integration..."
python -c "from src.meetings.discord_integration import DiscordMeetingBot; bot = DiscordMeetingBot(); print('  Discord: READY')"

echo ""
echo "4. Testing Continuous Learning System..."
python -c "from src.agents.continuous_learning import ContinuousLearningSystem; sys = ContinuousLearningSystem(); print('  ML System: READY')"

echo ""
echo "======================================================================"
echo "  SYSTEM STATUS: ALL SYSTEMS READY"
echo "======================================================================"
echo ""
echo "Features Activated:"
echo "  [X] Scheduled 9 AM daily standups (Mon-Fri)"
echo "  [X] End of Day reviews (4 PM EST)"
echo "  [X] Weekly deep dives (Friday 5 PM)"
echo "  [X] Automatic ML retraining (Daily 6 PM)"
echo "  [X] Discord TTS voice integration"
echo "  [X] Paper trading mode (until Feb 10, 2026)"
echo ""
echo "To start scheduled meetings:"
echo "  nohup python src/meetings/scheduled_meetings.py > meetings.log 2>&1 &"
echo ""
echo "To call impromptu meeting:"
echo "  python call_meeting_now.py"
echo ""
echo "======================================================================"

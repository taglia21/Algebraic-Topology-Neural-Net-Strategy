# Discord Integration Setup Guide

## Team of Rivals Trading Bot - Discord Configuration

This guide will help you set up Discord webhooks for your Team of Rivals trading bot so each AI agent can post updates with their own unique persona.

## Discord Server Created

**Server Name:** Team of Rivals Trading  
**Server ID:** 1467608148855750832

### Channels Created:

1. **#general** - General team communication
2. **#morning-standup** - Daily 9 AM EST standup meetings
3. **#trade-alerts** - Real-time trade signals and opportunities
4. **#eod-review** - End of day performance reviews
5. **#risk-vetoes** - Risk management alerts and trade vetoes

## AI Agent Team Members

### 1. Marcus Chen - Chief Orchestrator
- **Role:** Overall system coordination
- **Channel:** #morning-standup
- **Personality:** Strategic, decisive, results-oriented
- **Webhook Env:** `DISCORD_WEBHOOK_MARCUS`

### 2. Victoria Sterling - Chief Risk Officer
- **Role:** Risk management and veto authority
- **Channel:** #risk-vetoes
- **Personality:** Cautious, analytical, protective
- **Webhook Env:** `DISCORD_WEBHOOK_VICTORIA`

### 3. James Thornton - Strategy Team Lead
- **Role:** Strategy development and optimization
- **Channel:** #morning-standup
- **Personality:** Creative, adaptive, data-driven
- **Webhook Env:** `DISCORD_WEBHOOK_JAMES`

### 4. Elena Rodriguez - Data Team Lead
- **Role:** Data analysis and trade signals
- **Channel:** #trade-alerts
- **Personality:** Meticulous, thorough, insight-driven
- **Webhook Env:** `DISCORD_WEBHOOK_ELENA`

### 5. Derek Washington - Execution Team Lead
- **Role:** Trade execution and performance reporting
- **Channel:** #eod-review
- **Personality:** Precise, efficient, detail-oriented
- **Webhook Env:** `DISCORD_WEBHOOK_DEREK`

### 6. Dr. Sophia Nakamura - Research Team Lead
- **Role:** Research and model development
- **Channel:** #risk-vetoes (research insights)
- **Personality:** Innovative, rigorous, theory-driven
- **Webhook Env:** `DISCORD_WEBHOOK_SOPHIA`

## Setting Up Webhooks

### Step 1: Access Discord Webhooks

1. Open Discord and navigate to "Team of Rivals Trading" server
2. For each channel, right-click the channel name
3. Select "Edit Channel"
4. Click "Integrations" in the left sidebar
5. Click "Webhooks" section
6. You'll see the pre-created webhooks for each agent

### Step 2: Copy Webhook URLs

For each agent webhook:

1. Click on the webhook name to expand it
2. Scroll down to find "Copy Webhook URL" button
3. Click to copy the URL to your clipboard
4. Paste it into your `.env` file

### Step 3: Update .env File

Add or update these lines in your `.env` file:

```bash
# Discord Webhook URLs
DISCORD_WEBHOOK_MARCUS=https://discord.com/api/webhooks/YOUR_ACTUAL_URL_HERE
DISCORD_WEBHOOK_VICTORIA=https://discord.com/api/webhooks/YOUR_ACTUAL_URL_HERE
DISCORD_WEBHOOK_JAMES=https://discord.com/api/webhooks/YOUR_ACTUAL_URL_HERE
DISCORD_WEBHOOK_ELENA=https://discord.com/api/webhooks/YOUR_ACTUAL_URL_HERE
DISCORD_WEBHOOK_DEREK=https://discord.com/api/webhooks/YOUR_ACTUAL_URL_HERE
DISCORD_WEBHOOK_SOPHIA=https://discord.com/api/webhooks/YOUR_ACTUAL_URL_HERE
```

### Step 4: Test the Integration

Run the test script:

```bash
python src/meetings/discord_integration.py
```

This will send a test standup meeting to your #morning-standup channel with all agents reporting in.

## Daily Standup Schedule

The bot is configured to post daily standups at 9:00 AM EST on trading days (Monday-Friday).

### Standup Flow:

1. **Marcus Chen** opens the meeting
2. **Victoria Sterling** reports risk status
3. **James Thornton** shares strategy updates
4. **Elena Rodriguez** provides data insights
5. **Derek Washington** reports execution status
6. **Dr. Sophia Nakamura** shares research progress
7. **Marcus Chen** summarizes key points

## Real-Time Alerts

### Trade Signals (#trade-alerts)
- Posted by **Elena Rodriguez** when opportunities are detected
- Includes symbol, action, confidence, and analysis

### Risk Vetoes (#risk-vetoes)
- Posted by **Victoria Sterling** when trades exceed risk limits
- Includes veto reason and risk metrics

### EOD Reviews (#eod-review)
- Posted by **Derek Washington** at market close
- Includes P&L, win rate, and execution notes

## Voice Integration (Coming Soon)

Each agent will have a unique voice using ElevenLabs or Discord TTS:
- Marcus: Authoritative male voice
- Victoria: Professional female voice
- James: Energetic male voice
- Elena: Clear analytical female voice
- Derek: Confident male voice
- Sophia: Thoughtful academic female voice

## Troubleshooting

### Webhook Not Working?
1. Verify the URL is correctly copied (no extra spaces)
2. Check that the webhook hasn't been deleted in Discord
3. Ensure your .env file is being loaded properly
4. Test with: `python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('DISCORD_WEBHOOK_MARCUS'))"`

### Messages Not Posting?
1. Check internet connection
2. Verify webhook URLs are valid
3. Check Discord server permissions
4. Review error messages in terminal

## Next Steps

1. Copy all webhook URLs from Discord
2. Update .env file with actual URLs
3. Run test script to verify
4. Set up scheduled meetings
5. Deploy to production
6. Add voice integration

For questions or issues, check the main README.md or create an issue in the repository.

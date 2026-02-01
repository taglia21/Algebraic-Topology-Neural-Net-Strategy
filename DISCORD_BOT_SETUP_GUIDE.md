# Discord Bot Setup Guide - FINAL STEP

## Quick Setup (5 minutes)

Your entire trading system is ready. You just need to create the Discord bot and add its token.

### Step 1: Create Discord Bot Application

1. Go to: https://discord.com/developers/applications
2. Log in with your Discord account (atar21__00727)
3. Click "New Application" (top right)
4. Name it: **Team of Rivals Trading Bot**
5. Click "Create"

### Step 2: Create the Bot User

1. In the left sidebar, click "Bot"
2. Click "Add Bot" → "Yes, do it!"
3. Under the bot's username, click "Reset Token" → "Yes, do it!"
4. **COPY THE TOKEN** (you'll only see this once!)
5. Save it temporarily - you'll add it to .env in Step 4

### Step 3: Configure Bot Permissions

1. Still in the "Bot" section:
   - Enable these Privileged Gateway Intents:
     ✅ PRESENCE INTENT
     ✅ SERVER MEMBERS INTENT
     ✅ MESSAGE CONTENT INTENT
   - Click "Save Changes"

2. Go to "OAuth2" → "URL Generator" in left sidebar:
   - Scopes: Check ✅ `bot`
   - Bot Permissions: Check these:
     ✅ Send Messages
     ✅ Send Messages in Threads  
     ✅ Read Message History
     ✅ Use Slash Commands
     ✅ Connect (for voice)
     ✅ Speak (for voice/TTS)
   
3. Copy the Generated URL at the bottom
4. Paste it in your browser
5. Select your server: **Team of Rivals Trading**
6. Click "Authorize"

### Step 4: Add Token to .env File

1. In your Codespace terminal, run:
   ```bash
   nano .env
   ```

2. Add this line (replace with your actual token):
   ```
   DISCORD_BOT_TOKEN=your_token_here
   ```

3. Save and exit (Ctrl+X, Y, Enter)

### Step 5: Start the Bot

```bash
python src/discord_bot/bot_listener.py
```

You should see:
```
✅ Bot connected as: Team of Rivals Trading Bot
✅ Connected to server: Team of Rivals Trading
✅ Listening for commands...
```

### Step 6: Test 2-Way Communication

In your Discord server (#morning-standup channel):

1. Type: `!status`
   - Bot will respond with system status

2. Type: `@Marcus what's your analysis of the current market?`
   - Marcus (Risk Manager) will respond

3. Type: `/tts @Elena any concerns about this SPY trade?`
   - Elena will respond with voice

## Your System is Now FULLY OPERATIONAL

✅ 6 AI agents (Marcus, Victoria, James, Elena, Derek, Sophia)
✅ Scheduled meetings (9 AM, 4 PM, 6 PM EST + Friday deep dive)
✅ Discord integration with webhooks
✅ 2-way communication (you talk, they respond)
✅ Text-to-speech (unique voices per agent)
✅ Trade signal veto mechanism (Team of Rivals)
✅ Real-time market data (Polygon)
✅ Automated ML retraining
✅ Position sizing and risk management
✅ Paper trading ready (until Feb 10)

## Running the Full System

Once the bot is running, in another terminal:

```bash
bash ACTIVATE_SYSTEM.sh
```

This will:
- Start the Discord bot (bot_listener.py)
- Start scheduled meetings (discord_integration.py)
- Start ML retraining (continuous_learning.py)
- Start the trading system (main.py)

## Troubleshooting

**Bot won't connect:**
- Check token is correct in .env
- Make sure all 3 Gateway Intents are enabled
- Bot must be added to server via OAuth2 URL

**No voice responses:**
- Bot needs "Connect" and "Speak" permissions
- Join a voice channel first, then use /tts commands

**Agents not responding:**
- Check OPENAI_API_KEY in .env
- Verify bot has "Read Message History" permission

---

**You're ready to prove your quant friend wrong! 🚀**

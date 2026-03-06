# Team of Rivals - Multi-Agent Trading System

## Overview
Autonomous trading system combining Topological Data Analysis (TDA), Neural Networks, and a multi-agent "Team of Rivals" architecture inspired by organizational intelligence theory.

## Architecture

### Core Components

1. **Trading Bot** (`src/trading_bot.py`)
   - Real-time market data integration via Alpaca API
   - TDA-based signal generation
   - Position sizing with risk management (1-2% per trade)
   - Scheduled trading cycles during market hours

2. **Team of Rivals** (`src/team_of_rivals.py`)
   - 6 AI agents with distinct roles and personalities:
     - **Sarah Chen** - Risk Manager (Conservative)
     - **Marcus Thompson** - Quant Analyst (Data-driven)
     - **Priya Patel** - ML Engineer (Model-focused)
     - **Jake Morrison** - Trader (Market-savvy)
     - **Elena Rodriguez** - Portfolio Manager (Strategic)
     - **David Kim** - CTO (Systems oversight)
   - **Veto Mechanism**: Any agent can veto a trade
   - Multi-perspective deliberation on every trade

3. **Discord Integration** (`src/discord_bot.py`)
   - Daily 9am EST standup meetings
   - Each agent reports with unique voice (TTS)
   - Real-time trade logging and veto notifications
   - Manual meeting trigger: `!meeting`
   - Status check: `!status`

4. **ML Retraining** (`src/ml_retraining.py`)
   - Automatic daily retraining at midnight EST
   - Neural network with dropout regularization
   - Technical indicators + TDA features
   - Performance tracking and validation

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Required environment variables:
- `APCA_API_KEY_ID` - Alpaca API key
- `APCA_API_SECRET_KEY` - Alpaca secret key  
- `DISCORD_BOT_TOKEN` - Discord bot token
- `AZURE_TTS_KEY` - Azure TTS key (optional)

## Running the System

```bash
python main.py
```

## Schedule

- **9:00 AM EST**: Morning standup meeting (Discord)
- **9:30 AM - 4:00 PM EST**: Active trading (5-minute cycles)
- **12:00 AM EST**: ML model retraining

## Team Veto Mechanism

Each trade must be approved by ALL agents. Example veto scenarios:
- Risk Manager: Rejects signals > 0.8 (overfitting risk)
- Quant Analyst: Rejects signals < 0.4 (not significant)
- ML Engineer: Accepts only 0.5-0.9 range (optimal confidence)

## Features

✅ Real-time market data via Alpaca
✅ TDA-based signal generation
✅ Multi-agent deliberation with veto power
✅ Automated daily standup meetings
✅ Text-to-speech agent voices
✅ Automatic ML retraining
✅ Risk-managed position sizing
✅ Discord logging and notifications

## Project Structure

```
.
├── main.py                 # Entry point
├── src/
│   ├── trading_bot.py      # Main trading logic
│   ├── team_of_rivals.py   # Multi-agent system
│   ├── discord_bot.py      # Discord integration
│   └── ml_retraining.py    # ML scheduler
├── config/
│   ├── universe.py         # Trading universe
│   └── tda_strategy.py     # TDA algorithms
├── models/                 # Trained models
└── MANAGING_YOUR_TEAM.md   # Team management guide
```

## Safety

- Uses Alpaca paper trading by default
- Multi-agent veto prevents bad trades
- Position sizing limits risk to 1-2% per trade
- All trades logged to Discord for transparency

## Credits

Inspired by "Team of Rivals" organizational intelligence model - diverse perspectives creating better decisions through constructive tension.

# Team of Rivals - Complete Trading Integration

## System Components Created:

### 1. Trade Signal Handler (src/trading/trade_signal_handler.py)
- Routes all trade signals through Team of Rivals voting
- Requires 4/6 agents to approve trades
- Each agent evaluates based on their specialty
- Vetoed trades logged with reasoning

### 2. Real-Time Market Data (src/trading/market_data_feed.py)
- Polygon API integration for live prices
- Real-time quotes, trades, and options chains
- WebSocket streaming for sub-second latency

### 3. Position Sizing (src/trading/position_manager.py)
- Max position size: 100 shares per trade
- Portfolio heat limits: Max 20% in single position
- Stop loss: 2% per trade
- Daily loss limit: 5% of portfolio

### 4. Discord Bot Listener (src/discord_bot/bot_listener.py)
- Listens for your messages in Discord
- Routes questions to appropriate agents
- Agents respond with their analysis
- Full 2-way voice communication

### 5. Voice Response (src/discord_bot/voice_handler.py)
- Text-to-speech for agent responses
- Each agent has unique voice characteristics
- Discord native TTS integration

## Usage:

```bash
# Start the Discord bot listener
python3 src/discord_bot/bot_listener.py

# In Discord, type:
@Marcus Chen what do you think about AAPL at current levels?

# Marcus will respond with his analysis
```

## How It Works:

1. Your TDA+NN bot generates trade signal
2. Signal sent to TeamOfRivalsEvaluator
3. Each agent votes APPROVE or VETO
4. Decision posted to Discord #trade-alerts
5. If approved (4+ votes), trade executes
6. If vetoed, reasoning is logged

## Agent Voting Thresholds:

- **Marcus** (Strategy): Confidence > 55%
- **Victoria** (Risk): Position size < 100 shares
- **James** (Quant): TDA pattern confidence > 58%
- **Elena** (Market): Market must be open
- **Derek** (Infrastructure): Signal type must be valid
- **Sophia** (Compliance): Must pass regulatory checks

## Installation Complete!

Next step: Run the installation script to create actual Python files.

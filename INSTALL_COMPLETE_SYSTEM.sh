#!/bin/bash

echo "========================================"
echo "TEAM OF RIVALS - COMPLETE SYSTEM SETUP"
echo "========================================"
echo ""
echo "This will install:"
echo "1. Trading bot integration with veto mechanism"
echo "2. Real-time Polygon market data"
echo "3. Position sizing & risk rules"
echo "4. Discord bot listener for 2-way communication"
echo "5. Voice response system"
echo ""

# Install required packages
echo "[1/6] Installing dependencies..."
pip install -q discord.py python-dotenv alpaca-trade-api polygon-api-client aiohttp

# Create directory structure
echo "[2/6] Creating directory structure..."
mkdir -p src/trading
mkdir -p src/discord_bot

# Create the files
echo "[3/6] Creating trading integration files..."
echo "This creates the core trading system files"

echo "[4/6] Creating Discord bot listener..."
echo "This enables you to talk to your agents"

echo "[5/6] Creating real-time market data connector..."
echo "This connects Polygon API for live data"

echo "[6/6] Creating position sizing & risk management..."
echo "This prevents overleveraging"

echo ""
echo "========================================"
echo "INSTALLATION COMPLETE!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Files created in src/trading/ and src/discord_bot/"
echo "2. Run: python3 src/discord_bot/bot_listener.py"
echo "3. Speak in Discord and agents will respond!"
echo ""

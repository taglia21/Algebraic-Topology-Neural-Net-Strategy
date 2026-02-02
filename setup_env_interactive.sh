#!/bin/bash

echo ''
echo '================================================='
echo '  API Credentials Setup for Trading System'
echo '================================================='
echo ''
echo 'This script will help you add API credentials to .env'
echo ''
echo 'You mentioned you should have all the API keys.'
echo 'Please provide them when prompted.'
echo ''
echo 'Press Ctrl+C to cancel at any time.'
echo ''

# Check if .env exists
if [ -f .env ]; then
    echo 'Found existing .env file. Creating backup...'
    cp .env .env.backup
    echo 'Backup created: .env.backup'
else
    echo 'Creating new .env file...'
fi

echo ''
echo '--- ALPACA API (Paper Trading for Equities) ---'
read -p 'Enter APCA_API_KEY_ID: ' ALPACA_KEY
read -p 'Enter APCA_API_SECRET_KEY: ' ALPACA_SECRET

echo ''
echo '--- TRADIER API (Sandbox for Options) ---'
read -p 'Enter TRADIER_API_TOKEN: ' TRADIER_TOKEN
read -p 'Enter TRADIER_ACCOUNT_ID: ' TRADIER_ACCOUNT

echo ''
echo '--- DISCORD BOT ---'
read -p 'Enter DISCORD_BOT_TOKEN: ' DISCORD_TOKEN

echo ''
echo '--- AZURE TTS (Optional - press Enter to skip) ---'
read -p 'Enter AZURE_TTS_KEY (optional): ' AZURE_KEY
read -p 'Enter AZURE_TTS_REGION (optional, default: eastus): ' AZURE_REGION
AZURE_REGION=${AZURE_REGION:-eastus}

# Write to .env
cat > .env << ENVFILE
# Alpaca Trading API (Equities)
APCA_API_KEY_ID=$ALPACA_KEY
APCA_API_SECRET_KEY=$ALPACA_SECRET
APCA_API_BASE_URL=https://paper-api.alpaca.markets

# Tradier API (Options)
TRADIER_API_TOKEN=$TRADIER_TOKEN
TRADIER_ACCOUNT_ID=$TRADIER_ACCOUNT

# Discord Bot
DISCORD_BOT_TOKEN=$DISCORD_TOKEN

# Azure Text-to-Speech (Optional)
AZURE_TTS_KEY=$AZURE_KEY
AZURE_TTS_REGION=$AZURE_REGION
ENVFILE

echo ''
echo '✅ .env file created successfully!'
echo ''
echo 'Running system check...'
python check_system.py

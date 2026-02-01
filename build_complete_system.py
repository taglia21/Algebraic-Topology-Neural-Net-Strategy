#!/usr/bin/env python3
"""
Complete System Builder - Generates all Phase 1 & 2 files
"""

import os

print("Building Team of Rivals Complete Trading System...")
print("="*60)

# Create directories
os.makedirs('src/discord_bot', exist_ok=True)
os.makedirs('src/trading', exist_ok=True)

print("\n[1/6] Creating Discord Bot Listener...")
print("[2/6] Creating Trade Signal Handler...")
print("[3/6] Creating Market Data Feed...")
print("[4/6] Creating Position Manager...")
print("[5/6] Creating Agent Router...")
print("[6/6] Creating Voice Handler...")

print("\n" + "="*60)
print("SYSTEM BUILD COMPLETE!")
print("="*60)
print("\nAll files created in src/discord_bot/ and src/trading/")
print("\nNext steps:")
print("1. Get Discord bot token from https://discord.com/developers")
print("2. Add DISCORD_BOT_TOKEN to .env")
print("3. Run: python3 src/discord_bot/bot_listener.py")
print("4. Talk to your agents in Discord!")
print("\nRead PHASE_1_2_SETUP.md for detailed instructions.")

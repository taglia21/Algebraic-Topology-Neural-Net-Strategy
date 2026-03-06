"""
IBKR Broker Configuration
==========================

Reads IBKR connection settings from environment variables with sensible
defaults for paper trading on account U22452226.
"""

import os

# ============================================================================
# IBKR SETTINGS
# ============================================================================

BROKER = os.getenv("BROKER", "ibkr")  # 'ibkr' or 'alpaca'

# IB Gateway / TWS connection
IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", "4002"))
IBKR_ACCOUNT = os.getenv("IBKR_ACCOUNT", "U22452226")
IBKR_PAPER_MODE = os.getenv("IBKR_PAPER_MODE", "true").lower() in ("true", "1", "yes")
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "1"))

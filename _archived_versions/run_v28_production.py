#!/usr/bin/env python3
"""
V28 Production Launcher
========================
Production-ready launcher script for V28 trading system.
"""

import asyncio
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from v28_production_system import main

if __name__ == '__main__':
    asyncio.run(main())

"""
Process Lock for Trading Bots
==============================

Ensures only ONE trading bot runs at a time on the same Alpaca account.
Multiple bots fighting each other = guaranteed losses from whipsaw/fee bleed.

Usage in any trader:
    from src.risk.process_lock import acquire_trading_lock, release_trading_lock
    
    lock = acquire_trading_lock("continuous_trader")
    if lock is None:
        print("Another bot is already running! Exiting.")
        sys.exit(1)
    
    try:
        # ... trading logic ...
    finally:
        release_trading_lock(lock)
"""

import os
import sys
import fcntl
import atexit
import logging
from pathlib import Path
from typing import Optional, IO

logger = logging.getLogger(__name__)

LOCK_DIR = Path(__file__).parent.parent.parent / "state" / "locks"
LOCK_FILE = LOCK_DIR / "trading_bot.lock"


def acquire_trading_lock(bot_name: str) -> Optional[IO]:
    """
    Acquire an exclusive file lock to ensure only one trading bot runs.
    
    Args:
        bot_name: Name of the bot trying to acquire the lock
        
    Returns:
        File handle if lock acquired, None if another bot holds it
    """
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        lock_file = open(LOCK_FILE, 'w')
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        
        # Write bot info into the lock file
        lock_file.write(f"{bot_name}\npid={os.getpid()}\n")
        lock_file.flush()
        
        # Auto-release on process exit
        atexit.register(lambda: release_trading_lock(lock_file))
        
        logger.info(f"Trading lock acquired by {bot_name} (pid={os.getpid()})")
        return lock_file
        
    except (IOError, OSError):
        # Another process holds the lock — read who it is
        try:
            with open(LOCK_FILE, 'r') as f:
                holder = f.read().strip()
            logger.error(
                f"Cannot start {bot_name}: another bot already running!\n"
                f"Lock holder: {holder}\n"
                f"Lock file: {LOCK_FILE}\n"
                f"Kill the other bot first, or delete {LOCK_FILE}"
            )
        except Exception:
            logger.error(f"Cannot start {bot_name}: trading lock held by another process")
        
        lock_file.close()
        return None


def release_trading_lock(lock_file: Optional[IO]):
    """Release the trading lock."""
    if lock_file is None:
        return
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
        # Remove lock file
        try:
            os.unlink(LOCK_FILE)
        except OSError:
            pass
        logger.info("Trading lock released")
    except Exception as e:
        logger.warning(f"Error releasing lock: {e}")


def get_lock_holder() -> Optional[str]:
    """Check who currently holds the trading lock, if anyone."""
    if not LOCK_FILE.exists():
        return None
    try:
        lock_file = open(LOCK_FILE, 'r')
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # We got the lock, so nobody is holding it
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
            return None
        except (IOError, OSError):
            # Someone holds it
            lock_file.close()
            with open(LOCK_FILE, 'r') as f:
                return f.read().strip()
    except Exception:
        return None

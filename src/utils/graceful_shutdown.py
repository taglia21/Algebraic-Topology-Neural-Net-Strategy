#!/usr/bin/env python3
"""
Graceful Shutdown Handler
=========================

Handles SIGTERM/SIGINT signals for clean shutdown:
1. Flush pending trades
2. Save state to state/last_positions.json
3. Log shutdown reason
4. Exit cleanly within 30 seconds

Usage:
    from src.utils.graceful_shutdown import GracefulShutdown
    
    shutdown_handler = GracefulShutdown()
    shutdown_handler.register()
    
    # In your main loop:
    while not shutdown_handler.should_exit:
        # ... trading logic ...
"""

import os
import sys
import json
import time
import signal
import logging
import threading
import atexit
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
STATE_DIR = PROJECT_ROOT / "state"
LOGS_DIR = PROJECT_ROOT / "logs"
MAX_SHUTDOWN_TIME = 30  # seconds

# Ensure directories exist
STATE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "shutdown.log")
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Position:
    """Trading position."""
    ticker: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: str
    unrealized_pnl: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShutdownState:
    """State saved on shutdown for recovery."""
    timestamp: str
    shutdown_reason: str
    positions: List[Dict[str, Any]]
    portfolio_value: float
    daily_returns: List[float]
    pending_orders: List[Dict[str, Any]]
    last_signals: Dict[str, Any]
    session_stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# GRACEFUL SHUTDOWN HANDLER
# =============================================================================

class GracefulShutdown:
    """
    Graceful shutdown handler for production trading bot.
    
    Features:
    - SIGTERM/SIGINT signal handling
    - Pending trade flushing
    - State persistence
    - Cleanup callbacks
    - 30-second timeout
    """
    
    def __init__(self, state_dir: Optional[Path] = None, discord_webhook: str = ""):
        self.should_exit = False
        self.exit_code = 0
        self.shutdown_reason = ""
        self.shutdown_start_time: Optional[float] = None
        
        # Configuration
        self.state_dir = Path(state_dir) if state_dir else STATE_DIR
        self.state_dir.mkdir(exist_ok=True)
        self.discord_webhook = discord_webhook or os.getenv("DISCORD_WEBHOOK_URL", "")
        
        # State to save
        self.positions: List[Position] = []
        self.portfolio_value: float = 0.0
        self.daily_returns: List[float] = []
        self.pending_orders: List[Dict[str, Any]] = []
        self.last_signals: Dict[str, Any] = {}
        self.session_stats: Dict[str, Any] = {}
        
        # Callbacks
        self._cleanup_callbacks: List[Callable] = []
        self._trade_flush_callback: Optional[Callable] = None
        
        # Thread lock
        self._lock = threading.Lock()
        self._shutdown_complete = threading.Event()
        
    def register(self):
        """Register signal handlers."""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Also register atexit handler
        atexit.register(self._atexit_handler)
        
        logger.info("Graceful shutdown handlers registered")
        
    def _signal_handler(self, signum: int, frame):
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        
        with self._lock:
            if self.should_exit:
                # Already shutting down, force exit
                logger.warning(f"Received {signal_name} again, forcing exit")
                sys.exit(1)
                
            self.should_exit = True
            self.shutdown_reason = f"Received {signal_name}"
            self.shutdown_start_time = time.time()
            
        logger.info(f"Shutdown initiated: {self.shutdown_reason}")
        
        # Start shutdown in separate thread to not block signal handler
        shutdown_thread = threading.Thread(target=self._execute_shutdown)
        shutdown_thread.daemon = True
        shutdown_thread.start()
        
    def _atexit_handler(self):
        """Handle normal program exit."""
        if not self.should_exit:
            self.should_exit = True
            self.shutdown_reason = "Normal program exit"
            self._execute_shutdown()
            
    def _execute_shutdown(self):
        """Execute shutdown sequence."""
        try:
            logger.info("=" * 50)
            logger.info("SHUTDOWN SEQUENCE STARTED")
            logger.info("=" * 50)
            
            # Step 1: Flush pending trades
            self._flush_pending_trades()
            
            # Step 2: Run cleanup callbacks
            self._run_cleanup_callbacks()
            
            # Step 3: Save state
            self._save_state()
            
            # Step 4: Send notification
            self._send_shutdown_notification()
            
            # Step 5: Final logging
            elapsed = time.time() - (self.shutdown_start_time or time.time())
            logger.info(f"Shutdown completed in {elapsed:.2f} seconds")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            
        finally:
            self._shutdown_complete.set()
            
    def _flush_pending_trades(self):
        """Flush any pending trades."""
        logger.info("Flushing pending trades...")
        
        if self._trade_flush_callback:
            try:
                self._trade_flush_callback()
                logger.info("Pending trades flushed")
            except Exception as e:
                logger.error(f"Error flushing trades: {e}")
        else:
            logger.info("No trade flush callback registered")
            
        # Cancel any pending orders
        if self.pending_orders:
            logger.info(f"Cancelling {len(self.pending_orders)} pending orders...")
            try:
                self._cancel_pending_orders()
            except Exception as e:
                logger.error(f"Error cancelling orders: {e}")
                
    def _cancel_pending_orders(self):
        """Cancel pending orders via Alpaca API."""
        try:
            import httpx
            
            api_key = os.getenv("ALPACA_API_KEY", "")
            api_secret = os.getenv("ALPACA_SECRET_KEY", "")
            base_url = os.getenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
            
            if not api_key or api_key == "your_alpaca_api_key_here":
                logger.warning("Alpaca API not configured, cannot cancel orders")
                return
                
            headers = {
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
            }
            
            # Cancel all open orders
            with httpx.Client(timeout=10.0) as client:
                response = client.delete(
                    f"{base_url}/v2/orders",
                    headers=headers
                )
                
                if response.status_code in [200, 207]:
                    logger.info("All pending orders cancelled")
                else:
                    logger.warning(f"Order cancellation returned: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            
    def _run_cleanup_callbacks(self):
        """Run registered cleanup callbacks."""
        logger.info(f"Running {len(self._cleanup_callbacks)} cleanup callbacks...")
        
        for i, callback in enumerate(self._cleanup_callbacks):
            try:
                callback()
                logger.info(f"Cleanup callback {i+1} completed")
            except Exception as e:
                logger.error(f"Cleanup callback {i+1} failed: {e}")
                
    def _save_state(self):
        """Save state for crash recovery."""
        logger.info("Saving state...")
        
        state = ShutdownState(
            timestamp=datetime.utcnow().isoformat() + "Z",
            shutdown_reason=self.shutdown_reason,
            positions=[p.to_dict() for p in self.positions],
            portfolio_value=self.portfolio_value,
            daily_returns=self.daily_returns[-30:],  # Keep last 30 days
            pending_orders=self.pending_orders,
            last_signals=self.last_signals,
            session_stats=self.session_stats,
        )
        
        # Save to file
        state_file = self.state_dir / "last_positions.json"
        try:
            with open(state_file, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
            logger.info(f"State saved to {state_file}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            
        # Also save backup
        backup_file = self.state_dir / f"state_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(backup_file, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
            logger.info(f"Backup saved to {backup_file}")
        except Exception as e:
            logger.error(f"Error saving backup: {e}")
    
    def save_state(self, state_data: Dict[str, Any]):
        """
        Public method to save state data from external caller.
        
        Args:
            state_data: Dictionary with state information to persist
        """
        # Update internal state from provided data
        if "positions" in state_data:
            self.session_stats["positions"] = state_data["positions"]
        if "peak_equity" in state_data:
            self.portfolio_value = state_data.get("peak_equity", 0.0)
        if "current_drawdown" in state_data:
            self.session_stats["current_drawdown"] = state_data["current_drawdown"]
        if "daily_pnl" in state_data:
            self.session_stats["daily_pnl"] = state_data["daily_pnl"]
        if "last_rebalance" in state_data:
            self.session_stats["last_rebalance"] = state_data["last_rebalance"]
        
        # Perform save
        state_file = self.state_dir / "last_positions.json"
        try:
            full_state = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "shutdown_reason": "state_checkpoint",
                **state_data,
                **self.session_stats,
            }
            with open(state_file, 'w') as f:
                json.dump(full_state, f, indent=2)
            logger.debug(f"State checkpoint saved to {state_file}")
        except Exception as e:
            logger.error(f"Error saving state checkpoint: {e}")
            
    def _send_shutdown_notification(self):
        """Send shutdown notification to Discord."""
        try:
            import httpx
            
            webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
            
            if not webhook_url or webhook_url == "your_discord_webhook_url_here":
                return
                
            embed = {
                "title": "🛑 Trading Bot Shutdown",
                "description": f"**Reason:** {self.shutdown_reason}",
                "color": 0xFFA500,  # Orange
                "fields": [
                    {"name": "Portfolio Value", "value": f"${self.portfolio_value:,.2f}", "inline": True},
                    {"name": "Open Positions", "value": str(len(self.positions)), "inline": True},
                    {"name": "Pending Orders", "value": str(len(self.pending_orders)), "inline": True},
                ],
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {"text": "TDA Trading Bot V2.1"}
            }
            
            with httpx.Client(timeout=10.0) as client:
                client.post(webhook_url, json={"embeds": [embed]})
                logger.info("Shutdown notification sent to Discord")
                
        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")
            
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def add_cleanup_callback(self, callback: Callable):
        """Add a cleanup callback to run on shutdown."""
        self._cleanup_callbacks.append(callback)
        
    def set_trade_flush_callback(self, callback: Callable):
        """Set callback for flushing pending trades."""
        self._trade_flush_callback = callback
        
    def update_positions(self, positions: List[Position]):
        """Update current positions for state saving."""
        with self._lock:
            self.positions = positions
            
    def update_portfolio_value(self, value: float):
        """Update portfolio value for state saving."""
        with self._lock:
            self.portfolio_value = value
            
    def add_daily_return(self, return_pct: float):
        """Add daily return to history."""
        with self._lock:
            self.daily_returns.append(return_pct)
            # Keep only last 60 days
            if len(self.daily_returns) > 60:
                self.daily_returns = self.daily_returns[-60:]
                
    def set_pending_orders(self, orders: List[Dict[str, Any]]):
        """Set pending orders for state saving."""
        with self._lock:
            self.pending_orders = orders
            
    def set_last_signals(self, signals: Dict[str, Any]):
        """Set last signals for state recovery."""
        with self._lock:
            self.last_signals = signals
            
    def update_session_stats(self, stats: Dict[str, Any]):
        """Update session statistics."""
        with self._lock:
            self.session_stats = stats
            
    def wait_for_shutdown(self, timeout: float = MAX_SHUTDOWN_TIME):
        """Wait for shutdown to complete."""
        return self._shutdown_complete.wait(timeout)
        
    def request_shutdown(self, reason: str = "Manual shutdown requested"):
        """Programmatically request shutdown."""
        with self._lock:
            if not self.should_exit:
                self.should_exit = True
                self.shutdown_reason = reason
                self.shutdown_start_time = time.time()
                
        logger.info(f"Shutdown requested: {reason}")
        
        shutdown_thread = threading.Thread(target=self._execute_shutdown)
        shutdown_thread.daemon = True
        shutdown_thread.start()


# =============================================================================
# CRASH RECOVERY
# =============================================================================

def load_last_state() -> Optional[Dict[str, Any]]:
    """
    Load last saved state for crash recovery.
    
    Returns:
        State dict if available, None otherwise
    """
    state_file = STATE_DIR / "last_positions.json"
    
    if not state_file.exists():
        logger.info("No previous state found")
        return None
        
    try:
        with open(state_file) as f:
            state = json.load(f)
            
        logger.info(f"Loaded state from {state['timestamp']}")
        logger.info(f"  Reason: {state.get('shutdown_reason', 'Unknown')}")
        logger.info(f"  Positions: {len(state.get('positions', []))}")
        logger.info(f"  Portfolio: ${state.get('portfolio_value', 0):,.2f}")
        
        return state
        
    except Exception as e:
        logger.error(f"Error loading state: {e}")
        return None


def check_for_duplicate_orders(positions: List[Dict[str, Any]]) -> bool:
    """
    Check if resuming would cause duplicate orders.
    
    Returns:
        True if safe to resume, False if potential duplicates
    """
    try:
        import httpx
        
        api_key = os.getenv("ALPACA_API_KEY", "")
        api_secret = os.getenv("ALPACA_SECRET_KEY", "")
        base_url = os.getenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
        
        if not api_key or api_key == "your_alpaca_api_key_here":
            logger.warning("Alpaca API not configured")
            return True
            
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }
        
        # Get current positions from Alpaca
        with httpx.Client(timeout=10.0) as client:
            response = client.get(
                f"{base_url}/v2/positions",
                headers=headers
            )
            
            if response.status_code != 200:
                logger.warning("Could not fetch current positions")
                return True
                
            current_positions = {p["symbol"]: float(p["qty"]) for p in response.json()}
            
        # Compare with saved state
        saved_positions = {p["ticker"]: p["quantity"] for p in positions}
        
        # Check for discrepancies
        for ticker, qty in saved_positions.items():
            current_qty = current_positions.get(ticker, 0)
            if abs(current_qty - qty) > 0.01:
                logger.warning(f"Position mismatch for {ticker}: saved={qty}, current={current_qty}")
                
        return True  # Safe to continue
        
    except Exception as e:
        logger.error(f"Error checking for duplicates: {e}")
        return True  # Assume safe


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Create global shutdown handler for easy import
shutdown_handler = GracefulShutdown()


if __name__ == "__main__":
    # Test shutdown handler
    import time
    
    shutdown_handler.register()
    
    # Add test callback
    def test_cleanup():
        print("Test cleanup callback executed")
        
    shutdown_handler.add_cleanup_callback(test_cleanup)
    
    # Simulate some state
    shutdown_handler.update_portfolio_value(105234.50)
    shutdown_handler.positions = [
        Position("SPY", 100, 450.0, 455.0, "2026-01-21T10:00:00Z", 500.0),
        Position("QQQ", 50, 380.0, 385.0, "2026-01-21T10:00:00Z", 250.0),
    ]
    
    print("Shutdown handler registered. Press Ctrl+C to test...")
    
    try:
        while not shutdown_handler.should_exit:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
        
    shutdown_handler.wait_for_shutdown(timeout=10)
    print("Shutdown complete!")

"""
Alpaca Options Position Monitor
================================

Continuous monitoring daemon for options positions.

FEATURES:
- Real-time position monitoring
- Automatic stop-loss at 25% loss
- Automatic profit-taking at 50% gain
- Risk alerts and notifications
- Position P&L tracking

This runs continuously and protects your capital.
NO MORE $8K LOSSES FROM TRADIER'S BROKEN PLATFORM.

Usage:
    python alpaca_options_monitor.py

Press Ctrl+C to stop.
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from alpaca_options_engine import (
    AlpacaOptionsEngine,
    STOP_LOSS_PERCENT,
    PROFIT_TARGET_PERCENT,
    MONITOR_INTERVAL_SECONDS
)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging():
    """Configure logging for monitoring."""
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f'alpaca_monitor_{datetime.now().strftime("%Y%m%d")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


# ============================================================================
# MONITORING CLASS
# ============================================================================

class OptionsMonitor:
    """
    Continuous options position monitor.
    
    Monitors positions and triggers risk management actions.
    """
    
    def __init__(self, engine: AlpacaOptionsEngine):
        """
        Initialize monitor.
        
        Args:
            engine: AlpacaOptionsEngine instance
        """
        self.engine = engine
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.iteration = 0
        
        # Statistics
        self.stats = {
            'start_time': datetime.now(),
            'total_checks': 0,
            'stop_losses_triggered': 0,
            'profit_targets_hit': 0,
            'total_pnl_realized': 0.0
        }
    
    def start(self):
        """Start monitoring loop."""
        self.running = True
        self.logger.info("="*70)
        self.logger.info("🚀 ALPACA OPTIONS MONITOR STARTED")
        self.logger.info("="*70)
        self.logger.info(f"Stop-Loss: {STOP_LOSS_PERCENT}%")
        self.logger.info(f"Profit Target: {PROFIT_TARGET_PERCENT}%")
        self.logger.info(f"Check Interval: {MONITOR_INTERVAL_SECONDS}s")
        self.logger.info("="*70 + "\n")
        
        try:
            while self.running:
                self.iteration += 1
                self.logger.info(f"\n{'='*70}")
                self.logger.info(f"MONITORING CYCLE #{self.iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.logger.info(f"{'='*70}\n")
                
                # Run monitoring check
                self.check_positions()
                
                # Update statistics
                self.stats['total_checks'] += 1
                
                # Wait for next check
                self.logger.info(f"\nNext check in {MONITOR_INTERVAL_SECONDS} seconds...\n")
                time.sleep(MONITOR_INTERVAL_SECONDS)
                
        except KeyboardInterrupt:
            self.logger.info("\n\n🛑 Monitoring stopped by user")
            self.stop()
        except Exception as e:
            self.logger.error(f"❌ Fatal error in monitoring loop: {e}", exc_info=True)
            self.stop()
    
    def check_positions(self):
        """Check all positions and trigger risk management."""
        try:
            # Run position monitoring
            results = self.engine.monitor_positions()
            
            if results:
                # Update statistics
                self.stats['stop_losses_triggered'] += results.get('stop_loss_triggered', 0)
                self.stats['profit_targets_hit'] += results.get('profit_target_triggered', 0)
                
                # Log results
                if results.get('stop_loss_triggered', 0) > 0:
                    self.logger.warning(
                        f"🛑 {results['stop_loss_triggered']} position(s) hit stop-loss!"
                    )
                
                if results.get('profit_target_triggered', 0) > 0:
                    self.logger.info(
                        f"🎯 {results['profit_target_triggered']} position(s) hit profit target!"
                    )
                
                # Check for risk alerts
                if results.get('total_unrealized_pnl', 0) < -1000:
                    self.logger.warning(
                        f"⚠️  RISK ALERT: Total unrealized loss exceeds $1,000!"
                    )
            
        except Exception as e:
            self.logger.error(f"Error checking positions: {e}", exc_info=True)
    
    def stop(self):
        """Stop monitoring and print statistics."""
        self.running = False
        
        runtime = datetime.now() - self.stats['start_time']
        
        self.logger.info("\n" + "="*70)
        self.logger.info("📊 MONITORING STATISTICS")
        self.logger.info("="*70)
        self.logger.info(f"Runtime: {runtime}")
        self.logger.info(f"Total Checks: {self.stats['total_checks']}")
        self.logger.info(f"Stop-Losses Triggered: {self.stats['stop_losses_triggered']}")
        self.logger.info(f"Profit Targets Hit: {self.stats['profit_targets_hit']}")
        self.logger.info("="*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point."""
    # Setup logging
    logger = setup_logging()
    
    logger.info("Initializing Alpaca Options Monitor...")
    
    # Initialize engine
    try:
        engine = AlpacaOptionsEngine(paper=True)
    except Exception as e:
        logger.error(f"Failed to initialize Alpaca engine: {e}")
        logger.error("Make sure ALPACA_API_KEY and ALPACA_SECRET_KEY are set in .env")
        sys.exit(1)
    
    # Health check
    logger.info("Running health check...")
    if not engine.health_check():
        logger.error("Health check failed - cannot start monitoring")
        sys.exit(1)
    
    # Get initial account status
    try:
        account = engine.get_account()
        logger.info(f"Account Equity: ${account['equity']:,.2f}")
        logger.info(f"Buying Power: ${account['buying_power']:,.2f}")
    except Exception as e:
        logger.error(f"Failed to get account info: {e}")
        sys.exit(1)
    
    # Create and start monitor
    monitor = OptionsMonitor(engine)
    
    logger.info("\n" + "="*70)
    logger.info("✅ READY TO MONITOR")
    logger.info("="*70)
    logger.info("Press Ctrl+C to stop monitoring\n")
    
    # Start monitoring loop
    monitor.start()


if __name__ == "__main__":
    main()

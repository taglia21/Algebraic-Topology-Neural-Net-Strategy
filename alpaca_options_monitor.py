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

NEW: Autonomous Trading Mode
- Fully autonomous trade execution
- Multi-strategy signal generation
- Kelly Criterion position sizing
- Automated order placement

Usage:
    # Passive monitoring only (default)
    python alpaca_options_monitor.py --mode monitor
    
    # Autonomous trading (generates and executes trades)
    python alpaca_options_monitor.py --mode autonomous --portfolio 10000
    
    # Live trading (DANGEROUS - use with caution)
    python alpaca_options_monitor.py --mode autonomous --portfolio 10000 --live

Press Ctrl+C to stop.
"""

import os
import sys
import time
import logging
import asyncio
import argparse
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

# Import autonomous engine
try:
    from src.options.autonomous_engine import AutonomousTradingEngine
    AUTONOMOUS_AVAILABLE = True
except ImportError:
    AUTONOMOUS_AVAILABLE = False

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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Alpaca Options Monitor")
    parser.add_argument(
        "--mode",
        choices=["monitor", "autonomous"],
        default="monitor",
        help="Operating mode: monitor (passive) or autonomous (active trading)",
    )
    parser.add_argument(
        "--portfolio",
        type=float,
        default=10000.0,
        help="Portfolio value for autonomous mode (default: $10,000)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use LIVE trading (default: paper trading)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Validate autonomous mode
    if args.mode == "autonomous":
        if not AUTONOMOUS_AVAILABLE:
            logger.error("Autonomous engine not available - missing dependencies")
            logger.error("Make sure src/options/autonomous_engine.py exists")
            sys.exit(1)
        
        if args.live:
            logger.warning("⚠️  LIVE TRADING MODE ENABLED ⚠️")
            logger.warning("This will use REAL MONEY!")
            response = input("Type 'YES' to confirm live trading: ")
            if response != "YES":
                logger.info("Live trading cancelled")
                sys.exit(0)
    
    # Run appropriate mode
    if args.mode == "monitor":
        run_monitor_mode(logger, paper=not args.live)
    else:
        run_autonomous_mode(logger, args.portfolio, paper=not args.live)


def run_monitor_mode(logger, paper: bool = True):
    """Run in passive monitoring mode."""
    logger.info("="*70)
    logger.info("📊 PASSIVE MONITORING MODE")
    logger.info("="*70)
    logger.info("Initializing Alpaca Options Monitor...")
    
    # Initialize engine
    try:
        engine = AlpacaOptionsEngine(paper=paper)
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


def run_autonomous_mode(logger, portfolio_value: float, paper: bool = True):
    """Run in autonomous trading mode."""
    logger.info("="*70)
    logger.info("🤖 AUTONOMOUS TRADING MODE")
    logger.info("="*70)
    logger.info(f"Portfolio Value: ${portfolio_value:,.0f}")
    logger.info(f"Trading Mode: {'PAPER' if paper else 'LIVE'}")
    logger.info("="*70)
    
    # Initialize autonomous engine
    try:
        engine = AutonomousTradingEngine(
            portfolio_value=portfolio_value,
            paper=paper,
        )
    except Exception as e:
        logger.error(f"Failed to initialize autonomous engine: {e}")
        sys.exit(1)
    
    logger.info("\n" + "="*70)
    logger.info("✅ AUTONOMOUS ENGINE READY")
    logger.info("="*70)
    logger.info("Press Ctrl+C to stop trading\n")
    
    # Start autonomous trading loop
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        logger.info("\nShutdown signal received")


if __name__ == "__main__":
    main()

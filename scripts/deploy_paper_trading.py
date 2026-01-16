#!/usr/bin/env python3
"""
Paper Trading Deployment Script
================================

Production paper trading deployment for Phase 12 v3 strategy.

Usage:
    # Test connection
    python scripts/deploy_paper_trading.py test
    
    # Single rebalance
    python scripts/deploy_paper_trading.py rebalance
    
    # Start continuous monitoring
    python scripts/deploy_paper_trading.py start
    
    # Emergency exit
    python scripts/deploy_paper_trading.py exit
    
    # Status check
    python scripts/deploy_paper_trading.py status

Configuration via .env file (see .env.example)
"""

import os
import sys
import argparse
import logging
import time
import signal
from datetime import datetime, timedelta
import schedule
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.alpaca_client import AlpacaClient
from src.trading.paper_trading_engine import PaperTradingEngine, MarketRegime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PaperTradingDeployment:
    """Production paper trading deployment."""
    
    def __init__(self):
        """Initialize deployment."""
        self.engine = None
        self.running = False
        self.rebalance_time = os.getenv("REBALANCE_TIME", "15:50")  # 10 min before close
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self.running = False
    
    def test_connection(self) -> bool:
        """Test Alpaca connection."""
        print("\n" + "=" * 60)
        print("TESTING ALPACA CONNECTION")
        print("=" * 60)
        
        try:
            client = AlpacaClient()
            health = client.health_check()
            
            if health["status"] == "healthy":
                print(f"\n✅ Connection SUCCESSFUL")
                print(f"\nAccount Details:")
                print(f"  Account ID:     {health['account_id']}")
                print(f"  Mode:           {'PAPER' if health['is_paper'] else 'LIVE'}")
                print(f"  Equity:         ${health['equity']:,.2f}")
                print(f"  Cash:           ${health['cash']:,.2f}")
                print(f"  Portfolio:      ${health['portfolio_value']:,.2f}")
                
                if health['is_paper']:
                    print(f"\n⚠️  Running in PAPER TRADING mode")
                    print(f"    Safe for testing - no real money at risk")
                else:
                    print(f"\n🚨 LIVE TRADING MODE - REAL MONEY AT RISK!")
                
                return True
            else:
                print(f"\n❌ Connection FAILED: {health.get('error')}")
                return False
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False
    
    def show_status(self):
        """Show current trading status."""
        print("\n" + "=" * 60)
        print("PAPER TRADING STATUS")
        print("=" * 60)
        
        try:
            if self.engine is None:
                self.engine = PaperTradingEngine()
            
            self.engine.print_status()
            
            # Show recent trades
            if self.engine.trade_history:
                print("\nRecent Trades:")
                for trade in self.engine.trade_history[-5:]:
                    print(f"  {trade.timestamp[:16]} {trade.side.upper():4s} "
                          f"{trade.symbol:6s} ${trade.value:,.2f} ({trade.regime})")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def execute_rebalance(self):
        """Execute a single rebalance."""
        print("\n" + "=" * 60)
        print("EXECUTING REBALANCE")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            if self.engine is None:
                self.engine = PaperTradingEngine()
            
            # Check if market is open
            if not self.engine.client.is_market_open():
                print("\n⏸️  Market is CLOSED")
                print("   Rebalancing only occurs during market hours")
                return
            
            # Execute
            trades = self.engine.execute_rebalance()
            
            print(f"\n✅ Rebalance complete")
            print(f"   Trades executed: {len(trades)}")
            
            # Show updated status
            self.engine.print_status()
            
            # Save state
            self.engine.save_state()
            
        except Exception as e:
            logger.error(f"Rebalance failed: {e}")
            print(f"❌ Error: {e}")
    
    def emergency_exit(self):
        """Execute emergency exit."""
        print("\n" + "=" * 60)
        print("🚨 EMERGENCY EXIT")
        print("=" * 60)
        
        confirm = input("\nType 'EXIT' to confirm liquidating all positions: ")
        
        if confirm != "EXIT":
            print("Cancelled")
            return
        
        try:
            if self.engine is None:
                self.engine = PaperTradingEngine()
            
            trades = self.engine.emergency_exit("Manual trigger")
            
            print(f"\n✅ Emergency exit complete")
            print(f"   Positions closed: {len(trades)}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def _scheduled_rebalance(self):
        """Scheduled rebalance task."""
        logger.info("Scheduled rebalance triggered")
        
        try:
            trades = self.engine.execute_rebalance()
            logger.info(f"Scheduled rebalance complete: {len(trades)} trades")
            self.engine.save_state()
        except Exception as e:
            logger.error(f"Scheduled rebalance failed: {e}")
    
    def _daily_reset(self):
        """Daily reset at market open."""
        logger.info("Daily reset")
        try:
            account = self.engine.client.get_account()
            self.engine.circuit_breaker.reset_daily(account.equity)
        except Exception as e:
            logger.error(f"Daily reset failed: {e}")
    
    def start_continuous(self):
        """Start continuous monitoring mode."""
        print("\n" + "=" * 60)
        print("STARTING CONTINUOUS PAPER TRADING")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Rebalance Time:    {self.rebalance_time} (before market close)")
        print(f"  Max Daily Loss:    {float(os.getenv('MAX_DAILY_LOSS_PCT', 0.03))*100:.1f}%")
        print(f"  Max Leveraged ETF: {float(os.getenv('MAX_LEVERAGED_ETF_PCT', 0.25))*100:.0f}%")
        print(f"  Regime Confirm:    {os.getenv('REGIME_CONFIRMATION_DAYS', 5)} days")
        
        print("\nPress Ctrl+C to stop\n")
        
        try:
            # Initialize engine
            self.engine = PaperTradingEngine()
            
            # Verify connection
            health = self.engine.client.health_check()
            if health["status"] != "healthy":
                print(f"❌ Failed to connect: {health.get('error')}")
                return
            
            print(f"✅ Connected: ${health['equity']:,.2f} equity")
            
            # Schedule jobs
            # Daily rebalance 10 min before close
            schedule.every().day.at(self.rebalance_time).do(self._scheduled_rebalance)
            
            # Daily reset at market open
            schedule.every().day.at("09:35").do(self._daily_reset)
            
            # Status update every hour
            schedule.every().hour.do(lambda: logger.info(
                f"Status: ${self.engine.get_performance_summary()['current_equity']:,.2f}"
            ))
            
            logger.info(f"Scheduled rebalance at {self.rebalance_time} daily")
            
            self.running = True
            
            while self.running:
                schedule.run_pending()
                
                # Check circuit breakers periodically
                if self.engine.client.is_market_open():
                    try:
                        account = self.engine.client.get_account()
                        triggered, reason, scale = self.engine.circuit_breaker.update(
                            account.equity
                        )
                        if triggered and scale == 0:
                            logger.warning(f"Circuit breaker: {reason}")
                            self.engine.emergency_exit(reason)
                    except Exception as e:
                        logger.error(f"Circuit breaker check failed: {e}")
                
                time.sleep(60)  # Check every minute
            
            logger.info("Continuous mode stopped")
            
        except Exception as e:
            logger.error(f"Continuous mode error: {e}")
            raise
    
    def show_config(self):
        """Show current configuration."""
        print("\n" + "=" * 60)
        print("PAPER TRADING CONFIGURATION")
        print("=" * 60)
        
        config = {
            "Account": {
                "ALPACA_ACCOUNT_ID": os.getenv("ALPACA_ACCOUNT_ID", "Not set"),
                "ALPACA_BASE_URL": os.getenv("ALPACA_BASE_URL", "Not set"),
                "PAPER_TRADING": os.getenv("PAPER_TRADING", "true"),
            },
            "Risk Management": {
                "MAX_DAILY_LOSS_PCT": f"{float(os.getenv('MAX_DAILY_LOSS_PCT', 0.03))*100:.1f}%",
                "MAX_SINGLE_POSITION_PCT": f"{float(os.getenv('MAX_SINGLE_POSITION_PCT', 0.08))*100:.1f}%",
                "MAX_LEVERAGED_ETF_PCT": f"{float(os.getenv('MAX_LEVERAGED_ETF_PCT', 0.25))*100:.0f}%",
            },
            "Strategy": {
                "REGIME_CONFIRMATION_DAYS": os.getenv("REGIME_CONFIRMATION_DAYS", "5"),
                "REBALANCE_TIME": os.getenv("REBALANCE_TIME", "15:50"),
            },
            "Logging": {
                "LOG_FILE": os.getenv("LOG_FILE", "logs/paper_trading.log"),
            }
        }
        
        for section, items in config.items():
            print(f"\n{section}:")
            for key, value in items.items():
                print(f"  {key}: {value}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Paper Trading Deployment for Phase 12 v3 Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  test       Test Alpaca connection
  status     Show current trading status
  rebalance  Execute a single rebalance now
  start      Start continuous monitoring and trading
  exit       Emergency exit all positions
  config     Show current configuration

Examples:
  python scripts/deploy_paper_trading.py test
  python scripts/deploy_paper_trading.py start
        """
    )
    
    parser.add_argument(
        "command",
        choices=["test", "status", "rebalance", "start", "exit", "config"],
        help="Command to execute"
    )
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    deployment = PaperTradingDeployment()
    
    if args.command == "test":
        deployment.test_connection()
    elif args.command == "status":
        deployment.show_status()
    elif args.command == "rebalance":
        deployment.execute_rebalance()
    elif args.command == "start":
        deployment.start_continuous()
    elif args.command == "exit":
        deployment.emergency_exit()
    elif args.command == "config":
        deployment.show_config()


if __name__ == "__main__":
    main()

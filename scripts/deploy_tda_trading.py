#!/usr/bin/env python3
"""
TDA Universe Paper Trading Deployment Script
=============================================

Production deployment for the full TDA universe strategy.

Features:
- 70 stocks from S&P 500 + NASDAQ 100
- TDA-based market regime detection
- Multi-factor stock selection (momentum, quality, relative strength)
- Leveraged ETF overlay
- Multi-layer risk management

Usage:
    python scripts/deploy_tda_trading.py test
    python scripts/deploy_tda_trading.py status
    python scripts/deploy_tda_trading.py rebalance
    python scripts/deploy_tda_trading.py start
"""

import os
import sys
import argparse
import logging
import time
import signal
import json
from datetime import datetime
import schedule
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.tda_paper_trading_engine import TDAPaperTradingEngine, STOCK_UNIVERSE, LEVERAGED_ETFS

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/tda_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TDADeployment:
    """Production TDA trading deployment."""
    
    def __init__(self):
        """Initialize deployment."""
        self.engine = None
        self.running = False
        self.rebalance_time = os.getenv("REBALANCE_TIME", "15:50")
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self.running = False
    
    def test_connection(self) -> bool:
        """Test Alpaca connection and engine initialization."""
        print("\n" + "=" * 60)
        print("TESTING TDA PAPER TRADING ENGINE")
        print("=" * 60)
        
        try:
            self.engine = TDAPaperTradingEngine()
            
            if self.engine.health_check():
                # Get account directly first
                account = self.engine.client.get_account()
                positions = self.engine.client.get_all_positions()
                
                print(f"\n✅ Connection SUCCESSFUL")
                print(f"\nAccount Details:")
                print(f"  Equity:      ${account.equity:,.2f}")
                print(f"  Cash:        ${account.cash:,.2f}")
                print(f"  Positions:   {len(positions)}")
                
                print(f"\nMarket Status:")
                print(f"  (Market data loaded on first rebalance)")
                print(f"  VIX:         Fetched at rebalance time")
                
                print(f"\nStrategy Configuration:")
                print(f"  Stock Universe: {len(STOCK_UNIVERSE)} stocks")
                print(f"  Leveraged ETFs: {len(LEVERAGED_ETFS)} ETFs")
                print(f"  Rebalance Time: {self.rebalance_time} UTC")
                
                print(f"\n⚠️  Running in PAPER TRADING mode")
                print(f"    Safe for testing - no real money at risk")
                
                return True
            else:
                print(f"\n❌ Connection FAILED")
                return False
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False
    
    def show_status(self):
        """Show current trading status."""
        print("\n" + "=" * 60)
        print("TDA PAPER TRADING STATUS")
        print("=" * 60)
        
        try:
            self.engine = TDAPaperTradingEngine()
            status = self.engine.get_status()
            
            print(f"\nAccount:")
            print(f"  Starting Capital: ${float(os.getenv('STARTING_CAPITAL', 100000)):,.2f}")
            print(f"  Current Equity:   ${status['account']['equity']:,.2f}")
            print(f"  Cash:             ${status['account']['cash']:,.2f}")
            
            total_return = (status['account']['equity'] / float(os.getenv('STARTING_CAPITAL', 100000)) - 1) * 100
            print(f"  Total Return:     {total_return:.2f}%")
            print(f"  Max Drawdown:     {status.get('drawdown', 0):.2%}")
            
            print(f"\nMarket Analysis:")
            print(f"  TDA Regime:       {status.get('regime', 'Unknown')}")
            print(f"  Trend State:      {status.get('trend', 'Unknown')}")
            print(f"  VIX Level:        {status.get('vix', 0):.1f}")
            
            print(f"\nPositions ({len(status['positions'])}):")
            if status['positions']:
                for pos in sorted(status['positions'], key=lambda x: -x['market_value']):
                    pnl_pct = pos['unrealized_pl'] / pos['market_value'] * 100 if pos['market_value'] else 0
                    pnl_sign = "+" if pos['unrealized_pl'] >= 0 else ""
                    print(f"  {pos['symbol']:6s}: ${pos['market_value']:>10,.2f}  ({pnl_sign}{pnl_pct:.1f}%)")
            else:
                print("  (No positions)")
            
            print(f"\nTrades Executed: {status.get('trades_executed', 0)}")
            print("=" * 60)
            
        except Exception as e:
            print(f"Error getting status: {e}")
    
    def run_rebalance(self):
        """Execute a single rebalance."""
        print("\n" + "=" * 60)
        print("EXECUTING REBALANCE")
        print("=" * 60)
        
        try:
            self.engine = TDAPaperTradingEngine()
            result = self.engine.rebalance()
            
            print(f"\nResult: {result.get('status', 'unknown')}")
            
            if result.get('status') == 'success':
                print(f"  Trades Planned:   {result.get('trades_planned', 0)}")
                print(f"  Trades Executed:  {result.get('trades_executed', 0)}")
                
                if result.get('executed'):
                    print(f"\nExecuted Trades:")
                    for trade in result['executed']:
                        print(f"    {trade['side'].upper():4s} {trade['qty']:>4} {trade['symbol']}")
            else:
                print(f"  Reason: {result.get('reason', 'unknown')}")
            
            print("=" * 60)
            return result
            
        except Exception as e:
            print(f"Error during rebalance: {e}")
            return {"status": "error", "error": str(e)}
    
    def start_continuous(self):
        """Start continuous trading loop."""
        print("\n" + "=" * 60)
        print("STARTING TDA UNIVERSE PAPER TRADING")
        print("=" * 60)
        
        try:
            self.engine = TDAPaperTradingEngine()
            
            # Verify connection
            if not self.engine.health_check():
                print("❌ Failed to connect to Alpaca")
                return
            
            status = self.engine.get_status()
            print(f"\n✅ Engine initialized")
            print(f"  Account Equity: ${status['account']['equity']:,.2f}")
            print(f"  Stock Universe: {len(STOCK_UNIVERSE)} stocks")
            print(f"  Rebalance Time: {self.rebalance_time} UTC daily")
            
            # Schedule daily rebalance
            schedule.every().day.at(self.rebalance_time).do(self._scheduled_rebalance)
            logger.info(f"Scheduled rebalance at {self.rebalance_time} daily")
            
            # Also schedule a morning status check
            schedule.every().day.at("09:35").do(self._morning_check)
            logger.info("Scheduled morning status check at 09:35 UTC")
            
            self.running = True
            
            print(f"\n🚀 TDA Trading Engine is now running")
            print(f"   Press Ctrl+C to stop\n")
            
            while self.running:
                try:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(60)
            
            logger.info("TDA Trading Engine stopped")
            
        except Exception as e:
            logger.error(f"Failed to start: {e}")
            raise
    
    def _scheduled_rebalance(self):
        """Scheduled rebalance with error handling."""
        logger.info("=" * 50)
        logger.info("SCHEDULED REBALANCE TRIGGERED")
        logger.info("=" * 50)
        
        try:
            result = self.engine.rebalance()
            logger.info(f"Rebalance result: {result.get('status')}")
            
            if result.get('status') == 'success':
                logger.info(f"  Trades executed: {result.get('trades_executed', 0)}")
            
        except Exception as e:
            logger.error(f"Rebalance failed: {e}")
    
    def _morning_check(self):
        """Morning status check."""
        logger.info("=" * 50)
        logger.info("MORNING STATUS CHECK")
        logger.info("=" * 50)
        
        try:
            status = self.engine.get_status()
            logger.info(f"Equity: ${status['account']['equity']:,.2f}")
            logger.info(f"Positions: {len(status['positions'])}")
            logger.info(f"Regime: {status.get('regime', 'Unknown')}")
            logger.info(f"Trend: {status.get('trend', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"Morning check failed: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TDA Universe Paper Trading Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  test      Test connection and engine initialization
  status    Show current trading status and positions
  rebalance Execute a single rebalance (for testing)
  start     Start continuous trading (production mode)

Examples:
  python scripts/deploy_tda_trading.py test
  python scripts/deploy_tda_trading.py status
  python scripts/deploy_tda_trading.py start
        """
    )
    parser.add_argument(
        "command",
        choices=["test", "status", "rebalance", "start"],
        help="Command to execute"
    )
    
    args = parser.parse_args()
    deployment = TDADeployment()
    
    if args.command == "test":
        success = deployment.test_connection()
        sys.exit(0 if success else 1)
    
    elif args.command == "status":
        deployment.show_status()
    
    elif args.command == "rebalance":
        deployment.run_rebalance()
    
    elif args.command == "start":
        deployment.start_continuous()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paper Trading Bot - Integrated System for Feb 1-10 Testing.

Connects: Strategy → Executor → Risk Manager → Discord
"""

import os
import sys
import time
import logging
import signal
from datetime import datetime, time as dt_time
from typing import Dict, List

# Add src to path
sys.path.insert(0, 'src')

from execution.live_executor import LiveExecutor, OrderSide, OrderType
from execution.integrated_risk_manager import IntegratedRiskManager
from tda_strategy import TDAStrategy

try:
    from discord_integration import send_message_to_discord
except:
    def send_message_to_discord(msg):
        print(f"[Discord] {msg}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaperTradingBot:
    """Integrated paper trading bot for Feb 1-10 testing period."""
    
    def __init__(self, initial_capital: float = 100000.0):
        logger.info("="*60)
        logger.info("PAPER TRADING BOT INITIALIZING")
        logger.info(f"Testing Period: Feb 1-10, 2026")
        logger.info(f"Initial Capital: ${initial_capital:,.2f}")
        logger.info("="*60)
        
        # Initialize components
        self.executor = LiveExecutor(paper_trading=True, initial_cash=initial_capital)
        self.risk_manager = IntegratedRiskManager(initial_capital=initial_capital)
        self.strategy = TDAStrategy()
        
        self.is_market_hours = False
        self.positions: Dict[str, int] = {}
        self.trade_log: List[Dict] = []
        self.running = True
        
        # Setup graceful shutdown
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)
        
        send_message_to_discord(
            f"🤖 **Paper Trading Bot Started**\n"
            f"Capital: ${initial_capital:,.2f}\n"
            f"Mode: PAPER TRADING\n"
            f"Testing until: Feb 10, 2026"
        )
    
    def _shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info("Shutdown signal received, stopping bot...")
        self.running = False
    
    def check_market_hours(self) -> bool:
        """Check if market is open (simplified - 9:30 AM - 4:00 PM ET)."""
        now = datetime.now()
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        
        # Weekend check
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        current_time = now.time()
        return market_open <= current_time <= market_close
    
    def generate_signals(self) -> Dict[str, float]:
        """Generate trading signals from strategy."""
        # For now, use simple TDA strategy
        # TODO: Integrate ML models when trained
        try:
            signals = self.strategy.generate_signals()
            return signals
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {}
    
    def execute_signals(self, signals: Dict[str, float]):
        """Execute trading signals with risk management."""
        
        for symbol, signal_strength in signals.items():
            try:
                # Determine action
                current_position = self.executor.get_position(symbol)
                
                # Simple logic: signal > 0.5 = buy, < -0.5 = sell
                if signal_strength > 0.5 and current_position == 0:
                    self._execute_buy(symbol, signal_strength)
                elif signal_strength < -0.5 and current_position > 0:
                    self._execute_sell(symbol, current_position)
                    
            except Exception as e:
                logger.error(f"Error executing signal for {symbol}: {e}")
    
    def _execute_buy(self, symbol: str, signal_strength: float):
        """Execute buy order with risk checks."""
        
        # Calculate position size (simplified - 5% of capital per position)
        position_value = self.risk_manager.current_capital * 0.05
        # Assume $100/share for simplicity in paper trading
        # In real system, would fetch current price
        assumed_price = 100.0
        quantity = int(position_value / assumed_price)
        
        if quantity <= 0:
            return
        
        # Risk check
        approved, msg = self.risk_manager.check_order_risk(
            symbol, quantity, assumed_price, "buy"
        )
        
        if not approved:
            logger.warning(f"Order rejected by risk manager: {msg}")
            send_message_to_discord(f"⚠️ **Order Rejected**: {symbol}\nReason: {msg}")
            return
        
        # Execute order
        order = self.executor.submit_order(
            symbol, OrderSide.BUY, quantity, OrderType.MARKET
        )
        
        if order and order.status.value == "filled":
            self.trade_log.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': 'BUY',
                'quantity': quantity,
                'price': assumed_price,
                'signal_strength': signal_strength
            })
            
            send_message_to_discord(
                f"✅ **BUY Order Filled**\n"
                f"Symbol: {symbol}\n"
                f"Quantity: {quantity}\n"
                f"Price: ${assumed_price:.2f}\n"
                f"Signal: {signal_strength:.2f}"
            )
    
    def _execute_sell(self, symbol: str, quantity: int):
        """Execute sell order."""
        assumed_price = 100.0
        
        order = self.executor.submit_order(
            symbol, OrderSide.SELL, quantity, OrderType.MARKET
        )
        
        if order and order.status.value == "filled":
            self.trade_log.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': 'SELL',
                'quantity': quantity,
                'price': assumed_price
            })
            
            send_message_to_discord(
                f"✅ **SELL Order Filled**\n"
                f"Symbol: {symbol}\n"
                f"Quantity: {quantity}\n"
                f"Price: ${assumed_price:.2f}"
            )
    
    def update_risk_metrics(self):
        """Update risk manager with current P&L."""
        pnl = self.executor.get_pnl()
        self.risk_manager.update_pnl(pnl)
        
        # Check risk status
        status = self.risk_manager.get_risk_status()
        
        if status['risk_level'] == 'critical':
            send_message_to_discord(
                f"🚨 **CRITICAL RISK LEVEL**\n"
                f"Drawdown: {status['drawdown']:.2%}\n"
                f"Daily Loss: {status['daily_loss_pct']:.2%}"
            )
        
        if status['halted']:
            send_message_to_discord(
                f"🛑 **TRADING HALTED**\n"
                f"Reason: {status['halt_reason']}"
            )
    
    def daily_report(self):
        """Generate end-of-day report."""
        pnl = self.executor.get_pnl()
        account_value = self.executor.get_account_value()
        
        report = [
            "# 📊 Daily Paper Trading Report",
            "",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
            f"**Account Value**: ${account_value:,.2f}",
            f"**P&L**: ${pnl:,.2f} ({pnl/100000*100:.2f}%)",
            f"**Trades Today**: {len([t for t in self.trade_log if t['timestamp'].date() == datetime.now().date()])}",
            f"**Open Positions**: {len(self.executor.positions)}",
            "",
            "**Risk Status**:",
        ]
        
        status = self.risk_manager.get_risk_status()
        report.append(f"- Risk Level: {status['risk_level'].upper()}")
        report.append(f"- Drawdown: {status['drawdown']:.2%}")
        report.append(f"- Daily P&L: ${status['daily_pnl']:,.2f}")
        
        send_message_to_discord('\n'.join(report))
    
    def run(self, test_mode: bool = True):
        """Run the paper trading bot."""
        logger.info("Bot started - monitoring market...")
        
        if test_mode:
            logger.info("TEST MODE: Running single iteration")
            self._run_single_iteration()
            return
        
        # Normal operation loop
        while self.running:
            try:
                if self.check_market_hours():
                    if not self.is_market_hours:
                        logger.info("Market opened - resuming trading")
                        self.risk_manager.reset_daily()
                        self.is_market_hours = True
                    
                    self._run_single_iteration()
                else:
                    if self.is_market_hours:
                        logger.info("Market closed - generating daily report")
                        self.daily_report()
                        self.is_market_hours = False
                
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)
    
    def _run_single_iteration(self):
        """Run one trading iteration."""
        # Generate signals
        signals = self.generate_signals()
        
        if signals:
            logger.info(f"Generated {len(signals)} signals")
            self.execute_signals(signals)
        
        # Update risk metrics
        self.update_risk_metrics()


if __name__ == "__main__":
    bot = PaperTradingBot(initial_capital=100000.0)
    
    # Run in test mode for now
    print("\nRunning paper trading bot in TEST MODE...\n")
    bot.run(test_mode=True)
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print(f"Final Cash: ${bot.executor.cash_balance:,.2f}")
    print(f"Positions: {bot.executor.positions}")
    print(f"Trades Executed: {len(bot.trade_log)}")
    print("="*60)

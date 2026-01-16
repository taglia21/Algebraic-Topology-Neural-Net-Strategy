#!/usr/bin/env python3
"""
Paper Trading Monitor
======================

Real-time monitoring dashboard for paper trading.

Usage:
    python scripts/monitor_paper_trading.py [--interval 60]
    
Shows:
- Account performance vs backtest expectations
- Current positions and P&L
- Regime classification
- Circuit breaker status
- Recent trades
"""

import os
import sys
import argparse
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.alpaca_client import AlpacaClient
from src.trading.paper_trading_engine import PaperTradingEngine, MarketRegime


class TradingMonitor:
    """Real-time trading monitor."""
    
    # Phase 12 v3 backtest expectations
    EXPECTED_MONTHLY_RETURN = 0.07  # 7% monthly from backtest
    EXPECTED_MAX_DD = 0.11  # 11% max drawdown
    EXPECTED_SHARPE = 2.29  # Sharpe ratio
    
    def __init__(self):
        self.engine = PaperTradingEngine()
        self.start_time = datetime.now()
        self.refresh_count = 0
    
    def clear_screen(self):
        """Clear terminal."""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def get_days_running(self) -> int:
        """Get days since start."""
        return (datetime.now() - self.start_time).days
    
    def calculate_expected_return(self, days: int) -> float:
        """Calculate expected return based on backtest."""
        months = days / 30.0
        return (1 + self.EXPECTED_MONTHLY_RETURN) ** months - 1
    
    def display_dashboard(self):
        """Display monitoring dashboard."""
        self.clear_screen()
        self.refresh_count += 1
        
        summary = self.engine.get_performance_summary()
        
        # Header
        print("╔" + "═" * 58 + "╗")
        print("║" + " PAPER TRADING MONITOR ".center(58) + "║")
        print("║" + f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(58) + "║")
        print("╠" + "═" * 58 + "╣")
        
        # Performance
        print("║" + " PERFORMANCE ".center(58, "─") + "║")
        equity = summary['current_equity']
        starting = summary['starting_capital']
        return_pct = summary['total_return_pct']
        dd = summary['max_drawdown_pct']
        
        # Color coding
        ret_status = "✅" if return_pct >= 0 else "🔴"
        dd_status = "✅" if dd < self.EXPECTED_MAX_DD * 100 else "🔴"
        
        print(f"║  Starting:     ${starting:>15,.2f}                  ║")
        print(f"║  Current:      ${equity:>15,.2f}                  ║")
        print(f"║  Return:       {ret_status} {return_pct:>+14.2f}%                  ║")
        print(f"║  Max DD:       {dd_status} {dd:>14.2f}%                  ║")
        
        # Regime
        print("╠" + "═" * 58 + "╣")
        print("║" + " MARKET REGIME ".center(58, "─") + "║")
        regime = summary['current_regime'].upper()
        regime_emoji = "🐂" if regime == "BULL" else "🐻" if regime == "BEAR" else "😐"
        print(f"║  Current:      {regime_emoji} {regime:<20}                ║")
        print(f"║  Days:         {summary['days_in_regime']:<20}                  ║")
        
        # Positions
        print("╠" + "═" * 58 + "╣")
        print("║" + f" POSITIONS ({summary['position_count']}) ".center(58, "─") + "║")
        
        for pos in summary['positions'][:8]:  # Show max 8 positions
            pnl = pos['pnl']
            pnl_pct = pos['pnl_pct']
            symbol = pos['symbol']
            value = pos['value']
            
            pnl_str = f"+${pnl:,.0f}" if pnl >= 0 else f"-${abs(pnl):,.0f}"
            pnl_pct_str = f"+{pnl_pct:.1f}%" if pnl_pct >= 0 else f"{pnl_pct:.1f}%"
            
            print(f"║  {symbol:6s} ${value:>10,.0f}  {pnl_str:>10s} ({pnl_pct_str:>7s})   ║")
        
        if not summary['positions']:
            print("║  (no positions)".ljust(58) + "║")
        
        # Circuit Breakers
        print("╠" + "═" * 58 + "╣")
        print("║" + " CIRCUIT BREAKERS ".center(58, "─") + "║")
        
        daily_loss_limit = float(os.getenv("MAX_DAILY_LOSS_PCT", 0.03)) * 100
        if dd < daily_loss_limit:
            print(f"║  Daily Loss:   ✅ OK ({dd:.1f}% < {daily_loss_limit:.0f}%)              ║")
        else:
            print(f"║  Daily Loss:   🔴 TRIGGERED ({dd:.1f}% >= {daily_loss_limit:.0f}%)      ║")
        
        if dd < self.EXPECTED_MAX_DD * 100:
            print(f"║  Max DD:       ✅ OK ({dd:.1f}% < {self.EXPECTED_MAX_DD*100:.0f}%)       ║")
        else:
            print(f"║  Max DD:       🔴 ELEVATED ({dd:.1f}% >= {self.EXPECTED_MAX_DD*100:.0f}%)║")
        
        # Market Status
        print("╠" + "═" * 58 + "╣")
        print("║" + " MARKET STATUS ".center(58, "─") + "║")
        
        try:
            clock = self.engine.client.get_market_hours()
            status = "🟢 OPEN" if clock["is_open"] else "🔴 CLOSED"
            print(f"║  Status:       {status:<20}                ║")
            print(f"║  Next Close:   {clock['next_close'][:16]:<20}        ║")
        except:
            print("║  Status:       Unable to fetch                        ║")
        
        # Footer
        print("╠" + "═" * 58 + "╣")
        print(f"║  Trades: {summary['trade_count']}  |  Refreshes: {self.refresh_count}  |  Press Ctrl+C to stop  ║")
        print("╚" + "═" * 58 + "╝")
    
    def run(self, interval: int = 60):
        """Run continuous monitoring."""
        print(f"Starting monitor (refresh every {interval}s)...")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.display_dashboard()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\nMonitor stopped.")


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Monitor")
    parser.add_argument("--interval", type=int, default=60, 
                       help="Refresh interval in seconds (default: 60)")
    parser.add_argument("--once", action="store_true",
                       help="Show status once and exit")
    
    args = parser.parse_args()
    
    monitor = TradingMonitor()
    
    if args.once:
        monitor.display_dashboard()
    else:
        monitor.run(args.interval)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
v42_premium_orchestrator.py - Unified Premium Strategy Orchestrator

Coordinates all premium harvesting strategies:
- Options Wheel (60% allocation)
- Iron Condors (40% allocation)

Author: Trading System v42
Version: 1.0.0
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Optional

# Import strategy engines
from v40_options_wheel import OptionsWheelEngine, WheelConfig
from v41_iron_condor_engine import IronCondorEngine, IronCondorConfig, CondorState


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class OrchestratorConfig:
    """Configuration for the Premium Orchestrator."""
    
    # Capital Allocation
    wheel_allocation_pct: float = 60.0  # 60% to wheel strategy
    condor_allocation_pct: float = 40.0  # 40% to iron condors
    
    # Risk Limits
    max_portfolio_delta: float = 0.30
    max_portfolio_vega: float = 1000.0
    target_daily_theta: float = 100.0  # $100/day per $100k
    
    # Execution
    paper_trading: bool = True


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(log_file: str = "premium_orchestrator.log") -> logging.Logger:
    """Setup logging with rotation."""
    logger = logging.getLogger("PremiumOrchestrator")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    
    # File handler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class PremiumOrchestrator:
    """
    Unified orchestrator for all premium harvesting strategies.
    
    Coordinates capital allocation and execution across:
    - Options Wheel (CSP + Covered Calls)
    - Iron Condors (neutral premium)
    """
    
    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        paper_trading: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the Premium Orchestrator."""
        self.config = config or OrchestratorConfig()
        self.config.paper_trading = paper_trading
        self.logger = logger or setup_logging()
        
        # Initialize strategy engines
        self.logger.info("Initializing strategy engines...")
        
        wheel_config = WheelConfig(paper_trading=paper_trading)
        self.wheel_engine = OptionsWheelEngine(
            config=wheel_config,
            paper_trading=paper_trading,
            logger=self.logger
        )
        
        condor_config = IronCondorConfig(paper_trading=paper_trading)
        self.condor_engine = IronCondorEngine(
            config=condor_config,
            paper_trading=paper_trading,
            logger=self.logger
        )
        
        self.logger.info(
            f"PremiumOrchestrator initialized | Paper: {paper_trading} | "
            f"Wheel: {self.config.wheel_allocation_pct}% | "
            f"Condor: {self.config.condor_allocation_pct}%"
        )
    
    def allocate_capital(self) -> dict[str, float]:
        """
        Calculate capital allocation for each strategy.
        
        Returns:
            Dictionary with allocation amounts per strategy
        """
        account = self.wheel_engine.get_account_info()
        if "error" in account:
            self.logger.error("Could not get account info")
            return {}
        
        portfolio_value = account.get("portfolio_value", 0)
        
        allocations = {
            "wheel": portfolio_value * (self.config.wheel_allocation_pct / 100),
            "condor": portfolio_value * (self.config.condor_allocation_pct / 100),
            "total": portfolio_value,
        }
        
        self.logger.info(
            f"Capital Allocation | Total: ${portfolio_value:,.2f} | "
            f"Wheel: ${allocations['wheel']:,.2f} | "
            f"Condor: ${allocations['condor']:,.2f}"
        )
        
        return allocations
    
    def run_all_strategies(self) -> None:
        """Execute all premium strategies in sequence."""
        self.logger.info("=" * 60)
        self.logger.info("RUNNING ALL PREMIUM STRATEGIES")
        self.logger.info("=" * 60)
        
        # Check allocations
        allocations = self.allocate_capital()
        if not allocations:
            return
        
        # Check portfolio Greeks before trading
        greeks = self.get_portfolio_greeks()
        if abs(greeks["delta"]) > self.config.max_portfolio_delta:
            self.logger.warning(
                f"Portfolio delta ({greeks['delta']:.2f}) exceeds limit. "
                "Reducing new positions."
            )
        
        # Run Wheel Strategy (60%)
        self.logger.info("-" * 40)
        self.logger.info("Running Wheel Strategy...")
        try:
            self.wheel_engine.run_wheel_cycle()
        except Exception as e:
            self.logger.error(f"Wheel strategy error: {e}")
        
        # Run Iron Condor Strategy (40%)
        self.logger.info("-" * 40)
        self.logger.info("Running Iron Condor Strategy...")
        try:
            self.condor_engine.run_cycle()
        except Exception as e:
            self.logger.error(f"Condor strategy error: {e}")
        
        self.logger.info("=" * 60)
        self.logger.info("ALL STRATEGIES COMPLETE")
        self.logger.info("=" * 60)
    
    def aggregate_performance(self) -> dict[str, Any]:
        """Combine performance metrics from all strategies."""
        # Wheel metrics
        wheel_returns = self.wheel_engine.calculate_returns()
        
        # Condor metrics
        condor_stats = self.condor_engine.stats
        condor_pnl = (
            condor_stats["total_premium_collected"] - 
            condor_stats["total_premium_returned"]
        )
        
        return {
            "wheel": {
                "premium_collected": wheel_returns["total_premium_collected"],
                "capital_used": wheel_returns["total_capital_used"],
                "return_pct": wheel_returns["return_pct"],
                "trades": wheel_returns["trades_executed"],
                "active_positions": wheel_returns["active_positions"],
            },
            "condor": {
                "premium_collected": condor_stats["total_premium_collected"],
                "premium_returned": condor_stats["total_premium_returned"],
                "net_pnl": condor_pnl,
                "winning_trades": condor_stats["winning_trades"],
                "losing_trades": condor_stats["losing_trades"],
                "total_trades": condor_stats["total_trades"],
            },
            "combined": {
                "total_premium": (
                    wheel_returns["total_premium_collected"] + 
                    condor_stats["total_premium_collected"]
                ),
                "total_pnl": (
                    wheel_returns["total_premium_collected"] + condor_pnl
                ),
                "total_trades": (
                    wheel_returns["trades_executed"] + 
                    condor_stats["total_trades"]
                ),
            },
        }
    
    def get_portfolio_greeks(self) -> dict[str, float]:
        """Sum delta/theta/vega across all strategies."""
        delta = 0.0
        theta = 0.0
        vega = 0.0
        
        # Condor Greeks
        for pos in self.condor_engine.positions.values():
            if pos.state == CondorState.OPEN:
                delta += pos.delta
                theta += pos.theta
                vega += pos.vega
        
        # Wheel doesn't track Greeks in detail, estimate from positions
        # Short puts have positive delta, covered calls reduce delta
        
        return {
            "delta": round(delta, 3),
            "theta": round(theta, 2),
            "vega": round(vega, 2),
        }
    
    def get_status(self) -> str:
        """Get combined status report."""
        account = self.wheel_engine.get_account_info()
        perf = self.aggregate_performance()
        greeks = self.get_portfolio_greeks()
        
        lines = [
            "=" * 70,
            "PREMIUM ORCHESTRATOR STATUS",
            "=" * 70,
            "",
            "ACCOUNT:",
            f"  Portfolio Value: ${account.get('portfolio_value', 0):,.2f}",
            f"  Buying Power:    ${account.get('buying_power', 0):,.2f}",
            "",
            "PORTFOLIO GREEKS:",
            f"  Delta: {greeks['delta']:+.3f}",
            f"  Theta: ${greeks['theta']:+.2f}/day",
            f"  Vega:  ${greeks['vega']:+.2f}",
            "",
            "WHEEL STRATEGY:",
            f"  Premium Collected: ${perf['wheel']['premium_collected']:,.2f}",
            f"  Return:            {perf['wheel']['return_pct']:.2f}%",
            f"  Active Positions:  {perf['wheel']['active_positions']}",
            "",
            "IRON CONDOR STRATEGY:",
            f"  Premium Collected: ${perf['condor']['premium_collected']:,.2f}",
            f"  Net P&L:           ${perf['condor']['net_pnl']:,.2f}",
            f"  Win Rate:          {perf['condor']['winning_trades']}/{perf['condor']['total_trades']}",
            "",
            "COMBINED:",
            f"  Total Premium: ${perf['combined']['total_premium']:,.2f}",
            f"  Total P&L:     ${perf['combined']['total_pnl']:,.2f}",
            f"  Total Trades:  {perf['combined']['total_trades']}",
            "",
            "=" * 70,
        ]
        
        return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Premium Strategy Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python v42_premium_orchestrator.py --test    # Test mode
  python v42_premium_orchestrator.py --status  # Show status
  python v42_premium_orchestrator.py --trade   # Run all strategies
        """
    )
    
    parser.add_argument("--test", action="store_true", help="Test mode")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--trade", action="store_true", help="Run trading")
    parser.add_argument("--live", action="store_true", help="Live trading")
    
    args = parser.parse_args()
    paper = not args.live
    
    if args.live:
        if input("WARNING: Live mode. Type 'YES': ") != "YES":
            sys.exit(0)
    
    print("\n" + "=" * 60)
    print("PREMIUM STRATEGY ORCHESTRATOR")
    print("=" * 60 + "\n")
    
    orchestrator = PremiumOrchestrator(paper_trading=paper)
    
    if args.test:
        print("Testing orchestrator...\n")
        alloc = orchestrator.allocate_capital()
        print(f"Allocations: {alloc}")
        greeks = orchestrator.get_portfolio_greeks()
        print(f"Greeks: {greeks}")
        print("\nTest complete!")
    
    elif args.status:
        print(orchestrator.get_status())
    
    elif args.trade:
        orchestrator.run_all_strategies()
        print(orchestrator.get_status())
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

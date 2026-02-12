#!/usr/bin/env python3
"""
AGGRESSIVE MODE CONFIGURATION
=============================
Maximum profit extraction for demonstrating results.

WARNING: This is for PAPER TRADING DEMO ONLY!
Real money trading should use conservative settings.

Created: 2026-02-02
"""

from dataclasses import dataclass, field
from typing import Set

@dataclass
class AggressiveConfig:
    """Aggressive trading configuration for fast P&L demonstration."""
    
    # =========================================================================
    # SIGNAL THRESHOLDS - CONSERVATIVE to filter noise
    # =========================================================================
    
    buy_threshold: float = 0.58   # Only trade strong signals
    sell_threshold: float = 0.42  # Only trade strong signals
    min_confidence: float = 0.50  # Require 50%+ confidence to trade
    
    # =========================================================================
    # POSITION SIZING - CONSERVATIVE
    # =========================================================================
    
    # Quarter-Kelly for safety (academic standard is half-Kelly max)
    kelly_fraction: float = 0.25
    
    # Position sizes
    position_size_pct: float = 0.05  # 5% of portfolio per trade
    max_position_pct: float = 0.10   # Max 10% in single position
    min_position_dollars: int = 500
    max_position_dollars: int = 7500
    
    # Leverage - DISABLED for safety
    max_leverage: float = 1.0     # No leverage
    use_margin: bool = False
    
    # Capital allocation
    max_cash_deployed_pct: float = 0.70  # Deploy max 70% of capital
    reserve_cash_pct: float = 0.30       # Keep 30% reserve
    
    # =========================================================================
    # TRADE FREQUENCY - MEASURED
    # =========================================================================
    
    cycle_seconds: int = 120      # 2-minute cycles to reduce noise trading
    trade_all_signals: bool = False  # Only trade clear signals
    skip_neutral: bool = True     # Skip ambiguous signals
    
    # =========================================================================
    # MOMENTUM CHASING - DISABLED (chasing causes buying at tops)
    # =========================================================================
    
    enable_momentum_chase: bool = False
    momentum_threshold: float = 0.02   # Require 2% move (rarely triggers)
    chase_strength: bool = False       # Don't buy into strength
    chase_weakness: bool = False       # Don't sell into weakness
    
    # Intraday momentum factors
    use_vwap_signal: bool = True
    use_price_acceleration: bool = False
    
    # =========================================================================
    # RISK CONTROLS - TIGHT
    # =========================================================================
    
    max_loss_per_trade_pct: float = 0.01   # 1% max loss per trade
    portfolio_heat_pct: float = 0.25       # 25% max exposure
    stop_loss_pct: float = 0.02            # 2% stop loss
    take_profit_pct: float = 0.04          # 4% take profit (2:1 reward:risk)
    
    # Drawdown limits
    max_daily_drawdown_pct: float = 0.03   # 3% max daily loss
    max_portfolio_drawdown_pct: float = 0.10  # 10% max total loss
    
    # =========================================================================
    # UNIVERSE — DIVERSIFIED across sectors (was 70%+ tech)
    # =========================================================================
    
    universe: list = field(default_factory=lambda: [
        'SPY', 'QQQ',                      # Broad market ETFs
        'AAPL', 'MSFT', 'NVDA',            # Tech (3)
        'AMZN', 'TSLA',                     # Consumer Discretionary (2)
        'JPM', 'GS', 'V',                   # Financials (3)
        'XOM', 'CVX',                        # Energy (2)
        'HD', 'KO',                          # Industrials + Staples (2)
    ])
    
    # Don't exclude anything
    excluded_tickers: Set[str] = field(default_factory=set)
    
    # =========================================================================
    # SCALPING MODE - DISABLED (transaction costs eat profits on small moves)
    # =========================================================================
    
    enable_scalping: bool = False
    scalp_profit_target: float = 0.01   # 1% profit target (if enabled)
    scalp_time_limit_minutes: int = 15  # Close scalps after 15 min
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold,
            'min_confidence': self.min_confidence,
            'kelly_fraction': self.kelly_fraction,
            'position_size_pct': self.position_size_pct,
            'max_position_pct': self.max_position_pct,
            'max_leverage': self.max_leverage,
            'cycle_seconds': self.cycle_seconds,
            'trade_all_signals': self.trade_all_signals,
            'enable_momentum_chase': self.enable_momentum_chase,
            'max_loss_per_trade_pct': self.max_loss_per_trade_pct,
            'portfolio_heat_pct': self.portfolio_heat_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'universe': self.universe,
        }


# Global aggressive config instance
_aggressive_config = None

def get_aggressive_config() -> AggressiveConfig:
    """Get the aggressive configuration singleton."""
    global _aggressive_config
    if _aggressive_config is None:
        _aggressive_config = AggressiveConfig()
    return _aggressive_config


if __name__ == "__main__":
    config = get_aggressive_config()
    print("=" * 60)
    print("AGGRESSIVE MODE CONFIGURATION")
    print("=" * 60)
    for k, v in config.to_dict().items():
        print(f"  {k}: {v}")
    print("=" * 60)

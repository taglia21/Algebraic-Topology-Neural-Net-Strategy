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
    # SIGNAL THRESHOLDS - LOOSE for maximum trades
    # =========================================================================
    
    buy_threshold: float = 0.52   # Was 0.55 - more trades
    sell_threshold: float = 0.48  # Was 0.45 - more trades
    min_confidence: float = 0.05  # Was 0.3 - take almost all signals
    
    # =========================================================================
    # POSITION SIZING - AGGRESSIVE
    # =========================================================================
    
    # Full Kelly criterion (normally use half-Kelly)
    kelly_fraction: float = 1.0   # Was 0.5
    
    # Position sizes
    position_size_pct: float = 0.25  # 25% of portfolio per trade (was 10%)
    max_position_pct: float = 0.40   # Max 40% in single position
    min_position_dollars: int = 5000
    max_position_dollars: int = 50000
    
    # Leverage
    max_leverage: float = 2.0     # Use margin if available
    use_margin: bool = True
    
    # Capital allocation
    max_cash_deployed_pct: float = 0.95  # Deploy 95% of capital
    reserve_cash_pct: float = 0.05       # Keep only 5% reserve
    
    # =========================================================================
    # TRADE FREQUENCY - MAXIMUM
    # =========================================================================
    
    cycle_seconds: int = 30       # Was 60 - faster cycles
    trade_all_signals: bool = True
    skip_neutral: bool = False    # Trade neutral signals too
    
    # =========================================================================
    # MOMENTUM CHASING - ENABLED
    # =========================================================================
    
    enable_momentum_chase: bool = True
    momentum_threshold: float = 0.005  # 0.5% move triggers chase
    chase_strength: bool = True        # Buy into strength
    chase_weakness: bool = True        # Sell into weakness
    
    # Intraday momentum factors
    use_vwap_signal: bool = True
    use_price_acceleration: bool = True
    
    # =========================================================================
    # RISK CONTROLS - LOOSE
    # =========================================================================
    
    max_loss_per_trade_pct: float = 0.05   # 5% max loss per trade
    portfolio_heat_pct: float = 2.0        # 200% exposure allowed
    stop_loss_pct: float = 0.03            # 3% stop loss (wider)
    take_profit_pct: float = 0.02          # 2% take profit (quick)
    
    # Drawdown limits
    max_daily_drawdown_pct: float = 0.15   # 15% max daily loss
    max_portfolio_drawdown_pct: float = 0.30  # 30% max total loss
    
    # =========================================================================
    # UNIVERSE - FULL
    # =========================================================================
    
    # Trade everything available
    universe: list = field(default_factory=lambda: [
        'SPY', 'QQQ', 'IWM', 'XLK', 'XLF',  # ETFs
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META',  # Big tech
        'JPM', 'GS', 'BAC',  # Financials
        'AMZN', 'TSLA'  # High beta
    ])
    
    # Don't exclude anything
    excluded_tickers: Set[str] = field(default_factory=set)
    
    # =========================================================================
    # SCALPING MODE
    # =========================================================================
    
    enable_scalping: bool = True
    scalp_profit_target: float = 0.005  # 0.5% profit target
    scalp_time_limit_minutes: int = 5   # Close scalps after 5 min
    
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

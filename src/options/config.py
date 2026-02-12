"""
Options Trading Configuration
==============================

Risk parameters and trading configuration for autonomous engine.

This module defines all risk limits, trading parameters, and strategy
configurations used by the autonomous trading system.
"""

from typing import Dict, Any

# ============================================================================
# RISK MANAGEMENT PARAMETERS
# ============================================================================

RISK_CONFIG: Dict[str, Any] = {
    # Portfolio-level risk limits
    "max_portfolio_delta": 50.0,  # Maximum net delta exposure (shares equivalent)
    "max_position_size_pct": 0.03,  # 3% max per position (was 5%)
    "max_daily_loss_pct": 0.02,  # 2% max daily drawdown (was 3%)
    "max_portfolio_heat": 0.08,  # 8% max total risk exposure (was 10%)
    
    # Position-level risk limits
    "max_risk_per_trade_pct": 0.02,  # 2% max risk per trade
    "max_contracts_per_trade": 5,  # Maximum contracts per order
    "max_positions": 15,  # Maximum concurrent positions
    
    # Time-based parameters
    "min_dte": 7,  # Minimum days to expiration
    "max_dte": 60,  # Maximum days to expiration
    "optimal_dte_min": 21,  # Optimal DTE range start
    "optimal_dte_max": 45,  # Optimal DTE range end
    
    # Profit/loss targets — options need wider stops due to gamma/IV swings
    "target_profit_pct": 0.50,  # Take profit at 50% of max gain
    "stop_loss_pct": 0.75,  # Stop loss at 75% loss (was 25% — fired on normal IV movement)
    "trailing_stop_pct": 0.50,  # Trailing stop at 50% (was 35%)
    
    # IV-based thresholds
    "iv_rank_sell_threshold": 65.0,  # Sell premium above this IV rank (was 50 - too aggressive)
    "iv_rank_buy_threshold": 25.0,  # Buy options below this IV rank
    "iv_rank_extreme_high": 80.0,  # Extremely high IV
    "iv_rank_extreme_low": 20.0,  # Extremely low IV
    
    # Strategy-specific parameters
    "min_probability_of_profit": 0.60,  # Minimum 60% PoP (was 50% — coin-flip too risky)
    "min_premium_credit": 0.30,  # Minimum $0.30 credit per contract
    "max_bid_ask_spread_pct": 0.15,  # Max 15% bid-ask spread
    
    # Mean reversion
    "z_score_entry": 2.0,  # Enter when z-score exceeds +/-2.0
    "z_score_exit": 0.5,  # Exit when z-score returns to +/-0.5
    "lookback_period": 252,  # 1 year lookback for z-score
    
    # Delta hedging
    "delta_hedge_threshold": 25.0,  # Hedge when portfolio delta > +/-25 shares equivalent
    "delta_rebalance_threshold": 10.0,  # Rebalance at +/-10 shares equivalent
    
    # Kelly Criterion
    "kelly_fraction": 0.25,  # Quarter-Kelly for safety
    "max_kelly_fraction": 0.50,  # Absolute maximum Kelly
    "min_kelly_fraction": 0.01,  # Minimum position size
    "kelly_max": 0.50,  # Alias for max_kelly_fraction
    "kelly_min": 0.01,  # Alias for min_kelly_fraction
    
    # Execution
    "order_timeout_seconds": 60,  # Order timeout
    "max_slippage_pct": 0.05,  # Max 5% slippage tolerance
    "retry_attempts": 3,  # Retry failed orders 3 times
    "retry_delay_seconds": 5,  # Wait 5s between retries
}


# ============================================================================
# STRATEGY WEIGHTS
# ============================================================================

STRATEGY_WEIGHTS: Dict[str, float] = {
    "iv_rank": 0.35,  # 35% weight to IV rank strategy
    "theta_decay": 0.30,  # 30% weight to theta strategy
    "mean_reversion": 0.20,  # 20% weight to mean reversion
    "delta_hedging": 0.15,  # 15% weight to delta hedging
}


# ============================================================================
# MARKET HOURS (Eastern Time)
# ============================================================================

MARKET_HOURS = {
    "market_open": "09:30",  # Market opens 9:30 AM ET
    "market_close": "16:00",  # Market closes 4:00 PM ET
    "pre_market_start": "04:00",  # Pre-market starts 4:00 AM ET
    "after_hours_end": "20:00",  # After-hours ends 8:00 PM ET
    "safe_entry_start": "09:45",  # Safe entry after 9:45 AM
    "safe_entry_end": "15:45",  # Safe entry before 3:45 PM
}


# ============================================================================
# VOLATILITY REGIMES
# ============================================================================

VOLATILITY_REGIMES = [
    {"name": "extreme_low", "min_iv_rank": 0, "max_iv_rank": 20, "position_size_multiplier": 1.5},
    {"name": "low", "min_iv_rank": 20, "max_iv_rank": 30, "position_size_multiplier": 1.2},
    {"name": "normal", "min_iv_rank": 30, "max_iv_rank": 70, "position_size_multiplier": 1.0},
    {"name": "high", "min_iv_rank": 70, "max_iv_rank": 80, "position_size_multiplier": 0.8},
    {"name": "extreme_high", "min_iv_rank": 80, "max_iv_rank": 100, "position_size_multiplier": 0.5},
]


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "log_dir": "logs",
    "log_file": "autonomous_trading_{date}.log",
}


# ============================================================================
# MONITORING INTERVALS
# ============================================================================

MONITORING_CONFIG = {
    "signal_scan_interval": 60,  # Scan for signals every 60 seconds
    "signal_scan_interval_seconds": 60,  # Alias for compatibility
    "position_check_interval": 30,  # Check positions every 30 seconds
    "risk_check_interval": 15,  # Check risk every 15 seconds
    "heartbeat_interval": 300,  # Log heartbeat every 5 minutes
    "regime_update_interval": 3600,  # Update regime every hour
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config(key: str, default: Any = None) -> Any:
    """
    Get configuration value.
    
    Args:
        key: Configuration key
        default: Default value if key not found
        
    Returns:
        Configuration value
    """
    return RISK_CONFIG.get(key, default)


def validate_config() -> bool:
    """
    Validate configuration parameters.
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    # Validate percentage values
    pct_keys = [
        "max_position_size_pct",
        "max_daily_loss_pct",
        "max_portfolio_heat",
        "max_risk_per_trade_pct",
        "target_profit_pct",
        "stop_loss_pct",
    ]
    
    for key in pct_keys:
        value = RISK_CONFIG.get(key, 0)
        if not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1, got {value}")
    
    # Validate DTE ranges
    if RISK_CONFIG["min_dte"] >= RISK_CONFIG["max_dte"]:
        raise ValueError("min_dte must be less than max_dte")
    
    # Validate Kelly fraction
    if not 0 < RISK_CONFIG["kelly_fraction"] <= 1:
        raise ValueError("kelly_fraction must be between 0 and 1")
    
    return True


# Validate on import
validate_config()

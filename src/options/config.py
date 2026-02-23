"""
Options Trading Configuration
==============================

Risk parameters and trading configuration for the autonomous options engine.

This module defines all risk limits, trading parameters, strategy
configurations, and monitoring intervals used by the autonomous trading
system.  Values are validated on import.

Attributes:
    RISK_CONFIG: Portfolio- and position-level risk limits.
    STRATEGY_WEIGHTS: Default strategy allocation weights.
    MARKET_HOURS: NYSE trading-session time boundaries.
    VOLATILITY_REGIMES: IV-rank bucketed position-size multipliers.
    LOGGING_CONFIG: Log format and rotation settings.
    MONITORING_CONFIG: Scan, check, and heartbeat intervals.
"""

from typing import Dict, Any

# ============================================================================
# RISK MANAGEMENT PARAMETERS
# ============================================================================

RISK_CONFIG: Dict[str, Any] = {
    # Portfolio-level risk limits
    "max_portfolio_delta": 500.0,  # Maximum net delta exposure (shares equiv; ~7% of $73K)
    "max_position_size_pct": 0.02,  # 2% max per position (was 3%) — FIX 7
    "max_daily_loss_pct": 0.02,  # 2% max daily drawdown (was 3%)
    "max_portfolio_heat": 0.08,  # 8% max total risk exposure (was 10%)
    
    # Position-level risk limits
    "max_risk_per_trade_pct": 0.02,  # 2% max risk per trade
    "max_contracts_per_trade": 2,  # Maximum contracts per order (was 3) — FIX 7
    "max_positions": 8,  # Maximum concurrent positions (was 15)

    # ===== EMERGENCY FIX 2026-02-23: PER-SYMBOL CONTRACT CAP =====
    "max_contracts_per_symbol": 5,  # HARD CAP: never hold > 5 contracts per underlying
    "max_underlying_concentration_pct": 0.15,  # Max 15% of portfolio in one underlying
    "daily_loss_halt_usd": -1500,  # Halt new entries when day P&L < -$1,500
    "daily_loss_emergency_usd": -3000,  # Emergency close ALL when day P&L < -$3,000
    "signal_dedup_hours": 4,  # Min hours between identical contract signals
    "position_age_loss_24h_pct": -0.40,  # Close if >24h old and >40% loss
    "position_age_loss_48h_pct": -0.20,  # Close if >48h old and >20% loss
    "min_bayesian_confidence": 0.65,  # Minimum Bayesian confidence for entry — FIX 6
    "max_kelly_fraction": 0.25,  # Quarter Kelly cap — FIX 7
    
    # Time-based parameters
    "min_dte": 7,  # Minimum days to expiration
    "max_dte": 60,  # Maximum days to expiration
    "optimal_dte_min": 21,  # Optimal DTE range start
    "optimal_dte_max": 45,  # Optimal DTE range end
    
    # Profit/loss targets — options need wider stops due to gamma/IV swings
    "target_profit_pct": 0.50,  # Take profit at 50% of max gain
    "stop_loss_pct": 0.50,  # Stop loss at 50% loss (was 75% — too loose for small acct)
    "trailing_stop_pct": 0.50,  # Trailing stop at 50% (was 35%)
    
    # IV-based thresholds — FIX 6: tightened
    "iv_rank_sell_threshold": 60.0,  # Sell premium above IVR 60 (was 65 — FIX 6)
    "iv_rank_buy_threshold": 25.0,  # Buy options below this IV rank
    "iv_rank_extreme_high": 80.0,  # Extremely high IV
    "iv_rank_extreme_low": 20.0,  # Extremely low IV
    
    # Strategy-specific parameters
    "min_probability_of_profit": 0.60,  # Minimum 60% PoP (was 50% — coin-flip too risky)
    "min_premium_credit": 0.50,  # Minimum $0.50 credit per contract (was $0.30)
    "max_bid_ask_spread_pct": 0.10,  # Max 10% bid-ask spread (was 15%)
    
    # Mean reversion
    "z_score_entry": 2.0,  # Enter when z-score exceeds +/-2.0
    "z_score_exit": 0.5,  # Exit when z-score returns to +/-0.5
    "lookback_period": 252,  # 1 year lookback for z-score
    "multi_tf_zscore_windows": [10, 20, 50],  # Multi-timeframe z-score windows
    
    # Delta hedging
    "delta_hedge_threshold": 25.0,  # Hedge when portfolio delta > +/-25 shares equivalent
    "delta_rebalance_threshold": 10.0,  # Rebalance at +/-10 shares equivalent
    
    # Volatility Risk Premium (VRP) — FIX 6: tightened from 3% to 5 vol pts
    "vrp_threshold": 0.05,  # 5% IV-RV spread to trigger VRP strategy (was 3%)
    
    # IV Crush strategy
    "iv_crush_min_rank": 80,  # Min IV rank for IV crush strategy
    "iv_crush_min_historical_drop": 0.20,  # 20% min historical IV drop
    
    # Theta/Gamma efficiency
    "theta_gamma_min_ratio": 0.5,  # Min theta/gamma ratio for theta decay signals
    
    # Signal convergence
    "signal_convergence_boost": True,  # Enable Bayesian confidence boosting
    
    # Position sizing (fixed-fractional — Kelly capped at 0.25)
    "fixed_risk_fraction": 0.01,  # 1% of portfolio risked per trade
    "max_single_position_pct": 0.02,  # Max 2% of portfolio in single position — FIX 7
    "min_entry_sharpe": 1.5,  # Minimum backtest Sharpe for entry — FIX 7
    
    # Execution
    "order_timeout_seconds": 60,  # Order timeout
    "max_slippage_pct": 0.05,  # Max 5% slippage tolerance
    "retry_attempts": 3,  # Retry failed orders 3 times
    "retry_delay_seconds": 5,  # Wait 5s between retries

    # ===== PHASE 6: EXIT MANAGEMENT =====
    "exit_profit_target_pct": 0.50,        # Close at 50% of max profit
    "exit_stop_loss_multiplier": 2.0,      # Close at 2x premium collected
    "exit_dte_threshold": 7,               # Close at 7 DTE remaining
    "exit_trailing_stop_activate": 0.30,   # Activate trailing stop at 30% gain
    "exit_trailing_stop_trail": 0.50,      # Trail 50% of peak profit
    "exit_time_accel_dte_pct": 0.50,       # Early exit after 50% time elapsed
    "exit_time_accel_profit_pct": 0.25,    # At 25% profit with time accel
    "exit_use_mleg_close": True,           # Use MLEG orders for closing spreads

    # ===== PHASE 6: GEX AWARENESS =====
    "gex_enabled": True,                    # Enable GEX-based filtering
    "gex_sticky_strike_threshold": 0.30,   # Top 30% GEX = sticky
    "gex_avoidance_radius_pct": 0.005,     # Avoid strikes within 0.5% of sticky
    "gex_cache_ttl_minutes": 15,           # Cache GEX for 15 min
    "gex_negative_size_reduction": 0.50,   # Reduce size 50% in neg GEX
}


# ============================================================================
# TRANSACTION COST MODEL
# ============================================================================

TRANSACTION_COSTS: Dict[str, float] = {
    "commission_per_contract": 0.65,        # $0.65 per contract (Alpaca/IBKR)
    "slippage_pct_of_mid": 0.15,           # 15% slippage from mid price
    "min_expected_edge_after_costs": 0.10,  # 10% minimum profit margin
}


# ============================================================================
# MINIMUM LIQUIDITY GATES
# ============================================================================

LIQUIDITY_GATES: Dict[str, Any] = {
    "min_avg_daily_volume": 1_000_000,     # 1M shares minimum
    "min_option_open_interest": 500,        # 500 contracts OI minimum
    "max_bid_ask_spread_abs": 0.10,        # $0.10 max absolute spread
    "max_bid_ask_spread_pct": 0.05,        # 5% of mid price max spread
}


# ============================================================================
# STRATEGY WEIGHTS
# ============================================================================

STRATEGY_WEIGHTS: Dict[str, float] = {
    "iv_rank": 0.20,  # 20% weight to IV rank strategy
    "theta_decay": 0.20,  # 20% weight to theta strategy
    "mean_reversion": 0.15,  # 15% weight to mean reversion
    "delta_hedging": 0.10,  # 10% weight to delta hedging
    "vrp": 0.25,  # 25% weight to VRP (top alpha source)
    "iv_crush": 0.10,  # 10% weight to IV crush
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
    "signal_scan_interval": 60,  # Scan for signals every 60 seconds (alpha decays in minutes)
    "signal_scan_interval_seconds": 60,  # Alias for compatibility
    "position_check_interval": 30,  # Check positions every 30 seconds
    "risk_check_interval": 15,  # Check risk every 15 seconds
    "heartbeat_interval": 300,  # Log heartbeat every 5 minutes
    "regime_update_interval": 3600,  # Update regime every hour
    # Phase 3 additions
    "greeks_log_interval": 1,  # Log Greeks every cycle
    "vix_cache_seconds": 300,  # Cache VIX for 5 min to avoid excess API calls
    "max_underlying_concentration": 0.15,  # Max 15% of options risk in one underlying — FIX 5
    "max_positions_per_underlying": 2,  # Max option positions per underlying symbol
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config(key: str, default: Any = None) -> Any:
    """Retrieve a single configuration value from ``RISK_CONFIG``.

    Args:
        key: Configuration key to look up.
        default: Fallback value when *key* is absent.

    Returns:
        The configuration value, or *default* if not found.
    """
    return RISK_CONFIG.get(key, default)


def validate_config() -> bool:
    """Validate critical configuration parameters on import.

    Checks that percentage values fall within [0, 1], DTE ranges
    are consistent, and the fixed-risk fraction is sensible.

    Returns:
        ``True`` if all checks pass.

    Raises:
        ValueError: If any parameter is out of range.
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
    
    # Validate fixed risk fraction
    if not 0 < RISK_CONFIG["fixed_risk_fraction"] <= 0.05:
        raise ValueError("fixed_risk_fraction must be between 0 and 0.05")
    
    return True


# Validate on import
validate_config()

"""
Constants for Options Trading System
====================================

Production configuration constants and defaults.
"""

from typing import Tuple
from datetime import time

# ============================================================================
# MARKET CONSTANTS
# ============================================================================

TRADING_DAYS_PER_YEAR = 252
MINUTES_PER_DAY = 390  # 6.5 hours * 60 minutes
SECONDS_PER_DAY = MINUTES_PER_DAY * 60

# Risk-free rate (use current T-bill rate)
DEFAULT_RISK_FREE_RATE = 0.05  # 5% annualized

# ============================================================================
# POSITION SIZING
# ============================================================================

# Maximum risk per trade as % of portfolio
MAX_POSITION_PCT = 0.02  # 2% max risk per position
MAX_POSITIONS = 6  # Maximum concurrent options positions

# Capital management
RESERVE_CAPITAL_PCT = 0.10  # Reserve 10% of capital
MARGIN_REQUIREMENT_PCT = 0.20  # 20% margin for naked options

# Kelly criterion settings
KELLY_FRACTION_BASE = 0.25  # Quarter-Kelly (conservative)
KELLY_FRACTION = 0.25  # Deprecated, use KELLY_FRACTION_BASE
MIN_KELLY_FRACTION = 0.01
MAX_KELLY_FRACTION = 0.50  # Absolute maximum

# ============================================================================
# THETA DECAY PARAMETERS
# ============================================================================

# Theta accelerates exponentially in final 21 days
THETA_ACCELERATION_DTE = 21

# Optimal DTE ranges for different strategies
WHEEL_DTE_RANGE = (30, 45)  # Sell cash-secured puts
SPREAD_DTE_RANGE = (30, 45)  # Credit spreads
IRON_CONDOR_DTE_RANGE = (35, 50)  # Iron condors
IC_DTE_RANGE = IRON_CONDOR_DTE_RANGE  # Deprecated alias

# Exit thresholds - CRITICAL RISK PARAMETERS
PROFIT_TARGET_PCT = 0.50  # Take profit at 50% of max gain
PROFIT_TARGET_PERCENT = 50.0  # Same as above, percentage form
STOP_LOSS_PERCENT = 25.0  # Exit if loss exceeds 25% - SAFE (was 100% - INSANE!)
STOP_LOSS_MULTIPLIER = 2.0  # Stop loss at 2x credit received (for spreads)
TIME_EXIT_DTE = 21  # Exit at 21 DTE regardless of P&L

# ============================================================================
# IV ANALYSIS
# ============================================================================

# IV Rank/Percentile thresholds
IV_HIGH_THRESHOLD = 70.0  # >70 = sell premium
IV_MEDIUM_THRESHOLD = 30.0  # 30-70 = spreads
IV_LOW_THRESHOLD = 30.0  # <30 = buy premium or skip

# IV lookback period
IV_LOOKBACK_DAYS = 252  # 1 year of IV history

# HV/IV ratio thresholds
HVIV_OVERPRICED_THRESHOLD = 0.8  # IV > HV, sell premium
HVIV_UNDERPRICED_THRESHOLD = 1.2  # IV < HV, buy premium

# ============================================================================
# GREEKS LIMITS (per $100K capital)
# ============================================================================

# Delta limits (directional exposure)
MAX_PORTFOLIO_DELTA_PER_100K = 20.0  # ±$2,000 per $1 move
MAX_POSITION_DELTA = 0.40  # Max 0.40 delta per position

# Gamma limits (acceleration risk)
MAX_PORTFOLIO_GAMMA_PER_100K = 5.0

# Theta targets (positive = earning from decay)
TARGET_PORTFOLIO_THETA_PER_100K = 50.0  # $50/day goal
MIN_PORTFOLIO_THETA_PER_100K = 30.0  # Minimum theta target

# Vega limits (volatility exposure)
MAX_PORTFOLIO_VEGA_PER_100K = -100.0  # Negative (short vol)

# ============================================================================
# STRIKE SELECTION
# ============================================================================

# Delta targets
DELTA_TARGET = 0.30  # Typical target for CSP
WHEEL_DELTA_TARGET = 0.30  # Wheel strategy delta target
WHEEL_PUT_DELTA_RANGE = (0.20, 0.30)  # Sell puts at 20-30 delta
WHEEL_CALL_DELTA_RANGE = (0.25, 0.35)  # Sell calls at 25-35 delta
SPREAD_DELTA_TARGET = 0.30  # Typical target for credit spreads
SPREAD_SHORT_DELTA_RANGE = (0.20, 0.30)
SPREAD_LONG_DELTA_OFFSET = 5  # Points between short and long strike
IRON_CONDOR_WING_DELTA = 0.16  # 16 delta for iron condor wings (~1 std dev)
IC_WING_DELTA = IRON_CONDOR_WING_DELTA  # Deprecated alias

# ============================================================================
# 15-MINUTE DELAY COMPENSATION
# ============================================================================

# Delay in minutes (Tradier sandbox)
PRICE_DELAY_MINUTES = 15

# Safety buffers (standard deviations)
CREDIT_ENTRY_BUFFER_SIGMA = 1.5  # Conservative for selling premium
DEBIT_ENTRY_BUFFER_SIGMA = 2.0  # Extra conservative for buying
CREDIT_ENTRY_BUFFER = CREDIT_ENTRY_BUFFER_SIGMA  # Deprecated alias
DEBIT_ENTRY_BUFFER = DEBIT_ENTRY_BUFFER_SIGMA  # Deprecated alias

# Safe trading windows (Eastern Time)
SAFE_TRADING_WINDOWS: list[Tuple[time, time]] = [
    (time(10, 0), time(11, 30)),  # Morning session
    (time(13, 0), time(15, 0)),   # Afternoon session
]

# Avoid trading windows
AVOID_FIRST_MINUTES = 30  # After market open
AVOID_LAST_MINUTES = 30  # Before market close

# VIX thresholds
MAX_VIX_FOR_ENTRY = 35.0  # Don't enter new positions if VIX > 35
MAX_VIX_FOR_DELAYED_ENTRY = MAX_VIX_FOR_ENTRY  # Deprecated alias

# Minimum DTE for delayed data trading
MIN_DTE_FOR_DELAYED_DATA = 14  # Avoid very short DTE with stale data

# ============================================================================
# ORDER EXECUTION
# ============================================================================

# Order timeouts
ORDER_TIMEOUT_SECONDS = 60
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2  # Initial backoff, doubles each retry
RETRY_DELAY_SECONDS = RETRY_BACKOFF_SECONDS  # Deprecated alias

# Slippage estimates
EXPECTED_SLIPPAGE_PCT = 0.02  # 2% slippage on options
MAX_ACCEPTABLE_SLIPPAGE_PCT = 0.05  # 5% max

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

# Maximum daily loss (circuit breaker)
MAX_DAILY_LOSS_PCT = 0.05  # 5% of portfolio

# Maximum drawdown before stopping
MAX_DRAWDOWN_PCT = 0.15  # 15% max drawdown

# Correlation limits
MAX_CORRELATED_POSITIONS = 3  # Max positions in same sector

# ============================================================================
# BACKTESTING
# ============================================================================

BACKTEST_INITIAL_CAPITAL = 100000  # $100K starting capital
BACKTEST_COMMISSION_PER_CONTRACT = 0.65  # Per contract per side
BACKTEST_MIN_LIQUIDITY = 100  # Minimum daily volume

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# DATA PERSISTENCE
# ============================================================================

STATE_FILE = "options_engine_state.json"
TRADES_FILE = "options_trades.json"
METRICS_FILE = "options_metrics.json"
IV_HISTORY_FILE = "iv_history.json"

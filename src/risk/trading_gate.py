"""
Centralized Trading Gate
========================

A simple, importable module that ALL trading scripts should call before
placing any trade. Wraps the V26 circuit breaker system with Alpaca
portfolio data fetching.

Usage in any trader:
    from src.risk.trading_gate import check_trading_allowed, get_safe_position_size
    
    # Before any trade:
    allowed, reason = check_trading_allowed()
    if not allowed:
        logger.warning(f"Trade blocked: {reason}")
        return
    
    # For position sizing:
    size_pct = get_safe_position_size(confidence=0.65, volatility=0.02)
"""

import os
import logging
from typing import Tuple, Optional
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)

# Lazy-load the circuit breaker to avoid import-time failures
_breaker = None
_last_reset_date = None


def _get_breaker():
    """Lazy-initialize the V26 circuit breaker singleton."""
    global _breaker
    if _breaker is None:
        try:
            from src.risk.circuit_breakers import V26CircuitBreakers, V26CircuitBreakerConfig
            config = V26CircuitBreakerConfig(
                level_1_threshold=0.01,    # -1% daily loss → reduce 50%
                level_2_threshold=0.015,   # -1.5% → high confidence only
                level_3_threshold=0.02,    # -2% → halt all trading
                max_drawdown_pct=0.10,     # 10% max drawdown from peak
                warning_drawdown_pct=0.06, # 6% warning
                kelly_fraction=0.25,
                max_position_pct=0.05,
            )
            _breaker = V26CircuitBreakers(config)
            logger.info("Trading gate initialized with V26 circuit breakers")
        except Exception as e:
            logger.error(f"Failed to initialize circuit breakers: {e}")
            _breaker = None
    return _breaker


def _get_portfolio_value() -> Optional[float]:
    """Fetch current portfolio equity from Alpaca."""
    try:
        from alpaca.trading.client import TradingClient
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_SECRET_KEY')
        paper = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'
        
        if not api_key or not api_secret:
            logger.warning("Alpaca credentials not set, cannot fetch portfolio value")
            return None
        
        client = TradingClient(api_key, api_secret, paper=paper)
        account = client.get_account()
        return float(account.equity)
    except Exception as e:
        logger.error(f"Failed to fetch portfolio value: {e}")
        return None


def update_breaker_state() -> bool:
    """
    Update circuit breaker with current portfolio value.
    Call this at startup and periodically during trading.
    
    Returns:
        True if update succeeded
    """
    global _last_reset_date
    
    breaker = _get_breaker()
    if breaker is None:
        return False
    
    equity = _get_portfolio_value()
    if equity is None:
        return False
    
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).date()
    
    # Reset daily if new trading day
    if _last_reset_date != today:
        breaker.reset_daily(equity)
        _last_reset_date = today
        logger.info(f"Circuit breaker daily reset: equity=${equity:,.2f}")
    
    # Update with current equity
    state = breaker.update(equity)
    
    if not state.can_trade:
        logger.warning(f"⚠️ CIRCUIT BREAKER: {state.message} | "
                       f"Level: {state.level.value} | "
                       f"Daily loss: {state.daily_loss_pct:.2%}")
    
    return True


def check_trading_allowed(signal_confidence: float = 0.5) -> Tuple[bool, str]:
    """
    Check if trading is allowed based on circuit breakers.
    
    Args:
        signal_confidence: Confidence of the signal (0-1)
        
    Returns:
        Tuple of (allowed, reason)
    """
    breaker = _get_breaker()
    if breaker is None:
        # If circuit breakers fail to load, BLOCK trading (fail-safe)
        return False, "Circuit breaker system not initialized"
    
    # Update state first
    update_breaker_state()
    
    # Check if trading allowed at all
    can_trade, reason = breaker.can_trade()
    if not can_trade:
        return False, reason
    
    # Check signal-level filter
    should_trade, reason = breaker.should_trade_signal(signal_confidence)
    if not should_trade:
        return False, reason
    
    return True, "Trading allowed"


def get_safe_position_size(
    confidence: float,
    volatility: float = 0.02,
    base_size: Optional[float] = None
) -> float:
    """
    Get risk-adjusted position size as fraction of portfolio.
    
    Args:
        confidence: Signal confidence (0-1)
        volatility: Asset volatility (annualized)
        base_size: Optional base size override
        
    Returns:
        Position size as fraction of portfolio (0.0 to max_position_pct)
    """
    breaker = _get_breaker()
    if breaker is None:
        return 0.0  # No breaker = no trading
    
    return breaker.get_position_size(
        signal_confidence=confidence,
        volatility=volatility,
        base_size=base_size
    )


def get_max_daily_loss_remaining() -> Optional[float]:
    """
    Get how much more the portfolio can lose today before circuit breakers trigger.
    
    Returns:
        Remaining loss budget as percentage (e.g. 0.015 = 1.5%), or None if unknown
    """
    breaker = _get_breaker()
    if breaker is None:
        return None
    
    update_breaker_state()
    
    # How much of the daily budget is left before Level 3 halt?
    remaining = breaker.config.level_3_threshold - breaker.daily_loss_pct
    return max(0.0, remaining)

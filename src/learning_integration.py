#!/usr/bin/env python3
"""
Learning Integration - Hooks the Adaptive Learning Engine into the trading bot
==============================================================================
This module provides the glue code to integrate continuous learning with live trading.
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adaptive_learning_engine import (
    get_learning_engine, 
    TradeRecord, 
    AdaptiveLearningEngine
)

logger = logging.getLogger(__name__)


class TradingBotLearningIntegration:
    """
    Integrates the Adaptive Learning Engine with the live trading bot.
    Call this after every trade to enable continuous learning.
    """
    
    def __init__(self):
        self.engine = get_learning_engine()
        self.trade_count = 0
        logger.info("Learning Integration initialized")
        
    def on_trade_closed(self, 
                        trade_id: str,
                        symbol: str,
                        side: str,
                        entry_price: float,
                        exit_price: float,
                        quantity: int,
                        entry_time: datetime,
                        exit_time: datetime,
                        pnl: float,
                        signal_strength: float = 0.5,
                        predicted_direction: int = 0,
                        features: Optional[Dict[str, float]] = None,
                        market_regime: str = 'unknown') -> Dict[str, Any]:
        """
        Called when a trade is closed. Triggers learning.
        
        Parameters:
        -----------
        trade_id: Unique identifier for the trade
        symbol: Stock ticker symbol
        side: 'buy' or 'sell'
        entry_price: Price at which position was entered
        exit_price: Price at which position was closed
        quantity: Number of shares
        entry_time: When the trade was opened
        exit_time: When the trade was closed
        pnl: Profit/loss in dollars
        signal_strength: Confidence of the signal (0-1)
        predicted_direction: -1 (bearish), 0 (neutral), 1 (bullish)
        features: Dictionary of features used for the prediction
        market_regime: Current market regime
        
        Returns:
        --------
        Dict with learning results
        """
        # Calculate derived values
        pnl_percent = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
        if side == 'sell':  # Short trade
            pnl_percent = -pnl_percent
        
        hold_duration = int((exit_time - entry_time).total_seconds() / 60)
        
        # Determine actual direction
        if exit_price > entry_price:
            actual_direction = 1
        elif exit_price < entry_price:
            actual_direction = -1
        else:
            actual_direction = 0
        
        # Create trade record
        trade = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            entry_time=entry_time,
            exit_time=exit_time,
            pnl=pnl,
            pnl_percent=pnl_percent,
            hold_duration_minutes=hold_duration,
            signal_strength=signal_strength,
            predicted_direction=predicted_direction,
            actual_direction=actual_direction,
            features=features or {},
            market_regime=market_regime
        )
        
        # Record trade and trigger learning
        result = self.engine.record_trade(trade)
        self.trade_count += 1
        
        logger.info(f"Trade {trade_id} recorded. Learning updates: {result['learning_updates']}")
        
        return result
    
    def get_prediction(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Get prediction from the online learning model.
        Use this for real-time trading decisions.
        """
        return self.engine.get_prediction(features)
    
    def get_optimized_params(self) -> Dict[str, float]:
        """
        Get the current best trading parameters from evolution.
        Apply these to your trading logic.
        """
        return self.engine.get_active_params()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the learning system.
        """
        status = self.engine.get_status()
        status['integration'] = {
            'trades_processed': self.trade_count,
            'last_check': datetime.now().isoformat()
        }
        return status
    
    def should_trade(self, symbol: str, direction: int, confidence: float) -> bool:
        """
        Determine if a trade should be taken based on learning insights.
        
        Parameters:
        -----------
        symbol: Stock to trade
        direction: -1 (short), 0 (hold), 1 (long)
        confidence: Model confidence (0-1)
        
        Returns:
        --------
        bool: True if trade should be taken
        """
        params = self.get_optimized_params()
        threshold = params.get('signal_threshold', 0.3)
        
        # Check if confidence meets threshold
        if confidence < threshold:
            return False
        
        # Check if drift detected (might want to be more conservative)
        if self.engine.drift_detector.drift_detected:
            threshold *= 1.2  # Require higher confidence during drift
            if confidence < threshold:
                logger.warning(f"Trade rejected due to drift: {symbol}, conf={confidence:.2f}")
                return False
        
        return True
    
    def get_position_size(self, symbol: str, account_value: float) -> float:
        """
        Get recommended position size based on evolved parameters.
        
        Parameters:
        -----------
        symbol: Stock to trade
        account_value: Current account value
        
        Returns:
        --------
        float: Recommended position size in dollars
        """
        params = self.get_optimized_params()
        position_pct = params.get('position_size_pct', 0.05)
        
        # Reduce position during drift
        if self.engine.drift_detector.drift_detected:
            position_pct *= 0.5
            logger.warning("Reducing position size due to drift detection")
        
        return account_value * position_pct
    
    def get_stop_loss(self, entry_price: float) -> float:
        """
        Get evolved stop loss price.
        """
        params = self.get_optimized_params()
        stop_loss_pct = params.get('stop_loss', 0.02)
        return entry_price * (1 - stop_loss_pct)
    
    def get_take_profit(self, entry_price: float) -> float:
        """
        Get evolved take profit price.
        """
        params = self.get_optimized_params()
        profit_target_pct = params.get('profit_target', 0.015)
        return entry_price * (1 + profit_target_pct)

    def get_stats(self) -> dict:
        """Get learning statistics."""
        return {
            "trades_processed": len(self.trade_history),
            "learning_rate": self.learning_rate,
            "online_model_ready": self.online_model is not None,
            "last_updated": str(self.last_updated) if self.last_updated else None
        }

    def get_stats(self) -> dict:
        """Get learning statistics."""
        return {
            'trades_processed': len(self.trade_history) if hasattr(self, 'trade_history') else 0,
            'learning_rate': getattr(self, 'learning_rate', 0.01),
            'online_model_ready': hasattr(self, 'online_model') and self.online_model is not None
        }

# Singleton instance
_integration: Optional[TradingBotLearningIntegration] = None

def get_integration() -> TradingBotLearningIntegration:
    """Get or create the global integration instance"""
    global _integration
    if _integration is None:
        _integration = TradingBotLearningIntegration()
    return _integration


# Convenience functions for direct import
def on_trade_closed(**kwargs) -> Dict[str, Any]:
    """Record a closed trade and trigger learning"""
    return get_integration().on_trade_closed(**kwargs)

def get_prediction(features: Dict[str, float]) -> Dict[str, Any]:
    """Get prediction from learning model"""
    return get_integration().get_prediction(features)

def get_optimized_params() -> Dict[str, float]:
    """Get current best trading parameters"""
    return get_integration().get_optimized_params()

def get_learning_status() -> Dict[str, Any]:
    """Get learning system status"""
    return get_integration().get_status()


if __name__ == '__main__':
    import json
    
    # Test the integration
    integration = get_integration()
    
    # Simulate a trade
    result = integration.on_trade_closed(
        trade_id='test_001',
        symbol='AAPL',
        side='buy',
        entry_price=150.00,
        exit_price=152.25,
        quantity=10,
        entry_time=datetime(2026, 2, 2, 9, 30),
        exit_time=datetime(2026, 2, 2, 10, 45),
        pnl=22.50,
        signal_strength=0.65,
        predicted_direction=1,
        features={'momentum': 0.8, 'volatility': 0.2, 'rsi': 55}
    )
    
    print("Trade recorded successfully!")
    print(f"Learning updates: {result['learning_updates']}")
    print(f"Reward: {result.get('reward', 'N/A')}")
    print("\nCurrent optimized params:")
    print(json.dumps(get_optimized_params(), indent=2))
    print("\nLearning status:")
    status = get_learning_status()
    print(f"Total trades: {status['total_trades_recorded']}")
    print(f"Model fitted: {status['model_state']['is_fitted']}")

# ============================================
# Continuous Learning Helper Functions
# ============================================

_continuous_learner_instance = None

def get_continuous_learner():
    """Get or create the global continuous learner instance."""
    global _continuous_learner_instance
    if _continuous_learner_instance is None:
        _continuous_learner_instance = get_integration()
    return _continuous_learner_instance


def initialize_continuous_learning():
    """Initialize the continuous learning system."""
    learner = get_continuous_learner()
    # Initialize any additional setup here
    return learner


class ContinuousLearnerWrapper:
    """Wrapper class providing consistent interface for continuous learning."""
    
    def __init__(self):
        self.integration = get_integration()
    
    def on_trade_closed(self, symbol: str, trade_result: dict, features: dict = None):
        """Process a closed trade for learning."""
        return on_trade_closed(
            trade_result=trade_result,
            features=features or {}
        )
    
    def get_stats(self) -> dict:
        """Get learning statistics."""
        return get_learning_status()
    
    def get_prediction(self, features: dict) -> dict:
        """Get model prediction."""
        return get_prediction(features=features)


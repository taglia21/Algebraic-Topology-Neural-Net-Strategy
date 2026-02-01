#!/usr/bin/env python3
"""
Integrated Trading System
=========================
Combines TDA (Topological Data Analysis), Neural Networks, and Options Alpha Engine
for a complete trading solution with Team of Rivals consensus mechanism.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import json
import os

# Import our modules
from src.tda_strategy import TDAStrategy
from src.v50_options_alpha_engine import V50OptionsAlphaEngine, OptionsSignal
from src.risk.risk_manager import RiskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedTradingSystem:
    """
    Main trading system that integrates:
    - TDA-based market regime detection
    - Neural Network signal generation  
    - Options Alpha Engine for trade execution
    - Risk Manager for position controls
    - Team of Rivals consensus mechanism
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Initialize components
        self.tda_strategy = TDAStrategy(
            lookback_window=self.config.get('tda_lookback', 60),
            prediction_horizon=self.config.get('prediction_horizon', 5)
        )
        
        self.options_engine = V50OptionsAlphaEngine(
            paper_trading=self.config.get('paper_trading', True),
            max_position_pct=self.config.get('max_position_pct', 0.05),
            min_confidence=self.config.get('min_confidence', 0.6)
        )
        
        self.risk_manager = RiskManager(
            max_position_size=self.config.get('max_position_size', 0.05),
            max_portfolio_risk=self.config.get('max_portfolio_risk', 0.15),
            max_daily_loss=self.config.get('max_daily_loss', 0.02),
            max_correlation=self.config.get('max_correlation', 0.7)
        )
        
        # State tracking
        self.portfolio_value = self.config.get('initial_capital', 100000)
        self.positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        self.daily_pnl = 0.0
        self.is_halted = False
        
        logger.info("Integrated Trading System initialized")
    
    def _default_config(self) -> Dict:
        return {
            'paper_trading': True,
            'initial_capital': 100000,
            'max_position_pct': 0.05,
            'max_portfolio_risk': 0.15,
            'max_daily_loss': 0.02,
            'max_correlation': 0.7,
            'min_confidence': 0.6,
            'tda_lookback': 60,
            'prediction_horizon': 5,
            'universe': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AMD']
        }
    
    def analyze_symbol(self, symbol: str, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run full analysis on a symbol using TDA and NN.
        
        Returns:
            Dict with TDA analysis, NN prediction, and combined signal
        """
        try:
            # Run TDA analysis
            tda_result = self.tda_strategy.analyze(price_data)
            
            # Get NN prediction
            nn_prediction = self.tda_strategy.predict(price_data)
            
            # Combine signals
            combined = self._combine_signals(tda_result, nn_prediction)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'tda_analysis': tda_result,
                'nn_prediction': nn_prediction,
                'combined_signal': combined,
                'success': True
            }
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'error': str(e),
                'success': False
            }
    
    def _combine_signals(self, tda_result: Dict, nn_prediction: float) -> Dict:
        """Combine TDA and NN signals using Team of Rivals consensus."""
        # Extract TDA metrics
        persistence = tda_result.get('persistence_score', 0.5)
        regime = tda_result.get('regime', 'neutral')
        trend_strength = tda_result.get('trend_strength', 0.0)
        
        # Weight the signals (can be tuned)
        tda_weight = 0.4
        nn_weight = 0.6
        
        # Convert regime to numeric
        regime_signal = {'bullish': 0.5, 'bearish': -0.5, 'neutral': 0.0}.get(regime, 0.0)
        
        # Combined signal
        tda_signal = (persistence - 0.5) * 2 + regime_signal * 0.5 + trend_strength * 0.3
        combined = tda_weight * tda_signal + nn_weight * nn_prediction
        
        # Determine direction
        if combined > 0.2:
            direction = 'bullish'
        elif combined < -0.2:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        # Calculate confidence
        confidence = min(1.0, 0.3 + abs(combined) * 0.7)
        
        return {
            'signal': combined,
            'direction': direction,
            'confidence': confidence,
            'tda_contribution': tda_signal,
            'nn_contribution': nn_prediction
        }

    
    def generate_trade_signal(self, symbol: str, price_data: pd.DataFrame,
                               current_price: float) -> Optional[OptionsSignal]:
        """
        Generate a trade signal for a symbol.
        
        Returns:
            OptionsSignal if conditions are met, None otherwise
        """
        # Check if trading is halted
        if self.is_halted:
            logger.warning("Trading is halted - no new signals")
            return None
        
        # Check risk manager
        can_trade, reason = self.risk_manager.can_open_position(
            symbol=symbol,
            position_value=current_price * 100,  # Assuming 100 shares equivalent
            portfolio_value=self.portfolio_value,
            existing_positions=self.positions
        )
        
        if not can_trade:
            logger.info(f"Risk check failed for {symbol}: {reason}")
            return None
        
        # Analyze symbol
        analysis = self.analyze_symbol(symbol, price_data)
        if not analysis.get('success', False):
            return None
        
        combined = analysis['combined_signal']
        
        # Generate options signal
        tda_signal = {
            'persistence_score': analysis['tda_analysis'].get('persistence_score', 0.5),
            'trend_strength': analysis['tda_analysis'].get('trend_strength', 0.0),
            'regime': combined['direction']
        }
        
        signal = self.options_engine.generate_signal(
            symbol=symbol,
            underlying_price=current_price,
            tda_signal=tda_signal,
            nn_prediction=analysis['nn_prediction']
        )
        
        return signal
    
    def execute_signal(self, signal: OptionsSignal, dry_run: bool = True) -> Dict:
        """Execute a trading signal (paper or live)."""
        execution_result = {
            'signal': signal.to_dict(),
            'executed': False,
            'dry_run': dry_run,
            'timestamp': datetime.now().isoformat()
        }
        
        if dry_run or self.config.get('paper_trading', True):
            # Paper trade execution
            logger.info(f"PAPER TRADE: {signal.symbol} {signal.strategy.value}")
            execution_result['executed'] = True
            execution_result['execution_type'] = 'paper'
            
            # Track the position
            self.positions[signal.symbol] = {
                'strategy': signal.strategy.value,
                'direction': signal.direction,
                'entry_time': datetime.now(),
                'confidence': signal.confidence
            }
            
            self.trade_history.append(execution_result)
        else:
            # Live trade would go here
            logger.warning("Live trading not implemented - use Alpaca API")
            execution_result['execution_type'] = 'blocked'
        
        return execution_result
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        options_status = self.options_engine.get_status()
        risk_status = self.risk_manager.get_status()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.daily_pnl,
            'is_halted': self.is_halted,
            'active_positions': len(self.positions),
            'positions': self.positions,
            'total_trades': len(self.trade_history),
            'options_engine': options_status,
            'risk_manager': risk_status,
            'config': {
                'paper_trading': self.config.get('paper_trading', True),
                'universe': self.config.get('universe', [])
            }
        }
    
    def reset_daily_stats(self):
        """Reset daily statistics (call at market open)."""
        self.daily_pnl = 0.0
        self.risk_manager.reset_daily_stats()
        logger.info("Daily stats reset")



async def run_trading_loop(system: IntegratedTradingSystem, interval_minutes: int = 5):
    """Main trading loop that runs continuously."""
    import yfinance as yf
    
    logger.info(f"Starting trading loop (interval: {interval_minutes} min)")
    
    while True:
        try:
            universe = system.config.get('universe', ['SPY'])
            
            for symbol in universe:
                # Fetch latest data
                ticker = yf.Ticker(symbol)
                df = ticker.history(period='3mo', interval='1d')
                
                if df.empty:
                    continue
                
                current_price = df['Close'].iloc[-1]
                
                # Generate signal
                signal = system.generate_trade_signal(symbol, df, current_price)
                
                if signal:
                    # Execute paper trade
                    result = system.execute_signal(signal, dry_run=True)
                    logger.info(f"Trade executed: {result}")
            
            # Log status
            status = system.get_system_status()
            logger.info(f"System status: {json.dumps(status, default=str, indent=2)}")
            
            # Wait for next iteration
            await asyncio.sleep(interval_minutes * 60)
            
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            await asyncio.sleep(60)  # Wait 1 minute on error


def demo_integrated_system():
    """Demo the integrated trading system."""
    import yfinance as yf
    
    print("=" * 60)
    print("INTEGRATED TRADING SYSTEM - DEMO")
    print("=" * 60)
    
    # Initialize system
    config = {
        'paper_trading': True,
        'initial_capital': 100000,
        'universe': ['SPY', 'QQQ', 'AAPL']
    }
    
    system = IntegratedTradingSystem(config)
    
    # Test with SPY
    print("\n[Fetching SPY data...]")
    spy = yf.Ticker('SPY')
    df = spy.history(period='3mo', interval='1d')
    
    if not df.empty:
        current_price = df['Close'].iloc[-1]
        print(f"SPY Current Price: ${current_price:.2f}")
        
        # Generate signal
        print("\n[Generating trade signal...]")
        signal = system.generate_trade_signal('SPY', df, current_price)
        
        if signal:
            print(f"Signal Generated:")
            print(f"  Strategy: {signal.strategy.value}")
            print(f"  Direction: {signal.direction}")
            print(f"  Confidence: {signal.confidence:.2%}")
            print(f"  IV Percentile: {signal.iv_percentile:.1f}")
            
            # Execute
            result = system.execute_signal(signal)
            print(f"\nExecution: {result['execution_type']}")
        else:
            print("No signal generated (below confidence threshold)")
    
    # Get status
    print("\n[System Status]")
    status = system.get_system_status()
    print(json.dumps(status, indent=2, default=str))
    
    print("\n" + "=" * 60)
    print("INTEGRATED SYSTEM READY")
    print("=" * 60)
    
    return system


if __name__ == "__main__":
    demo_integrated_system()


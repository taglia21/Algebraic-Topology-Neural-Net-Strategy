#!/usr/bin/env python3
"""
V23 Position Sizer
===================
Fractional Kelly position sizing with dynamic adjustments.

Features:
- Kelly criterion calculation
- Drawdown-responsive sizing
- Regime-based adjustments (VIX proxy)
- Position limits and risk controls
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('V23_PositionSizer')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SizingConfig:
    """Position sizing configuration."""
    
    # Kelly parameters
    kelly_fraction: float = 0.5  # Half-Kelly (conservative)
    max_position_pct: float = 0.10  # 10% max per position
    min_position_pct: float = 0.02  # 2% min per position
    
    # Position limits
    max_positions: int = 30
    max_sector_weight: float = 0.25
    max_single_stock_weight: float = 0.10
    
    # Drawdown adjustments
    dd_reduce_threshold: float = -0.10  # Start reducing at -10% DD
    dd_halt_threshold: float = -0.15  # Halt new positions at -15% DD
    dd_emergency_threshold: float = -0.20  # Emergency sizing at -20% DD
    
    # Drawdown multipliers
    dd_reduce_mult: float = 0.50  # 50% of normal size
    dd_emergency_mult: float = 0.25  # 25% of normal size
    
    # Regime adjustments (VIX-based)
    bull_threshold: float = 18.0  # VIX < 18
    bear_threshold: float = 30.0  # VIX > 30
    bull_mult: float = 1.10  # 10% larger in bull
    bear_mult: float = 0.60  # 40% smaller in bear
    
    # Volatility scaling
    target_volatility: float = 0.25  # 25% annual vol target
    vol_scale_min: float = 0.5
    vol_scale_max: float = 2.0


# =============================================================================
# KELLY CALCULATOR
# =============================================================================

class KellyCalculator:
    """
    Kelly Criterion position sizing calculator.
    
    Kelly formula: f* = (p * b - q) / b
    where:
        f* = optimal fraction of capital
        p = probability of winning
        q = probability of losing (1 - p)
        b = win/loss ratio (avg_win / avg_loss)
    """
    
    def __init__(self, config: Optional[SizingConfig] = None):
        self.config = config or SizingConfig()
        self.history: List[Dict] = []
        
    def calculate_kelly(self,
                       win_rate: float,
                       avg_win: float,
                       avg_loss: float) -> float:
        """
        Calculate optimal Kelly fraction.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)
        
        Returns:
            Optimal position size as fraction of capital
        """
        if avg_loss <= 0:
            logger.warning("avg_loss must be positive, using default")
            return self.config.max_position_pct
        
        if win_rate <= 0 or win_rate >= 1:
            logger.warning(f"Invalid win_rate {win_rate}, using default")
            return self.config.max_position_pct
        
        # Calculate win/loss ratio
        b = avg_win / avg_loss
        
        # Kelly formula
        p = win_rate
        q = 1 - p
        kelly = (p * b - q) / b
        
        # Ensure non-negative
        kelly = max(0, kelly)
        
        # Apply Kelly fraction (half-Kelly for safety)
        kelly_adjusted = kelly * self.config.kelly_fraction
        
        # Cap at max position
        kelly_final = min(kelly_adjusted, self.config.max_position_pct)
        
        logger.debug(f"Kelly: win_rate={win_rate:.2f}, b={b:.2f}, "
                    f"raw={kelly:.3f}, adj={kelly_adjusted:.3f}, final={kelly_final:.3f}")
        
        return kelly_final
    
    def calculate_from_trades(self, trades: pd.DataFrame) -> float:
        """
        Calculate Kelly from historical trades.
        
        Args:
            trades: DataFrame with 'return' column
        
        Returns:
            Optimal position size
        """
        if len(trades) < 10:
            logger.warning("Insufficient trade history, using default sizing")
            return self.config.max_position_pct / 2
        
        returns = trades['return']
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            logger.warning("No wins or losses in history, using default")
            return self.config.max_position_pct / 2
        
        win_rate = len(wins) / len(returns)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        return self.calculate_kelly(win_rate, avg_win, avg_loss)


# =============================================================================
# POSITION SIZER
# =============================================================================

class PositionSizer:
    """
    Dynamic position sizing with multiple adjustment factors.
    """
    
    def __init__(self, config: Optional[SizingConfig] = None):
        self.config = config or SizingConfig()
        self.kelly = KellyCalculator(self.config)
        
        # State tracking
        self.current_drawdown: float = 0.0
        self.current_vix: float = 20.0
        self.realized_volatility: float = 0.20
        
        # Sizing history
        self.sizing_log: List[Dict] = []
        
        # State persistence
        self.state_dir = Path('state/sizing')
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("PositionSizer initialized")
    
    def update_market_state(self,
                           drawdown: float,
                           vix: Optional[float] = None,
                           realized_vol: Optional[float] = None):
        """Update current market state for sizing decisions."""
        self.current_drawdown = drawdown
        if vix is not None:
            self.current_vix = vix
        if realized_vol is not None:
            self.realized_volatility = realized_vol
        
        logger.debug(f"Market state updated: DD={drawdown:.1%}, VIX={self.current_vix:.1f}, "
                    f"Vol={self.realized_volatility:.1%}")
    
    def calculate_position_size(self,
                               account_value: float,
                               symbol: str,
                               base_kelly: Optional[float] = None,
                               symbol_volatility: Optional[float] = None) -> Tuple[float, Dict]:
        """
        Calculate position size with all adjustments.
        
        Args:
            account_value: Total account value
            symbol: Stock symbol
            base_kelly: Pre-calculated Kelly fraction (optional)
            symbol_volatility: Stock's realized volatility (optional)
        
        Returns:
            (position_value, adjustment_details)
        """
        # Start with base Kelly or default
        if base_kelly is not None:
            base_pct = base_kelly
        else:
            base_pct = self.config.max_position_pct / 2  # Conservative default
        
        adjustments = {
            'base_pct': base_pct,
            'drawdown_mult': 1.0,
            'regime_mult': 1.0,
            'volatility_mult': 1.0,
            'final_pct': base_pct
        }
        
        # 1. Drawdown adjustment
        dd_mult = self._get_drawdown_multiplier()
        adjustments['drawdown_mult'] = dd_mult
        
        # 2. Regime adjustment (VIX-based)
        regime_mult = self._get_regime_multiplier()
        adjustments['regime_mult'] = regime_mult
        
        # 3. Volatility scaling
        if symbol_volatility is not None:
            vol_mult = self._get_volatility_multiplier(symbol_volatility)
        else:
            vol_mult = 1.0
        adjustments['volatility_mult'] = vol_mult
        
        # Calculate final percentage
        final_pct = base_pct * dd_mult * regime_mult * vol_mult
        
        # Apply limits
        final_pct = max(self.config.min_position_pct, final_pct)
        final_pct = min(self.config.max_position_pct, final_pct)
        
        # Check for halt condition
        if self.current_drawdown < self.config.dd_halt_threshold:
            final_pct = 0.0
            logger.warning(f"Drawdown {self.current_drawdown:.1%} exceeds halt threshold - no new positions")
        
        adjustments['final_pct'] = final_pct
        
        # Calculate dollar value
        position_value = account_value * final_pct
        
        # Log
        self.sizing_log.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'account_value': account_value,
            'adjustments': adjustments,
            'position_value': position_value
        })
        
        return position_value, adjustments
    
    def _get_drawdown_multiplier(self) -> float:
        """Get position size multiplier based on drawdown."""
        dd = self.current_drawdown
        
        if dd > self.config.dd_reduce_threshold:
            return 1.0
        elif dd > self.config.dd_halt_threshold:
            # Linear interpolation between reduce and halt
            progress = (dd - self.config.dd_reduce_threshold) / \
                      (self.config.dd_halt_threshold - self.config.dd_reduce_threshold)
            return 1.0 - progress * (1.0 - self.config.dd_reduce_mult)
        elif dd > self.config.dd_emergency_threshold:
            return self.config.dd_reduce_mult
        else:
            return self.config.dd_emergency_mult
    
    def _get_regime_multiplier(self) -> float:
        """Get position size multiplier based on market regime (VIX)."""
        vix = self.current_vix
        
        if vix < self.config.bull_threshold:
            return self.config.bull_mult
        elif vix > self.config.bear_threshold:
            return self.config.bear_mult
        else:
            # Linear interpolation
            progress = (vix - self.config.bull_threshold) / \
                      (self.config.bear_threshold - self.config.bull_threshold)
            return self.config.bull_mult - progress * (self.config.bull_mult - self.config.bear_mult)
    
    def _get_volatility_multiplier(self, symbol_vol: float) -> float:
        """Get position size multiplier based on volatility targeting."""
        if symbol_vol <= 0:
            return 1.0
        
        vol_mult = self.config.target_volatility / symbol_vol
        vol_mult = max(self.config.vol_scale_min, vol_mult)
        vol_mult = min(self.config.vol_scale_max, vol_mult)
        
        return vol_mult
    
    def calculate_portfolio_weights(self,
                                   signals: pd.DataFrame,
                                   account_value: float) -> Dict[str, float]:
        """
        Calculate portfolio weights for all signals.
        
        Args:
            signals: DataFrame with columns ['symbol', 'signal_strength', 'volatility']
            account_value: Total account value
        
        Returns:
            {symbol: weight} dictionary
        """
        weights = {}
        total_raw_weight = 0.0
        
        # Calculate raw weights for each signal
        for _, row in signals.iterrows():
            symbol = row['symbol']
            signal = row.get('signal_strength', 1.0)
            vol = row.get('volatility', self.realized_volatility)
            
            # Calculate position value
            pos_value, _ = self.calculate_position_size(
                account_value=account_value,
                symbol=symbol,
                symbol_volatility=vol
            )
            
            # Scale by signal strength
            weighted_value = pos_value * signal
            weights[symbol] = weighted_value
            total_raw_weight += weighted_value
        
        # Normalize to sum to max exposure (e.g., 100%)
        max_exposure = min(len(weights) * self.config.max_position_pct, 1.0)
        
        if total_raw_weight > 0:
            scale = min(max_exposure * account_value / total_raw_weight, 1.0)
            weights = {k: v * scale / account_value for k, v in weights.items()}
        
        # Apply position limits
        weights = self._apply_position_limits(weights)
        
        return weights
    
    def _apply_position_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply position count and size limits."""
        # Sort by weight descending
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        # Limit position count
        if len(sorted_weights) > self.config.max_positions:
            sorted_weights = sorted_weights[:self.config.max_positions]
        
        # Cap individual weights
        limited = {}
        for symbol, weight in sorted_weights:
            limited[symbol] = min(weight, self.config.max_single_stock_weight)
        
        return limited
    
    def get_sizing_recommendation(self, account_value: float) -> Dict:
        """Get current sizing recommendation summary."""
        dd_mult = self._get_drawdown_multiplier()
        regime_mult = self._get_regime_multiplier()
        
        # Effective max position size
        effective_max = (self.config.max_position_pct * dd_mult * regime_mult)
        
        regime = 'BULL' if self.current_vix < self.config.bull_threshold else \
                 'BEAR' if self.current_vix > self.config.bear_threshold else 'NEUTRAL'
        
        dd_status = 'NORMAL' if self.current_drawdown > self.config.dd_reduce_threshold else \
                   'REDUCED' if self.current_drawdown > self.config.dd_halt_threshold else \
                   'HALTED' if self.current_drawdown > self.config.dd_emergency_threshold else 'EMERGENCY'
        
        return {
            'account_value': account_value,
            'current_drawdown': self.current_drawdown,
            'current_vix': self.current_vix,
            'regime': regime,
            'drawdown_status': dd_status,
            'drawdown_multiplier': dd_mult,
            'regime_multiplier': regime_mult,
            'effective_max_position_pct': effective_max,
            'effective_max_position_value': account_value * effective_max,
            'max_positions': self.config.max_positions,
            'kelly_fraction': self.config.kelly_fraction,
            'new_entries_allowed': self.current_drawdown > self.config.dd_halt_threshold
        }
    
    def save_state(self):
        """Save sizing state to disk."""
        state = {
            'current_drawdown': self.current_drawdown,
            'current_vix': self.current_vix,
            'realized_volatility': self.realized_volatility,
            'sizing_log': self.sizing_log[-100:],  # Keep last 100
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.state_dir / 'sizing_state.json', 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info("Sizing state saved")
    
    def load_state(self):
        """Load sizing state from disk."""
        state_file = self.state_dir / 'sizing_state.json'
        if not state_file.exists():
            return
        
        with open(state_file) as f:
            state = json.load(f)
        
        self.current_drawdown = state.get('current_drawdown', 0.0)
        self.current_vix = state.get('current_vix', 20.0)
        self.realized_volatility = state.get('realized_volatility', 0.20)
        self.sizing_log = state.get('sizing_log', [])
        
        logger.info("Sizing state loaded")


# =============================================================================
# MAIN / TESTING
# =============================================================================

def main():
    """Test position sizer."""
    logger.info("=" * 70)
    logger.info("🧮 V23 POSITION SIZER TEST")
    logger.info("=" * 70)
    
    # Initialize
    sizer = PositionSizer()
    account_value = 100_000
    
    # Test Kelly calculation
    logger.info("\n📊 Testing Kelly Calculator...")
    kelly = sizer.kelly.calculate_kelly(
        win_rate=0.55,
        avg_win=0.05,
        avg_loss=0.03
    )
    logger.info(f"   Win rate: 55%, Win/Loss: 5%/3%")
    logger.info(f"   Kelly fraction: {kelly:.2%}")
    
    # Test different market states
    logger.info("\n📈 Testing position sizing in different market states...")
    
    scenarios = [
        {'name': 'Normal Bull', 'dd': 0.0, 'vix': 15.0},
        {'name': 'Normal Neutral', 'dd': -0.05, 'vix': 20.0},
        {'name': 'Drawdown Warning', 'dd': -0.12, 'vix': 22.0},
        {'name': 'High Volatility', 'dd': -0.08, 'vix': 35.0},
        {'name': 'Drawdown Halt', 'dd': -0.18, 'vix': 28.0},
    ]
    
    logger.info(f"\n   {'Scenario':<20} {'DD':>8} {'VIX':>6} {'Position':>12} {'Status':>12}")
    logger.info("-" * 65)
    
    for scenario in scenarios:
        sizer.update_market_state(
            drawdown=scenario['dd'],
            vix=scenario['vix']
        )
        
        pos_value, adjustments = sizer.calculate_position_size(
            account_value=account_value,
            symbol='TEST',
            base_kelly=0.05
        )
        
        rec = sizer.get_sizing_recommendation(account_value)
        
        logger.info(f"   {scenario['name']:<20} {scenario['dd']:>8.1%} {scenario['vix']:>6.1f} "
                   f"${pos_value:>10,.0f} {rec['drawdown_status']:>12}")
    
    # Test portfolio weights
    logger.info("\n📋 Testing portfolio weight calculation...")
    
    sizer.update_market_state(drawdown=-0.05, vix=18.0)
    
    signals = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        'signal_strength': [1.0, 0.8, 0.9, 0.7, 0.85],
        'volatility': [0.25, 0.22, 0.28, 0.30, 0.35]
    })
    
    weights = sizer.calculate_portfolio_weights(signals, account_value)
    
    logger.info(f"   {'Symbol':<10} {'Weight':>10} {'Value':>12}")
    logger.info("-" * 35)
    for sym, wt in weights.items():
        logger.info(f"   {sym:<10} {wt:>10.2%} ${wt*account_value:>10,.0f}")
    
    logger.info(f"\n   Total exposure: {sum(weights.values()):.2%}")
    
    # Get recommendation summary
    logger.info("\n📊 Current Sizing Recommendation:")
    rec = sizer.get_sizing_recommendation(account_value)
    for key, value in rec.items():
        if isinstance(value, float):
            if key.endswith('pct') or key.endswith('multiplier') or key == 'kelly_fraction':
                logger.info(f"   {key}: {value:.2%}")
            elif 'value' in key:
                logger.info(f"   {key}: ${value:,.0f}")
            else:
                logger.info(f"   {key}: {value:.2f}")
        else:
            logger.info(f"   {key}: {value}")
    
    # Save state
    sizer.save_state()
    
    logger.info("\n✅ Position sizer test complete")
    
    return sizer


if __name__ == "__main__":
    main()

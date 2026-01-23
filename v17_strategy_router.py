#!/usr/bin/env python3
"""
V17.0 Strategy Router
======================
Routes trading decisions based on HMM regime state.

Regime -> Strategy Mapping:
- 0: LowVolTrend     -> Cross-sectional momentum
- 1: HighVolTrend    -> Trend following with breakouts
- 2: LowVolMeanRevert -> Statistical arbitrage / pairs
- 3: Crisis          -> Defensive (reduce exposure)

Each strategy defines:
- Position sizing method
- Signal generation logic
- Risk management rules
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('V17_Router')


@dataclass
class StrategyConfig:
    """Configuration for a strategy"""
    name: str
    max_positions: int
    max_position_size: float  # As fraction of portfolio
    min_holding_period: int   # Days
    volatility_target: float  # Annual vol target
    stop_loss: float          # Maximum loss per position
    take_profit: Optional[float] = None


# Default configs per regime
STRATEGY_CONFIGS = {
    0: StrategyConfig(  # LowVolTrend - Momentum
        name='momentum_xsection',
        max_positions=50,
        max_position_size=0.04,    # 4% max per position
        min_holding_period=5,
        volatility_target=0.15,    # 15% annual vol
        stop_loss=0.05,            # 5% stop
        take_profit=0.20           # 20% take profit
    ),
    1: StrategyConfig(  # HighVolTrend - Trend Following
        name='trend_follow',
        max_positions=30,
        max_position_size=0.05,
        min_holding_period=10,
        volatility_target=0.20,
        stop_loss=0.08,
        take_profit=0.30
    ),
    2: StrategyConfig(  # LowVolMeanRevert - Stat Arb
        name='stat_arb',
        max_positions=40,
        max_position_size=0.03,
        min_holding_period=3,
        volatility_target=0.12,
        stop_loss=0.04,
        take_profit=0.10
    ),
    3: StrategyConfig(  # Crisis - Defensive
        name='defensive',
        max_positions=10,
        max_position_size=0.02,
        min_holding_period=1,
        volatility_target=0.08,
        stop_loss=0.03,
        take_profit=None  # No take profit, just survive
    )
}


class BaseStrategy(ABC):
    """Abstract base class for all strategies"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = config.name
    
    @abstractmethod
    def generate_signals(
        self, 
        prices: pd.DataFrame,
        factors: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Returns:
            DataFrame with columns: symbol, signal (-1 to 1), alpha_score
        """
        pass
    
    @abstractmethod
    def calculate_position_sizes(
        self,
        signals: pd.DataFrame,
        volatilities: pd.DataFrame,
        portfolio_value: float
    ) -> pd.DataFrame:
        """
        Calculate position sizes from signals.
        
        Returns:
            DataFrame with columns: symbol, target_weight, target_shares
        """
        pass
    
    def apply_risk_limits(
        self,
        positions: pd.DataFrame,
        current_positions: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply risk limits to positions"""
        positions = positions.copy()
        
        # Cap position size
        positions['target_weight'] = positions['target_weight'].clip(
            -self.config.max_position_size,
            self.config.max_position_size
        )
        
        # Limit number of positions
        if len(positions) > self.config.max_positions:
            positions = positions.nlargest(
                self.config.max_positions, 
                'alpha_score'
            )
        
        return positions


class MomentumStrategy(BaseStrategy):
    """Cross-sectional momentum strategy for LowVolTrend regime"""
    
    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = STRATEGY_CONFIGS[0]
        super().__init__(config)
        self.lookback_short = 20   # 1 month
        self.lookback_long = 252   # 1 year
        self.skip_recent = 5       # Skip most recent days (reversal effect)
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        factors: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate momentum signals using 12-1 momentum.
        
        Long top decile, short bottom decile.
        """
        # Calculate returns
        returns_12m = prices['close'].pct_change(self.lookback_long)
        returns_1m = prices['close'].pct_change(self.skip_recent)
        
        # 12-1 momentum (skip recent month to avoid reversal)
        momentum = returns_12m - returns_1m
        
        # Get latest momentum by symbol
        if 'symbol' in prices.columns:
            latest = prices.groupby('symbol').last()
            mom_df = pd.DataFrame({
                'symbol': latest.index,
                'momentum': momentum.groupby(prices['symbol']).last().values
            })
        else:
            # Single symbol
            mom_df = pd.DataFrame({
                'symbol': ['UNKNOWN'],
                'momentum': [momentum.iloc[-1]]
            })
        
        mom_df = mom_df.dropna()
        
        # Rank and create signals
        mom_df['rank'] = mom_df['momentum'].rank(pct=True)
        
        # Long top 20%, short bottom 20%
        mom_df['signal'] = 0.0
        mom_df.loc[mom_df['rank'] > 0.8, 'signal'] = 1.0
        mom_df.loc[mom_df['rank'] < 0.2, 'signal'] = -1.0
        
        mom_df['alpha_score'] = mom_df['momentum'].abs()
        
        return mom_df[['symbol', 'signal', 'alpha_score', 'momentum']]
    
    def calculate_position_sizes(
        self,
        signals: pd.DataFrame,
        volatilities: pd.DataFrame,
        portfolio_value: float
    ) -> pd.DataFrame:
        """
        Calculate position sizes using volatility weighting.
        """
        positions = signals.merge(volatilities, on='symbol', how='left')
        
        # Target volatility per position
        target_vol = self.config.volatility_target / np.sqrt(self.config.max_positions)
        
        # Vol-weighted position size
        positions['vol'] = positions.get('volatility', 0.20)  # Default 20%
        positions['raw_weight'] = target_vol / positions['vol']
        
        # Scale by signal strength
        positions['target_weight'] = positions['signal'] * positions['raw_weight']
        
        # Apply position limits
        positions['target_weight'] = positions['target_weight'].clip(
            -self.config.max_position_size,
            self.config.max_position_size
        )
        
        # Calculate shares
        positions['target_value'] = positions['target_weight'] * portfolio_value
        
        return positions[['symbol', 'signal', 'target_weight', 'target_value', 'alpha_score']]


class TrendFollowStrategy(BaseStrategy):
    """Trend following strategy for HighVolTrend regime"""
    
    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = STRATEGY_CONFIGS[1]
        super().__init__(config)
        self.breakout_period = 20
        self.atr_period = 14
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        factors: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate breakout signals.
        
        Long on breakout above 20-day high.
        """
        # Calculate rolling highs/lows
        high_20 = prices['high'].rolling(self.breakout_period).max()
        low_20 = prices['low'].rolling(self.breakout_period).min()
        
        # Breakout signal
        close = prices['close']
        breakout_long = close >= high_20
        breakout_short = close <= low_20
        
        # Calculate ATR for position sizing
        tr = pd.concat([
            prices['high'] - prices['low'],
            (prices['high'] - close.shift()).abs(),
            (prices['low'] - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()
        
        # Build signals
        if 'symbol' in prices.columns:
            symbols = prices.groupby('symbol').last().index
            signals_list = []
            
            for sym in symbols:
                mask = prices['symbol'] == sym
                sym_close = close[mask].iloc[-1] if mask.any() else 0
                sym_high = high_20[mask].iloc[-1] if mask.any() else 0
                sym_low = low_20[mask].iloc[-1] if mask.any() else 0
                sym_atr = atr[mask].iloc[-1] if mask.any() else 1
                
                if sym_close >= sym_high:
                    signal = 1.0
                elif sym_close <= sym_low:
                    signal = -1.0
                else:
                    signal = 0.0
                
                # Trend strength
                trend_strength = abs(sym_close - (sym_high + sym_low) / 2) / sym_atr if sym_atr > 0 else 0
                
                signals_list.append({
                    'symbol': sym,
                    'signal': signal,
                    'alpha_score': trend_strength,
                    'atr': sym_atr
                })
            
            return pd.DataFrame(signals_list)
        else:
            return pd.DataFrame()
    
    def calculate_position_sizes(
        self,
        signals: pd.DataFrame,
        volatilities: pd.DataFrame,
        portfolio_value: float
    ) -> pd.DataFrame:
        """
        ATR-based position sizing.
        """
        positions = signals.copy()
        
        # Use ATR for sizing
        atr = positions.get('atr', 1.0)
        
        # Risk 1% of portfolio per ATR
        risk_per_trade = 0.01 * portfolio_value
        
        # Position size = risk / ATR
        positions['target_value'] = risk_per_trade / atr * positions['signal']
        positions['target_weight'] = positions['target_value'] / portfolio_value
        
        # Apply limits
        positions['target_weight'] = positions['target_weight'].clip(
            -self.config.max_position_size,
            self.config.max_position_size
        )
        
        return positions[['symbol', 'signal', 'target_weight', 'target_value', 'alpha_score']]


class StatArbStrategy(BaseStrategy):
    """Statistical arbitrage for LowVolMeanRevert regime"""
    
    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = STRATEGY_CONFIGS[2]
        super().__init__(config)
        self.zscore_period = 20
        self.entry_zscore = 1.5
        self.exit_zscore = 0.5
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        factors: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate mean reversion signals based on z-score.
        """
        close = prices['close']
        
        # Calculate rolling z-score
        rolling_mean = close.rolling(self.zscore_period).mean()
        rolling_std = close.rolling(self.zscore_period).std()
        zscore = (close - rolling_mean) / rolling_std
        
        if 'symbol' in prices.columns:
            symbols = prices.groupby('symbol').last().index
            signals_list = []
            
            for sym in symbols:
                mask = prices['symbol'] == sym
                sym_zscore = zscore[mask].iloc[-1] if mask.any() else 0
                
                # Mean reversion: short when zscore > entry, long when < -entry
                if sym_zscore > self.entry_zscore:
                    signal = -1.0
                elif sym_zscore < -self.entry_zscore:
                    signal = 1.0
                else:
                    signal = 0.0
                
                signals_list.append({
                    'symbol': sym,
                    'signal': signal,
                    'alpha_score': abs(sym_zscore),
                    'zscore': sym_zscore
                })
            
            return pd.DataFrame(signals_list)
        else:
            return pd.DataFrame()
    
    def calculate_position_sizes(
        self,
        signals: pd.DataFrame,
        volatilities: pd.DataFrame,
        portfolio_value: float
    ) -> pd.DataFrame:
        """
        Equal weight positions for mean reversion.
        """
        positions = signals.copy()
        
        # Equal weight across positions
        n_positions = len(positions[positions['signal'] != 0])
        if n_positions > 0:
            weight_per_position = min(
                self.config.max_position_size,
                1.0 / n_positions
            )
        else:
            weight_per_position = 0
        
        positions['target_weight'] = positions['signal'] * weight_per_position
        positions['target_value'] = positions['target_weight'] * portfolio_value
        
        return positions[['symbol', 'signal', 'target_weight', 'target_value', 'alpha_score']]


class DefensiveStrategy(BaseStrategy):
    """Defensive strategy for Crisis regime"""
    
    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = STRATEGY_CONFIGS[3]
        super().__init__(config)
        self.safe_havens = ['GLD', 'TLT', 'SHY', 'IEF', 'VXX', 'UVXY']
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        factors: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Defensive positioning: safe havens only.
        """
        if 'symbol' in prices.columns:
            symbols = prices.groupby('symbol').last().index
        else:
            symbols = []
        
        signals_list = []
        for sym in symbols:
            if sym in self.safe_havens:
                signals_list.append({
                    'symbol': sym,
                    'signal': 1.0,
                    'alpha_score': 1.0
                })
            else:
                # Reduce all other positions
                signals_list.append({
                    'symbol': sym,
                    'signal': 0.0,  # Exit
                    'alpha_score': 0.0
                })
        
        return pd.DataFrame(signals_list) if signals_list else pd.DataFrame()
    
    def calculate_position_sizes(
        self,
        signals: pd.DataFrame,
        volatilities: pd.DataFrame,
        portfolio_value: float
    ) -> pd.DataFrame:
        """
        Conservative sizing in crisis.
        """
        positions = signals.copy()
        
        # Very small positions
        positions['target_weight'] = positions['signal'] * self.config.max_position_size
        positions['target_value'] = positions['target_weight'] * portfolio_value
        
        return positions[['symbol', 'signal', 'target_weight', 'target_value', 'alpha_score']]


class StrategyRouter:
    """
    Routes trading to appropriate strategy based on regime.
    """
    
    def __init__(self):
        self.strategies = {
            0: MomentumStrategy(),
            1: TrendFollowStrategy(),
            2: StatArbStrategy(),
            3: DefensiveStrategy()
        }
        
        self.current_regime: int = 2  # Default to mean revert
        self.regime_history: List[Tuple[str, int]] = []
    
    def set_regime(self, regime: int, date: Optional[str] = None):
        """Set current regime"""
        if regime not in self.strategies:
            raise ValueError(f"Invalid regime: {regime}")
        
        self.current_regime = regime
        self.regime_history.append((date or 'unknown', regime))
        
        logger.info(f"📊 Regime set to {regime}: {self.strategies[regime].name}")
    
    def get_strategy(self, regime: Optional[int] = None) -> BaseStrategy:
        """Get strategy for regime"""
        regime = regime or self.current_regime
        return self.strategies[regime]
    
    def get_config(self, regime: Optional[int] = None) -> StrategyConfig:
        """Get config for regime"""
        regime = regime or self.current_regime
        return STRATEGY_CONFIGS[regime]
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        factors: Optional[pd.DataFrame] = None,
        regime: Optional[int] = None
    ) -> pd.DataFrame:
        """Generate signals for current/specified regime"""
        strategy = self.get_strategy(regime)
        return strategy.generate_signals(prices, factors)
    
    def get_portfolio(
        self,
        prices: pd.DataFrame,
        portfolio_value: float,
        volatilities: Optional[pd.DataFrame] = None,
        factors: Optional[pd.DataFrame] = None,
        regime: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get complete portfolio for current regime.
        
        Returns:
            DataFrame with target positions
        """
        strategy = self.get_strategy(regime)
        
        # Generate signals
        signals = strategy.generate_signals(prices, factors)
        
        if signals.empty:
            return pd.DataFrame()
        
        # Default volatilities if not provided
        if volatilities is None:
            volatilities = pd.DataFrame({
                'symbol': signals['symbol'],
                'volatility': 0.20
            })
        
        # Calculate positions
        positions = strategy.calculate_position_sizes(
            signals, volatilities, portfolio_value
        )
        
        # Apply risk limits
        positions = strategy.apply_risk_limits(positions, pd.DataFrame())
        
        return positions
    
    def get_regime_allocation(self, regime: Optional[int] = None) -> Dict[str, float]:
        """Get typical allocation for regime"""
        regime = regime or self.current_regime
        config = STRATEGY_CONFIGS[regime]
        
        return {
            'max_equity_exposure': config.max_positions * config.max_position_size,
            'volatility_target': config.volatility_target,
            'max_positions': config.max_positions,
            'max_position_size': config.max_position_size
        }


def main():
    """Test strategy router"""
    print("\n" + "=" * 60)
    print("🚦 V17.0 STRATEGY ROUTER TEST")
    print("=" * 60)
    
    router = StrategyRouter()
    
    # Test each regime
    for regime in range(4):
        strategy = router.get_strategy(regime)
        config = router.get_config(regime)
        allocation = router.get_regime_allocation(regime)
        
        print(f"\n📊 Regime {regime}: {strategy.name}")
        print(f"   Max Positions: {config.max_positions}")
        print(f"   Max Position Size: {config.max_position_size:.1%}")
        print(f"   Vol Target: {config.volatility_target:.1%}")
        print(f"   Stop Loss: {config.stop_loss:.1%}")
        print(f"   Max Equity Exposure: {allocation['max_equity_exposure']:.1%}")
    
    # Load regime from HMM
    try:
        import pickle
        with open('cache/v17_hmm_regime.pkl', 'rb') as f:
            hmm_data = pickle.load(f)
        
        # Get current regime from saved state mapping
        state_mapping = hmm_data['state_mapping']
        print(f"\n✅ Loaded HMM state mapping: {state_mapping}")
        
    except Exception as e:
        print(f"\n⚠️ Could not load HMM model: {e}")
    
    print("\n✅ Strategy Router initialized")
    
    return router


if __name__ == "__main__":
    main()

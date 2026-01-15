"""Phase 9 Orchestrator - Main Integration Module.

Coordinates all Phase 9 components:
1. Hierarchical Regime Strategy
2. Advanced Alpha Engine
3. Adaptive Universe Screener
4. Dynamic Position Optimizer

Provides complete backtest integration and real-time signal generation.
"""

import os
import sys
import json
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd

# Phase 9 modules
from .regime_meta_strategy import (
    HierarchicalRegimeStrategy, RegimeMeta, MacroRegime, TDARegime
)
from .alpha_engine import AdvancedAlphaEngine, AlphaSignal
from .adaptive_screener import AdaptiveUniverseScreener, StockProfile, UniverseConfig
from .dynamic_optimizer import DynamicPositionOptimizer, PositionRecommendation, PortfolioState

logger = logging.getLogger(__name__)


@dataclass
class Phase9Config:
    """Configuration for Phase 9 system."""
    # Universe
    initial_universe: List[str] = None
    target_universe_size: int = 50
    min_universe_size: int = 15
    
    # Regime
    use_regime_adaptation: bool = True
    regime_update_frequency: int = 1  # Days
    
    # Alpha - more aggressive momentum focus
    momentum_weight: float = 0.55  # Increased from 0.40
    reversal_weight: float = 0.10  # Reduced from 0.15
    tda_weight: float = 0.20  # Reduced from 0.25
    cross_sectional_weight: float = 0.15  # Reduced from 0.20
    
    # Position sizing - more aggressive
    kelly_fraction: float = 0.40  # Increased from 0.25
    target_volatility: float = 0.20  # Increased from 0.15
    max_position: float = 0.12  # Increased from 0.08
    max_leverage: float = 1.0
    
    # Risk management - slightly relaxed for higher returns
    max_sector_weight: float = 0.30  # Increased from 0.25
    max_drawdown_threshold: float = 0.20  # Increased from 0.15
    stop_loss_atr: float = 2.5  # Increased from 2.0
    
    # Backtest
    initial_capital: float = 100000.0
    cost_bp_per_side: float = 5.0  # 0.05% per side
    
    def __post_init__(self):
        if self.initial_universe is None:
            self.initial_universe = []


@dataclass 
class DailyState:
    """Complete state for a single trading day."""
    date: str
    
    # Regime
    regime_meta: Optional[RegimeMeta] = None
    
    # Universe
    universe_size: int = 0
    universe_tickers: List[str] = None
    
    # Signals
    signals: Dict[str, float] = None
    top_signals: List[Tuple[str, float]] = None
    
    # Portfolio
    portfolio_value: float = 0.0
    cash: float = 0.0
    positions: Dict[str, float] = None
    
    # Risk
    drawdown: float = 0.0
    portfolio_volatility: float = 0.0
    
    def __post_init__(self):
        if self.universe_tickers is None:
            self.universe_tickers = []
        if self.signals is None:
            self.signals = {}
        if self.top_signals is None:
            self.top_signals = []
        if self.positions is None:
            self.positions = {}


class Phase9Orchestrator:
    """
    Main orchestrator for Phase 9 advanced alpha generation.
    
    Integrates all Phase 9 components for:
    - Real-time signal generation
    - Backtest execution
    - Performance analysis
    """
    
    def __init__(
        self,
        config: Optional[Phase9Config] = None,
    ):
        self.config = config or Phase9Config()
        
        # Initialize components
        self.regime_strategy = HierarchicalRegimeStrategy()
        self.alpha_engine = AdvancedAlphaEngine()
        self.universe_screener = AdaptiveUniverseScreener(
            config=UniverseConfig(
                target_universe_size=self.config.target_universe_size,
                min_universe_size=self.config.min_universe_size,
                max_sector_weight=self.config.max_sector_weight,
            )
        )
        self.position_optimizer = DynamicPositionOptimizer(
            kelly_fraction=self.config.kelly_fraction,
            target_volatility=self.config.target_volatility,
            max_position=self.config.max_position,
            max_leverage=self.config.max_leverage,
        )
        
        # State tracking
        self.current_state: Optional[DailyState] = None
        self.state_history: List[DailyState] = []
        
        # Performance tracking
        self.equity_curve: List[float] = []
        self.daily_returns: List[float] = []
        self.trades: List[Dict] = []
        
        # Caches
        self._price_cache: Dict[str, np.ndarray] = {}
        self._tda_cache: Dict[str, pd.DataFrame] = {}
        self._sector_cache: Dict[str, str] = {}
        
        logger.info("Phase 9 Orchestrator initialized")
    
    def process_day(
        self,
        date: str,
        spy_prices: pd.DataFrame,
        universe_prices: Dict[str, np.ndarray],
        universe_volumes: Dict[str, np.ndarray],
        sector_map: Dict[str, str],
        vix_data: Optional[pd.DataFrame] = None,
        tda_data: Optional[Dict[str, pd.DataFrame]] = None,
        sector_etf_data: Optional[Dict[str, pd.DataFrame]] = None,
        current_portfolio: Optional[PortfolioState] = None,
    ) -> DailyState:
        """
        Process a single trading day.
        
        Args:
            date: Trading date
            spy_prices: SPY OHLCV data for regime detection
            universe_prices: {ticker: price_array}
            universe_volumes: {ticker: volume_array}
            sector_map: {ticker: sector}
            vix_data: VIX data for volatility regime
            tda_data: {ticker: tda_features_df}
            sector_etf_data: {sector_etf: ohlcv_df} for sector rotation
            current_portfolio: Current portfolio state
            
        Returns:
            DailyState with complete analysis
        """
        # Step 1: Regime Analysis
        regime_meta = self._analyze_regime(
            spy_prices, vix_data, sector_etf_data, 
            tda_data.get('SPY') if tda_data else None,
            date
        )
        
        # Step 2: Universe Screening
        universe_tickers, profiles = self._screen_universe(
            universe_prices, universe_volumes, sector_map, tda_data,
            regime_meta.macro_regime.value if regime_meta else None
        )
        
        # Step 3: Generate Alpha Signals
        signals, alpha_signals = self._generate_signals(
            universe_tickers, universe_prices, tda_data,
            regime_meta.factor_weights if regime_meta else None
        )
        
        # Step 4: Optimize Positions
        if current_portfolio:
            recommendations = self._optimize_positions(
                signals, universe_prices, current_portfolio,
                regime_meta.position_scale if regime_meta else 1.0
            )
        else:
            recommendations = {}
        
        # Build state
        state = DailyState(
            date=date,
            regime_meta=regime_meta,
            universe_size=len(universe_tickers),
            universe_tickers=universe_tickers,
            signals=signals,
            top_signals=sorted(signals.items(), key=lambda x: x[1], reverse=True)[:10],
            portfolio_value=current_portfolio.total_value if current_portfolio else self.config.initial_capital,
            cash=current_portfolio.cash if current_portfolio else self.config.initial_capital,
            positions=current_portfolio.positions if current_portfolio else {},
            drawdown=current_portfolio.current_drawdown if current_portfolio else 0,
            portfolio_volatility=current_portfolio.portfolio_volatility if current_portfolio else 0,
        )
        
        self.current_state = state
        self.state_history.append(state)
        
        return state
    
    def _analyze_regime(
        self,
        spy_prices: pd.DataFrame,
        vix_data: Optional[pd.DataFrame],
        sector_data: Optional[Dict[str, pd.DataFrame]],
        spy_tda: Optional[pd.DataFrame],
        date: str,
    ) -> RegimeMeta:
        """Run hierarchical regime analysis."""
        if not self.config.use_regime_adaptation:
            # Return neutral regime
            return RegimeMeta(
                macro_regime=MacroRegime.TRANSITION,
                macro_confidence=0.5,
                macro_transition_prob=0.5,
                tda_regime=TDARegime.CONSOLIDATION,
                tda_turbulence=50.0,
                tda_fragmentation=0.5,
                tda_cyclicity=0.5,
                recommended_leverage=1.0,
                factor_weights=self._get_default_weights(),
                position_scale=1.0,
                stop_multiplier=2.0,
                trade_allowed=True,
                strategy_mode='neutral',
                timestamp=date,
                days_in_regime=0,
            )
        
        return self.regime_strategy.analyze(
            spy_prices=spy_prices,
            vix_data=vix_data,
            sector_data=sector_data,
            tda_features=spy_tda,
            timestamp=date,
        )
    
    def _screen_universe(
        self,
        prices: Dict[str, np.ndarray],
        volumes: Dict[str, np.ndarray],
        sector_map: Dict[str, str],
        tda_data: Optional[Dict[str, pd.DataFrame]],
        regime: Optional[str],
    ) -> Tuple[List[str], Dict[str, StockProfile]]:
        """Screen and rank universe."""
        return self.universe_screener.screen_universe(
            price_data=prices,
            volume_data=volumes,
            sector_map=sector_map,
            tda_data=tda_data,
            regime=regime,
        )
    
    def _generate_signals(
        self,
        tickers: List[str],
        prices: Dict[str, np.ndarray],
        tda_data: Optional[Dict[str, pd.DataFrame]],
        regime_weights: Optional[Dict[str, float]],
    ) -> Tuple[Dict[str, float], Dict[str, AlphaSignal]]:
        """Generate alpha signals for all tickers."""
        # Update alpha engine with regime weights
        if regime_weights:
            self.alpha_engine.update_weights(regime_weights)
        
        signals = {}
        alpha_signals = {}
        
        # First pass: compute raw signals
        raw_scores = {}
        for ticker in tickers:
            if ticker not in prices:
                continue
            
            price_array = prices[ticker]
            tda_features = tda_data.get(ticker) if tda_data else None
            
            alpha = self.alpha_engine.generate_alpha_signal(
                ticker=ticker,
                prices=price_array,
                tda_features=tda_features,
                regime_weights=regime_weights,
            )
            
            raw_scores[ticker] = alpha.momentum_signal  # For cross-sectional
            alpha_signals[ticker] = alpha
            signals[ticker] = alpha.weighted_signal
        
        # Second pass: add cross-sectional component
        for ticker in tickers:
            if ticker in alpha_signals:
                alpha = self.alpha_engine.generate_alpha_signal(
                    ticker=ticker,
                    prices=prices.get(ticker, np.array([100.0])),
                    tda_features=tda_data.get(ticker) if tda_data else None,
                    cross_sectional_data=raw_scores,
                    regime_weights=regime_weights,
                )
                signals[ticker] = alpha.weighted_signal
                alpha_signals[ticker] = alpha
        
        return signals, alpha_signals
    
    def _optimize_positions(
        self,
        signals: Dict[str, float],
        prices: Dict[str, np.ndarray],
        current_portfolio: PortfolioState,
        regime_scale: float,
    ) -> Dict[str, PositionRecommendation]:
        """Optimize position sizes."""
        # Compute volatilities and expected returns
        volatilities = {}
        expected_returns = {}
        returns_data = {}
        
        for ticker, price_array in prices.items():
            if ticker not in signals:
                continue
            
            if len(price_array) >= 60:
                returns = np.diff(price_array[-60:]) / price_array[-60:-1]
                vol = np.std(returns) * np.sqrt(252)
                volatilities[ticker] = vol
                
                # Simple expected return: signal * base return
                expected_returns[ticker] = signals[ticker] * 0.20
                
                returns_data[ticker] = returns
            else:
                volatilities[ticker] = 0.20
                expected_returns[ticker] = signals[ticker] * 0.20
        
        return self.position_optimizer.optimize_positions(
            signals=signals,
            expected_returns=expected_returns,
            volatilities=volatilities,
            returns_data=returns_data,
            current_portfolio=current_portfolio,
            regime_scale=regime_scale,
        )
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Get default signal weights."""
        return {
            'momentum': self.config.momentum_weight,
            'reversal': self.config.reversal_weight,
            'tda': self.config.tda_weight,
            'cross_sectional': self.config.cross_sectional_weight,
        }
    
    def get_regime_summary(self) -> Dict:
        """Get current regime summary."""
        return self.regime_strategy.get_regime_summary()
    
    def get_universe_summary(self) -> Dict:
        """Get current universe summary."""
        return self.universe_screener.get_universe_summary()
    
    def get_top_signals(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N signals."""
        if self.current_state:
            return self.current_state.top_signals[:n]
        return []
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary from backtest."""
        if not self.equity_curve:
            return {'status': 'no_data'}
        
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0])
        
        # Core metrics
        total_return = (equity[-1] / equity[0] - 1) if len(equity) > 0 else 0
        n_days = len(returns)
        n_years = n_days / 252
        
        if n_years > 0:
            cagr = (1 + total_return) ** (1 / n_years) - 1
        else:
            cagr = 0
        
        # Risk metrics
        if len(returns) > 0:
            daily_vol = np.std(returns)
            annual_vol = daily_vol * np.sqrt(252)
            sharpe = (cagr - 0.02) / annual_vol if annual_vol > 0 else 0
        else:
            annual_vol = 0
            sharpe = 0
        
        # Drawdown
        peak = np.maximum.accumulate(equity)
        drawdowns = equity / peak - 1
        max_dd = np.min(drawdowns)
        
        return {
            'total_return': f"{total_return:.1%}",
            'cagr': f"{cagr:.1%}",
            'sharpe_ratio': f"{sharpe:.2f}",
            'annual_volatility': f"{annual_vol:.1%}",
            'max_drawdown': f"{max_dd:.1%}",
            'n_trades': len(self.trades),
            'n_days': n_days,
        }
    
    def export_results(self, output_path: str):
        """Export results to JSON."""
        results = {
            'config': asdict(self.config),
            'performance': self.get_performance_summary(),
            'regime_history': [
                self.regime_strategy.get_regime_summary()
            ],
            'universe_summary': self.get_universe_summary(),
            'equity_curve': self.equity_curve[-100:] if self.equity_curve else [],  # Last 100 points
            'daily_returns': self.daily_returns[-252:] if self.daily_returns else [],  # Last year
            'trade_count': len(self.trades),
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results exported to {output_path}")


# Convenience function for quick initialization
def create_phase9_system(
    target_universe_size: int = 50,
    target_volatility: float = 0.15,
    max_position: float = 0.08,
) -> Phase9Orchestrator:
    """Create Phase 9 system with sensible defaults."""
    config = Phase9Config(
        target_universe_size=target_universe_size,
        target_volatility=target_volatility,
        max_position=max_position,
    )
    return Phase9Orchestrator(config)

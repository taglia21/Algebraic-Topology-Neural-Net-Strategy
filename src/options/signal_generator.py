"""
Multi-Strategy Signal Generator
================================

Generates trading signals using multiple options strategies:
- IV Rank Strategy: Sell premium when IV high, buy when IV low
- Theta Decay Strategy: Sell options in optimal DTE range
- Mean Reversion Strategy: Trade based on z-score extremes
- Delta Hedging Strategy: Hedge when portfolio delta exceeds threshold

Each strategy produces signals with confidence scores that are combined
using configured weights.
"""

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional
import logging

from .config import RISK_CONFIG
from .universe import get_universe, is_strategy_allowed, STRATEGY_DEFINITIONS
from .iv_analyzer import IVAnalyzer
from .theta_decay_engine import ThetaDecayEngine
from .iv_data_manager import IVDataManager
from .theta_decay_engine import IVRegime, TrendDirection


# ============================================================================
# DATA MODELS
# ============================================================================

class SignalType(Enum):
    """Signal direction."""
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    HOLD = "hold"


class SignalSource(Enum):
    """Signal source strategy."""
    IV_RANK = "iv_rank"
    THETA_DECAY = "theta_decay"
    MEAN_REVERSION = "mean_reversion"
    DELTA_HEDGING = "delta_hedging"


@dataclass
class Signal:
    """Trading signal with all metadata."""
    symbol: str
    signal_type: SignalType
    signal_source: SignalSource
    strategy: str  # e.g., "iron_condor", "credit_spread"
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    
    # Option parameters
    dte: Optional[int] = None
    strike_put: Optional[float] = None
    strike_call: Optional[float] = None
    
    # Market data
    iv_rank: Optional[float] = None
    current_price: Optional[float] = None
    z_score: Optional[float] = None
    delta: Optional[float] = None
    
    # Risk metrics
    probability_of_profit: Optional[float] = None
    expected_premium: Optional[float] = None
    max_loss: Optional[float] = None
    
    # Resolved contract fields (populated by OptionContractResolver after signal generation)
    occ_symbol: Optional[str] = None
    expiration_date: Optional[date] = None
    
    # Metadata
    reason: str = ""


# ============================================================================
# IV RANK STRATEGY
# ============================================================================

class IVRankStrategy:
    """
    Generate signals based on Implied Volatility Rank.
    
    Logic:
    - HIGH IV (>50): SELL premium (credit spreads, iron condors)
    - LOW IV (<30): BUY options (straddles, strangles)
    - NORMAL IV: No signal
    """
    
    def __init__(self):
        self.config = RISK_CONFIG
        self.logger = logging.getLogger(__name__)
        self.iv_analyzer = IVAnalyzer()
        self.iv_data_manager = IVDataManager()
        
    async def generate_signals(self, symbols: List[str]) -> List[Signal]:
        """
        Generate IV Rank signals for all symbols.
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            List of signals
        """
        signals = []
        
        for symbol in symbols:
            try:
                signal = await self._analyze_symbol(symbol)
                if signal:
                    signals.append(signal)
            except Exception as e:
                self.logger.error(f"IV Rank error for {symbol}: {e}")
                continue
        
        return signals
    
    async def _analyze_symbol(self, symbol: str) -> Optional[Signal]:
        """Analyze single symbol for IV Rank signal."""
        # Prefer the IV cache (IVDataManager) for a stable, production-safe IV rank.
        # If unavailable, default to neutral (50) and simply avoid IV-rank signals.
        iv_rank = self.iv_data_manager.get_iv_rank(symbol)
        if iv_rank is None:
            return None

        current_price = None
        
        # HIGH IV: SELL premium
        if iv_rank >= self.config["iv_rank_sell_threshold"]:
            # Prefer credit spreads and iron condors
            strategy = "iron_condor" if is_strategy_allowed(symbol, "iron_condor") else "credit_spread"
            
            confidence = min((iv_rank - 50) / 50, 1.0)  # Scale 50-100 to 0-1
            
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                signal_source=SignalSource.IV_RANK,
                strategy=strategy,
                confidence=confidence,
                timestamp=datetime.now(),
                iv_rank=iv_rank,
                current_price=current_price,
                dte=35,  # Mid-range optimal
                reason=f"High IV Rank ({iv_rank:.1f}) - sell premium",
            )
        
        # LOW IV: BUY options
        elif iv_rank <= self.config["iv_rank_buy_threshold"]:
            # Prefer straddles and strangles
            strategy = "straddle" if is_strategy_allowed(symbol, "straddle") else "strangle"
            
            confidence = min((30 - iv_rank) / 30, 1.0)  # Scale 0-30 to 1-0
            
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                signal_source=SignalSource.IV_RANK,
                strategy=strategy,
                confidence=confidence,
                timestamp=datetime.now(),
                iv_rank=iv_rank,
                current_price=current_price,
                dte=21,  # Shorter for vol expansion
                reason=f"Low IV Rank ({iv_rank:.1f}) - buy options",
            )
        
        return None


# ============================================================================
# THETA DECAY STRATEGY
# ============================================================================

class ThetaDecayStrategy:
    """
    Generate signals based on theta decay efficiency.
    
    Logic:
    - SELL options in 21-45 DTE sweet spot (maximum theta/gamma ratio)
    - Target high probability of profit (>50%)
    - Focus on premium collection
    """
    
    def __init__(self):
        self.config = RISK_CONFIG
        self.logger = logging.getLogger(__name__)
        self.theta_engine = ThetaDecayEngine()
        self.iv_data_manager = IVDataManager()
        
    async def generate_signals(self, symbols: List[str]) -> List[Signal]:
        """Generate theta decay signals."""
        signals = []
        
        for symbol in symbols:
            try:
                signal = await self._analyze_symbol(symbol)
                if signal:
                    signals.append(signal)
            except Exception as e:
                self.logger.error(f"Theta Decay error for {symbol}: {e}")
                continue
        
        return signals
    
    async def _analyze_symbol(self, symbol: str) -> Optional[Signal]:
        """Analyze symbol for theta decay opportunity."""
        iv_rank = self.iv_data_manager.get_iv_rank(symbol)
        if iv_rank is None:
            iv_rank = 50.0

        if iv_rank > 90:
            regime = IVRegime.EXTREME
        elif iv_rank > 70:
            regime = IVRegime.HIGH
        elif iv_rank < 30:
            regime = IVRegime.LOW
        else:
            regime = IVRegime.NORMAL

        rec = self.theta_engine.calculate_optimal_dte(
            iv_rank=iv_rank,
            trend=TrendDirection.NEUTRAL,
            volatility_regime=regime,
            strategy_type="spreads",
        )

        # Pick an entry DTE within our configured bounds.
        dte = int((rec.entry_dte_min + rec.entry_dte_max) / 2)
        dte = max(self.config["optimal_dte_min"], min(dte, self.config["optimal_dte_max"]))

        # Heuristic probability-of-profit so downstream sizing can function.
        # This is intentionally conservative and purely rule-based.
        pop = 0.55
        if iv_rank > 70:
            pop = 0.60
        if iv_rank < 30:
            pop = 0.48
        
        # Only signal if PoP meets minimum
        if pop < self.config["min_probability_of_profit"]:
            return None
        
        # Prefer credit spreads for theta
        strategy = "credit_spread" if is_strategy_allowed(symbol, "credit_spread") else "iron_condor"
        
        # Confidence based on theta efficiency
        confidence = min(pop, 0.95)  # Cap at 95%
        
        return Signal(
            symbol=symbol,
            signal_type=SignalType.SELL,
            signal_source=SignalSource.THETA_DECAY,
            strategy=strategy,
            confidence=confidence,
            timestamp=datetime.now(),
            dte=dte,
            probability_of_profit=pop,
            current_price=None,
            reason=f"Optimal theta decay at {dte} DTE (PoP: {pop:.1%})",
        )


# ============================================================================
# MEAN REVERSION STRATEGY
# ============================================================================

class MeanReversionStrategy:
    """
    Generate signals based on z-score extremes.
    
    Logic:
    - Z-score > +2.0: Price extended high, sell calls or buy puts
    - Z-score < -2.0: Price extended low, sell puts or buy calls
    - Z-score near 0: Exit positions
    """
    
    def __init__(self):
        self.config = RISK_CONFIG
        self.logger = logging.getLogger(__name__)
        
    async def generate_signals(self, symbols: List[str]) -> List[Signal]:
        """Generate mean reversion signals."""
        signals = []
        
        for symbol in symbols:
            try:
                signal = await self._analyze_symbol(symbol)
                if signal:
                    signals.append(signal)
            except Exception as e:
                self.logger.error(f"Mean Reversion error for {symbol}: {e}")
                continue
        
        return signals
    
    async def _analyze_symbol(self, symbol: str) -> Optional[Signal]:
        """Analyze symbol for mean reversion."""
        # Calculate z-score (this would use actual price history)
        z_score = await self._calculate_z_score(symbol)
        
        if z_score is None:
            return None
        
        # ENTRY: Z-score at extremes
        if abs(z_score) >= self.config["z_score_entry"]:
            if z_score > 0:
                # Price too high - sell calls or buy puts
                strategy = "credit_spread"  # Bear call spread
                signal_type = SignalType.SELL
                reason = f"Z-score {z_score:.2f} - price extended high"
            else:
                # Price too low - sell puts or buy calls
                strategy = "put_spread"  # Bull put spread
                signal_type = SignalType.SELL
                reason = f"Z-score {z_score:.2f} - price extended low"
            
            confidence = min(abs(z_score) / 3.0, 1.0)  # Scale to 1.0 at z=3
            
            return Signal(
                symbol=symbol,
                signal_type=signal_type,
                signal_source=SignalSource.MEAN_REVERSION,
                strategy=strategy,
                confidence=confidence,
                timestamp=datetime.now(),
                z_score=z_score,
                dte=30,
                reason=reason,
            )
        
        return None
    
    async def _calculate_z_score(self, symbol: str) -> Optional[float]:
        """
        Calculate z-score for symbol using actual historical price data.
        
        Uses a 20-day rolling mean and standard deviation to compute the
        z-score of the current price (how many stdevs from the mean).
        Returns None if insufficient data is available.
        """
        try:
            import yfinance as yf
            data = yf.download(symbol, period='60d', interval='1d', progress=False)
            if data is None or len(data) < 20:
                self.logger.debug(f"Insufficient data for z-score: {symbol}")
                return None
            
            # Handle multi-index columns from yfinance
            import pandas as pd
            if isinstance(data.columns, pd.MultiIndex):
                closes = data['Close'].iloc[:, 0].dropna().values
            else:
                closes = data['Close'].dropna().values
            
            if len(closes) < 20:
                return None
            
            # 20-day rolling statistics
            recent = closes[-20:]
            mean_price = float(recent.mean())
            std_price = float(recent.std())
            
            if std_price < 1e-8:  # Avoid division by zero
                return 0.0
            
            current_price = float(closes[-1])
            z_score = (current_price - mean_price) / std_price
            
            return z_score
            
        except Exception as e:
            self.logger.debug(f"Z-score calculation failed for {symbol}: {e}")
            return None


# ============================================================================
# DELTA HEDGING STRATEGY
# ============================================================================

class DeltaHedgingStrategy:
    """
    Generate signals to hedge portfolio delta.
    
    Logic:
    - Monitor portfolio delta
    - If delta > +threshold: Hedge with short delta
    - If delta < -threshold: Hedge with long delta
    """
    
    def __init__(self):
        self.config = RISK_CONFIG
        self.logger = logging.getLogger(__name__)
        
    async def generate_signals(self, portfolio_delta: float) -> List[Signal]:
        """
        Generate delta hedging signals.
        
        Args:
            portfolio_delta: Current portfolio delta
            
        Returns:
            List of hedge signals (usually 0 or 1)
        """
        signals = []
        threshold = self.config["delta_hedge_threshold"]
        
        # Portfolio too bullish
        if portfolio_delta > threshold:
            signal = Signal(
                symbol="SPY",  # Use SPY for hedging
                signal_type=SignalType.SELL,
                signal_source=SignalSource.DELTA_HEDGING,
                strategy="put_spread",
                confidence=min(portfolio_delta / (threshold * 2), 1.0),
                timestamp=datetime.now(),
                delta=portfolio_delta,
                dte=30,
                reason=f"Portfolio delta {portfolio_delta:.2f} - need bearish hedge",
            )
            signals.append(signal)
        
        # Portfolio too bearish
        elif portfolio_delta < -threshold:
            signal = Signal(
                symbol="SPY",
                signal_type=SignalType.BUY,
                signal_source=SignalSource.DELTA_HEDGING,
                strategy="call_spread",
                confidence=min(abs(portfolio_delta) / (threshold * 2), 1.0),
                timestamp=datetime.now(),
                delta=portfolio_delta,
                dte=30,
                reason=f"Portfolio delta {portfolio_delta:.2f} - need bullish hedge",
            )
            signals.append(signal)
        
        return signals


# ============================================================================
# MAIN SIGNAL GENERATOR
# ============================================================================

class SignalGenerator:
    """
    Main signal generator combining all strategies.
    """
    
    def __init__(self):
        self.config = RISK_CONFIG
        self.logger = logging.getLogger(__name__)
        
        # Initialize strategies
        self.iv_rank_strategy = IVRankStrategy()
        self.theta_decay_strategy = ThetaDecayStrategy()
        self.mean_reversion_strategy = MeanReversionStrategy()
        self.delta_hedging_strategy = DeltaHedgingStrategy()
        
    async def generate_all_signals(
        self,
        symbols: Optional[List[str]] = None,
        portfolio_delta: float = 0.0,
    ) -> List[Signal]:
        """
        Generate signals from all strategies.
        
        Args:
            symbols: Symbols to analyze (default: universe)
            portfolio_delta: Current portfolio delta for hedging
            
        Returns:
            Combined list of signals from all strategies
        """
        if symbols is None:
            symbols = get_universe()
        
        all_signals = []
        
        # Run strategies in parallel
        iv_signals, theta_signals, mean_rev_signals, delta_signals = await asyncio.gather(
            self.iv_rank_strategy.generate_signals(symbols),
            self.theta_decay_strategy.generate_signals(symbols),
            self.mean_reversion_strategy.generate_signals(symbols),
            self.delta_hedging_strategy.generate_signals(portfolio_delta),
            return_exceptions=True,
        )
        
        # Combine signals (handle exceptions)
        for signals in [iv_signals, theta_signals, mean_rev_signals, delta_signals]:
            if isinstance(signals, list):
                all_signals.extend(signals)
            else:
                self.logger.error(f"Strategy failed: {signals}")
        
        # Sort by confidence descending
        all_signals.sort(key=lambda s: s.confidence, reverse=True)
        
        self.logger.info(f"Generated {len(all_signals)} signals across all strategies")
        return all_signals

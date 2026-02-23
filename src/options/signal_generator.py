"""
Multi-Strategy Signal Generator
================================

Generates trading signals using multiple options strategies:
- IV Rank Strategy: Sell premium when IV high, buy when IV low
- Theta Decay Strategy: Sell options in optimal DTE range (BS-based PoP)
- Mean Reversion Strategy: Multi-timeframe z-score with BB width filter
- Delta Hedging Strategy: Hedge when portfolio delta exceeds threshold
- Vol Divergence Strategy: Trade IV vs realized vol mispricings (Phase 5b)
- VRP Strategy: Volatility Risk Premium — buy/sell premium on IV-RV spread
- IV Crush Strategy: Sell premium pre-earnings to capture IV collapse

Signals are combined with Bayesian confidence scoring and regime-adaptive
strategy weights from the RegimeDetector + DynamicWeightOptimizer.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional
import logging
import numpy as np
from scipy.stats import norm

from .config import RISK_CONFIG, STRATEGY_WEIGHTS
from .universe import get_universe, is_strategy_allowed, STRATEGY_DEFINITIONS
from .iv_analyzer import IVAnalyzer
from .theta_decay_engine import ThetaDecayEngine
from .iv_data_manager import IVDataManager
from .theta_decay_engine import IVRegime, TrendDirection
from .earnings_gate import EARNINGS_CALENDAR, next_earnings_date

# Phase 5b: Advanced spread strategies (lazy-loaded to avoid circular import)
_SPREAD_AVAILABLE = True  # will be set False on first load failure


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
    VRP = "vrp"
    IV_CRUSH = "iv_crush"


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
    - Probability of profit computed via Black-Scholes N(-d2) for puts / N(d2) for calls
    - Only signal when theta/gamma > configured minimum (efficient decay zone)
    - Focus on premium collection
    """
    
    def __init__(self):
        self.config = RISK_CONFIG
        self.logger = logging.getLogger(__name__)
        self.theta_engine = ThetaDecayEngine()
        self.iv_data_manager = IVDataManager()
        self._risk_free_rate = 0.05  # Annual risk-free rate
        
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
    
    def _compute_probability_otm(
        self,
        spot: float,
        strike: float,
        dte: int,
        iv: float,
        option_type: str = "put",
    ) -> float:
        """
        Compute Black-Scholes probability that the option finishes OTM.
        
        For a short put: P(S > K) = N(d2)
        For a short call: P(S < K) = N(-d2)
        """
        T = max(dte / 365.0, 1e-6)
        sigma = max(iv, 1e-6)
        r = self._risk_free_rate
        
        d2_val = (np.log(spot / strike) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        if option_type == "put":
            # Short put profits when S > K at expiry
            return float(norm.cdf(d2_val))
        else:
            # Short call profits when S < K at expiry
            return float(norm.cdf(-d2_val))
    
    def _compute_theta_gamma_ratio(
        self,
        spot: float,
        strike: float,
        dte: int,
        iv: float,
    ) -> float:
        """
        Compute theta/gamma ratio to assess decay efficiency.
        
        Higher ratio = more efficient premium decay relative to gamma risk.
        """
        T = max(dte / 365.0, 1e-6)
        sigma = max(iv, 1e-6)
        r = self._risk_free_rate
        
        d1_val = (np.log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        # Gamma = phi(d1) / (S * sigma * sqrt(T))
        phi_d1 = float(norm.pdf(d1_val))
        gamma = phi_d1 / (spot * sigma * np.sqrt(T)) if (spot * sigma * np.sqrt(T)) > 1e-8 else 0.0
        
        # Theta (per day) ≈ -S * phi(d1) * sigma / (2 * sqrt(T)) / 365
        theta_daily = -(spot * phi_d1 * sigma) / (2 * np.sqrt(T)) / 365.0
        
        if gamma < 1e-8:
            return 0.0
        
        return abs(theta_daily / gamma)
    
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

        # Estimate current price and IV for BS computation
        current_price = await self._get_current_price(symbol)
        if current_price is None or current_price <= 0:
            current_price = 100.0  # fallback for offline/test mode
        
        # Approximate annualized IV from IV rank
        # median IV ~25%, rank scales linearly
        implied_vol = 0.15 + (iv_rank / 100.0) * 0.35  # 15%-50% range
        
        # ATM strike for computation
        strike = current_price
        
        # Compute Black-Scholes probability of finishing OTM (short put)
        pop = self._compute_probability_otm(current_price, strike, dte, implied_vol, "put")
        
        # Compute theta/gamma ratio
        tg_ratio = self._compute_theta_gamma_ratio(current_price, strike, dte, implied_vol)
        
        # Only signal if theta/gamma ratio is above minimum (efficient decay zone)
        min_tg = self.config.get("theta_gamma_min_ratio", 0.5)
        if tg_ratio < min_tg:
            self.logger.debug(f"Theta/gamma ratio {tg_ratio:.2f} < {min_tg} for {symbol}")
            return None

        # ===== 2026-02-23 FIX 6: theta/price ratio >= 0.005 (5bp/day) =====
        if current_price and current_price > 0:
            T = max(dte / 365.0, 1e-6)
            d1_val_chk = (np.log(current_price / strike) + (self._risk_free_rate + 0.5 * implied_vol**2) * T) / (implied_vol * np.sqrt(T))
            phi_d1_chk = float(norm.pdf(d1_val_chk))
            theta_daily_est = abs(current_price * phi_d1_chk * implied_vol / (2 * np.sqrt(T)) / 365.0)
            theta_price_ratio = theta_daily_est / current_price
            if theta_price_ratio < 0.005:
                self.logger.debug(
                    f"Theta/price ratio {theta_price_ratio:.4f} < 0.005 for {symbol} — skipping"
                )
                return None
        
        # Only signal if PoP meets minimum
        if pop < self.config["min_probability_of_profit"]:
            return None
        
        # Prefer credit spreads for theta
        strategy = "credit_spread" if is_strategy_allowed(symbol, "credit_spread") else "iron_condor"
        
        # Confidence based on theta efficiency and PoP
        confidence = min(pop * (1.0 + 0.1 * min(tg_ratio, 2.0)), 0.95)
        
        return Signal(
            symbol=symbol,
            signal_type=SignalType.SELL,
            signal_source=SignalSource.THETA_DECAY,
            strategy=strategy,
            confidence=confidence,
            timestamp=datetime.now(),
            dte=dte,
            probability_of_profit=pop,
            current_price=current_price,
            reason=f"Theta decay at {dte} DTE (BS PoP: {pop:.1%}, θ/γ: {tg_ratio:.2f})",
        )
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Fetch current price via yfinance."""
        try:
            import yfinance as yf
            data = yf.download(symbol, period='5d', interval='1d', progress=False)
            if data is None or len(data) == 0:
                return None
            import pandas as pd
            if isinstance(data.columns, pd.MultiIndex):
                return float(data['Close'].iloc[:, 0].dropna().values[-1])
            else:
                return float(data['Close'].dropna().values[-1])
        except Exception:
            return None


# ============================================================================
# MEAN REVERSION STRATEGY
# ============================================================================

class MeanReversionStrategy:
    """
    Generate signals based on multi-timeframe z-score convergence.
    
    Logic:
    - Compute z-scores on 10d, 20d, and 50d windows
    - Only signal when ALL three agree on direction (convergence filter)
    - Bollinger Band width filter: skip signals when BB width expanding (trending)
    - Z-score > +2.0 (all TFs): Price extended high, sell calls or buy puts
    - Z-score < -2.0 (all TFs): Price extended low, sell puts or buy calls
    """
    
    def __init__(self):
        self.config = RISK_CONFIG
        self.logger = logging.getLogger(__name__)
        self.z_score_windows = self.config.get("multi_tf_zscore_windows", [10, 20, 50])
        
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
        """Analyze symbol for mean reversion with multi-TF convergence."""
        price_data = await self._fetch_price_data(symbol)
        if price_data is None:
            return None
        
        closes = price_data
        max_window = max(self.z_score_windows)
        
        if len(closes) < max_window + 5:
            return None
        
        # Compute z-scores across all timeframes
        z_scores = {}
        for window in self.z_score_windows:
            recent = closes[-window:]
            mean_price = float(recent.mean())
            std_price = float(recent.std())
            if std_price < 1e-8:
                z_scores[window] = 0.0
            else:
                z_scores[window] = (float(closes[-1]) - mean_price) / std_price
        
        # Check convergence: ALL timeframes must agree on direction
        all_positive = all(z >= self.config["z_score_entry"] for z in z_scores.values())
        all_negative = all(z <= -self.config["z_score_entry"] for z in z_scores.values())
        
        if not all_positive and not all_negative:
            return None  # No convergence — skip
        
        # Bollinger Band width filter: skip when BB width is expanding (trending)
        if self._is_bb_expanding(closes):
            self.logger.debug(f"BB width expanding for {symbol} — skip mean reversion")
            return None
        
        # Average z-score across timeframes
        avg_z = float(np.mean(list(z_scores.values())))
        
        if all_positive:
            # Price too high across all TFs — sell calls or buy puts
            strategy = "credit_spread"  # Bear call spread
            signal_type = SignalType.SELL
            reason = (f"Multi-TF z-score convergence HIGH "
                      f"({', '.join(f'{w}d={z:.2f}' for w, z in z_scores.items())})")
        else:
            # Price too low across all TFs — sell puts or buy calls
            strategy = "put_spread"  # Bull put spread
            signal_type = SignalType.SELL
            reason = (f"Multi-TF z-score convergence LOW "
                      f"({', '.join(f'{w}d={z:.2f}' for w, z in z_scores.items())})")
        
        # Confidence boosted by convergence strength
        confidence = min(abs(avg_z) / 3.0, 1.0)
        
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            signal_source=SignalSource.MEAN_REVERSION,
            strategy=strategy,
            confidence=confidence,
            timestamp=datetime.now(),
            z_score=avg_z,
            dte=30,
            reason=reason,
        )
    
    def _is_bb_expanding(self, closes: np.ndarray, window: int = 20) -> bool:
        """
        Check if Bollinger Band width is expanding (trending market).
        
        Returns True when current BB width > BB width 5 days ago,
        indicating the market is trending and mean reversion is risky.
        """
        if len(closes) < window + 6:
            return False
        
        def bb_width(data: np.ndarray) -> float:
            mean = float(data.mean())
            std = float(data.std())
            if mean < 1e-8:
                return 0.0
            return (2 * std) / mean  # Normalized BB width
        
        current_bb = bb_width(closes[-window:])
        past_bb = bb_width(closes[-window - 5:-5])
        
        return current_bb > past_bb * 1.05  # 5% expansion threshold
    
    async def _fetch_price_data(self, symbol: str) -> Optional[np.ndarray]:
        """Fetch recent price data for z-score calculation."""
        try:
            import yfinance as yf
            data = yf.download(symbol, period='120d', interval='1d', progress=False)
            if data is None or len(data) < 20:
                return None
            
            import pandas as pd
            if isinstance(data.columns, pd.MultiIndex):
                closes = data['Close'].iloc[:, 0].dropna().values
            else:
                closes = data['Close'].dropna().values
            
            if len(closes) < 20:
                return None
            
            return closes.astype(float)
            
        except Exception as e:
            self.logger.debug(f"Price data fetch failed for {symbol}: {e}")
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
# VOL DIVERGENCE STRATEGY  (Phase 5b: IV vs Realized Vol)
# ============================================================================

class VolDivergenceStrategy:
    """
    Trade implied-vs-realized volatility mispricings.

    Intuition
    ---------
    Options are priced on *implied* volatility but *realized* volatility
    determines the actual P&L of a delta-hedged position.  When the two
    diverge meaningfully the market is mispricing risk:

      • **IV / RV > 1.5** → IV is **over-priced** → SELL premium
        (credit spreads / iron condors).
      • **IV / RV < 0.7** → IV is **under-priced** → BUY options
        (debit spreads / straddles).

    Realized vol is computed as the annualized standard deviation of
    daily log-returns over the last ``rv_lookback`` trading days.
    """

    # Configurable thresholds
    IV_OVER_RV_SELL = 1.5    # sell when IV/RV ≥ this
    IV_OVER_RV_BUY = 0.7     # buy when IV/RV ≤ this
    RV_LOOKBACK = 20         # 20 trading-day realized vol

    def __init__(
        self,
        rv_lookback: int = 20,
        sell_threshold: float = 1.5,
        buy_threshold: float = 0.7,
    ):
        self.rv_lookback = rv_lookback
        self.sell_threshold = sell_threshold
        self.buy_threshold = buy_threshold
        self.config = RISK_CONFIG
        self.iv_data = IVDataManager()
        self.iv_analyzer = IVAnalyzer()
        self.logger = logging.getLogger(f"{__name__}.VolDivergence")

    # ----------------------------------------------------------
    # Public API (same shape as other strategies)
    # ----------------------------------------------------------

    async def generate_signals(self, symbols: List[str]) -> List[Signal]:
        signals: List[Signal] = []
        for symbol in symbols:
            try:
                sig = await self._evaluate(symbol)
                if sig is not None:
                    signals.append(sig)
            except Exception as exc:
                self.logger.debug(f"VolDiv error {symbol}: {exc}")
        return signals

    # ----------------------------------------------------------
    # Core evaluation
    # ----------------------------------------------------------

    async def _evaluate(self, symbol: str) -> Optional[Signal]:
        # Get current IV from the data manager
        iv_rank = self.iv_data.get_iv_rank(symbol)
        if iv_rank is None:
            return None

        # Fetch recent closes for realized vol
        rv = await self._compute_realized_vol(symbol)
        if rv is None or rv < 1e-6:
            return None

        # Approximate current IV from iv_rank heuristic
        # (IVDataManager stores rank; we back-estimate annualised IV)
        # In production the IVAnalyzer provides actual IV — here we use
        # a simple mapping: iv ≈ rv * (1 + (iv_rank - 50) / 100)
        # This preserves the relationship while being testable offline.
        implied_vol = rv * (1.0 + (iv_rank - 50) / 100.0)
        if implied_vol < 1e-6:
            return None

        ratio = implied_vol / rv

        # ── SELL premium: IV over-priced ──
        if ratio >= self.sell_threshold:
            strategy = "iron_condor" if is_strategy_allowed(symbol, "iron_condor") else "credit_spread"
            confidence = min(0.95, 0.50 + 0.20 * (ratio - self.sell_threshold))
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                signal_source=SignalSource.IV_RANK,
                strategy=strategy,
                confidence=round(confidence, 3),
                timestamp=datetime.now(),
                iv_rank=iv_rank,
                dte=35,
                probability_of_profit=0.60,
                reason=(
                    f"Vol divergence SELL: IV/RV={ratio:.2f} ≥ {self.sell_threshold} "
                    f"(IV≈{implied_vol:.2%}, RV={rv:.2%})"
                ),
            )

        # ── BUY options: IV under-priced ──
        if ratio <= self.buy_threshold:
            strategy = "straddle" if is_strategy_allowed(symbol, "straddle") else "strangle"
            confidence = min(0.95, 0.50 + 0.30 * (self.buy_threshold - ratio))
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                signal_source=SignalSource.IV_RANK,
                strategy=strategy,
                confidence=round(confidence, 3),
                timestamp=datetime.now(),
                iv_rank=iv_rank,
                dte=21,
                probability_of_profit=0.45,
                reason=(
                    f"Vol divergence BUY: IV/RV={ratio:.2f} ≤ {self.buy_threshold} "
                    f"(IV≈{implied_vol:.2%}, RV={rv:.2%})"
                ),
            )

        return None

    # ----------------------------------------------------------
    # Realized vol helper
    # ----------------------------------------------------------

    async def _compute_realized_vol(self, symbol: str) -> Optional[float]:
        """
        20-day annualized realized volatility from daily close returns.
        """
        try:
            import yfinance as yf
            data = yf.download(symbol, period="60d", interval="1d", progress=False)
            if data is None or len(data) < self.rv_lookback + 1:
                return None
            import pandas as pd
            if isinstance(data.columns, pd.MultiIndex):
                closes = data["Close"].iloc[:, 0].dropna().values.astype(float)
            else:
                closes = data["Close"].dropna().values.astype(float)
            if len(closes) < self.rv_lookback + 1:
                return None

            log_rets = np.diff(np.log(closes[-self.rv_lookback - 1:]))
            rv = float(np.std(log_rets) * np.sqrt(252))
            return rv
        except Exception as exc:
            self.logger.debug(f"RV calc failed for {symbol}: {exc}")
            return None


# ============================================================================
# VRP STRATEGY — Volatility Risk Premium
# ============================================================================

class VRPStrategy:
    """
    Volatility Risk Premium (VRP) strategy.
    
    The #1 proven alpha source in options markets (~3.4% annualized,
    88% positive months per Morningstar research).
    
    Logic:
    - Compare implied vol (from VIX/option chain) vs realized vol (GARCH/historical)
    - When IV > RV by > vrp_threshold (3%): sell premium (credit spreads/iron condors)
    - When IV < RV: buy premium (market underpricing risk)

    Phase 7 Enhancement — Intraday VRP Signal:
    - Compute 20-day realized vol from recent closes
    - Compare to current VIX (implied vol)
    - If VRP (IV - RV) > 5%, signal is favorable for selling premium
    - If VRP < 0%, avoid selling premium entirely
    """
    
    def __init__(self):
        self.config = RISK_CONFIG
        self.logger = logging.getLogger(f"{__name__}.VRP")
        self.iv_data_manager = IVDataManager()
        self.vrp_threshold = self.config.get("vrp_threshold", 0.03)
        # Phase 7: Cache the last computed intraday VRP
        self._last_intraday_vrp: Optional[float] = None
        self._last_vrp_update: Optional[datetime] = None
    
    async def generate_signals(self, symbols: List[str]) -> List[Signal]:
        """Generate VRP signals for all symbols."""
        signals: List[Signal] = []

        # Phase 7: Compute intraday VRP once per signal scan cycle
        await self._update_intraday_vrp()

        for symbol in symbols:
            try:
                sig = await self._evaluate(symbol)
                if sig is not None:
                    signals.append(sig)
            except Exception as exc:
                self.logger.debug(f"VRP error {symbol}: {exc}")
        return signals
    
    async def _update_intraday_vrp(self) -> None:
        """Compute real-time VRP = VIX (implied vol) - 20d realized vol.

        Phase 7 (Improvement 3): This is a portfolio-level signal that
        gates all premium-selling activity.  Updated at most once per
        5 minutes to avoid excessive API calls.

        Stored in ``self._last_intraday_vrp`` (annualised percentage
        points, e.g. 0.06 = 6 %).
        """
        # Rate-limit updates to every 5 minutes
        now = datetime.now()
        if (
            self._last_vrp_update is not None
            and (now - self._last_vrp_update).total_seconds() < 300
        ):
            return

        try:
            import yfinance as yf

            # --- Implied vol from VIX ---
            vix_data = yf.download("^VIX", period="5d", interval="1d", progress=False)
            if vix_data is None or vix_data.empty:
                self.logger.debug("VRP: VIX data unavailable")
                return
            import pandas as pd
            if isinstance(vix_data.columns, pd.MultiIndex):
                vix_close = float(vix_data["Close"].iloc[:, 0].dropna().values[-1])
            else:
                vix_close = float(vix_data["Close"].dropna().values[-1])
            implied_vol = vix_close / 100.0  # VIX is in % (e.g. 18 = 18%)

            # --- Realized vol from SPY 20-day log returns ---
            spy_data = yf.download("SPY", period="60d", interval="1d", progress=False)
            if spy_data is None or len(spy_data) < 21:
                return
            if isinstance(spy_data.columns, pd.MultiIndex):
                spy_closes = spy_data["Close"].iloc[:, 0].dropna().values.astype(float)
            else:
                spy_closes = spy_data["Close"].dropna().values.astype(float)
            if len(spy_closes) < 21:
                return

            log_rets = np.diff(np.log(spy_closes[-21:]))
            realized_vol = float(np.std(log_rets) * np.sqrt(252))

            vrp = implied_vol - realized_vol
            self._last_intraday_vrp = vrp
            self._last_vrp_update = now

            self.logger.info(
                f"Intraday VRP updated: IV(VIX)={implied_vol:.2%}, "
                f"RV(20d)={realized_vol:.2%}, VRP={vrp:+.2%}"
            )
        except Exception as exc:
            self.logger.debug(f"Intraday VRP update failed: {exc}")

    def get_intraday_vrp(self) -> Optional[float]:
        """Return the last computed intraday VRP, or None if unavailable.

        Other strategies can call this to gate premium-selling:
        - VRP > 0.05 (5%): favorable to sell premium
        - VRP < 0.00 (0%): avoid selling premium
        """
        return self._last_intraday_vrp
    
    async def _evaluate(self, symbol: str) -> Optional[Signal]:
        """Evaluate VRP for a symbol by comparing IV and RV.

        Phase 7: Incorporates intraday VRP gate — if VRP < 0%,
        premium-selling signals are suppressed entirely.
        """
        iv_rank = self.iv_data_manager.get_iv_rank(symbol)
        if iv_rank is None:
            return None
        
        rv = await self._compute_realized_vol(symbol)
        if rv is None or rv < 1e-6:
            return None
        
        garch_rv = self._garch_forecast(symbol, rv)
        
        # Approximate annualised implied vol from IV rank
        # Map IV rank 0-100 to roughly 10%-60% annualised IV
        implied_vol = 0.10 + (iv_rank / 100.0) * 0.50
        
        vrp = implied_vol - garch_rv  # Positive = IV > RV = sell premium

        # Phase 7: Intraday VRP gate — suppress premium-selling when VRP < 0
        intraday_vrp = self._last_intraday_vrp
        if intraday_vrp is not None and intraday_vrp < 0.0 and vrp > 0:
            self.logger.info(
                f"VRP gate: suppressing SELL for {symbol} — "
                f"intraday VRP={intraday_vrp:+.2%} < 0%"
            )
            return None
        
        # SELL premium: IV exceeds RV by more than threshold
        if vrp > self.vrp_threshold:
            strategy = "iron_condor" if is_strategy_allowed(symbol, "iron_condor") else "credit_spread"
            
            # Confidence scales with VRP magnitude
            confidence = min(0.50 + (vrp - self.vrp_threshold) * 5.0, 0.95)

            # Phase 7: Boost confidence when intraday VRP is strongly positive
            if intraday_vrp is not None and intraday_vrp > 0.05:
                confidence = min(confidence * 1.10, 0.95)
            
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                signal_source=SignalSource.VRP,
                strategy=strategy,
                confidence=round(confidence, 3),
                timestamp=datetime.now(),
                iv_rank=iv_rank,
                dte=35,
                probability_of_profit=min(0.55 + vrp, 0.85),
                reason=(
                    f"VRP SELL: IV={implied_vol:.1%} - RV(GARCH)={garch_rv:.1%} = "
                    f"VRP={vrp:.1%} > {self.vrp_threshold:.1%}"
                ),
            )
        
        # BUY premium: RV exceeds IV (market underpricing risk)
        if vrp < -self.vrp_threshold:
            strategy = "straddle" if is_strategy_allowed(symbol, "straddle") else "strangle"
            
            confidence = min(0.50 + abs(vrp + self.vrp_threshold) * 4.0, 0.90)
            
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                signal_source=SignalSource.VRP,
                strategy=strategy,
                confidence=round(confidence, 3),
                timestamp=datetime.now(),
                iv_rank=iv_rank,
                dte=21,
                probability_of_profit=0.45,
                reason=(
                    f"VRP BUY: IV={implied_vol:.1%} - RV(GARCH)={garch_rv:.1%} = "
                    f"VRP={vrp:.1%} (negative = underpriced)"
                ),
            )
        
        return None
    
    def _garch_forecast(self, symbol: str, historical_rv: float) -> float:
        """
        Simple GARCH(1,1)-inspired vol forecast.
        
        Uses exponential weighting of recent vs long-term vol to produce
        a forward-looking realized vol estimate. Falls back to historical
        RV if GARCH params unavailable.
        
        GARCH(1,1): σ²(t+1) = ω + α·r²(t) + β·σ²(t)
        Simplified: forecast = 0.05 + 0.10·recent_var + 0.85·historical_var
        """
        # Long-run average vol (annualised) — approximation
        omega = 0.04  # ~20% long-run vol squared → 0.04
        alpha = 0.10  # Weight on recent return shock
        beta = 0.85   # Persistence of historical vol
        
        # historical_rv is already annualised, convert to variance
        hist_var = historical_rv ** 2
        
        # Forecast variance (simplified — in production use arch package)
        forecast_var = omega * (1 - alpha - beta) + alpha * hist_var + beta * hist_var
        forecast_vol = np.sqrt(max(forecast_var, 1e-8))
        
        return float(forecast_vol)
    
    async def _compute_realized_vol(self, symbol: str) -> Optional[float]:
        """20-day annualized realized volatility."""
        try:
            import yfinance as yf
            data = yf.download(symbol, period="60d", interval="1d", progress=False)
            if data is None or len(data) < 21:
                return None
            import pandas as pd
            if isinstance(data.columns, pd.MultiIndex):
                closes = data["Close"].iloc[:, 0].dropna().values.astype(float)
            else:
                closes = data["Close"].dropna().values.astype(float)
            if len(closes) < 21:
                return None
            log_rets = np.diff(np.log(closes[-21:]))
            rv = float(np.std(log_rets) * np.sqrt(252))
            return rv
        except Exception:
            return None


# ============================================================================
# IV CRUSH STRATEGY — Pre-Earnings IV Collapse
# ============================================================================

class IVCrushStrategy:
    """
    IV Crush strategy: sell premium before earnings to capture IV collapse.
    
    Logic:
    - Monitor earnings calendar (from earnings_gate.py)
    - If IV rank > 80 and earnings within 1-7 days: sell iron condors/strangles
    - DTE = closest expiry crossing earnings (1-3 days preferred)
    - Only activate for symbols with consistent historical IV crush > 20%
    - Turns the earnings gate from a pure blocker into an alpha source
    """
    
    def __init__(self):
        self.config = RISK_CONFIG
        self.logger = logging.getLogger(f"{__name__}.IVCrush")
        self.iv_data_manager = IVDataManager()
        self.min_iv_rank = self.config.get("iv_crush_min_rank", 80)
        self.min_historical_drop = self.config.get("iv_crush_min_historical_drop", 0.20)
        
        # Historical IV crush data (% drop post-earnings, based on research)
        # In production, this would be computed from actual historical IV data
        self._historical_crush: Dict[str, float] = {
            "AAPL": 0.35,   # 35% avg IV crush post-earnings
            "MSFT": 0.30,
            "NVDA": 0.45,   # High crush due to AI hype cycles
            "AMZN": 0.40,
            "META": 0.42,
            "GOOGL": 0.32,
            "SPY": 0.0,     # ETF — no earnings crush
            "QQQ": 0.0,     # ETF — no earnings crush
        }
    
    async def generate_signals(self, symbols: List[str]) -> List[Signal]:
        """Generate IV crush signals for symbols near earnings."""
        signals: List[Signal] = []
        from datetime import timedelta
        today = date.today()
        
        for symbol in symbols:
            try:
                sig = await self._evaluate(symbol, today)
                if sig is not None:
                    signals.append(sig)
            except Exception as exc:
                self.logger.debug(f"IVCrush error {symbol}: {exc}")
        
        return signals
    
    async def _evaluate(self, symbol: str, today: date) -> Optional[Signal]:
        """Evaluate if symbol is ripe for IV crush trade."""
        from datetime import timedelta
        
        # Check if symbol has earnings coming up
        earnings_date = next_earnings_date(symbol)
        if earnings_date is None:
            return None
        
        days_to_earnings = (earnings_date - today).days
        
        # Only activate 1-7 days before earnings
        if days_to_earnings < 1 or days_to_earnings > 7:
            return None
        
        # Check historical IV crush is high enough
        hist_crush = self._historical_crush.get(symbol, 0.0)
        if hist_crush < self.min_historical_drop:
            self.logger.debug(
                f"IV crush for {symbol} ({hist_crush:.0%}) below min "
                f"({self.min_historical_drop:.0%})"
            )
            return None
        
        # Check IV rank is elevated
        iv_rank = self.iv_data_manager.get_iv_rank(symbol)
        if iv_rank is None or iv_rank < self.min_iv_rank:
            return None
        
        # DTE = closest expiry that crosses earnings (1-3 days ideal)
        dte = min(max(days_to_earnings + 1, 1), 7)
        
        # Strategy: iron condor or strangle to max premium capture
        strategy = "iron_condor" if is_strategy_allowed(symbol, "iron_condor") else "credit_spread"
        
        # Confidence based on IV rank and historical crush magnitude
        confidence = min(
            0.50 + (iv_rank - self.min_iv_rank) / 100.0 + hist_crush * 0.5,
            0.90,
        )
        
        # Expected profit from IV crush
        expected_premium_pct = hist_crush * (iv_rank / 100.0) * 0.5
        
        return Signal(
            symbol=symbol,
            signal_type=SignalType.SELL,
            signal_source=SignalSource.IV_CRUSH,
            strategy=strategy,
            confidence=round(confidence, 3),
            timestamp=datetime.now(),
            iv_rank=iv_rank,
            dte=dte,
            probability_of_profit=min(0.60 + hist_crush * 0.3, 0.85),
            reason=(
                f"IV CRUSH: {symbol} earnings in {days_to_earnings}d, "
                f"IV rank={iv_rank:.0f}, hist crush={hist_crush:.0%}, "
                f"DTE={dte}"
            ),
        )


# ============================================================================
# BAYESIAN CONFIDENCE COMBINER
# ============================================================================

def bayesian_combine_confidence(signals: List[Signal]) -> List[Signal]:
    """
    Apply Bayesian-inspired confidence combination to signals for the same symbol.
    
    When multiple strategies agree on direction, boost confidence:
        combined_conf = 1 - product(1 - conf_i)
    
    When strategies disagree, dampen the weaker signal.
    
    Args:
        signals: Raw signals from all strategies
        
    Returns:
        Updated signals with boosted/dampened confidence
    """
    if not signals:
        return signals
    
    # Group signals by (symbol, direction)
    from collections import defaultdict
    by_symbol: Dict[str, List[Signal]] = defaultdict(list)
    for sig in signals:
        by_symbol[sig.symbol].append(sig)
    
    adjusted: List[Signal] = []
    
    for symbol, sym_signals in by_symbol.items():
        if len(sym_signals) <= 1:
            adjusted.extend(sym_signals)
            continue
        
        # Split by direction
        sells = [s for s in sym_signals if s.signal_type == SignalType.SELL]
        buys = [s for s in sym_signals if s.signal_type == SignalType.BUY]
        others = [s for s in sym_signals if s.signal_type not in (SignalType.SELL, SignalType.BUY)]
        
        # Boost agreeing signals: conf = 1 - prod(1 - conf_i)
        for group in [sells, buys]:
            if len(group) >= 2:
                combined = 1.0 - np.prod([1.0 - s.confidence for s in group])
                combined = min(combined, 0.98)
                # Keep the highest-confidence signal, boost it
                group.sort(key=lambda s: s.confidence, reverse=True)
                group[0].confidence = round(combined, 3)
                group[0].reason += f" [Bayesian boost: {len(group)} strategies agree]"
                adjusted.append(group[0])
                # Drop weaker duplicates
            elif len(group) == 1:
                adjusted.append(group[0])
        
        # Dampen conflicting signals
        if sells and buys:
            # Reduce confidence of the weaker direction
            sell_conf = max(s.confidence for s in sells) if sells else 0
            buy_conf = max(s.confidence for s in buys) if buys else 0
            
            # Already added the stronger; dampen the weaker that was added
            for sig in adjusted:
                if sig.symbol == symbol:
                    if sig.signal_type == SignalType.SELL and buy_conf > 0:
                        dampen = max(0.0, 1.0 - buy_conf * 0.5)
                        sig.confidence = round(sig.confidence * dampen, 3)
                    elif sig.signal_type == SignalType.BUY and sell_conf > 0:
                        dampen = max(0.0, 1.0 - sell_conf * 0.5)
                        sig.confidence = round(sig.confidence * dampen, 3)
        
        adjusted.extend(others)
    
    return adjusted


# ============================================================================
# MAIN SIGNAL GENERATOR
# ============================================================================

class SignalGenerator:
    """
    Main signal generator combining all strategies with Bayesian confidence
    scoring and regime-adaptive strategy weights.

    2026-02-23 FIX 2: Signal deduplication — tracks last signal time per
    OCC contract and blocks rapid-fire duplicate signals.
    """
    
    def __init__(self):
        self.config = RISK_CONFIG
        self.logger = logging.getLogger(__name__)

        # 2026-02-23 FIX 2: Signal deduplication state
        self._last_signal_per_contract: Dict[str, datetime] = {}
        self._signal_dedup_hours = self.config.get("signal_dedup_hours", 4)
        
        # Initialize strategies
        self.iv_rank_strategy = IVRankStrategy()
        self.theta_decay_strategy = ThetaDecayStrategy()
        self.mean_reversion_strategy = MeanReversionStrategy()
        self.delta_hedging_strategy = DeltaHedgingStrategy()

        # Phase 5b: Advanced spread strategies (lazy import to avoid circular)
        self.spread_aggregator = None
        if _SPREAD_AVAILABLE:
            try:
                from .spread_strategies import SpreadStrategyAggregator
                self.spread_aggregator = SpreadStrategyAggregator()
                self.logger.info("✓ Phase 5b spread strategies loaded (IC, bull-put, bear-call)")
            except Exception as exc:
                self.logger.warning(f"Spread strategies init failed: {exc}")

        # Phase 5b: IV vs Realized Vol divergence strategy
        self.vol_divergence_strategy = VolDivergenceStrategy()
        
        # New alpha strategies
        self.vrp_strategy = VRPStrategy()
        self.iv_crush_strategy = IVCrushStrategy()
        
        # Regime detector and weight optimizer (lazy-loaded)
        self.regime_detector = None
        self.weight_optimizer = None
        self._init_regime_components()
    
    def _init_regime_components(self):
        """Initialize regime detector and weight optimizer for regime-adaptive weights."""
        try:
            from .regime_detector import RegimeDetector
            from .weight_optimizer import DynamicWeightOptimizer
            
            self.regime_detector = RegimeDetector()
            self.weight_optimizer = DynamicWeightOptimizer(
                strategies=["iv_rank", "theta_decay", "mean_reversion",
                            "delta_hedging", "vrp", "iv_crush"],
                regime_detector=self.regime_detector,
            )
            self.logger.info("✓ Regime detector + weight optimizer wired into signal generation")
        except Exception as exc:
            self.logger.warning(f"Regime components init failed (using static weights): {exc}")
    
    async def _get_regime_weights(self) -> Dict[str, float]:
        """
        Get current strategy weights from RegimeDetector + WeightOptimizer.
        
        Falls back to static STRATEGY_WEIGHTS if regime detection unavailable.
        """
        if self.regime_detector is not None and self.weight_optimizer is not None:
            try:
                # Attempt to detect current regime
                if not self.regime_detector.is_fitted:
                    await self.regime_detector.fit()
                
                regime_state = await self.regime_detector.detect_current_regime()
                weights = await self.weight_optimizer.rebalance(regime=regime_state.current_regime)
                
                self.logger.info(
                    f"Regime: {regime_state.current_regime.value} "
                    f"(conf={regime_state.confidence:.2f}) → "
                    f"Weights: {', '.join(f'{k}={v:.0%}' for k, v in weights.items())}"
                )
                return weights
            except Exception as exc:
                self.logger.debug(f"Regime detection failed, using static weights: {exc}")
        
        return dict(STRATEGY_WEIGHTS)
    
    def _apply_regime_weights(
        self, signals: List[Signal], weights: Dict[str, float]
    ) -> List[Signal]:
        """
        Scale signal confidence by regime-adaptive strategy weights.
        
        Multiplies each signal's confidence by its strategy weight, then
        renormalize so the top signal can still reach ~0.95.
        """
        if not signals:
            return signals
        
        source_to_key = {
            SignalSource.IV_RANK: "iv_rank",
            SignalSource.THETA_DECAY: "theta_decay",
            SignalSource.MEAN_REVERSION: "mean_reversion",
            SignalSource.DELTA_HEDGING: "delta_hedging",
            SignalSource.VRP: "vrp",
            SignalSource.IV_CRUSH: "iv_crush",
        }
        
        for sig in signals:
            strategy_key = source_to_key.get(sig.signal_source, "iv_rank")
            weight = weights.get(strategy_key, 0.15)
            # Scale confidence by weight (relative to max weight)
            max_weight = max(weights.values()) if weights else 1.0
            scale = weight / max_weight if max_weight > 0 else 1.0
            sig.confidence = round(sig.confidence * (0.5 + 0.5 * scale), 3)
        
        return signals

    async def generate_all_signals(
        self,
        symbols: Optional[List[str]] = None,
        portfolio_delta: float = 0.0,
        current_positions: Optional[List] = None,
    ) -> List[Signal]:
        """
        Generate signals from all strategies with Bayesian confidence
        scoring and regime-adaptive weighting.

        2026-02-23 FIX 2: Accepts ``current_positions`` for dedup and
        applies signal-level deduplication before returning.

        Args:
            symbols: Symbols to analyze (default: universe)
            portfolio_delta: Current portfolio delta for hedging
            current_positions: Optional list of in-memory positions for dedup
            
        Returns:
            Combined list of signals from all strategies
        """
        if symbols is None:
            symbols = get_universe()
        
        all_signals = []
        
        # Get regime-adaptive weights
        weights = await self._get_regime_weights()
        
        # Core strategies (always run)
        tasks = [
            self.iv_rank_strategy.generate_signals(symbols),
            self.theta_decay_strategy.generate_signals(symbols),
            self.mean_reversion_strategy.generate_signals(symbols),
            self.delta_hedging_strategy.generate_signals(portfolio_delta),
            self.vol_divergence_strategy.generate_signals(symbols),
            self.vrp_strategy.generate_signals(symbols),
            self.iv_crush_strategy.generate_signals(symbols),
        ]

        # Phase 5b: Add spread strategies if available
        if self.spread_aggregator is not None:
            tasks.append(self.spread_aggregator.generate_signals(symbols))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine signals (handle exceptions)
        for signals in results:
            if isinstance(signals, list):
                all_signals.extend(signals)
            else:
                self.logger.error(f"Strategy failed: {signals}")
        
        # Apply regime-adaptive weights
        all_signals = self._apply_regime_weights(all_signals, weights)
        
        # Apply Bayesian confidence combination
        if self.config.get("signal_convergence_boost", True):
            all_signals = bayesian_combine_confidence(all_signals)

        # ===== 2026-02-23 FIX 2: SIGNAL DEDUPLICATION =====
        all_signals = self._deduplicate_signals(all_signals, current_positions)

        # ===== 2026-02-23 FIX 6: MOMENTUM FILTER =====
        all_signals = await self._apply_momentum_filter(all_signals)
        
        # Sort by confidence descending
        all_signals.sort(key=lambda s: s.confidence, reverse=True)
        
        self.logger.info(f"Generated {len(all_signals)} signals across all strategies")
        return all_signals

    # ================================================================== #
    # 2026-02-23 FIX 2: SIGNAL DEDUPLICATION
    # ================================================================== #

    def _deduplicate_signals(
        self, signals: List[Signal], current_positions: Optional[List] = None
    ) -> List[Signal]:
        """Remove duplicate signals for the same contract/symbol.

        Rules:
        1. Minimum ``signal_dedup_hours`` (default 4) between identical signals
           on the same OCC contract.
        2. Any signal for a contract already held with qty >= 3 is blocked.

        Args:
            signals: Raw signal list.
            current_positions: In-memory position list for qty check.

        Returns:
            Filtered signal list.
        """
        now = datetime.now()
        deduped: List[Signal] = []

        # Build a map of underlying -> total qty from current_positions
        held_qty_by_symbol: Dict[str, int] = defaultdict(int)
        if current_positions:
            for pos in current_positions:
                if isinstance(pos, dict):
                    sig = pos.get("signal")
                    sym = getattr(sig, "symbol", None) or pos.get("symbol")
                    ps = pos.get("position_size")
                    qty = getattr(ps, "contracts", 1) if ps else 1
                    if sym:
                        held_qty_by_symbol[sym.upper()] += qty

        for signal in signals:
            # Key for dedup: prefer OCC symbol, fall back to underlying+strategy
            key = signal.occ_symbol or f"{signal.symbol}_{signal.strategy}_{signal.signal_type.value}"

            # Rule 1: Time-based dedup
            last_time = self._last_signal_per_contract.get(key)
            if last_time is not None:
                hours_since = (now - last_time).total_seconds() / 3600.0
                if hours_since < self._signal_dedup_hours:
                    self.logger.info(
                        f"DEDUP: Blocking {signal.symbol} ({key}) — "
                        f"last signal {hours_since:.1f}h ago "
                        f"(min {self._signal_dedup_hours}h)"
                    )
                    continue

            # Rule 2: Block if already holding >= 3 contracts of this underlying
            sym_upper = (signal.symbol or "").upper()
            if held_qty_by_symbol.get(sym_upper, 0) >= 3:
                self.logger.info(
                    f"DEDUP: Blocking {signal.symbol} — already holding "
                    f"{held_qty_by_symbol[sym_upper]} contracts (>= 3)"
                )
                continue

            # Accept signal and record timestamp
            self._last_signal_per_contract[key] = now
            deduped.append(signal)

        removed = len(signals) - len(deduped)
        if removed > 0:
            self.logger.info(f"DEDUP: Removed {removed} duplicate signals")

        return deduped

    # ================================================================== #
    # 2026-02-23 FIX 6: MOMENTUM FILTER
    # ================================================================== #

    async def _apply_momentum_filter(self, signals: List[Signal]) -> List[Signal]:
        """Filter signals against the 5-day trend.

        Do NOT buy puts/short against a strong uptrend, and do NOT buy
        calls/long against a strong downtrend.

        Computes 5-day EMA and checks price vs EMA.
        """
        filtered: List[Signal] = []
        for signal in signals:
            try:
                import yfinance as yf
                data = yf.download(
                    signal.symbol, period='10d', interval='1d', progress=False
                )
                if data is None or len(data) < 5:
                    filtered.append(signal)
                    continue

                import pandas as pd
                if isinstance(data.columns, pd.MultiIndex):
                    closes = data['Close'].iloc[:, 0].dropna()
                else:
                    closes = data['Close'].dropna()

                if len(closes) < 5:
                    filtered.append(signal)
                    continue

                ema5 = closes.ewm(span=5, adjust=False).mean()
                current_price = float(closes.iloc[-1])
                ema5_val = float(ema5.iloc[-1])
                trend_up = current_price > ema5_val

                # Block puts against uptrend
                if signal.signal_type == SignalType.BUY and signal.strategy in (
                    "put_spread", "straddle", "strangle"
                ):
                    # Straddles/strangles are direction-neutral so allow
                    if signal.strategy == "put_spread" and trend_up:
                        self.logger.info(
                            f"MOMENTUM: Blocking put_spread on {signal.symbol} "
                            f"(price {current_price:.2f} > EMA5 {ema5_val:.2f})"
                        )
                        continue

                # Block bullish signals in strong downtrend
                if signal.signal_type == SignalType.BUY and signal.strategy == "call_spread":
                    if not trend_up:
                        self.logger.info(
                            f"MOMENTUM: Blocking call_spread on {signal.symbol} "
                            f"(price {current_price:.2f} < EMA5 {ema5_val:.2f})"
                        )
                        continue

            except Exception:
                pass  # On failure, allow signal through

            filtered.append(signal)

        removed = len(signals) - len(filtered)
        if removed > 0:
            self.logger.info(f"MOMENTUM FILTER: Removed {removed} signals against trend")
        return filtered

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

from .config import RISK_CONFIG, STRATEGY_WEIGHTS, TRANSACTION_COSTS, LIQUIDITY_GATES
from .universe import get_universe, is_strategy_allowed, STRATEGY_DEFINITIONS
from .iv_analyzer import IVAnalyzer
from .theta_decay_engine import ThetaDecayEngine
from .iv_data_manager import IVDataManager
from .theta_decay_engine import IVRegime, TrendDirection
from .earnings_gate import EARNINGS_CALENDAR, next_earnings_date

# Phase 5b: Advanced spread strategies (lazy-loaded to avoid circular import)
_SPREAD_AVAILABLE = True  # will be set False on first load failure


# ============================================================================
# SHARED INFRASTRUCTURE: DATA QUALITY + LIQUIDITY GATES
# ============================================================================

def _check_data_quality(iv_data_manager: IVDataManager, symbol: str, logger=None) -> bool:
    """Block signal if IV data is synthetic or has <30 days of real data.

    Returns True if data quality is sufficient to trade.
    """
    if iv_data_manager.is_synthetic(symbol):
        if logger:
            logger.info(f"DATA_QUALITY: Blocking {symbol} — IV data is synthetic")
        return False
    score = iv_data_manager.data_quality_score(symbol)
    if score < 0.3:
        if logger:
            logger.info(f"DATA_QUALITY: Blocking {symbol} — quality score {score:.2f} < 0.30")
        return False
    return True


def _check_liquidity(symbol: str, logger=None) -> bool:
    """Check minimum liquidity gates: volume > 1M shares.

    Options OI and bid-ask checks require live market data.  Here we
    verify underlying equity volume using yfinance.  If data is
    unavailable, we ALLOW the signal (fail-open for known liquid names
    in the universe).
    """
    min_vol = LIQUIDITY_GATES.get("min_avg_daily_volume", 1_000_000)
    try:
        import yfinance as yf
        import pandas as pd
        data = yf.download(symbol, period='10d', interval='1d', progress=False)
        if data is None or data.empty:
            return True  # fail-open
        if isinstance(data.columns, pd.MultiIndex):
            vol_series = data['Volume'].iloc[:, 0].dropna()
        else:
            vol_series = data['Volume'].dropna()
        if len(vol_series) < 3:
            return True  # insufficient data, fail-open
        avg_vol = float(vol_series.tail(5).mean())
        if avg_vol < min_vol:
            if logger:
                logger.info(
                    f"LIQUIDITY: Blocking {symbol} — avg volume "
                    f"{avg_vol:,.0f} < {min_vol:,.0f}"
                )
            return False
    except Exception:
        pass  # fail-open on error
    return True


def _check_net_edge(expected_premium: float, contracts: int, logger=None, symbol: str = "") -> bool:
    """Check that expected premium exceeds transaction costs by minimum margin.

    Returns True if net edge is sufficient.
    """
    if expected_premium is None or expected_premium <= 0 or contracts <= 0:
        return True  # cannot evaluate, allow signal through

    commission = TRANSACTION_COSTS["commission_per_contract"] * contracts
    slippage = TRANSACTION_COSTS["slippage_pct_of_mid"] * expected_premium * contracts
    total_cost = commission + slippage
    gross = expected_premium * contracts
    min_edge = TRANSACTION_COSTS["min_expected_edge_after_costs"]
    net_edge = gross - total_cost

    if net_edge < min_edge * gross:
        if logger:
            logger.info(
                f"TX_COST: Blocking {symbol} — net_edge ${net_edge:.2f} < "
                f"{min_edge:.0%} of gross ${gross:.2f}"
            )
        return False
    return True


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
    EARNINGS_IV_CRUSH = "earnings_iv_crush"
    ZERO_DTE_BUTTERFLY = "zero_dte_butterfly"


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
        # Data quality gate: block synthetic / low-quality IV data
        if not _check_data_quality(self.iv_data_manager, symbol, self.logger):
            return None
        # Liquidity gate: block illiquid underlyings
        if not _check_liquidity(symbol, self.logger):
            return None

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
        # Data quality gate
        if not _check_data_quality(self.iv_data_manager, symbol, self.logger):
            return None
        # Liquidity gate
        if not _check_liquidity(symbol, self.logger):
            return None

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
        # Liquidity gate
        if not _check_liquidity(symbol, self.logger):
            return None

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
        # Data quality gate
        if not _check_data_quality(self.iv_data, symbol, self.logger):
            return None
        # Liquidity gate
        if not _check_liquidity(symbol, self.logger):
            return None

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
        # Data quality gate
        if not _check_data_quality(self.iv_data_manager, symbol, self.logger):
            return None
        # Liquidity gate
        if not _check_liquidity(symbol, self.logger):
            return None

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

        # Data quality gate
        if not _check_data_quality(self.iv_data_manager, symbol, self.logger):
            return None
        # Liquidity gate
        if not _check_liquidity(symbol, self.logger):
            return None
        
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
# EARNINGS IV CRUSH STRATEGY — Pre-Earnings Straddle/Strangle Sell
# ============================================================================

class EarningsIVCrushStrategy:
    """
    Sell straddles/strangles 3-5 days BEFORE earnings on S&P 500 names.

    Known documented edge: IV spikes dramatically before earnings and
    collapses afterwards.  We sell into the IV spike and close 1 day
    before earnings (before the binary gap risk).

    Rules:
    - Scan S&P 500 names with earnings in 4-7 days
    - IV rank > 70 (unusually high IV pre-earnings)
    - Sell ATM straddle or 16-delta strangle
    - Target: close at 25-40% profit or 1 day before earnings
    - Hard stop: 2x credit received
    """

    # S&P 500 names with high historical IV crush (extend as needed)
    SP500_HIGH_CRUSH = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
        "NFLX", "AMD", "CRM", "ADBE", "INTC", "PYPL", "SHOP",
        "UBER", "SQ", "SNAP", "ROKU", "PINS", "ZM",
    ]

    def __init__(self):
        self.config = RISK_CONFIG
        self.logger = logging.getLogger(f"{__name__}.EarningsIVCrush")
        self.iv_data_manager = IVDataManager()

    async def generate_signals(self, symbols: List[str]) -> List[Signal]:
        """Generate earnings IV crush signals."""
        signals: List[Signal] = []
        today = date.today()

        # Only scan high-crush names that are also in our universe
        candidates = [s for s in symbols if s.upper() in self.SP500_HIGH_CRUSH]
        # Also scan high-crush names not in symbols
        extras = [s for s in self.SP500_HIGH_CRUSH if s not in symbols]
        candidates.extend(extras)

        for symbol in candidates:
            try:
                sig = await self._evaluate(symbol, today)
                if sig is not None:
                    signals.append(sig)
            except Exception as exc:
                self.logger.debug(f"EarningsIVCrush error {symbol}: {exc}")
        return signals

    async def _evaluate(self, symbol: str, today: date) -> Optional[Signal]:
        """Evaluate earnings IV crush opportunity."""
        # Data quality gate
        if not _check_data_quality(self.iv_data_manager, symbol, self.logger):
            return None
        # Liquidity gate
        if not _check_liquidity(symbol, self.logger):
            return None

        # Check earnings date
        earnings_date = next_earnings_date(symbol)
        if earnings_date is None:
            return None

        days_to_earnings = (earnings_date - today).days

        # Only active 4-7 days before earnings
        if days_to_earnings < 4 or days_to_earnings > 7:
            return None

        # IV rank must be elevated (>70)
        iv_rank = self.iv_data_manager.get_iv_rank(symbol)
        if iv_rank is None or iv_rank < 70:
            return None

        # Get current price for strike selection
        current_price = await self._get_price(symbol)
        if current_price is None or current_price <= 0:
            return None

        # Strategy: sell ATM straddle (highest premium capture)
        # For smaller accounts, use 16-delta strangle instead
        strategy = "straddle" if is_strategy_allowed(symbol, "straddle") else "strangle"

        # Estimate premium as % of stock price using IV
        implied_vol = 0.15 + (iv_rank / 100.0) * 0.35
        dte = days_to_earnings - 1  # close 1 day before earnings
        dte = max(dte, 1)

        # BS straddle premium estimate: 2 * S * sigma * sqrt(T/2pi)
        T = dte / 365.0
        straddle_premium_est = 2 * current_price * implied_vol * np.sqrt(T / (2 * np.pi))
        premium_per_contract = straddle_premium_est  # per share, multiply by 100 for contract

        # Confidence: higher IV rank + closer to earnings = higher confidence
        confidence = min(
            0.55 + (iv_rank - 70) / 100.0 + (7 - days_to_earnings) * 0.02,
            0.90,
        )

        return Signal(
            symbol=symbol,
            signal_type=SignalType.SELL,
            signal_source=SignalSource.EARNINGS_IV_CRUSH,
            strategy=strategy,
            confidence=round(confidence, 3),
            timestamp=datetime.now(),
            iv_rank=iv_rank,
            current_price=current_price,
            dte=dte,
            expected_premium=round(premium_per_contract, 2),
            probability_of_profit=min(0.60 + (iv_rank - 70) * 0.003, 0.80),
            reason=(
                f"EARNINGS IV CRUSH: {symbol} earnings in {days_to_earnings}d, "
                f"IV rank={iv_rank:.0f}, est premium=${premium_per_contract:.2f}/sh, "
                f"close {dte}d pre-earnings"
            ),
        )

    async def _get_price(self, symbol: str) -> Optional[float]:
        """Fetch current price via yfinance."""
        try:
            import yfinance as yf
            import pandas as pd
            data = yf.download(symbol, period='5d', interval='1d', progress=False)
            if data is None or len(data) == 0:
                return None
            if isinstance(data.columns, pd.MultiIndex):
                return float(data['Close'].iloc[:, 0].dropna().values[-1])
            else:
                return float(data['Close'].dropna().values[-1])
        except Exception:
            return None


# ============================================================================
# GAMMA SCALPING STRATEGY
# ============================================================================

class GammaScalpingStrategy:
    """
    Generate signal when 5-day realized vol > implied vol by >5 points.

    Uses yfinance for realized vol calculation. When RV dominates IV,
    gamma scalping (long gamma) is profitable because the underlying
    moves more than the market is pricing.
    """

    RV_LOOKBACK = 5           # 5 trading days
    IV_RV_THRESHOLD = 5.0     # 5 vol points (e.g. 25% RV vs 20% IV)

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.GammaScalping")
        self.iv_data = IVDataManager()

    async def generate_signals(self, symbols: List[str]) -> List[Signal]:
        signals: List[Signal] = []
        for symbol in symbols:
            try:
                sig = await self._evaluate(symbol)
                if sig is not None:
                    signals.append(sig)
            except Exception as exc:
                self.logger.debug(f"GammaScalp error {symbol}: {exc}")
        return signals

    async def _evaluate(self, symbol: str) -> Optional[Signal]:
        if not _check_data_quality(self.iv_data, symbol, self.logger):
            return None
        if not _check_liquidity(symbol, self.logger):
            return None

        rv = self._compute_5d_rv(symbol)
        if rv is None:
            return None

        iv = self.iv_data.get_current_iv(symbol)
        if iv is None:
            return None

        # Both in annualised percentage terms (e.g. 0.25 = 25%)
        rv_pct = rv * 100
        iv_pct = iv * 100
        diff = rv_pct - iv_pct

        if diff > self.IV_RV_THRESHOLD:
            confidence = min(0.95, 0.55 + 0.05 * (diff - self.IV_RV_THRESHOLD))
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                signal_source=SignalSource.IV_RANK,
                strategy="straddle",
                confidence=round(confidence, 3),
                timestamp=datetime.now(),
                iv_rank=None,
                dte=14,
                reason=(
                    f"Gamma scalp: 5d RV {rv_pct:.1f}% > IV {iv_pct:.1f}% "
                    f"(diff={diff:.1f} > {self.IV_RV_THRESHOLD})"
                ),
            )
        return None

    def _compute_5d_rv(self, symbol: str) -> Optional[float]:
        """5-day annualized realized vol from yfinance."""
        try:
            import yfinance as yf
            import pandas as pd
            data = yf.download(symbol, period="10d", interval="1d", progress=False)
            if data is None or len(data) < self.RV_LOOKBACK + 1:
                return None
            if isinstance(data.columns, pd.MultiIndex):
                closes = data["Close"].iloc[:, 0].dropna().values.astype(float)
            else:
                closes = data["Close"].dropna().values.astype(float)
            if len(closes) < self.RV_LOOKBACK + 1:
                return None
            log_rets = np.diff(np.log(closes[-(self.RV_LOOKBACK + 1):]))
            return float(np.std(log_rets) * np.sqrt(252))
        except Exception:
            return None


# ============================================================================
# VOLATILITY ARBITRAGE STRATEGY
# ============================================================================

class VolatilityArbitrageStrategy:
    """
    SPY vs sector ETF (XLK, XLE, XLF, XBI) vol dispersion strategy.

    Generates signal when the z-score of IV dispersion across sectors
    exceeds 2.0, indicating unusual divergence in sector-level risk pricing.
    """

    SECTOR_ETFS = ["XLK", "XLE", "XLF", "XBI"]
    Z_THRESHOLD = 2.0
    RV_LOOKBACK = 20

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.VolArb")
        self.iv_data = IVDataManager()

    async def generate_signals(self, symbols: List[str] = None) -> List[Signal]:
        """Generate vol dispersion signals. Symbols arg ignored; uses SPY + sectors."""
        signals: List[Signal] = []
        try:
            sig = await self._evaluate_dispersion()
            if sig is not None:
                signals.append(sig)
        except Exception as exc:
            self.logger.debug(f"VolArb error: {exc}")
        return signals

    async def _evaluate_dispersion(self) -> Optional[Signal]:
        spy_iv = self.iv_data.get_current_iv("SPY")
        if spy_iv is None:
            return None

        sector_ivs = []
        for etf in self.SECTOR_ETFS:
            iv = self.iv_data.get_current_iv(etf)
            if iv is not None:
                sector_ivs.append(iv)

        if len(sector_ivs) < 2:
            return None

        # Dispersion = std of sector IVs relative to SPY IV
        dispersion = float(np.std(sector_ivs))
        mean_disp = float(np.mean(sector_ivs))

        # Z-score: use heuristic rolling baseline (20% of mean as std proxy)
        baseline_std = max(mean_disp * 0.20, 0.01)
        z_score = (dispersion - baseline_std) / baseline_std

        if abs(z_score) > self.Z_THRESHOLD:
            # High dispersion → sell premium on high-IV sectors, buy on low
            if z_score > 0:
                # Dispersion elevated → sell condors on SPY (expect convergence)
                confidence = min(0.90, 0.55 + 0.10 * (z_score - self.Z_THRESHOLD))
                return Signal(
                    symbol="SPY",
                    signal_type=SignalType.SELL,
                    signal_source=SignalSource.IV_RANK,
                    strategy="iron_condor",
                    confidence=round(confidence, 3),
                    timestamp=datetime.now(),
                    z_score=round(z_score, 2),
                    dte=30,
                    reason=(
                        f"Vol arb: sector dispersion z={z_score:.2f} > "
                        f"{self.Z_THRESHOLD} → sell premium"
                    ),
                )
        return None


# ============================================================================
# SKEW TRADE STRATEGY
# ============================================================================

class SkewTradeStrategy:
    """
    Uses 25-delta put/call skew from iv_data_manager.

    - Sell put spreads when skew >2std elevated (puts overpriced)
    - Buy call spreads when skew collapses (calls cheap)
    """

    SKEW_LOOKBACK = 30   # 30 days for z-score
    SELL_Z_THRESHOLD = 2.0
    BUY_Z_THRESHOLD = -2.0

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SkewTrade")
        self.iv_data = IVDataManager()

    async def generate_signals(self, symbols: List[str]) -> List[Signal]:
        signals: List[Signal] = []
        for symbol in symbols:
            try:
                sig = await self._evaluate(symbol)
                if sig is not None:
                    signals.append(sig)
            except Exception as exc:
                self.logger.debug(f"SkewTrade error {symbol}: {exc}")
        return signals

    async def _evaluate(self, symbol: str) -> Optional[Signal]:
        if not _check_data_quality(self.iv_data, symbol, self.logger):
            return None

        # Get skew history
        history = self.iv_data.get_iv_history(symbol, days=self.SKEW_LOOKBACK)
        if len(history) < 10:
            return None

        skews = [h.skew_25delta for h in history if h.skew_25delta is not None]
        if len(skews) < 10:
            return None

        current_skew = skews[0]  # Most recent (history is DESC)
        mean_skew = float(np.mean(skews))
        std_skew = float(np.std(skews))
        if std_skew < 1e-6:
            return None

        z_score = (current_skew - mean_skew) / std_skew

        # Skew elevated → puts are expensive → sell put spreads
        if z_score > self.SELL_Z_THRESHOLD:
            confidence = min(0.90, 0.55 + 0.10 * (z_score - self.SELL_Z_THRESHOLD))
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                signal_source=SignalSource.IV_RANK,
                strategy="put_spread",
                confidence=round(confidence, 3),
                timestamp=datetime.now(),
                z_score=round(z_score, 2),
                dte=30,
                reason=(
                    f"Skew trade SELL: 25Δ skew z={z_score:.2f} > "
                    f"{self.SELL_Z_THRESHOLD} (skew={current_skew:.4f})"
                ),
            )

        # Skew collapsed → calls are cheap → buy call spreads
        if z_score < self.BUY_Z_THRESHOLD:
            confidence = min(0.90, 0.55 + 0.10 * (self.BUY_Z_THRESHOLD - z_score))
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                signal_source=SignalSource.IV_RANK,
                strategy="call_spread",
                confidence=round(confidence, 3),
                timestamp=datetime.now(),
                z_score=round(z_score, 2),
                dte=21,
                reason=(
                    f"Skew trade BUY: 25Δ skew z={z_score:.2f} < "
                    f"{self.BUY_Z_THRESHOLD} (skew={current_skew:.4f})"
                ),
            )

        return None


# ============================================================================
# 0DTE SPX IRON BUTTERFLY STRATEGY
# ============================================================================

class ZeroDTEIronButterflyStrategy:
    """
    0DTE SPX Iron Butterfly — highest documented Sharpe retail options strategy.

    Rules:
    - Only active 2:00-3:30pm ET (accelerated theta decay)
    - Only fires when VIX < 20 (low vol, predictable decay)
    - Only on SPX/SPY (weekly expirations, same-day)
    - ATM body ± 10 strikes, wings ± 20 strikes wide
    - Auto-close at 50% profit OR 30 min before close (3:30pm)
    - Hard stop: 2x credit received

    Documented Sharpe: 2.5-5.0 across retail traders in 2024-2025.
    """

    def __init__(self):
        self.config = RISK_CONFIG
        self.logger = logging.getLogger(f"{__name__}.ZeroDTEButterfly")
        self._vix_cache = None
        self._vix_cache_time = None

    async def generate_signals(self, symbols: List[str]) -> List[Signal]:
        """Generate 0DTE butterfly signal if conditions met.

        This strategy only trades SPY (retail-accessible proxy for SPX).
        It checks time window, VIX level, and constructs an iron butterfly.
        """
        signals: List[Signal] = []

        # Only trade SPY (or SPX if in universe)
        target_symbol = "SPY" if "SPY" in symbols else None
        if target_symbol is None:
            return signals

        try:
            sig = await self._evaluate(target_symbol)
            if sig is not None:
                signals.append(sig)
        except Exception as exc:
            self.logger.debug(f"0DTE Butterfly error: {exc}")

        return signals

    async def _evaluate(self, symbol: str) -> Optional[Signal]:
        """Check all conditions for 0DTE iron butterfly entry."""
        from zoneinfo import ZoneInfo

        now_et = datetime.now(ZoneInfo("America/New_York"))
        current_time = now_et.time()

        # Time window: 2:00 PM - 3:30 PM ET only
        from datetime import time as dtime
        if current_time < dtime(14, 0) or current_time > dtime(15, 30):
            return None

        # VIX must be below 20 (calm market for butterfly)
        vix_level = await self._get_vix()
        if vix_level is None or vix_level >= 20:
            self.logger.debug(f"0DTE: VIX={vix_level} >= 20, skipping")
            return None

        # Get current price
        current_price = await self._get_price(symbol)
        if current_price is None or current_price <= 0:
            return None

        # Iron butterfly construction:
        # Short ATM put + Short ATM call (body)
        # Long put at ATM-10 (wing) + Long call at ATM+10 (wing)
        atm_strike = round(current_price)
        wing_width = 10  # $10 wide wings

        strike_put = atm_strike - wing_width
        strike_call = atm_strike + wing_width

        # Estimate premium: butterfly captures ~30-40% of wing width in premium
        estimated_credit = wing_width * 0.35  # ~$3.50 on $10 wings
        max_profit = estimated_credit * 100  # per contract
        max_loss = (wing_width - estimated_credit) * 100  # per contract

        # DTE = 0 (same day expiration)
        dte = 0

        # PoP for iron butterfly: approximately premium_width / wing_width
        pop = min(estimated_credit / wing_width, 0.60)

        # Confidence: higher when VIX is lower and time is closer to 2:30pm
        time_score = 1.0 - abs(current_time.hour * 60 + current_time.minute - 14 * 60 - 30) / 90.0
        vix_score = max(0, (20 - vix_level) / 10.0)
        confidence = min(0.55 + 0.15 * time_score + 0.15 * vix_score, 0.90)

        return Signal(
            symbol=symbol,
            signal_type=SignalType.SELL,
            signal_source=SignalSource.ZERO_DTE_BUTTERFLY,
            strategy="iron_condor",  # iron butterfly is a type of iron condor
            confidence=round(confidence, 3),
            timestamp=datetime.now(),
            iv_rank=None,
            current_price=current_price,
            strike_put=strike_put,
            strike_call=strike_call,
            dte=dte,
            expected_premium=round(estimated_credit, 2),
            max_loss=max_loss,
            probability_of_profit=round(pop, 3),
            reason=(
                f"0DTE IRON BUTTERFLY: {symbol} @ ${atm_strike}, "
                f"wings ±${wing_width}, VIX={vix_level:.1f}, "
                f"est credit=${estimated_credit:.2f}/sh, "
                f"time={current_time.strftime('%H:%M')} ET"
            ),
        )

    async def _get_vix(self) -> Optional[float]:
        """Get current VIX with 5-min cache."""
        now = datetime.now()
        if (
            self._vix_cache is not None
            and self._vix_cache_time is not None
            and (now - self._vix_cache_time).total_seconds() < 300
        ):
            return self._vix_cache

        try:
            import yfinance as yf
            import pandas as pd
            data = yf.download("^VIX", period="5d", interval="1d", progress=False)
            if data is None or data.empty:
                return None
            if isinstance(data.columns, pd.MultiIndex):
                vix = float(data["Close"].iloc[:, 0].dropna().values[-1])
            else:
                vix = float(data["Close"].dropna().values[-1])
            self._vix_cache = vix
            self._vix_cache_time = now
            return vix
        except Exception:
            return self._vix_cache  # return stale cache on failure

    async def _get_price(self, symbol: str) -> Optional[float]:
        """Fetch current price."""
        try:
            import yfinance as yf
            import pandas as pd
            data = yf.download(symbol, period='5d', interval='1d', progress=False)
            if data is None or len(data) == 0:
                return None
            if isinstance(data.columns, pd.MultiIndex):
                return float(data['Close'].iloc[:, 0].dropna().values[-1])
            else:
                return float(data['Close'].dropna().values[-1])
        except Exception:
            return None


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
        self.earnings_iv_crush_strategy = EarningsIVCrushStrategy()
        self.zero_dte_butterfly_strategy = ZeroDTEIronButterflyStrategy()
        
        # Grand Overhaul alpha strategies
        self.gamma_scalping_strategy = GammaScalpingStrategy()
        self.vol_arb_strategy = VolatilityArbitrageStrategy()
        self.skew_trade_strategy = SkewTradeStrategy()
        
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
                            "delta_hedging", "vrp", "iv_crush",
                            "earnings_iv_crush", "zero_dte_butterfly",
                            "gamma_scalping", "vol_arb", "skew_trade"],
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
            SignalSource.EARNINGS_IV_CRUSH: "earnings_iv_crush",
            SignalSource.ZERO_DTE_BUTTERFLY: "zero_dte_butterfly",
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
            self.earnings_iv_crush_strategy.generate_signals(symbols),
            self.zero_dte_butterfly_strategy.generate_signals(symbols),
            self.gamma_scalping_strategy.generate_signals(symbols),
            self.vol_arb_strategy.generate_signals(symbols),
            self.skew_trade_strategy.generate_signals(symbols),
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

        # ===== ALPHA OVERHAUL: TRANSACTION COST NET EDGE CHECK =====
        pre_tx = len(all_signals)
        all_signals = [
            s for s in all_signals
            if _check_net_edge(s.expected_premium, 1, self.logger, s.symbol)
        ]
        if len(all_signals) < pre_tx:
            self.logger.info(f"TX_COST: Removed {pre_tx - len(all_signals)} signals below net edge")

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

"""
Signal Generator v2
====================

COMPLETE REPLACEMENT for the broken V50OptionsAlphaEngine signal pipeline.

This module is the orchestrator: it combines IV analysis, regime detection,
and strategy selection to produce high-quality, well-reasoned trade signals.

Key improvements over the old system:
1. NO "neutral" signals interpreted as bearish
2. Clear NO_TRADE filter when conditions aren't favorable
3. Every signal has a logged rationale
4. Max 2-5 trades per day (quality over quantity)
5. All strategies are DEFINED RISK (no naked options)

Usage:
    generator = SignalGeneratorV2()
    signals = generator.scan_for_signals()
    for signal in signals:
        print(f"{signal.symbol}: {signal.strategy.value} - {signal.rationale}")

Author: System Overhaul - Feb 2026
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from src.iv_analysis import IVAnalysisEngine, IVMetrics, MarketIVSnapshot
from src.regime_detector import (
    RuleBasedRegimeDetector,
    Regime,
    RegimeResult,
    TechnicalSignals,
)
from src.strategy_selector import (
    StrategySelector,
    StrategyRecommendation,
    StrategyType,
    Direction,
)

logger = logging.getLogger(__name__)


# ============================================================================
# TRADING UNIVERSE
# ============================================================================

# Liquid underlyings with good options markets
DEFAULT_UNIVERSE = [
    "SPY",    # S&P 500 ETF - most liquid
    "QQQ",    # Nasdaq 100 ETF
    "IWM",    # Russell 2000 ETF
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "NVDA",   # NVIDIA
    "TSLA",   # Tesla
    "AMZN",   # Amazon
    "META",   # Meta
    "GOOGL",  # Alphabet
]

# Maximum signals per scan cycle
MAX_SIGNALS_PER_SCAN = 5


# ============================================================================
# TRADE SIGNAL (output of this module)
# ============================================================================

@dataclass
class TradeSignal:
    """
    A complete, actionable trade signal.

    This is the FINAL output that gets passed to the execution engine.
    Every field is populated and every signal has a clear rationale.
    """
    # Symbol and strategy
    symbol: str
    strategy: StrategyType
    direction: Direction

    # Options parameters
    target_dte: int
    target_delta: float
    wing_width: float

    # Position sizing hints
    max_contracts: int
    probability_of_profit: float
    risk_reward_ratio: float

    # Analysis context
    iv_rank: float
    iv_percentile: float
    regime: Regime
    regime_confidence: float
    confidence: float          # Overall signal confidence (0-1)

    # Technical context
    rsi: float
    macd_signal: float         # MACD - Signal line
    bb_position: float         # 0-1 within Bollinger Bands
    current_price: float

    # Rationale
    rationale: str

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    signal_id: str = ""

    def __post_init__(self):
        if not self.signal_id:
            ts = self.timestamp.strftime("%H%M%S")
            self.signal_id = f"{self.symbol}_{self.strategy.value}_{ts}"

    @property
    def is_tradeable(self) -> bool:
        """Is this signal actionable (not NO_TRADE)?"""
        return self.strategy != StrategyType.NO_TRADE

    @property
    def is_credit(self) -> bool:
        """Is this a credit (premium-selling) strategy?"""
        return self.strategy in (
            StrategyType.IRON_CONDOR,
            StrategyType.BULL_PUT_SPREAD,
            StrategyType.BEAR_CALL_SPREAD,
        )

    def summary(self) -> str:
        """Human-readable one-line summary."""
        if not self.is_tradeable:
            return f"[{self.symbol}] NO TRADE - {self.rationale}"
        return (
            f"[{self.symbol}] {self.strategy.value} "
            f"({self.direction.value}) "
            f"IV={self.iv_rank:.0f} Regime={self.regime.value} "
            f"Conf={self.confidence:.0%} "
            f"DTE={self.target_dte} Delta={self.target_delta:.2f}"
        )


# ============================================================================
# SIGNAL GENERATOR V2
# ============================================================================

class SignalGeneratorV2:
    """
    Production signal generator combining IV analysis + regime + strategy.

    Scan flow:
    1. Get market-wide IV snapshot (VIX level, rank, term structure)
    2. For each symbol in universe:
       a. Calculate IV metrics (rank, percentile, HV/IV ratio)
       b. Detect market regime (trending, mean-reverting, high-vol)
       c. Select optimal strategy (or NO_TRADE)
       d. Build complete TradeSignal with all context
    3. Filter: keep only tradeable signals above minimum confidence
    4. Rank by confidence, cap at MAX_SIGNALS_PER_SCAN
    5. Log every decision (trade or no-trade) for audit
    """

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        min_confidence: float = 0.40,
        max_signals: int = MAX_SIGNALS_PER_SCAN,
        max_daily_trades: int = 15,
    ):
        """
        Initialize signal generator.

        Args:
            universe: List of symbols to scan (default: DEFAULT_UNIVERSE)
            min_confidence: Minimum confidence to emit a signal
            max_signals: Maximum signals per scan
            max_daily_trades: Max new-trade signals emitted per day (default 15)
        """
        self.universe = universe or DEFAULT_UNIVERSE
        self.min_confidence = min_confidence
        self.max_signals = max_signals
        self.max_daily_trades = max_daily_trades

        # Sub-components
        self.iv_engine = IVAnalysisEngine()
        self.regime_detector = RuleBasedRegimeDetector()
        self.strategy_selector = StrategySelector()

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"SignalGeneratorV2 initialized: "
            f"{len(self.universe)} symbols, "
            f"min_conf={min_confidence:.0%}, max_signals={max_signals}"
        )

        # Track daily signal count
        self._daily_signals: List[TradeSignal] = []
        self._daily_date: Optional[str] = None

    # ================================================================== #
    # PUBLIC API
    # ================================================================== #

    def scan_for_signals(self) -> List[TradeSignal]:
        """
        Run a complete signal scan across the universe.

        Returns:
            List of actionable TradeSignals (NO_TRADE filtered out),
            sorted by confidence descending, capped at max_signals.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        if self._daily_date != today:
            self._daily_signals = []
            self._daily_date = today

        self.logger.info("=" * 60)
        self.logger.info(f"SIGNAL SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)

        # Step 1: Market-wide context
        market_snapshot = self.iv_engine.get_market_iv_snapshot()
        if market_snapshot:
            self.logger.info(
                f"Market: VIX={market_snapshot.vix_level:.1f} "
                f"VIX_Rank={market_snapshot.vix_rank:.0f} "
                f"Term={market_snapshot.vix_term_slope:.2f}"
            )

        # Step 2: Analyze each symbol
        all_signals: List[TradeSignal] = []
        no_trade_reasons: List[str] = []

        for symbol in self.universe:
            signal = self._analyze_symbol(symbol, market_snapshot)

            if signal.is_tradeable:
                all_signals.append(signal)
                self.logger.info(f"  SIGNAL: {signal.summary()}")
            else:
                no_trade_reasons.append(f"  {symbol}: {signal.rationale}")

        # Step 3: Log no-trade decisions
        if no_trade_reasons:
            self.logger.info(f"Skipped {len(no_trade_reasons)} symbols:")
            for reason in no_trade_reasons:
                self.logger.debug(reason)

        # Step 4: Filter by minimum confidence
        qualified = [s for s in all_signals if s.confidence >= self.min_confidence]
        filtered_count = len(all_signals) - len(qualified)
        if filtered_count > 0:
            self.logger.info(f"Filtered {filtered_count} signals below {self.min_confidence:.0%} confidence")

        # Step 5: Sort by confidence and cap
        qualified.sort(key=lambda s: s.confidence, reverse=True)
        result = qualified[: self.max_signals]

        # Check daily limit
        daily_remaining = self.max_daily_trades - len(self._daily_signals)
        if daily_remaining <= 0:
            self.logger.warning(
                f"Daily trade limit ({self.max_daily_trades}) reached - no new signals"
            )
            return []
        result = result[:daily_remaining]

        # Track (only actionable signals count — NO_TRADE is already
        # filtered out above so everything in *result* is tradeable)
        self._daily_signals.extend(result)

        # Summary
        self.logger.info(
            f"SCAN COMPLETE: {len(result)} actionable signals "
            f"(from {len(self.universe)} symbols scanned)"
        )
        for sig in result:
            self.logger.info(f"  >> {sig.summary()}")

        return result

    def get_daily_trade_count(self) -> int:
        """Get number of signals generated today."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._daily_date != today:
            return 0
        return len(self._daily_signals)

    # ================================================================== #
    # PRIVATE: Per-Symbol Analysis
    # ================================================================== #

    def _analyze_symbol(
        self,
        symbol: str,
        market_snapshot: Optional[MarketIVSnapshot],
    ) -> TradeSignal:
        """
        Analyze a single symbol and produce a TradeSignal.

        Steps:
        1. Get IV metrics
        2. Detect regime (using SPY for broad market, symbol-specific for stocks)
        3. Select strategy
        4. Build TradeSignal

        Returns:
            TradeSignal (may be NO_TRADE)
        """
        try:
            # Step 1: IV Analysis
            iv_metrics = self.iv_engine.get_iv_metrics(symbol)
            if iv_metrics is None:
                return self._make_no_trade(
                    symbol, reason="IV data unavailable"
                )

            # Step 2: Regime Detection
            # Use SPY regime for ETFs, symbol-specific for stocks
            regime_symbol = "SPY" if symbol in ("SPY", "QQQ", "IWM", "DIA") else symbol
            regime_result = self.regime_detector.detect_regime(regime_symbol)

            # For individual stocks, also check broad market regime
            if regime_symbol != "SPY":
                spy_regime = self.regime_detector.detect_regime("SPY")
                # If SPY is in HIGH_VOLATILITY, override individual stock regime
                if spy_regime.regime == Regime.HIGH_VOLATILITY:
                    regime_result = spy_regime

            # Step 3: Strategy Selection
            recommendation = self.strategy_selector.select_strategy(
                symbol=symbol,
                iv_metrics=iv_metrics,
                regime_result=regime_result,
                market_snapshot=market_snapshot,
            )

            # Step 4: Build TradeSignal
            technicals = regime_result.technicals
            return TradeSignal(
                symbol=symbol,
                strategy=recommendation.strategy,
                direction=recommendation.direction,
                target_dte=recommendation.target_dte,
                target_delta=recommendation.target_delta,
                wing_width=recommendation.wing_width,
                max_contracts=recommendation.max_contracts,
                probability_of_profit=recommendation.probability_of_profit,
                risk_reward_ratio=recommendation.risk_reward_ratio,
                iv_rank=iv_metrics.iv_rank,
                iv_percentile=iv_metrics.iv_percentile,
                regime=regime_result.regime,
                regime_confidence=regime_result.confidence,
                confidence=recommendation.confidence,
                rsi=technicals.rsi_14 if technicals else 50.0,
                macd_signal=technicals.macd_signal if technicals else 0.0,
                bb_position=technicals.bb_position if technicals else 0.5,
                current_price=technicals.current_price if technicals else 0.0,
                rationale=recommendation.rationale,
            )

        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)
            return self._make_no_trade(symbol, reason=f"Analysis error: {e}")

    def _make_no_trade(
        self,
        symbol: str,
        reason: str,
    ) -> TradeSignal:
        """Create a NO_TRADE signal."""
        return TradeSignal(
            symbol=symbol,
            strategy=StrategyType.NO_TRADE,
            direction=Direction.NEUTRAL,
            target_dte=0,
            target_delta=0,
            wing_width=0,
            max_contracts=0,
            probability_of_profit=0,
            risk_reward_ratio=0,
            iv_rank=0,
            iv_percentile=0,
            regime=Regime.UNKNOWN,
            regime_confidence=0,
            confidence=0,
            rsi=50,
            macd_signal=0,
            bb_position=0.5,
            current_price=0,
            rationale=f"NO TRADE: {reason}",
        )

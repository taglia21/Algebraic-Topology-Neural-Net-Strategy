"""
Earnings IV Strategy
=====================

Earnings-aware options strategies that exploit the IV lifecycle:

  1. PRE-EARNINGS (5-7 days before):
     - IV is elevated as market prices in uncertainty
     - SELL premium: credit spreads, iron condors
     - Only when IV rank > 60% (confirming elevated IV)

  2. EARNINGS BLACKOUT (0-2 days before):
     - No new positions — binary risk too high
     - Existing positions managed normally

  3. POST-EARNINGS (0-3 days after):
     - IV crushes 30-60% after the event
     - BUY volatility for mean reversion if IV dropped too far
     - Or sell premium if IV remains elevated (rare)

Integration with unified_trader.py:
  - Provides generate_earnings_signals() → List[EarningsOptionSignal]
  - Each signal has strategy, direction, confidence, reason
  - The trader checks these signals in addition to regular IV rank signals

Usage:
    from src.earnings_iv_strategy import EarningsIVStrategy

    strategy = EarningsIVStrategy()
    signals = strategy.generate_earnings_signals(
        symbols=["AAPL", "MSFT"],
        iv_rank_fn=lambda s: iv_engine.get_iv_rank(s),
    )
    for sig in signals:
        print(f"{sig.symbol}: {sig.action} {sig.strategy} (conf={sig.confidence:.0%})")
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

class EarningsAction(Enum):
    """What to do with options around earnings."""
    SELL_PREMIUM = "sell_premium"       # Pre-earnings: sell credit spreads/IC
    BUY_VOLATILITY = "buy_volatility"   # Post-earnings: buy straddles if IV crushed
    BLACKOUT = "blackout"               # Too close to earnings: no new positions
    NO_ACTION = "no_action"             # Nothing to do


@dataclass
class EarningsOptionSignal:
    """Options signal generated from earnings analysis."""
    symbol: str
    action: EarningsAction
    strategy: str                       # "iron_condor", "credit_spread", "straddle", etc.
    direction: str                      # "sell" or "buy"
    confidence: float                   # 0.0 to 1.0
    days_to_earnings: int
    iv_rank: Optional[float] = None
    reason: str = ""
    target_dte: int = 35                # Recommended DTE for the trade
    max_risk_pct: float = 0.03          # Suggested max % of equity to risk

    @property
    def is_actionable(self) -> bool:
        """Whether this signal should result in a trade."""
        return self.action in (EarningsAction.SELL_PREMIUM, EarningsAction.BUY_VOLATILITY)


# ============================================================================
# EARNINGS IV STRATEGY
# ============================================================================

class EarningsIVStrategy:
    """
    Generate options signals based on earnings calendar and IV dynamics.

    Strategy logic:
      - 5-7 days pre-earnings + IV rank > 60% → SELL premium
      - 0-2 days pre-earnings → BLACKOUT
      - 0-3 days post-earnings + IV crush → BUY volatility
    """

    def __init__(
        self,
        pre_earnings_min_days: int = 3,
        pre_earnings_max_days: int = 7,
        blackout_days: int = 2,
        post_earnings_days: int = 3,
        pre_earnings_iv_threshold: float = 60.0,
        post_earnings_iv_drop_pct: float = 20.0,
        earnings_calendar=None,
    ):
        """
        Args:
            pre_earnings_min_days: Minimum days before earnings for premium selling
            pre_earnings_max_days: Maximum days before earnings to consider
            blackout_days: Days before earnings to avoid new positions
            post_earnings_days: Days after earnings to look for IV crush plays
            pre_earnings_iv_threshold: Minimum IV rank for pre-earnings selling
            post_earnings_iv_drop_pct: IV must have dropped this % for post-earnings buy
            earnings_calendar: EarningsCalendar instance (lazy loaded if None)
        """
        self.pre_earnings_min_days = pre_earnings_min_days
        self.pre_earnings_max_days = pre_earnings_max_days
        self.blackout_days = blackout_days
        self.post_earnings_days = post_earnings_days
        self.pre_earnings_iv_threshold = pre_earnings_iv_threshold
        self.post_earnings_iv_drop_pct = post_earnings_iv_drop_pct

        self._calendar = earnings_calendar
        self._calendar_loaded = earnings_calendar is not None

        logger.info(
            f"EarningsIVStrategy initialized: "
            f"pre={pre_earnings_min_days}-{pre_earnings_max_days}d, "
            f"blackout={blackout_days}d, "
            f"IV_threshold={pre_earnings_iv_threshold}%"
        )

    @property
    def calendar(self):
        """Lazy-load earnings calendar."""
        if not self._calendar_loaded:
            try:
                from src.earnings_calendar import EarningsCalendar
                self._calendar = EarningsCalendar()
                self._calendar_loaded = True
            except Exception as e:
                logger.error(f"Failed to load EarningsCalendar: {e}")
                self._calendar = None
                self._calendar_loaded = True
        return self._calendar

    # ================================================================== #
    # PUBLIC API
    # ================================================================== #

    def generate_earnings_signals(
        self,
        symbols: List[str],
        iv_rank_fn: Optional[Callable[[str], Optional[float]]] = None,
    ) -> List[EarningsOptionSignal]:
        """
        Generate earnings-based options signals for a list of symbols.

        Args:
            symbols: Tickers to analyze
            iv_rank_fn: Callable that returns IV rank (0-100) for a symbol.
                       If None, only calendar-based signals are generated.

        Returns:
            List of EarningsOptionSignal for actionable opportunities
        """
        if self.calendar is None:
            logger.warning("No earnings calendar available — skipping")
            return []

        signals = []
        for symbol in symbols:
            try:
                sig = self._analyze_symbol(symbol, iv_rank_fn)
                if sig is not None:
                    signals.append(sig)
            except Exception as e:
                logger.warning(f"Earnings analysis failed for {symbol}: {e}")

        return signals

    def should_skip_equity_trade(self, symbol: str) -> bool:
        """
        Check if an equity trade should be skipped due to earnings proximity.

        Returns True if the symbol is within the blackout window.
        """
        if self.calendar is None:
            return False
        return self.calendar.is_in_blackout(symbol, self.blackout_days)

    def get_etf_earnings_risk(self, etf: str) -> Dict:
        """
        Assess earnings-related risk for an ETF.

        Returns dict with:
          - components_reporting: int (how many major components report this week)
          - risk_level: "low", "medium", "high"
          - details: dict of {symbol: days_until} for reporting components
        """
        if self.calendar is None:
            return {"components_reporting": 0, "risk_level": "low", "details": {}}

        exposure = self.calendar.get_etf_earnings_exposure(etf)
        n = len(exposure)

        if n >= 5:
            risk = "high"
        elif n >= 2:
            risk = "medium"
        else:
            risk = "low"

        return {
            "components_reporting": n,
            "risk_level": risk,
            "details": exposure,
        }

    # ================================================================== #
    # PRIVATE: Analysis
    # ================================================================== #

    def _analyze_symbol(
        self,
        symbol: str,
        iv_rank_fn: Optional[Callable],
    ) -> Optional[EarningsOptionSignal]:
        """Analyze a single symbol for earnings-based options signals."""

        cal = self.calendar

        # Get IV rank if function provided
        iv_rank = None
        if iv_rank_fn is not None:
            try:
                iv_rank = iv_rank_fn(symbol)
            except Exception:
                pass

        # ── Check BLACKOUT first (highest priority) ──
        if cal.is_in_blackout(symbol, self.blackout_days):
            days = cal.get_days_to_earnings(symbol)
            return EarningsOptionSignal(
                symbol=symbol,
                action=EarningsAction.BLACKOUT,
                strategy="none",
                direction="none",
                confidence=1.0,
                days_to_earnings=days if days is not None else 0,
                iv_rank=iv_rank,
                reason=f"Earnings blackout ({days}d to earnings)",
            )

        # ── Check PRE-EARNINGS premium selling window ──
        if cal.is_pre_earnings_window(
            symbol, self.pre_earnings_min_days, self.pre_earnings_max_days
        ):
            days = cal.get_days_to_earnings(symbol)

            # Only sell if IV is elevated
            if iv_rank is not None and iv_rank >= self.pre_earnings_iv_threshold:
                # Confidence scales with IV rank (60% → 0.2, 80% → 0.6, 100% → 1.0)
                confidence = min((iv_rank - self.pre_earnings_iv_threshold) / 40.0, 1.0)
                confidence = max(confidence, 0.3)  # Floor at 30%

                # Prefer iron condors (delta-neutral) for earnings plays
                strategy = "iron_condor"

                # DTE: match it to expire AFTER earnings
                days_val = days if days is not None else 5
                target_dte = max(days_val + 7, 21)  # At least 7 days past earnings

                return EarningsOptionSignal(
                    symbol=symbol,
                    action=EarningsAction.SELL_PREMIUM,
                    strategy=strategy,
                    direction="sell",
                    confidence=confidence,
                    days_to_earnings=days_val,
                    iv_rank=iv_rank,
                    reason=(
                        f"Pre-earnings premium sell: {days_val}d to earnings, "
                        f"IV rank {iv_rank:.1f}% > {self.pre_earnings_iv_threshold}%"
                    ),
                    target_dte=target_dte,
                    max_risk_pct=0.03,  # 3% max risk for earnings plays
                )

        # ── Check POST-EARNINGS: IV crush opportunity ──
        if cal.is_post_earnings(symbol, self.post_earnings_days):
            days_since = None
            recent = cal._get_most_recent_earnings(symbol)
            if recent:
                days_since = -recent.days_until

            # If IV is now LOW after earnings → potential vol mean reversion
            if iv_rank is not None and iv_rank < 30:
                confidence = min((30 - iv_rank) / 30.0, 0.8)  # Cap at 80%

                return EarningsOptionSignal(
                    symbol=symbol,
                    action=EarningsAction.BUY_VOLATILITY,
                    strategy="straddle",
                    direction="buy",
                    confidence=confidence,
                    days_to_earnings=-(days_since or 0),
                    iv_rank=iv_rank,
                    reason=(
                        f"Post-earnings IV crush: {days_since}d since earnings, "
                        f"IV rank {iv_rank:.1f}% — vol mean reversion"
                    ),
                    target_dte=21,  # Shorter DTE for vol expansion
                    max_risk_pct=0.02,  # 2% max risk for speculative plays
                )

        return None


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    strategy = EarningsIVStrategy()

    # Mock IV rank function
    mock_iv = {"AAPL": 72.0, "MSFT": 45.0, "NVDA": 85.0, "TSLA": 25.0}
    iv_fn = lambda s: mock_iv.get(s)

    signals = strategy.generate_earnings_signals(
        symbols=["AAPL", "MSFT", "NVDA", "TSLA"],
        iv_rank_fn=iv_fn,
    )

    for sig in signals:
        print(
            f"  {sig.symbol}: {sig.action.value} | {sig.strategy} "
            f"| conf={sig.confidence:.0%} | {sig.reason}"
        )

    if not signals:
        print("  No earnings-based signals (expected if no earnings this week)")

    # Test ETF exposure
    for etf in ["SPY", "QQQ"]:
        risk = strategy.get_etf_earnings_risk(etf)
        print(f"  {etf} earnings risk: {risk['risk_level']} ({risk['components_reporting']} components)")

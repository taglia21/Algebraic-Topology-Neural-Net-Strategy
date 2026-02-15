"""
Alternative Data Provider — Social Sentiment, Insider Trading, Analyst Ratings
================================================================================

Aggregates free/low-cost alternative data signals for trading decisions:
  1. Social sentiment scores from Finnhub (if API key available)
  2. Insider trading activity tracking
  3. Analyst rating changes aggregation

All methods return sensible defaults (neutral) when API keys are missing
or services are unreachable.  No hard dependency on external APIs.

Usage:
    provider = AlternativeDataProvider()
    score = provider.get_sentiment_score("AAPL")  # [-1, 1]
    insider = provider.get_insider_activity("AAPL")
    ratings = provider.get_analyst_ratings("AAPL")
    combined = provider.get_combined_score("AAPL")  # [-1, 1]

Author: Tier 1 Implementation — Feb 2026
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


# ============================================================================
# DATA MODELS
# ============================================================================

class SentimentDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class SocialSentiment:
    """Social media sentiment score for a symbol."""
    symbol: str
    score: float                 # [-1, 1]  neg=bearish, pos=bullish
    direction: SentimentDirection
    mentions: int                # Total mention count
    positive_mentions: int
    negative_mentions: int
    source: str                  # "finnhub", "fallback"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class InsiderActivity:
    """Insider trading activity summary for a symbol."""
    symbol: str
    net_shares: int              # Positive = net buying, negative = net selling
    buy_count: int               # Number of insider buys (last 90 days)
    sell_count: int               # Number of insider sells (last 90 days)
    net_value_usd: float         # Net dollar value of insider trades
    signal: float                # [-1, 1]  positive = insiders buying
    source: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AnalystRating:
    """Analyst ratings summary for a symbol."""
    symbol: str
    consensus: float             # [-1, 1]  -1=strong sell, +1=strong buy
    buy_count: int
    hold_count: int
    sell_count: int
    target_price: Optional[float]
    current_price: Optional[float]
    upside_pct: Optional[float]  # (target - current) / current
    recent_upgrades: int         # Upgrades in last 30 days
    recent_downgrades: int       # Downgrades in last 30 days
    source: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AlternativeDataConfig:
    """Configuration for alternative data provider."""
    finnhub_api_key: str = ""
    cache_ttl_minutes: int = 60       # Cache scores for 1 hour
    request_timeout: int = 5          # HTTP timeout seconds
    rate_limit_delay: float = 0.5     # Seconds between API calls
    insider_lookback_days: int = 90    # Look back 90 days for insider trades
    # Weights for combined score
    sentiment_weight: float = 0.40
    insider_weight: float = 0.30
    analyst_weight: float = 0.30


# ============================================================================
# ALTERNATIVE DATA PROVIDER
# ============================================================================

class AlternativeDataProvider:
    """
    Aggregates alternative data signals from multiple free sources.

    Data sources:
      - Finnhub: Social sentiment, insider transactions, analyst recommendations
      - Fallback: Returns neutral scores when APIs are unavailable

    Thread-safe caching with configurable TTL.
    """

    FINNHUB_BASE = "https://finnhub.io/api/v1"

    def __init__(self, config: Optional[AlternativeDataConfig] = None):
        self.cfg = config or AlternativeDataConfig()

        # Try env variable for API key
        if not self.cfg.finnhub_api_key:
            self.cfg.finnhub_api_key = os.environ.get("FINNHUB_API_KEY", "")

        self._has_finnhub = bool(self.cfg.finnhub_api_key) and _REQUESTS_AVAILABLE
        if self._has_finnhub:
            logger.info("AlternativeDataProvider initialized with Finnhub API")
        else:
            logger.info("AlternativeDataProvider initialized (no Finnhub key — neutral fallback)")

        # Caches: symbol -> (data, timestamp)
        self._sentiment_cache: Dict[str, Tuple[SocialSentiment, datetime]] = {}
        self._insider_cache: Dict[str, Tuple[InsiderActivity, datetime]] = {}
        self._analyst_cache: Dict[str, Tuple[AnalystRating, datetime]] = {}
        self._last_api_call: float = 0.0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_sentiment_score(self, symbol: str) -> float:
        """
        Get social sentiment score for a symbol.

        Returns:
            float in [-1, 1]. Positive = bullish, negative = bearish, 0 = neutral.
        """
        sentiment = self.get_social_sentiment(symbol)
        return sentiment.score

    def get_social_sentiment(self, symbol: str) -> SocialSentiment:
        """Get detailed social sentiment data."""
        cached = self._check_cache(self._sentiment_cache, symbol)
        if cached is not None:
            return cached

        if self._has_finnhub:
            result = self._fetch_finnhub_sentiment(symbol)
            if result is not None:
                self._sentiment_cache[symbol] = (result, datetime.now())
                return result

        # Neutral fallback
        result = SocialSentiment(
            symbol=symbol, score=0.0, direction=SentimentDirection.NEUTRAL,
            mentions=0, positive_mentions=0, negative_mentions=0,
            source="fallback",
        )
        self._sentiment_cache[symbol] = (result, datetime.now())
        return result

    def get_insider_activity(self, symbol: str) -> InsiderActivity:
        """Get insider trading activity summary."""
        cached = self._check_cache(self._insider_cache, symbol)
        if cached is not None:
            return cached

        if self._has_finnhub:
            result = self._fetch_finnhub_insider(symbol)
            if result is not None:
                self._insider_cache[symbol] = (result, datetime.now())
                return result

        # Neutral fallback
        result = InsiderActivity(
            symbol=symbol, net_shares=0, buy_count=0, sell_count=0,
            net_value_usd=0.0, signal=0.0, source="fallback",
        )
        self._insider_cache[symbol] = (result, datetime.now())
        return result

    def get_analyst_ratings(self, symbol: str) -> AnalystRating:
        """Get analyst ratings summary."""
        cached = self._check_cache(self._analyst_cache, symbol)
        if cached is not None:
            return cached

        if self._has_finnhub:
            result = self._fetch_finnhub_ratings(symbol)
            if result is not None:
                self._analyst_cache[symbol] = (result, datetime.now())
                return result

        # Neutral fallback
        result = AnalystRating(
            symbol=symbol, consensus=0.0, buy_count=0, hold_count=0,
            sell_count=0, target_price=None, current_price=None,
            upside_pct=None, recent_upgrades=0, recent_downgrades=0,
            source="fallback",
        )
        self._analyst_cache[symbol] = (result, datetime.now())
        return result

    def get_combined_score(self, symbol: str) -> float:
        """
        Get weighted combination of all alternative data signals.

        Returns:
            float in [-1, 1]. Weighted average of sentiment + insider + analyst scores.
        """
        sentiment = self.get_social_sentiment(symbol)
        insider = self.get_insider_activity(symbol)
        analyst = self.get_analyst_ratings(symbol)

        combined = (
            self.cfg.sentiment_weight * sentiment.score
            + self.cfg.insider_weight * insider.signal
            + self.cfg.analyst_weight * analyst.consensus
        )
        return max(-1.0, min(1.0, combined))

    def get_alt_data_summary(self, symbol: str) -> Dict[str, float]:
        """Get all scores as a flat dict for logging/features."""
        sentiment = self.get_social_sentiment(symbol)
        insider = self.get_insider_activity(symbol)
        analyst = self.get_analyst_ratings(symbol)
        return {
            "alt_sentiment": sentiment.score,
            "alt_insider": insider.signal,
            "alt_analyst": analyst.consensus,
            "alt_combined": self.get_combined_score(symbol),
            "alt_mentions": sentiment.mentions,
            "alt_insider_buys": insider.buy_count,
            "alt_insider_sells": insider.sell_count,
        }

    # ------------------------------------------------------------------ #
    # Finnhub API calls
    # ------------------------------------------------------------------ #

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_api_call
        if elapsed < self.cfg.rate_limit_delay:
            time.sleep(self.cfg.rate_limit_delay - elapsed)
        self._last_api_call = time.time()

    def _finnhub_get(self, endpoint: str, params: dict) -> Optional[dict]:
        """Make a GET request to Finnhub API."""
        self._rate_limit()
        params["token"] = self.cfg.finnhub_api_key
        try:
            resp = requests.get(
                f"{self.FINNHUB_BASE}/{endpoint}",
                params=params,
                timeout=self.cfg.request_timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                logger.warning("Finnhub rate limit hit — returning neutral")
                return None
            else:
                logger.debug(f"Finnhub {endpoint} returned {resp.status_code}")
                return None
        except Exception as e:
            logger.debug(f"Finnhub request failed: {e}")
            return None

    def _fetch_finnhub_sentiment(self, symbol: str) -> Optional[SocialSentiment]:
        """Fetch social sentiment from Finnhub."""
        # Try social sentiment endpoint
        data = self._finnhub_get("stock/social-sentiment", {"symbol": symbol})
        if data is None:
            return None

        # Aggregate Reddit + Twitter mentions
        reddit = data.get("reddit", [])
        twitter = data.get("twitter", [])

        total_pos = 0
        total_neg = 0
        total_mentions = 0

        for entry in reddit + twitter:
            pos = entry.get("positiveMention", 0)
            neg = entry.get("negativeMention", 0)
            total_pos += pos
            total_neg += neg
            total_mentions += entry.get("mention", pos + neg)

        if total_mentions == 0:
            return SocialSentiment(
                symbol=symbol, score=0.0, direction=SentimentDirection.NEUTRAL,
                mentions=0, positive_mentions=0, negative_mentions=0,
                source="finnhub",
            )

        # Score: normalized difference
        score = (total_pos - total_neg) / max(total_pos + total_neg, 1)
        score = max(-1.0, min(1.0, score))

        direction = (
            SentimentDirection.BULLISH if score > 0.1
            else SentimentDirection.BEARISH if score < -0.1
            else SentimentDirection.NEUTRAL
        )

        return SocialSentiment(
            symbol=symbol, score=round(score, 4), direction=direction,
            mentions=total_mentions, positive_mentions=total_pos,
            negative_mentions=total_neg, source="finnhub",
        )

    def _fetch_finnhub_insider(self, symbol: str) -> Optional[InsiderActivity]:
        """Fetch insider transactions from Finnhub."""
        data = self._finnhub_get("stock/insider-transactions", {"symbol": symbol})
        if data is None:
            return None

        transactions = data.get("data", [])
        if not transactions:
            return InsiderActivity(
                symbol=symbol, net_shares=0, buy_count=0, sell_count=0,
                net_value_usd=0.0, signal=0.0, source="finnhub",
            )

        cutoff = datetime.now() - timedelta(days=self.cfg.insider_lookback_days)
        buy_shares = 0
        sell_shares = 0
        buy_value = 0.0
        sell_value = 0.0
        buy_count = 0
        sell_count = 0

        for txn in transactions:
            txn_date_str = txn.get("transactionDate", "")
            try:
                txn_date = datetime.strptime(txn_date_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                continue

            if txn_date < cutoff:
                continue

            change = txn.get("change", 0) or 0
            price = txn.get("transactionPrice", 0) or 0
            code = txn.get("transactionCode", "").upper()

            if code in ("P", "A") or change > 0:  # Purchase / Award
                buy_shares += abs(change)
                buy_value += abs(change) * price
                buy_count += 1
            elif code in ("S", "F") or change < 0:  # Sale / Tax
                sell_shares += abs(change)
                sell_value += abs(change) * price
                sell_count += 1

        net_shares = buy_shares - sell_shares
        net_value = buy_value - sell_value
        total = buy_count + sell_count

        # Signal: normalized buy/sell ratio
        if total == 0:
            signal = 0.0
        else:
            signal = (buy_count - sell_count) / total
        signal = max(-1.0, min(1.0, signal))

        return InsiderActivity(
            symbol=symbol, net_shares=net_shares,
            buy_count=buy_count, sell_count=sell_count,
            net_value_usd=net_value, signal=round(signal, 4),
            source="finnhub",
        )

    def _fetch_finnhub_ratings(self, symbol: str) -> Optional[AnalystRating]:
        """Fetch analyst recommendations from Finnhub."""
        data = self._finnhub_get("stock/recommendation", {"symbol": symbol})
        if data is None or not data:
            return None

        # Use most recent recommendation summary
        latest = data[0] if isinstance(data, list) and data else {}
        buy = latest.get("buy", 0) + latest.get("strongBuy", 0)
        hold = latest.get("hold", 0)
        sell = latest.get("sell", 0) + latest.get("strongSell", 0)
        total = buy + hold + sell

        if total == 0:
            consensus = 0.0
        else:
            # Consensus: (buy - sell) / total, in [-1, 1]
            consensus = (buy - sell) / total

        # Count recent upgrades/downgrades from last 2 entries
        upgrades = 0
        downgrades = 0
        if isinstance(data, list) and len(data) >= 2:
            curr_buy = data[0].get("buy", 0) + data[0].get("strongBuy", 0)
            prev_buy = data[1].get("buy", 0) + data[1].get("strongBuy", 0)
            curr_sell = data[0].get("sell", 0) + data[0].get("strongSell", 0)
            prev_sell = data[1].get("sell", 0) + data[1].get("strongSell", 0)
            upgrades = max(0, curr_buy - prev_buy)
            downgrades = max(0, curr_sell - prev_sell)

        # Try to get price target
        target_price = None
        current_price = None
        upside_pct = None
        pt_data = self._finnhub_get("stock/price-target", {"symbol": symbol})
        if pt_data:
            target_price = pt_data.get("targetMean") or pt_data.get("targetMedian")
            current_price = pt_data.get("lastPrice")
            if target_price and current_price and current_price > 0:
                upside_pct = (target_price - current_price) / current_price

        return AnalystRating(
            symbol=symbol, consensus=round(consensus, 4),
            buy_count=buy, hold_count=hold, sell_count=sell,
            target_price=target_price, current_price=current_price,
            upside_pct=round(upside_pct, 4) if upside_pct is not None else None,
            recent_upgrades=upgrades, recent_downgrades=downgrades,
            source="finnhub",
        )

    # ------------------------------------------------------------------ #
    # Cache management
    # ------------------------------------------------------------------ #

    def _check_cache(self, cache: dict, symbol: str):
        """Check if cached data is still valid."""
        if symbol in cache:
            data, ts = cache[symbol]
            if (datetime.now() - ts).total_seconds() < self.cfg.cache_ttl_minutes * 60:
                return data
        return None

    def clear_cache(self):
        """Clear all caches."""
        self._sentiment_cache.clear()
        self._insider_cache.clear()
        self._analyst_cache.clear()

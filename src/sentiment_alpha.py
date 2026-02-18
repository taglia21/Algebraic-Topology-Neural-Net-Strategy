"""
Sentiment Alpha — Multi-Source Sentiment Scoring (TIER 4)
==========================================================

Aggregates sentiment from multiple sources into a single alpha signal
per symbol, ranging from −1 (extreme bearish) to +1 (extreme bullish).

Sources:
1. **News API** — headline sentiment via keyword scoring + VADER
2. **FinBERT embeddings** — transformer-based financial NLP (optional)
3. **Social media** — Reddit WSB / StockTwits buzz proxy
4. **Earnings transcripts** — key-phrase sentiment scoring
5. **get_sentiment_score(symbol)** — composite score [-1, +1]

Design:
- Each source returns [-1, 1]; sources are combined with configurable weights.
- Results are cached with configurable TTL.
- Heavy models (FinBERT) are optional — falls back to keyword scoring.

Usage:
    from src.sentiment_alpha import SentimentAlpha, SentimentConfig

    sa = SentimentAlpha()
    score = sa.get_sentiment_score("AAPL")
    detail = sa.get_detailed_sentiment("AAPL")
"""

import logging
import math
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional heavy dependencies
try:
    import requests as _requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_OK = True
except ImportError:
    VADER_OK = False

try:
    import torch
    from torch import nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


# =============================================================================
# CONFIGURATION
# =============================================================================

class SentimentSource(Enum):
    NEWS = "news"
    FINBERT = "finbert"
    SOCIAL = "social"
    EARNINGS = "earnings"
    KEYWORD = "keyword"


@dataclass
class SentimentConfig:
    """Configuration for multi-source sentiment."""
    # API keys (from env if empty)
    news_api_key: str = ""
    finnhub_api_key: str = ""

    # Source weights (must sum to 1.0 approximately)
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "news": 0.35,
        "keyword": 0.25,
        "social": 0.25,
        "earnings": 0.15,
    })

    # Cache
    cache_ttl_seconds: int = 900            # 15 min
    cache_max_size: int = 200

    # News
    news_lookback_hours: int = 48
    max_news_items: int = 20

    # Time decay
    decay_half_life_hours: float = 24.0

    # FinBERT
    use_finbert: bool = False               # heavy — off by default

    # Thresholds
    strong_threshold: float = 0.5
    moderate_threshold: float = 0.2


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class SentimentItem:
    """Single scored item (headline, post, transcript snippet)."""
    text: str = ""
    source: str = ""
    score: float = 0.0              # [-1, 1]
    timestamp: str = ""
    relevance: float = 1.0          # 0-1


@dataclass
class SentimentDetail:
    """Detailed sentiment breakdown for a symbol."""
    symbol: str = ""
    composite_score: float = 0.0    # [-1, 1]
    signal: str = "neutral"         # bearish / neutral / bullish
    source_scores: Dict[str, float] = field(default_factory=dict)
    items: List[SentimentItem] = field(default_factory=list)
    item_count: int = 0
    cached: bool = False
    timestamp: str = ""


# =============================================================================
# KEYWORD SCORER
# =============================================================================

class FinancialKeywordScorer:
    """
    Rule-based financial sentiment using keyword / phrase lists.
    Fast fallback when VADER or FinBERT are unavailable.
    """

    BULLISH_WORDS = {
        "beat", "beats", "exceeded", "exceeds", "upgrade", "upgrades",
        "outperform", "buy", "bullish", "surged", "surges", "rally",
        "rallied", "growth", "strong", "record", "breakthrough",
        "profit", "dividend", "innovation", "expansion", "upside",
        "positive", "optimistic", "momentum", "recovery",
    }

    BEARISH_WORDS = {
        "miss", "missed", "misses", "downgrade", "downgrades",
        "underperform", "sell", "bearish", "plunged", "plunges",
        "crash", "decline", "declined", "weak", "loss", "losses",
        "risk", "warning", "layoffs", "bankruptcy", "default",
        "negative", "pessimistic", "recession", "investigation",
        "lawsuit", "fraud", "overvalued", "debt",
    }

    @classmethod
    def score(cls, text: str) -> float:
        """Score text ∈ [-1, 1] based on financial keywords."""
        words = set(re.findall(r"\b\w+\b", text.lower()))
        bull = len(words & cls.BULLISH_WORDS)
        bear = len(words & cls.BEARISH_WORDS)
        total = bull + bear
        if total == 0:
            return 0.0
        return (bull - bear) / total


# =============================================================================
# VADER SCORER (OPTIONAL)
# =============================================================================

class VaderScorer:
    """Wrapper around VADER sentiment analyser."""

    def __init__(self):
        self._analyzer = SentimentIntensityAnalyzer() if VADER_OK else None

    def score(self, text: str) -> float:
        if not self._analyzer:
            return FinancialKeywordScorer.score(text)
        scores = self._analyzer.polarity_scores(text)
        return scores["compound"]  # [-1, 1]

    @property
    def available(self) -> bool:
        return self._analyzer is not None


# =============================================================================
# LRU CACHE
# =============================================================================

class _LRUCache:
    def __init__(self, max_size: int = 200, ttl: int = 900):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["ts"] < self._ttl:
                self._cache.move_to_end(key)
                return entry["val"]
            del self._cache[key]
        return None

    def put(self, key: str, val: Any) -> None:
        self._cache[key] = {"val": val, "ts": time.time()}
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


# =============================================================================
# SENTIMENT ALPHA
# =============================================================================

class SentimentAlpha:
    """
    Multi-source sentiment engine producing a composite score.

    Usage:
        sa = SentimentAlpha()
        score = sa.get_sentiment_score("AAPL")       # float [-1, 1]
        detail = sa.get_detailed_sentiment("AAPL")    # SentimentDetail
    """

    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or SentimentConfig()
        self._cache = _LRUCache(self.config.cache_max_size, self.config.cache_ttl_seconds)
        self._vader = VaderScorer()
        self._keyword_scorer = FinancialKeywordScorer()

        # Resolve API keys from env
        if not self.config.news_api_key:
            self.config.news_api_key = os.environ.get("NEWS_API_KEY", "")
        if not self.config.finnhub_api_key:
            self.config.finnhub_api_key = os.environ.get("FINNHUB_API_KEY", "")

        logger.info("SentimentAlpha initialised (vader=%s, finbert=%s)",
                     self._vader.available, self.config.use_finbert)

    # ── Public API ───────────────────────────────────────────────────────

    def get_sentiment_score(self, symbol: str) -> float:
        """
        Get composite sentiment score for a symbol.

        Returns
        -------
        float ∈ [-1, 1]
        """
        detail = self.get_detailed_sentiment(symbol)
        return detail.composite_score

    def get_detailed_sentiment(self, symbol: str) -> SentimentDetail:
        """
        Get full breakdown of sentiment sources.

        Returns
        -------
        SentimentDetail with per-source scores and individual items.
        """
        cached = self._cache.get(symbol)
        if cached is not None:
            cached.cached = True
            return cached

        source_scores: Dict[str, float] = {}
        items: List[SentimentItem] = []

        # 1. News headlines
        news_items = self._fetch_news(symbol)
        if news_items:
            scored = self._score_items(news_items, "news")
            items.extend(scored)
            source_scores["news"] = self._aggregate_scores(scored)

        # 2. Keyword fallback (always available)
        if news_items:
            kw_scores = [self._keyword_scorer.score(it.text) for it in news_items]
            source_scores["keyword"] = float(np.mean(kw_scores)) if kw_scores else 0.0
        else:
            source_scores["keyword"] = 0.0

        # 3. Social buzz (simulated proxy — in production would call APIs)
        social_items = self._fetch_social(symbol)
        if social_items:
            scored = self._score_items(social_items, "social")
            items.extend(scored)
            source_scores["social"] = self._aggregate_scores(scored)
        else:
            source_scores["social"] = 0.0

        # 4. Earnings sentiment
        earnings_items = self._fetch_earnings(symbol)
        if earnings_items:
            scored = self._score_items(earnings_items, "earnings")
            items.extend(scored)
            source_scores["earnings"] = self._aggregate_scores(scored)
        else:
            source_scores["earnings"] = 0.0

        # ── Composite weighted score ──
        weights = self.config.source_weights
        total_weight = sum(weights.get(src, 0) for src in source_scores)
        if total_weight > 0:
            composite = sum(
                source_scores.get(src, 0) * weights.get(src, 0)
                for src in source_scores
            ) / total_weight
        else:
            composite = 0.0

        composite = float(np.clip(composite, -1, 1))

        # Signal
        if composite >= self.config.strong_threshold:
            signal = "strong_bullish"
        elif composite >= self.config.moderate_threshold:
            signal = "bullish"
        elif composite <= -self.config.strong_threshold:
            signal = "strong_bearish"
        elif composite <= -self.config.moderate_threshold:
            signal = "bearish"
        else:
            signal = "neutral"

        detail = SentimentDetail(
            symbol=symbol,
            composite_score=composite,
            signal=signal,
            source_scores=source_scores,
            items=items,
            item_count=len(items),
            cached=False,
            timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        )
        self._cache.put(symbol, detail)
        return detail

    def score_text(self, text: str) -> float:
        """Score arbitrary text [-1, 1]."""
        if self._vader.available:
            return self._vader.score(text)
        return self._keyword_scorer.score(text)

    def inject_items(self, symbol: str, items: List[SentimentItem]) -> None:
        """Inject pre-scored items (for testing or manual override)."""
        self._cache.put(f"_inject_{symbol}", items)

    # ── Data fetching ────────────────────────────────────────────────────

    def _fetch_news(self, symbol: str) -> List[SentimentItem]:
        """Fetch recent news headlines."""
        items: List[SentimentItem] = []

        # Try Finnhub first
        if self.config.finnhub_api_key and REQUESTS_OK:
            try:
                now = datetime.utcnow()
                _from = (now - timedelta(hours=self.config.news_lookback_hours)).strftime("%Y-%m-%d")
                to = now.strftime("%Y-%m-%d")
                url = (
                    f"https://finnhub.io/api/v1/company-news"
                    f"?symbol={symbol}&from={_from}&to={to}"
                    f"&token={self.config.finnhub_api_key}"
                )
                resp = _requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    for article in data[:self.config.max_news_items]:
                        headline = article.get("headline", "")
                        if headline:
                            items.append(SentimentItem(
                                text=headline,
                                source="finnhub",
                                timestamp=datetime.fromtimestamp(
                                    article.get("datetime", 0)
                                ).isoformat(),
                            ))
            except Exception as e:
                logger.debug("Finnhub fetch failed: %s", e)

        # Fallback: yfinance news
        if not items:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                news = getattr(ticker, "news", None)
                if news:
                    for n in news[:self.config.max_news_items]:
                        title = n.get("title", "")
                        if title:
                            items.append(SentimentItem(
                                text=title,
                                source="yfinance",
                            ))
            except Exception as e:
                logger.debug("yfinance news fetch failed: %s", e)

        return items

    def _fetch_social(self, symbol: str) -> List[SentimentItem]:
        """
        Fetch social media mentions.
        In production this would call Reddit / StockTwits APIs.
        Returns empty list as placeholder — inject via inject_items() for testing.
        """
        cached = self._cache.get(f"_inject_{symbol}")
        if cached is not None:
            return [it for it in cached if it.source == "social"]
        return []

    def _fetch_earnings(self, symbol: str) -> List[SentimentItem]:
        """
        Fetch earnings transcript snippets.
        In production this would call an earnings API.
        Returns empty list as placeholder.
        """
        cached = self._cache.get(f"_inject_{symbol}")
        if cached is not None:
            return [it for it in cached if it.source == "earnings"]
        return []

    # ── Scoring helpers ──────────────────────────────────────────────────

    def _score_items(
        self, items: List[SentimentItem], source: str,
    ) -> List[SentimentItem]:
        """Score a list of items in-place."""
        for item in items:
            if item.score == 0.0:
                item.score = self.score_text(item.text)
            item.source = source
        return items

    def _aggregate_scores(self, items: List[SentimentItem]) -> float:
        """Aggregate scores with time decay."""
        if not items:
            return 0.0

        now = time.time()
        half_life = self.config.decay_half_life_hours * 3600
        weighted_sum = 0.0
        weight_total = 0.0

        for item in items:
            # Parse timestamp for decay
            age = 0.0
            if item.timestamp:
                try:
                    ts = datetime.fromisoformat(item.timestamp)
                    age = now - ts.timestamp()
                except (ValueError, TypeError):
                    pass

            decay = math.exp(-0.693 * age / half_life) if half_life > 0 else 1.0
            w = decay * item.relevance
            weighted_sum += item.score * w
            weight_total += w

        return weighted_sum / weight_total if weight_total > 0 else 0.0


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sa = SentimentAlpha()

    # Score some headlines
    headlines = [
        "Apple beats earnings expectations, stock surges",
        "Tesla recalls 500,000 vehicles amid safety investigation",
        "Microsoft announces record cloud growth",
        "Economic recession fears mount as Fed hikes rates",
        "NVIDIA stock rallies on AI breakthrough news",
    ]
    for h in headlines:
        score = sa.score_text(h)
        print(f"  [{score:+.3f}] {h}")

    # Get detailed sentiment (will use cache / API if available)
    detail = sa.get_detailed_sentiment("AAPL")
    print(f"\nAAPL sentiment: {detail.composite_score:+.3f} ({detail.signal})")
    print(f"  Sources: {detail.source_scores}")
    print(f"  Items: {detail.item_count}")


# =============================================================================
# SENTIMENT SIGNAL PROCESSOR — decay + contrarian for strategy_engine
# =============================================================================

@dataclass
class SentimentSignal:
    """Processed sentiment signal for a single symbol."""
    symbol: str
    raw_score: float               # composite [-1, 1]
    decayed_score_6h: float        # fast-decay (intraday)
    decayed_score_24h: float       # slow-decay (swing)
    blended_score: float           # 0.6 × 6h + 0.4 × 24h
    contrarian_flag: bool          # True if extreme → fade
    confidence_boost: float        # 0-0.15 confidence increment for signals
    signal_label: str              # "bullish" / "neutral" / "bearish"
    timestamp: str = ""

    def describe(self) -> str:
        c = "CONTRARIAN" if self.contrarian_flag else ""
        return (
            f"{self.symbol}: {self.signal_label} "
            f"blend={self.blended_score:+.3f} "
            f"(6h={self.decayed_score_6h:+.3f} 24h={self.decayed_score_24h:+.3f}) "
            f"boost={self.confidence_boost:+.3f} {c}"
        ).strip()


class SentimentSignalProcessor:
    """
    Convert raw multi-source sentiment into an actionable trading signal.

    Features:
      - Dual half-life decay: 6h (intraday) and 24h (swing)
      - Blended score: 60% fast + 40% slow
      - Contrarian filter: extreme scores (|score| > 0.8) are faded
      - Confidence boost: moderate agreement → +0-15% signal confidence
      - Source weighting inherited from SentimentAlpha

    Usage::

        processor = SentimentSignalProcessor()
        sig = processor.process("AAPL")
        if sig.confidence_boost > 0:
            trade_signal.confidence += sig.confidence_boost
    """

    def __init__(
        self,
        fast_half_life_hours: float = 6.0,
        slow_half_life_hours: float = 24.0,
        fast_weight: float = 0.60,
        contrarian_threshold: float = 0.80,
        max_confidence_boost: float = 0.15,
        sentiment_config: Optional[SentimentConfig] = None,
    ):
        self.fast_hl = fast_half_life_hours
        self.slow_hl = slow_half_life_hours
        self.fast_w = fast_weight
        self.slow_w = 1.0 - fast_weight
        self.contrarian_thresh = contrarian_threshold
        self.max_boost = max_confidence_boost

        # Two SentimentAlpha instances with different decay rates
        fast_cfg = sentiment_config or SentimentConfig()
        slow_cfg = sentiment_config or SentimentConfig()
        fast_cfg.decay_half_life_hours = self.fast_hl
        slow_cfg.decay_half_life_hours = self.slow_hl
        self._sa_fast = SentimentAlpha(config=fast_cfg)
        self._sa_slow = SentimentAlpha(config=slow_cfg)
        self.logger = logging.getLogger(f"{__name__}.SentimentSignalProcessor")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def process(self, symbol: str) -> SentimentSignal:
        """
        Process sentiment for a symbol with dual-decay and contrarian logic.

        Parameters
        ----------
        symbol : str
            Ticker symbol.

        Returns
        -------
        SentimentSignal
        """
        score_fast = self._sa_fast.get_sentiment_score(symbol)
        score_slow = self._sa_slow.get_sentiment_score(symbol)

        blended = self.fast_w * score_fast + self.slow_w * score_slow

        # Contrarian filter: extreme consensus → fade
        contrarian = abs(blended) > self.contrarian_thresh
        if contrarian:
            # Reduce magnitude by 50% (market often reverses at extremes)
            blended *= 0.50
            self.logger.info(
                f"Contrarian fade on {symbol}: raw_blend={score_fast:.3f}/{score_slow:.3f} "
                f"→ faded {blended:+.3f}"
            )

        # Confidence boost: proportional to |blended|, capped
        boost = min(abs(blended) * self.max_boost, self.max_boost)

        # Label
        if blended > 0.15:
            label = "bullish"
        elif blended < -0.15:
            label = "bearish"
        else:
            label = "neutral"

        raw_composite = self._sa_fast.get_sentiment_score(symbol)  # use cached

        return SentimentSignal(
            symbol=symbol,
            raw_score=raw_composite,
            decayed_score_6h=score_fast,
            decayed_score_24h=score_slow,
            blended_score=blended,
            contrarian_flag=contrarian,
            confidence_boost=boost,
            signal_label=label,
            timestamp=datetime.utcnow().isoformat(),
        )

    def process_batch(self, symbols: List[str]) -> Dict[str, SentimentSignal]:
        """Process sentiment for multiple symbols."""
        return {sym: self.process(sym) for sym in symbols}

    def inject_items(self, symbol: str, items: List[SentimentItem]) -> None:
        """Inject items into both fast and slow analyzers (for testing)."""
        self._sa_fast.inject_items(symbol, items)
        self._sa_slow.inject_items(symbol, items)

"""
Production-Ready Sentiment Analysis Module
===========================================

Market sentiment analysis from news and social signals with time decay.

Features:
- Primary: Finnhub News API (free tier)
- Fallback: yfinance news property
- VADER sentiment scoring (optimized for financial text)
- Exponential time decay (24hr half-life)
- Aggregate score: -1.0 (extreme fear) to +1.0 (extreme greed)
- 15-minute result caching

Author: Trading System
Version: 1.0.0
"""

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math

import requests
import yfinance as yf

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logging.warning("VADER not available. Install with: pip install vaderSentiment")

logger = logging.getLogger(__name__)


class SentimentLevel(Enum):
    """Sentiment classification levels."""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


@dataclass
class NewsArticle:
    """Represents a single news article."""
    headline: str
    summary: str
    source: str
    published_at: datetime
    url: Optional[str] = None
    sentiment_score: Optional[float] = None
    
    @property
    def age_hours(self) -> float:
        """Calculate article age in hours."""
        return (datetime.now() - self.published_at).total_seconds() / 3600


@dataclass
class SentimentResult:
    """Complete sentiment analysis result."""
    symbol: str
    timestamp: datetime
    score: float  # -1.0 to 1.0
    level: SentimentLevel
    article_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    articles: List[NewsArticle]
    data_source: str
    is_valid: bool = True  # HIGH-SEVERITY FIX: Track data validity
    
    @property
    def is_bullish(self) -> bool:
        """Check if sentiment is bullish."""
        return self.score > 0.3
    
    @property
    def is_bearish(self) -> bool:
        """Check if sentiment is bearish."""
        return self.score < -0.3
    
    @property
    def confidence(self) -> float:
        """Calculate confidence based on article count and score magnitude."""
        # More articles = higher confidence
        article_factor = min(1.0, self.article_count / 20)
        # Stronger sentiment = higher confidence
        magnitude_factor = abs(self.score)
        return (article_factor + magnitude_factor) / 2


@dataclass
class SentimentConfig:
    """Sentiment analyzer configuration."""
    # API keys
    finnhub_api_key: Optional[str] = None
    
    # Data sources
    use_finnhub: bool = True
    use_yfinance: bool = True
    
    # Sentiment parameters
    lookback_hours: int = 48  # How far back to fetch news
    time_decay_halflife_hours: float = 24.0  # Exponential decay half-life
    min_articles: int = 3  # Minimum articles for valid sentiment
    
    # Scoring thresholds
    extreme_fear_threshold: float = -0.6
    fear_threshold: float = -0.3
    greed_threshold: float = 0.3
    extreme_greed_threshold: float = 0.6
    
    # Caching
    cache_ttl_seconds: int = 900  # 15 minutes
    
    # VADER customization
    use_vader: bool = True
    vader_threshold: float = 0.2  # HIGH-SEVERITY FIX: More realistic threshold for financial news


class SentimentAnalyzer:
    """
    Market sentiment analyzer using news and social signals.
    
    Fetches news from multiple sources, scores sentiment using VADER,
    and applies time decay for recency weighting.
    """
    
    def __init__(self, config: Optional[SentimentConfig] = None):
        """
        Initialize sentiment analyzer.
        
        Args:
            config: Sentiment configuration (uses defaults if None)
        """
        self.config = config or SentimentConfig()
        
        # Try to get API key from environment if not provided
        if self.config.finnhub_api_key is None:
            self.config.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        
        # Initialize VADER
        self.vader = None
        if self.config.use_vader and VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer initialized")
        else:
            logger.warning("VADER not available, using basic sentiment")
        
        # CRITICAL FIX: Use OrderedDict for LRU cache with size limit
        self.cache: OrderedDict = OrderedDict()
        self.max_cache_size = 100  # Limit to 100 symbols
        self.cache_ttl = timedelta(seconds=self.config.cache_ttl_seconds)
        
        # HIGH-SEVERITY FIX: Create session with default timeout
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'TradingSystem/1.0'})
        self.default_timeout = 10  # 10 second timeout
        
        logger.info(f"SentimentAnalyzer initialized (Finnhub: {bool(self.config.finnhub_api_key)}, "
                   f"VADER: {bool(self.vader)}, max_cache={self.max_cache_size})")
    
    def _fetch_finnhub_news(self, symbol: str) -> List[NewsArticle]:
        """
        Fetch news from Finnhub API.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of NewsArticle objects
        """
        if not self.config.use_finnhub or not self.config.finnhub_api_key:
            return []
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=self.config.lookback_hours)
            
            # Format dates for API
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')
            
            # API endpoint
            url = "https://finnhub.io/api/v1/company-news"
            
            # HIGH-SEVERITY FIX: Use headers for API key instead of query params
            headers = {
                'X-Finnhub-Token': self.config.finnhub_api_key
            }
            
            params = {
                'symbol': symbol,
                'from': from_date,
                'to': to_date
            }
            
            # Make request with timeout
            response = self.session.get(url, params=params, headers=headers, 
                                      timeout=self.default_timeout)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            for item in data:
                try:
                    # Parse timestamp (Unix timestamp)
                    published_at = datetime.fromtimestamp(item.get('datetime', 0))
                    
                    article = NewsArticle(
                        headline=item.get('headline', ''),
                        summary=item.get('summary', ''),
                        source=item.get('source', 'Finnhub'),
                        published_at=published_at,
                        url=item.get('url')
                    )
                    
                    if article.headline:  # Only add if has headline
                        articles.append(article)
                
                except Exception as e:
                    logger.debug(f"Error parsing Finnhub article: {e}")
                    continue
            
            logger.info(f"Fetched {len(articles)} articles from Finnhub for {symbol}")
            return articles
        
        except Exception as e:
            logger.error(f"Error fetching Finnhub news for {symbol}: {e}")
            return []
    
    def _fetch_yfinance_news(self, symbol: str) -> List[NewsArticle]:
        """
        Fetch news from yfinance (fallback).
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of NewsArticle objects
        """
        if not self.config.use_yfinance:
            return []
        
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                logger.warning(f"No news from yfinance for {symbol}")
                return []
            
            articles = []
            cutoff_time = datetime.now() - timedelta(hours=self.config.lookback_hours)
            
            for item in news:
                try:
                    # Parse timestamp
                    timestamp = item.get('providerPublishTime', 0)
                    published_at = datetime.fromtimestamp(timestamp)
                    
                    # Skip old articles
                    if published_at < cutoff_time:
                        continue
                    
                    article = NewsArticle(
                        headline=item.get('title', ''),
                        summary=item.get('summary', item.get('title', '')),
                        source=item.get('publisher', 'Yahoo Finance'),
                        published_at=published_at,
                        url=item.get('link')
                    )
                    
                    if article.headline:
                        articles.append(article)
                
                except Exception as e:
                    logger.debug(f"Error parsing yfinance article: {e}")
                    continue
            
            logger.info(f"Fetched {len(articles)} articles from yfinance for {symbol}")
            return articles
        
        except Exception as e:
            logger.error(f"Error fetching yfinance news for {symbol}: {e}")
            return []
    
    def _score_sentiment_vader(self, text: str) -> float:
        """
        Score sentiment using VADER.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score (-1.0 to 1.0)
        """
        if not self.vader:
            return 0.0
        
        try:
            scores = self.vader.polarity_scores(text)
            # Use compound score (normalized -1 to 1)
            return scores['compound']
        except Exception as e:
            logger.debug(f"Error scoring sentiment: {e}")
            return 0.0
    
    def _score_sentiment_basic(self, text: str) -> float:
        """
        Basic sentiment scoring (fallback if VADER unavailable).
        
        Uses simple keyword matching for positive/negative words.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score (-1.0 to 1.0)
        """
        text_lower = text.lower()
        
        # Basic positive words
        positive_words = [
            'bullish', 'surge', 'soar', 'rally', 'gain', 'profit', 'beat',
            'upgrade', 'strong', 'growth', 'positive', 'up', 'rise', 'high',
            'outperform', 'excellent', 'good', 'success', 'win', 'breakthrough'
        ]
        
        # Basic negative words
        negative_words = [
            'bearish', 'plunge', 'crash', 'fall', 'loss', 'miss', 'downgrade',
            'weak', 'decline', 'negative', 'down', 'drop', 'low', 'underperform',
            'poor', 'bad', 'fail', 'concern', 'risk', 'warning'
        ]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        # Normalize to -1 to 1
        score = (pos_count - neg_count) / total
        return score
    
    def _calculate_time_decay_weight(self, age_hours: float) -> float:
        """
        Calculate exponential time decay weight.
        
        Uses half-life decay: weight = 0.5^(age / half_life)
        
        Args:
            age_hours: Age of article in hours
            
        Returns:
            Decay weight (0.0 to 1.0)
        """
        halflife = self.config.time_decay_halflife_hours
        decay_weight = math.pow(0.5, age_hours / halflife)
        return decay_weight
    
    def _analyze_articles(self, articles: List[NewsArticle]) -> Tuple[float, int, int, int]:
        """
        Analyze sentiment of multiple articles with time decay.
        
        Args:
            articles: List of NewsArticle objects
            
        Returns:
            (weighted_score, positive_count, negative_count, neutral_count)
        """
        if not articles:
            return 0.0, 0, 0, 0
        
        weighted_sum = 0.0
        weight_sum = 0.0
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for article in articles:
            # Combine headline and summary for scoring
            text = f"{article.headline} {article.summary}"
            
            # Score sentiment
            if self.vader:
                score = self._score_sentiment_vader(text)
            else:
                score = self._score_sentiment_basic(text)
            
            # Store in article
            article.sentiment_score = score
            
            # Classify
            if score > self.config.vader_threshold:
                positive_count += 1
            elif score < -self.config.vader_threshold:
                negative_count += 1
            else:
                neutral_count += 1
            
            # Calculate time decay weight
            weight = self._calculate_time_decay_weight(article.age_hours)
            
            # Add to weighted sum
            weighted_sum += score * weight
            weight_sum += weight
        
        # Calculate final weighted score
        if weight_sum > 0:
            final_score = weighted_sum / weight_sum
        else:
            final_score = 0.0
        
        # Clip to [-1, 1]
        final_score = max(-1.0, min(1.0, final_score))
        
        return final_score, positive_count, negative_count, neutral_count
    
    def _classify_sentiment(self, score: float) -> SentimentLevel:
        """
        Classify sentiment score into level.
        
        Args:
            score: Sentiment score (-1.0 to 1.0)
            
        Returns:
            SentimentLevel enum
        """
        if score <= self.config.extreme_fear_threshold:
            return SentimentLevel.EXTREME_FEAR
        elif score <= self.config.fear_threshold:
            return SentimentLevel.FEAR
        elif score >= self.config.extreme_greed_threshold:
            return SentimentLevel.EXTREME_GREED
        elif score >= self.config.greed_threshold:
            return SentimentLevel.GREED
        else:
            return SentimentLevel.NEUTRAL
    
    def get_sentiment(self, symbol: str, use_cache: bool = True) -> SentimentResult:
        """
        Get sentiment analysis for a symbol.
        
        Args:
            symbol: Stock symbol to analyze
            use_cache: Whether to use cached results
            
        Returns:
            SentimentResult with complete analysis
        """
        # Check cache
        if use_cache and symbol in self.cache:
            cached_result, cache_time = self.cache[symbol]
            age = (datetime.now() - cache_time).total_seconds()
            
            if age < self.config.cache_ttl_seconds:
                logger.debug(f"Using cached sentiment for {symbol} ({age:.0f}s old)")
                return cached_result
        
        logger.info(f"Analyzing sentiment for {symbol}")
        
        # Fetch news from multiple sources
        articles = []
        data_sources = []
        
        # Try Finnhub first
        finnhub_articles = self._fetch_finnhub_news(symbol)
        if finnhub_articles:
            articles.extend(finnhub_articles)
            data_sources.append("Finnhub")
        
        # Fallback to yfinance if needed
        if len(articles) < self.config.min_articles:
            yf_articles = self._fetch_yfinance_news(symbol)
            if yf_articles:
                articles.extend(yf_articles)
                data_sources.append("YahooFinance")
        
        # Sort by date (newest first)
        articles.sort(key=lambda x: x.published_at, reverse=True)
        
        # Analyze sentiment
        if len(articles) >= self.config.min_articles:
            score, pos_count, neg_count, neu_count = self._analyze_articles(articles)
            level = self._classify_sentiment(score)
            data_source = " + ".join(data_sources)
            is_valid = True
        else:
            # HIGH-SEVERITY FIX: Mark as INVALID when insufficient data
            logger.warning(f"SENTIMENT DATA UNAVAILABLE for {symbol}: {len(articles)} < {self.config.min_articles}")
            score = 0.0
            pos_count = neg_count = neu_count = 0
            level = SentimentLevel.NEUTRAL
            data_source = "INSUFFICIENT_DATA"
            is_valid = False  # Mark as invalid for downstream rejection
        
        # Create result
        result = SentimentResult(
            symbol=symbol,
            timestamp=datetime.now(),
            score=score,
            level=level,
            article_count=len(articles),
            positive_count=pos_count,
            negative_count=neg_count,
            neutral_count=neu_count,
            articles=articles[:20],  # Keep top 20 for reference
            data_source=data_source,
            is_valid=is_valid
        )
        
        # CRITICAL FIX: Add to cache with LRU eviction
        self.cache[symbol] = (result, datetime.now())
        
        # Evict oldest if over limit
        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)  # Remove oldest entry
            logger.debug(f"Cache LRU eviction: size was {len(self.cache) + 1}, now {len(self.cache)}")
        
        # Periodic cleanup of expired entries (every 10th addition)
        if len(self.cache) % 10 == 0:
            self._cleanup_expired_cache()
        
        logger.info(f"{symbol} sentiment: {score:.2f} ({level.value}), "
                   f"{len(articles)} articles, source: {data_source}, valid: {is_valid}")
        
        return result
    
    def _cleanup_expired_cache(self):
        """Remove expired cache entries based on TTL."""
        now = datetime.now()
        expired = [k for k, (_, ts) in self.cache.items() 
                  if now - ts > self.cache_ttl]
        
        for key in expired:
            del self.cache[key]
        
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired cache entries")
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear sentiment cache.
        
        Args:
            symbol: Clear specific symbol, or all if None
        """
        if symbol:
            if symbol in self.cache:
                del self.cache[symbol]
                logger.debug(f"Cleared cache for {symbol}")
        else:
            self.cache.clear()
            logger.debug("Cleared all cache")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Analyze a symbol
    symbol = "AMD"
    result = analyzer.get_sentiment(symbol)
    
    print(f"\n{'='*60}")
    print(f"Sentiment Analysis: {symbol}")
    print(f"{'='*60}")
    print(f"Score: {result.score:+.2f} ({result.level.value})")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Articles: {result.article_count} ({result.positive_count}+ / {result.negative_count}- / {result.neutral_count}=)")
    print(f"Source: {result.data_source}")
    print(f"Bullish: {'YES' if result.is_bullish else 'NO'}")
    print(f"Bearish: {'YES' if result.is_bearish else 'NO'}")
    
    if result.articles:
        print(f"\nTop Headlines:")
        for i, article in enumerate(result.articles[:5], 1):
            age = article.age_hours
            score = article.sentiment_score or 0.0
            print(f"  {i}. [{score:+.2f}] ({age:.1f}h ago) {article.headline[:80]}")

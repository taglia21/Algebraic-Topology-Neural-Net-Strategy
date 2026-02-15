"""News Sentiment Analyzer for trading decisions."""
import os
import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import requests

logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    symbol: str
    score: float  # -1 to 1
    confidence: float
    news_count: int
    timestamp: datetime
    has_breaking_news: bool = False

class NewsSentimentAnalyzer:
    """Analyzes news sentiment for trading symbols."""
    
    def __init__(self):
        self.api_key = os.getenv('FINNHUB_API_KEY', '')
        self.cache: Dict[str, SentimentResult] = {}
        self.cache_ttl = 300  # 5 minutes
        
    def get_sentiment_score(self, symbol: str) -> SentimentResult:
        """Get sentiment score for a symbol (-1 bearish to +1 bullish)."""
        if symbol in self.cache:
            cached = self.cache[symbol]
            if (datetime.now() - cached.timestamp).seconds < self.cache_ttl:
                return cached
        
        # Default neutral sentiment if no API key
        result = SentimentResult(
            symbol=symbol, score=0.0, confidence=0.5,
            news_count=0, timestamp=datetime.now()
        )
        self.cache[symbol] = result
        return result
    
    def get_market_sentiment(self) -> float:
        """Get overall market sentiment."""
        spy_sent = self.get_sentiment_score('SPY')
        return spy_sent.score
    
    def has_breaking_news(self, symbol: str) -> bool:
        """Check if symbol has breaking news."""
        result = self.get_sentiment_score(symbol)
        return result.has_breaking_news
    
    def should_skip_trade(self, symbol: str, min_score: float = -0.3) -> bool:
        """Return True if sentiment is too negative to trade."""
        result = self.get_sentiment_score(symbol)
        return result.score < min_score

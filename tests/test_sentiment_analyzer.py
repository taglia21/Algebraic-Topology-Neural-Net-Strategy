"""
Unit Tests for Sentiment Analyzer Module
=========================================

Tests all sentiment analysis functionality including:
- News fetching (Finnhub, yfinance)
- VADER sentiment scoring
- Time decay calculations
- Sentiment aggregation
- Caching
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import math

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentConfig,
    SentimentLevel,
    NewsArticle,
    SentimentResult
)


class TestNewsArticle(unittest.TestCase):
    """Test NewsArticle dataclass."""
    
    def test_age_calculation(self):
        """Test article age calculation."""
        # Article from 2 hours ago
        published = datetime.now() - timedelta(hours=2)
        article = NewsArticle(
            headline="Test",
            summary="Summary",
            source="Test",
            published_at=published
        )
        
        self.assertAlmostEqual(article.age_hours, 2.0, places=1)
    
    def test_fresh_article(self):
        """Test fresh article age."""
        article = NewsArticle(
            headline="Test",
            summary="Summary",
            source="Test",
            published_at=datetime.now()
        )
        
        self.assertLess(article.age_hours, 0.1)


class TestSentimentResult(unittest.TestCase):
    """Test SentimentResult dataclass."""
    
    def test_bullish_detection(self):
        """Test bullish sentiment detection."""
        result = SentimentResult(
            symbol="TEST",
            timestamp=datetime.now(),
            score=0.5,
            level=SentimentLevel.GREED,
            article_count=10,
            positive_count=7,
            negative_count=3,
            neutral_count=0,
            articles=[],
            data_source="test"
        )
        
        self.assertTrue(result.is_bullish)
        self.assertFalse(result.is_bearish)
    
    def test_bearish_detection(self):
        """Test bearish sentiment detection."""
        result = SentimentResult(
            symbol="TEST",
            timestamp=datetime.now(),
            score=-0.5,
            level=SentimentLevel.FEAR,
            article_count=10,
            positive_count=3,
            negative_count=7,
            neutral_count=0,
            articles=[],
            data_source="test"
        )
        
        self.assertTrue(result.is_bearish)
        self.assertFalse(result.is_bullish)
    
    def test_neutral_detection(self):
        """Test neutral sentiment detection."""
        result = SentimentResult(
            symbol="TEST",
            timestamp=datetime.now(),
            score=0.1,
            level=SentimentLevel.NEUTRAL,
            article_count=10,
            positive_count=5,
            negative_count=5,
            neutral_count=0,
            articles=[],
            data_source="test"
        )
        
        self.assertFalse(result.is_bullish)
        self.assertFalse(result.is_bearish)
    
    def test_confidence_calculation(self):
        """Test confidence calculation."""
        # High confidence: many articles, strong sentiment
        high = SentimentResult(
            symbol="TEST",
            timestamp=datetime.now(),
            score=0.8,
            level=SentimentLevel.EXTREME_GREED,
            article_count=20,
            positive_count=18,
            negative_count=2,
            neutral_count=0,
            articles=[],
            data_source="test"
        )
        
        # Low confidence: few articles, weak sentiment
        low = SentimentResult(
            symbol="TEST",
            timestamp=datetime.now(),
            score=0.1,
            level=SentimentLevel.NEUTRAL,
            article_count=3,
            positive_count=2,
            negative_count=1,
            neutral_count=0,
            articles=[],
            data_source="test"
        )
        
        self.assertGreater(high.confidence, low.confidence)


class TestTimeDecay(unittest.TestCase):
    """Test time decay calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
    
    def test_fresh_article_full_weight(self):
        """Test fresh article gets full weight."""
        weight = self.analyzer._calculate_time_decay_weight(0.0)
        self.assertAlmostEqual(weight, 1.0, places=2)
    
    def test_halflife_decay(self):
        """Test half-life decay at exactly half-life hours."""
        halflife = self.analyzer.config.time_decay_halflife_hours
        weight = self.analyzer._calculate_time_decay_weight(halflife)
        self.assertAlmostEqual(weight, 0.5, places=2)
    
    def test_double_halflife(self):
        """Test weight at 2x half-life."""
        halflife = self.analyzer.config.time_decay_halflife_hours
        weight = self.analyzer._calculate_time_decay_weight(halflife * 2)
        self.assertAlmostEqual(weight, 0.25, places=2)
    
    def test_old_article_low_weight(self):
        """Test very old article has minimal weight."""
        weight = self.analyzer._calculate_time_decay_weight(100.0)
        self.assertLess(weight, 0.1)


class TestSentimentScoring(unittest.TestCase):
    """Test sentiment scoring methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
    
    def test_basic_positive_scoring(self):
        """Test basic positive sentiment detection."""
        text = "Stock surges on strong earnings beat and positive outlook"
        score = self.analyzer._score_sentiment_basic(text)
        self.assertGreater(score, 0)
    
    def test_basic_negative_scoring(self):
        """Test basic negative sentiment detection."""
        text = "Stock plunges on weak earnings miss and concerns about outlook"
        score = self.analyzer._score_sentiment_basic(text)
        self.assertLess(score, 0)
    
    def test_basic_neutral_scoring(self):
        """Test basic neutral sentiment."""
        text = "Company announces new product availability"
        score = self.analyzer._score_sentiment_basic(text)
        # Should be close to neutral
        self.assertAlmostEqual(score, 0.0, places=1)
    
    @unittest.skipIf(not hasattr(SentimentAnalyzer(), 'vader') or 
                     SentimentAnalyzer().vader is None,
                     "VADER not available")
    def test_vader_positive_scoring(self):
        """Test VADER positive sentiment."""
        text = "Excellent quarter with outstanding growth and great prospects!"
        score = self.analyzer._score_sentiment_vader(text)
        self.assertGreater(score, 0.3)
    
    @unittest.skipIf(not hasattr(SentimentAnalyzer(), 'vader') or 
                     SentimentAnalyzer().vader is None,
                     "VADER not available")
    def test_vader_negative_scoring(self):
        """Test VADER negative sentiment."""
        text = "Terrible results, horrible performance, very disappointing"
        score = self.analyzer._score_sentiment_vader(text)
        self.assertLess(score, -0.3)


class TestSentimentClassification(unittest.TestCase):
    """Test sentiment level classification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
    
    def test_extreme_fear(self):
        """Test extreme fear classification."""
        level = self.analyzer._classify_sentiment(-0.7)
        self.assertEqual(level, SentimentLevel.EXTREME_FEAR)
    
    def test_fear(self):
        """Test fear classification."""
        level = self.analyzer._classify_sentiment(-0.4)
        self.assertEqual(level, SentimentLevel.FEAR)
    
    def test_neutral(self):
        """Test neutral classification."""
        level = self.analyzer._classify_sentiment(0.0)
        self.assertEqual(level, SentimentLevel.NEUTRAL)
    
    def test_greed(self):
        """Test greed classification."""
        level = self.analyzer._classify_sentiment(0.4)
        self.assertEqual(level, SentimentLevel.GREED)
    
    def test_extreme_greed(self):
        """Test extreme greed classification."""
        level = self.analyzer._classify_sentiment(0.7)
        self.assertEqual(level, SentimentLevel.EXTREME_GREED)


class TestArticleAnalysis(unittest.TestCase):
    """Test multi-article sentiment analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
    
    def create_test_articles(self, sentiments: list) -> list:
        """Create test articles with specified sentiments."""
        articles = []
        base_time = datetime.now()
        
        for i, sentiment in enumerate(sentiments):
            if sentiment == 'positive':
                headline = "Stock surges on strong earnings beat"
            elif sentiment == 'negative':
                headline = "Stock plunges on weak outlook"
            else:
                headline = "Company announces new product"
            
            article = NewsArticle(
                headline=headline,
                summary="",
                source="Test",
                published_at=base_time - timedelta(hours=i)
            )
            articles.append(article)
        
        return articles
    
    def test_all_positive_articles(self):
        """Test analysis of all positive articles."""
        articles = self.create_test_articles(['positive'] * 5)
        score, pos, neg, neu = self.analyzer._analyze_articles(articles)
        
        self.assertGreater(score, 0)
        self.assertGreater(pos, neg)
    
    def test_all_negative_articles(self):
        """Test analysis of all negative articles."""
        articles = self.create_test_articles(['negative'] * 5)
        score, pos, neg, neu = self.analyzer._analyze_articles(articles)
        
        self.assertLess(score, 0)
        self.assertGreater(neg, pos)
    
    def test_mixed_articles(self):
        """Test analysis of mixed articles."""
        articles = self.create_test_articles(['positive', 'negative', 'neutral'])
        score, pos, neg, neu = self.analyzer._analyze_articles(articles)
        
        # Should be relatively neutral
        self.assertGreater(score, -0.5)
        self.assertLess(score, 0.5)
    
    def test_recent_articles_weighted_more(self):
        """Test that recent articles have more impact."""
        # Recent positive, old negative
        recent_positive = self.create_test_articles(['positive'])
        recent_positive[0].published_at = datetime.now()
        
        old_negative = self.create_test_articles(['negative'])
        old_negative[0].published_at = datetime.now() - timedelta(hours=48)
        
        articles = recent_positive + old_negative
        score, _, _, _ = self.analyzer._analyze_articles(articles)
        
        # Recent positive should dominate
        self.assertGreater(score, 0)
    
    def test_empty_articles(self):
        """Test handling of empty article list."""
        score, pos, neg, neu = self.analyzer._analyze_articles([])
        
        self.assertEqual(score, 0.0)
        self.assertEqual(pos, 0)
        self.assertEqual(neg, 0)
        self.assertEqual(neu, 0)


class TestNewsFetching(unittest.TestCase):
    """Test news fetching from various sources."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
    
    @patch('requests.get')
    def test_finnhub_fetch_success(self, mock_get):
        """Test successful Finnhub news fetch."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                'headline': 'Test headline',
                'summary': 'Test summary',
                'source': 'Test Source',
                'datetime': datetime.now().timestamp(),
                'url': 'http://test.com'
            }
        ]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Set API key
        self.analyzer.config.finnhub_api_key = 'test_key'
        
        articles = self.analyzer._fetch_finnhub_news('TEST')
        
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].headline, 'Test headline')
    
    @patch('requests.get')
    def test_finnhub_fetch_error(self, mock_get):
        """Test Finnhub fetch error handling."""
        mock_get.side_effect = Exception("Network error")
        
        self.analyzer.config.finnhub_api_key = 'test_key'
        articles = self.analyzer._fetch_finnhub_news('TEST')
        
        self.assertEqual(len(articles), 0)
    
    @patch('yfinance.Ticker')
    def test_yfinance_fetch_success(self, mock_ticker):
        """Test successful yfinance news fetch."""
        # Mock news data
        mock_ticker.return_value.news = [
            {
                'title': 'Test headline',
                'summary': 'Test summary',
                'publisher': 'Yahoo',
                'providerPublishTime': datetime.now().timestamp(),
                'link': 'http://test.com'
            }
        ]
        
        articles = self.analyzer._fetch_yfinance_news('TEST')
        
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].headline, 'Test headline')
    
    @patch('yfinance.Ticker')
    def test_yfinance_fetch_error(self, mock_ticker):
        """Test yfinance fetch error handling."""
        mock_ticker.side_effect = Exception("API error")
        
        articles = self.analyzer._fetch_yfinance_news('TEST')
        
        self.assertEqual(len(articles), 0)
    
    @patch('yfinance.Ticker')
    def test_yfinance_filters_old_articles(self, mock_ticker):
        """Test yfinance filters out old articles."""
        old_time = (datetime.now() - timedelta(hours=100)).timestamp()
        
        mock_ticker.return_value.news = [
            {
                'title': 'Old article',
                'providerPublishTime': old_time,
            }
        ]
        
        articles = self.analyzer._fetch_yfinance_news('TEST')
        
        # Should filter out old article
        self.assertEqual(len(articles), 0)


class TestSentimentIntegration(unittest.TestCase):
    """Test complete sentiment analysis flow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
    
    @patch.object(SentimentAnalyzer, '_fetch_finnhub_news')
    @patch.object(SentimentAnalyzer, '_fetch_yfinance_news')
    def test_complete_analysis(self, mock_yf, mock_fh):
        """Test complete sentiment analysis."""
        # Mock articles
        articles = [
            NewsArticle(
                headline="Stock surges on earnings",
                summary="Strong results",
                source="Test",
                published_at=datetime.now()
            ),
            NewsArticle(
                headline="Positive outlook announced",
                summary="Great prospects",
                source="Test",
                published_at=datetime.now() - timedelta(hours=1)
            ),
            NewsArticle(
                headline="Analyst upgrades stock",
                summary="Buy rating",
                source="Test",
                published_at=datetime.now() - timedelta(hours=2)
            )
        ]
        
        mock_fh.return_value = articles
        mock_yf.return_value = []
        
        result = self.analyzer.get_sentiment('TEST', use_cache=False)
        
        self.assertEqual(result.symbol, 'TEST')
        self.assertEqual(result.article_count, 3)
        self.assertGreater(result.score, 0)
    
    @patch.object(SentimentAnalyzer, '_fetch_finnhub_news')
    @patch.object(SentimentAnalyzer, '_fetch_yfinance_news')
    def test_insufficient_articles(self, mock_yf, mock_fh):
        """Test handling of insufficient articles."""
        mock_fh.return_value = []
        mock_yf.return_value = [NewsArticle(
            headline="Single article",
            summary="",
            source="Test",
            published_at=datetime.now()
        )]
        
        result = self.analyzer.get_sentiment('TEST', use_cache=False)
        
        # Should return neutral with insufficient data
        self.assertEqual(result.level, SentimentLevel.NEUTRAL)
        self.assertEqual(result.data_source, "insufficient_data")
    
    @patch.object(SentimentAnalyzer, '_fetch_finnhub_news')
    @patch.object(SentimentAnalyzer, '_fetch_yfinance_news')
    def test_fallback_to_yfinance(self, mock_yf, mock_fh):
        """Test fallback to yfinance when Finnhub fails."""
        mock_fh.return_value = []  # Finnhub fails
        
        # Provide enough yfinance articles
        mock_yf.return_value = [
            NewsArticle(f"Article {i}", "", "YF", datetime.now())
            for i in range(5)
        ]
        
        result = self.analyzer.get_sentiment('TEST', use_cache=False)
        
        self.assertIn("YahooFinance", result.data_source)
        self.assertGreaterEqual(result.article_count, 3)


class TestCaching(unittest.TestCase):
    """Test caching functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
    
    @patch.object(SentimentAnalyzer, '_fetch_finnhub_news')
    def test_cache_hit(self, mock_fetch):
        """Test cache returns stored results."""
        # Create mock result
        mock_result = SentimentResult(
            symbol="TEST",
            timestamp=datetime.now(),
            score=0.5,
            level=SentimentLevel.GREED,
            article_count=10,
            positive_count=7,
            negative_count=3,
            neutral_count=0,
            articles=[],
            data_source="test"
        )
        
        # Add to cache
        self.analyzer.cache["TEST"] = (mock_result, datetime.now())
        
        # Get sentiment should use cache
        result = self.analyzer.get_sentiment("TEST", use_cache=True)
        
        # Should not have called fetch
        mock_fetch.assert_not_called()
        self.assertEqual(result.score, 0.5)
    
    def test_cache_expiry(self):
        """Test cache expires after TTL."""
        mock_result = SentimentResult(
            symbol="TEST",
            timestamp=datetime.now(),
            score=0.5,
            level=SentimentLevel.GREED,
            article_count=10,
            positive_count=7,
            negative_count=3,
            neutral_count=0,
            articles=[],
            data_source="test"
        )
        
        # Add expired cache entry
        old_time = datetime.now() - timedelta(seconds=self.analyzer.config.cache_ttl_seconds + 10)
        self.analyzer.cache["TEST"] = (mock_result, old_time)
        
        # Should not use expired cache (will fetch new data which fails in test)
        with patch.object(self.analyzer, '_fetch_finnhub_news', return_value=[]):
            with patch.object(self.analyzer, '_fetch_yfinance_news', return_value=[]):
                result = self.analyzer.get_sentiment("TEST", use_cache=True)
    
    def test_clear_cache_specific(self):
        """Test clearing specific symbol from cache."""
        self.analyzer.cache["TEST1"] = (Mock(), datetime.now())
        self.analyzer.cache["TEST2"] = (Mock(), datetime.now())
        
        self.analyzer.clear_cache("TEST1")
        
        self.assertNotIn("TEST1", self.analyzer.cache)
        self.assertIn("TEST2", self.analyzer.cache)
    
    def test_clear_cache_all(self):
        """Test clearing entire cache."""
        self.analyzer.cache["TEST1"] = (Mock(), datetime.now())
        self.analyzer.cache["TEST2"] = (Mock(), datetime.now())
        
        self.analyzer.clear_cache()
        
        self.assertEqual(len(self.analyzer.cache), 0)


class TestSentimentConfig(unittest.TestCase):
    """Test SentimentConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = SentimentConfig()
        
        self.assertEqual(config.lookback_hours, 48)
        self.assertEqual(config.time_decay_halflife_hours, 24.0)
        self.assertEqual(config.cache_ttl_seconds, 900)
        self.assertEqual(config.min_articles, 3)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SentimentConfig(
            finnhub_api_key="test_key",
            lookback_hours=72,
            cache_ttl_seconds=600,
            min_articles=5
        )
        
        self.assertEqual(config.finnhub_api_key, "test_key")
        self.assertEqual(config.lookback_hours, 72)
        self.assertEqual(config.cache_ttl_seconds, 600)
        self.assertEqual(config.min_articles, 5)


if __name__ == '__main__':
    unittest.main(verbosity=2)

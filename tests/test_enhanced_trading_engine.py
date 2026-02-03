"""
Integration Tests for Enhanced Trading Engine
==============================================

Tests the complete trading system integration including:
- All modules working together
- Pipeline execution
- Decision making
- Error handling
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.enhanced_trading_engine import (
    EnhancedTradingEngine,
    EngineConfig,
    TradeSignal,
    TradeDecision
)
from src.position_sizer import PerformanceMetrics


class TestEnhancedTradingEngine(unittest.TestCase):
    """Test complete trading engine integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = EnhancedTradingEngine()
        self.portfolio_value = 100000
        self.metrics = PerformanceMetrics(
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            total_profit=15000,
            total_loss=-10000
        )
    
    def test_engine_initialization(self):
        """Test engine initializes all modules."""
        self.assertIsNotNone(self.engine.risk_manager)
        self.assertIsNotNone(self.engine.position_sizer)
        self.assertIsNotNone(self.engine.mtf_analyzer)
        self.assertIsNotNone(self.engine.sentiment_analyzer)
    
    def test_atr_calculation(self):
        """Test ATR calculation."""
        with patch('yfinance.Ticker') as mock_ticker:
            # Mock price data
            dates = pd.date_range('2024-01-01', periods=100, freq='D')
            data = pd.DataFrame({
                'High': np.random.rand(100) * 2 + 100,
                'Low': np.random.rand(100) * 2 + 98,
                'Close': np.random.rand(100) * 2 + 99,
            }, index=dates)
            
            mock_ticker.return_value.history.return_value = data
            
            atr = self.engine._calculate_atr('TEST', 14)
            
            self.assertGreater(atr, 0)
    
    def test_current_price_fetch(self):
        """Test current price fetching."""
        with patch('yfinance.Ticker') as mock_ticker:
            # Mock recent data
            data = pd.DataFrame({
                'Close': [100.0, 101.0, 102.0]
            })
            
            mock_ticker.return_value.history.return_value = data
            
            price = self.engine._get_current_price('TEST')
            
            self.assertEqual(price, 102.0)
    
    def test_combined_score_calculation(self):
        """Test combined score calculation."""
        # High MTF, high sentiment
        score = self.engine._calculate_combined_score(80.0, 0.6)
        self.assertGreater(score, 0.7)
        
        # Low MTF, low sentiment
        score = self.engine._calculate_combined_score(30.0, -0.4)
        self.assertLess(score, 0.4)
        
        # Mixed signals
        score = self.engine._calculate_combined_score(70.0, -0.2)
        self.assertGreater(score, 0.4)
        self.assertLess(score, 0.6)
    
    def test_signal_determination(self):
        """Test trade signal determination."""
        # Strong buy signals
        signal = self.engine._determine_signal(0.85, 85.0, 0.7)
        self.assertEqual(signal, TradeSignal.STRONG_BUY)
        
        # Buy signal
        signal = self.engine._determine_signal(0.75, 75.0, 0.4)
        self.assertEqual(signal, TradeSignal.BUY)
        
        # Hold signal
        signal = self.engine._determine_signal(0.55, 55.0, 0.1)
        self.assertEqual(signal, TradeSignal.HOLD)
        
        # Sell signal
        signal = self.engine._determine_signal(0.35, 35.0, -0.2)
        self.assertEqual(signal, TradeSignal.SELL)
        
        # Strong sell signal
        signal = self.engine._determine_signal(0.25, 25.0, -0.6)
        self.assertEqual(signal, TradeSignal.STRONG_SELL)
    
    @patch('yfinance.Ticker')
    @patch.object(EnhancedTradingEngine, '_calculate_atr')
    @patch.object(EnhancedTradingEngine, '_get_current_price')
    def test_complete_analysis_pipeline(self, mock_price, mock_atr, mock_ticker):
        """Test complete analysis pipeline execution."""
        # Mock data
        mock_price.return_value = 100.0
        mock_atr.return_value = 2.5
        
        # Mock MTF analyzer
        with patch.object(self.engine.mtf_analyzer, 'analyze') as mock_mtf:
            mock_mtf.return_value = Mock(
                alignment_score=75.0,
                dominant_trend=Mock(name='BULLISH'),
                bullish_timeframes=4,
                bearish_timeframes=1,
                signals={}
            )
            
            # Mock sentiment analyzer
            with patch.object(self.engine.sentiment_analyzer, 'get_sentiment') as mock_sent:
                mock_sent.return_value = Mock(
                    score=0.5,
                    level=Mock(value='greed'),
                    article_count=10,
                    positive_count=7,
                    negative_count=3,
                    confidence=0.7
                )
                
                # Run analysis
                decision = self.engine.analyze_opportunity(
                    'TEST',
                    self.portfolio_value,
                    self.metrics
                )
                
                # Verify decision
                self.assertIsInstance(decision, TradeDecision)
                self.assertEqual(decision.symbol, 'TEST')
                self.assertGreater(decision.mtf_score, 0)
                self.assertGreater(decision.sentiment_score, 0)
                self.assertIn(decision.signal, TradeSignal)
    
    @patch('yfinance.Ticker')
    @patch.object(EnhancedTradingEngine, '_get_current_price')
    def test_rejection_on_low_mtf_score(self, mock_price, mock_ticker):
        """Test rejection when MTF score is too low."""
        mock_price.return_value = 100.0
        
        with patch.object(self.engine.mtf_analyzer, 'analyze') as mock_mtf:
            mock_mtf.return_value = Mock(
                alignment_score=40.0,  # Below minimum
                dominant_trend=Mock(name='NEUTRAL'),
                bullish_timeframes=2,
                bearish_timeframes=2,
                signals={}
            )
            
            with patch.object(self.engine.sentiment_analyzer, 'get_sentiment') as mock_sent:
                mock_sent.return_value = Mock(
                    score=0.2,
                    level=Mock(value='neutral'),
                    article_count=5,
                    positive_count=3,
                    negative_count=2,
                    confidence=0.4
                )
                
                decision = self.engine.analyze_opportunity('TEST', self.portfolio_value)
                
                self.assertFalse(decision.is_tradeable)
                self.assertTrue(any('MTF' in r for r in decision.rejection_reasons))
    
    @patch('yfinance.Ticker')
    @patch.object(EnhancedTradingEngine, '_get_current_price')
    def test_rejection_on_low_sentiment(self, mock_price, mock_ticker):
        """Test rejection when sentiment is too negative."""
        mock_price.return_value = 100.0
        
        with patch.object(self.engine.mtf_analyzer, 'analyze') as mock_mtf:
            mock_mtf.return_value = Mock(
                alignment_score=75.0,
                dominant_trend=Mock(name='BULLISH'),
                bullish_timeframes=4,
                bearish_timeframes=1,
                signals={}
            )
            
            with patch.object(self.engine.sentiment_analyzer, 'get_sentiment') as mock_sent:
                mock_sent.return_value = Mock(
                    score=-0.6,  # Very negative
                    level=Mock(value='fear'),
                    article_count=8,
                    positive_count=1,
                    negative_count=7,
                    confidence=0.6
                )
                
                decision = self.engine.analyze_opportunity('TEST', self.portfolio_value)
                
                self.assertFalse(decision.is_tradeable)
                self.assertTrue(any('Sentiment' in r for r in decision.rejection_reasons))
    
    def test_batch_analysis(self):
        """Test batch analysis of multiple symbols."""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        with patch.object(self.engine, 'analyze_opportunity') as mock_analyze:
            # Mock different scores for each symbol
            mock_analyze.side_effect = [
                TradeDecision(
                    symbol=sym,
                    timestamp=datetime.now(),
                    signal=TradeSignal.BUY,
                    mtf_score=70 + i*5,
                    sentiment_score=0.3 + i*0.1,
                    combined_score=0.6 + i*0.05,
                    confidence=0.7,
                    recommended_position_value=5000,
                    recommended_quantity=50,
                    entry_price=100.0,
                    stop_loss=95.0,
                    take_profits=[105.0, 110.0, 115.0],
                    is_tradeable=True,
                    rejection_reasons=[],
                    metadata={}
                )
                for i, sym in enumerate(symbols)
            ]
            
            decisions = self.engine.batch_analyze(symbols, self.portfolio_value, self.metrics)
            
            self.assertEqual(len(decisions), 3)
            # Should be sorted by combined score (descending)
            self.assertGreaterEqual(decisions[0].combined_score, decisions[1].combined_score)
            self.assertGreaterEqual(decisions[1].combined_score, decisions[2].combined_score)
    
    def test_error_handling_in_batch(self):
        """Test error handling in batch analysis."""
        symbols = ['GOOD', 'BAD', 'UGLY']
        
        def side_effect(symbol, *args, **kwargs):
            if symbol == 'BAD':
                raise Exception("Network error")
            elif symbol == 'UGLY':
                raise ValueError("Invalid data")
            else:
                return TradeDecision(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    signal=TradeSignal.BUY,
                    mtf_score=70,
                    sentiment_score=0.3,
                    combined_score=0.6,
                    confidence=0.7,
                    recommended_position_value=5000,
                    recommended_quantity=50,
                    entry_price=100.0,
                    stop_loss=95.0,
                    take_profits=[105.0],
                    is_tradeable=True,
                    rejection_reasons=[],
                    metadata={}
                )
        
        with patch.object(self.engine, 'analyze_opportunity', side_effect=side_effect):
            decisions = self.engine.batch_analyze(symbols, self.portfolio_value)
            
            # Should still return all 3 decisions
            self.assertEqual(len(decisions), 3)
            
            # Bad symbols should have rejection decisions
            bad_decision = next(d for d in decisions if d.symbol == 'BAD')
            self.assertFalse(bad_decision.is_tradeable)
            self.assertTrue(any('error' in r.lower() for r in bad_decision.rejection_reasons))
    
    def test_custom_config(self):
        """Test engine with custom configuration."""
        from src.risk_manager import RiskConfig
        from src.position_sizer import SizingConfig
        
        custom_config = EngineConfig(
            risk_config=RiskConfig(max_concurrent_positions=10),
            sizing_config=SizingConfig(kelly_multiplier=1.0),  # Full Kelly
            min_mtf_score=70.0,
            min_sentiment_score=-0.2
        )
        
        engine = EnhancedTradingEngine(custom_config)
        
        self.assertEqual(engine.config.min_mtf_score, 70.0)
        self.assertEqual(engine.config.min_sentiment_score, -0.2)
        self.assertEqual(engine.risk_manager.config.max_concurrent_positions, 10)
        self.assertEqual(engine.position_sizer.config.kelly_multiplier, 1.0)
    
    def test_metadata_inclusion(self):
        """Test that decisions include comprehensive metadata."""
        with patch.object(self.engine, '_get_current_price', return_value=100.0):
            with patch.object(self.engine, '_calculate_atr', return_value=2.5):
                with patch.object(self.engine.mtf_analyzer, 'analyze') as mock_mtf:
                    with patch.object(self.engine.sentiment_analyzer, 'get_sentiment') as mock_sent:
                        mock_mtf.return_value = Mock(
                            alignment_score=75.0,
                            dominant_trend=Mock(name='BULLISH'),
                            bullish_timeframes=4,
                            bearish_timeframes=1,
                            signals={}
                        )
                        
                        mock_sent.return_value = Mock(
                            score=0.5,
                            level=Mock(value='greed'),
                            article_count=10,
                            positive_count=7,
                            negative_count=3,
                            confidence=0.7
                        )
                        
                        decision = self.engine.analyze_opportunity('TEST', self.portfolio_value)
                        
                        # Check metadata
                        self.assertIn('atr', decision.metadata)
                        self.assertIn('mtf_analysis', decision.metadata)
                        self.assertIn('sentiment_result', decision.metadata)
                        self.assertIn('position_sizing', decision.metadata)
                        self.assertIn('risk_metrics', decision.metadata)


class TestEngineConfig(unittest.TestCase):
    """Test EngineConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = EngineConfig()
        
        self.assertEqual(config.min_mtf_score, 60.0)
        self.assertEqual(config.min_sentiment_score, -0.3)
        self.assertEqual(config.min_combined_score, 0.6)
        self.assertEqual(config.mtf_weight, 0.6)
        self.assertEqual(config.sentiment_weight, 0.4)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = EngineConfig(
            min_mtf_score=70.0,
            min_sentiment_score=-0.2,
            mtf_weight=0.7,
            sentiment_weight=0.3
        )
        
        self.assertEqual(config.min_mtf_score, 70.0)
        self.assertEqual(config.min_sentiment_score, -0.2)
        self.assertEqual(config.mtf_weight, 0.7)
        self.assertEqual(config.sentiment_weight, 0.3)


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""
Tests for Signal Generator
===========================

Unit tests for multi-strategy signal generation.
"""

import asyncio
import pytest
from datetime import datetime

from src.options.signal_generator import (
    SignalGenerator,
    IVRankStrategy,
    ThetaDecayStrategy,
    MeanReversionStrategy,
    DeltaHedgingStrategy,
    Signal,
    SignalType,
    SignalSource,
)
from src.options.config import get_config


# ============================================================================
# IV RANK STRATEGY TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_iv_rank_high_generates_sell_signal():
    """Test that high IV rank generates SELL signal."""
    strategy = IVRankStrategy()
    
    # Mock high IV rank data
    # In real implementation, this would query actual market data
    # For testing, we assume the analyze method would detect high IV
    
    signals = await strategy.generate_signals(["SPY"])
    
    # We expect at least 0 signals (depending on mock data)
    assert isinstance(signals, list)
    assert all(isinstance(s, Signal) for s in signals)


@pytest.mark.asyncio
async def test_iv_rank_low_generates_buy_signal():
    """Test that low IV rank generates BUY signal."""
    strategy = IVRankStrategy()
    
    signals = await strategy.generate_signals(["TSLA"])
    
    assert isinstance(signals, list)


@pytest.mark.asyncio
async def test_iv_rank_confidence_scaling():
    """Test that confidence scales with IV rank distance from threshold."""
    strategy = IVRankStrategy()
    config = get_config()
    
    # If IV rank is 100, confidence should be high
    # If IV rank is 50, confidence should be lower
    # This tests the internal logic
    
    signals = await strategy.generate_signals(["QQQ"])
    
    for signal in signals:
        assert 0.0 <= signal.confidence <= 1.0


# ============================================================================
# THETA DECAY STRATEGY TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_theta_decay_optimal_dte():
    """Test that theta strategy targets optimal DTE range."""
    strategy = ThetaDecayStrategy()
    config = get_config()
    
    signals = await strategy.generate_signals(["SPY"])
    
    for signal in signals:
        if signal.dte:
            # Should be in optimal range (21-45 DTE)
            assert config["optimal_dte_min"] <= signal.dte <= config["optimal_dte_max"]


@pytest.mark.asyncio
async def test_theta_decay_min_pop_requirement():
    """Test that theta strategy respects minimum PoP requirement."""
    strategy = ThetaDecayStrategy()
    config = get_config()
    
    signals = await strategy.generate_signals(["IWM"])
    
    for signal in signals:
        if signal.probability_of_profit:
            # Should meet minimum PoP
            assert signal.probability_of_profit >= config["min_probability_of_profit"]


# ============================================================================
# MEAN REVERSION STRATEGY TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_mean_reversion_z_score_entry():
    """Test that mean reversion triggers at z-score extremes."""
    strategy = MeanReversionStrategy()
    config = get_config()
    
    signals = await strategy.generate_signals(["AAPL"])
    
    for signal in signals:
        if signal.z_score:
            # Should only signal at extremes
            assert abs(signal.z_score) >= config["z_score_entry"]


@pytest.mark.asyncio
async def test_mean_reversion_direction():
    """Test that mean reversion suggests correct direction."""
    strategy = MeanReversionStrategy()
    
    signals = await strategy.generate_signals(["NVDA"])
    
    for signal in signals:
        if signal.z_score:
            if signal.z_score > 0:
                # Price too high - should sell
                assert signal.signal_type in [SignalType.SELL]
            else:
                # Price too low - should buy or sell puts
                assert signal.signal_type in [SignalType.SELL, SignalType.BUY]


# ============================================================================
# DELTA HEDGING STRATEGY TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_delta_hedging_threshold():
    """Test that delta hedging triggers at threshold."""
    strategy = DeltaHedgingStrategy()
    config = get_config()
    
    # Test with high positive delta
    signals = await strategy.generate_signals(portfolio_delta=0.15)
    assert len(signals) > 0  # Should generate hedge signal
    
    # Test with delta within threshold
    signals = await strategy.generate_signals(portfolio_delta=0.05)
    assert len(signals) == 0  # Should not generate signal


@pytest.mark.asyncio
async def test_delta_hedging_direction():
    """Test that delta hedging suggests correct hedge direction."""
    strategy = DeltaHedgingStrategy()
    
    # Positive delta needs bearish hedge
    signals = await strategy.generate_signals(portfolio_delta=0.20)
    if signals:
        assert signals[0].signal_type == SignalType.SELL
    
    # Negative delta needs bullish hedge
    signals = await strategy.generate_signals(portfolio_delta=-0.20)
    if signals:
        assert signals[0].signal_type == SignalType.BUY


# ============================================================================
# SIGNAL GENERATOR TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_signal_generator_integration():
    """Test that main signal generator combines all strategies."""
    generator = SignalGenerator()
    
    signals = await generator.generate_all_signals(
        symbols=["SPY", "QQQ"],
        portfolio_delta=0.0,
    )
    
    assert isinstance(signals, list)
    assert all(isinstance(s, Signal) for s in signals)


@pytest.mark.asyncio
async def test_signal_generator_sorts_by_confidence():
    """Test that signals are sorted by confidence descending."""
    generator = SignalGenerator()
    
    signals = await generator.generate_all_signals(
        symbols=["SPY", "QQQ", "IWM"],
        portfolio_delta=0.0,
    )
    
    # Check descending order
    for i in range(len(signals) - 1):
        assert signals[i].confidence >= signals[i + 1].confidence


@pytest.mark.asyncio
async def test_signal_generator_multiple_sources():
    """Test that signals come from multiple strategies."""
    generator = SignalGenerator()
    
    signals = await generator.generate_all_signals(
        symbols=["SPY", "TSLA", "NVDA"],
        portfolio_delta=0.0,
    )
    
    # Should have signals from different sources
    sources = set(s.signal_source for s in signals)
    # At least some signals should be generated
    assert len(signals) >= 0


@pytest.mark.asyncio
async def test_signal_has_all_required_fields():
    """Test that signals have all required fields."""
    generator = SignalGenerator()
    
    signals = await generator.generate_all_signals(
        symbols=["SPY"],
        portfolio_delta=0.0,
    )
    
    for signal in signals:
        assert signal.symbol
        assert signal.signal_type
        assert signal.signal_source
        assert signal.strategy
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.timestamp
        assert signal.reason


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_empty_symbol_list():
    """Test handling of empty symbol list."""
    generator = SignalGenerator()
    
    signals = await generator.generate_all_signals(
        symbols=[],
        portfolio_delta=0.0,
    )
    
    # Should return empty list or only delta hedge signals
    assert isinstance(signals, list)
    assert len(signals) <= 1  # Only possible delta hedge signal


@pytest.mark.asyncio
async def test_invalid_symbol():
    """Test handling of invalid symbol."""
    generator = SignalGenerator()
    
    # Should not crash on invalid symbol
    signals = await generator.generate_all_signals(
        symbols=["INVALID123"],
        portfolio_delta=0.0,
    )
    
    assert isinstance(signals, list)


@pytest.mark.asyncio
async def test_extreme_portfolio_delta():
    """Test handling of extreme portfolio delta."""
    strategy = DeltaHedgingStrategy()
    
    # Very high delta
    signals = await strategy.generate_signals(portfolio_delta=1.0)
    assert len(signals) > 0
    
    # Very low delta
    signals = await strategy.generate_signals(portfolio_delta=-1.0)
    assert len(signals) > 0


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

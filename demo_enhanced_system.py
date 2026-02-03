#!/usr/bin/env python3
"""
Enhanced Trading System - Quick Demo
=====================================

Demonstrates the complete production trading system with all modules.

Usage:
    python demo_enhanced_system.py [SYMBOL] [--portfolio-value VALUE]

Example:
    python demo_enhanced_system.py AMD --portfolio-value 100000
"""

import argparse
import logging
from datetime import datetime

from src.enhanced_trading_engine import EnhancedTradingEngine, EngineConfig
from src.position_sizer import PerformanceMetrics
from src.risk_manager import RiskConfig
from src.position_sizer import SizingConfig
from src.multi_timeframe_analyzer import AnalyzerConfig
from src.sentiment_analyzer import SentimentConfig


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def print_header(text: str, width: int = 80):
    """Print formatted header."""
    print(f"\n{'='*width}")
    print(f"{text:^{width}}")
    print(f"{'='*width}\n")


def print_section(title: str):
    """Print section title."""
    print(f"\n{title}")
    print("-" * len(title))


def display_decision(decision):
    """Display trading decision in formatted output."""
    
    print_header(f"TRADING DECISION: {decision.symbol}")
    
    # Signal and Status
    print(f"Signal:     {decision.signal.value.upper()}")
    print(f"Tradeable:  {'✓ YES' if decision.is_tradeable else '✗ NO'}")
    print(f"Timestamp:  {decision.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Scores
    print_section("Analysis Scores")
    print(f"  Multi-Timeframe Alignment: {decision.mtf_score:6.1f}/100")
    print(f"  Market Sentiment:          {decision.sentiment_score:+6.2f} (-1 to +1)")
    print(f"  Combined Score:            {decision.combined_score:6.2f} (0 to 1)")
    print(f"  Overall Confidence:        {decision.confidence:6.1%}")
    
    # Position Details
    print_section("Position Recommendation")
    print(f"  Recommended Value:     ${decision.recommended_position_value:>12,.2f}")
    print(f"  Recommended Quantity:  {decision.recommended_quantity:>12,} shares")
    print(f"  Entry Price:           ${decision.entry_price:>12,.2f}")
    print(f"  Stop Loss:             ${decision.stop_loss:>12,.2f} "
          f"({((decision.stop_loss/decision.entry_price-1)*100):+.1f}%)")
    
    # Take Profit Levels
    if decision.take_profits:
        print(f"\n  Take Profit Levels:")
        for i, tp in enumerate(decision.take_profits, 1):
            pct = ((tp/decision.entry_price-1)*100)
            print(f"    TP{i}: ${tp:>12,.2f} ({pct:+.1f}%)")
    
    # Multi-Timeframe Details
    if 'mtf_analysis' in decision.metadata:
        mtf = decision.metadata['mtf_analysis']
        print_section("Multi-Timeframe Analysis")
        print(f"  Dominant Trend:        {mtf.dominant_trend.name}")
        print(f"  Bullish Timeframes:    {mtf.bullish_timeframes}")
        print(f"  Bearish Timeframes:    {mtf.bearish_timeframes}")
        print(f"  Neutral Timeframes:    {mtf.neutral_timeframes}")
        
        if mtf.signals:
            print(f"\n  Per-Timeframe Signals:")
            for tf, signals in mtf.signals.items():
                bias = "🟢 BULL" if signals.is_bullish else "🔴 BEAR" if signals.is_bearish else "⚪ NEUT"
                print(f"    {tf.description:12s}: {bias}  Score={signals.trend_score:+.2f}")
    
    # Sentiment Details
    if 'sentiment_result' in decision.metadata:
        sent = decision.metadata['sentiment_result']
        print_section("Sentiment Analysis")
        print(f"  Sentiment Level:       {sent.level.value.upper()}")
        print(f"  Article Count:         {sent.article_count}")
        print(f"  Positive Articles:     {sent.positive_count}")
        print(f"  Negative Articles:     {sent.negative_count}")
        print(f"  Neutral Articles:      {sent.neutral_count}")
        print(f"  Data Source:           {sent.data_source}")
        
        if sent.articles:
            print(f"\n  Recent Headlines:")
            for i, article in enumerate(sent.articles[:3], 1):
                score = article.sentiment_score or 0.0
                age = article.age_hours
                sentiment_emoji = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "⚪"
                print(f"    {i}. {sentiment_emoji} [{score:+.2f}] ({age:.1f}h) {article.headline[:60]}...")
    
    # Position Sizing Details
    if 'position_sizing' in decision.metadata:
        pos = decision.metadata['position_sizing']
        print_section("Position Sizing Details")
        if pos.is_valid:
            print(f"  Kelly Fraction:        {pos.kelly_fraction:.2%}")
            print(f"  Confidence Adjusted:   {pos.confidence_adjusted:.2%}")
            print(f"  Volatility Adjusted:   {pos.volatility_adjusted:.2%}")
            print(f"  Final Size:            {pos.final_size:.2%}")
            
            if pos.sizing_factors:
                print(f"\n  Sizing Factors:")
                print(f"    Win Rate:          {pos.sizing_factors.get('win_rate', 0):.1%}")
                print(f"    Payoff Ratio:      {pos.sizing_factors.get('payoff_ratio', 0):.2f}")
                print(f"    Expectancy:        ${pos.sizing_factors.get('expectancy', 0):.2f}")
    
    # Risk Metrics
    if 'risk_metrics' in decision.metadata:
        risk = decision.metadata['risk_metrics']
        print_section("Portfolio Risk Metrics")
        print(f"  Current Drawdown:      {risk.current_drawdown:.2%}")
        print(f"  Daily Drawdown:        {risk.daily_drawdown:.2%}")
        print(f"  Open Positions:        {risk.open_positions}")
        print(f"  Total Exposure:        ${risk.total_exposure:,.2f}")
        print(f"  Largest Position:      {risk.largest_position_pct:.1%}")
        print(f"  Risk Limit Reached:    {'YES' if risk.risk_limit_reached else 'NO'}")
    
    # Rejection Reasons
    if decision.rejection_reasons:
        print_section("⚠️  Rejection Reasons")
        for reason in decision.rejection_reasons:
            print(f"  • {reason}")
    
    # Summary
    print_section("Summary")
    if decision.is_tradeable:
        print("✓ This is a TRADEABLE opportunity")
        print(f"  Recommended Action: {decision.signal.value.upper()}")
        print(f"  Position Size: ${decision.recommended_position_value:,.2f} ({decision.recommended_quantity} shares)")
    else:
        print("✗ This opportunity is NOT tradeable")
        print(f"  Reason: See rejection reasons above")
    
    print(f"\n{'='*80}\n")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Enhanced Trading System Demo')
    parser.add_argument('symbol', nargs='?', default='AMD', help='Stock symbol to analyze')
    parser.add_argument('--portfolio-value', type=float, default=100000,
                       help='Portfolio value for position sizing')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--batch', nargs='+', help='Analyze multiple symbols')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    print_header("ENHANCED TRADING SYSTEM - PRODUCTION DEMO", 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Portfolio Value: ${args.portfolio_value:,.2f}")
    
    # Initialize engine
    print("\nInitializing trading engine...")
    engine = EnhancedTradingEngine()
    
    # Sample performance metrics (would come from actual trading history)
    metrics = PerformanceMetrics(
        total_trades=100,
        winning_trades=58,
        losing_trades=42,
        total_profit=14500.00,
        total_loss=-9200.00
    )
    
    print(f"✓ Engine initialized with {metrics.total_trades} historical trades")
    print(f"  Win Rate: {metrics.win_rate:.1%}")
    print(f"  Payoff Ratio: {metrics.payoff_ratio:.2f}")
    print(f"  Expectancy: ${metrics.expectancy:.2f}")
    
    # Analyze symbol(s)
    if args.batch:
        # Batch analysis
        print(f"\nAnalyzing {len(args.batch)} symbols...")
        decisions = engine.batch_analyze(args.batch, args.portfolio_value, metrics)
        
        print(f"\n{'='*80}")
        print(f"BATCH ANALYSIS RESULTS")
        print(f"{'='*80}\n")
        
        tradeable = [d for d in decisions if d.is_tradeable]
        print(f"Tradeable Opportunities: {len(tradeable)}/{len(decisions)}\n")
        
        # Display summary table
        print(f"{'Symbol':<8} {'Signal':<12} {'MTF':<6} {'Sent':<6} {'Comb':<6} {'Size':<12} {'Status'}")
        print("-" * 80)
        for d in decisions:
            status = "✓ TRADE" if d.is_tradeable else "✗ SKIP"
            print(f"{d.symbol:<8} {d.signal.value:<12} {d.mtf_score:>5.1f} "
                  f"{d.sentiment_score:>+5.2f} {d.combined_score:>5.2f} "
                  f"${d.recommended_position_value:>10,.0f} {status}")
        
        # Display best opportunity
        if tradeable:
            best = tradeable[0]
            print(f"\n{'='*80}")
            print(f"BEST OPPORTUNITY: {best.symbol}")
            print(f"{'='*80}")
            display_decision(best)
    else:
        # Single symbol analysis
        print(f"\nAnalyzing {args.symbol}...")
        decision = engine.analyze_opportunity(args.symbol, args.portfolio_value, metrics)
        display_decision(decision)


if __name__ == "__main__":
    main()

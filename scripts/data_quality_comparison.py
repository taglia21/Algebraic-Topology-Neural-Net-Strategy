"""Data Quality Comparison: Polygon vs yfinance.

Compares data quality between Polygon and yfinance providers:
- Data completeness (missing bars)
- Price accuracy
- Volume consistency
- Gap detection
- OHLCV integrity

Generates comprehensive report for validation.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_provider import get_ohlcv_data, validate_ohlcv_data

# Configuration
TICKERS = ["SPY", "QQQ", "IWM", "XLF", "XLK"]
START_DATE = "2020-01-01"
END_DATE = "2025-01-15"  # Current date
RESULTS_DIR = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results'


def fetch_data_both_providers(ticker: str, start: str, end: str) -> dict:
    """Fetch data from both providers for comparison."""
    result = {
        'ticker': ticker,
        'polygon': None,
        'yfinance': None,
        'polygon_error': None,
        'yfinance_error': None,
    }
    
    # Fetch from Polygon
    print(f"  Fetching {ticker} from Polygon...")
    try:
        result['polygon'] = get_ohlcv_data(
            ticker, start, end,
            timeframe="1d",
            provider="polygon"
        )
        print(f"    ✓ Polygon: {len(result['polygon'])} bars")
    except Exception as e:
        result['polygon_error'] = str(e)
        print(f"    ✗ Polygon error: {e}")
    
    # Fetch from yfinance
    print(f"  Fetching {ticker} from yfinance...")
    try:
        result['yfinance'] = get_ohlcv_data(
            ticker, start, end,
            timeframe="1d",
            provider="yfinance",
            use_subprocess=True
        )
        print(f"    ✓ yfinance: {len(result['yfinance'])} bars")
    except Exception as e:
        result['yfinance_error'] = str(e)
        print(f"    ✗ yfinance error: {e}")
    
    return result


def compare_data_quality(polygon_df: pd.DataFrame, yfinance_df: pd.DataFrame, ticker: str) -> dict:
    """Compare data quality between two DataFrames."""
    comparison = {
        'ticker': ticker,
        'polygon_bars': 0,
        'yfinance_bars': 0,
        'common_dates': 0,
        'polygon_only': 0,
        'yfinance_only': 0,
        'price_diff_stats': {},
        'volume_diff_stats': {},
        'ohlcv_integrity': {},
        'data_gaps': {},
    }
    
    if polygon_df is None or polygon_df.empty:
        comparison['polygon_error'] = 'No data'
        return comparison
    
    if yfinance_df is None or yfinance_df.empty:
        comparison['yfinance_error'] = 'No data'
        return comparison
    
    # Basic counts
    comparison['polygon_bars'] = len(polygon_df)
    comparison['yfinance_bars'] = len(yfinance_df)
    
    # Normalize indices (just dates, no time)
    poly_dates = set(polygon_df.index.date)
    yf_dates = set(yfinance_df.index.date)
    
    common_dates = poly_dates & yf_dates
    comparison['common_dates'] = len(common_dates)
    comparison['polygon_only'] = len(poly_dates - yf_dates)
    comparison['yfinance_only'] = len(yf_dates - poly_dates)
    
    # Compare prices on common dates
    if common_dates:
        poly_norm = polygon_df.copy()
        poly_norm.index = poly_norm.index.date
        yf_norm = yfinance_df.copy()
        yf_norm.index = yf_norm.index.date
        
        # Select common dates
        common_list = sorted(list(common_dates))
        poly_common = poly_norm.loc[common_list]
        yf_common = yf_norm.loc[common_list]
        
        # Price difference statistics (close price)
        close_diff = (poly_common['close'] - yf_common['close']).abs()
        close_pct_diff = (close_diff / yf_common['close'] * 100)
        
        comparison['price_diff_stats'] = {
            'mean_abs_diff': float(close_diff.mean()),
            'max_abs_diff': float(close_diff.max()),
            'mean_pct_diff': float(close_pct_diff.mean()),
            'max_pct_diff': float(close_pct_diff.max()),
            'bars_matching_exactly': int((close_diff < 0.01).sum()),
        }
        
        # Volume comparison
        vol_diff = (poly_common['volume'] - yf_common['volume']).abs()
        vol_pct_diff = (vol_diff / yf_common['volume'].replace(0, 1) * 100)
        
        comparison['volume_diff_stats'] = {
            'mean_abs_diff': float(vol_diff.mean()),
            'max_abs_diff': float(vol_diff.max()),
            'mean_pct_diff': float(vol_pct_diff.mean()),
        }
    
    # OHLCV integrity checks
    comparison['ohlcv_integrity'] = {
        'polygon': check_ohlcv_integrity(polygon_df),
        'yfinance': check_ohlcv_integrity(yfinance_df),
    }
    
    # Gap detection
    comparison['data_gaps'] = {
        'polygon': detect_gaps(polygon_df),
        'yfinance': detect_gaps(yfinance_df),
    }
    
    return comparison


def check_ohlcv_integrity(df: pd.DataFrame) -> dict:
    """Check OHLCV data integrity."""
    if df is None or df.empty:
        return {'error': 'No data'}
    
    high_ge_low = (df['high'] >= df['low']).all()
    high_ge_open = (df['high'] >= df['open']).all()
    high_ge_close = (df['high'] >= df['close']).all()
    low_le_open = (df['low'] <= df['open']).all()
    low_le_close = (df['low'] <= df['close']).all()
    volume_non_neg = (df['volume'] >= 0).all()
    
    zero_vol_count = (df['volume'] == 0).sum()
    zero_vol_pct = zero_vol_count / len(df) * 100
    
    return {
        'high_ge_low': bool(high_ge_low),
        'high_ge_open': bool(high_ge_open),
        'high_ge_close': bool(high_ge_close),
        'low_le_open': bool(low_le_open),
        'low_le_close': bool(low_le_close),
        'volume_non_neg': bool(volume_non_neg),
        'zero_volume_bars': int(zero_vol_count),
        'zero_volume_pct': float(zero_vol_pct),
        'all_checks_pass': all([
            high_ge_low, high_ge_open, high_ge_close,
            low_le_open, low_le_close, volume_non_neg
        ])
    }


def detect_gaps(df: pd.DataFrame) -> dict:
    """Detect data gaps in time series."""
    if df is None or df.empty or len(df) < 2:
        return {'error': 'Insufficient data'}
    
    df_sorted = df.sort_index()
    date_diffs = df_sorted.index.to_series().diff().dropna()
    
    # Convert to days
    if hasattr(date_diffs.iloc[0], 'days'):
        day_gaps = date_diffs.apply(lambda x: x.days)
    else:
        day_gaps = pd.to_timedelta(date_diffs).dt.days
    
    # Count gaps (more than 3 days = significant gap, accounting for weekends)
    small_gaps = ((day_gaps > 3) & (day_gaps <= 5)).sum()  # 3-4 day weekends
    medium_gaps = ((day_gaps > 5) & (day_gaps <= 10)).sum()  # ~1 week
    large_gaps = (day_gaps > 10).sum()  # More than 2 weeks
    
    largest_gap = int(day_gaps.max()) if len(day_gaps) > 0 else 0
    
    return {
        'small_gaps_3_5_days': int(small_gaps),
        'medium_gaps_5_10_days': int(medium_gaps),
        'large_gaps_10plus_days': int(large_gaps),
        'largest_gap_days': largest_gap,
        'total_gaps_over_3_days': int(small_gaps + medium_gaps + large_gaps),
    }


def generate_summary_report(comparisons: list) -> dict:
    """Generate summary report from all comparisons."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'date_range': f"{START_DATE} to {END_DATE}",
        'tickers_analyzed': len(comparisons),
        'total_polygon_bars': 0,
        'total_yfinance_bars': 0,
        'avg_price_diff_pct': 0,
        'data_quality_scores': {},
        'recommendations': [],
    }
    
    price_diffs = []
    
    for comp in comparisons:
        ticker = comp['ticker']
        summary['total_polygon_bars'] += comp.get('polygon_bars', 0)
        summary['total_yfinance_bars'] += comp.get('yfinance_bars', 0)
        
        if 'price_diff_stats' in comp and comp['price_diff_stats']:
            price_diffs.append(comp['price_diff_stats'].get('mean_pct_diff', 0))
        
        # Score each data source (0-100)
        poly_score = calculate_quality_score(comp, 'polygon')
        yf_score = calculate_quality_score(comp, 'yfinance')
        
        summary['data_quality_scores'][ticker] = {
            'polygon': poly_score,
            'yfinance': yf_score,
            'winner': 'polygon' if poly_score > yf_score else 'yfinance' if yf_score > poly_score else 'tie',
        }
    
    if price_diffs:
        summary['avg_price_diff_pct'] = float(np.mean(price_diffs))
    
    # Generate recommendations
    polygon_wins = sum(1 for s in summary['data_quality_scores'].values() if s['winner'] == 'polygon')
    
    if polygon_wins > len(comparisons) / 2:
        summary['recommendations'].append("Polygon provides higher quality data overall - recommend as primary source.")
    
    if summary['avg_price_diff_pct'] < 0.1:
        summary['recommendations'].append("Price differences are minimal (<0.1%) - both sources are reliable for backtesting.")
    
    return summary


def calculate_quality_score(comp: dict, provider: str) -> int:
    """Calculate quality score (0-100) for a provider."""
    score = 100
    
    # Deduct for missing data
    bars = comp.get(f'{provider}_bars', 0)
    if bars == 0:
        return 0
    
    # Deduct for integrity issues
    integrity = comp.get('ohlcv_integrity', {}).get(provider, {})
    if not integrity.get('all_checks_pass', True):
        score -= 20
    
    zero_vol_pct = integrity.get('zero_volume_pct', 0)
    if zero_vol_pct > 5:
        score -= 10
    if zero_vol_pct > 10:
        score -= 10
    
    # Deduct for gaps
    gaps = comp.get('data_gaps', {}).get(provider, {})
    large_gaps = gaps.get('large_gaps_10plus_days', 0)
    score -= large_gaps * 5
    
    return max(0, score)


def main():
    """Run data quality comparison."""
    print("=" * 70)
    print("DATA QUALITY COMPARISON: Polygon vs yfinance")
    print("=" * 70)
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Tickers: {', '.join(TICKERS)}")
    print("=" * 70)
    
    comparisons = []
    
    for ticker in TICKERS:
        print(f"\n[{ticker}]")
        
        # Fetch from both providers
        data = fetch_data_both_providers(ticker, START_DATE, END_DATE)
        
        # Compare quality
        comparison = compare_data_quality(
            data['polygon'],
            data['yfinance'],
            ticker
        )
        
        # Add any fetch errors
        if data['polygon_error']:
            comparison['polygon_fetch_error'] = data['polygon_error']
        if data['yfinance_error']:
            comparison['yfinance_fetch_error'] = data['yfinance_error']
        
        comparisons.append(comparison)
        
        # Print summary for this ticker
        print(f"  Polygon: {comparison['polygon_bars']} bars, yfinance: {comparison['yfinance_bars']} bars")
        print(f"  Common dates: {comparison['common_dates']}")
        
        if comparison['price_diff_stats']:
            print(f"  Avg price diff: {comparison['price_diff_stats']['mean_pct_diff']:.4f}%")
    
    # Generate summary
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    
    summary = generate_summary_report(comparisons)
    
    print(f"\nTotal bars - Polygon: {summary['total_polygon_bars']}, yfinance: {summary['total_yfinance_bars']}")
    print(f"Average price difference: {summary['avg_price_diff_pct']:.4f}%")
    
    print("\nData Quality Scores (0-100):")
    for ticker, scores in summary['data_quality_scores'].items():
        print(f"  {ticker}: Polygon={scores['polygon']}, yfinance={scores['yfinance']} → {scores['winner'].upper()}")
    
    print("\nRecommendations:")
    for rec in summary['recommendations']:
        print(f"  • {rec}")
    
    # Save detailed report
    full_report = {
        'summary': summary,
        'detailed_comparisons': comparisons,
    }
    
    report_path = f"{RESULTS_DIR}/data_quality_comparison.json"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: {report_path}")
    
    return full_report


if __name__ == "__main__":
    main()

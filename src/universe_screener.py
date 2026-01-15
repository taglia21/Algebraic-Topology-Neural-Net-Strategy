"""
Universe Screener - Phase 7 Multi-Stage Filtering.

Reduces Russell 3000 (~3000 stocks) to tradeable candidates (~200-300):
1. Liquidity & Data Quality filters
2. Momentum ranking  
3. Sector diversification
4. Optional TDA quality filter

Target: High-quality, liquid, momentum stocks with sector balance
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Sector mapping for major stocks (simplified)
# In production, would use yfinance .info['sector'] or GICS codes
SECTOR_MAP = {
    # Technology
    **{t: 'Technology' for t in [
        "AAPL", "MSFT", "GOOGL", "GOOG", "META", "NVDA", "AVGO", "ORCL", "CSCO", "CRM",
        "AMD", "ADBE", "ACN", "INTC", "IBM", "TXN", "QCOM", "NOW", "INTU", "AMAT",
        "ADI", "LRCX", "MU", "PANW", "SNPS", "KLAC", "CDNS", "MRVL", "ROP", "FTNT",
        "ADSK", "NXPI", "MCHP", "APH", "CTSH", "IT", "KEYS", "ANSS", "FSLR", "HPQ",
        "ANET", "CRWD", "DDOG", "ZS", "SNOW", "NET", "MDB", "TEAM", "OKTA", "ZM",
        "PLTR", "DELL", "HPE", "SMCI", "MPWR",
    ]},
    # Financials
    **{t: 'Financials' for t in [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "SPGI",
        "PNC", "USB", "TFC", "COF", "BK", "CME", "ICE", "MCO", "CB", "MMC",
        "AON", "MET", "PRU", "AIG", "TRV", "ALL", "AFL", "PGR", "AJG", "HIG",
        "CINF", "WRB", "NTRS", "STT", "MSCI", "NDAQ", "CBOE", "TROW", "BEN", "IVZ",
    ]},
    # Healthcare
    **{t: 'Healthcare' for t in [
        "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "GILD", "CVS", "ELV", "CI", "ISRG", "VRTX", "REGN", "MDT", "SYK",
        "BSX", "BDX", "EW", "ZBH", "IDXX", "IQV", "MTD", "A", "DXCM", "WST",
        "MRNA", "BIIB", "ILMN", "ZTS", "HCA", "CNC", "MOH", "HUM", "VEEV",
    ]},
    # Consumer Discretionary
    **{t: 'Consumer Discretionary' for t in [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG",
        "ORLY", "AZO", "ROST", "MAR", "HLT", "YUM", "DG", "DLTR", "ULTA", "BBY",
        "EBAY", "ETSY", "F", "GM", "APTV", "RCL", "CCL", "ABNB", "EXPE",
    ]},
    # Industrials
    **{t: 'Industrials' for t in [
        "UNP", "UPS", "HON", "RTX", "CAT", "DE", "BA", "LMT", "GE", "MMM",
        "FDX", "EMR", "ITW", "ETN", "NSC", "CSX", "PCAR", "PH", "CMI", "ROK",
        "WM", "RSG", "CTAS", "PAYX", "ADP", "GWW", "FAST", "TT", "CARR", "IR",
    ]},
    # Consumer Staples
    **{t: 'Consumer Staples' for t in [
        "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "EL", "MDLZ",
        "KMB", "GIS", "K", "CAG", "HSY", "MKC", "CHD", "CLX", "KR", "SYY",
    ]},
    # Energy
    **{t: 'Energy' for t in [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "PXD", "OXY",
        "HAL", "BKR", "DVN", "MRO", "WMB", "KMI", "OKE", "FANG", "HES", "APA",
    ]},
    # Utilities
    **{t: 'Utilities' for t in [
        "NEE", "DUK", "SO", "D", "SRE", "AEP", "EXC", "XEL", "WEC", "ED",
        "PEG", "ES", "EIX", "DTE", "FE", "AWK", "CEG", "NRG", "VST",
    ]},
    # Materials
    **{t: 'Materials' for t in [
        "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "DD", "PPG", "VMC",
        "MLM", "DOW", "LYB", "CTVA", "ALB", "EMN", "CE",
    ]},
    # Real Estate
    **{t: 'Real Estate' for t in [
        "PLD", "AMT", "CCI", "EQIX", "PSA", "O", "SPG", "WELL", "DLR", "AVB",
        "EQR", "VTR", "ARE", "ESS", "MAA", "UDR", "EXR", "IRM",
    ]},
    # Communication Services
    **{t: 'Communication Services' for t in [
        "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "CHTR", "EA", "TTWO",
        "WBD", "PARA", "OMC", "IPG", "LYV", "SPOT", "ROKU", "PINS", "SNAP",
    ]},
}


@dataclass
class ScreeningResult:
    """Result of universe screening."""
    initial_count: int
    after_liquidity: int
    after_momentum: int
    after_sector: int
    final_count: int
    
    passed_tickers: List[str]
    momentum_scores: Dict[str, float]
    sector_distribution: Dict[str, int]
    
    # Detailed filter stats
    failed_liquidity: List[str]
    failed_momentum: List[str]
    excluded_sector: List[str]


@dataclass
class LiquidityMetrics:
    """Liquidity metrics for a stock."""
    ticker: str
    avg_dollar_volume: float  # Average daily $ volume
    avg_volume: float  # Average daily shares
    min_price: float  # Minimum price in period
    trading_days: int  # Number of trading days
    zero_volume_days: int  # Days with 0 volume
    max_gap_days: int  # Maximum consecutive 0-volume days
    passes_filter: bool


class UniverseScreener:
    """
    Multi-stage universe screening for scalable stock selection.
    
    Filters:
    1. Liquidity (volume, price, data quality)
    2. Momentum (return ranking)
    3. Sector diversification
    4. Optional TDA quality
    """
    
    def __init__(
        self,
        # Liquidity filters
        min_dollar_volume: float = 5_000_000,  # $5M daily
        min_price: float = 5.0,  # >$5
        min_trading_days: int = 252,  # 1 year
        max_zero_volume_streak: int = 5,  # Max 5 consecutive 0-vol days
        
        # Momentum filters
        momentum_lookback: int = 126,  # 6-month momentum
        momentum_percentile: float = 0.5,  # Top 50%
        
        # Sector limits
        max_sector_weight: float = 0.40,  # Max 40% in one sector
        min_sectors: int = 5,  # At least 5 sectors
        
        # Universe size
        target_universe_size: int = 200,
    ):
        self.min_dollar_volume = min_dollar_volume
        self.min_price = min_price
        self.min_trading_days = min_trading_days
        self.max_zero_volume_streak = max_zero_volume_streak
        
        self.momentum_lookback = momentum_lookback
        self.momentum_percentile = momentum_percentile
        
        self.max_sector_weight = max_sector_weight
        self.min_sectors = min_sectors
        
        self.target_universe_size = target_universe_size
        
        self.sector_map = SECTOR_MAP
    
    def calculate_liquidity_metrics(
        self,
        df: pd.DataFrame,
        ticker: str,
    ) -> LiquidityMetrics:
        """
        Calculate liquidity metrics for a stock.
        
        Args:
            df: OHLCV DataFrame
            ticker: Stock symbol
            
        Returns:
            LiquidityMetrics
        """
        if df is None or len(df) == 0:
            return LiquidityMetrics(
                ticker=ticker,
                avg_dollar_volume=0,
                avg_volume=0,
                min_price=0,
                trading_days=0,
                zero_volume_days=0,
                max_gap_days=999,
                passes_filter=False,
            )
        
        # Ensure required columns
        if 'close' not in df.columns or 'volume' not in df.columns:
            return LiquidityMetrics(
                ticker=ticker,
                avg_dollar_volume=0,
                avg_volume=0,
                min_price=0,
                trading_days=0,
                zero_volume_days=0,
                max_gap_days=999,
                passes_filter=False,
            )
        
        # Calculate metrics - handle potential Series/scalar issues
        try:
            close_col = df['close']
            volume_col = df['volume']
            
            # Ensure they're 1D series (not multi-column)
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.iloc[:, 0]
            if isinstance(volume_col, pd.DataFrame):
                volume_col = volume_col.iloc[:, 0]
            
            dollar_volume = close_col * volume_col
            avg_dollar_volume = float(dollar_volume.mean())
            avg_volume = float(volume_col.mean())
            min_price = float(close_col.min())
            trading_days = len(df)
            
            # Zero volume analysis
            zero_vol_mask = volume_col == 0
            zero_volume_days = int(zero_vol_mask.sum())
            
            # Maximum consecutive zero-volume days
            max_gap = 0
            current_gap = 0
            for is_zero in zero_vol_mask:
                if is_zero:
                    current_gap += 1
                    max_gap = max(max_gap, current_gap)
                else:
                    current_gap = 0
        except Exception as e:
            logger.debug(f"Error calculating metrics for {ticker}: {e}")
            return LiquidityMetrics(
                ticker=ticker,
                avg_dollar_volume=0,
                avg_volume=0,
                min_price=0,
                trading_days=0,
                zero_volume_days=0,
                max_gap_days=999,
                passes_filter=False,
            )
        
        # Check if passes filter
        passes = bool(
            avg_dollar_volume >= self.min_dollar_volume and
            min_price >= self.min_price and
            trading_days >= self.min_trading_days and
            max_gap <= self.max_zero_volume_streak
        )
        
        return LiquidityMetrics(
            ticker=ticker,
            avg_dollar_volume=avg_dollar_volume,
            avg_volume=avg_volume,
            min_price=min_price,
            trading_days=trading_days,
            zero_volume_days=zero_volume_days,
            max_gap_days=max_gap,
            passes_filter=passes,
        )
    
    def apply_liquidity_filters(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
    ) -> Tuple[List[str], List[str], Dict[str, LiquidityMetrics]]:
        """
        Apply liquidity filters to universe.
        
        Args:
            ohlcv_dict: Dict of {ticker: OHLCV DataFrame}
            
        Returns:
            Tuple of (passed_tickers, failed_tickers, metrics_dict)
        """
        passed = []
        failed = []
        metrics = {}
        
        for ticker, df in ohlcv_dict.items():
            m = self.calculate_liquidity_metrics(df, ticker)
            metrics[ticker] = m
            
            if m.passes_filter:
                passed.append(ticker)
            else:
                failed.append(ticker)
        
        logger.info(f"Liquidity filter: {len(passed)}/{len(ohlcv_dict)} passed "
                   f"(${self.min_dollar_volume/1e6:.0f}M volume, ${self.min_price} min price)")
        
        return passed, failed, metrics
    
    def calculate_momentum_score(
        self,
        df: pd.DataFrame,
        lookback: int = None,
    ) -> float:
        """
        Calculate volatility-adjusted momentum score.
        
        Args:
            df: OHLCV DataFrame
            lookback: Lookback period in days
            
        Returns:
            Momentum score (return / volatility)
        """
        lookback = lookback or self.momentum_lookback
        
        if df is None or len(df) < lookback:
            return -np.inf
        
        prices = df['close'].iloc[-lookback:]
        
        if len(prices) < lookback:
            return -np.inf
        
        # Raw return
        raw_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        
        # Volatility (annualized)
        returns = prices.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        
        if volatility < 0.001:  # Avoid division by zero
            return raw_return
        
        # Risk-adjusted momentum
        momentum_score = raw_return / volatility
        
        return momentum_score
    
    def calculate_momentum_scores(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
        tickers: List[str] = None,
    ) -> pd.Series:
        """
        Calculate momentum scores for all tickers.
        
        Args:
            ohlcv_dict: Dict of {ticker: OHLCV DataFrame}
            tickers: Optional list of tickers to score (defaults to all)
            
        Returns:
            Series of {ticker: momentum_score}, sorted descending
        """
        tickers = tickers or list(ohlcv_dict.keys())
        
        scores = {}
        for ticker in tickers:
            if ticker in ohlcv_dict:
                scores[ticker] = self.calculate_momentum_score(ohlcv_dict[ticker])
        
        # Convert to series and sort
        score_series = pd.Series(scores).sort_values(ascending=False)
        
        # Remove -inf scores
        score_series = score_series[score_series > -np.inf]
        
        return score_series
    
    def apply_momentum_filter(
        self,
        momentum_scores: pd.Series,
        percentile: float = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Filter to top momentum percentile.
        
        Args:
            momentum_scores: Series of momentum scores
            percentile: Keep top X percentile (0.5 = top 50%)
            
        Returns:
            Tuple of (passed_tickers, failed_tickers)
        """
        percentile = percentile or self.momentum_percentile
        
        # Keep stocks with positive momentum above percentile
        threshold = momentum_scores.quantile(1 - percentile)
        
        passed = momentum_scores[momentum_scores >= threshold].index.tolist()
        failed = momentum_scores[momentum_scores < threshold].index.tolist()
        
        # Also filter out negative momentum
        passed = [t for t in passed if momentum_scores[t] > 0]
        
        logger.info(f"Momentum filter: {len(passed)}/{len(momentum_scores)} passed "
                   f"(top {percentile*100:.0f}%, threshold={threshold:.3f})")
        
        return passed, failed
    
    def get_sector(self, ticker: str) -> str:
        """Get sector for ticker."""
        return self.sector_map.get(ticker, 'Other')
    
    def apply_sector_diversification(
        self,
        candidates: List[str],
        momentum_scores: pd.Series,
    ) -> Tuple[List[str], List[str], Dict[str, int]]:
        """
        Apply sector diversification constraints.
        
        Args:
            candidates: List of candidate tickers
            momentum_scores: Series of momentum scores for ranking
            
        Returns:
            Tuple of (passed_tickers, excluded_tickers, sector_counts)
        """
        # Count by sector
        sector_counts = defaultdict(int)
        for ticker in candidates:
            sector = self.get_sector(ticker)
            sector_counts[sector] += 1
        
        max_per_sector = int(len(candidates) * self.max_sector_weight)
        
        # Select tickers respecting sector limits
        passed = []
        excluded = []
        selected_sector_counts = defaultdict(int)
        
        # Sort candidates by momentum score
        sorted_candidates = sorted(
            candidates,
            key=lambda t: momentum_scores.get(t, -np.inf),
            reverse=True
        )
        
        for ticker in sorted_candidates:
            sector = self.get_sector(ticker)
            
            if selected_sector_counts[sector] < max_per_sector:
                passed.append(ticker)
                selected_sector_counts[sector] += 1
            else:
                excluded.append(ticker)
        
        # Verify sector diversity
        n_sectors = len([s for s, c in selected_sector_counts.items() if c > 0])
        
        logger.info(f"Sector diversification: {len(passed)} selected, "
                   f"{n_sectors} sectors, max {max_per_sector} per sector")
        
        return passed, excluded, dict(selected_sector_counts)
    
    def get_final_universe(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
        size: int = None,
    ) -> ScreeningResult:
        """
        Run full screening pipeline.
        
        Args:
            ohlcv_dict: Dict of {ticker: OHLCV DataFrame}
            size: Target universe size (default: self.target_universe_size)
            
        Returns:
            ScreeningResult with all details
        """
        size = size or self.target_universe_size
        initial_count = len(ohlcv_dict)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Universe Screening: {initial_count} stocks → target {size}")
        logger.info(f"{'='*60}")
        
        # Stage 1: Liquidity filters
        logger.info("\nStage 1: Liquidity & Data Quality Filters")
        liquidity_passed, liquidity_failed, _ = self.apply_liquidity_filters(ohlcv_dict)
        after_liquidity = len(liquidity_passed)
        
        # Stage 2: Momentum ranking
        logger.info("\nStage 2: Momentum Ranking")
        momentum_scores = self.calculate_momentum_scores(ohlcv_dict, liquidity_passed)
        momentum_passed, momentum_failed = self.apply_momentum_filter(momentum_scores)
        after_momentum = len(momentum_passed)
        
        # Stage 3: Sector diversification
        logger.info("\nStage 3: Sector Diversification")
        sector_passed, sector_excluded, sector_dist = self.apply_sector_diversification(
            momentum_passed,
            momentum_scores,
        )
        after_sector = len(sector_passed)
        
        # Limit to target size
        final_passed = sector_passed[:size]
        final_count = len(final_passed)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Screening Complete: {initial_count} → {final_count} stocks")
        logger.info(f"{'='*60}")
        
        return ScreeningResult(
            initial_count=initial_count,
            after_liquidity=after_liquidity,
            after_momentum=after_momentum,
            after_sector=after_sector,
            final_count=final_count,
            passed_tickers=final_passed,
            momentum_scores={t: momentum_scores.get(t, 0) for t in final_passed},
            sector_distribution=sector_dist,
            failed_liquidity=liquidity_failed,
            failed_momentum=momentum_failed,
            excluded_sector=sector_excluded,
        )
    
    def print_screening_summary(self, result: ScreeningResult):
        """Print screening funnel summary."""
        print("\n" + "="*60)
        print("UNIVERSE SCREENING FUNNEL")
        print("="*60)
        
        print(f"\n{'Stage':<30} {'Count':>10} {'Pass Rate':>12}")
        print("-"*55)
        print(f"{'Initial Universe':<30} {result.initial_count:>10}")
        print(f"{'After Liquidity Filter':<30} {result.after_liquidity:>10} {result.after_liquidity/result.initial_count:>11.1%}")
        print(f"{'After Momentum Filter':<30} {result.after_momentum:>10} {result.after_momentum/max(1,result.after_liquidity):>11.1%}")
        print(f"{'After Sector Diversification':<30} {result.after_sector:>10}")
        print(f"{'Final Universe':<30} {result.final_count:>10}")
        
        print(f"\n{'Sector Distribution':}")
        print("-"*40)
        for sector, count in sorted(result.sector_distribution.items(), key=lambda x: -x[1]):
            pct = count / max(1, result.final_count)
            bar = "█" * int(pct * 20)
            print(f"  {sector:<25} {count:>4} ({pct:>5.1%}) {bar}")
        
        print(f"\nTop 10 Momentum Stocks:")
        print("-"*40)
        sorted_mom = sorted(result.momentum_scores.items(), key=lambda x: -x[1])[:10]
        for ticker, score in sorted_mom:
            sector = self.get_sector(ticker)
            print(f"  {ticker:<8} {score:>8.3f}  ({sector})")


def test_screener():
    """Test the universe screener."""
    print("\n" + "="*60)
    print("Phase 7: Universe Screener Test")
    print("="*60)
    
    # Generate synthetic test data
    import numpy as np
    np.random.seed(42)
    
    test_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "BAC", "WFC",
        "JNJ", "UNH", "PFE", "XOM", "CVX", "HD", "PG", "KO", "MCD", "CAT",
    ]
    
    ohlcv_dict = {}
    for ticker in test_tickers:
        n = 300
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        
        # Random walk with drift
        drift = np.random.uniform(-0.001, 0.003)
        returns = np.random.randn(n) * 0.02 + drift
        prices = 100 * np.cumprod(1 + returns)
        
        volume = np.random.uniform(1e6, 1e8, n)
        
        ohlcv_dict[ticker] = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': volume,
        }, index=dates)
    
    # Test screener
    screener = UniverseScreener(
        min_dollar_volume=1_000_000,  # Lower for test
        min_trading_days=100,  # Lower for test
        target_universe_size=10,
    )
    
    result = screener.get_final_universe(ohlcv_dict)
    screener.print_screening_summary(result)


if __name__ == "__main__":
    test_screener()

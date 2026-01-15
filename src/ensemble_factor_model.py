"""
Phase 8: Ensemble Multi-Factor Model

Combines 4 alpha factors with regime-adaptive weighting:
1. Momentum (35% base) - Cross-sectional relative momentum
2. TDA Topological (25% base) - Persistence diagram features
3. Value (20% base) - Price/earnings, book/market
4. Quality (20% base) - Profitability, stability metrics

Targets: CAGR >18%, Sharpe >1.2, MaxDD <15%
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


@dataclass
class FactorWeights:
    """Factor weights that sum to 1.0."""
    momentum: float = 0.35
    tda: float = 0.25
    value: float = 0.20
    quality: float = 0.20
    
    def __post_init__(self):
        total = self.momentum + self.tda + self.value + self.quality
        if abs(total - 1.0) > 0.01:
            # Normalize
            self.momentum /= total
            self.tda /= total
            self.value /= total
            self.quality /= total
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'momentum': self.momentum,
            'tda': self.tda,
            'value': self.value,
            'quality': self.quality,
        }


# Regime-specific factor weights
REGIME_WEIGHTS = {
    'bull': FactorWeights(momentum=0.45, tda=0.25, value=0.15, quality=0.15),
    'bear': FactorWeights(momentum=0.15, tda=0.25, value=0.20, quality=0.40),
    'recovery': FactorWeights(momentum=0.30, tda=0.20, value=0.35, quality=0.15),
    'sideways': FactorWeights(momentum=0.25, tda=0.30, value=0.25, quality=0.20),
    'volatile': FactorWeights(momentum=0.20, tda=0.35, value=0.15, quality=0.30),
    'default': FactorWeights(),  # Base weights
}


def _get_close_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """Extract close price series from DataFrame, handling multi-level columns and case variations."""
    if df is None or len(df) == 0:
        return None
    
    try:
        # Handle multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                return df['Close'].iloc[:, 0]
            elif 'Adj Close' in df.columns.get_level_values(0):
                return df['Adj Close'].iloc[:, 0]
            elif 'close' in df.columns.get_level_values(0):
                return df['close'].iloc[:, 0]
        else:
            # Try different case variations
            for col in ['Close', 'close', 'Adj Close', 'adj close']:
                if col in df.columns:
                    return df[col]
        
        return None
    except Exception:
        return None


def zscore_cross_sectional(values: pd.Series, winsorize_pct: float = 0.01) -> pd.Series:
    """
    Compute cross-sectional z-scores with winsorization.
    
    Args:
        values: Raw factor values
        winsorize_pct: Percentage to winsorize at each tail
        
    Returns:
        Z-scored values
    """
    if len(values) == 0:
        return values
    
    # Winsorize outliers
    lower = np.nanpercentile(values, winsorize_pct * 100)
    upper = np.nanpercentile(values, (1 - winsorize_pct) * 100)
    clipped = values.clip(lower, upper)
    
    # Z-score
    mean = clipped.mean()
    std = clipped.std()
    
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=values.index)
    
    return (clipped - mean) / std


class MomentumFactor:
    """
    Cross-sectional momentum factor.
    
    Combines multiple lookback periods:
    - 12-month momentum (skip last month)
    - 6-month momentum
    - 3-month momentum
    - 1-month reversal (negative weight)
    """
    
    def __init__(
        self,
        lookback_12m: int = 252,
        lookback_6m: int = 126,
        lookback_3m: int = 63,
        lookback_1m: int = 21,
        skip_last: int = 21,  # Skip last month to avoid reversal
        weight_12m: float = 0.40,
        weight_6m: float = 0.30,
        weight_3m: float = 0.25,
        weight_1m: float = -0.05,  # Negative for reversal
    ):
        self.lookback_12m = lookback_12m
        self.lookback_6m = lookback_6m
        self.lookback_3m = lookback_3m
        self.lookback_1m = lookback_1m
        self.skip_last = skip_last
        self.weight_12m = weight_12m
        self.weight_6m = weight_6m
        self.weight_3m = weight_3m
        self.weight_1m = weight_1m
    
    def compute(
        self,
        prices: Dict[str, pd.DataFrame],
        end_date: Optional[str] = None,
    ) -> pd.Series:
        """
        Compute momentum scores for all tickers.
        
        Args:
            prices: Dict of {ticker: DataFrame with 'Close' column}
            end_date: Optional end date for calculation
            
        Returns:
            Series of z-scored momentum values
        """
        momentum_scores = {}
        
        for ticker, df in prices.items():
            try:
                if df is None or len(df) < self.lookback_12m + self.skip_last:
                    continue
                
                # Get close prices using helper
                close = _get_close_series(df)
                if close is None:
                    continue
                
                # Apply end date filter
                if end_date:
                    close = close[:end_date]
                
                if len(close) < self.lookback_12m + self.skip_last:
                    continue
                
                # Skip last N days
                close_adj = close.iloc[:-self.skip_last] if self.skip_last > 0 else close
                
                # Calculate returns for each period
                ret_12m = close_adj.iloc[-1] / close_adj.iloc[-self.lookback_12m] - 1 if len(close_adj) >= self.lookback_12m else np.nan
                ret_6m = close_adj.iloc[-1] / close_adj.iloc[-self.lookback_6m] - 1 if len(close_adj) >= self.lookback_6m else np.nan
                ret_3m = close_adj.iloc[-1] / close_adj.iloc[-self.lookback_3m] - 1 if len(close_adj) >= self.lookback_3m else np.nan
                ret_1m = close.iloc[-1] / close.iloc[-self.lookback_1m] - 1 if len(close) >= self.lookback_1m else np.nan  # Recent reversal
                
                # Weighted combination
                if pd.notna(ret_12m):
                    score = (
                        self.weight_12m * ret_12m +
                        self.weight_6m * ret_6m +
                        self.weight_3m * ret_3m +
                        self.weight_1m * ret_1m
                    )
                    momentum_scores[ticker] = score
                    
            except Exception as e:
                logger.debug(f"Momentum calc failed for {ticker}: {e}")
                continue
        
        if not momentum_scores:
            return pd.Series(dtype=float)
        
        raw = pd.Series(momentum_scores)
        return zscore_cross_sectional(raw)


class TDAFactor:
    """
    TDA Topological factor wrapper.
    
    Uses pre-computed TDA features from the parallel engine.
    """
    
    def __init__(self, feature_weights: Optional[Dict[str, float]] = None):
        self.feature_weights = feature_weights or {
            'persistence_0': 0.25,  # H0 persistence entropy
            'persistence_1': 0.30,  # H1 holes (trend patterns)
            'betti_0': 0.15,        # Connected components
            'betti_1': 0.30,        # Cyclical patterns
        }
    
    def compute(
        self,
        tda_features: Dict[str, Dict],
    ) -> pd.Series:
        """
        Compute TDA factor scores from pre-computed features.
        
        Args:
            tda_features: Dict of {ticker: {feature_name: value}}
            
        Returns:
            Series of z-scored TDA values
        """
        tda_scores = {}
        
        for ticker, features in tda_features.items():
            try:
                if not features:
                    continue
                
                # Combine available features with weights
                score = 0.0
                total_weight = 0.0
                
                for feature, weight in self.feature_weights.items():
                    if feature in features and pd.notna(features[feature]):
                        score += weight * features[feature]
                        total_weight += weight
                
                if total_weight > 0:
                    tda_scores[ticker] = score / total_weight
                    
            except Exception as e:
                logger.debug(f"TDA score failed for {ticker}: {e}")
                continue
        
        if not tda_scores:
            return pd.Series(dtype=float)
        
        raw = pd.Series(tda_scores)
        return zscore_cross_sectional(raw)


class ValueFactor:
    """
    Value factor based on valuation ratios.
    
    Uses:
    - Earnings yield (inverse P/E)
    - Book/Market ratio
    - Cash flow yield
    """
    
    def __init__(self, cache: Dict[str, Dict] = None):
        self.cache = cache or {}
    
    def _fetch_fundamentals(self, ticker: str) -> Dict[str, float]:
        """Fetch fundamental data from yfinance."""
        if ticker in self.cache:
            return self.cache[ticker]
        
        if not HAS_YFINANCE:
            return {}
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Calculate value metrics
            pe = info.get('trailingPE') or info.get('forwardPE')
            pb = info.get('priceToBook')
            ps = info.get('priceToSalesTrailing12Months')
            pcf = info.get('priceToFreeCashFlow') or info.get('priceToCashflow')
            
            result = {
                'earnings_yield': 1 / pe if pe and pe > 0 else np.nan,
                'book_to_market': 1 / pb if pb and pb > 0 else np.nan,
                'sales_yield': 1 / ps if ps and ps > 0 else np.nan,
                'cf_yield': 1 / pcf if pcf and pcf > 0 else np.nan,
            }
            
            self.cache[ticker] = result
            return result
            
        except Exception as e:
            logger.debug(f"Fundamentals fetch failed for {ticker}: {e}")
            return {}
    
    def compute(
        self,
        tickers: List[str],
        fundamentals: Optional[Dict[str, Dict]] = None,
    ) -> pd.Series:
        """
        Compute value factor scores.
        
        Args:
            tickers: List of tickers
            fundamentals: Pre-fetched fundamentals (optional)
            
        Returns:
            Series of z-scored value scores
        """
        value_scores = {}
        
        for ticker in tickers:
            try:
                # Use provided fundamentals or fetch
                if fundamentals and ticker in fundamentals:
                    data = fundamentals[ticker]
                else:
                    data = self._fetch_fundamentals(ticker)
                
                if not data:
                    continue
                
                # Combine value metrics
                metrics = []
                weights = []
                
                if pd.notna(data.get('earnings_yield')):
                    metrics.append(data['earnings_yield'])
                    weights.append(0.40)
                    
                if pd.notna(data.get('book_to_market')):
                    metrics.append(data['book_to_market'])
                    weights.append(0.30)
                    
                if pd.notna(data.get('cf_yield')):
                    metrics.append(data['cf_yield'])
                    weights.append(0.30)
                
                if metrics:
                    # Weighted average
                    total_weight = sum(weights)
                    score = sum(m * w for m, w in zip(metrics, weights)) / total_weight
                    value_scores[ticker] = score
                    
            except Exception as e:
                logger.debug(f"Value calc failed for {ticker}: {e}")
                continue
        
        if not value_scores:
            return pd.Series(dtype=float)
        
        raw = pd.Series(value_scores)
        return zscore_cross_sectional(raw)


class QualityFactor:
    """
    Quality factor based on profitability and stability.
    
    Uses:
    - Return on Equity (ROE)
    - Return on Assets (ROA)
    - Profit margin
    - Debt/Equity (negative)
    - Earnings stability
    """
    
    def __init__(self, cache: Dict[str, Dict] = None):
        self.cache = cache or {}
    
    def _fetch_quality_metrics(self, ticker: str) -> Dict[str, float]:
        """Fetch quality metrics from yfinance."""
        if ticker in self.cache:
            return self.cache[ticker]
        
        if not HAS_YFINANCE:
            return {}
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            result = {
                'roe': info.get('returnOnEquity') or np.nan,
                'roa': info.get('returnOnAssets') or np.nan,
                'profit_margin': info.get('profitMargins') or np.nan,
                'debt_equity': info.get('debtToEquity') or np.nan,
                'current_ratio': info.get('currentRatio') or np.nan,
            }
            
            self.cache[ticker] = result
            return result
            
        except Exception as e:
            logger.debug(f"Quality fetch failed for {ticker}: {e}")
            return {}
    
    def compute(
        self,
        tickers: List[str],
        prices: Optional[Dict[str, pd.DataFrame]] = None,
        quality_data: Optional[Dict[str, Dict]] = None,
    ) -> pd.Series:
        """
        Compute quality factor scores.
        
        Args:
            tickers: List of tickers
            prices: Price data for volatility calculation
            quality_data: Pre-fetched quality metrics
            
        Returns:
            Series of z-scored quality scores
        """
        quality_scores = {}
        
        for ticker in tickers:
            try:
                # Get quality metrics
                if quality_data and ticker in quality_data:
                    data = quality_data[ticker]
                else:
                    data = self._fetch_quality_metrics(ticker)
                
                components = []
                weights = []
                
                # ROE (higher is better)
                if pd.notna(data.get('roe')):
                    components.append(data['roe'])
                    weights.append(0.30)
                
                # ROA (higher is better)
                if pd.notna(data.get('roa')):
                    components.append(data['roa'])
                    weights.append(0.25)
                
                # Profit margin (higher is better)
                if pd.notna(data.get('profit_margin')):
                    components.append(data['profit_margin'])
                    weights.append(0.25)
                
                # Debt/Equity (lower is better - invert)
                if pd.notna(data.get('debt_equity')) and data['debt_equity'] > 0:
                    # Invert: lower D/E = higher score
                    inverted_de = 1 / (1 + data['debt_equity'])
                    components.append(inverted_de)
                    weights.append(0.20)
                
                # Volatility component (if prices available)
                if prices and ticker in prices:
                    df = prices[ticker]
                    if df is not None and len(df) >= 60:
                        close = _get_close_series(df)
                        if close is not None:
                            returns = close.pct_change().dropna()
                            vol = returns.std() * np.sqrt(252)
                            # Lower vol = higher quality
                            if vol > 0:
                                inv_vol = 1 / vol
                                components.append(inv_vol)
                                weights.append(0.15)
                
                if components:
                    total_weight = sum(weights)
                    score = sum(c * w for c, w in zip(components, weights)) / total_weight
                    quality_scores[ticker] = score
                    
            except Exception as e:
                logger.debug(f"Quality calc failed for {ticker}: {e}")
                continue
        
        if not quality_scores:
            return pd.Series(dtype=float)
        
        raw = pd.Series(quality_scores)
        return zscore_cross_sectional(raw)


class RegimeDetector:
    """
    Simple market regime detector based on SPY behavior.
    
    Regimes:
    - bull: Strong uptrend with low volatility
    - bear: Downtrend
    - recovery: Transition from bear to bull
    - sideways: Low directional movement
    - volatile: High volatility regime
    """
    
    def __init__(
        self,
        lookback_trend: int = 60,
        lookback_vol: int = 20,
        vol_threshold_high: float = 0.25,
        vol_threshold_low: float = 0.15,
    ):
        self.lookback_trend = lookback_trend
        self.lookback_vol = lookback_vol
        self.vol_threshold_high = vol_threshold_high
        self.vol_threshold_low = vol_threshold_low
    
    def detect(
        self,
        spy_prices: pd.DataFrame,
        end_date: Optional[str] = None,
    ) -> str:
        """
        Detect current market regime.
        
        Args:
            spy_prices: SPY price DataFrame
            end_date: Optional end date
            
        Returns:
            Regime string
        """
        try:
            close = _get_close_series(spy_prices)
            if close is None:
                return 'default'
            
            if end_date:
                close = close[:end_date]
            
            if len(close) < self.lookback_trend:
                return 'default'
            
            # Calculate metrics
            returns = close.pct_change().dropna()
            
            # Trend: recent return vs long-term
            ret_short = close.iloc[-20] / close.iloc[-self.lookback_trend] - 1 if len(close) >= self.lookback_trend else 0
            ret_recent = close.iloc[-1] / close.iloc[-20] - 1 if len(close) >= 20 else 0
            
            # Volatility
            vol = returns.tail(self.lookback_vol).std() * np.sqrt(252)
            
            # SMA trend
            sma_50 = close.tail(50).mean()
            sma_200 = close.tail(200).mean() if len(close) >= 200 else sma_50
            
            current = close.iloc[-1]
            
            # Decision logic
            if vol > self.vol_threshold_high:
                return 'volatile'
            elif current < sma_200 and ret_short < -0.10:
                return 'bear'
            elif current > sma_50 > sma_200 and ret_short > 0.10:
                return 'bull'
            elif current > sma_200 and ret_recent > 0.05 and ret_short < 0:
                return 'recovery'
            else:
                return 'sideways'
                
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return 'default'


class EnsembleFactorModel:
    """
    Main ensemble model combining all factors with regime-adaptive weighting.
    """
    
    def __init__(
        self,
        base_weights: Optional[FactorWeights] = None,
        use_regime_rotation: bool = True,
    ):
        self.base_weights = base_weights or FactorWeights()
        self.use_regime_rotation = use_regime_rotation
        
        # Factor calculators
        self.momentum = MomentumFactor()
        self.tda = TDAFactor()
        self.value = ValueFactor()
        self.quality = QualityFactor()
        self.regime_detector = RegimeDetector()
        
        # Results storage
        self.last_regime = None
        self.last_weights = None
        self.factor_scores = {}
        
        logger.info(f"Initialized EnsembleFactorModel with base weights: {self.base_weights.to_dict()}")
    
    def compute_composite_score(
        self,
        prices: Dict[str, pd.DataFrame],
        tda_features: Dict[str, Dict],
        spy_prices: Optional[pd.DataFrame] = None,
        end_date: Optional[str] = None,
        fundamentals: Optional[Dict[str, Dict]] = None,
        quality_data: Optional[Dict[str, Dict]] = None,
    ) -> pd.DataFrame:
        """
        Compute composite factor scores for all tickers.
        
        Args:
            prices: Dict of {ticker: price DataFrame}
            tda_features: Dict of {ticker: TDA features}
            spy_prices: SPY prices for regime detection
            end_date: Optional end date
            fundamentals: Pre-fetched fundamentals for value factor
            quality_data: Pre-fetched quality metrics
            
        Returns:
            DataFrame with factor scores and composite
        """
        tickers = list(prices.keys())
        
        # Detect regime and get weights
        if self.use_regime_rotation and spy_prices is not None:
            regime = self.regime_detector.detect(spy_prices, end_date)
            weights = REGIME_WEIGHTS.get(regime, self.base_weights)
        else:
            regime = 'default'
            weights = self.base_weights
        
        self.last_regime = regime
        self.last_weights = weights
        
        logger.info(f"Detected regime: {regime}, weights: {weights.to_dict()}")
        
        # Compute individual factors
        logger.info("Computing momentum scores...")
        momentum_scores = self.momentum.compute(prices, end_date)
        
        logger.info("Computing TDA scores...")
        tda_scores = self.tda.compute(tda_features)
        
        logger.info("Computing value scores...")
        value_scores = self.value.compute(tickers, fundamentals)
        
        logger.info("Computing quality scores...")
        quality_scores = self.quality.compute(tickers, prices, quality_data)
        
        # Store raw scores
        self.factor_scores = {
            'momentum': momentum_scores,
            'tda': tda_scores,
            'value': value_scores,
            'quality': quality_scores,
        }
        
        # Build results DataFrame
        results = []
        
        for ticker in tickers:
            mom = momentum_scores.get(ticker, 0.0)
            tda_s = tda_scores.get(ticker, 0.0)
            val = value_scores.get(ticker, 0.0)
            qual = quality_scores.get(ticker, 0.0)
            
            # Weighted composite
            composite = (
                weights.momentum * mom +
                weights.tda * tda_s +
                weights.value * val +
                weights.quality * qual
            )
            
            results.append({
                'ticker': ticker,
                'momentum': mom,
                'tda': tda_s,
                'value': val,
                'quality': qual,
                'composite': composite,
                'regime': regime,
            })
        
        df = pd.DataFrame(results)
        df = df.set_index('ticker')
        df = df.sort_values('composite', ascending=False)
        
        return df
    
    def rank_stocks(
        self,
        prices: Dict[str, pd.DataFrame],
        tda_features: Dict[str, Dict],
        spy_prices: Optional[pd.DataFrame] = None,
        end_date: Optional[str] = None,
        top_n: int = 50,
    ) -> List[str]:
        """
        Rank stocks by composite score and return top N.
        
        Returns:
            List of top N tickers
        """
        scores = self.compute_composite_score(
            prices, tda_features, spy_prices, end_date
        )
        
        return scores.head(top_n).index.tolist()
    
    def get_portfolio_weights(
        self,
        prices: Dict[str, pd.DataFrame],
        tda_features: Dict[str, Dict],
        spy_prices: Optional[pd.DataFrame] = None,
        end_date: Optional[str] = None,
        n_positions: int = 25,
        equal_weight: bool = True,
    ) -> Dict[str, float]:
        """
        Get portfolio weights based on factor scores.
        
        Args:
            prices: Price data
            tda_features: TDA features
            spy_prices: SPY for regime
            end_date: Date filter
            n_positions: Number of positions
            equal_weight: If True, equal weight; else score-weighted
            
        Returns:
            Dict of {ticker: weight}
        """
        scores = self.compute_composite_score(
            prices, tda_features, spy_prices, end_date
        )
        
        # Get top N
        top = scores.head(n_positions)
        
        if equal_weight:
            weight = 1.0 / n_positions
            return {t: weight for t in top.index}
        else:
            # Score-weighted (shift to positive)
            shifted = top['composite'] - top['composite'].min() + 0.01
            total = shifted.sum()
            return {t: s / total for t, s in shifted.items()}
    
    def print_summary(self, scores: pd.DataFrame, top_n: int = 20):
        """Print factor model summary."""
        print("\n" + "="*70)
        print("ENSEMBLE FACTOR MODEL SUMMARY")
        print("="*70)
        
        print(f"\nRegime: {self.last_regime.upper()}")
        print(f"Weights: Mom={self.last_weights.momentum:.0%}, TDA={self.last_weights.tda:.0%}, "
              f"Val={self.last_weights.value:.0%}, Qual={self.last_weights.quality:.0%}")
        
        print(f"\nTop {top_n} Stocks by Composite Score:")
        print("-"*70)
        print(f"{'Ticker':<8} {'Composite':>10} {'Momentum':>10} {'TDA':>10} {'Value':>10} {'Quality':>10}")
        print("-"*70)
        
        for i, (ticker, row) in enumerate(scores.head(top_n).iterrows()):
            print(f"{ticker:<8} {row['composite']:>10.3f} {row['momentum']:>10.3f} "
                  f"{row['tda']:>10.3f} {row['value']:>10.3f} {row['quality']:>10.3f}")
        
        print("-"*70)
        print(f"\nFactor Correlations:")
        factor_cols = ['momentum', 'tda', 'value', 'quality']
        corr = scores[factor_cols].corr()
        for f1 in factor_cols:
            row = "  " + f1.ljust(10)
            for f2 in factor_cols:
                row += f"{corr.loc[f1, f2]:>8.2f}"
            print(row)


def test_ensemble_model():
    """Test the ensemble factor model."""
    print("\n" + "="*60)
    print("Phase 8: Ensemble Factor Model Test")
    print("="*60)
    
    # Create sample price data
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2024-01-01', freq='D')
    
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'XOM', 'JNJ', 'UNH', 'HD', 'PG']
    
    prices = {}
    for i, ticker in enumerate(test_tickers):
        # Generate trending price with noise
        trend = 0.0003 + 0.0002 * (i - 5)  # Some trending up, some down
        noise = np.random.randn(len(dates)) * 0.015
        returns = trend + noise
        price = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'Open': price * 0.995,
            'High': price * 1.01,
            'Low': price * 0.99,
            'Close': price,
            'Volume': np.random.randint(1e6, 1e7, len(dates)),
        }, index=dates)
        prices[ticker] = df
    
    # Create fake TDA features
    tda_features = {}
    for ticker in test_tickers:
        tda_features[ticker] = {
            'persistence_0': np.random.randn() * 0.5,
            'persistence_1': np.random.randn() * 0.5,
            'betti_0': np.random.randint(1, 10),
            'betti_1': np.random.randint(0, 5),
        }
    
    # SPY for regime detection
    spy_returns = 0.0002 + np.random.randn(len(dates)) * 0.012
    spy_price = 400 * np.exp(np.cumsum(spy_returns))
    spy_df = pd.DataFrame({'Close': spy_price}, index=dates)
    
    # Test the model
    model = EnsembleFactorModel(use_regime_rotation=True)
    
    scores = model.compute_composite_score(
        prices=prices,
        tda_features=tda_features,
        spy_prices=spy_df,
    )
    
    model.print_summary(scores)
    
    # Get portfolio weights
    weights = model.get_portfolio_weights(
        prices=prices,
        tda_features=tda_features,
        spy_prices=spy_df,
        n_positions=5,
    )
    
    print("\nPortfolio Weights (Top 5):")
    for t, w in weights.items():
        print(f"  {t}: {w:.1%}")


if __name__ == "__main__":
    test_ensemble_model()

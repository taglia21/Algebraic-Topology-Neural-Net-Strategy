"""
Factor Engine for Phase 11
===========================

Multi-factor stock scoring system for large-scale stock selection.

5-Factor Model:
1. Momentum (40%): 12-month price momentum, skip last month
2. Quality (25%): ROE, earnings growth, debt ratio
3. Volatility-Adjusted Returns (20%): Returns / volatility
4. Relative Strength vs SPY (10%): Beta-adjusted performance
5. Liquidity (5%): Log(avg daily dollar volume)

Each factor is z-score normalized across the universe.
Final composite score is weighted average of factor z-scores.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class FactorWeights:
    """Weights for each factor in composite score."""
    momentum: float = 0.40
    quality: float = 0.25
    vol_adjusted: float = 0.20
    relative_strength: float = 0.10
    liquidity: float = 0.05
    
    def validate(self):
        """Ensure weights sum to 1.0."""
        total = self.momentum + self.quality + self.vol_adjusted + \
                self.relative_strength + self.liquidity
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Factor weights must sum to 1.0, got {total}")


class FactorEngine:
    """
    Multi-factor stock scoring engine.
    
    Computes factor scores for each stock in the universe and
    creates a composite ranking for stock selection.
    """
    
    def __init__(
        self,
        weights: FactorWeights = None,
        lookback_momentum: int = 252,  # 12 months
        lookback_short: int = 63,       # 3 months
        skip_momentum: int = 21,        # Skip last month (reversal)
        min_history: int = 100,         # Minimum days of data (reduced from 200)
    ):
        self.weights = weights or FactorWeights()
        self.weights.validate()
        
        self.lookback_momentum = lookback_momentum
        self.lookback_short = lookback_short
        self.skip_momentum = skip_momentum
        self.min_history = min_history
        
        # Cache for factor scores
        self.factor_cache = {}
    
    def compute_all_factors(
        self,
        price_data: Dict[str, pd.DataFrame],
        date: str,
        spy_data: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Compute all factor scores for all tickers.
        
        Args:
            price_data: Dict of ticker -> DataFrame with OHLCV
            date: Date to compute factors for
            spy_data: SPY data for relative strength calculation
            
        Returns:
            DataFrame with columns: ticker, momentum, quality, vol_adj, rel_strength, liquidity, composite
        """
        date_dt = pd.to_datetime(date)
        
        results = []
        
        for ticker, df in price_data.items():
            # Filter data up to date
            df_to_date = df[df.index <= date_dt]
            
            if len(df_to_date) < self.min_history:
                continue
            
            try:
                factors = self._compute_ticker_factors(ticker, df_to_date, spy_data, date_dt)
                if factors is not None:
                    results.append(factors)
            except Exception as e:
                logger.debug(f"Error computing factors for {ticker}: {e}")
                continue
        
        if not results:
            logger.warning("No factors computed for any ticker")
            return pd.DataFrame()
        
        # Create DataFrame
        df_factors = pd.DataFrame(results)
        
        # Z-score normalize each factor across universe
        for col in ['momentum', 'quality', 'vol_adjusted', 'relative_strength', 'liquidity']:
            if col in df_factors.columns:
                values = df_factors[col].values
                # Winsorize to handle outliers
                values = np.clip(values, np.percentile(values, 1), np.percentile(values, 99))
                # Z-score
                mean = np.mean(values)
                std = np.std(values)
                if std > 0:
                    df_factors[f'{col}_z'] = (values - mean) / std
                else:
                    df_factors[f'{col}_z'] = 0.0
        
        # Compute composite score
        df_factors['composite'] = (
            self.weights.momentum * df_factors.get('momentum_z', 0) +
            self.weights.quality * df_factors.get('quality_z', 0) +
            self.weights.vol_adjusted * df_factors.get('vol_adjusted_z', 0) +
            self.weights.relative_strength * df_factors.get('relative_strength_z', 0) +
            self.weights.liquidity * df_factors.get('liquidity_z', 0)
        )
        
        # Rank by composite
        df_factors = df_factors.sort_values('composite', ascending=False)
        df_factors['rank'] = range(1, len(df_factors) + 1)
        
        logger.info(f"Computed factors for {len(df_factors)} tickers on {date}")
        
        return df_factors
    
    def _compute_ticker_factors(
        self,
        ticker: str,
        df: pd.DataFrame,
        spy_data: pd.DataFrame,
        date: pd.Timestamp,
    ) -> Optional[Dict]:
        """Compute all factors for a single ticker."""
        # Handle both capitalized and lowercase column names
        close = df['Close'] if 'Close' in df.columns else df['close']
        volume = df['Volume'] if 'Volume' in df.columns else df['volume']
        
        # Factor 1: Momentum (12-month minus last month)
        momentum = self._compute_momentum(close)
        
        # Factor 2: Quality (simplified - use volatility stability as proxy)
        quality = self._compute_quality(close, volume)
        
        # Factor 3: Volatility-Adjusted Returns
        vol_adjusted = self._compute_vol_adjusted_returns(close)
        
        # Factor 4: Relative Strength vs SPY
        relative_strength = self._compute_relative_strength(close, spy_data, date)
        
        # Factor 5: Liquidity
        liquidity = self._compute_liquidity(close, volume)
        
        if any(pd.isna([momentum, quality, vol_adjusted, liquidity])):
            return None
        
        return {
            'ticker': ticker,
            'momentum': momentum,
            'quality': quality,
            'vol_adjusted': vol_adjusted,
            'relative_strength': relative_strength,
            'liquidity': liquidity,
            'current_price': close.iloc[-1],
            'avg_volume': volume.tail(20).mean(),
        }
    
    def _compute_momentum(self, close: pd.Series) -> float:
        """
        Compute momentum factor.
        
        12-month momentum with last month skipped (avoid reversal effect).
        Also blend with 3-month momentum for trend confirmation.
        """
        if len(close) < self.lookback_momentum + self.skip_momentum:
            return np.nan
        
        # 12-month momentum (skip last month)
        price_now = close.iloc[-self.skip_momentum]
        price_12m_ago = close.iloc[-(self.lookback_momentum + self.skip_momentum)]
        mom_12m = (price_now / price_12m_ago - 1) if price_12m_ago > 0 else 0
        
        # 3-month momentum (for trend confirmation)
        if len(close) >= self.lookback_short + self.skip_momentum:
            price_3m_ago = close.iloc[-(self.lookback_short + self.skip_momentum)]
            mom_3m = (price_now / price_3m_ago - 1) if price_3m_ago > 0 else 0
        else:
            mom_3m = mom_12m
        
        # Blend: 70% 12-month + 30% 3-month
        momentum = 0.7 * mom_12m + 0.3 * mom_3m
        
        return momentum
    
    def _compute_quality(self, close: pd.Series, volume: pd.Series) -> float:
        """
        Compute quality factor.
        
        Since we don't have fundamental data, use price-based proxies:
        - Consistency of returns (lower volatility of returns is higher quality)
        - Price stability (fewer large drawdowns)
        - Volume consistency (steady volume patterns)
        """
        returns = close.pct_change().dropna()
        
        if len(returns) < 60:
            return np.nan
        
        recent_returns = returns.tail(60)
        
        # Downside deviation (penalize downside more than upside)
        downside = recent_returns[recent_returns < 0]
        downside_std = downside.std() if len(downside) > 5 else 0.05
        
        # Consistency (inverse of vol of vol)
        rolling_vol = returns.rolling(20).std()
        vol_of_vol = rolling_vol.std() / rolling_vol.mean() if rolling_vol.mean() > 0 else 1
        
        # Combine: lower downside deviation and vol_of_vol = higher quality
        quality = 1 / (downside_std + 0.01) - vol_of_vol
        
        return quality
    
    def _compute_vol_adjusted_returns(self, close: pd.Series) -> float:
        """
        Compute volatility-adjusted returns (like Sharpe numerator).
        
        Higher return per unit of volatility is better.
        """
        returns = close.pct_change().dropna()
        
        if len(returns) < 60:
            return np.nan
        
        # Use recent 12-month period
        recent = returns.tail(min(252, len(returns)))
        
        annual_return = (1 + recent.mean()) ** 252 - 1
        annual_vol = recent.std() * np.sqrt(252)
        
        if annual_vol > 0.01:
            return annual_return / annual_vol
        else:
            return 0.0
    
    def _compute_relative_strength(
        self,
        close: pd.Series,
        spy_data: pd.DataFrame,
        date: pd.Timestamp,
    ) -> float:
        """
        Compute relative strength vs SPY.
        
        Stock return minus beta-adjusted SPY return.
        """
        if spy_data is None or spy_data.empty:
            return 0.0
        
        # Handle both capitalized and lowercase column names
        spy_close = spy_data['Close'] if 'Close' in spy_data.columns else spy_data['close']
        spy_to_date = spy_close[spy_close.index <= date]
        stock_to_date = close[close.index <= date]
        
        if len(spy_to_date) < 60 or len(stock_to_date) < 60:
            return 0.0
        
        # Get overlapping dates
        common_dates = stock_to_date.index.intersection(spy_to_date.index)
        if len(common_dates) < 60:
            return 0.0
        
        stock_aligned = stock_to_date.loc[common_dates].tail(60)
        spy_aligned = spy_to_date.loc[common_dates].tail(60)
        
        stock_returns = stock_aligned.pct_change().dropna()
        spy_returns = spy_aligned.pct_change().dropna()
        
        if len(stock_returns) < 20:
            return 0.0
        
        # Calculate beta
        cov = np.cov(stock_returns.values, spy_returns.values)
        if cov[1, 1] > 0:
            beta = cov[0, 1] / cov[1, 1]
        else:
            beta = 1.0
        
        # Relative strength = stock return - beta * spy return
        stock_total_ret = (stock_aligned.iloc[-1] / stock_aligned.iloc[0] - 1)
        spy_total_ret = (spy_aligned.iloc[-1] / spy_aligned.iloc[0] - 1)
        
        relative_strength = stock_total_ret - beta * spy_total_ret
        
        return relative_strength
    
    def _compute_liquidity(self, close: pd.Series, volume: pd.Series) -> float:
        """
        Compute liquidity factor.
        
        Log of average daily dollar volume (higher is better for trading).
        """
        if len(close) < 20 or len(volume) < 20:
            return np.nan
        
        # Dollar volume = price * shares traded
        dollar_volume = close.tail(20) * volume.tail(20)
        avg_dollar_volume = dollar_volume.mean()
        
        if avg_dollar_volume > 0:
            return np.log10(avg_dollar_volume)
        else:
            return 0.0
    
    def get_top_stocks(
        self,
        factor_df: pd.DataFrame,
        n: int = 50,
        exclude_leveraged: bool = False,
    ) -> List[str]:
        """
        Get top N stocks by composite score.
        
        Args:
            factor_df: DataFrame from compute_all_factors
            n: Number of stocks to select
            exclude_leveraged: Whether to exclude leveraged ETFs
            
        Returns:
            List of top N ticker symbols
        """
        if factor_df.empty:
            return []
        
        df = factor_df.copy()
        
        # Exclude leveraged ETFs if requested
        if exclude_leveraged:
            leveraged = {'TQQQ', 'SPXL', 'UPRO', 'TNA', 'SOXL', 'TECL', 'FAS', 
                        'FNGU', 'LABU', 'QLD', 'SSO', 'UWM', 'ROM', 'USD',
                        'ERX', 'CURE', 'DPST', 'RETL', 'NAIL', 'DFEN'}
            df = df[~df['ticker'].isin(leveraged)]
        
        # Take top N
        top = df.head(n)['ticker'].tolist()
        
        return top

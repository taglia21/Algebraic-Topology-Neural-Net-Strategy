"""Regime Labeler for market state classification.

V1.3: Uses returns, volatility, and TDA entropy/complexity to derive coarse regimes.

Regime labels:
- trend_up: Strong positive returns with moderate volatility
- trend_down: Strong negative returns with moderate volatility
- high_vol: High volatility regardless of returns
- choppy: Low signal, high TDA entropy (complex structure)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class RegimeLabeler:
    """Label market regimes based on returns, volatility, and TDA features.
    
    V1.3: Integrates TDA entropy as a complexity proxy.
    """
    
    # Default quantile thresholds for regime classification
    DEFAULT_THRESHOLDS = {
        'ret_hi_quantile': 0.70,     # Top 30% of returns = trend
        'vol_hi_quantile': 0.80,     # Top 20% of vol = high_vol regime
        'vol_mid_quantile': 0.50,    # Median volatility
        'entropy_hi_quantile': 0.75, # Top 25% entropy = choppy
    }
    
    def __init__(self, window: int = 20, thresholds: Optional[Dict] = None):
        """Initialize regime labeler.
        
        Args:
            window: Rolling window for return/volatility calculation
            thresholds: Optional custom thresholds for regime classification
        """
        self.window = window
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
    
    def compute_rolling_metrics(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling return and volatility from price DataFrame.
        
        Args:
            prices: DataFrame with 'close' or 'Close' column
            
        Returns:
            DataFrame with 'ret_window', 'vol_window' columns
        """
        close_col = 'close' if 'close' in prices.columns else 'Close'
        close = prices[close_col].values
        
        # Log returns
        log_prices = np.log(close + 1e-10)
        daily_returns = np.diff(log_prices)
        daily_returns = np.concatenate([[0], daily_returns])
        
        # Rolling return (sum of log returns over window)
        ret_window = pd.Series(daily_returns).rolling(self.window).sum().values
        
        # Rolling volatility (std of daily returns over window)
        vol_window = pd.Series(daily_returns).rolling(self.window).std().values
        
        return pd.DataFrame({
            'ret_window': ret_window,
            'vol_window': vol_window
        }, index=prices.index)
    
    def label_regimes(self, 
                      prices: pd.DataFrame,
                      tda_features: Optional[pd.DataFrame] = None,
                      return_details: bool = False) -> pd.Series:
        """Label each date with a regime based on returns, volatility, TDA.
        
        Args:
            prices: DataFrame with price data (must have 'close' or 'Close')
            tda_features: Optional TDA feature DataFrame (aligned with prices)
            return_details: If True, return detailed DataFrame instead of Series
            
        Returns:
            pd.Series of regime labels aligned with prices index (after warmup)
        """
        # Compute rolling metrics
        metrics = self.compute_rolling_metrics(prices)
        
        # Drop NaN rows from warmup
        valid_mask = ~metrics['ret_window'].isna()
        metrics = metrics[valid_mask]
        
        # Compute quantile thresholds from the data
        ret_hi = np.quantile(metrics['ret_window'].dropna(), self.thresholds['ret_hi_quantile'])
        ret_lo = -ret_hi  # Symmetric for trend_down
        vol_hi = np.quantile(metrics['vol_window'].dropna(), self.thresholds['vol_hi_quantile'])
        vol_mid = np.quantile(metrics['vol_window'].dropna(), self.thresholds['vol_mid_quantile'])
        
        # Extract TDA entropy if available
        has_tda_entropy = (tda_features is not None and 
                          'entropy_l0' in tda_features.columns and 
                          'entropy_l1' in tda_features.columns)
        
        if has_tda_entropy:
            # Align TDA features with metrics
            tda_aligned = tda_features.loc[metrics.index] if hasattr(tda_features, 'loc') else tda_features
            
            # Use combined entropy as complexity measure
            if len(tda_aligned) == len(metrics):
                entropy_combined = np.sqrt(
                    tda_aligned['entropy_l0'].values**2 + 
                    tda_aligned['entropy_l1'].values**2
                )
                entropy_hi = np.quantile(entropy_combined[~np.isnan(entropy_combined)], 
                                         self.thresholds['entropy_hi_quantile'])
            else:
                has_tda_entropy = False
                entropy_combined = None
                entropy_hi = None
        else:
            entropy_combined = None
            entropy_hi = None
        
        # Classify each row
        labels = []
        for i, idx in enumerate(metrics.index):
            ret = metrics.loc[idx, 'ret_window']
            vol = metrics.loc[idx, 'vol_window']
            
            # Priority 1: High volatility → high_vol
            if vol > vol_hi:
                labels.append('high_vol')
            # Priority 2: Strong positive return + moderate vol → trend_up
            elif ret > ret_hi and vol <= vol_mid:
                labels.append('trend_up')
            # Priority 3: Strong negative return + moderate vol → trend_down
            elif ret < ret_lo and vol <= vol_mid:
                labels.append('trend_down')
            # Priority 4: High TDA entropy → choppy
            elif has_tda_entropy and entropy_combined is not None and entropy_combined[i] > entropy_hi:
                labels.append('choppy')
            # Default: choppy (catch-all)
            else:
                labels.append('choppy')
        
        regime_series = pd.Series(labels, index=metrics.index, name='regime')
        
        if return_details:
            details = metrics.copy()
            details['regime'] = regime_series
            if has_tda_entropy:
                details['entropy_combined'] = entropy_combined
            return details
        
        return regime_series
    
    def compute_regime_performance(self,
                                   regime_labels: pd.Series,
                                   daily_returns: pd.Series,
                                   annualize: bool = True) -> Dict:
        """Compute performance metrics per regime.
        
        Args:
            regime_labels: Series of regime labels
            daily_returns: Series of daily returns (aligned with regime_labels)
            annualize: If True, annualize Sharpe ratio
            
        Returns:
            Dict with performance by regime
        """
        # Align indices
        common_idx = regime_labels.index.intersection(daily_returns.index)
        regimes = regime_labels.loc[common_idx]
        returns = daily_returns.loc[common_idx]
        
        unique_regimes = ['trend_up', 'trend_down', 'high_vol', 'choppy']
        performance = {}
        
        for regime in unique_regimes:
            mask = regimes == regime
            regime_returns = returns[mask]
            
            if len(regime_returns) < 2:
                performance[regime] = {
                    'count': int(mask.sum()),
                    'pct_of_total': float(mask.mean() * 100),
                    'mean_return': 0.0,
                    'sharpe': 0.0,
                    'total_return': 0.0,
                }
                continue
            
            mean_ret = regime_returns.mean()
            std_ret = regime_returns.std()
            sharpe = (mean_ret / std_ret) if std_ret > 0 else 0
            if annualize:
                sharpe *= np.sqrt(252)
            
            performance[regime] = {
                'count': int(mask.sum()),
                'pct_of_total': float(mask.mean() * 100),
                'mean_return': float(mean_ret),
                'sharpe': float(sharpe),
                'total_return': float(regime_returns.sum()),
            }
        
        return performance
    
    def get_regime_summary(self, regime_labels: pd.Series) -> Dict:
        """Get summary counts of each regime.
        
        Args:
            regime_labels: Series of regime labels
            
        Returns:
            Dict with regime counts and percentages
        """
        counts = regime_labels.value_counts().to_dict()
        total = len(regime_labels)
        
        summary = {
            'total_samples': total,
            'regimes': {}
        }
        
        for regime in ['trend_up', 'trend_down', 'high_vol', 'choppy']:
            count = counts.get(regime, 0)
            summary['regimes'][regime] = {
                'count': count,
                'pct': round(count / total * 100, 1) if total > 0 else 0
            }
        
        return summary


def label_regimes(prices: pd.DataFrame,
                  tda_features: Optional[np.ndarray] = None,
                  window: int = 20) -> pd.Series:
    """
    Convenience function to label regimes aligned with prices.
    
    Returns a Series of regime labels per date, aligned with `prices.index`
    (after initial warmup). Uses:
      - rolling return,
      - rolling volatility,
      - TDA entropy / sum_lifetime as a complexity proxy.

    Example labels: "trend_up", "trend_down", "high_vol", "choppy".
    
    Args:
        prices: DataFrame with price data
        tda_features: Optional numpy array or DataFrame of TDA features
        window: Rolling window for metrics calculation
        
    Returns:
        pd.Series of regime labels
    """
    labeler = RegimeLabeler(window=window)
    
    # Convert numpy array to DataFrame if needed
    if tda_features is not None and isinstance(tda_features, np.ndarray):
        # Assume standard TDA feature order
        if tda_features.shape[1] >= 6:  # Has entropy columns
            tda_df = pd.DataFrame({
                'entropy_l0': tda_features[:, 4] if tda_features.shape[1] > 4 else np.zeros(len(tda_features)),
                'entropy_l1': tda_features[:, 5] if tda_features.shape[1] > 5 else np.zeros(len(tda_features)),
            })
        else:
            tda_df = None
    else:
        tda_df = tda_features
    
    return labeler.label_regimes(prices, tda_df)


def test():
    """Test regime labeler on synthetic data."""
    np.random.seed(42)
    n_bars = 200
    
    # Create synthetic price data with regime patterns
    # First 50 bars: trend up
    # Next 50 bars: choppy
    # Next 50 bars: high vol
    # Last 50 bars: trend down
    
    returns = np.concatenate([
        np.random.randn(50) * 0.01 + 0.005,   # Trend up
        np.random.randn(50) * 0.005,          # Choppy
        np.random.randn(50) * 0.03,           # High vol
        np.random.randn(50) * 0.01 - 0.005,   # Trend down
    ])
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'close': prices,
        'open': prices * (1 + np.random.randn(n_bars) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n_bars)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(n_bars)) * 0.01),
        'volume': np.random.randint(1000, 10000, n_bars),
    }, index=pd.date_range('2023-01-01', periods=n_bars))
    
    # Test regime labeler
    labeler = RegimeLabeler(window=20)
    regimes = labeler.label_regimes(df)
    
    assert len(regimes) > 0, "No regimes labeled"
    assert set(regimes.unique()).issubset({'trend_up', 'trend_down', 'high_vol', 'choppy'}), \
        f"Unexpected regimes: {regimes.unique()}"
    
    # Test summary
    summary = labeler.get_regime_summary(regimes)
    assert 'total_samples' in summary
    assert 'regimes' in summary
    
    # Test performance calculation
    daily_returns = pd.Series(returns, index=df.index)
    perf = labeler.compute_regime_performance(regimes, daily_returns.loc[regimes.index])
    
    assert 'trend_up' in perf
    assert 'sharpe' in perf['trend_up']
    
    # Test convenience function
    regimes2 = label_regimes(df, window=20)
    assert len(regimes2) == len(regimes)
    
    print("Regime Labeler V1.3: All tests passed!")
    print(f"  Sample regime distribution: {summary['regimes']}")
    
    return True


if __name__ == "__main__":
    success = test()
    if success:
        import sys
        sys.stdout.write("Regime Labeler V1.3: All tests passed\n")

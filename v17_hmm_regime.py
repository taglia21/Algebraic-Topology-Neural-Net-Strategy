#!/usr/bin/env python3
"""
V17.0 HMM Regime Detection
===========================
4-State Hidden Markov Model for market regime classification.

States:
- 0: LowVolTrend     - Calm uptrend, momentum works
- 1: HighVolTrend    - Volatile uptrend, breakout/trend works
- 2: LowVolMeanRevert - Range-bound, stat arb works
- 3: Crisis          - High vol drawdown, go defensive

Features:
- returns (10-day)
- realized_vol (10-day)
- volume_ratio (vs 20-day avg)
- trend_strength (price vs MA)

Trained on SPY, applied market-wide.
"""

import os
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('V17_HMM')


# Regime names and characteristics
REGIME_NAMES = {
    0: 'LowVolTrend',
    1: 'HighVolTrend', 
    2: 'LowVolMeanRevert',
    3: 'Crisis'
}

REGIME_STRATEGIES = {
    0: 'v17_momentum_xsection',   # Cross-sectional momentum
    1: 'v17_trend_follow',        # Breakout + ATR stops
    2: 'v17_stat_arb',            # Cointegrated pairs
    3: 'v17_defensive'            # Cash + vol targeting
}


class RegimeFeatureBuilder:
    """Build features for HMM regime detection"""
    
    def __init__(self, lookback: int = 10):
        self.lookback = lookback
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regime detection features from price data.
        
        Args:
            df: DataFrame with 'close', 'volume' columns (SPY data)
            
        Returns:
            DataFrame with feature columns
        """
        df = df.copy()
        
        # Ensure sorted by date
        if 'date' in df.columns:
            df = df.sort_values('date')
        
        close = df['close']
        volume = df.get('volume', pd.Series([1e8] * len(df)))
        
        # Feature 1: N-day returns
        df['returns_10d'] = close.pct_change(self.lookback)
        
        # Feature 2: Realized volatility (10-day)
        daily_ret = close.pct_change()
        df['realized_vol_10d'] = daily_ret.rolling(self.lookback).std() * np.sqrt(252)
        
        # Feature 3: Volume ratio (vs 20-day average)
        vol_ma = volume.rolling(20).mean()
        df['volume_ratio'] = volume / vol_ma
        
        # Feature 4: Trend strength (price vs 50-day MA)
        ma50 = close.rolling(50).mean()
        df['trend_strength'] = (close - ma50) / ma50
        
        # Feature 5: Price momentum (20-day)
        df['momentum_20d'] = close.pct_change(20)
        
        # Feature 6: Volatility of volatility
        df['vol_of_vol'] = df['realized_vol_10d'].rolling(10).std()
        
        return df
    
    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract feature matrix for HMM training.
        
        Returns:
            2D numpy array of shape (n_samples, n_features)
        """
        feature_cols = [
            'returns_10d',
            'realized_vol_10d', 
            'volume_ratio',
            'trend_strength'
        ]
        
        # Calculate features if not present
        if 'returns_10d' not in df.columns:
            df = self.calculate_features(df)
        
        # Drop NaN rows
        features_df = df[feature_cols].dropna()
        
        return features_df.values


class HMMRegimeDetector:
    """
    4-State Hidden Markov Model for regime detection.
    """
    
    def __init__(self, n_states: int = 4, n_iter: int = 100, random_state: int = 42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.model: Optional[GaussianHMM] = None
        self.feature_builder = RegimeFeatureBuilder()
        self.is_fitted = False
        
        # State characteristics (learned after fitting)
        self.state_means: Optional[np.ndarray] = None
        self.state_covars: Optional[np.ndarray] = None
        self.state_mapping: Dict[int, str] = {}
        
    def fit(self, spy_data: pd.DataFrame) -> 'HMMRegimeDetector':
        """
        Fit HMM on SPY data.
        
        Args:
            spy_data: DataFrame with 'close', 'volume' columns
        """
        logger.info("🔄 Fitting HMM regime detector...")
        
        # Build features
        df = self.feature_builder.calculate_features(spy_data)
        X = self.feature_builder.get_feature_matrix(df)
        
        if len(X) < 100:
            raise ValueError(f"Insufficient data for HMM training: {len(X)} samples")
        
        logger.info(f"   Training samples: {len(X)}")
        
        # Initialize and fit HMM
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type='full',
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        self.model.fit(X)
        
        # Store learned parameters
        self.state_means = self.model.means_
        self.state_covars = self.model.covars_
        
        # Map states to regimes based on characteristics
        self._map_states_to_regimes()
        
        self.is_fitted = True
        logger.info(f"✅ HMM fitted with {self.n_states} states")
        
        return self
    
    def _map_states_to_regimes(self):
        """
        Map learned HMM states to semantic regime names.
        
        Uses state means to classify based on return/volatility quadrants:
        - High return + Low vol = LowVolTrend (0)
        - High return + High vol = HighVolTrend (1)  
        - Low return + Low vol = LowVolMeanRevert (2)
        - Negative return + High vol = Crisis (3)
        """
        if self.state_means is None:
            return
        
        # Columns: returns_10d, realized_vol_10d, volume_ratio, trend_strength
        returns_idx = 0
        vol_idx = 1
        
        # Get median thresholds
        median_ret = np.median(self.state_means[:, returns_idx])
        median_vol = np.median(self.state_means[:, vol_idx])
        
        state_chars = []
        for i in range(self.n_states):
            ret = self.state_means[i, returns_idx]
            vol = self.state_means[i, vol_idx]
            state_chars.append({
                'state': i,
                'return': ret,
                'vol': vol,
                'high_ret': ret >= median_ret,
                'high_vol': vol >= median_vol
            })
        
        # Classify into 4 quadrants ensuring unique assignments
        regime_assignments = {}
        used_regimes = set()
        
        # Priority order: Crisis first (worst return + high vol)
        # then LowVolTrend (best conditions), then others
        
        # 1. Crisis: Negative return + High vol
        crisis_candidates = sorted(
            [s for s in state_chars if s['return'] < 0],
            key=lambda x: (-x['vol'], x['return'])  # Highest vol, worst return
        )
        if crisis_candidates:
            regime_assignments[crisis_candidates[0]['state']] = 3
            used_regimes.add(3)
        
        # 2. LowVolTrend: High return + Low vol
        trend_candidates = sorted(
            [s for s in state_chars if s['state'] not in regime_assignments and s['high_ret'] and not s['high_vol']],
            key=lambda x: (-x['return'], x['vol'])  # Highest return, lowest vol
        )
        if trend_candidates:
            regime_assignments[trend_candidates[0]['state']] = 0
            used_regimes.add(0)
        
        # 3. HighVolTrend: High return + High vol  
        hvt_candidates = sorted(
            [s for s in state_chars if s['state'] not in regime_assignments and s['high_ret']],
            key=lambda x: (-x['return'], -x['vol'])
        )
        if hvt_candidates and 1 not in used_regimes:
            regime_assignments[hvt_candidates[0]['state']] = 1
            used_regimes.add(1)
        
        # 4. LowVolMeanRevert: Low return + Low vol (remaining)
        for s in state_chars:
            if s['state'] not in regime_assignments:
                if 2 not in used_regimes:
                    regime_assignments[s['state']] = 2
                    used_regimes.add(2)
                elif 0 not in used_regimes:
                    regime_assignments[s['state']] = 0
                    used_regimes.add(0)
                elif 1 not in used_regimes:
                    regime_assignments[s['state']] = 1
                    used_regimes.add(1)
                else:
                    # Fallback: assign to closest matching regime
                    if s['high_ret'] and not s['high_vol']:
                        regime_assignments[s['state']] = 0
                    elif s['high_ret'] and s['high_vol']:
                        regime_assignments[s['state']] = 1
                    elif s['return'] < 0 and s['high_vol']:
                        regime_assignments[s['state']] = 3
                    else:
                        regime_assignments[s['state']] = 2
        
        self.state_mapping = regime_assignments
        
        # Log mapping
        for state, regime in self.state_mapping.items():
            logger.info(f"   State {state} -> {REGIME_NAMES[regime]}: "
                       f"ret={self.state_means[state, 0]:.4f}, "
                       f"vol={self.state_means[state, 1]:.4f}")
    
    def predict(self, spy_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime for each date.
        
        Returns:
            DataFrame with 'regime' and 'regime_name' columns
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Build features
        df = self.feature_builder.calculate_features(spy_data)
        
        # Get feature matrix
        feature_cols = ['returns_10d', 'realized_vol_10d', 'volume_ratio', 'trend_strength']
        valid_mask = df[feature_cols].notna().all(axis=1)
        
        X = df.loc[valid_mask, feature_cols].values
        
        # Predict states
        states = self.model.predict(X)
        
        # Map to regimes
        regimes = np.array([self.state_mapping[s] for s in states])
        regime_names = np.array([REGIME_NAMES[r] for r in regimes])
        
        # Add to dataframe
        result = df.loc[valid_mask].copy()
        result['hmm_state'] = states
        result['regime'] = regimes
        result['regime_name'] = regime_names
        
        return result
    
    def get_current_regime(self, spy_data: pd.DataFrame) -> Tuple[int, str]:
        """Get the current (latest) regime"""
        result = self.predict(spy_data)
        
        if result.empty:
            return 2, 'LowVolMeanRevert'  # Default
        
        latest = result.iloc[-1]
        return int(latest['regime']), str(latest['regime_name'])
    
    def get_regime_probabilities(self, spy_data: pd.DataFrame) -> pd.DataFrame:
        """Get regime probabilities for each date"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        df = self.feature_builder.calculate_features(spy_data)
        feature_cols = ['returns_10d', 'realized_vol_10d', 'volume_ratio', 'trend_strength']
        valid_mask = df[feature_cols].notna().all(axis=1)
        
        X = df.loc[valid_mask, feature_cols].values
        
        # Get posterior probabilities
        posteriors = self.model.predict_proba(X)
        
        result = df.loc[valid_mask].copy()
        for i in range(self.n_states):
            regime = self.state_mapping[i]
            result[f'prob_{REGIME_NAMES[regime]}'] = posteriors[:, i]
        
        return result
    
    def save(self, filepath: str):
        """Save fitted model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'state_means': self.state_means,
                'state_covars': self.state_covars,
                'state_mapping': self.state_mapping,
                'is_fitted': self.is_fitted
            }, f)
        
        logger.info(f"💾 Model saved: {filepath}")
    
    def load(self, filepath: str):
        """Load fitted model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.state_means = data['state_means']
        self.state_covars = data['state_covars']
        self.state_mapping = data['state_mapping']
        self.is_fitted = data['is_fitted']
        
        logger.info(f"📂 Model loaded: {filepath}")
    
    def get_transition_matrix(self) -> pd.DataFrame:
        """Get regime transition probability matrix"""
        if not self.is_fitted:
            return pd.DataFrame()
        
        trans_mat = self.model.transmat_
        
        # Map to regime names
        regime_order = [REGIME_NAMES[self.state_mapping[i]] for i in range(self.n_states)]
        
        return pd.DataFrame(
            trans_mat,
            index=regime_order,
            columns=regime_order
        )


def main():
    """Main entry point"""
    import yfinance as yf
    
    print("\n" + "=" * 60)
    print("🧠 V17.0 HMM REGIME DETECTOR")
    print("=" * 60)
    
    # Fetch SPY data for training
    logger.info("📥 Fetching SPY data for training...")
    spy = yf.download('SPY', period='5y', progress=False)
    
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.columns = [c.lower() for c in spy.columns]
    spy = spy.reset_index()
    spy.columns = [c.lower() if c != 'Date' else 'date' for c in spy.columns]
    
    logger.info(f"   Training data: {len(spy)} days")
    
    # Initialize and fit HMM
    detector = HMMRegimeDetector(n_states=4)
    detector.fit(spy)
    
    # Predict regimes
    result = detector.predict(spy)
    
    # Get regime distribution
    regime_counts = result['regime_name'].value_counts()
    
    print(f"\n📊 Regime Distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(result) * 100
        print(f"   {regime}: {count} days ({pct:.1f}%)")
    
    # Current regime
    current_regime, current_name = detector.get_current_regime(spy)
    print(f"\n🎯 Current Regime: {current_name} (state {current_regime})")
    print(f"   Recommended Strategy: {REGIME_STRATEGIES[current_regime]}")
    
    # Transition matrix
    trans_mat = detector.get_transition_matrix()
    print(f"\n📈 Transition Matrix:")
    print(trans_mat.round(3).to_string())
    
    # Save model
    detector.save('cache/v17_hmm_regime.pkl')
    
    # Save regime history
    regime_history = result[['date', 'regime', 'regime_name']].copy()
    regime_history.to_parquet('cache/v17_regime_history.parquet', index=False)
    
    print(f"\n✅ HMM regime detector ready")
    print(f"💾 Model saved to cache/v17_hmm_regime.pkl")
    
    return detector


if __name__ == "__main__":
    main()

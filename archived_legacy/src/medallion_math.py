# src/medallion_math.py

import numpy as np
from scipy import stats
from hmmlearn import hmm
import pywt
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class HurstExponent:
    """
    Calculates Hurst exponent using R/S analysis.
    H > 0.5: trending (use momentum)
    H < 0.5: mean-reverting (use stat arb)
    H ≈ 0.5: random walk (reduce position)
    """
    @staticmethod
    def calculate(series: np.ndarray, max_lag: int = 100) -> float:
        lags = range(2, min(max_lag, len(series) // 2))
        rs_values = []
        for lag in lags:
            subseries = [series[i:i+lag] for i in range(0, len(series)-lag, lag)]
            rs = []
            for s in subseries:
                if len(s) < 2: continue
                mean = np.mean(s)
                cumdev = np.cumsum(s - mean)
                R = np.max(cumdev) - np.min(cumdev)
                S = np.std(s, ddof=1) if np.std(s, ddof=1) > 0 else 1e-10
                rs.append(R / S)
            if rs:
                rs_values.append((np.log(lag), np.log(np.mean(rs))))
        if len(rs_values) < 2:
            return 0.5
        x, y = zip(*rs_values)
        slope, _, _, _, _ = stats.linregress(x, y)
        return np.clip(slope, 0, 1)


class MarketRegimeHMM:
    """
    4-state Hidden Markov Model for regime detection.
    States: Bull, Bear, High-Volatility, Sideways
    Uses Baum-Welch for parameter estimation.
    """
    def __init__(self, n_states: int = 4):
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.state_names = ['Bull', 'Bear', 'HighVol', 'Sideways']
        self.is_fitted = False
    
    def fit(self, returns: np.ndarray, volumes: np.ndarray) -> 'MarketRegimeHMM':
        X = np.column_stack([
            returns,
            np.abs(returns),  # volatility proxy
            np.log1p(volumes) if volumes is not None else np.zeros_like(returns)
        ])
        try:
            self.model.fit(X)
            self.is_fitted = True
        except (ValueError, np.linalg.LinAlgError):
            # Fallback to diagonal covariance if full fails
            self.model = hmm.GaussianHMM(
                n_components=self.model.n_components,
                covariance_type="diag",
                n_iter=100,
                random_state=42
            )
            self.model.fit(X)
            self.is_fitted = True
        return self
    
    def predict_regime(self, returns: np.ndarray, volumes: np.ndarray = None) -> Tuple[int, str, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X = np.column_stack([
            returns,
            np.abs(returns),
            np.log1p(volumes) if volumes is not None else np.zeros_like(returns)
        ])
        states = self.model.predict(X)
        probs = self.model.predict_proba(X)
        current_state = states[-1]
        return current_state, self.state_names[current_state], probs[-1]


class OrnsteinUhlenbeck:
    """
    Ornstein-Uhlenbeck process for mean reversion.
    dX = θ(μ - X)dt + σdW
    
    Calculates optimal entry/exit levels based on:
    - Half-life of mean reversion
    - Z-score thresholds
    - Sharpe-optimal stop-loss/take-profit
    """
    def __init__(self):
        self.theta = None  # mean reversion speed
        self.mu = None     # long-term mean
        self.sigma = None  # volatility
        self.half_life = None
    
    def fit(self, spread: np.ndarray, dt: float = 1/252) -> 'OrnsteinUhlenbeck':
        n = len(spread)
        X = spread[:-1]
        Y = spread[1:]
        
        # OLS regression: Y = a + b*X + epsilon
        X_mean = np.mean(X)
        Y_mean = np.mean(Y)
        b = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean)**2)
        a = Y_mean - b * X_mean
        
        # OU parameters from regression
        self.theta = -np.log(b) / dt if b > 0 else 1.0
        self.mu = a / (1 - b) if b != 1 else Y_mean
        residuals = Y - (a + b * X)
        self.sigma = np.std(residuals) * np.sqrt(2 * self.theta / (1 - b**2)) if b**2 < 1 else np.std(residuals)
        self.half_life = np.log(2) / self.theta if self.theta > 0 else np.inf
        
        return self
    
    def get_signal(self, current_value: float, entry_z: float = 2.0, exit_z: float = 0.5) -> dict:
        if self.mu is None:
            raise ValueError("Model not fitted")
        
        eq_std = self.sigma / np.sqrt(2 * self.theta) if self.theta > 0 else self.sigma
        z_score = (current_value - self.mu) / eq_std if eq_std > 0 else 0
        
        signal = {
            'z_score': z_score,
            'half_life_days': self.half_life * 252,
            'mean': self.mu,
            'current': current_value,
            'action': 'HOLD'
        }
        
        if z_score > entry_z:
            signal['action'] = 'SHORT'
        elif z_score < -entry_z:
            signal['action'] = 'LONG'
        elif abs(z_score) < exit_z:
            signal['action'] = 'EXIT'
            
        return signal


class WaveletDenoiser:
    """
    Wavelet-based signal denoising using Daubechies wavelets.
    Separates signal from noise at multiple time scales.
    """
    def __init__(self, wavelet: str = 'db4', level: int = 4):
        self.wavelet = wavelet
        self.level = level
    
    def denoise(self, signal: np.ndarray, threshold_mode: str = 'soft') -> np.ndarray:
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        
        # Calculate threshold using universal threshold (VisuShrink)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        
        # Apply thresholding to detail coefficients (not approximation)
        denoised_coeffs = [coeffs[0]]  # keep approximation
        for i in range(1, len(coeffs)):
            if threshold_mode == 'soft':
                denoised_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
            else:
                denoised_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='hard'))
        
        return pywt.waverec(denoised_coeffs, self.wavelet)[:len(signal)]


class PersistentHomologyTurbulence:
    """
    Simplified persistent homology turbulence index.
    Tracks topological changes in correlation structure.
    High turbulence = potential crash warning.
    
    Note: Full implementation requires giotto-tda or ripser.
    This is a correlation-based approximation.
    """
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.baseline_eigenvalues = None
    
    def fit_baseline(self, returns_matrix: np.ndarray) -> 'PersistentHomologyTurbulence':
        corr = np.corrcoef(returns_matrix.T)
        self.baseline_eigenvalues = np.linalg.eigvalsh(corr)
        return self
    
    def get_turbulence(self, returns_matrix: np.ndarray) -> float:
        if self.baseline_eigenvalues is None:
            self.fit_baseline(returns_matrix)
        
        corr = np.corrcoef(returns_matrix.T)
        current_eigenvalues = np.linalg.eigvalsh(corr)
        
        # Turbulence = change in eigenvalue distribution
        # High turbulence when correlation structure changes
        turbulence = np.sqrt(np.sum((current_eigenvalues - self.baseline_eigenvalues)**2))
        return turbulence


class MedallionStrategy:
    """
    Main orchestrator combining all mathematical components.
    Implements the Medallion-lite framework.
    """
    def __init__(self):
        self.hurst = HurstExponent()
        self.hmm = MarketRegimeHMM(n_states=4)
        self.ou = OrnsteinUhlenbeck()
        self.denoiser = WaveletDenoiser()
        self.turbulence = PersistentHomologyTurbulence()
    
    def analyze(self, prices: np.ndarray, volumes: np.ndarray = None) -> dict:
        # 1. Denoise the signal
        clean_prices = self.denoiser.denoise(prices)
        returns = np.diff(np.log(clean_prices))
        
        # 2. Calculate Hurst exponent
        h = self.hurst.calculate(returns)
        
        # 3. Detect regime
        if len(returns) > 50:
            self.hmm.fit(returns, volumes[1:] if volumes is not None else None)
            regime_id, regime_name, regime_probs = self.hmm.predict_regime(
                returns, volumes[1:] if volumes is not None else None
            )
        else:
            regime_id, regime_name, regime_probs = 3, 'Sideways', np.array([0.25]*4)
        
        # 4. O-U analysis for mean reversion
        self.ou.fit(clean_prices)
        ou_signal = self.ou.get_signal(clean_prices[-1])
        
        # 5. Strategy selection based on Hurst
        if h > 0.55:
            strategy = 'TREND_FOLLOWING'
            confidence = (h - 0.5) * 2  # 0 to 1
        elif h < 0.45:
            strategy = 'MEAN_REVERSION'
            confidence = (0.5 - h) * 2
        else:
            strategy = 'NEUTRAL'
            confidence = 0.0
        
        return {
            'hurst_exponent': h,
            'regime': regime_name,
            'regime_probabilities': regime_probs.tolist(),
            'ou_signal': ou_signal,
            'recommended_strategy': strategy,
            'strategy_confidence': confidence,
            'half_life_days': self.ou.half_life * 252 if self.ou.half_life else None
        }

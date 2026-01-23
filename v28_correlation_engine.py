#!/usr/bin/env python3
"""
V28 Cross-Asset Correlation Engine
====================================
Dynamic correlation matrix tracking with breakdown alerts and sector rotation signals.

Features:
- Real-time dynamic correlation matrix
- Rolling correlation with exponential weighting
- Correlation breakdown detection and alerts
- Sector rotation signals based on correlation shifts
- Portfolio diversification metrics
- Cross-asset risk monitoring

Key Metrics:
- Average pairwise correlation
- Correlation regime (normal/elevated/crisis)
- Sector correlation clustering
- Diversification benefit ratio
"""

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('V28_Correlation')


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class CorrelationRegime(Enum):
    """Correlation regime classification."""
    LOW = "low"          # Below 25th percentile - high diversification
    NORMAL = "normal"    # 25th-75th percentile - typical conditions
    ELEVATED = "elevated"  # 75th-90th percentile - rising correlations
    CRISIS = "crisis"    # Above 90th percentile - correlations spike


class AlertType(Enum):
    """Correlation alert types."""
    BREAKDOWN = "correlation_breakdown"
    SPIKE = "correlation_spike"
    REGIME_CHANGE = "regime_change"
    SECTOR_ROTATION = "sector_rotation"


@dataclass
class CorrelationAlert:
    """Correlation-based alert."""
    alert_type: AlertType
    severity: str  # 'info', 'warning', 'critical'
    message: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_type': self.alert_type.value,
            'severity': self.severity,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class SectorRotationSignal:
    """Sector rotation signal based on correlation analysis."""
    from_sector: str
    to_sector: str
    strength: float  # 0-1 strength of signal
    reason: str
    recommended_action: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'from_sector': self.from_sector,
            'to_sector': self.to_sector,
            'strength': self.strength,
            'reason': self.reason,
            'recommended_action': self.recommended_action,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class CorrelationState:
    """Current state of cross-asset correlations."""
    regime: CorrelationRegime
    average_correlation: float
    correlation_matrix: np.ndarray
    sector_correlations: Dict[str, Dict[str, float]]
    diversification_ratio: float
    effective_n_assets: float
    highest_pairs: List[Tuple[str, str, float]]
    lowest_pairs: List[Tuple[str, str, float]]
    regime_percentile: float
    rolling_volatility: float
    alerts: List[CorrelationAlert] = field(default_factory=list)
    rotation_signals: List[SectorRotationSignal] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'regime': self.regime.value,
            'average_correlation': self.average_correlation,
            'diversification_ratio': self.diversification_ratio,
            'effective_n_assets': self.effective_n_assets,
            'highest_pairs': self.highest_pairs[:5],
            'lowest_pairs': self.lowest_pairs[:5],
            'regime_percentile': self.regime_percentile,
            'rolling_volatility': self.rolling_volatility,
            'alerts': [a.to_dict() for a in self.alerts],
            'rotation_signals': [s.to_dict() for s in self.rotation_signals],
            'timestamp': self.timestamp.isoformat()
        }


# =============================================================================
# CORRELATION MATRIX TRACKER
# =============================================================================

class DynamicCorrelationMatrix:
    """
    Track and analyze dynamic correlations between assets.
    
    Uses exponentially weighted correlation for responsiveness
    while maintaining stability.
    """
    
    def __init__(
        self,
        lookback: int = 60,
        ewm_span: int = 20,
        min_periods: int = 30
    ):
        self.lookback = lookback
        self.ewm_span = ewm_span
        self.min_periods = min_periods
        
        # Storage
        self.returns_history: Optional[pd.DataFrame] = None
        self.correlation_history: List[np.ndarray] = []
        self.avg_corr_history: List[float] = []
        self.symbols: List[str] = []
        
        # Historical percentiles for regime classification
        self.corr_percentile_25: float = 0.15
        self.corr_percentile_75: float = 0.40
        self.corr_percentile_90: float = 0.55
    
    def update(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Update correlation matrix with new returns data.
        
        Args:
            returns: DataFrame with columns as assets, rows as dates
            
        Returns:
            Current correlation matrix
        """
        self.returns_history = returns
        self.symbols = list(returns.columns)
        
        # Calculate exponentially weighted correlation
        ewm_cov = returns.ewm(span=self.ewm_span, min_periods=self.min_periods).cov()
        
        # Get the most recent correlation matrix
        n_assets = len(self.symbols)
        last_idx = returns.index[-1]
        
        try:
            # Extract correlation from EWM covariance
            cov_slice = ewm_cov.loc[last_idx].values.reshape(n_assets, n_assets)
            std = np.sqrt(np.diag(cov_slice))
            std_outer = np.outer(std, std)
            corr_matrix = cov_slice / (std_outer + 1e-8)
            
            # Ensure diagonal is 1
            np.fill_diagonal(corr_matrix, 1.0)
            
            # Clip to valid range
            corr_matrix = np.clip(corr_matrix, -1, 1)
        except Exception:
            # Fallback to simple rolling correlation
            corr_matrix = returns.iloc[-self.lookback:].corr().values
        
        # Update history
        self.correlation_history.append(corr_matrix)
        
        # Calculate average correlation (excluding diagonal)
        mask = ~np.eye(n_assets, dtype=bool)
        avg_corr = np.mean(corr_matrix[mask])
        self.avg_corr_history.append(avg_corr)
        
        # Trim history
        if len(self.correlation_history) > 252:
            self.correlation_history = self.correlation_history[-252:]
            self.avg_corr_history = self.avg_corr_history[-252:]
        
        # Update percentiles
        if len(self.avg_corr_history) >= 60:
            self.corr_percentile_25 = np.percentile(self.avg_corr_history, 25)
            self.corr_percentile_75 = np.percentile(self.avg_corr_history, 75)
            self.corr_percentile_90 = np.percentile(self.avg_corr_history, 90)
        
        return corr_matrix
    
    def get_rolling_correlation(self, window: int = 20) -> np.ndarray:
        """Get simple rolling correlation matrix."""
        if self.returns_history is None or len(self.returns_history) < window:
            return np.eye(len(self.symbols))
        
        return self.returns_history.iloc[-window:].corr().values
    
    def get_correlation_change(self, periods: int = 5) -> np.ndarray:
        """Calculate change in correlation over periods."""
        if len(self.correlation_history) < periods + 1:
            return np.zeros_like(self.correlation_history[-1])
        
        return self.correlation_history[-1] - self.correlation_history[-1-periods]
    
    def classify_regime(self, avg_corr: float) -> Tuple[CorrelationRegime, float]:
        """Classify correlation regime and return percentile."""
        # Calculate percentile
        if len(self.avg_corr_history) > 0:
            percentile = stats.percentileofscore(self.avg_corr_history, avg_corr)
        else:
            percentile = 50.0
        
        # Classify
        if avg_corr < self.corr_percentile_25:
            return CorrelationRegime.LOW, percentile
        elif avg_corr < self.corr_percentile_75:
            return CorrelationRegime.NORMAL, percentile
        elif avg_corr < self.corr_percentile_90:
            return CorrelationRegime.ELEVATED, percentile
        else:
            return CorrelationRegime.CRISIS, percentile


# =============================================================================
# CORRELATION BREAKDOWN DETECTOR
# =============================================================================

class CorrelationBreakdownDetector:
    """
    Detect correlation breakdowns and unusual patterns.
    
    Monitors for:
    - Sudden correlation spikes
    - Correlation regime changes
    - Decorrelation opportunities
    """
    
    def __init__(
        self,
        spike_threshold: float = 0.20,  # Change in avg corr
        breakdown_threshold: float = -0.15,
        regime_change_periods: int = 5
    ):
        self.spike_threshold = spike_threshold
        self.breakdown_threshold = breakdown_threshold
        self.regime_change_periods = regime_change_periods
        
        self.previous_regime: Optional[CorrelationRegime] = None
        self.alert_callbacks: List[Callable[[CorrelationAlert], None]] = []
    
    def register_alert_callback(self, callback: Callable[[CorrelationAlert], None]):
        """Register callback for alerts."""
        self.alert_callbacks.append(callback)
    
    def check_for_alerts(
        self,
        current_matrix: np.ndarray,
        previous_matrix: np.ndarray,
        current_regime: CorrelationRegime,
        symbols: List[str]
    ) -> List[CorrelationAlert]:
        """
        Check for correlation alerts.
        
        Returns list of triggered alerts.
        """
        alerts = []
        
        # Calculate average correlation change
        n = len(symbols)
        mask = ~np.eye(n, dtype=bool)
        
        current_avg = np.mean(current_matrix[mask])
        previous_avg = np.mean(previous_matrix[mask])
        change = current_avg - previous_avg
        
        # Check for correlation spike
        if change > self.spike_threshold:
            alert = CorrelationAlert(
                alert_type=AlertType.SPIKE,
                severity='warning' if change < 0.30 else 'critical',
                message=f"Correlation spike detected: +{change:.1%}",
                details={
                    'previous_avg': previous_avg,
                    'current_avg': current_avg,
                    'change': change
                }
            )
            alerts.append(alert)
        
        # Check for correlation breakdown (decorrelation)
        if change < self.breakdown_threshold:
            alert = CorrelationAlert(
                alert_type=AlertType.BREAKDOWN,
                severity='info',
                message=f"Correlation breakdown detected: {change:.1%}",
                details={
                    'previous_avg': previous_avg,
                    'current_avg': current_avg,
                    'change': change
                }
            )
            alerts.append(alert)
        
        # Check for regime change
        if self.previous_regime and current_regime != self.previous_regime:
            severity = 'warning'
            if current_regime == CorrelationRegime.CRISIS:
                severity = 'critical'
            elif current_regime == CorrelationRegime.LOW:
                severity = 'info'
            
            alert = CorrelationAlert(
                alert_type=AlertType.REGIME_CHANGE,
                severity=severity,
                message=f"Correlation regime change: {self.previous_regime.value} → {current_regime.value}",
                details={
                    'from_regime': self.previous_regime.value,
                    'to_regime': current_regime.value
                }
            )
            alerts.append(alert)
        
        self.previous_regime = current_regime
        
        # Check for unusual pair correlations
        pair_alerts = self._check_pair_anomalies(current_matrix, previous_matrix, symbols)
        alerts.extend(pair_alerts)
        
        # Notify callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.warning(f"Alert callback error: {e}")
        
        return alerts
    
    def _check_pair_anomalies(
        self,
        current: np.ndarray,
        previous: np.ndarray,
        symbols: List[str]
    ) -> List[CorrelationAlert]:
        """Check for unusual changes in specific pair correlations."""
        alerts = []
        
        n = len(symbols)
        change_matrix = current - previous
        
        # Find pairs with biggest changes
        for i in range(n):
            for j in range(i + 1, n):
                change = change_matrix[i, j]
                
                if abs(change) > 0.30:  # 30% change threshold
                    alert = CorrelationAlert(
                        alert_type=AlertType.BREAKDOWN if change < 0 else AlertType.SPIKE,
                        severity='info',
                        message=f"Large correlation change: {symbols[i]}/{symbols[j]} {change:+.1%}",
                        details={
                            'pair': (symbols[i], symbols[j]),
                            'previous_corr': previous[i, j],
                            'current_corr': current[i, j],
                            'change': change
                        }
                    )
                    alerts.append(alert)
        
        return alerts[:3]  # Limit to top 3 pair alerts


# =============================================================================
# SECTOR ROTATION ENGINE
# =============================================================================

class SectorRotationEngine:
    """
    Generate sector rotation signals based on correlation analysis.
    
    Uses correlation clustering and momentum to identify
    rotation opportunities.
    """
    
    # Default sector mappings
    DEFAULT_SECTORS = {
        'SPY': 'Market',
        'QQQ': 'Technology',
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLE': 'Energy',
        'XLV': 'Healthcare',
        'XLI': 'Industrials',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLU': 'Utilities',
        'XLB': 'Materials',
        'XLRE': 'Real Estate',
        'XLC': 'Communications'
    }
    
    def __init__(
        self,
        sector_map: Dict[str, str] = None,
        rotation_threshold: float = 0.20
    ):
        self.sector_map = sector_map or self.DEFAULT_SECTORS
        self.rotation_threshold = rotation_threshold
        
        self.sector_momentum: Dict[str, float] = {}
        self.sector_correlations: Dict[str, Dict[str, float]] = {}
        self.rotation_history: List[SectorRotationSignal] = []
    
    def update_sector_mapping(self, symbols: List[str], sectors: Dict[str, str]):
        """Update sector mappings for symbols."""
        self.sector_map.update(sectors)
    
    def calculate_sector_correlations(
        self,
        returns: pd.DataFrame,
        symbols: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate average correlations between sectors."""
        # Map symbols to sectors
        symbol_sectors = {}
        for sym in symbols:
            symbol_sectors[sym] = self.sector_map.get(sym, 'Other')
        
        # Get unique sectors
        sectors = list(set(symbol_sectors.values()))
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Aggregate to sector level
        sector_corrs = {}
        for s1 in sectors:
            sector_corrs[s1] = {}
            s1_symbols = [s for s in symbols if symbol_sectors[s] == s1]
            
            for s2 in sectors:
                s2_symbols = [s for s in symbols if symbol_sectors[s] == s2]
                
                if s1 == s2:
                    sector_corrs[s1][s2] = 1.0
                else:
                    # Average pairwise correlation between sectors
                    corrs = []
                    for sym1 in s1_symbols:
                        for sym2 in s2_symbols:
                            if sym1 in corr_matrix.index and sym2 in corr_matrix.columns:
                                corrs.append(corr_matrix.loc[sym1, sym2])
                    
                    sector_corrs[s1][s2] = np.mean(corrs) if corrs else 0.0
        
        self.sector_correlations = sector_corrs
        return sector_corrs
    
    def calculate_sector_momentum(
        self,
        returns: pd.DataFrame,
        lookback: int = 20
    ) -> Dict[str, float]:
        """Calculate momentum for each sector."""
        # Map symbols to sectors
        symbol_sectors = {sym: self.sector_map.get(sym, 'Other') for sym in returns.columns}
        sectors = list(set(symbol_sectors.values()))
        
        momentum = {}
        for sector in sectors:
            sector_symbols = [s for s in returns.columns if symbol_sectors[s] == sector]
            if sector_symbols:
                sector_returns = returns[sector_symbols].mean(axis=1)
                momentum[sector] = sector_returns.iloc[-lookback:].sum()
            else:
                momentum[sector] = 0.0
        
        self.sector_momentum = momentum
        return momentum
    
    def generate_rotation_signals(
        self,
        returns: pd.DataFrame,
        correlation_matrix: np.ndarray,
        symbols: List[str]
    ) -> List[SectorRotationSignal]:
        """
        Generate sector rotation signals.
        
        Identifies opportunities to rotate from underperforming to
        outperforming sectors with low correlation.
        """
        signals = []
        
        # Calculate sector metrics
        sector_corrs = self.calculate_sector_correlations(returns, symbols)
        momentum = self.calculate_sector_momentum(returns)
        
        # Rank sectors by momentum
        momentum_rank = sorted(momentum.items(), key=lambda x: x[1], reverse=True)
        
        if len(momentum_rank) < 2:
            return signals
        
        # Find rotation opportunities
        top_sectors = [s for s, m in momentum_rank[:3] if m > 0]
        bottom_sectors = [s for s, m in momentum_rank[-3:] if m < 0]
        
        for bottom in bottom_sectors:
            for top in top_sectors:
                if bottom == top:
                    continue
                
                # Check correlation between sectors
                corr = sector_corrs.get(bottom, {}).get(top, 0.5)
                
                # Calculate signal strength
                momentum_diff = momentum[top] - momentum[bottom]
                decorr_bonus = max(0, 0.5 - corr)  # Bonus for low correlation
                
                strength = min(1.0, (abs(momentum_diff) * 10 + decorr_bonus) / 2)
                
                if strength > self.rotation_threshold:
                    signal = SectorRotationSignal(
                        from_sector=bottom,
                        to_sector=top,
                        strength=strength,
                        reason=f"Momentum divergence: {momentum_diff:.1%}, Corr: {corr:.2f}",
                        recommended_action=f"Reduce {bottom}, increase {top}"
                    )
                    signals.append(signal)
        
        # Sort by strength
        signals.sort(key=lambda x: x.strength, reverse=True)
        
        # Keep top signals
        self.rotation_history.extend(signals[:3])
        if len(self.rotation_history) > 100:
            self.rotation_history = self.rotation_history[-100:]
        
        return signals[:3]


# =============================================================================
# PORTFOLIO DIVERSIFICATION METRICS
# =============================================================================

class DiversificationAnalyzer:
    """
    Calculate portfolio diversification metrics.
    
    Metrics:
    - Diversification ratio (weighted avg vol / portfolio vol)
    - Effective number of assets
    - Correlation concentration
    """
    
    def __init__(self):
        pass
    
    def calculate_diversification_ratio(
        self,
        weights: np.ndarray,
        volatilities: np.ndarray,
        correlation_matrix: np.ndarray
    ) -> float:
        """
        Calculate diversification ratio.
        
        DR = (Σ w_i * σ_i) / σ_portfolio
        
        Higher is better (more diversification benefit).
        """
        # Weighted average volatility
        weighted_avg_vol = np.sum(weights * volatilities)
        
        # Portfolio volatility
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        portfolio_var = weights @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_var)
        
        if portfolio_vol == 0:
            return 1.0
        
        return weighted_avg_vol / portfolio_vol
    
    def calculate_effective_n_assets(
        self,
        weights: np.ndarray,
        correlation_matrix: np.ndarray
    ) -> float:
        """
        Calculate effective number of uncorrelated assets.
        
        Uses eigenvalue decomposition of correlation matrix.
        """
        # Get eigenvalues
        eigenvalues = np.linalg.eigvalsh(correlation_matrix)
        eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
        
        # Effective number based on eigenvalues
        total = np.sum(eigenvalues)
        if total == 0:
            return len(weights)
        
        normalized = eigenvalues / total
        # Entropy-based effective number
        nonzero = normalized[normalized > 1e-10]
        entropy = -np.sum(nonzero * np.log(nonzero))
        effective_n = np.exp(entropy)
        
        return min(effective_n, len(weights))
    
    def calculate_concentration(
        self,
        correlation_matrix: np.ndarray
    ) -> float:
        """
        Calculate correlation concentration (HHI-like measure).
        
        Returns value 0-1, higher means more concentrated correlations.
        """
        n = len(correlation_matrix)
        if n <= 1:
            return 0.0
        
        # Get off-diagonal correlations
        mask = ~np.eye(n, dtype=bool)
        corrs = np.abs(correlation_matrix[mask])
        
        if len(corrs) == 0:
            return 0.0
        
        # Calculate HHI-like concentration
        avg_corr = np.mean(corrs)
        return float(avg_corr)


# =============================================================================
# COMBINED CORRELATION ENGINE
# =============================================================================

class V28CorrelationEngine:
    """
    Combined cross-asset correlation analysis engine.
    
    Integrates:
    - Dynamic correlation tracking
    - Breakdown detection and alerts
    - Sector rotation signals
    - Diversification metrics
    """
    
    def __init__(
        self,
        lookback: int = 60,
        alert_threshold: float = 0.20,
        sector_map: Dict[str, str] = None
    ):
        self.matrix_tracker = DynamicCorrelationMatrix(lookback=lookback)
        self.breakdown_detector = CorrelationBreakdownDetector(spike_threshold=alert_threshold)
        self.rotation_engine = SectorRotationEngine(sector_map=sector_map)
        self.diversification_analyzer = DiversificationAnalyzer()
        
        self.current_state: Optional[CorrelationState] = None
        self.state_history: List[CorrelationState] = []
    
    def analyze(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray = None
    ) -> CorrelationState:
        """
        Perform full correlation analysis.
        
        Args:
            returns: DataFrame of asset returns
            weights: Optional portfolio weights
            
        Returns:
            CorrelationState with full analysis
        """
        symbols = list(returns.columns)
        n_assets = len(symbols)
        
        # Default equal weights if not provided
        if weights is None:
            weights = np.ones(n_assets) / n_assets
        
        # Update correlation matrix
        corr_matrix = self.matrix_tracker.update(returns)
        
        # Get previous matrix for comparison
        if len(self.matrix_tracker.correlation_history) >= 2:
            prev_matrix = self.matrix_tracker.correlation_history[-2]
        else:
            prev_matrix = corr_matrix
        
        # Calculate average correlation
        mask = ~np.eye(n_assets, dtype=bool)
        avg_corr = np.mean(corr_matrix[mask])
        
        # Classify regime
        regime, percentile = self.matrix_tracker.classify_regime(avg_corr)
        
        # Check for alerts
        alerts = self.breakdown_detector.check_for_alerts(
            corr_matrix, prev_matrix, regime, symbols
        )
        
        # Generate rotation signals
        rotation_signals = self.rotation_engine.generate_rotation_signals(
            returns, corr_matrix, symbols
        )
        
        # Calculate diversification metrics
        volatilities = returns.std() * np.sqrt(252)
        div_ratio = self.diversification_analyzer.calculate_diversification_ratio(
            weights, volatilities.values, corr_matrix
        )
        effective_n = self.diversification_analyzer.calculate_effective_n_assets(
            weights, corr_matrix
        )
        
        # Get highest and lowest correlation pairs
        highest_pairs = []
        lowest_pairs = []
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                highest_pairs.append((symbols[i], symbols[j], corr_matrix[i, j]))
                lowest_pairs.append((symbols[i], symbols[j], corr_matrix[i, j]))
        
        highest_pairs.sort(key=lambda x: x[2], reverse=True)
        lowest_pairs.sort(key=lambda x: x[2])
        
        # Create state
        state = CorrelationState(
            regime=regime,
            average_correlation=float(avg_corr),
            correlation_matrix=corr_matrix,
            sector_correlations=self.rotation_engine.sector_correlations,
            diversification_ratio=float(div_ratio),
            effective_n_assets=float(effective_n),
            highest_pairs=highest_pairs[:10],
            lowest_pairs=lowest_pairs[:10],
            regime_percentile=float(percentile),
            rolling_volatility=float(volatilities.mean()),
            alerts=alerts,
            rotation_signals=rotation_signals
        )
        
        # Update history
        self.current_state = state
        self.state_history.append(state)
        
        if len(self.state_history) > 252:
            self.state_history = self.state_history[-252:]
        
        return state
    
    def get_correlation_adjusted_weights(
        self,
        target_weights: np.ndarray,
        max_correlation: float = 0.80
    ) -> np.ndarray:
        """
        Adjust weights to reduce highly correlated positions.
        
        Args:
            target_weights: Initial target weights
            max_correlation: Maximum allowed correlation
            
        Returns:
            Adjusted weights
        """
        if self.current_state is None:
            return target_weights
        
        corr_matrix = self.current_state.correlation_matrix
        n = len(target_weights)
        adjusted = target_weights.copy()
        
        # Find highly correlated pairs
        for i in range(n):
            for j in range(i + 1, n):
                if corr_matrix[i, j] > max_correlation:
                    # Reduce the smaller position
                    reduction_factor = (corr_matrix[i, j] - max_correlation) / (1 - max_correlation)
                    reduction_factor = min(0.5, reduction_factor)  # Max 50% reduction
                    
                    if adjusted[i] < adjusted[j]:
                        adjusted[i] *= (1 - reduction_factor)
                    else:
                        adjusted[j] *= (1 - reduction_factor)
        
        # Renormalize
        total = np.sum(adjusted)
        if total > 0:
            adjusted = adjusted / total
        
        return adjusted
    
    def should_reduce_exposure(self) -> bool:
        """Check if overall exposure should be reduced due to high correlations."""
        if self.current_state is None:
            return False
        
        return self.current_state.regime in [CorrelationRegime.ELEVATED, CorrelationRegime.CRISIS]
    
    def get_exposure_multiplier(self) -> float:
        """Get exposure multiplier based on correlation regime."""
        if self.current_state is None:
            return 1.0
        
        multipliers = {
            CorrelationRegime.LOW: 1.2,
            CorrelationRegime.NORMAL: 1.0,
            CorrelationRegime.ELEVATED: 0.7,
            CorrelationRegime.CRISIS: 0.4
        }
        
        return multipliers.get(self.current_state.regime, 1.0)
    
    def register_alert_callback(self, callback: Callable[[CorrelationAlert], None]):
        """Register callback for correlation alerts."""
        self.breakdown_detector.register_alert_callback(callback)


# =============================================================================
# MAIN / DEMO
# =============================================================================

def demo():
    """Demo the correlation engine."""
    np.random.seed(42)
    
    logger.info("📊 V28 Correlation Engine Demo")
    logger.info("=" * 50)
    
    # Create synthetic data
    n_days = 252
    n_assets = 10
    symbols = ['SPY', 'QQQ', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU']
    
    # Generate correlated returns
    base_market = np.random.randn(n_days) * 0.01
    
    returns_data = {}
    for i, sym in enumerate(symbols):
        # Each asset has market exposure + idiosyncratic
        market_beta = 0.5 + np.random.rand() * 0.5
        idio = np.random.randn(n_days) * 0.015
        returns_data[sym] = market_beta * base_market + idio
    
    returns = pd.DataFrame(returns_data)
    
    # Initialize engine
    engine = V28CorrelationEngine()
    
    # Register alert callback
    def alert_handler(alert: CorrelationAlert):
        logger.info(f"🔔 Alert: {alert.message}")
    
    engine.register_alert_callback(alert_handler)
    
    # Analyze
    state = engine.analyze(returns)
    
    # Print results
    logger.info(f"\n📈 Correlation Analysis Results:")
    logger.info(f"   Regime: {state.regime.value.upper()}")
    logger.info(f"   Average Correlation: {state.average_correlation:.3f}")
    logger.info(f"   Regime Percentile: {state.regime_percentile:.1f}%")
    logger.info(f"   Diversification Ratio: {state.diversification_ratio:.2f}")
    logger.info(f"   Effective # Assets: {state.effective_n_assets:.1f}")
    
    logger.info(f"\n📊 Highest Correlated Pairs:")
    for sym1, sym2, corr in state.highest_pairs[:3]:
        logger.info(f"   {sym1}/{sym2}: {corr:.3f}")
    
    logger.info(f"\n📊 Lowest Correlated Pairs:")
    for sym1, sym2, corr in state.lowest_pairs[:3]:
        logger.info(f"   {sym1}/{sym2}: {corr:.3f}")
    
    if state.rotation_signals:
        logger.info(f"\n🔄 Sector Rotation Signals:")
        for signal in state.rotation_signals[:2]:
            logger.info(f"   {signal.from_sector} → {signal.to_sector} (strength: {signal.strength:.2f})")
    
    logger.info(f"\n💹 Exposure Multiplier: {engine.get_exposure_multiplier():.2f}")


if __name__ == '__main__':
    demo()

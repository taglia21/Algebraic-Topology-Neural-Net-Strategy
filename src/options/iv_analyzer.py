"""
IV (Implied Volatility) Analyzer
=================================

Track and analyze implied volatility to identify premium selling opportunities.

Key Metrics:
- IV Rank: Current IV vs 52-week range (0-100)
- IV Percentile: % of days where IV was below current level
- HV/IV Ratio: Realized vs implied volatility comparison
- IV Regime: Classification (low/normal/high/extreme)
"""

import json
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Deque
from pathlib import Path
import logging

from .theta_decay_engine import IVRegime
from .utils.constants import IV_LOOKBACK_DAYS, IV_HIGH_THRESHOLD, IV_LOW_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class IVMetrics:
    """IV analysis metrics for a symbol."""
    symbol: str
    current_iv: float
    iv_rank: float  # 0-100
    iv_percentile: float  # 0-100
    hv_iv_ratio: Optional[float]  # HV/IV ratio
    regime: IVRegime
    min_iv_52w: float
    max_iv_52w: float
    avg_iv_52w: float
    std_iv_52w: float
    days_of_data: int
    timestamp: datetime


class IVAnalyzer:
    """
    Implied Volatility Analyzer.
    
    Tracks IV history and calculates rank/percentile metrics to identify
    when premium is expensive (sell) or cheap (buy).
    
    Usage:
        analyzer = IVAnalyzer()
        analyzer.update('SPY', 0.18)
        metrics = analyzer.analyze('SPY', current_iv=0.20, historical_vol=0.15)
    """
    
    def __init__(
        self,
        lookback_days: int = IV_LOOKBACK_DAYS,
        state_file: Optional[Path] = None
    ):
        """
        Initialize IV analyzer.
        
        Args:
            lookback_days: Number of days of IV history to maintain
            state_file: Path to save/load IV history (for persistence)
        """
        self.lookback_days = lookback_days
        self.state_file = state_file or Path("iv_history.json")
        
        # Rolling window of IV observations: symbol -> deque of (timestamp, iv)
        self.history: Dict[str, Deque[Tuple[datetime, float]]] = {}
        
        # Load persisted history if exists
        if self.state_file.exists():
            self._load_history()
        
        logger.info(f"IV Analyzer initialized (lookback={lookback_days} days)")
    
    def update(self, symbol: str, iv: float, timestamp: Optional[datetime] = None) -> None:
        """
        Add new IV observation to rolling window.
        
        Args:
            symbol: Stock symbol
            iv: Implied volatility (annualized, e.g., 0.20 for 20%)
            timestamp: Observation time (defaults to now)
        """
        if iv <= 0:
            logger.warning(f"Invalid IV {iv} for {symbol}, skipping")
            return
        
        timestamp = timestamp or datetime.now()
        
        # Initialize deque for symbol if needed
        if symbol not in self.history:
            self.history[symbol] = deque(maxlen=self.lookback_days)
        
        # Add observation
        self.history[symbol].append((timestamp, iv))
        
        logger.debug(f"Updated {symbol} IV: {iv:.4f} ({len(self.history[symbol])} days)")
    
    def get_iv_rank(self, symbol: str, current_iv: float) -> float:
        """
        Calculate IV Rank: (Current - Min) / (Max - Min) * 100
        
        IV Rank shows where current IV sits in the 52-week range.
        
        Args:
            symbol: Stock symbol
            current_iv: Current implied volatility
            
        Returns:
            IV Rank (0-100), or 50 if insufficient data
        """
        if symbol not in self.history or len(self.history[symbol]) < 20:
            logger.warning(f"Insufficient data for {symbol} IV rank (need 20 days)")
            return 50.0  # Default to neutral
        
        ivs = [iv for _, iv in self.history[symbol]]
        min_iv = min(ivs)
        max_iv = max(ivs)
        
        if max_iv == min_iv:
            return 50.0  # No range = neutral
        
        iv_rank = ((current_iv - min_iv) / (max_iv - min_iv)) * 100
        return float(np.clip(iv_rank, 0, 100))
    
    def get_iv_percentile(self, symbol: str, current_iv: float) -> float:
        """
        Calculate IV Percentile: % of days where IV was below current level.
        
        More robust than IV Rank (not affected by outliers).
        
        Args:
            symbol: Stock symbol
            current_iv: Current implied volatility
            
        Returns:
            IV Percentile (0-100), or 50 if insufficient data
        """
        if symbol not in self.history or len(self.history[symbol]) < 20:
            logger.warning(f"Insufficient data for {symbol} IV percentile")
            return 50.0
        
        ivs = np.array([iv for _, iv in self.history[symbol]])
        below_current = np.sum(ivs < current_iv)
        percentile = (below_current / len(ivs)) * 100
        
        return float(percentile)
    
    def calculate_hv_iv_ratio(
        self,
        historical_vol: float,
        implied_vol: float
    ) -> float:
        """
        Calculate HV/IV ratio to identify mispricings.
        
        HV/IV > 1.0 = IV underpriced (options cheap, consider buying)
        HV/IV < 1.0 = IV overpriced (options expensive, sell premium)
        
        Args:
            historical_vol: Realized historical volatility
            implied_vol: Current implied volatility
            
        Returns:
            HV/IV ratio
        """
        if implied_vol <= 0:
            logger.warning(f"Invalid IV {implied_vol} for HV/IV ratio")
            return 1.0
        
        return historical_vol / implied_vol
    
    def detect_iv_regime(self, symbol: str, current_iv: float) -> IVRegime:
        """
        Classify current IV environment.
        
        Args:
            symbol: Stock symbol
            current_iv: Current implied volatility
            
        Returns:
            IVRegime (LOW, NORMAL, HIGH, EXTREME)
        """
        iv_rank = self.get_iv_rank(symbol, current_iv)
        
        if iv_rank > 90:
            return IVRegime.EXTREME
        elif iv_rank > IV_HIGH_THRESHOLD:
            return IVRegime.HIGH
        elif iv_rank < IV_LOW_THRESHOLD:
            return IVRegime.LOW
        else:
            return IVRegime.NORMAL
    
    def analyze(
        self,
        symbol: str,
        current_iv: float,
        historical_vol: Optional[float] = None
    ) -> IVMetrics:
        """
        Perform complete IV analysis.
        
        Args:
            symbol: Stock symbol
            current_iv: Current implied volatility
            historical_vol: Realized historical volatility (optional)
            
        Returns:
            IVMetrics with complete analysis
        """
        # Update history with current IV
        self.update(symbol, current_iv)
        
        # Calculate metrics
        iv_rank = self.get_iv_rank(symbol, current_iv)
        iv_percentile = self.get_iv_percentile(symbol, current_iv)
        regime = self.detect_iv_regime(symbol, current_iv)
        
        # HV/IV ratio if HV provided
        hv_iv_ratio = None
        if historical_vol is not None and historical_vol > 0:
            hv_iv_ratio = self.calculate_hv_iv_ratio(historical_vol, current_iv)
        
        # Historical statistics
        if symbol in self.history and len(self.history[symbol]) > 0:
            ivs = np.array([iv for _, iv in self.history[symbol]])
            min_iv = float(ivs.min())
            max_iv = float(ivs.max())
            avg_iv = float(ivs.mean())
            std_iv = float(ivs.std())
            days_of_data = len(ivs)
        else:
            min_iv = max_iv = avg_iv = current_iv
            std_iv = 0.0
            days_of_data = 0
        
        metrics = IVMetrics(
            symbol=symbol,
            current_iv=current_iv,
            iv_rank=iv_rank,
            iv_percentile=iv_percentile,
            hv_iv_ratio=hv_iv_ratio,
            regime=regime,
            min_iv_52w=min_iv,
            max_iv_52w=max_iv,
            avg_iv_52w=avg_iv,
            std_iv_52w=std_iv,
            days_of_data=days_of_data,
            timestamp=datetime.now()
        )
        
        logger.info(
            f"{symbol} IV Analysis: IV={current_iv:.2%}, Rank={iv_rank:.0f}, "
            f"Percentile={iv_percentile:.0f}, Regime={regime.value}"
        )
        
        return metrics
    
    def should_sell_premium(
        self,
        symbol: str,
        current_iv: float,
        min_iv_rank: float = 50.0
    ) -> Tuple[bool, str]:
        """
        Determine if conditions are favorable for selling premium.
        
        Args:
            symbol: Stock symbol
            current_iv: Current implied volatility
            min_iv_rank: Minimum IV rank to sell (default 50)
            
        Returns:
            (should_sell: bool, reason: str)
        """
        metrics = self.analyze(symbol, current_iv)
        
        # Check IV rank
        if metrics.iv_rank < min_iv_rank:
            return False, f"IV Rank too low ({metrics.iv_rank:.0f} < {min_iv_rank})"
        
        # Check IV regime
        if metrics.regime == IVRegime.LOW:
            return False, f"IV regime is LOW (rank={metrics.iv_rank:.0f})"
        
        # Check HV/IV if available
        if metrics.hv_iv_ratio is not None:
            if metrics.hv_iv_ratio > 1.2:
                return False, f"IV underpriced (HV/IV={metrics.hv_iv_ratio:.2f})"
        
        # Favorable conditions
        reason = f"IV Rank {metrics.iv_rank:.0f}, Regime {metrics.regime.value}"
        if metrics.hv_iv_ratio is not None:
            reason += f", HV/IV {metrics.hv_iv_ratio:.2f}"
        
        return True, reason
    
    def get_statistics(self, symbol: str) -> Dict:
        """
        Get statistical summary of IV history.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with statistics
        """
        if symbol not in self.history or len(self.history[symbol]) == 0:
            return {'error': 'No data available'}
        
        ivs = np.array([iv for _, iv in self.history[symbol]])
        
        return {
            'symbol': symbol,
            'days_of_data': len(ivs),
            'current_iv': float(ivs[-1]),
            'min_iv': float(ivs.min()),
            'max_iv': float(ivs.max()),
            'mean_iv': float(ivs.mean()),
            'median_iv': float(np.median(ivs)),
            'std_iv': float(ivs.std()),
            'percentiles': {
                '10th': float(np.percentile(ivs, 10)),
                '25th': float(np.percentile(ivs, 25)),
                '50th': float(np.percentile(ivs, 50)),
                '75th': float(np.percentile(ivs, 75)),
                '90th': float(np.percentile(ivs, 90)),
            }
        }
    
    def persist_history(self) -> None:
        """Save IV history to disk for persistence across sessions."""
        try:
            # Convert deques to lists for JSON serialization
            data = {}
            for symbol, history in self.history.items():
                data[symbol] = [
                    {'timestamp': ts.isoformat(), 'iv': iv}
                    for ts, iv in history
                ]
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved IV history for {len(data)} symbols to {self.state_file}")
            
        except Exception as e:
            logger.error(f"Error persisting IV history: {e}")
    
    def _load_history(self) -> None:
        """Load IV history from disk."""
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            
            for symbol, history_list in data.items():
                self.history[symbol] = deque(maxlen=self.lookback_days)
                for entry in history_list:
                    ts = datetime.fromisoformat(entry['timestamp'])
                    iv = entry['iv']
                    self.history[symbol].append((ts, iv))
            
            logger.info(f"Loaded IV history for {len(data)} symbols from {self.state_file}")
            
        except Exception as e:
            logger.error(f"Error loading IV history: {e}")
    
    def cleanup_old_data(self, days_to_keep: Optional[int] = None) -> int:
        """
        Remove IV data older than specified days.
        
        Args:
            days_to_keep: Number of days to retain (default: lookback_days)
            
        Returns:
            Number of observations removed
        """
        days_to_keep = days_to_keep or self.lookback_days
        cutoff = datetime.now() - timedelta(days=days_to_keep)
        removed = 0
        
        for symbol in self.history:
            original_len = len(self.history[symbol])
            # Filter to keep only recent data
            self.history[symbol] = deque(
                [(ts, iv) for ts, iv in self.history[symbol] if ts >= cutoff],
                maxlen=self.lookback_days
            )
            removed += original_len - len(self.history[symbol])
        
        if removed > 0:
            logger.info(f"Cleaned up {removed} old IV observations")
        
        return removed

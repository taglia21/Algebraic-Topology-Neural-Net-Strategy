import numpy as np
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class GlobalRiskManager:
    """Manages risk across all active strategies and asset classes."""
    def __init__(self, max_total_drawdown: float = 0.15, 
                 correlation_threshold: float = 0.7):
        self.max_total_drawdown = max_total_drawdown
        self.correlation_threshold = correlation_threshold
        self.strategy_returns = {} # symbol -> list of returns
        self.peak_value = 0.0
        self.current_value = 0.0

    def update_metrics(self, total_equity: float):
        self.current_value = total_equity
        if total_equity > self.peak_value:
            self.peak_value = total_equity
            
    def check_risk_limits(self) -> bool:
        """Returns True if within limits, False if breach occurred."""
        if self.peak_value == 0: return True
        
        drawdown = (self.peak_value - self.current_value) / self.peak_value
        if drawdown > self.max_total_drawdown:
            logger.error(f"GLOBAL RISK BREACH: Drawdown {drawdown:.2%} exceeds limit {self.max_total_drawdown:.2%}")
            return False
        return True

    def calculate_correlation_adjustment(self, returns_matrix: np.ndarray) -> np.ndarray:
        """Adjusts position sizes based on cross-asset correlation."""
        if returns_matrix.shape[1] < 2:
            return np.ones(returns_matrix.shape[1])
            
        corr = np.corrcoef(returns_matrix.T)
        avg_corr = np.mean(corr[np.triu_indices(corr.shape[0], k=1)])
        
        # If correlation is high, scale down all positions
        if avg_corr > self.correlation_threshold:
            adjustment = 1.0 - (avg_corr - self.correlation_threshold) / (1.0 - self.correlation_threshold)
            return np.full(returns_matrix.shape[1], max(0.2, adjustment))
            
        return np.ones(returns_matrix.shape[1])


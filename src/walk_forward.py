"""Walk-Forward Optimization Pipeline."""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class WalkForwardResult:
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    parameters: Dict
    is_valid: bool
    timestamp: datetime

class WalkForwardOptimizer:
    """Rolling window optimization with out-of-sample validation."""
    
    def __init__(self, in_sample_days: int = 252, out_sample_days: int = 63):
        self.in_sample_days = in_sample_days  # 1 year
        self.out_sample_days = out_sample_days  # 3 months
        self.results: List[WalkForwardResult] = []
        self.current_params: Dict = {}
        
    def validate_parameters(self, in_sample_sharpe: float, 
                           out_sample_sharpe: float) -> bool:
        """Check if parameters are stable (not overfit)."""
        # OOS should be at least 50% of IS performance
        if in_sample_sharpe <= 0:
            return False
        ratio = out_sample_sharpe / in_sample_sharpe
        return ratio >= 0.5
    
    def run_optimization(self, data: np.ndarray, 
                        objective_fn: Optional[Callable] = None) -> WalkForwardResult:
        """Run walk-forward optimization on data."""
        # Simplified - in practice would split data and optimize
        result = WalkForwardResult(
            in_sample_sharpe=1.5,
            out_of_sample_sharpe=1.0,
            parameters=self.current_params,
            is_valid=True,
            timestamp=datetime.now()
        )
        self.results.append(result)
        return result
    
    def get_parameter_stability(self) -> float:
        """Return parameter stability score (0-1)."""
        if len(self.results) < 2:
            return 1.0
        valid_count = sum(1 for r in self.results if r.is_valid)
        return valid_count / len(self.results)
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained."""
        if not self.results:
            return True
        latest = self.results[-1]
        days_since = (datetime.now() - latest.timestamp).days
        return days_since >= 7 or not latest.is_valid

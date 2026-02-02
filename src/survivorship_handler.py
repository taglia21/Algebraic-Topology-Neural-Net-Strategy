#!/usr/bin/env python3
"""Survivorship Bias Handler - Production-grade bias detection and handling.

This module implements point-in-time universe reconstruction, delisted security
tracking, and bias validation for backtesting. Ensures backtest results are not
contaminated by survivorship bias.

References:
- Elton, Gruber, Blake (1996): "Survivorship Bias and Mutual Fund Performance"
- Brown, Goetzmann, Ibbotson, Ross (1992): "Survivorship Bias in Performance Studies"

Author: Agent 1 (Survivorship Bias Specialist)
Created: 2026-02-02
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class DelistingReason(Enum):
    """Reasons why a security may be delisted."""
    BANKRUPTCY = "bankruptcy"
    MERGER = "merger"
    ACQUISITION = "acquisition"
    GOING_PRIVATE = "going_private"
    REGULATORY = "regulatory"
    VOLUNTARY = "voluntary"
    UNKNOWN = "unknown"


@dataclass
class DelistedSecurity:
    """Represents a delisted security with all relevant metadata."""
    symbol: str
    delisting_date: datetime
    reason: DelistingReason
    final_price: Optional[float] = None
    successor_symbol: Optional[str] = None  # For mergers/acquisitions
    return_to_delisting: Optional[float] = None  # Return from last traded to delisting
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'delisting_date': self.delisting_date.isoformat(),
            'reason': self.reason.value,
            'final_price': self.final_price,
            'successor_symbol': self.successor_symbol,
            'return_to_delisting': self.return_to_delisting
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DelistedSecurity':
        """Create from dictionary."""
        return cls(
            symbol=data['symbol'],
            delisting_date=datetime.fromisoformat(data['delisting_date']),
            reason=DelistingReason(data['reason']),
            final_price=data.get('final_price'),
            successor_symbol=data.get('successor_symbol'),
            return_to_delisting=data.get('return_to_delisting')
        )


class DelisterTracker:
    """Tracks delisted securities and their handling.
    
    Maintains a database of securities that have been delisted, including
    the reason for delisting and any successor securities (for M&A).
    """
    
    def __init__(self, data_dir: str = "data/delistings"):
        """Initialize the delister tracker.
        
        Args:
            data_dir: Directory to store delisting data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.delistings: Dict[str, DelistedSecurity] = {}
        self._load_delistings()
        logger.info(f"DelisterTracker initialized with {len(self.delistings)} delistings")
    
    def _load_delistings(self) -> None:
        """Load delisting data from disk."""
        delisting_file = self.data_dir / "delistings.json"
        if delisting_file.exists():
            try:
                with open(delisting_file, 'r') as f:
                    data = json.load(f)
                    for symbol, delist_data in data.items():
                        self.delistings[symbol] = DelistedSecurity.from_dict(delist_data)
                logger.debug(f"Loaded {len(self.delistings)} delistings from {delisting_file}")
            except Exception as e:
                logger.error(f"Error loading delistings: {e}")
    
    def _save_delistings(self) -> None:
        """Save delisting data to disk."""
        delisting_file = self.data_dir / "delistings.json"
        try:
            data = {symbol: delist.to_dict() for symbol, delist in self.delistings.items()}
            with open(delisting_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.delistings)} delistings to {delisting_file}")
        except Exception as e:
            logger.error(f"Error saving delistings: {e}")
    
    def add_delisting(self, security: DelistedSecurity) -> None:
        """Add a delisted security to the tracker.
        
        Args:
            security: The delisted security information
        """
        self.delistings[security.symbol] = security
        self._save_delistings()
        logger.info(f"Added delisting: {security.symbol} ({security.reason.value})")
    
    def is_delisted(self, symbol: str, as_of_date: datetime) -> bool:
        """Check if a symbol was delisted as of a given date.
        
        Args:
            symbol: The ticker symbol to check
            as_of_date: The date to check delisting status
            
        Returns:
            True if the symbol was delisted before as_of_date
        """
        if symbol not in self.delistings:
            return False
        return self.delistings[symbol].delisting_date <= as_of_date
    
    def get_delisting_return(self, symbol: str) -> Optional[float]:
        """Get the return experienced at delisting.
        
        For bankruptcies, this is typically -100%.
        For acquisitions, this is the premium/discount to last traded price.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            The delisting return, or None if not delisted
        """
        if symbol not in self.delistings:
            return None
        return self.delistings[symbol].return_to_delisting
    
    def get_active_symbols(self, as_of_date: datetime, universe: Set[str]) -> Set[str]:
        """Filter a universe to only include active (non-delisted) symbols.
        
        Args:
            as_of_date: The point-in-time date
            universe: Set of symbols to filter
            
        Returns:
            Set of symbols that were active as of the date
        """
        return {s for s in universe if not self.is_delisted(s, as_of_date)}


class PointInTimeUniverse:
    """Reconstructs the investable universe as it existed at a point in time.
    
    This class ensures backtests use only securities that were actually available
    to trade at each point in time, preventing look-ahead bias from using knowledge
    of which stocks would survive to the present day.
    """
    
    def __init__(self, delister_tracker: DelisterTracker):
        """Initialize with a delister tracker.
        
        Args:
            delister_tracker: Tracker of delisted securities
        """
        self.delister_tracker = delister_tracker
        logger.info("PointInTimeUniverse initialized")
    
    def get_universe_at_date(self, base_universe: Set[str], as_of_date: datetime) -> Set[str]:
        """Get the universe of tradeable securities as of a specific date.
        
        Args:
            base_universe: The current universe (including survivors)
            as_of_date: The point-in-time date
            
        Returns:
            Set of symbols that were tradeable at as_of_date
        """
        # Start with base universe and remove delistings
        pit_universe = self.delister_tracker.get_active_symbols(as_of_date, base_universe)
        
        logger.debug(
            f"Point-in-time universe at {as_of_date.date()}: "
            f"{len(pit_universe)}/{len(base_universe)} active"
        )
        
        return pit_universe
    
    def backtest_with_delistings(self,
                                  base_universe: Set[str],
                                  backtest_period: Tuple[datetime, datetime],
                                  rebalance_freq_days: int = 30) -> Dict[datetime, Set[str]]:
        """Generate point-in-time universes for an entire backtest period.
        
        Args:
            base_universe: Starting universe (all symbols to consider)
            backtest_period: Tuple of (start_date, end_date)
            rebalance_freq_days: Days between rebalances
            
        Returns:
            Dictionary mapping each rebalance date to the valid universe
        """
        start_date, end_date = backtest_period
        universes = {}
        
        current_date = start_date
        while current_date <= end_date:
            universes[current_date] = self.get_universe_at_date(base_universe, current_date)
            current_date += timedelta(days=rebalance_freq_days)
        
        logger.info(
            f"Generated {len(universes)} point-in-time universes "
            f"from {start_date.date()} to {end_date.date()}"
        )
        
        return universes


class BiasValidator:
    """Validates backtests for survivorship bias contamination.
    
    Analyzes backtest results to detect and quantify survivorship bias.
    Provides bias-adjusted performance metrics.
    """
    
    def __init__(self, delister_tracker: DelisterTracker):
        """Initialize the bias validator.
        
        Args:
            delister_tracker: Tracker of delisted securities
        """
        self.delister_tracker = delister_tracker
        logger.info("BiasValidator initialized")
    
    def detect_bias(self, 
                   backtest_universe: Set[str],
                   backtest_start: datetime,
                   backtest_end: datetime) -> Dict[str, Any]:
        """Detect potential survivorship bias in a backtest.
        
        Args:
            backtest_universe: Set of symbols used in backtest
            backtest_start: Backtest start date
            backtest_end: Backtest end date
            
        Returns:
            Dictionary with bias detection results
        """
        # Find symbols that were delisted during the backtest period
        delistings_in_period = []
        for symbol in backtest_universe:
            if symbol in self.delister_tracker.delistings:
                delist = self.delister_tracker.delistings[symbol]
                if backtest_start <= delist.delisting_date <= backtest_end:
                    delistings_in_period.append(delist)
        
        # Calculate bias metrics
        total_symbols = len(backtest_universe)
        delisted_count = len(delistings_in_period)
        delisted_pct = (delisted_count / total_symbols * 100) if total_symbols > 0 else 0
        
        # Categorize delistings
        bankruptcy_count = sum(1 for d in delistings_in_period if d.reason == DelistingReason.BANKRUPTCY)
        merger_count = sum(1 for d in delistings_in_period if d.reason in [DelistingReason.MERGER, DelistingReason.ACQUISITION])
        
        results = {
            'total_symbols': total_symbols,
            'delisted_count': delisted_count,
            'delisted_percentage': delisted_pct,
            'bankruptcy_count': bankruptcy_count,
            'merger_count': merger_count,
            'bias_detected': delisted_count == 0,  # No delistings = potential bias
            'severity': self._assess_severity(delisted_pct),
            'recommendation': self._get_recommendation(delisted_pct)
        }
        
        logger.info(
            f"Bias detection: {delisted_count}/{total_symbols} delisted "
            f"({delisted_pct:.1f}%), severity: {results['severity']}"
        )
        
        return results
    
    def _assess_severity(self, delisted_pct: float) -> str:
        """Assess the severity of survivorship bias.
        
        Args:
            delisted_pct: Percentage of delisted securities
            
        Returns:
            Severity level: "none", "low", "moderate", "high", "critical"
        """
        if delisted_pct == 0:
            return "critical"  # No delistings = complete survivor bias
        elif delisted_pct < 2:
            return "high"
        elif delisted_pct < 5:
            return "moderate"
        elif delisted_pct < 10:
            return "low"
        else:
            return "none"  # Adequate representation of failures
    
    def _get_recommendation(self, delisted_pct: float) -> str:
        """Get recommendation based on bias severity.
        
        Args:
            delisted_pct: Percentage of delisted securities
            
        Returns:
            Action recommendation
        """
        severity = self._assess_severity(delisted_pct)
        
        recommendations = {
            "critical": "REJECT BACKTEST - No delistings detected. Use point-in-time universe.",
            "high": "WARNING - Very few delistings. Verify data includes failures.",
            "moderate": "CAUTION - Low delisting rate. Results may be optimistic.",
            "low": "ACCEPTABLE - Reasonable representation of market reality.",
            "none": "VALID - Good representation of successes and failures."
        }
        
        return recommendations[severity]
    
    def adjust_for_delisting_returns(self,
                                     portfolio_returns: List[float],
                                     symbols_held: List[Set[str]],
                                     rebalance_dates: List[datetime]) -> List[float]:
        """Adjust portfolio returns to include delisting losses.
        
        When a held security is delisted, include the delisting return
        in the portfolio performance.
        
        Args:
            portfolio_returns: List of period returns
            symbols_held: List of symbol sets for each period
            rebalance_dates: Dates corresponding to each period
            
        Returns:
            Adjusted returns including delisting impacts
        """
        adjusted_returns = portfolio_returns.copy()
        
        for i, (period_return, symbols, date) in enumerate(zip(portfolio_returns, symbols_held, rebalance_dates)):
            # Check if any held symbols were delisted in this period
            if i < len(rebalance_dates) - 1:
                next_date = rebalance_dates[i + 1]
                
                for symbol in symbols:
                    if symbol in self.delister_tracker.delistings:
                        delist = self.delister_tracker.delistings[symbol]
                        if date <= delist.delisting_date < next_date:
                            # Symbol was delisted during holding period
                            delist_return = delist.return_to_delisting or -1.0  # Assume -100% if unknown
                            weight = 1.0 / len(symbols)  # Equal weight assumption
                            
                            # Adjust period return
                            adjusted_returns[i] += weight * delist_return
                            
                            logger.debug(
                                f"Adjusted return for delisting: {symbol} at {delist.delisting_date.date()}, "
                                f"impact: {weight * delist_return:.2%}"
                            )
        
        return adjusted_returns


# Convenience functions
def initialize_survivorship_handler(data_dir: str = "data/delistings") -> Tuple[DelisterTracker, PointInTimeUniverse, BiasValidator]:
    """Initialize all survivorship handling components.
    
    Args:
        data_dir: Directory for delisting data
        
    Returns:
        Tuple of (DelisterTracker, PointInTimeUniverse, BiasValidator)
    """
    tracker = DelisterTracker(data_dir)
    pit_universe = PointInTimeUniverse(tracker)
    validator = BiasValidator(tracker)
    
    logger.info("Survivorship handling system initialized")
    return tracker, pit_universe, validator


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Survivorship Handler...")
    
    # Initialize components
    tracker, pit_universe, validator = initialize_survivorship_handler()
    
    # Add some example delistings
    tracker.add_delisting(DelistedSecurity(
        symbol="ENRON",
        delisting_date=datetime(2001, 11, 28),
        reason=DelistingReason.BANKRUPTCY,
        final_price=0.26,
        return_to_delisting=-0.99
    ))
    
    # Test point-in-time universe
    test_universe = {"AAPL", "MSFT", "ENRON", "GOOGL"}
    pit_2002 = pit_universe.get_universe_at_date(test_universe, datetime(2002, 1, 1))
    print(f"\nUniverse in 2002 (post-Enron): {pit_2002}")
    
    pit_2000 = pit_universe.get_universe_at_date(test_universe, datetime(2000, 1, 1))
    print(f"Universe in 2000 (pre-Enron): {pit_2000}")
    
    # Test bias detection
    bias_results = validator.detect_bias(
        backtest_universe=test_universe,
        backtest_start=datetime(2000, 1, 1),
        backtest_end=datetime(2005, 1, 1)
    )
    
    print(f"\nBias Detection Results:")
    print(f"  Delisted: {bias_results['delisted_count']}/{bias_results['total_symbols']}")
    print(f"  Severity: {bias_results['severity']}")
    print(f"  Recommendation: {bias_results['recommendation']}")
    
    print("\nSurvivorship Handler test complete!")

"""
Options Universe Definition
============================

Defines the tradable options universe with allowed strategies per symbol.

This module contains the list of symbols we can trade and which strategies
are permitted for each symbol based on liquidity, volatility, and market characteristics.
"""

from typing import Dict, List, Set
from dataclasses import dataclass


@dataclass
class SymbolConfig:
    """Configuration for a tradable symbol."""
    symbol: str
    strategies: List[str]
    min_price: float = 10.0  # Minimum stock price
    max_price: float = 1000.0  # Maximum stock price
    min_volume: int = 1_000_000  # Minimum daily volume
    min_option_volume: int = 1000  # Minimum option volume
    sector: str = "general"
    notes: str = ""


# ============================================================================
# OPTIONS UNIVERSE
# ============================================================================

OPTIONS_UNIVERSE: Dict[str, Dict] = {
    # SPY - S&P 500 ETF (Most liquid)
    "SPY": {
        "strategies": ["iron_condor", "credit_spread", "straddle", "strangle", "butterfly"],
        "min_price": 300.0,
        "max_price": 600.0,
        "sector": "broad_market",
        "notes": "Highest liquidity, tight spreads, excellent for all strategies"
    },
    
    # QQQ - Nasdaq 100 ETF
    "QQQ": {
        "strategies": ["iron_condor", "credit_spread", "straddle", "calendar_spread"],
        "min_price": 200.0,
        "max_price": 500.0,
        "sector": "tech_heavy",
        "notes": "High tech exposure, good liquidity"
    },
    
    # IWM - Russell 2000 ETF
    "IWM": {
        "strategies": ["iron_condor", "credit_spread", "put_spread"],
        "min_price": 150.0,
        "max_price": 250.0,
        "sector": "small_cap",
        "notes": "Small cap exposure, moderate volatility"
    },
    
    # AAPL - Apple
    "AAPL": {
        "strategies": ["credit_spread", "covered_call", "put_spread", "iron_condor"],
        "min_price": 100.0,
        "max_price": 250.0,
        "sector": "technology",
        "notes": "Mega-cap tech, excellent option liquidity"
    },
    
    # TSLA - Tesla
    "TSLA": {
        "strategies": ["iron_condor", "straddle", "strangle", "credit_spread"],
        "min_price": 150.0,
        "max_price": 400.0,
        "sector": "automotive_tech",
        "notes": "High volatility, large moves, rich premiums"
    },
    
    # NVDA - NVIDIA
    "NVDA": {
        "strategies": ["credit_spread", "iron_condor", "put_spread"],
        "min_price": 200.0,
        "max_price": 1000.0,
        "sector": "semiconductors",
        "notes": "AI leader, high volatility, good liquidity"
    },
    
    # MSFT - Microsoft
    "MSFT": {
        "strategies": ["credit_spread", "iron_condor", "covered_call"],
        "min_price": 250.0,
        "max_price": 450.0,
        "sector": "technology",
        "notes": "Stable mega-cap, moderate volatility"
    },
    
    # AMZN - Amazon
    "AMZN": {
        "strategies": ["credit_spread", "iron_condor", "straddle"],
        "min_price": 100.0,
        "max_price": 200.0,
        "sector": "e-commerce_cloud",
        "notes": "High beta, good for premium selling"
    },
    
    # META - Meta Platforms
    "META": {
        "strategies": ["credit_spread", "iron_condor", "strangle"],
        "min_price": 200.0,
        "max_price": 600.0,
        "sector": "social_media",
        "notes": "Elevated volatility, good option volume"
    },
    
    # DIA - Dow Jones ETF
    "DIA": {
        "strategies": ["iron_condor", "credit_spread"],
        "min_price": 300.0,
        "max_price": 450.0,
        "sector": "blue_chip",
        "notes": "Lower volatility, stable premium collection"
    },
}


# ============================================================================
# STRATEGY DEFINITIONS
# ============================================================================

STRATEGY_DEFINITIONS = {
    "iron_condor": {
        "legs": 4,
        "min_dte": 21,
        "max_dte": 45,
        "ideal_iv_rank": "high",
        "description": "Sell OTM put spread + call spread",
    },
    "credit_spread": {
        "legs": 2,
        "min_dte": 21,
        "max_dte": 45,
        "ideal_iv_rank": "high",
        "description": "Sell vertical spread for credit",
    },
    "straddle": {
        "legs": 2,
        "min_dte": 14,
        "max_dte": 30,
        "ideal_iv_rank": "low",
        "description": "Buy ATM call + put for volatility",
    },
    "strangle": {
        "legs": 2,
        "min_dte": 14,
        "max_dte": 30,
        "ideal_iv_rank": "low",
        "description": "Buy OTM call + put for volatility",
    },
    "covered_call": {
        "legs": 1,
        "min_dte": 30,
        "max_dte": 45,
        "ideal_iv_rank": "high",
        "description": "Sell call against stock position",
    },
    "put_spread": {
        "legs": 2,
        "min_dte": 21,
        "max_dte": 45,
        "ideal_iv_rank": "high", 
        "description": "Bull put spread for credit",
    },
    "calendar_spread": {
        "legs": 2,
        "min_dte": 30,
        "max_dte": 60,
        "ideal_iv_rank": "normal",
        "description": "Sell near-term, buy far-term",
    },
    "butterfly": {
        "legs": 4,
        "min_dte": 21,
        "max_dte": 45,
        "ideal_iv_rank": "low",
        "description": "Balanced wings for neutral outlook",
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_universe() -> List[str]:
    """
    Get list of all tradable symbols.
    
    Returns:
        List of symbol strings
    """
    return list(OPTIONS_UNIVERSE.keys())


def get_strategies_for_symbol(symbol: str) -> List[str]:
    """
    Get allowed strategies for a symbol.
    
    Args:
        symbol: Stock/ETF symbol
        
    Returns:
        List of allowed strategy names
    """
    return OPTIONS_UNIVERSE.get(symbol, {}).get("strategies", [])


def is_strategy_allowed(symbol: str, strategy: str) -> bool:
    """
    Check if strategy is allowed for symbol.
    
    Args:
        symbol: Stock/ETF symbol
        strategy: Strategy name
        
    Returns:
        True if allowed, False otherwise
    """
    return strategy in get_strategies_for_symbol(symbol)


def get_high_liquidity_symbols() -> List[str]:
    """
    Get symbols with highest liquidity (ETFs).
    
    Returns:
        List of high-liquidity symbols
    """
    return ["SPY", "QQQ", "IWM", "DIA"]


def get_tech_symbols() -> List[str]:
    """
    Get technology sector symbols.
    
    Returns:
        List of tech symbols
    """
    tech_symbols = []
    for symbol, config in OPTIONS_UNIVERSE.items():
        if "tech" in config.get("sector", "").lower():
            tech_symbols.append(symbol)
    return tech_symbols


def get_symbols_by_strategy(strategy: str) -> List[str]:
    """
    Get symbols that support a specific strategy.
    
    Args:
        strategy: Strategy name
        
    Returns:
        List of symbols supporting the strategy
    """
    symbols = []
    for symbol, config in OPTIONS_UNIVERSE.items():
        if strategy in config.get("strategies", []):
            symbols.append(symbol)
    return symbols


def validate_universe() -> bool:
    """
    Validate universe configuration.
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check all symbols have required fields
    required_fields = ["strategies"]
    
    for symbol, config in OPTIONS_UNIVERSE.items():
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Symbol {symbol} missing required field: {field}")
        
        # Validate strategies exist in definitions
        for strategy in config["strategies"]:
            if strategy not in STRATEGY_DEFINITIONS:
                raise ValueError(f"Unknown strategy '{strategy}' for symbol {symbol}")
    
    return True


# Validate on import
validate_universe()

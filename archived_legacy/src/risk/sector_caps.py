"""
Sector-based position caps to prevent portfolio concentration.

Problem: All standalone bots traded 70%+ tech stocks, so a single sector
drawdown wiped out the entire portfolio.  This module classifies tickers
by sector and enforces maximum per-sector exposure.
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Sector Classification ───────────────────────────────────────────────
# GICS-like grouping for the bot's combined trading universe.
# Tickers not listed here are classified as 'unknown'.

SECTOR_MAP: Dict[str, str] = {
    # Technology
    'AAPL': 'technology', 'MSFT': 'technology', 'GOOGL': 'technology',
    'NVDA': 'technology', 'META': 'technology', 'AMD': 'technology',
    'CRM': 'technology', 'PLTR': 'technology', 'SHOP': 'technology',
    'XLK': 'technology', 'ADBE': 'technology', 'ORCL': 'technology',
    'AVGO': 'technology', 'QCOM': 'technology',
    # Consumer Discretionary
    'AMZN': 'consumer_discretionary', 'TSLA': 'consumer_discretionary',
    'NFLX': 'consumer_discretionary', 'DIS': 'consumer_discretionary',
    # Financials
    'JPM': 'financials', 'GS': 'financials', 'BAC': 'financials',
    'V': 'financials', 'MA': 'financials', 'XLF': 'financials',
    'SQ': 'financials', 'PYPL': 'financials', 'COIN': 'financials',
    'MSTR': 'financials',
    # Healthcare
    'UNH': 'healthcare', 'JNJ': 'healthcare', 'LLY': 'healthcare',
    'PFE': 'healthcare', 'ABBV': 'healthcare', 'MRK': 'healthcare',
    # Energy
    'XOM': 'energy', 'CVX': 'energy', 'COP': 'energy', 'SLB': 'energy',
    # Consumer Staples
    'KO': 'consumer_staples', 'PEP': 'consumer_staples',
    'PG': 'consumer_staples', 'COST': 'consumer_staples', 'WMT': 'consumer_staples',
    # Industrials
    'HD': 'industrials', 'LOW': 'industrials',
    'CAT': 'industrials', 'HON': 'industrials', 'UPS': 'industrials',
    'GE': 'industrials', 'RTX': 'industrials', 'DE': 'industrials',
    # Utilities
    'NEE': 'utilities', 'SO': 'utilities',
    # REITs
    'AMT': 'reits',
    # Materials
    'LIN': 'materials', 'FCX': 'materials', 'NEM': 'materials',
    # Broad Market ETFs (treated as their own "sector" so they don't crowd out)
    'SPY': 'broad_market', 'QQQ': 'broad_market', 'IWM': 'broad_market',
    'DIA': 'broad_market',
}

# Default maximum % of equity that any single sector can consume.
# Broad-market ETFs get a higher cap since they're already diversified.
DEFAULT_SECTOR_CAPS: Dict[str, float] = {
    'technology': 0.30,            # max 30% in tech (reduced from 35%)
    'consumer_discretionary': 0.20,
    'financials': 0.20,
    'healthcare': 0.20,
    'energy': 0.15,
    'consumer_staples': 0.15,
    'industrials': 0.15,
    'utilities': 0.10,
    'reits': 0.10,
    'materials': 0.10,
    'broad_market': 0.30,          # SPY/QQQ/IWM/DIA can be up to 30%
    'unknown': 0.10,               # unknown tickers limited to 10%
}


def get_sector(symbol: str) -> str:
    """Return the sector for a symbol, or 'unknown'."""
    return SECTOR_MAP.get(symbol, 'unknown')


def sector_allows_trade(
    symbol: str,
    proposed_cost: float,
    current_positions: Dict[str, float],
    total_equity: float,
    caps: Optional[Dict[str, float]] = None,
) -> Tuple[bool, str]:
    """
    Check whether adding *proposed_cost* for *symbol* would breach a
    sector cap.

    Parameters
    ----------
    symbol : str
        Ticker to trade.
    proposed_cost : float
        Dollar cost of the proposed new position.
    current_positions : dict
        Mapping of symbol → current position market_value (dollars).
    total_equity : float
        Current account equity.
    caps : dict, optional
        Override sector caps.  Defaults to DEFAULT_SECTOR_CAPS.

    Returns
    -------
    (allowed, reason) : (bool, str)
    """
    if caps is None:
        caps = DEFAULT_SECTOR_CAPS

    sector = get_sector(symbol)
    cap_pct = caps.get(sector, caps.get('unknown', 0.10))
    max_dollars = total_equity * cap_pct

    # Sum current exposure in this sector
    sector_exposure = sum(
        abs(val) for sym, val in current_positions.items()
        if get_sector(sym) == sector
    )

    if sector_exposure + proposed_cost > max_dollars:
        reason = (
            f"Sector '{sector}' exposure would be "
            f"${sector_exposure + proposed_cost:,.0f} "
            f"(cap ${max_dollars:,.0f} = {cap_pct:.0%} of ${total_equity:,.0f})"
        )
        logger.warning(f"🚫 SECTOR CAP: {symbol} blocked — {reason}")
        return False, reason

    return True, "ok"


def filter_universe_by_sector(
    universe: List[str],
    current_positions: Dict[str, float],
    total_equity: float,
    caps: Optional[Dict[str, float]] = None,
) -> List[str]:
    """
    Return the subset of *universe* whose sectors are not already at cap.
    Useful for pre-filtering scan lists so bots don't waste API calls on
    symbols they can't trade.
    """
    if caps is None:
        caps = DEFAULT_SECTOR_CAPS

    allowed = []
    for sym in universe:
        sector = get_sector(sym)
        cap_pct = caps.get(sector, caps.get('unknown', 0.10))
        max_dollars = total_equity * cap_pct
        sector_exposure = sum(
            abs(val) for s, val in current_positions.items()
            if get_sector(s) == sector
        )
        if sector_exposure < max_dollars:
            allowed.append(sym)
    return allowed

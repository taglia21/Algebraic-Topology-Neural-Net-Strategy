#!/usr/bin/env python3
"""
Trading Universe Configuration
==============================
Expanded universe for the Team of Rivals trading system.
Includes S&P 500 components, major ETFs, and high-liquidity stocks.
"""

# Major Market ETFs
ETFS = [
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO',
    'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLY', 'XLB', 'XLRE', 'XLC',
    'TLT', 'IEF', 'HYG', 'LQD', 'GLD', 'SLV', 'USO',
    'EEM', 'EFA', 'VWO', 'ARKK', 'ARKG', 'ARKF',
]

# Mega Cap Tech
MEGA_CAP_TECH = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
    'AVGO', 'ORCL', 'CRM', 'ADBE', 'AMD', 'INTC', 'CSCO', 'QCOM',
    'TXN', 'IBM', 'NOW', 'INTU', 'AMAT', 'MU', 'LRCX', 'ADI',
]

FINANCIALS = [
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW',
    'AXP', 'SPGI', 'CME', 'ICE', 'COF', 'USB', 'PNC', 'TFC',
]

HEALTHCARE = [
    'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT',
    'DHR', 'BMY', 'AMGN', 'GILD', 'ISRG', 'VRTX', 'REGN', 'MDT',
]

CONSUMER = [
    'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'CMG',
    'PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'MDLZ',
]

INDUSTRIALS = [
    'CAT', 'DE', 'UNP', 'UPS', 'RTX', 'HON', 'BA', 'LMT',
    'GE', 'MMM', 'EMR', 'ETN', 'FDX', 'NSC', 'CSX', 'GD',
]

ENERGY = [
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO',
    'OXY', 'DVN', 'HES', 'HAL', 'BKR', 'KMI', 'WMB', 'OKE',
]

GROWTH_TECH = [
    'SNOW', 'PLTR', 'CRWD', 'DDOG', 'ZS', 'NET', 'MDB', 'TEAM',
    'WDAY', 'SQ', 'PYPL', 'COIN', 'SHOP', 'MELI', 'BABA', 'JD',
]

MATERIALS = ['LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'NUE']

COMMS = ['NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'EA', 'CHTR']

UTILITIES = ['NEE', 'DUK', 'SO', 'D', 'AEP', 'AMT', 'PLD', 'CCI', 'EQIX', 'PSA']

def get_full_universe():
    """Get complete trading universe - ~200 liquid stocks."""
    all_symbols = set()
    for lst in [ETFS, MEGA_CAP_TECH, FINANCIALS, HEALTHCARE, CONSUMER,
                INDUSTRIALS, ENERGY, GROWTH_TECH, MATERIALS, COMMS, UTILITIES]:
        all_symbols.update(lst)
    return sorted(list(all_symbols))

def get_core_universe(apply_overrides: bool = True):
    """
    Get core trading universe - 50 most liquid.
    
    Args:
        apply_overrides: If True, apply strategy_overrides exclusions (removes weak assets like QQQ)
    """
    base_universe = [
        'SPY', 'IWM', 'DIA',  # QQQ removed - Sharpe 0.39 drags portfolio
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'AVGO', 'CRM',
        'JPM', 'BAC', 'GS', 'MS', 'BLK',
        'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK',
        'HD', 'MCD', 'NKE', 'COST', 'WMT', 'PG', 'KO',
        'CAT', 'DE', 'UNP', 'HON', 'BA',
        'XOM', 'CVX', 'COP', 'SLB',
        'NFLX', 'DIS', 'VZ', 'T',
        'XLF', 'XLK',  # Added high-Sharpe sector ETFs
        'SNOW', 'CRWD', 'DDOG', 'PLTR', 'SQ', 'PYPL',
    ]
    
    if apply_overrides:
        try:
            from config.strategy_overrides import get_overrides
            overrides = get_overrides()
            return [t for t in base_universe if overrides.should_include_ticker(t)]
        except ImportError:
            pass
    
    return base_universe


def get_core_universe_legacy():
    """Original core universe without overrides (for comparison backtests)."""
    return [
        'SPY', 'QQQ', 'IWM', 'DIA',
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'AVGO', 'CRM',
        'JPM', 'BAC', 'GS', 'MS', 'BLK',
        'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK',
        'HD', 'MCD', 'NKE', 'COST', 'WMT', 'PG', 'KO',
        'CAT', 'DE', 'UNP', 'HON', 'BA',
        'XOM', 'CVX', 'COP', 'SLB',
        'NFLX', 'DIS', 'VZ', 'T',
        'SNOW', 'CRWD', 'DDOG', 'PLTR', 'SQ', 'PYPL',
    ]

if __name__ == "__main__":
    full = get_full_universe()
    core = get_core_universe()
    print(f"Full Universe: {len(full)} symbols")
    print(f"Core Universe: {len(core)} symbols")

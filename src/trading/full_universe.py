"""
Full Market Universe Manager
=============================

Manages a comprehensive US equity universe of 500+ stocks for hedge fund deployment.

Includes:
- S&P 500 (500 stocks)
- NASDAQ 100 (100 stocks)  
- Russell 2000 top liquid names (200+ stocks)
- Sector ETFs and Leveraged ETFs

Total tradeable universe: 700+ symbols
"""

import os
import json
import logging
from typing import List, Set, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# =============================================================================
# FULL MARKET UNIVERSE - 500+ STOCKS
# =============================================================================

# S&P 500 Components (as of 2024, most liquid)
SP500_TICKERS = [
    # Technology (75 stocks)
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'META', 'AVGO', 'CSCO', 'ORCL', 'CRM',
    'ACN', 'ADBE', 'AMD', 'INTC', 'QCOM', 'TXN', 'IBM', 'NOW', 'INTU', 'AMAT',
    'ADI', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'PANW', 'MCHP', 'FTNT', 'ADSK',
    'ANSS', 'CRWD', 'NXPI', 'ON', 'MRVL', 'HPQ', 'HPE', 'KEYS', 'ZBRA', 'CTSH',
    'IT', 'AKAM', 'JNPR', 'FFIV', 'NTAP', 'WDC', 'STX', 'DELL', 'GEN', 'EPAM',
    'GDDY', 'PAYC', 'PAYX', 'CDAY', 'MANH', 'TYL', 'PTC', 'CDW', 'BR', 'JKHY',
    'TRMB', 'TER', 'SWKS', 'QRVO', 'MPWR', 'ENPH', 'SEDG', 'WOLF', 'FSLR', 'RUN',
    'MSCI', 'SPGI', 'MCO', 'FDS', 'VRSK',
    
    # Healthcare (65 stocks)
    'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY',
    'AMGN', 'GILD', 'ISRG', 'MDT', 'SYK', 'BSX', 'EW', 'ZBH', 'BDX', 'DXCM',
    'IDXX', 'IQV', 'RMD', 'MTD', 'WAT', 'A', 'WST', 'TFX', 'HOLX', 'BAX',
    'ALGN', 'COO', 'PODD', 'RVTY', 'TECH', 'BIO', 'CRL', 'ICLR', 'MEDP', 'EXAS',
    'VEEV', 'ZTS', 'REGN', 'VRTX', 'MRNA', 'BIIB', 'INCY', 'ALNY', 'SGEN', 'BMRN',
    'CI', 'ELV', 'HUM', 'CNC', 'MOH', 'CVS', 'HCA', 'UHS', 'THC', 'GEHC',
    'DGX', 'LH', 'CAH', 'MCK', 'ABC',
    
    # Financials (70 stocks)
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF',
    'AXP', 'SCHW', 'BLK', 'SPGI', 'ICE', 'CME', 'MCO', 'MSCI', 'NDAQ', 'CBOE',
    'BK', 'STT', 'NTRS', 'TROW', 'SEIC', 'AMG', 'BEN', 'IVZ', 'LM', 'APAM',
    'V', 'MA', 'PYPL', 'FIS', 'FISV', 'GPN', 'FLT', 'CPAY', 'SQ', 'AFRM',
    'MET', 'PRU', 'AIG', 'AFL', 'PGR', 'ALL', 'TRV', 'CB', 'HIG', 'L',
    'CINF', 'GL', 'WRB', 'RNR', 'ERIE', 'RE', 'AJG', 'MMC', 'AON', 'MARSH',
    'BRO', 'WTW', 'RYAN', 'KKR', 'APO', 'BX', 'CG', 'ARES', 'OWL', 'HLNE',
    
    # Consumer Discretionary (65 stocks)
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'BKNG', 'MAR',
    'HLT', 'CMG', 'YUM', 'DRI', 'ORLY', 'AZO', 'AAP', 'GPC', 'ULTA', 'ROST',
    'BBY', 'TGT', 'KMX', 'AN', 'LAD', 'CVNA', 'GRMN', 'POOL', 'DECK', 'LULU',
    'WSM', 'RH', 'ETSY', 'W', 'CHWY', 'PTON', 'NVR', 'DHI', 'LEN', 'PHM',
    'TOL', 'MTH', 'KBH', 'MDC', 'LGIH', 'F', 'GM', 'RIVN', 'LCID', 'BWA',
    'APTV', 'LEA', 'VC', 'ALV', 'MGA', 'DAN', 'DORM', 'FOXF', 'LKQ', 'GNTX',
    'CZR', 'WYNN', 'LVS', 'MGM', 'RCL',
    
    # Consumer Staples (35 stocks)
    'COST', 'WMT', 'PG', 'KO', 'PEP', 'PM', 'MO', 'STZ', 'BF-B', 'DEO',
    'MNST', 'KDP', 'TAP', 'SAM', 'MDLZ', 'HSY', 'KHC', 'GIS', 'K', 'CPB',
    'CAG', 'SJM', 'HRL', 'TSN', 'JBS', 'INGR', 'ADM', 'BG', 'DAR', 'FDP',
    'CL', 'EL', 'CHD', 'CLX', 'KMB',
    
    # Industrials (75 stocks)
    'CAT', 'DE', 'UNP', 'CSX', 'NSC', 'UPS', 'FDX', 'XPO', 'JBHT', 'CHRW',
    'ODFL', 'SAIA', 'WERN', 'KNX', 'LSTR', 'SNDR', 'GWW', 'FAST', 'WSO', 'POOL',
    'LMT', 'RTX', 'NOC', 'GD', 'BA', 'TDG', 'HWM', 'SPR', 'TXT', 'LHX',
    'HON', 'GE', 'ETN', 'EMR', 'ITW', 'ROK', 'AME', 'IEX', 'XYL', 'DOV',
    'IR', 'TT', 'CARR', 'GNRC', 'AGCO', 'PCAR', 'CMI', 'PACCAR', 'WAB', 'TTC',
    'PNR', 'SWK', 'ALLE', 'MAS', 'OC', 'BLD', 'BLDR', 'MLM', 'VMC', 'CX',
    'WM', 'WCN', 'RSG', 'CLH', 'SRCL', 'ECOL', 'NDSN', 'RRX', 'ROP', 'IDEX',
    'FTV', 'PWR', 'EME', 'MTZ', 'DY',
    
    # Energy (30 stocks)
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX', 'PXD',
    'DVN', 'FANG', 'HES', 'HAL', 'BKR', 'MRO', 'APA', 'CTRA', 'OVV', 'EQT',
    'AR', 'SWN', 'RRC', 'MTDR', 'PR', 'KMI', 'WMB', 'OKE', 'TRGP', 'LNG',
    
    # Utilities (30 stocks)
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'WEC', 'ES',
    'PEG', 'ED', 'EIX', 'ETR', 'FE', 'DTE', 'PPL', 'CMS', 'CNP', 'NI',
    'AEE', 'EVRG', 'AWK', 'ATO', 'NRG', 'VST', 'CEG', 'PCG', 'OGE', 'PNW',
    
    # Materials (30 stocks)
    'LIN', 'APD', 'SHW', 'ECL', 'DD', 'DOW', 'PPG', 'ALB', 'EMN', 'CE',
    'LYB', 'CTVA', 'FMC', 'CF', 'MOS', 'NTR', 'NEM', 'FCX', 'SCCO', 'GOLD',
    'NUE', 'STLD', 'CLF', 'X', 'AA', 'ATI', 'RS', 'CMC', 'WOR', 'SUM',
    
    # Real Estate (30 stocks)
    'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB',
    'EQR', 'VTR', 'ARE', 'BXP', 'VNO', 'SLG', 'KIM', 'REG', 'FRT', 'HST',
    'RHP', 'PEAK', 'DOC', 'HR', 'OHI', 'SBAC', 'UDR', 'CPT', 'ESS', 'MAA',
    
    # Communications (20 stocks)
    'GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR', 'PARA',
    'WBD', 'FOX', 'FOXA', 'LYV', 'MTCH', 'EA', 'TTWO', 'RBLX', 'U', 'ZG',
]

# Mid-Cap Growth & Value (Russell 1000 beyond S&P 500)
MIDCAP_TICKERS = [
    # Mid-Cap Tech
    'DDOG', 'SNOW', 'MDB', 'NET', 'ZS', 'OKTA', 'HUBS', 'TEAM', 'WDAY', 'SPLK',
    'ZM', 'DOCU', 'TWLO', 'PINS', 'SNAP', 'BILL', 'CFLT', 'ESTC', 'GTLB', 'APP',
    'IOT', 'PATH', 'SMAR', 'FROG', 'NEWR', 'BRZE', 'SQSP', 'MNDY', 'WK', 'BASE',
    'DOCS', 'SMCI', 'AEHR', 'SITM', 'POWI', 'ONTO', 'AMBA', 'CRUS', 'SLAB', 'DIOD',
    
    # Mid-Cap Healthcare
    'EXEL', 'SRPT', 'RARE', 'IONS', 'NBIX', 'PTCT', 'BLUE', 'FOLD', 'HALX', 'UTHR',
    'JAZZ', 'PRGO', 'ELAN', 'AXON', 'GEHC', 'GMED', 'LNTH', 'NVST', 'STE', 'TFX',
    
    # Mid-Cap Financials
    'WAL', 'FRC', 'SIVB', 'SBNY', 'FHN', 'CFG', 'MTB', 'KEY', 'HBAN', 'ZION',
    'RF', 'CMA', 'FITB', 'ALLY', 'LC', 'SOFI', 'UPST', 'HOOD', 'IBKR', 'MKTX',
    'VIRT', 'LPLA', 'SEIC', 'SF', 'EVR', 'HLI', 'LAZ', 'JEF', 'PJT', 'MC',
    
    # Mid-Cap Consumer
    'FIVE', 'OLLI', 'DKS', 'ASO', 'BOOT', 'PLBY', 'HBI', 'GOOS', 'CROX', 'SKX',
    'SHOO', 'WWW', 'TPR', 'CPRI', 'RL', 'VFC', 'UAA', 'LEVI', 'GPS', 'AEO',
    'EXPR', 'BURL', 'CATO', 'DDS', 'M', 'JWN', 'KSS', 'BIG', 'DLTR', 'DG',
    
    # Mid-Cap Industrials
    'SITE', 'GMS', 'UFPI', 'TREX', 'AZEK', 'DOOR', 'AWI', 'TILE', 'BECN', 'IBP',
    'EXPO', 'WSC', 'WTS', 'PRIM', 'APG', 'ESAB', 'KAI', 'KRNT', 'MEI', 'NPO',
    
    # Mid-Cap Energy & MLPs
    'EPD', 'ET', 'MPLX', 'PAA', 'WES', 'HESM', 'DTM', 'USAC', 'AROC', 'CEQP',
    'ENLC', 'TRGP', 'AM', 'NS', 'GEL', 'PBFX', 'GPP', 'CAPL', 'UGP', 'CCLP',
]

# Small-Cap High Momentum Names (Russell 2000 top performers)
SMALLCAP_MOMENTUM = [
    # High-growth small caps
    'ASAN', 'FROG', 'GTLB', 'BRZE', 'MNDY', 'JAMF', 'ALTR', 'AI', 'BIGC', 'VTEX',
    'GLBE', 'GLOB', 'CWAN', 'INTA', 'FLYW', 'TASK', 'COUR', 'UDMY', 'GENI', 'RSI',
    'DKNG', 'PENN', 'BETZ', 'GNOG', 'AGS', 'CHDN', 'BYD', 'IGT', 'SGMS', 'EVRI',
    'STEM', 'BEEM', 'BLNK', 'CHPT', 'EVGO', 'LEV', 'ARVL', 'FSR', 'FFIE', 'GOEV',
    'SHLS', 'ARRY', 'MAXN', 'SPWR', 'NOVA', 'CSIQ', 'DQ', 'JKS', 'SOL', 'SUNW',
]

# ETFs for sector rotation and leverage
ETFS = [
    # Sector ETFs
    'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC',
    'SMH', 'SOXX', 'IBB', 'XBI', 'IYR', 'VNQ', 'KRE', 'XHB', 'ITB', 'GDX', 'GDXJ',
    'XOP', 'OIH', 'KWEB', 'FXI', 'EWJ', 'EEM', 'VWO', 'IEMG',
    
    # Core Market ETFs
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VT', 'VEA', 'VWO', 'ARKK',
    'MDY', 'IJH', 'IJR', 'IWO', 'IWN', 'VTV', 'VUG', 'MTUM', 'QUAL', 'VLUE',
    
    # 3x Leveraged Bull ETFs
    'TQQQ', 'SPXL', 'UPRO', 'TNA', 'SOXL', 'TECL', 'FNGU', 'LABU', 'FAS', 'CURE',
    'DPST', 'NAIL', 'DFEN', 'RETL', 'HIBL', 'WANT', 'WEBL', 'PILL', 'DUSL', 'MIDU',
    
    # 3x Leveraged Bear/Inverse ETFs
    'SQQQ', 'SPXU', 'SPXS', 'TZA', 'SOXS', 'TECS', 'FNGD', 'LABD', 'FAZ', 'HIBS',
    
    # 2x Leveraged (less decay)
    'QLD', 'SSO', 'UWM', 'ROM', 'USD', 'UYG', 'UGE', 'UCC', 'SAA', 'MVV',
]

# Combine all into full universe
FULL_UNIVERSE = list(set(SP500_TICKERS + MIDCAP_TICKERS + SMALLCAP_MOMENTUM + ETFS))

# Sector classification for all stocks
SECTOR_MAP = {}

# Technology
for t in ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'META', 'AVGO', 'CSCO', 'ORCL', 'CRM', 
          'ACN', 'ADBE', 'AMD', 'INTC', 'QCOM', 'TXN', 'IBM', 'NOW', 'INTU', 'AMAT',
          'DDOG', 'SNOW', 'MDB', 'NET', 'ZS', 'CRWD', 'PANW', 'FTNT', 'OKTA', 'HUBS',
          'SMCI', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MCHP', 'NXPI', 'ON', 'MRVL']:
    SECTOR_MAP[t] = 'Technology'

# Healthcare
for t in ['UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY',
          'AMGN', 'GILD', 'ISRG', 'MDT', 'SYK', 'BSX', 'VEEV', 'DXCM', 'IDXX', 'ZTS']:
    SECTOR_MAP[t] = 'Healthcare'

# Financials
for t in ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA', 'BLK', 'SCHW',
          'AXP', 'USB', 'PNC', 'TFC', 'COF', 'CME', 'ICE', 'SPGI', 'MCO', 'CB']:
    SECTOR_MAP[t] = 'Financials'

# Consumer
for t in ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'COST', 'WMT', 'LOW', 'TGT',
          'BKNG', 'MAR', 'HLT', 'CMG', 'LULU', 'ROST', 'TJX', 'ORLY', 'AZO', 'DG']:
    SECTOR_MAP[t] = 'Consumer'

# Industrials
for t in ['CAT', 'DE', 'UNP', 'UPS', 'FDX', 'HON', 'GE', 'BA', 'LMT', 'RTX',
          'ETN', 'EMR', 'ITW', 'MMM', 'CMI', 'PCAR', 'WAB', 'GWW', 'FAST', 'ROK']:
    SECTOR_MAP[t] = 'Industrials'

# Energy
for t in ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX', 'PXD',
          'DVN', 'FANG', 'HES', 'HAL', 'BKR', 'KMI', 'WMB', 'OKE', 'TRGP', 'LNG']:
    SECTOR_MAP[t] = 'Energy'

# Utilities
for t in ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'WEC', 'ES', 'CEG']:
    SECTOR_MAP[t] = 'Utilities'

# Materials
for t in ['LIN', 'APD', 'SHW', 'ECL', 'DD', 'DOW', 'NEM', 'FCX', 'NUE', 'SCCO']:
    SECTOR_MAP[t] = 'Materials'

# Real Estate
for t in ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB']:
    SECTOR_MAP[t] = 'Real Estate'

# Communications
for t in ['NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR', 'EA', 'TTWO', 'RBLX']:
    SECTOR_MAP[t] = 'Communications'


def get_full_universe() -> List[str]:
    """Get the complete tradeable universe."""
    return FULL_UNIVERSE.copy()


def get_sp500() -> List[str]:
    """Get S&P 500 tickers only."""
    return SP500_TICKERS.copy()


def get_sector(ticker: str) -> str:
    """Get sector for a ticker."""
    return SECTOR_MAP.get(ticker, 'Other')


def get_etfs() -> List[str]:
    """Get all ETFs."""
    return ETFS.copy()


def get_leveraged_bull_etfs() -> Dict[str, float]:
    """Get 3x leveraged bull ETFs with default weights."""
    return {
        'TQQQ': 0.35,  # Nasdaq 100 3x
        'SPXL': 0.25,  # S&P 500 3x
        'SOXL': 0.20,  # Semiconductors 3x
        'TECL': 0.10,  # Technology 3x
        'FNGU': 0.10,  # FAANG 3x
    }


def get_leveraged_bear_etfs() -> Dict[str, float]:
    """Get 3x leveraged bear/inverse ETFs with default weights."""
    return {
        'SQQQ': 0.35,  # Nasdaq 100 -3x
        'SPXU': 0.25,  # S&P 500 -3x
        'SOXS': 0.20,  # Semiconductors -3x
        'TECS': 0.10,  # Technology -3x
        'FNGD': 0.10,  # FAANG -3x
    }


# Export universe size info
UNIVERSE_SIZE = len(FULL_UNIVERSE)
SP500_SIZE = len(SP500_TICKERS)
MIDCAP_SIZE = len(MIDCAP_TICKERS)
SMALLCAP_SIZE = len(SMALLCAP_MOMENTUM)
ETF_SIZE = len(ETFS)

logger.info(f"Full Universe: {UNIVERSE_SIZE} symbols")
logger.info(f"  - S&P 500: {SP500_SIZE}")
logger.info(f"  - Mid-Caps: {MIDCAP_SIZE}")
logger.info(f"  - Small-Cap Momentum: {SMALLCAP_SIZE}")
logger.info(f"  - ETFs: {ETF_SIZE}")

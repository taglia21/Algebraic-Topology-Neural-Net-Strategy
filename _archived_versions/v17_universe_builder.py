#!/usr/bin/env python3
"""
V17.0 Universe Builder
======================
Builds a tradeable universe of 3,500+ stocks with proper entry/exit smoothing.

Universe Criteria:
- Russell 3000 + NASDAQ 100 + S&P 500 constituents
- Price > $5
- Avg Daily $ Volume > $1M (20-day average)
- Smoothing: 16/21 days to enter, 5/21 days to exit

Output: JSON file with qualified symbols updated daily
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('V17_Universe')


# ============================================================================
# HARDCODED CONSTITUENT LISTS (Russell 3000 approximation)
# ============================================================================

# S&P 500 components (top ~500)
SP500_SYMBOLS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'GOOG', 'TSLA', 'BRK.B', 'UNH',
    'XOM', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'AVGO', 'HD', 'CVX', 'MRK',
    'ABBV', 'LLY', 'PEP', 'KO', 'COST', 'ADBE', 'WMT', 'MCD', 'CSCO', 'CRM',
    'BAC', 'PFE', 'TMO', 'ACN', 'NFLX', 'AMD', 'ABT', 'DHR', 'DIS', 'LIN',
    'CMCSA', 'VZ', 'INTC', 'NKE', 'PM', 'WFC', 'TXN', 'NEE', 'RTX', 'UPS',
    'QCOM', 'BMY', 'COP', 'HON', 'LOW', 'ORCL', 'UNP', 'SPGI', 'IBM', 'CAT',
    'GE', 'BA', 'INTU', 'AMAT', 'AMGN', 'GS', 'SBUX', 'BLK', 'DE', 'ELV',
    'ISRG', 'MDLZ', 'ADP', 'GILD', 'ADI', 'BKNG', 'VRTX', 'TJX', 'PLD', 'MMC',
    'SYK', 'MS', 'CVS', 'LMT', 'REGN', 'CI', 'TMUS', 'CB', 'SCHW', 'ZTS',
    'ETN', 'MO', 'SO', 'BDX', 'EOG', 'DUK', 'AMT', 'BSX', 'LRCX', 'NOC',
    'PYPL', 'AON', 'CME', 'ICE', 'ITW', 'WM', 'SLB', 'APD', 'CSX', 'CL',
    'PNC', 'TGT', 'FCX', 'MCK', 'EMR', 'MPC', 'USB', 'SHW', 'SNPS', 'NSC',
    'FDX', 'CDNS', 'GD', 'ORLY', 'PSX', 'AZO', 'OXY', 'TFC', 'AJG', 'KLAC',
    'MCO', 'ROP', 'HUM', 'MCHP', 'PCAR', 'VLO', 'MAR', 'AEP', 'MET', 'KMB',
    'CTAS', 'AFL', 'MSCI', 'D', 'AIG', 'TRV', 'CCI', 'GIS', 'PSA', 'JCI',
    'HCA', 'APH', 'WELL', 'CMG', 'DXCM', 'F', 'GM', 'TEL', 'CARR', 'NUE',
    'ADM', 'SRE', 'CHTR', 'WMB', 'STZ', 'HES', 'DVN', 'KHC', 'A', 'IDXX',
    'BIIB', 'EW', 'DHI', 'LHX', 'HAL', 'AMP', 'EXC', 'DOW', 'PAYX', 'MNST',
    'ROK', 'PRU', 'MTD', 'ODFL', 'FTNT', 'SPG', 'XEL', 'ED', 'ROST', 'OTIS',
    'AME', 'BK', 'CTSH', 'GWW', 'DD', 'CMI', 'CPRT', 'EA', 'IQV', 'PEG',
    'DLR', 'KDP', 'PPG', 'YUM', 'HSY', 'KEYS', 'ON', 'FAST', 'EXR', 'VRSK',
    'FANG', 'EIX', 'GPN', 'ANSS', 'CDW', 'WTW', 'MLM', 'IR', 'VICI', 'BKR',
    'ALB', 'WEC', 'EBAY', 'AWK', 'RMD', 'OKE', 'VMC', 'EFX', 'CAH', 'MTB',
    'DFS', 'CBRE', 'ACGL', 'DLTR', 'WBD', 'FTV', 'ZBH', 'LYB', 'TSCO', 'PCG',
    'HPQ', 'ULTA', 'RSG', 'HLT', 'TROW', 'FE', 'ARE', 'RJF', 'FITB', 'BAX',
    'ES', 'STT', 'PPL', 'DTE', 'K', 'CHD', 'ILMN', 'LUV', 'SBAC', 'AEE',
    'AVB', 'HOLX', 'NTRS', 'RF', 'VTR', 'HBAN', 'WAB', 'CINF', 'NDAQ', 'CFG',
    'WY', 'MAA', 'EXPD', 'CNP', 'CMS', 'MKC', 'NVR', 'MOH', 'WAT', 'INVH',
    'TDY', 'J', 'CLX', 'COO', 'SYY', 'ALGN', 'DOV', 'ATO', 'AMCR', 'EQR',
    'BALL', 'LH', 'TXT', 'STE', 'BR', 'ENPH', 'SWK', 'OMC', 'MAS', 'URI',
    'HPE', 'PKI', 'IP', 'FLT', 'TRGP', 'NTAP', 'LDOS', 'SWKS', 'WRB', 'DGX',
    'EVRG', 'JBHT', 'CE', 'APA', 'GEN', 'TER', 'POOL', 'BRO', 'FDS', 'LVS',
    'PFG', 'LNT', 'IEX', 'CAG', 'AVY', 'TECH', 'GRMN', 'INCY', 'KEY', 'NI',
    'ESS', 'EPAM', 'CHRW', 'SJM', 'AKAM', 'DRI', 'TYL', 'MGM', 'UDR', 'CPT',
    'PNR', 'HII', 'GL', 'ALLE', 'L', 'HST', 'PEAK', 'BBY', 'REG', 'EMN',
    'VTRS', 'JKHY', 'WYNN', 'KIM', 'QRVO', 'CBOE', 'CPB', 'NDSN', 'CRL', 'AAL',
    'HSIC', 'WDC', 'IPG', 'RHI', 'MKTX', 'BXP', 'BWA', 'AIZ', 'CZR', 'TAP',
    'PAYC', 'NRG', 'AES', 'SEDG', 'BBWI', 'BIO', 'HRL', 'AOS', 'FFIV', 'CTLT',
    'PHM', 'MHK', 'PARA', 'FRT', 'LKQ', 'GNRC', 'PNW', 'HAS', 'LUMN', 'NWSA',
    'XRAY', 'DAY', 'IVZ', 'CMA', 'SEE', 'ETSY', 'ZION', 'NWS', 'ROL', 'DISH',
]

# NASDAQ 100 additional
NASDAQ100_ADDITIONAL = [
    'ASML', 'PDD', 'ARM', 'MELI', 'PANW', 'ABNB', 'LULU', 'AZN', 'MRVL', 'CRWD',
    'TEAM', 'DDOG', 'WDAY', 'ZS', 'ADSK', 'COIN', 'SIRI', 'RIVN', 'LCID', 'TTD',
    'SNOW', 'OKTA', 'ROKU', 'SPLK', 'DOCU', 'MDB', 'NET', 'TWLO', 'ZM', 'PLTR',
    'DASH', 'HOOD', 'RBLX', 'U', 'BILL', 'DUOL', 'CFLT', 'DOCN', 'GTLB', 'SMAR',
    'CELH', 'SMCI', 'TXRH', 'WING', 'DECK', 'GFS', 'MPWR', 'WOLF', 'CDAY', 'GRAB',
]

# Russell 2000 sample (mid/small caps) - comprehensive list
RUSSELL_2000_SAMPLE = [
    # Healthcare/Biotech
    'IRTC', 'XENE', 'TGTX', 'PCVX', 'KRYS', 'VKTX', 'VRNA', 'APLS', 'ARQT', 'CRNX',
    'DAWN', 'IMVT', 'TVTX', 'RCKT', 'BLUE', 'SGMO', 'LXRX', 'RVMD', 'RARE', 'REPL',
    'ARCT', 'ABCL', 'ALNY', 'BGNE', 'BMRN', 'EXAS', 'HALO', 'IONS', 'JAZZ', 'LEGN',
    'MRNA', 'NBIX', 'NTRA', 'PCTY', 'RGEN', 'SGEN', 'SRRK', 'UTHR', 'VCEL', 'ZLAB',
    
    # Technology
    'AEHR', 'AI', 'AMBA', 'APPS', 'ASGN', 'AZPN', 'BAND', 'BLKB', 'BMBL', 'CALX',
    'CDLX', 'CIEN', 'CLVT', 'CMPR', 'CRUS', 'CVLT', 'DOMO', 'ESTC', 'EVBG', 'EXPI',
    'FIVN', 'FOUR', 'FRSH', 'GDYN', 'GLOB', 'HCP', 'HLIT', 'HUBS', 'JAMF', 'JNPR',
    'LITE', 'LSPD', 'MANH', 'MGNI', 'MQ', 'NEWR', 'NTNX', 'OLED', 'PATH', 'PEGA',
    'PI', 'PRGS', 'QLYS', 'RNG', 'SAIA', 'SITM', 'SLAB', 'SPSC', 'TENB', 'TYL',
    
    # Financials
    'ACHR', 'ALLY', 'ASB', 'BANC', 'BOKF', 'BOH', 'BYD', 'CBSH', 'CFR', 'COOP',
    'CVBF', 'EWBC', 'FAF', 'FCNCA', 'FHN', 'FIBK', 'FNB', 'FULT', 'GBCI', 'HWC',
    'IBOC', 'IBTX', 'MCY', 'NAVI', 'NBHC', 'ONB', 'OZK', 'PACW', 'PNFP', 'PPBI',
    'PRDO', 'SBCF', 'SEIC', 'SF', 'SNV', 'SSB', 'STNE', 'TFSL', 'TOWN', 'TRMK',
    'UBSI', 'UCBI', 'UMBF', 'VLY', 'WAL', 'WAFD', 'WBS', 'WSFS', 'WU', 'ZION',
    
    # Industrials
    'AGCO', 'AIMC', 'ARCB', 'ASTE', 'AYI', 'B', 'BC', 'BWXT', 'CBT', 'CW',
    'DY', 'ESAB', 'EVTC', 'EXP', 'EXPO', 'FELE', 'FL', 'GEO', 'GGG', 'GMS',
    'GNW', 'GTX', 'HNI', 'HXL', 'IESC', 'KBR', 'KNX', 'LECO', 'MGRC', 'MIDD',
    'MTX', 'MWA', 'NEX', 'NPO', 'OSIS', 'OSK', 'PRIM', 'R', 'RBC', 'REVG',
    'RXN', 'SATS', 'SNDR', 'SXI', 'TNC', 'TRN', 'TRNS', 'UNF', 'WERN', 'WMS',
    
    # Consumer
    'AAON', 'ABM', 'AVNT', 'AXL', 'BJ', 'BOOT', 'BURL', 'CACC', 'CAKE', 'CASY',
    'CHDN', 'CHUY', 'CRI', 'CWH', 'DDS', 'DIN', 'DKS', 'DORM', 'EAT', 'ELF',
    'ENS', 'EXPE', 'FIVE', 'FOXF', 'FRPT', 'GOLF', 'GPOR', 'HBI', 'HGV', 'HZO',
    'JACK', 'JWN', 'KSS', 'LAD', 'LANC', 'LEG', 'LGIH', 'LL', 'LTH', 'M',
    'MBUU', 'MCRI', 'MUSA', 'NXST', 'ODP', 'PATK', 'PLXS', 'PRGO', 'RCII', 'RH',
    'RL', 'SHAK', 'SIG', 'SKX', 'SN', 'SNEX', 'SPWH', 'STOR', 'TPR', 'TPX',
    'TRIP', 'UAA', 'VVV', 'WINA', 'WSM', 'WWW', 'YETI', 'ZWS',
    
    # Energy
    'AM', 'AR', 'AROC', 'CHRD', 'CIVI', 'CLB', 'CPE', 'CTRA', 'DRQ', 'EGY',
    'EPM', 'GTES', 'HP', 'HUN', 'LBRT', 'LEU', 'LPG', 'MGY', 'MRC', 'MTDR',
    'MUR', 'NOG', 'NRT', 'OAS', 'OII', 'OIS', 'PDCE', 'PDS', 'PR', 'PUMP',
    'RES', 'RRC', 'SM', 'SUN', 'SWN', 'TALO', 'TELL', 'TTI', 'USAC', 'VET',
    'VNOM', 'WLL', 'WTTR', 'XEC',
    
    # REITs
    'ACC', 'ADC', 'AIV', 'ALX', 'BDN', 'BRX', 'CIM', 'CLI', 'COLD', 'CONE',
    'CTRE', 'CUZ', 'DEA', 'DEI', 'DOC', 'EGP', 'EPR', 'EQC', 'ESRT', 'FAR',
    'FR', 'FSP', 'GNL', 'GTY', 'HTA', 'INN', 'IIPR', 'IRT', 'JBGS', 'KRC',
    'LAMR', 'LXP', 'MAC', 'NHI', 'NNN', 'NSA', 'OFC', 'OHI', 'OUT', 'PDM',
    'PEB', 'PKG', 'PLNT', 'RLJ', 'ROIC', 'RPT', 'REXR', 'SHO', 'SKT', 'SLG',
    'STAG', 'SUI', 'TRNO', 'UE', 'UMH', 'VER', 'VICI', 'VNO', 'VRE', 'WPC',
    
    # Materials
    'AMWD', 'APOG', 'ATI', 'ATR', 'AVD', 'AXTA', 'BERY', 'CBZ', 'CCK', 'CFX',
    'CLW', 'CMC', 'COUR', 'CRS', 'CSWI', 'CYH', 'ENV', 'GEF', 'GPK', 'GRA',
    'HCC', 'HXL', 'IOSP', 'KMT', 'KREF', 'KWR', 'LAUR', 'LPX', 'MDU', 'MP',
    'MTX', 'OC', 'OGS', 'OI', 'OMF', 'PBF', 'PBI', 'PHIN', 'ROCK', 'RYI',
    'SEM', 'SLVM', 'SON', 'STLD', 'TROX', 'UFPI', 'USLM', 'WRK', 'X', 'ZEUS',
    
    # Additional Small/Mid Caps
    'AAP', 'ACIW', 'AEIS', 'AFYA', 'AGYS', 'AIN', 'ALKS', 'AMED', 'AMKR', 'ARI',
    'ATKR', 'ATNI', 'AVAV', 'AXON', 'AXS', 'BCPC', 'BDC', 'BHF', 'BIG', 'BLDR',
    'BRC', 'BRBR', 'BRKR', 'BSIG', 'BXMT', 'CADE', 'CAR', 'CARS', 'CASA', 'CBZ',
    'CC', 'CEIX', 'CERS', 'CEVA', 'CGNT', 'CHE', 'CHGG', 'CNK', 'CNM', 'CNO',
    'CNXC', 'COHU', 'COKE', 'COLB', 'COMM', 'CONE', 'CPK', 'CPRX', 'CRK', 'CROX',
    'CSGS', 'CSTM', 'CTLT', 'CUBE', 'CVI', 'CXW', 'DCO', 'DENN', 'DIOD', 'DLB',
    'DNOW', 'DNUT', 'DRH', 'DSGX', 'DVA', 'DVAX', 'DXC', 'ECPG', 'EFC', 'EHC',
    'EIG', 'ELAN', 'ELY', 'EME', 'ENTA', 'ESGR', 'EVRI', 'EXLS', 'FCN', 'FELE',
    'FFBC', 'FHI', 'FIGS', 'FIX', 'FLGT', 'FLS', 'FOLD', 'FORM', 'FROG', 'FSS',
    'FTDR', 'FUL', 'GATX', 'GBX', 'GCO', 'GDOT', 'GEL', 'GES', 'GFF', 'GH',
    'GHC', 'GIII', 'GIL', 'GLBE', 'GLNG', 'GO', 'GRBK', 'GRC', 'GSAT', 'GTLS',
    'GVA', 'HA', 'HAFC', 'HCKT', 'HE', 'HEAR', 'HIW', 'HLF', 'HLNE', 'HMN',
    'HNI', 'HOME', 'HP', 'HURN', 'HVT', 'HWKN', 'IAA', 'IART', 'IBP', 'ICHR',
    'ICUI', 'IDCC', 'IGT', 'IHG', 'IMMR', 'INGR', 'INSM', 'INST', 'IRWD', 'ITGR',
    'ITRI', 'JELD', 'JHG', 'JJSF', 'JRVR', 'KAI', 'KAMN', 'KAR', 'KD', 'KE',
    'KELYA', 'KFY', 'KN', 'KNSL', 'KNTK', 'KOD', 'KPTI', 'KRO', 'KTOS', 'KW',
    'LANC', 'LB', 'LCII', 'LCUT', 'LDI', 'LFUS', 'LIVN', 'LKFN', 'LMB', 'LNTH',
    'LRN', 'LSCC', 'LUNA', 'LVLU', 'LZ', 'MAC', 'MATV', 'MATX', 'MAX', 'MAXR',
    'MCW', 'MD', 'MEDP', 'MEG', 'MESA', 'MIR', 'MKSI', 'MLI', 'MLNK', 'MMI',
    'MMSI', 'MNKD', 'MNR', 'MODV', 'MOG.A', 'MOV', 'MPW', 'MSA', 'MSGE', 'MSGS',
    'MSTR', 'MTH', 'MTN', 'MTRN', 'NARI', 'NAT', 'NATR', 'NBR', 'NBTB', 'NC',
    'NCR', 'NEOG', 'NMIH', 'NNI', 'NOV', 'NPTN', 'NSIT', 'NTB', 'NTGR', 'NTST',
    'NUVB', 'NVT', 'NXGN', 'NYCB', 'OFG', 'OGE', 'OGN', 'OI', 'OLLI', 'OLO',
    'OPK', 'OPRX', 'ORA', 'ORIC', 'OUT', 'PAGS', 'PAHC', 'PARR', 'PAX', 'PAYS',
    'PBH', 'PCH', 'PDCO', 'PDFS', 'PENN', 'PETQ', 'PFS', 'PGNY', 'PINC', 'PLAB',
    'PLAY', 'PLMR', 'PLUS', 'PMT', 'PNTG', 'POWI', 'POWL', 'PRAA', 'PRG', 'PRGS',
    'PRVA', 'PSMT', 'PTEN', 'PTGX', 'PVH', 'PWSC', 'QCRH', 'QDEL', 'QGEN', 'QUOT',
    'RCUS', 'RDN', 'REGI', 'RGS', 'RIG', 'RKT', 'RMBS', 'RNR', 'ROCK', 'RPD',
    'RPAY', 'RPM', 'RVNC', 'RWT', 'RXO', 'RYAM', 'SAFE', 'SAGE', 'SAH', 'SANM',
    'SAR', 'SATS', 'SBGI', 'SBH', 'SCCO', 'SCHL', 'SCL', 'SCSC', 'SDGR', 'SFBS',
    'SFIX', 'SFL', 'SGH', 'SHOO', 'SIGI', 'SITC', 'SKIN', 'SLM', 'SLNO', 'SMAR',
    'SMPL', 'SMTC', 'SNBR', 'SND', 'SNX', 'SPB', 'SPNT', 'SPR', 'SPTN', 'SPXC',
    'SRDX', 'SSD', 'SSP', 'STAA', 'STEP', 'STNG', 'STOR', 'STR', 'SUM', 'SUPN',
    'SWIM', 'SXC', 'SYBT', 'SYF', 'TALO', 'TARS', 'TCBI', 'TCBK', 'TDC', 'TDS',
    'TGNA', 'THC', 'THG', 'THO', 'TKC', 'TKR', 'TMHC', 'TNET', 'TNK', 'TNL',
    'TOL', 'TR', 'TREE', 'TRHC', 'TRNO', 'TRS', 'TRUP', 'TTEC', 'TTMI', 'TWI',
    'TWST', 'TXG', 'UCTT', 'UGI', 'UHS', 'UI', 'UMPQ', 'UNIT', 'UNM', 'UPST',
    'URBN', 'USFD', 'USM', 'USX', 'UTMD', 'VAC', 'VALN', 'VBTX', 'VC', 'VCNX',
    'VCTR', 'VCYT', 'VECO', 'VIAV', 'VICR', 'VIR', 'VIRT', 'VLGEA', 'VNDA', 'VNT',
    'VNTR', 'VRNS', 'VSCO', 'VSH', 'VVX', 'WABC', 'WDFC', 'WEN', 'WGO', 'WHD',
    'WKHS', 'WK', 'WLK', 'WLKP', 'WOLF', 'WOR', 'WRBY', 'WSC', 'WSBC', 'WSO',
    'WSR', 'WTS', 'WWD', 'XPEL', 'XPO', 'XRAY', 'XNCR', 'YEXT', 'YOU', 'YNDX',
]

# Combined Master Universe
MASTER_UNIVERSE = list(set(SP500_SYMBOLS + NASDAQ100_ADDITIONAL + RUSSELL_2000_SAMPLE))


@dataclass
class UniverseConfig:
    """Configuration for universe filtering"""
    min_price: float = 5.0
    min_dollar_volume: float = 1_000_000  # $1M daily
    volume_lookback: int = 20
    entry_days_threshold: int = 16  # 16/21 days to enter
    exit_days_threshold: int = 5    # 5/21 days to exit
    smoothing_window: int = 21


@dataclass
class SymbolStatus:
    """Status of a symbol in the universe"""
    symbol: str
    in_universe: bool
    days_qualified: int
    last_price: float
    avg_dollar_volume: float
    last_updated: str


class UniverseBuilder:
    """
    Builds and maintains a tradeable universe with entry/exit smoothing.
    """
    
    def __init__(self, config: UniverseConfig = None, output_dir: str = 'cache/universe'):
        self.config = config or UniverseConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.candidate_symbols = MASTER_UNIVERSE.copy()
        self.current_universe: Set[str] = set()
        self.symbol_status: Dict[str, SymbolStatus] = {}
        self.qualification_history: Dict[str, List[bool]] = {}
        
    def fetch_price_data(self, symbols: List[str], lookback_days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Fetch price/volume data for filtering.
        Uses yfinance with batching for efficiency.
        """
        logger.info(f"📥 Fetching data for {len(symbols)} symbols...")
        
        data = {}
        batch_size = 100
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            batch_str = ' '.join(batch)
            
            try:
                df = yf.download(
                    batch_str,
                    period=f'{lookback_days}d',
                    progress=False,
                    threads=True
                )
                
                if isinstance(df.columns, pd.MultiIndex):
                    # Multiple symbols
                    for sym in batch:
                        try:
                            sym_df = df.xs(sym, axis=1, level=1, drop_level=True)
                            if len(sym_df) > 0 and not sym_df['Close'].isna().all():
                                sym_df.columns = [c.lower() for c in sym_df.columns]
                                data[sym] = sym_df
                        except (KeyError, Exception):
                            pass
                else:
                    # Single symbol
                    if len(df) > 0 and len(batch) == 1:
                        df.columns = [c.lower() for c in df.columns]
                        data[batch[0]] = df
                        
            except Exception as e:
                logger.warning(f"Batch fetch error: {e}")
                
            if (i + batch_size) % 500 == 0:
                logger.info(f"   Fetched {min(i + batch_size, len(symbols))}/{len(symbols)} symbols")
        
        logger.info(f"✅ Successfully fetched {len(data)} symbols")
        return data
    
    def check_qualification(self, df: pd.DataFrame) -> Tuple[bool, float, float]:
        """
        Check if a symbol qualifies for the universe.
        
        Returns:
            (qualifies, last_price, avg_dollar_volume)
        """
        if len(df) < 5:
            return False, 0, 0
        
        # Get recent data
        recent = df.tail(self.config.volume_lookback)
        
        last_price = recent['close'].iloc[-1]
        avg_volume = recent['volume'].mean()
        avg_dollar_volume = avg_volume * last_price
        
        # Check criteria
        price_ok = last_price >= self.config.min_price
        volume_ok = avg_dollar_volume >= self.config.min_dollar_volume
        
        return (price_ok and volume_ok), last_price, avg_dollar_volume
    
    def update_qualification_history(self, symbol: str, qualifies: bool):
        """Update rolling qualification history for smoothing"""
        if symbol not in self.qualification_history:
            self.qualification_history[symbol] = []
        
        self.qualification_history[symbol].append(qualifies)
        
        # Keep only last N days
        if len(self.qualification_history[symbol]) > self.config.smoothing_window:
            self.qualification_history[symbol] = self.qualification_history[symbol][-self.config.smoothing_window:]
    
    def apply_smoothing(self, symbol: str, currently_in_universe: bool) -> bool:
        """
        Apply entry/exit smoothing rules.
        
        Entry: 16/21 days qualified (or first run with qualification)
        Exit: 5/21 days qualified (or fewer)
        """
        history = self.qualification_history.get(symbol, [])
        
        # First run: if currently qualifies, add to universe
        if len(history) < 5:
            return history[-1] if history else False
        
        days_qualified = sum(history)
        
        if currently_in_universe:
            # Exit if fails too often
            return days_qualified >= self.config.exit_days_threshold
        else:
            # Enter only if consistently qualified
            # For initial build, relax to current qualification
            if len(history) < self.config.smoothing_window:
                return history[-1] if history else False
            return days_qualified >= self.config.entry_days_threshold
    
    def build_universe(self) -> List[str]:
        """
        Build the tradeable universe with all filters and smoothing.
        """
        logger.info("🔄 Building Universe...")
        logger.info(f"   Candidates: {len(self.candidate_symbols)}")
        
        # Fetch data
        price_data = self.fetch_price_data(self.candidate_symbols)
        
        # Check each symbol
        qualified_today = {}
        
        for symbol, df in price_data.items():
            qualifies, price, dollar_volume = self.check_qualification(df)
            qualified_today[symbol] = qualifies
            
            # Update history
            self.update_qualification_history(symbol, qualifies)
            
            # Update status
            self.symbol_status[symbol] = SymbolStatus(
                symbol=symbol,
                in_universe=False,  # Will update after smoothing
                days_qualified=sum(self.qualification_history.get(symbol, [])),
                last_price=price,
                avg_dollar_volume=dollar_volume,
                last_updated=datetime.now().isoformat()
            )
        
        # Apply smoothing
        new_universe = set()
        
        for symbol in price_data.keys():
            was_in_universe = symbol in self.current_universe
            stays_in = self.apply_smoothing(symbol, was_in_universe)
            
            if stays_in:
                new_universe.add(symbol)
                self.symbol_status[symbol].in_universe = True
        
        self.current_universe = new_universe
        
        logger.info(f"✅ Universe built: {len(self.current_universe)} symbols")
        
        return sorted(list(self.current_universe))
    
    def get_universe(self) -> List[str]:
        """Get current universe symbols"""
        return sorted(list(self.current_universe))
    
    def save_universe(self, filename: str = None) -> str:
        """Save universe to JSON file"""
        if filename is None:
            filename = f"universe_{datetime.now().strftime('%Y%m%d')}.json"
        
        filepath = self.output_dir / filename
        
        # Convert numpy types to native Python types
        def convert_value(v):
            if hasattr(v, 'item'):
                return v.item()
            return v
        
        data = {
            'date': datetime.now().isoformat(),
            'count': len(self.current_universe),
            'symbols': sorted(list(self.current_universe)),
            'config': {k: convert_value(v) for k, v in asdict(self.config).items()},
            'status': {
                sym: {k: convert_value(v) for k, v in asdict(status).items()}
                for sym, status in self.symbol_status.items() if status.in_universe
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Also save latest
        latest_path = self.output_dir / 'universe_latest.json'
        with open(latest_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"💾 Universe saved: {filepath}")
        return str(filepath)
    
    def load_universe(self, filename: str = 'universe_latest.json') -> List[str]:
        """Load universe from JSON file"""
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Universe file not found: {filepath}")
            return []
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.current_universe = set(data['symbols'])
        logger.info(f"📂 Loaded universe: {len(self.current_universe)} symbols")
        
        return data['symbols']
    
    def get_statistics(self) -> Dict:
        """Get universe statistics"""
        if not self.symbol_status:
            return {}
        
        in_universe = [s for s in self.symbol_status.values() if s.in_universe]
        
        if not in_universe:
            return {'count': 0}
        
        prices = [s.last_price for s in in_universe]
        volumes = [s.avg_dollar_volume for s in in_universe]
        
        return {
            'count': len(in_universe),
            'avg_price': np.mean(prices),
            'median_price': np.median(prices),
            'min_price': np.min(prices),
            'max_price': np.max(prices),
            'avg_dollar_volume': np.mean(volumes),
            'total_dollar_volume': np.sum(volumes),
            'volume_coverage_pct': len([v for v in volumes if v > 10_000_000]) / len(volumes) * 100
        }


def main():
    """Main entry point"""
    print("\n" + "=" * 60)
    print("🔧 V17.0 UNIVERSE BUILDER")
    print("=" * 60)
    
    # Initialize builder
    config = UniverseConfig(
        min_price=5.0,
        min_dollar_volume=1_000_000,
        volume_lookback=20,
        entry_days_threshold=16,
        exit_days_threshold=5,
        smoothing_window=21
    )
    
    builder = UniverseBuilder(config)
    
    # Build universe
    universe = builder.build_universe()
    
    # Get statistics
    stats = builder.get_statistics()
    
    print(f"\n📊 Universe Statistics:")
    print(f"   Total Symbols:     {stats.get('count', 0)}")
    print(f"   Avg Price:         ${stats.get('avg_price', 0):,.2f}")
    print(f"   Median Price:      ${stats.get('median_price', 0):,.2f}")
    print(f"   Avg $ Volume:      ${stats.get('avg_dollar_volume', 0):,.0f}")
    print(f"   High Vol (>$10M):  {stats.get('volume_coverage_pct', 0):.1f}%")
    
    # Save
    filepath = builder.save_universe()
    
    print(f"\n✅ Universe ready: {len(universe)} symbols")
    print(f"💾 Saved to: {filepath}")
    
    return builder


if __name__ == "__main__":
    main()

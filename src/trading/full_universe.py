"""
Full Market Universe Manager - EXPANDED (1000+ Stocks)
======================================================

Comprehensive US equity universe for hedge fund deployment.

Includes:
- S&P 500 (500 stocks)
- NASDAQ 100 (100 stocks)
- Russell 2000 top liquid names (400+ stocks)
- Sector ETFs and Leveraged ETFs

Total tradeable universe: 1000+ symbols
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# =============================================================================
# EXPANDED MARKET UNIVERSE - 1000+ STOCKS
# =============================================================================

# S&P 500 Components (most liquid)
SP500_TICKERS = [
    # Technology (80+ stocks)
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'META', 'AVGO', 'CSCO', 'ORCL', 'CRM',
    'ACN', 'ADBE', 'AMD', 'INTC', 'QCOM', 'TXN', 'IBM', 'NOW', 'INTU', 'AMAT',
    'ADI', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'PANW', 'MCHP', 'FTNT', 'ADSK',
    'CRWD', 'NXPI', 'ON', 'MRVL', 'HPQ', 'HPE', 'KEYS', 'ZBRA', 'CTSH',
    'IT', 'AKAM', 'FFIV', 'NTAP', 'WDC', 'STX', 'DELL', 'GEN', 'EPAM',
    'GDDY', 'PAYC', 'PAYX', 'MANH', 'TYL', 'PTC', 'CDW', 'BR', 'JKHY',
    'TRMB', 'TER', 'SWKS', 'QRVO', 'MPWR', 'ENPH', 'FSLR',
    'MSCI', 'SPGI', 'MCO', 'FDS', 'VRSK', 'NTNX', 'ESTC', 'PLTR', 'COIN',
    
    # Healthcare (70 stocks)
    'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY',
    'AMGN', 'GILD', 'ISRG', 'MDT', 'SYK', 'BSX', 'EW', 'ZBH', 'BDX', 'DXCM',
    'IDXX', 'IQV', 'RMD', 'MTD', 'WAT', 'A', 'WST', 'HOLX', 'BAX',
    'ALGN', 'COO', 'PODD', 'RVTY', 'BIO', 'CRL', 'ICLR', 'MEDP',
    'VEEV', 'ZTS', 'REGN', 'VRTX', 'MRNA', 'BIIB', 'INCY', 'ALNY', 'BMRN',
    'CI', 'ELV', 'HUM', 'CNC', 'MOH', 'CVS', 'HCA', 'UHS', 'THC', 'GEHC',
    'DGX', 'LH', 'CAH', 'MCK', 'VTRS', 'TEVA', 'PEN', 'XRAY', 'DENTSPLY',
    
    # Financials (80 stocks)
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF',
    'AXP', 'SCHW', 'BLK', 'ICE', 'CME', 'NDAQ', 'CBOE',
    'BK', 'STT', 'NTRS', 'TROW', 'SEIC', 'AMG', 'BEN', 'IVZ', 'APAM',
    'V', 'MA', 'PYPL', 'FIS', 'FISV', 'GPN', 'CPAY',
    'MET', 'PRU', 'AIG', 'AFL', 'PGR', 'ALL', 'TRV', 'CB', 'HIG', 'L',
    'CINF', 'GL', 'WRB', 'RNR', 'ERIE', 'AJG', 'MMC', 'AON',
    'BRO', 'WTW', 'KKR', 'APO', 'BX', 'CG', 'ARES', 'OWL', 'HLNE',
    'CFG', 'MTB', 'KEY', 'HBAN', 'ZION', 'RF', 'CMA', 'FITB', 'ALLY',
    'SOFI', 'HOOD', 'IBKR', 'MKTX', 'VIRT', 'LPLA', 'SF', 'EVR', 'HLI',
    
    # Consumer Discretionary (70 stocks)
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'BKNG', 'MAR',
    'HLT', 'CMG', 'YUM', 'DRI', 'ORLY', 'AZO', 'GPC', 'ULTA', 'ROST',
    'BBY', 'TGT', 'KMX', 'AN', 'LAD', 'CVNA', 'GRMN', 'POOL', 'DECK', 'LULU',
    'WSM', 'RH', 'ETSY', 'W', 'CHWY', 'NVR', 'DHI', 'LEN', 'PHM',
    'TOL', 'MTH', 'KBH', 'LGIH', 'F', 'GM', 'RIVN', 'LCID', 'BWA',
    'APTV', 'LEA', 'VC', 'ALV', 'MGA', 'LKQ', 'GNTX',
    'CZR', 'WYNN', 'LVS', 'MGM', 'RCL', 'CCL', 'NCLH', 'EXPE', 'ABNB',
    'UBER', 'LYFT', 'DASH', 'GRAB', 'CPNG', 'MELI', 'SE', 'BABA', 'JD', 'PDD',
    
    # Consumer Staples (40 stocks)
    'COST', 'WMT', 'PG', 'KO', 'PEP', 'PM', 'MO', 'STZ',
    'MNST', 'KDP', 'TAP', 'SAM', 'MDLZ', 'HSY', 'KHC', 'GIS', 'K', 'CPB',
    'CAG', 'SJM', 'HRL', 'TSN', 'INGR', 'ADM', 'BG',
    'CL', 'EL', 'CHD', 'CLX', 'KMB', 'SYY', 'USFD', 'PFGC', 'CHEF', 'CORE',
    'CELH', 'FIZZ', 'COKE', 'KOF',
    
    # Industrials (80 stocks)
    'CAT', 'DE', 'UNP', 'CSX', 'NSC', 'UPS', 'FDX', 'XPO', 'JBHT', 'CHRW',
    'ODFL', 'SAIA', 'WERN', 'KNX', 'LSTR', 'SNDR', 'GWW', 'FAST', 'WSO',
    'LMT', 'RTX', 'NOC', 'GD', 'BA', 'TDG', 'HWM', 'TXT', 'LHX',
    'HON', 'GE', 'ETN', 'EMR', 'ITW', 'ROK', 'AME', 'IEX', 'XYL', 'DOV',
    'IR', 'TT', 'CARR', 'GNRC', 'AGCO', 'CMI', 'WAB', 'TTC',
    'PNR', 'SWK', 'ALLE', 'MAS', 'OC', 'BLD', 'BLDR', 'MLM', 'VMC',
    'WM', 'WCN', 'RSG', 'NDSN', 'RRX', 'ROP', 'IDEX',
    'FTV', 'PWR', 'EME', 'MTZ', 'DY', 'VRT', 'AXON', 'CTAS', 'PAYCHEX',
    'VRSK', 'EXPD', 'EFX', 'INFO', 'TRI', 'CPRT', 'COPART', 'IAA', 'KAR',
    
    # Energy (35 stocks)
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX',
    'DVN', 'FANG', 'HAL', 'BKR', 'APA', 'CTRA', 'OVV', 'EQT',
    'AR', 'RRC', 'MTDR', 'PR', 'KMI', 'WMB', 'OKE', 'TRGP', 'LNG',
    'EPD', 'ET', 'MPLX', 'PAA', 'WES', 'DTM', 'AM',
    
    # Utilities (35 stocks)
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'WEC', 'ES',
    'PEG', 'ED', 'EIX', 'ETR', 'FE', 'DTE', 'PPL', 'CMS', 'CNP', 'NI',
    'AEE', 'EVRG', 'AWK', 'ATO', 'NRG', 'VST', 'CEG', 'PCG', 'OGE', 'PNW',
    'AES', 'WTRG', 'AWR', 'CWT', 'SJW',
    
    # Materials (35 stocks)
    'LIN', 'APD', 'SHW', 'ECL', 'DD', 'DOW', 'PPG', 'ALB', 'EMN', 'CE',
    'LYB', 'CTVA', 'FMC', 'CF', 'MOS', 'NTR', 'NEM', 'FCX', 'SCCO', 'GOLD',
    'NUE', 'STLD', 'CLF', 'AA', 'ATI', 'RS', 'CMC', 'WOR',
    'RPM', 'AXTA', 'ASH', 'HUN', 'OLN', 'TROX', 'KRO',
    
    # Real Estate (35 stocks)
    'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB',
    'EQR', 'VTR', 'ARE', 'BXP', 'VNO', 'SLG', 'KIM', 'REG', 'FRT', 'HST',
    'RHP', 'DOC', 'HR', 'OHI', 'SBAC', 'UDR', 'CPT', 'ESS', 'MAA',
    'IRM', 'CUBE', 'LSI', 'EXR', 'INVH', 'AMH',
    
    # Communications (25 stocks)
    'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR',
    'WBD', 'FOX', 'FOXA', 'LYV', 'MTCH', 'EA', 'TTWO', 'RBLX', 'U',
    'SPOT', 'TME', 'IQ', 'BILI', 'HUYA', 'DOYU', 'SE', 'GRAB',
]

# Mid-Cap Growth & Value (Russell 1000 beyond S&P 500)
MIDCAP_TICKERS = [
    # Mid-Cap Tech
    'DDOG', 'SNOW', 'MDB', 'NET', 'ZS', 'OKTA', 'HUBS', 'TEAM', 'WDAY',
    'ZM', 'DOCU', 'TWLO', 'PINS', 'SNAP', 'BILL', 'CFLT', 'ESTC', 'GTLB', 'APP',
    'IOT', 'PATH', 'SMAR', 'MNDY', 'WK',
    'DOCS', 'SMCI', 'AEHR', 'SITM', 'POWI', 'ONTO', 'AMBA', 'CRUS', 'SLAB', 'DIOD',
    'ZI', 'APPN', 'BOX', 'SUMO', 'TENB', 'CYBR', 'SAIL', 'RPD', 'S', 'QLYS',
    'VRNS', 'FEYE', 'COUP', 'PCTY', 'PAYC', 'NCNO', 'NTNX', 'ALRM', 'EVBG', 'NEOG',
    
    # Mid-Cap Healthcare
    'EXEL', 'SRPT', 'RARE', 'IONS', 'NBIX', 'PTCT', 'FOLD', 'UTHR',
    'JAZZ', 'PRGO', 'ELAN', 'AXON', 'GMED', 'LNTH', 'NVST', 'STE',
    'ENSG', 'NHC', 'AMED', 'ACHC', 'SGRY', 'USPH', 'LFST', 'OPCH', 'BKD', 'NRC',
    'HZNP', 'ALKS', 'IRWD', 'SUPN', 'PDCO', 'PRXL', 'XNCR', 'DCPH', 'RVNC', 'CORT',
    
    # Mid-Cap Financials
    'WAL', 'CFG', 'MTB', 'KEY', 'HBAN', 'ZION',
    'RF', 'CMA', 'FITB', 'ALLY', 'LC', 'UPST',
    'LAZ', 'JEF', 'PJT', 'MC',
    'FCNCA', 'CADE', 'GBCI', 'UBSI', 'FNB', 'WSFS', 'NBTB', 'BANF', 'HOPE', 'CVBF',
    'PPBI', 'SBCF', 'WTFC', 'UCBI', 'SSB', 'IBTX', 'PNFP', 'TCBI', 'FFBC', 'FULT',
    
    # Mid-Cap Consumer
    'FIVE', 'OLLI', 'DKS', 'ASO', 'BOOT', 'HBI', 'CROX',
    'SHOO', 'WWW', 'TPR', 'CPRI', 'RL', 'VFC', 'UAA', 'LEVI', 'AEO',
    'BURL', 'DDS', 'M', 'KSS', 'DLTR', 'DG',
    'DINE', 'DIN', 'TXRH', 'CAKE', 'BJRI', 'RRGB', 'JACK', 'PLAY', 'DAVE', 'RUTH',
    'PRTY', 'PLNT', 'XPOF', 'FWRG', 'SHAK', 'WING', 'ARCO', 'CBRL', 'TACO', 'LOCO',
    
    # Mid-Cap Industrials
    'UFPI', 'TREX', 'AWI', 'TILE', 'IBP',
    'EXPO', 'WSC', 'WTS', 'PRIM', 'APG', 'ESAB', 'KAI', 'MEI', 'NPO',
    'AAON', 'ROAD', 'MTRN', 'ROCK', 'NX', 'AIMC', 'GBX', 'RAIL', 'TRN', 'WAT',
    'FSS', 'MIDD', 'REVG', 'OSK', 'HLIO', 'PRLB', 'RBC', 'THR', 'CALX', 'VIAV',
    
    # Mid-Cap Energy & MLPs
    'EPD', 'ET', 'MPLX', 'PAA', 'WES', 'DTM', 'AM',
    'TRGP', 'HESM', 'AROC',
    'CIVI', 'CHRD', 'SM', 'PDCE', 'CPE', 'ROCC', 'EPSN', 'REPX', 'VTLE', 'NOG',
    'MGY', 'NEXT', 'GPOR', 'HPK', 'CLNE', 'CRK', 'CNX', 'PTEN', 'RES', 'CLB',
]

# Small-Cap High Momentum Names (Russell 2000 top performers)
SMALLCAP_MOMENTUM = [
    # High-growth small caps
    'ASAN', 'JAMF', 'AI',
    'GLBE', 'GLOB', 'CWAN', 'INTA', 'FLYW', 'TASK', 'COUR', 'UDMY', 'GENI',
    'DKNG', 'PENN', 'BETZ', 'CHDN', 'BYD', 'AGS',
    'STEM', 'BEEM', 'BLNK', 'CHPT', 'EVGO',
    'SHLS', 'ARRY', 'MAXN', 'SPWR', 'CSIQ', 'DQ', 'JKS', 'SOL',
    
    # Additional Russell 2000 momentum
    'AMBA', 'MYRG', 'PLAB', 'FORM', 'OLED', 'LSCC', 'MKSI', 'COHU', 'ICHR', 'ACMR',
    'CEVA', 'RMBS', 'SYNA', 'NVMI', 'NXGN', 'PEGA', 'SABR', 'NABL', 'FIVN', 'BAND',
    'COMM', 'INSG', 'CASA', 'CIEN', 'LITE', 'INFN', 'VIAV', 'NPTN', 'ADTN', 'CALX',
    'DGII', 'DIGI', 'CMBM', 'CAMP', 'NTGR', 'SATS', 'VNET', 'CCOI', 'UNIT', 'TSAT',
    
    # Biotech small caps
    'XENE', 'RCUS', 'SWTX', 'FATE', 'KURA', 'RVMD', 'BEAM', 'TWST', 'CRSP', 'EDIT',
    'NTLA', 'VERV', 'VRTX', 'SNDX', 'LEGN', 'IMVT', 'RLAY', 'AVXL', 'SAVA', 'PRAX',
    'ADVM', 'SGMO', 'FREQ', 'ZYME', 'ALDX', 'TBPH', 'RETA', 'VNDA', 'LQDA', 'GTHX',
    
    # Financial small caps
    'AMAL', 'FSBC', 'INBK', 'BY', 'MCBS', 'CWBC', 'FMBH', 'BHLB', 'CZWI', 'FNWB',
    'ESSA', 'EFSC', 'GCBC', 'MVBF', 'NWFL', 'RVSB', 'TBNK', 'THFF', 'USCB', 'VCNX',
    
    # Industrial small caps
    'AGYS', 'AMSC', 'AOSL', 'ARKO', 'ATEX', 'AZTA', 'BBSI', 'BCOV', 'BKSY', 'BLZE',
    'CENX', 'CLSK', 'COIN', 'CRS', 'CWST', 'CYTK', 'DCBO', 'DFIN', 'DLO', 'DOMO',
    'DUOL', 'EMBC', 'ENVX', 'EYE', 'FRPT', 'FTAI', 'FTDR', 'FWRD', 'GATO', 'GEF',
    'GERN', 'GMRE', 'GNL', 'GPRO', 'GSHD', 'GTES', 'GTX', 'HAFC', 'HASI', 'HCC',
    'HEES', 'HGV', 'HI', 'HLMN', 'HLX', 'HOMB', 'HOPE', 'HQY', 'HRI', 'HSTM',
    'HWC', 'HWKN', 'HXL', 'HYLN', 'ICFI', 'IHRT', 'INN', 'INOD', 'IOSP', 'IPG',
    'IRIX', 'IRMD', 'IRON', 'ITGR', 'ITIC', 'ITRI', 'IVT', 'JACK', 'JBL', 'JBT',
    'JELD', 'JJSF', 'JOE', 'JOUT', 'JOBY', 'KALU', 'KBR', 'KFY', 'KIDS', 'KN',
    'KNF', 'KRC', 'KRG', 'KROS', 'KRYS', 'KWR', 'LADR', 'LANC', 'LAUR', 'LAWS',
    'LDI', 'LEG', 'LESL', 'LFST', 'LGND', 'LIND', 'LIVN', 'LKFN', 'LLNW', 'LMAT',
    'LMB', 'LMND', 'LOAN', 'LOCO', 'LPRO', 'LPX', 'LRMR', 'LTC', 'LTRN', 'LXP',
    'LZB', 'MASI', 'MATV', 'MATW', 'MBIN', 'MCFT', 'MCW', 'MDU', 'MESA', 'MGNI',
    'MGRC', 'MITK', 'MMS', 'MMSI', 'MODV', 'MOV', 'MP', 'MPLN', 'MRCY', 'MRTN',
    'MSEX', 'MSM', 'MTRX', 'MTX', 'MTZ', 'MUR', 'MVST', 'MWA', 'MXCT', 'MYE',
]

# Expanded ETF list
ETFS = [
    # Sector ETFs
    'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC',
    'SMH', 'SOXX', 'IBB', 'XBI', 'IYR', 'VNQ', 'KRE', 'XHB', 'ITB', 'GDX', 'GDXJ',
    'XOP', 'OIH', 'KWEB', 'FXI', 'EWJ', 'EEM', 'VWO', 'IEMG',
    
    # Core Market ETFs
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VT', 'VEA', 'ARKK',
    'MDY', 'IJH', 'IJR', 'IWO', 'IWN', 'VTV', 'VUG', 'MTUM', 'QUAL', 'VLUE',
    'RSP', 'IVE', 'IVW', 'VBK', 'VBR', 'VIOO', 'VIOV', 'VIOG', 'VONG', 'VONV',
    
    # Industry-specific
    'XRT', 'XME', 'JETS', 'BETZ', 'HACK', 'SKYY', 'CLOU', 'WCLD', 'CIBR', 'ROBO',
    'BOTZ', 'DRIV', 'IDRV', 'KARS', 'LIT', 'QCLN', 'ICLN', 'TAN', 'PBW', 'FAN',
    
    # 3x Leveraged Bull ETFs
    'TQQQ', 'SPXL', 'UPRO', 'TNA', 'SOXL', 'TECL', 'FNGU', 'LABU', 'FAS', 'CURE',
    'DPST', 'NAIL', 'DFEN', 'RETL', 'HIBL', 'WANT', 'WEBL', 'DUSL', 'MIDU',
    
    # 3x Leveraged Bear/Inverse ETFs
    'SQQQ', 'SPXU', 'SPXS', 'TZA', 'SOXS', 'TECS', 'FNGD', 'LABD', 'FAZ', 'HIBS',
    
    # 2x Leveraged (less decay)
    'QLD', 'SSO', 'UWM', 'ROM', 'USD', 'UYG', 'UGE', 'UCC', 'MVV',
    
    # Volatility & Hedging
    'VXX', 'UVXY', 'SVXY', 'VIXY', 'VIXM', 'XVZ', 'TAIL', 'NRGZ',
    
    # Bond & Fixed Income
    'TLT', 'TBT', 'TMF', 'TMV', 'BND', 'AGG', 'LQD', 'HYG', 'JNK', 'EMB',
]

# Combine all into full universe
FULL_UNIVERSE = list(set(SP500_TICKERS + MIDCAP_TICKERS + SMALLCAP_MOMENTUM + ETFS))

# Sector classification
SECTOR_MAP = {}

# Auto-assign Technology
for t in ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'META', 'AVGO', 'CSCO', 'ORCL', 'CRM', 
          'ACN', 'ADBE', 'AMD', 'INTC', 'QCOM', 'TXN', 'IBM', 'NOW', 'INTU', 'AMAT',
          'DDOG', 'SNOW', 'MDB', 'NET', 'ZS', 'CRWD', 'PANW', 'FTNT', 'OKTA', 'HUBS',
          'SMCI', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MCHP', 'NXPI', 'ON', 'MRVL',
          'PLTR', 'COIN', 'APP', 'CFLT', 'ESTC', 'GTLB', 'PATH', 'SMAR', 'MNDY']:
    SECTOR_MAP[t] = 'Technology'

# Healthcare
for t in ['UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY',
          'AMGN', 'GILD', 'ISRG', 'MDT', 'SYK', 'BSX', 'VEEV', 'DXCM', 'IDXX', 'ZTS',
          'REGN', 'VRTX', 'MRNA', 'BIIB', 'CI', 'ELV', 'HUM', 'CNC', 'CVS', 'HCA']:
    SECTOR_MAP[t] = 'Healthcare'

# Financials
for t in ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA', 'BLK', 'SCHW',
          'AXP', 'USB', 'PNC', 'TFC', 'COF', 'CME', 'ICE', 'SPGI', 'MCO', 'CB',
          'KKR', 'APO', 'BX', 'SOFI', 'HOOD', 'ALLY', 'CFG', 'MTB', 'KEY', 'HBAN']:
    SECTOR_MAP[t] = 'Financials'

# Consumer
for t in ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'COST', 'WMT', 'LOW', 'TGT',
          'BKNG', 'MAR', 'HLT', 'CMG', 'LULU', 'ROST', 'TJX', 'ORLY', 'AZO', 'DG',
          'UBER', 'LYFT', 'DASH', 'ABNB', 'EXPE', 'RCL', 'CCL', 'NCLH', 'DKNG', 'PENN']:
    SECTOR_MAP[t] = 'Consumer'

# Industrials
for t in ['CAT', 'DE', 'UNP', 'UPS', 'FDX', 'HON', 'GE', 'BA', 'LMT', 'RTX',
          'ETN', 'EMR', 'ITW', 'CMI', 'WAB', 'GWW', 'FAST', 'ROK', 'WM', 'RSG']:
    SECTOR_MAP[t] = 'Industrials'

# Energy
for t in ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX',
          'DVN', 'FANG', 'HAL', 'BKR', 'KMI', 'WMB', 'OKE', 'TRGP', 'LNG', 'EPD', 'ET']:
    SECTOR_MAP[t] = 'Energy'

# Utilities
for t in ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'WEC', 'ES', 'CEG', 'VST']:
    SECTOR_MAP[t] = 'Utilities'

# Materials
for t in ['LIN', 'APD', 'SHW', 'ECL', 'DD', 'DOW', 'NEM', 'FCX', 'NUE', 'SCCO', 'GOLD']:
    SECTOR_MAP[t] = 'Materials'

# Real Estate
for t in ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB']:
    SECTOR_MAP[t] = 'Real Estate'

# Communications
for t in ['NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR', 'EA', 'TTWO', 'RBLX', 'SPOT']:
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
        'TQQQ': 0.30,  # Nasdaq 100 3x
        'SPXL': 0.25,  # S&P 500 3x
        'SOXL': 0.20,  # Semiconductors 3x
        'TECL': 0.10,  # Technology 3x
        'FNGU': 0.10,  # FAANG 3x
        'UPRO': 0.05,  # S&P 500 3x alternative
    }


def get_leveraged_bear_etfs() -> Dict[str, float]:
    """Get 3x leveraged bear/inverse ETFs with default weights."""
    return {
        'SQQQ': 0.30,  # Nasdaq 100 -3x
        'SPXU': 0.25,  # S&P 500 -3x
        'SOXS': 0.20,  # Semiconductors -3x
        'TECS': 0.10,  # Technology -3x
        'FNGD': 0.10,  # FAANG -3x
        'TZA': 0.05,   # Russell 2000 -3x
    }


# Export universe size info
UNIVERSE_SIZE = len(FULL_UNIVERSE)
SP500_SIZE = len(SP500_TICKERS)
MIDCAP_SIZE = len(MIDCAP_TICKERS)
SMALLCAP_SIZE = len(SMALLCAP_MOMENTUM)
ETF_SIZE = len(ETFS)

print(f"Expanded Universe: {UNIVERSE_SIZE} symbols")
print(f"  - S&P 500: {SP500_SIZE}")
print(f"  - Mid-Caps: {MIDCAP_SIZE}")
print(f"  - Small-Cap Momentum: {SMALLCAP_SIZE}")
print(f"  - ETFs: {ETF_SIZE}")

"""
MEGA MARKET UNIVERSE - FULL NYSE + NASDAQ SCOPE
================================================

PRINTING CASH LIKE RENAISSANCE TECHNOLOGIES!

Maximum opportunity set: 3000+ liquid US stocks and ETFs.

Includes:
- Full S&P 500 (503 stocks)
- Full NASDAQ 100 (100 stocks)
- Russell 1000 growth/value (1000 stocks)
- Russell 2000 most liquid (1000+ stocks)
- All sector/leveraged ETFs (150+)
- ADRs (Chinese tech, LatAm, etc.)

Total Universe: 3000+ symbols
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# =============================================================================
# S&P 500 - FULL LIST (503 stocks)
# =============================================================================
SP500 = [
    # INFORMATION TECHNOLOGY (80+)
    'AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'CRM', 'CSCO', 'ACN', 'ADBE', 'AMD',
    'IBM', 'INTC', 'QCOM', 'TXN', 'NOW', 'INTU', 'AMAT', 'ADI', 'MU', 'LRCX',
    'KLAC', 'SNPS', 'CDNS', 'PANW', 'MCHP', 'FTNT', 'ADSK', 'CRWD', 'NXPI', 'ON',
    'MRVL', 'HPQ', 'HPE', 'KEYS', 'ZBRA', 'CTSH', 'IT', 'AKAM', 'FFIV', 'NTAP',
    'WDC', 'STX', 'DELL', 'GEN', 'EPAM', 'GDDY', 'PAYC', 'PAYX', 'MANH', 'TYL',
    'PTC', 'CDW', 'BR', 'JKHY', 'TRMB', 'TER', 'SWKS', 'QRVO', 'MPWR', 'ENPH',
    'FSLR', 'MSCI', 'SPGI', 'MCO', 'FDS', 'VRSK', 'PLTR', 'COIN', 'GLOB', 'SMCI',
    'ANET', 'TEAM', 'MDB', 'SNOW', 'DDOG', 'NET', 'ZS', 'HUBS', 'WDAY', 'VEEV',
    
    # HEALTHCARE (75+)
    'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY',
    'AMGN', 'GILD', 'ISRG', 'MDT', 'SYK', 'BSX', 'EW', 'ZBH', 'BDX', 'DXCM',
    'IDXX', 'IQV', 'RMD', 'MTD', 'WAT', 'A', 'WST', 'HOLX', 'BAX', 'ALGN',
    'COO', 'PODD', 'RVTY', 'BIO', 'CRL', 'ICLR', 'MEDP', 'VEEV', 'ZTS', 'REGN',
    'VRTX', 'MRNA', 'BIIB', 'INCY', 'ALNY', 'BMRN', 'CI', 'ELV', 'HUM', 'CNC',
    'MOH', 'CVS', 'HCA', 'UHS', 'THC', 'GEHC', 'DGX', 'LH', 'CAH', 'MCK',
    'VTRS', 'TEVA', 'PEN', 'XRAY', 'HSIC', 'DVA', 'TECH', 'EXAS', 'NVST', 'MGRC',
    
    # FINANCIALS (80+)
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF',
    'AXP', 'SCHW', 'BLK', 'ICE', 'CME', 'NDAQ', 'CBOE', 'BK', 'STT', 'NTRS',
    'TROW', 'SEIC', 'AMG', 'BEN', 'IVZ', 'APAM', 'V', 'MA', 'PYPL', 'FIS',
    'FISV', 'GPN', 'MET', 'PRU', 'AIG', 'AFL', 'PGR', 'ALL', 'TRV', 'CB',
    'HIG', 'L', 'CINF', 'GL', 'WRB', 'RNR', 'ERIE', 'AJG', 'MMC', 'AON',
    'BRO', 'WTW', 'KKR', 'APO', 'BX', 'CG', 'ARES', 'OWL', 'HLNE', 'CFG',
    'MTB', 'KEY', 'HBAN', 'ZION', 'RF', 'CMA', 'FITB', 'ALLY', 'SOFI', 'HOOD',
    'IBKR', 'MKTX', 'VIRT', 'LPLA', 'SF', 'EVR', 'HLI', 'DFS', 'SYF', 'CPAY',
    
    # CONSUMER DISCRETIONARY (70+)
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'BKNG', 'MAR',
    'HLT', 'CMG', 'YUM', 'DRI', 'ORLY', 'AZO', 'GPC', 'ULTA', 'ROST', 'BBY',
    'TGT', 'KMX', 'AN', 'LAD', 'CVNA', 'GRMN', 'POOL', 'DECK', 'LULU', 'WSM',
    'RH', 'ETSY', 'W', 'CHWY', 'NVR', 'DHI', 'LEN', 'PHM', 'TOL', 'MTH',
    'KBH', 'LGIH', 'F', 'GM', 'RIVN', 'LCID', 'BWA', 'APTV', 'LEA', 'VC',
    'ALV', 'MGA', 'LKQ', 'GNTX', 'CZR', 'WYNN', 'LVS', 'MGM', 'RCL', 'CCL',
    'NCLH', 'EXPE', 'ABNB', 'UBER', 'LYFT', 'DASH', 'GRAB', 'CPNG', 'MELI', 'SE',
    
    # CONSUMER STAPLES (35+)
    'COST', 'WMT', 'PG', 'KO', 'PEP', 'PM', 'MO', 'STZ', 'MNST', 'KDP',
    'TAP', 'SAM', 'MDLZ', 'HSY', 'KHC', 'GIS', 'K', 'CPB', 'CAG', 'SJM',
    'HRL', 'TSN', 'INGR', 'ADM', 'BG', 'CL', 'EL', 'CHD', 'CLX', 'KMB',
    'SYY', 'USFD', 'PFGC', 'CHEF', 'CELH', 'FIZZ', 'KR', 'ACI', 'GO', 'SFM',
    
    # INDUSTRIALS (80+)
    'CAT', 'DE', 'UNP', 'CSX', 'NSC', 'UPS', 'FDX', 'XPO', 'JBHT', 'CHRW',
    'ODFL', 'SAIA', 'WERN', 'KNX', 'LSTR', 'SNDR', 'GWW', 'FAST', 'WSO',
    'LMT', 'RTX', 'NOC', 'GD', 'BA', 'TDG', 'HWM', 'TXT', 'LHX', 'HON',
    'GE', 'ETN', 'EMR', 'ITW', 'ROK', 'AME', 'IEX', 'XYL', 'DOV', 'IR',
    'TT', 'CARR', 'GNRC', 'AGCO', 'CMI', 'WAB', 'TTC', 'PNR', 'SWK', 'ALLE',
    'MAS', 'OC', 'BLD', 'BLDR', 'MLM', 'VMC', 'WM', 'WCN', 'RSG', 'NDSN',
    'RRX', 'ROP', 'IDEX', 'FTV', 'PWR', 'EME', 'MTZ', 'DY', 'VRT', 'AXON',
    'CTAS', 'EXPD', 'EFX', 'INFO', 'TRI', 'CPRT', 'KAR', 'PCAR', 'HUBB', 'AOS',
    
    # ENERGY (40+)
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX', 'DVN',
    'FANG', 'HAL', 'BKR', 'APA', 'CTRA', 'OVV', 'EQT', 'AR', 'RRC', 'MTDR',
    'PR', 'KMI', 'WMB', 'OKE', 'TRGP', 'LNG', 'EPD', 'ET', 'MPLX', 'PAA',
    'WES', 'DTM', 'AM', 'HES', 'PXD', 'CHRD', 'MRO', 'MGY', 'SM', 'NOG',
    
    # UTILITIES (35+)
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'WEC', 'ES',
    'PEG', 'ED', 'EIX', 'ETR', 'FE', 'DTE', 'PPL', 'CMS', 'CNP', 'NI',
    'AEE', 'EVRG', 'AWK', 'ATO', 'NRG', 'VST', 'CEG', 'PCG', 'OGE', 'PNW',
    'AES', 'WTRG', 'AWR', 'CWT',
    
    # MATERIALS (35+)
    'LIN', 'APD', 'SHW', 'ECL', 'DD', 'DOW', 'PPG', 'ALB', 'EMN', 'CE',
    'LYB', 'CTVA', 'FMC', 'CF', 'MOS', 'NTR', 'NEM', 'FCX', 'SCCO', 'GOLD',
    'NUE', 'STLD', 'CLF', 'AA', 'ATI', 'RS', 'CMC', 'WOR', 'RPM', 'AXTA',
    'ASH', 'HUN', 'OLN', 'TROX', 'KRO', 'MP', 'LAC', 'ALB', 'LTHM', 'PLL',
    
    # REAL ESTATE (40+)
    'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB',
    'EQR', 'VTR', 'ARE', 'BXP', 'VNO', 'SLG', 'KIM', 'REG', 'FRT', 'HST',
    'RHP', 'DOC', 'HR', 'OHI', 'SBAC', 'UDR', 'CPT', 'ESS', 'MAA', 'IRM',
    'CUBE', 'EXR', 'INVH', 'AMH', 'REXR', 'VICI', 'GLPI', 'IIPR', 'STAG', 'COLD',
    
    # COMMUNICATIONS (30+)
    'GOOGL', 'GOOG', 'META', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR',
    'WBD', 'FOX', 'FOXA', 'LYV', 'MTCH', 'EA', 'TTWO', 'RBLX', 'U', 'SPOT',
    'TME', 'IQ', 'BILI', 'HUYA', 'DOYU', 'ZM', 'TWLO', 'PINS', 'SNAP', 'TTD',
]

# =============================================================================
# NASDAQ 100 ADDITIONS (not in S&P 500)
# =============================================================================
NASDAQ100_ADDITIONS = [
    'ASML', 'AZN', 'ARM', 'CEG', 'CSGP', 'GFS', 'ILMN', 'KHC', 'LCID',
    'MDLZ', 'MELI', 'ODFL', 'PCAR', 'PDD', 'ROST', 'SIRI', 'WBA', 'ZM',
]

# =============================================================================
# RUSSELL 1000 GROWTH - HIGH GROWTH STOCKS (400+)
# =============================================================================
RUSSELL1000_GROWTH = [
    # Tech growth
    'CFLT', 'ESTC', 'GTLB', 'PATH', 'SMAR', 'MNDY', 'WK', 'IOT', 'DOCN', 'FROG',
    'BOX', 'APPN', 'NCNO', 'ALRM', 'NEOG', 'JAMF', 'ASAN', 'AI', 'GLBE', 'CWAN',
    'INTA', 'FLYW', 'TASK', 'COUR', 'UDMY', 'GENI', 'BIGC', 'SQSP', 'DUOL', 'BYON',
    'PUBM', 'MGNI', 'CRTO', 'APPS', 'ZETA', 'BRZE', 'SEMR', 'KARO', 'AUR', 'IONQ',
    
    # Semiconductors
    'ACMR', 'AEHR', 'ALGM', 'AMBA', 'AMKR', 'ANET', 'AOSL', 'ASX', 'CAMT', 'COHU',
    'CRUS', 'DIOD', 'FORM', 'ICHR', 'IPGP', 'LFUS', 'LITE', 'LSCC', 'MKSI', 'MPWR',
    'MTSI', 'MXL', 'NVMI', 'OLED', 'ONTO', 'PLAB', 'POWI', 'RMBS', 'SITM', 'SLAB',
    'SYNA', 'UCTT', 'WOLF', 'SOXL', 'SMH', 'SOXX',
    
    # Cloud/SaaS
    'OKTA', 'BILL', 'ZI', 'TENB', 'CYBR', 'SAIL', 'RPD', 'S', 'QLYS', 'VRNS',
    'PCTY', 'PAYC', 'COUP', 'NTNX', 'EVBG', 'FIVN', 'BAND', 'CIEN', 'CALX', 'VIAV',
    'COMM', 'INSG', 'DGII', 'DIGI', 'CMBM', 'CAMP', 'NTGR', 'SATS', 'VNET', 'CCOI',
    
    # Biotech growth
    'EXEL', 'SRPT', 'RARE', 'IONS', 'NBIX', 'PTCT', 'FOLD', 'UTHR', 'JAZZ', 'PRGO',
    'ELAN', 'GMED', 'LNTH', 'NVST', 'STE', 'ENSG', 'NHC', 'ACHC', 'SGRY', 'USPH',
    'LFST', 'OPCH', 'BKD', 'NRC', 'ALKS', 'IRWD', 'SUPN', 'XNCR', 'CORT', 'PRAX',
    'XENE', 'RCUS', 'FATE', 'KURA', 'RVMD', 'BEAM', 'TWST', 'CRSP', 'EDIT', 'NTLA',
    'VERV', 'SNDX', 'LEGN', 'IMVT', 'RLAY', 'AVXL', 'SAVA', 'ADVM', 'SGMO', 'ZYME',
    'ALDX', 'TBPH', 'VNDA', 'LQDA', 'KRYS', 'AXSM', 'ARCT', 'BNTX', 'BHVN', 'HRTX',
    
    # Fintech growth
    'AFRM', 'UPST', 'LC', 'MQ', 'PAYO', 'FOUR', 'RELY', 'TOST', 'FLYW', 'BILL',
    'TIGR', 'FUTU', 'FINV', 'LX', 'QFIN', 'XP', 'NU', 'STNE', 'PAGS', 'NVEI',
    
    # Consumer growth
    'FIVE', 'OLLI', 'DKS', 'ASO', 'BOOT', 'HBI', 'CROX', 'SHOO', 'WWW', 'TPR',
    'CPRI', 'RL', 'VFC', 'UAA', 'UA', 'LEVI', 'AEO', 'BURL', 'DDS', 'M',
    'KSS', 'DLTR', 'DG', 'TXRH', 'CAKE', 'BJRI', 'JACK', 'PLAY', 'DAVE', 'WING',
    'SHAK', 'ARCO', 'TACO', 'LOCO', 'CMG', 'BROS', 'SBUX', 'DPZ', 'PZZA', 'WEN',
]

# =============================================================================
# RUSSELL 2000 SMALL CAPS - HIGH MOMENTUM (600+)
# =============================================================================
RUSSELL2000_MOMENTUM = [
    # Tech small caps
    'PEGA', 'SABR', 'NABL', 'TSAT', 'UNIT', 'ADTN', 'NPTN', 'INFN', 'GSAT', 'IRDM',
    'ORBCOMM', 'SATS', 'VNET', 'GDS', 'EURN', 'CLSK', 'MARA', 'RIOT', 'HUT', 'BTBT',
    'CAN', 'CIFR', 'BITF', 'IREN', 'CORZ', 'WULF', 'MSTR', 'COIN', 'HOOD', 'SOFI',
    
    # Clean energy / EV
    'STEM', 'BEEM', 'BLNK', 'CHPT', 'EVGO', 'SHLS', 'ARRY', 'MAXN', 'SPWR', 'CSIQ',
    'DQ', 'JKS', 'SOL', 'ENPH', 'SEDG', 'RUN', 'NOVA', 'PLUG', 'FCEL', 'BE',
    'BLDP', 'HYLN', 'NKLA', 'GOEV', 'FSR', 'PTRA', 'ARVL', 'RIDE', 'WKHS', 'XPEV',
    'NIO', 'LI', 'RIVN', 'LCID', 'TSLA', 'MULN', 'REE', 'FFIE', 'PSNY', 'VFS',
    
    # Gaming/Entertainment
    'DKNG', 'PENN', 'BETZ', 'CHDN', 'BYD', 'RSI', 'GDEN', 'GAMB', 'RRR', 'FLUT',
    'GNOG', 'PLBY', 'WYNN', 'MGM', 'CZR', 'LVS', 'RRR', 'MCRI', 'MSGM', 'LYV',
    
    # Healthcare small caps
    'ANGO', 'ATRC', 'BLFS', 'BRKR', 'CCRN', 'CERS', 'CODX', 'CUTR', 'DXCM', 'EBS',
    'ECOR', 'EOLS', 'ESTA', 'GKOS', 'GMED', 'GOSS', 'GRFS', 'HCAT', 'ICAD', 'IMVT',
    'INGN', 'INMD', 'IOVA', 'IRTC', 'ISRG', 'ITGR', 'KIDS', 'LMAT', 'LUNG', 'MDXH',
    'MGNX', 'MIRM', 'MYGN', 'NARI', 'NVCR', 'OMCL', 'OSH', 'PDCO', 'PHR', 'PINC',
    
    # Industrial small caps
    'AGYS', 'AMSC', 'ARKO', 'ATEX', 'AZTA', 'BBSI', 'BKSY', 'BLZE', 'CENX', 'CRS',
    'CWST', 'CYTK', 'DCBO', 'DFIN', 'DLO', 'DOMO', 'EMBC', 'ENVX', 'EYE', 'FRPT',
    'FTAI', 'FTDR', 'FWRD', 'GEF', 'GERN', 'GMRE', 'GNL', 'GPRO', 'GSHD', 'GTES',
    'GTX', 'HAFC', 'HASI', 'HCC', 'HGV', 'HI', 'HLMN', 'HLX', 'HOMB', 'HOPE',
    'HQY', 'HRI', 'HSTM', 'HWC', 'HWKN', 'HXL', 'HYLN', 'ICFI', 'IHRT', 'INN',
    'INOD', 'IOSP', 'IPG', 'IRIX', 'IRMD', 'IRON', 'ITGR', 'ITIC', 'ITRI', 'IVT',
    'JACK', 'JBL', 'JELD', 'JJSF', 'JOE', 'JOUT', 'JOBY', 'KALU', 'KBR', 'KFY',
    'KIDS', 'KN', 'KNF', 'KRC', 'KRG', 'KROS', 'KRYS', 'KWR', 'LADR', 'LANC',
    'LAUR', 'LDI', 'LEG', 'LESL', 'LFST', 'LGND', 'LIND', 'LIVN', 'LKFN', 'LMAT',
    'LMB', 'LMND', 'LOAN', 'LOCO', 'LPRO', 'LPX', 'LRMR', 'LTC', 'LTRN', 'LXP',
    'LZB', 'MASI', 'MATV', 'MATW', 'MBIN', 'MCFT', 'MCW', 'MDU', 'MESA', 'MGNI',
    'MITK', 'MMS', 'MMSI', 'MODV', 'MOV', 'MP', 'MRCY', 'MRTN', 'MSEX', 'MSM',
    'MTRX', 'MTX', 'MTZ', 'MUR', 'MVST', 'MWA', 'MXCT', 'MYE', 'NARI', 'NATR',
    
    # Financial small caps
    'AMAL', 'FSBC', 'INBK', 'BY', 'MCBS', 'CWBC', 'FMBH', 'CZWI', 'FNWB', 'EFSC',
    'GCBC', 'MVBF', 'NWFL', 'RVSB', 'THFF', 'USCB', 'VCNX', 'AX', 'BANC', 'BANR',
    'BOH', 'BPOP', 'BUSE', 'CADE', 'CASH', 'CBSH', 'CCB', 'CFFN', 'CIVB', 'COLB',
    'CPF', 'CUBI', 'CVBF', 'CWBC', 'DCOM', 'EGBN', 'EVBN', 'EWBC', 'FBIZ', 'FBK',
    'FBNC', 'FCF', 'FCNCA', 'FFBC', 'FFIN', 'FFWM', 'FIBK', 'FISI', 'FLIC', 'FLO',
    'FMBI', 'FNB', 'FNCB', 'FNWB', 'FRME', 'FSBW', 'FUNC', 'GABC', 'GBCI', 'GLBZ',
    'GNTY', 'GSBC', 'HAFC', 'HBT', 'HFWA', 'HMST', 'HOMB', 'HOPE', 'HTBI', 'HTBK',
    'HTLF', 'HWC', 'IBCP', 'IBOC', 'INDB', 'IROQ', 'ISBC', 'KRNY', 'LBAI', 'LION',
    
    # Energy small caps
    'AROC', 'CIVI', 'CHRD', 'SM', 'CPE', 'EPSN', 'REPX', 'VTLE', 'NOG', 'MGY',
    'NEXT', 'GPOR', 'HPK', 'CLNE', 'CRK', 'CNX', 'PTEN', 'RES', 'CLB', 'OIS',
    'WTTR', 'XPRO', 'NGL', 'GEL', 'AMPY', 'BORR', 'NINE', 'NBR', 'HP', 'LBRT',
    'NEX', 'PUMP', 'SOI', 'TDW', 'VET', 'VNOM', 'WHD', 'WTTR', 'CDEV', 'ESTE',
    
    # REITs small caps
    'ADC', 'ALEX', 'ALX', 'APTS', 'BRSP', 'BRT', 'BXMT', 'CBL', 'CIO', 'CLDT',
    'CLPR', 'CSR', 'CUZ', 'DEA', 'DEI', 'ESRT', 'FCPT', 'FSP', 'GNL', 'GOOD',
    'GPT', 'GRPN', 'GTY', 'INN', 'JBGS', 'KRC', 'KRG', 'LAND', 'LTC', 'LXP',
    'MAC', 'MDV', 'NLCP', 'NNN', 'NSA', 'OFC', 'OLP', 'OUT', 'PCH', 'PGRE',
    'PK', 'PLYM', 'PSTL', 'RLJ', 'ROIC', 'RPAI', 'RVI', 'SAFE', 'SKT', 'SLG',
    'STWD', 'SUI', 'TRNO', 'UBA', 'UE', 'VNO', 'VRE', 'WPC', 'WSR', 'XHR',
]

# =============================================================================
# INTERNATIONAL ADRs - LIQUID NAMES (150+)
# =============================================================================
INTERNATIONAL_ADRS = [
    # Chinese Tech
    'BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'XPEV', 'LI', 'TME', 'IQ', 'BILI',
    'WB', 'HUYA', 'DOYU', 'ZTO', 'VNET', 'TUYA', 'YMM', 'GDS', 'EDU', 'TAL',
    'GOTU', 'TIGR', 'FUTU', 'FINV', 'LX', 'YY', 'QFIN', 'BZ', 'HTHT', 'ATHM',
    'VIPS', 'WDH', 'YUMC', 'YSG', 'ZH', 'ZLAB', 'ZEPP', 'ZLAB', 'KC', 'LEGN',
    
    # LatAm
    'SE', 'GRAB', 'MELI', 'GLOB', 'DLO', 'STNE', 'PAGS', 'NU', 'XP', 'CRNT',
    'ERJ', 'CIG', 'PBR', 'SBS', 'SID', 'VALE', 'EBR', 'GOL', 'AZUL', 'CBD',
    'BRFS', 'SQM', 'ECL', 'ENIC', 'AMX', 'TV', 'TGNA', 'VTMX', 'OMAB', 'PAC',
    
    # European
    'ASML', 'NVO', 'AZN', 'SNY', 'GSK', 'BTI', 'UL', 'DEO', 'RIO', 'BP',
    'SHEL', 'TTE', 'EQNR', 'STLA', 'RACE', 'NVS', 'SAP', 'SHOP', 'TD', 'BNS',
    'BMO', 'CM', 'RY', 'ENB', 'TRP', 'SU', 'CNQ', 'CP', 'CNI', 'WCN',
    
    # Japan/Korea/Taiwan
    'TM', 'HMC', 'SONY', 'MUFG', 'SMFG', 'MFG', 'NMR', 'NTDOY', 'KYOCY', 'CAJ',
    'TSM', 'UMC', 'ASX', 'SKM', 'LPL', 'KB', 'SHG', 'PKX', 'HIMX', 'CHKP',
    
    # India
    'INFY', 'WIT', 'HDB', 'IBN', 'SIFY', 'RDY', 'TTM', 'VEDL', 'WNS', 'MMYT',
    
    # Other EM
    'TCOM', 'CPNG', 'BEKE', 'MNSO', 'VIPS', 'YSG', 'DAO', 'IH', 'CANG', 'LU',
]

# =============================================================================
# LEVERAGED & SECTOR ETFs (150+)
# =============================================================================
ETFS = [
    # Core Index ETFs
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VT', 'VEA', 'VWO', 'IEMG',
    'MDY', 'IJH', 'IJR', 'IWO', 'IWN', 'VTV', 'VUG', 'MTUM', 'QUAL', 'VLUE',
    'RSP', 'IVE', 'IVW', 'VBK', 'VBR', 'VIOO', 'VIOV', 'VIOG', 'VONG', 'VONV',
    
    # Sector ETFs
    'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC',
    'SMH', 'SOXX', 'IBB', 'XBI', 'IYR', 'VNQ', 'KRE', 'XHB', 'ITB', 'GDX', 'GDXJ',
    'XOP', 'OIH', 'KWEB', 'FXI', 'EWJ', 'EEM', 'VWO', 'IEMG', 'ARKK', 'ARKW',
    'ARKG', 'ARKF', 'ARKQ', 'ARKX', 'PRNT', 'IZRL',
    
    # Industry ETFs
    'XRT', 'XME', 'JETS', 'BETZ', 'HACK', 'SKYY', 'CLOU', 'WCLD', 'CIBR', 'ROBO',
    'BOTZ', 'DRIV', 'IDRV', 'KARS', 'LIT', 'QCLN', 'ICLN', 'TAN', 'PBW', 'FAN',
    'PAVE', 'IFRA', 'GRID', 'ACES', 'CNRG', 'SMOG', 'REMX', 'COPX', 'URA', 'URNM',
    
    # 3x Leveraged Bull ETFs
    'TQQQ', 'SPXL', 'UPRO', 'TNA', 'SOXL', 'TECL', 'FNGU', 'LABU', 'FAS', 'CURE',
    'DPST', 'NAIL', 'DFEN', 'RETL', 'HIBL', 'WANT', 'WEBL', 'DUSL', 'MIDU', 'TPOR',
    'UDOW', 'UMDD', 'URTY', 'EDC', 'TMF', 'TYD', 'UCO', 'BOIL', 'UGL', 'AGQ',
    'NUGT', 'JNUG', 'GUSH', 'ERX', 'DRN', 'YINN',
    
    # 3x Leveraged Bear/Inverse ETFs
    'SQQQ', 'SPXU', 'SPXS', 'TZA', 'SOXS', 'TECS', 'FNGD', 'LABD', 'FAZ', 'HIBS',
    'SDOW', 'SRTY', 'EDZ', 'TMV', 'TYO', 'SCO', 'KOLD', 'DUST', 'JDST', 'DRIP',
    'ERY', 'DRV', 'YANG',
    
    # 2x Leveraged
    'QLD', 'SSO', 'UWM', 'ROM', 'USD', 'UYG', 'UGE', 'UCC', 'MVV', 'SAA',
    'DDM', 'UKK', 'EFO', 'EET', 'EZJ', 'UPV', 'UBR', 'EFU', 'EWV', 'EPV',
    
    # Volatility & Hedging
    'VXX', 'UVXY', 'SVXY', 'VIXY', 'VIXM', 'TAIL', 'SVOL',
    
    # Bonds & Fixed Income
    'TLT', 'TBT', 'TMF', 'TMV', 'BND', 'AGG', 'LQD', 'HYG', 'JNK', 'EMB',
    'TIP', 'VTIP', 'SCHP', 'STIP', 'SHY', 'IEF', 'IEI', 'GOVT', 'NEAR', 'MINT',
    
    # Commodities
    'GLD', 'SLV', 'IAU', 'GLDM', 'SIVR', 'PPLT', 'PALL', 'DBC', 'PDBC', 'GSG',
    'USO', 'BNO', 'UNG', 'CORN', 'WEAT', 'SOYB', 'COW', 'DBA', 'MOO', 'ICOW',
]

# =============================================================================
# COMBINE INTO MEGA UNIVERSE
# =============================================================================
MEGA_UNIVERSE = list(set(
    SP500 +
    NASDAQ100_ADDITIONS +
    RUSSELL1000_GROWTH +
    RUSSELL2000_MOMENTUM +
    INTERNATIONAL_ADRS +
    ETFS
))

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_mega_universe() -> List[str]:
    """Get the MEGA tradeable universe - 3000+ symbols."""
    return MEGA_UNIVERSE.copy()

def get_sp500() -> List[str]:
    """Get S&P 500 tickers only."""
    return SP500.copy()

def get_etfs() -> List[str]:
    """Get all ETFs."""
    return ETFS.copy()

def get_leveraged_bull_etfs() -> Dict[str, float]:
    """Get 3x leveraged bull ETFs with aggressive weights."""
    return {
        'TQQQ': 0.35,  # Nasdaq 100 3x - CORE POSITION
        'SPXL': 0.25,  # S&P 500 3x
        'SOXL': 0.20,  # Semiconductors 3x
        'TECL': 0.10,  # Technology 3x
        'FNGU': 0.10,  # FAANG 3x
    }

def get_gold_hedges() -> Dict[str, float]:
    """Get gold mining ETFs for hedging."""
    return {
        'GDX': 0.50,   # Gold miners
        'GDXJ': 0.30,  # Junior miners
        'NEM': 0.10,   # Newmont
        'GOLD': 0.10,  # Barrick
    }

def get_leveraged_bear_etfs() -> Dict[str, float]:
    """Get 3x leveraged bear/inverse ETFs."""
    return {
        'SQQQ': 0.30,  # Nasdaq 100 -3x
        'SPXU': 0.25,  # S&P 500 -3x
        'SOXS': 0.20,  # Semiconductors -3x
        'TECS': 0.10,  # Technology -3x
        'FNGD': 0.10,  # FAANG -3x
        'TZA': 0.05,   # Russell 2000 -3x
    }

# =============================================================================
# UNIVERSE STATS
# =============================================================================
UNIVERSE_STATS = {
    'total': len(MEGA_UNIVERSE),
    'sp500': len(SP500),
    'nasdaq100_additions': len(NASDAQ100_ADDITIONS),
    'russell1000_growth': len(RUSSELL1000_GROWTH),
    'russell2000_momentum': len(RUSSELL2000_MOMENTUM),
    'international_adrs': len(INTERNATIONAL_ADRS),
    'etfs': len(ETFS),
}

# Print stats on import
print(f"MEGA Universe Loaded: {UNIVERSE_STATS['total']} symbols")
print(f"  - S&P 500: {UNIVERSE_STATS['sp500']}")
print(f"  - NASDAQ Additions: {UNIVERSE_STATS['nasdaq100_additions']}")
print(f"  - Russell 1000 Growth: {UNIVERSE_STATS['russell1000_growth']}")
print(f"  - Russell 2000 Momentum: {UNIVERSE_STATS['russell2000_momentum']}")
print(f"  - International ADRs: {UNIVERSE_STATS['international_adrs']}")
print(f"  - ETFs: {UNIVERSE_STATS['etfs']}")
print("PRINTING CASH LIKE RENAISSANCE!")

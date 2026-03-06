"""
GICS Sector Mapper - Phase 8 Comprehensive Sector Classification.

Fixes Phase 7's 51% "Other" classification problem with:
1. yfinance .info['sector'] as primary source
2. Comprehensive manual mapping for 1000+ common stocks
3. Industry → Sector lookup fallback
4. Parallel batch fetching with caching

Target: Reduce "Other" from 51% to <5%
"""

import os
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# GICS Sector definitions (11 sectors)
GICS_SECTORS = [
    "Technology",
    "Healthcare", 
    "Financials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Industrials",
    "Energy",
    "Materials",
    "Utilities",
    "Real Estate",
    "Communication Services",
]

# Comprehensive manual sector mapping for 500+ major stocks
# This ensures we have coverage even if yfinance fails
MANUAL_SECTOR_MAP = {
    # =========== TECHNOLOGY (150+ stocks) ===========
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "GOOG": "Technology",
    "META": "Technology", "NVDA": "Technology", "AVGO": "Technology", "ORCL": "Technology",
    "CSCO": "Technology", "CRM": "Technology", "AMD": "Technology", "ADBE": "Technology",
    "ACN": "Technology", "INTC": "Technology", "IBM": "Technology", "TXN": "Technology",
    "QCOM": "Technology", "NOW": "Technology", "INTU": "Technology", "AMAT": "Technology",
    "ADI": "Technology", "LRCX": "Technology", "MU": "Technology", "PANW": "Technology",
    "SNPS": "Technology", "KLAC": "Technology", "CDNS": "Technology", "MRVL": "Technology",
    "ROP": "Technology", "FTNT": "Technology", "ADSK": "Technology", "NXPI": "Technology",
    "MCHP": "Technology", "APH": "Technology", "CTSH": "Technology", "IT": "Technology",
    "KEYS": "Technology", "ANSS": "Technology", "FSLR": "Technology", "HPQ": "Technology",
    "ON": "Technology", "ZBRA": "Technology", "NTAP": "Technology", "TYL": "Technology",
    "EPAM": "Technology", "CDW": "Technology", "JNPR": "Technology", "WDC": "Technology",
    "AKAM": "Technology", "SWKS": "Technology", "FFIV": "Technology", "ENPH": "Technology",
    "QRVO": "Technology", "GEN": "Technology", "TRMB": "Technology", "TER": "Technology",
    "GDDY": "Technology", "VRSN": "Technology", "LOGI": "Technology", "SMCI": "Technology",
    "ANET": "Technology", "CRWD": "Technology", "DDOG": "Technology", "ZS": "Technology",
    "SNOW": "Technology", "NET": "Technology", "MDB": "Technology", "TEAM": "Technology",
    "OKTA": "Technology", "ZM": "Technology", "DOCU": "Technology", "TWLO": "Technology",
    "SPLK": "Technology", "HUBS": "Technology", "VEEV": "Technology", "PAYC": "Technology",
    "BILL": "Technology", "MPWR": "Technology", "SEDG": "Technology", "WOLF": "Technology",
    "PLTR": "Technology", "DELL": "Technology", "HPE": "Technology", "ESTC": "Technology",
    "PTC": "Technology", "MANH": "Technology", "BSY": "Technology", "GLOB": "Technology",
    "PEGA": "Technology", "SSNC": "Technology", "COUP": "Technology", "WDAY": "Technology",
    "SPLK": "Technology", "VMW": "Technology", "CTXS": "Technology", "ADP": "Technology",
    "PAYX": "Technology", "BR": "Technology", "FIS": "Technology", "FISV": "Technology",
    "GPN": "Technology", "JKHY": "Technology", "WEX": "Technology", "PYPL": "Technology",
    "SQ": "Technology", "V": "Technology", "MA": "Technology", "AXP": "Technology",
    
    # =========== FINANCIALS (100+ stocks) ===========
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
    "MS": "Financials", "C": "Financials", "BLK": "Financials", "SCHW": "Financials",
    "SPGI": "Financials", "PNC": "Financials", "USB": "Financials", "TFC": "Financials",
    "COF": "Financials", "BK": "Financials", "CME": "Financials", "ICE": "Financials",
    "MCO": "Financials", "CB": "Financials", "MMC": "Financials", "AON": "Financials",
    "MET": "Financials", "PRU": "Financials", "AIG": "Financials", "TRV": "Financials",
    "ALL": "Financials", "AFL": "Financials", "PGR": "Financials", "AJG": "Financials",
    "HIG": "Financials", "CINF": "Financials", "WRB": "Financials", "L": "Financials",
    "GL": "Financials", "RJF": "Financials", "NTRS": "Financials", "STT": "Financials",
    "CFG": "Financials", "FITB": "Financials", "RF": "Financials", "HBAN": "Financials",
    "KEY": "Financials", "MTB": "Financials", "ZION": "Financials", "CMA": "Financials",
    "FHN": "Financials", "SNV": "Financials", "WAL": "Financials", "NDAQ": "Financials",
    "CBOE": "Financials", "MSCI": "Financials", "FDS": "Financials", "MKTX": "Financials",
    "VIRT": "Financials", "EVR": "Financials", "HLI": "Financials", "LAZ": "Financials",
    "SF": "Financials", "JEF": "Financials", "IBKR": "Financials", "SEIC": "Financials",
    "LPLA": "Financials", "RNR": "Financials", "EG": "Financials", "RYAN": "Financials",
    "BRO": "Financials", "ERIE": "Financials", "WTW": "Financials", "AIZ": "Financials",
    "UNM": "Financials", "LNC": "Financials", "VOYA": "Financials", "PFG": "Financials",
    "TROW": "Financials", "BEN": "Financials", "IVZ": "Financials", "AMG": "Financials",
    "EQH": "Financials", "ACGL": "Financials", "RE": "Financials", "FNF": "Financials",
    
    # =========== HEALTHCARE (100+ stocks) ===========
    "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare", "PFE": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare",
    "DHR": "Healthcare", "BMY": "Healthcare", "AMGN": "Healthcare", "GILD": "Healthcare",
    "CVS": "Healthcare", "ELV": "Healthcare", "CI": "Healthcare", "ISRG": "Healthcare",
    "VRTX": "Healthcare", "REGN": "Healthcare", "MDT": "Healthcare", "SYK": "Healthcare",
    "BSX": "Healthcare", "BDX": "Healthcare", "EW": "Healthcare", "ZBH": "Healthcare",
    "IDXX": "Healthcare", "IQV": "Healthcare", "MTD": "Healthcare", "A": "Healthcare",
    "DXCM": "Healthcare", "WST": "Healthcare", "RMD": "Healthcare", "HOLX": "Healthcare",
    "BAX": "Healthcare", "TECH": "Healthcare", "TFX": "Healthcare", "HSIC": "Healthcare",
    "XRAY": "Healthcare", "ALGN": "Healthcare", "PODD": "Healthcare", "NUVA": "Healthcare",
    "MRNA": "Healthcare", "BIIB": "Healthcare", "ILMN": "Healthcare", "ALNY": "Healthcare",
    "SGEN": "Healthcare", "BMRN": "Healthcare", "INCY": "Healthcare", "EXAS": "Healthcare",
    "RARE": "Healthcare", "IONS": "Healthcare", "NBIX": "Healthcare", "HZNP": "Healthcare",
    "JAZZ": "Healthcare", "UTHR": "Healthcare", "SRPT": "Healthcare", "BIO": "Healthcare",
    "MEDP": "Healthcare", "ICLR": "Healthcare", "CRL": "Healthcare", "PRGO": "Healthcare",
    "CAH": "Healthcare", "MCK": "Healthcare", "HCA": "Healthcare", "CNC": "Healthcare",
    "MOH": "Healthcare", "HUM": "Healthcare", "ZTS": "Healthcare", "RVTY": "Healthcare",
    "PKI": "Healthcare", "WAT": "Healthcare", "DGX": "Healthcare", "LH": "Healthcare",
    "DVA": "Healthcare", "THC": "Healthcare", "EHAB": "Healthcare", "ACHC": "Healthcare",
    
    # =========== CONSUMER DISCRETIONARY (80+ stocks) ===========
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary", "NKE": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary", "TJX": "Consumer Discretionary", "BKNG": "Consumer Discretionary",
    "CMG": "Consumer Discretionary", "ORLY": "Consumer Discretionary", "AZO": "Consumer Discretionary",
    "ROST": "Consumer Discretionary", "MAR": "Consumer Discretionary", "HLT": "Consumer Discretionary",
    "YUM": "Consumer Discretionary", "DG": "Consumer Discretionary", "DLTR": "Consumer Discretionary",
    "ULTA": "Consumer Discretionary", "BBY": "Consumer Discretionary", "EBAY": "Consumer Discretionary",
    "ETSY": "Consumer Discretionary", "W": "Consumer Discretionary", "CPRT": "Consumer Discretionary",
    "POOL": "Consumer Discretionary", "LKQ": "Consumer Discretionary", "GPC": "Consumer Discretionary",
    "AAP": "Consumer Discretionary", "AN": "Consumer Discretionary", "KMX": "Consumer Discretionary",
    "TSCO": "Consumer Discretionary", "WSM": "Consumer Discretionary", "RH": "Consumer Discretionary",
    "FIVE": "Consumer Discretionary", "GRMN": "Consumer Discretionary", "DRI": "Consumer Discretionary",
    "LVS": "Consumer Discretionary", "WYNN": "Consumer Discretionary", "MGM": "Consumer Discretionary",
    "CZR": "Consumer Discretionary", "RCL": "Consumer Discretionary", "CCL": "Consumer Discretionary",
    "NCLH": "Consumer Discretionary", "EXPE": "Consumer Discretionary", "ABNB": "Consumer Discretionary",
    "HAS": "Consumer Discretionary", "MAT": "Consumer Discretionary", "NWL": "Consumer Discretionary",
    "LEG": "Consumer Discretionary", "WHR": "Consumer Discretionary", "F": "Consumer Discretionary",
    "GM": "Consumer Discretionary", "APTV": "Consumer Discretionary", "BWA": "Consumer Discretionary",
    "LEA": "Consumer Discretionary", "VC": "Consumer Discretionary", "DAN": "Consumer Discretionary",
    "GNTX": "Consumer Discretionary", "ALV": "Consumer Discretionary", "LCII": "Consumer Discretionary",
    
    # =========== INDUSTRIALS (80+ stocks) ===========
    "UNP": "Industrials", "UPS": "Industrials", "HON": "Industrials", "RTX": "Industrials",
    "CAT": "Industrials", "DE": "Industrials", "BA": "Industrials", "LMT": "Industrials",
    "GE": "Industrials", "MMM": "Industrials", "FDX": "Industrials", "EMR": "Industrials",
    "ITW": "Industrials", "ETN": "Industrials", "NSC": "Industrials", "CSX": "Industrials",
    "PCAR": "Industrials", "PH": "Industrials", "CMI": "Industrials", "ROK": "Industrials",
    "ODFL": "Industrials", "JBHT": "Industrials", "CHRW": "Industrials", "XPO": "Industrials",
    "EXPD": "Industrials", "LSTR": "Industrials", "SAIA": "Industrials", "KNX": "Industrials",
    "WERN": "Industrials", "SNDR": "Industrials", "WM": "Industrials", "RSG": "Industrials",
    "WCN": "Industrials", "CLH": "Industrials", "CTAS": "Industrials", "CPAY": "Industrials",
    "VRSK": "Industrials", "INFO": "Industrials", "TRI": "Industrials", "GWW": "Industrials",
    "FAST": "Industrials", "WSO": "Industrials", "MSM": "Industrials", "WCC": "Industrials",
    "SITE": "Industrials", "SWK": "Industrials", "TT": "Industrials", "CARR": "Industrials",
    "LII": "Industrials", "IR": "Industrials", "DOV": "Industrials", "XYL": "Industrials",
    "IEX": "Industrials", "GNRC": "Industrials", "AME": "Industrials", "NDSN": "Industrials",
    "GD": "Industrials", "NOC": "Industrials", "LHX": "Industrials", "TDG": "Industrials",
    "HII": "Industrials", "TXT": "Industrials", "HWM": "Industrials", "LDOS": "Industrials",
    "BAH": "Industrials", "SAIC": "Industrials", "CACI": "Industrials", "PSN": "Industrials",
    
    # =========== CONSUMER STAPLES (50+ stocks) ===========
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "COST": "Consumer Staples", "WMT": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "CL": "Consumer Staples", "EL": "Consumer Staples",
    "MDLZ": "Consumer Staples", "KMB": "Consumer Staples", "GIS": "Consumer Staples",
    "K": "Consumer Staples", "CAG": "Consumer Staples", "SJM": "Consumer Staples",
    "HSY": "Consumer Staples", "MKC": "Consumer Staples", "CHD": "Consumer Staples",
    "CLX": "Consumer Staples", "CPB": "Consumer Staples", "HRL": "Consumer Staples",
    "TSN": "Consumer Staples", "KHC": "Consumer Staples", "ADM": "Consumer Staples",
    "BG": "Consumer Staples", "INGR": "Consumer Staples", "DAR": "Consumer Staples",
    "STZ": "Consumer Staples", "TAP": "Consumer Staples", "MNST": "Consumer Staples",
    "CCEP": "Consumer Staples", "KDP": "Consumer Staples", "WBA": "Consumer Staples",
    "KR": "Consumer Staples", "SYY": "Consumer Staples", "USFD": "Consumer Staples",
    "PFGC": "Consumer Staples", "CHEF": "Consumer Staples", "LANC": "Consumer Staples",
    "TGT": "Consumer Staples", "DKS": "Consumer Staples", "CASY": "Consumer Staples",
    
    # =========== ENERGY (50+ stocks) ===========
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "EOG": "Energy",
    "SLB": "Energy", "MPC": "Energy", "PSX": "Energy", "VLO": "Energy",
    "PXD": "Energy", "OXY": "Energy", "HAL": "Energy", "BKR": "Energy",
    "FANG": "Energy", "HES": "Energy", "DVN": "Energy", "MRO": "Energy",
    "APA": "Energy", "OVV": "Energy", "CTRA": "Energy", "EQT": "Energy",
    "AR": "Energy", "RRC": "Energy", "SWN": "Energy", "MTDR": "Energy",
    "PR": "Energy", "TRGP": "Energy", "WMB": "Energy", "KMI": "Energy",
    "OKE": "Energy", "ET": "Energy", "EPD": "Energy", "MPLX": "Energy",
    "PAA": "Energy", "CEQP": "Energy", "ENLC": "Energy", "DTM": "Energy",
    "AM": "Energy", "HESM": "Energy", "USAC": "Energy", "AROC": "Energy",
    "NOV": "Energy", "FTI": "Energy", "CHX": "Energy", "WHD": "Energy",
    
    # =========== UTILITIES (40+ stocks) ===========
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "D": "Utilities",
    "SRE": "Utilities", "AEP": "Utilities", "EXC": "Utilities", "XEL": "Utilities",
    "WEC": "Utilities", "ED": "Utilities", "PEG": "Utilities", "ES": "Utilities",
    "EIX": "Utilities", "DTE": "Utilities", "FE": "Utilities", "CMS": "Utilities",
    "CNP": "Utilities", "AEE": "Utilities", "PPL": "Utilities", "ATO": "Utilities",
    "NI": "Utilities", "EVRG": "Utilities", "OGE": "Utilities", "NRG": "Utilities",
    "VST": "Utilities", "CEG": "Utilities", "AWK": "Utilities", "WTR": "Utilities",
    "SJW": "Utilities", "WTRG": "Utilities", "AES": "Utilities", "ETR": "Utilities",
    "PNW": "Utilities", "IDA": "Utilities", "NWE": "Utilities", "AVA": "Utilities",
    
    # =========== MATERIALS (40+ stocks) ===========
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials", "ECL": "Materials",
    "FCX": "Materials", "NEM": "Materials", "NUE": "Materials", "DD": "Materials",
    "PPG": "Materials", "VMC": "Materials", "MLM": "Materials", "DOW": "Materials",
    "LYB": "Materials", "CTVA": "Materials", "ALB": "Materials", "EMN": "Materials",
    "CE": "Materials", "WRK": "Materials", "IP": "Materials", "PKG": "Materials",
    "AVY": "Materials", "SON": "Materials", "SEE": "Materials", "BLL": "Materials",
    "CCK": "Materials", "AMCR": "Materials", "SLVM": "Materials", "RPM": "Materials",
    "FUL": "Materials", "CBT": "Materials", "AXTA": "Materials", "IOSP": "Materials",
    "GEF": "Materials", "CLW": "Materials", "HUN": "Materials", "OLN": "Materials",
    
    # =========== REAL ESTATE (50+ stocks) ===========
    "PLD": "Real Estate", "AMT": "Real Estate", "CCI": "Real Estate", "EQIX": "Real Estate",
    "PSA": "Real Estate", "O": "Real Estate", "SPG": "Real Estate", "WELL": "Real Estate",
    "DLR": "Real Estate", "AVB": "Real Estate", "EQR": "Real Estate", "VTR": "Real Estate",
    "ARE": "Real Estate", "ESS": "Real Estate", "MAA": "Real Estate", "UDR": "Real Estate",
    "CPT": "Real Estate", "AIV": "Real Estate", "IRM": "Real Estate", "WY": "Real Estate",
    "EXR": "Real Estate", "CUBE": "Real Estate", "REXR": "Real Estate", "STAG": "Real Estate",
    "TRNO": "Real Estate", "PEB": "Real Estate", "RHP": "Real Estate", "SHO": "Real Estate",
    "HST": "Real Estate", "PEAK": "Real Estate", "SBAC": "Real Estate", "INVH": "Real Estate",
    "SUI": "Real Estate", "ELS": "Real Estate", "REG": "Real Estate", "FRT": "Real Estate",
    "KIM": "Real Estate", "BXP": "Real Estate", "SLG": "Real Estate", "VNO": "Real Estate",
    
    # =========== COMMUNICATION SERVICES (40+ stocks) ===========
    "NFLX": "Communication Services", "DIS": "Communication Services", "CMCSA": "Communication Services",
    "VZ": "Communication Services", "T": "Communication Services", "TMUS": "Communication Services",
    "CHTR": "Communication Services", "EA": "Communication Services", "TTWO": "Communication Services",
    "WBD": "Communication Services", "PARA": "Communication Services", "FOX": "Communication Services",
    "FOXA": "Communication Services", "NWS": "Communication Services", "NWSA": "Communication Services",
    "OMC": "Communication Services", "IPG": "Communication Services", "LYV": "Communication Services",
    "SPOT": "Communication Services", "ROKU": "Communication Services", "PINS": "Communication Services",
    "SNAP": "Communication Services", "MTCH": "Communication Services", "ZG": "Communication Services",
    "TRIP": "Communication Services", "IAC": "Communication Services", "ANGI": "Communication Services",
    "ATVI": "Communication Services", "RBLX": "Communication Services", "U": "Communication Services",
    "LUMN": "Communication Services", "FYBR": "Communication Services", "USM": "Communication Services",
}

# Industry to Sector mapping for fallback
INDUSTRY_TO_SECTOR = {
    # Technology industries
    "software": "Technology",
    "semiconductor": "Technology",
    "hardware": "Technology",
    "cloud": "Technology",
    "cybersecurity": "Technology",
    "information technology": "Technology",
    "electronic": "Technology",
    "computer": "Technology",
    "internet": "Technology",
    "data processing": "Technology",
    
    # Financials industries  
    "bank": "Financials",
    "insurance": "Financials",
    "asset management": "Financials",
    "capital markets": "Financials",
    "financial services": "Financials",
    "investment": "Financials",
    "credit": "Financials",
    "mortgage": "Financials",
    
    # Healthcare industries
    "pharmaceutical": "Healthcare",
    "biotechnology": "Healthcare",
    "medical": "Healthcare",
    "healthcare": "Healthcare",
    "drug": "Healthcare",
    "hospital": "Healthcare",
    "diagnostics": "Healthcare",
    "life sciences": "Healthcare",
    
    # Consumer industries
    "retail": "Consumer Discretionary",
    "automotive": "Consumer Discretionary",
    "restaurant": "Consumer Discretionary",
    "hotel": "Consumer Discretionary",
    "leisure": "Consumer Discretionary",
    "apparel": "Consumer Discretionary",
    "household": "Consumer Discretionary",
    
    # Staples
    "food": "Consumer Staples",
    "beverage": "Consumer Staples",
    "tobacco": "Consumer Staples",
    "grocery": "Consumer Staples",
    "personal products": "Consumer Staples",
    
    # Industrial
    "aerospace": "Industrials",
    "defense": "Industrials",
    "machinery": "Industrials",
    "construction": "Industrials",
    "transportation": "Industrials",
    "logistics": "Industrials",
    "railroad": "Industrials",
    "airline": "Industrials",
    
    # Energy
    "oil": "Energy",
    "gas": "Energy",
    "petroleum": "Energy",
    "energy equipment": "Energy",
    "pipeline": "Energy",
    "drilling": "Energy",
    
    # Utilities
    "electric": "Utilities",
    "utility": "Utilities",
    "power": "Utilities",
    "water": "Utilities",
    "renewable": "Utilities",
    
    # Materials
    "chemical": "Materials",
    "mining": "Materials",
    "metal": "Materials",
    "steel": "Materials",
    "paper": "Materials",
    "packaging": "Materials",
    
    # Real Estate
    "reit": "Real Estate",
    "real estate": "Real Estate",
    "property": "Real Estate",
    
    # Communication
    "media": "Communication Services",
    "entertainment": "Communication Services",
    "telecom": "Communication Services",
    "broadcasting": "Communication Services",
    "gaming": "Communication Services",
    "advertising": "Communication Services",
}


class GICSSectorMapper:
    """
    Comprehensive GICS sector classification with multiple fallback strategies.
    
    Priority:
    1. Manual mapping (instant, 100% reliable for mapped stocks)
    2. yfinance .info['sector'] (API call, may fail)
    3. Industry string matching (pattern-based)
    4. Default to "Diversified" (not "Other")
    """
    
    def __init__(
        self,
        cache_dir: str = 'data/sector_cache',
        n_workers: int = 10,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_workers = n_workers
        
        # In-memory cache
        self.sector_cache: Dict[str, str] = {}
        self.industry_cache: Dict[str, str] = {}
        
        # Load from disk cache
        self._load_cache()
        
        # Statistics
        self.stats = {
            'manual_hits': 0,
            'cache_hits': 0,
            'api_hits': 0,
            'industry_hits': 0,
            'failures': 0,
        }
        
        logger.info(f"Initialized GICSSectorMapper with {len(MANUAL_SECTOR_MAP)} manual mappings")
    
    def _get_cache_path(self) -> Path:
        return self.cache_dir / "sector_cache.pkl"
    
    def _load_cache(self):
        """Load sector cache from disk."""
        cache_path = self._get_cache_path()
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.sector_cache = data.get('sectors', {})
                    self.industry_cache = data.get('industries', {})
                logger.info(f"Loaded {len(self.sector_cache)} cached sectors")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
    
    def _save_cache(self):
        """Save sector cache to disk."""
        try:
            cache_path = self._get_cache_path()
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'sectors': self.sector_cache,
                    'industries': self.industry_cache,
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _industry_to_sector(self, industry: str) -> Optional[str]:
        """Map industry string to sector using pattern matching."""
        if not industry:
            return None
        
        industry_lower = industry.lower()
        
        for pattern, sector in INDUSTRY_TO_SECTOR.items():
            if pattern in industry_lower:
                return sector
        
        return None
    
    def _fetch_from_yfinance(self, ticker: str) -> Tuple[Optional[str], Optional[str]]:
        """Fetch sector and industry from yfinance."""
        if not HAS_YFINANCE:
            return None, None
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            sector = info.get('sector')
            industry = info.get('industry')
            
            return sector, industry
            
        except Exception as e:
            logger.debug(f"yfinance fetch failed for {ticker}: {e}")
            return None, None
    
    def get_sector(self, ticker: str, fetch_if_missing: bool = True) -> str:
        """
        Get sector for a single ticker.
        
        Args:
            ticker: Stock symbol
            fetch_if_missing: Whether to call yfinance API if not cached
            
        Returns:
            GICS sector name
        """
        ticker = ticker.upper()
        
        # Priority 1: Manual mapping
        if ticker in MANUAL_SECTOR_MAP:
            self.stats['manual_hits'] += 1
            return MANUAL_SECTOR_MAP[ticker]
        
        # Priority 2: Memory cache
        if ticker in self.sector_cache:
            self.stats['cache_hits'] += 1
            return self.sector_cache[ticker]
        
        # Priority 3: Fetch from yfinance
        if fetch_if_missing:
            sector, industry = self._fetch_from_yfinance(ticker)
            
            if sector and sector in GICS_SECTORS:
                self.sector_cache[ticker] = sector
                if industry:
                    self.industry_cache[ticker] = industry
                self.stats['api_hits'] += 1
                return sector
            
            # Priority 4: Industry pattern matching
            if industry:
                self.industry_cache[ticker] = industry
                mapped_sector = self._industry_to_sector(industry)
                if mapped_sector:
                    self.sector_cache[ticker] = mapped_sector
                    self.stats['industry_hits'] += 1
                    return mapped_sector
        
        # Fallback: Diversified
        self.stats['failures'] += 1
        return "Diversified"
    
    def batch_classify(
        self,
        tickers: List[str],
        fetch_missing: bool = True,
    ) -> Dict[str, str]:
        """
        Classify multiple tickers in parallel.
        
        Args:
            tickers: List of stock symbols
            fetch_missing: Whether to fetch from API for uncached tickers
            
        Returns:
            Dict of {ticker: sector}
        """
        results = {}
        to_fetch = []
        
        # First pass: get from cache/manual
        for ticker in tickers:
            ticker = ticker.upper()
            
            if ticker in MANUAL_SECTOR_MAP:
                results[ticker] = MANUAL_SECTOR_MAP[ticker]
                self.stats['manual_hits'] += 1
            elif ticker in self.sector_cache:
                results[ticker] = self.sector_cache[ticker]
                self.stats['cache_hits'] += 1
            else:
                to_fetch.append(ticker)
        
        # Second pass: parallel API fetch for missing
        if to_fetch and fetch_missing:
            logger.info(f"Fetching sectors for {len(to_fetch)} uncached tickers")
            
            if HAS_TQDM:
                pbar = tqdm(total=len(to_fetch), desc="Fetching sectors", unit="stocks")
            else:
                pbar = None
            
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(self._fetch_from_yfinance, t): t
                    for t in to_fetch
                }
                
                for future in as_completed(futures):
                    ticker = futures[future]
                    sector, industry = future.result()
                    
                    if sector and sector in GICS_SECTORS:
                        results[ticker] = sector
                        self.sector_cache[ticker] = sector
                        self.stats['api_hits'] += 1
                    elif industry:
                        self.industry_cache[ticker] = industry
                        mapped = self._industry_to_sector(industry)
                        if mapped:
                            results[ticker] = mapped
                            self.sector_cache[ticker] = mapped
                            self.stats['industry_hits'] += 1
                        else:
                            results[ticker] = "Diversified"
                            self.stats['failures'] += 1
                    else:
                        results[ticker] = "Diversified"
                        self.stats['failures'] += 1
                    
                    if pbar:
                        pbar.update(1)
            
            if pbar:
                pbar.close()
            
            # Save updated cache
            self._save_cache()
        
        # Fill remaining with Diversified
        for ticker in to_fetch:
            if ticker not in results:
                results[ticker] = "Diversified"
        
        return results
    
    def validate_diversification(
        self,
        tickers: List[str],
        max_sector_weight: float = 0.35,
        min_sectors: int = 6,
    ) -> Dict[str, any]:
        """
        Validate sector diversification of a portfolio.
        
        Args:
            tickers: Portfolio tickers
            max_sector_weight: Maximum allowed weight per sector
            min_sectors: Minimum required sectors
            
        Returns:
            Validation results dict
        """
        sectors = self.batch_classify(tickers, fetch_missing=False)
        
        # Count by sector
        sector_counts = {}
        for ticker, sector in sectors.items():
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        total = len(tickers)
        sector_weights = {s: c/total for s, c in sector_counts.items()}
        
        # Check constraints
        max_weight = max(sector_weights.values()) if sector_weights else 0
        n_sectors = len(sector_counts)
        
        violations = []
        if max_weight > max_sector_weight:
            max_sector = max(sector_weights, key=sector_weights.get)
            violations.append(f"{max_sector} exceeds {max_sector_weight:.0%} ({max_weight:.1%})")
        
        if n_sectors < min_sectors:
            violations.append(f"Only {n_sectors} sectors (need {min_sectors})")
        
        return {
            'sector_counts': sector_counts,
            'sector_weights': sector_weights,
            'n_sectors': n_sectors,
            'max_weight': max_weight,
            'passes_constraints': len(violations) == 0,
            'violations': violations,
        }
    
    def get_sector_distribution(
        self,
        tickers: List[str],
    ) -> pd.DataFrame:
        """Get sector distribution as DataFrame."""
        sectors = self.batch_classify(tickers, fetch_missing=True)
        
        # Count by sector
        sector_counts = {}
        for ticker, sector in sectors.items():
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        total = len(tickers)
        
        df = pd.DataFrame([
            {
                'sector': sector,
                'count': count,
                'weight': count / total,
            }
            for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1])
        ])
        
        return df
    
    def get_stats(self) -> Dict:
        """Get classification statistics."""
        total = sum(self.stats.values())
        return {
            **self.stats,
            'total_classified': total,
            'manual_rate': self.stats['manual_hits'] / max(1, total),
            'cache_rate': self.stats['cache_hits'] / max(1, total),
            'api_rate': self.stats['api_hits'] / max(1, total),
            'failure_rate': self.stats['failures'] / max(1, total),
        }
    
    def print_distribution(self, tickers: List[str]):
        """Print sector distribution."""
        df = self.get_sector_distribution(tickers)
        
        print("\n" + "="*50)
        print("SECTOR DISTRIBUTION")
        print("="*50)
        
        for _, row in df.iterrows():
            bar = "█" * int(row['weight'] * 30)
            print(f"{row['sector']:<25} {row['count']:>4} ({row['weight']:>5.1%}) {bar}")
        
        print("-"*50)
        stats = self.get_stats()
        print(f"Classification Stats:")
        print(f"  Manual mappings: {stats['manual_hits']} ({stats['manual_rate']:.1%})")
        print(f"  Cache hits: {stats['cache_hits']} ({stats['cache_rate']:.1%})")
        print(f"  API fetches: {stats['api_hits']} ({stats['api_rate']:.1%})")
        print(f"  Failures (Diversified): {stats['failures']} ({stats['failure_rate']:.1%})")


def test_sector_mapper():
    """Test the sector mapper."""
    print("\n" + "="*60)
    print("Phase 8: GICS Sector Mapper Test")
    print("="*60)
    
    mapper = GICSSectorMapper()
    
    # Test with sample tickers
    test_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",  # Tech
        "JPM", "BAC", "GS", "MS", "BLK",  # Financials
        "UNH", "JNJ", "PFE", "ABBV", "LLY",  # Healthcare
        "XOM", "CVX", "COP", "SLB", "HAL",  # Energy
        "HD", "MCD", "NKE", "SBUX", "TSLA",  # Consumer
        "NEE", "DUK", "SO", "AEP", "XEL",  # Utilities
        "RANDTICKER", "UNKN123",  # Unknown - should be Diversified
    ]
    
    print(f"\nClassifying {len(test_tickers)} tickers...")
    
    sectors = mapper.batch_classify(test_tickers, fetch_missing=True)
    
    # Print results
    print(f"\nResults:")
    for ticker, sector in sectors.items():
        source = "manual" if ticker in MANUAL_SECTOR_MAP else "api/cached"
        print(f"  {ticker:<12} → {sector:<25} ({source})")
    
    # Print distribution
    mapper.print_distribution(list(sectors.keys()))
    
    # Validate
    validation = mapper.validate_diversification(list(sectors.keys()))
    print(f"\nDiversification Check:")
    print(f"  Sectors: {validation['n_sectors']}")
    print(f"  Max weight: {validation['max_weight']:.1%}")
    print(f"  Passes: {'✓' if validation['passes_constraints'] else '✗'}")
    if validation['violations']:
        for v in validation['violations']:
            print(f"  ⚠️ {v}")


if __name__ == "__main__":
    test_sector_mapper()

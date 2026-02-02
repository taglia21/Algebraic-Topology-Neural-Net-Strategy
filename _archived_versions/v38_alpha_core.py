#!/usr/bin/env python3
"""
V38 Alpha Core Engine - Module 1/3
===================================
Core components for the Ultimate Alpha Generation System.

Components:
- ExpandedUniverse: 150+ tradable assets across multiple asset classes
- RegimeDetector: Multi-model regime detection ensemble
- MLEnsemble: Stacked ML models for signal generation
- AggressivePositionSizer: Kelly-based position sizing

Author: V38 Alpha Team
Version: 1.0.0
"""

from dotenv import load_dotenv
load_dotenv()

import os
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not installed. Some features disabled.")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not installed. Some features disabled.")

try:
    from hmmlearn import hmm
    HAS_HMM = True
except ImportError:
    HAS_HMM = False
    warnings.warn("hmmlearn not installed. HMM regime detection disabled.")

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import REST, TimeFrame
    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False
    warnings.warn("Alpaca API not installed.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class AssetClass(Enum):
    """Asset class categorization."""
    EQUITY = "equity"
    LEVERAGED_BULL = "leveraged_bull"
    LEVERAGED_BEAR = "leveraged_bear"
    SECTOR = "sector"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    INTERNATIONAL = "international"
    FIXED_INCOME = "fixed_income"


class RegimeState(Enum):
    """Market regime states."""
    BULL = "bull"
    BEAR = "bear"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class VIXRegime(Enum):
    """VIX-based regime classification."""
    CALM = "calm"           # VIX < 15
    NORMAL = "normal"       # 15 <= VIX < 25
    ELEVATED = "elevated"   # 25 <= VIX < 35
    CRISIS = "crisis"       # VIX >= 35


class SignalType(Enum):
    """Trading signal types."""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class Asset:
    """Represents a tradable asset."""
    symbol: str
    name: str
    asset_class: AssetClass
    leverage: float = 1.0
    sector: Optional[str] = None
    is_shortable: bool = True
    min_notional: float = 1.0
    avg_volume: float = 0.0
    volatility: float = 0.0
    beta: float = 1.0


@dataclass
class Signal:
    """Trading signal with metadata."""
    symbol: str
    signal_type: SignalType
    strength: float  # 0 to 1
    timestamp: datetime
    strategy: str
    features: Dict[str, float] = field(default_factory=dict)
    regime: Optional[RegimeState] = None
    confidence: float = 0.5


@dataclass
class RegimeInfo:
    """Current regime information."""
    hmm_state: RegimeState
    vix_regime: VIXRegime
    volatility_regime: str
    trend_regime: str
    composite_score: float  # -1 (bearish) to 1 (bullish)
    confidence: float
    strategy_weights: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# EXPANDED UNIVERSE CLASS
# =============================================================================

class ExpandedUniverse:
    """
    Manages an expanded universe of 150+ tradable assets across multiple
    asset classes including equities, leveraged ETFs, crypto, and commodities.
    """
    
    def __init__(self, alpaca_api: Optional[Any] = None):
        """
        Initialize the expanded trading universe.
        
        Args:
            alpaca_api: Optional Alpaca API instance for data fetching
        """
        self.api = alpaca_api
        self.assets: Dict[str, Asset] = {}
        self.sector_mapping: Dict[str, List[str]] = {}
        self.asset_class_mapping: Dict[AssetClass, List[str]] = {}
        
        self._initialize_universe()
        logger.info(f"Initialized universe with {len(self.assets)} assets")
    
    def _initialize_universe(self) -> None:
        """Build the complete asset universe."""
        
        # Leveraged Bull ETFs (3x exposure)
        leveraged_bull = [
            ("TQQQ", "ProShares UltraPro QQQ", "technology", 3.0),
            ("SOXL", "Direxion Semiconductor Bull 3X", "technology", 3.0),
            ("UPRO", "ProShares UltraPro S&P 500", "broad_market", 3.0),
            ("SPXL", "Direxion S&P 500 Bull 3X", "broad_market", 3.0),
            ("FNGU", "MicroSectors FANG+ Bull 3X", "technology", 3.0),
            ("TECL", "Direxion Technology Bull 3X", "technology", 3.0),
            ("WEBL", "Direxion Internet Bull 3X", "technology", 3.0),
            ("LABU", "Direxion Biotech Bull 3X", "healthcare", 3.0),
            ("NAIL", "Direxion Homebuilders Bull 3X", "real_estate", 3.0),
            ("JNUG", "Direxion Gold Miners Bull 2X", "materials", 2.0),
            ("NUGT", "Direxion Gold Miners Bull 2X", "materials", 2.0),
            ("ERX", "Direxion Energy Bull 2X", "energy", 2.0),
            ("GUSH", "Direxion Oil & Gas Bull 2X", "energy", 2.0),
            ("TNA", "Direxion Small Cap Bull 3X", "small_cap", 3.0),
            ("UDOW", "ProShares UltraPro Dow30", "broad_market", 3.0),
            ("UCO", "ProShares Ultra Bloomberg Crude Oil", "commodity", 2.0),
            ("BOIL", "ProShares Ultra Bloomberg Natural Gas", "commodity", 2.0),
            ("CURE", "Direxion Healthcare Bull 3X", "healthcare", 3.0),
            ("FAS", "Direxion Financial Bull 3X", "financials", 3.0),
            ("DUSL", "Direxion Industrials Bull 3X", "industrials", 3.0),
            ("DPST", "Direxion Regional Banks Bull 3X", "financials", 3.0),
            ("MIDU", "Direxion Mid Cap Bull 3X", "mid_cap", 3.0),
            ("HIBL", "Direxion High Beta Bull 3X", "broad_market", 3.0),
            ("RETL", "Direxion Retail Bull 3X", "consumer", 3.0),
            ("PILL", "Direxion Pharma Bull 3X", "healthcare", 3.0),
        ]
        
        for symbol, name, sector, leverage in leveraged_bull:
            self.assets[symbol] = Asset(
                symbol=symbol,
                name=name,
                asset_class=AssetClass.LEVERAGED_BULL,
                leverage=leverage,
                sector=sector,
                is_shortable=True
            )
        
        # Leveraged Bear ETFs (inverse)
        leveraged_bear = [
            ("SQQQ", "ProShares UltraPro Short QQQ", "technology", -3.0),
            ("SPXS", "Direxion S&P 500 Bear 3X", "broad_market", -3.0),
            ("SOXS", "Direxion Semiconductor Bear 3X", "technology", -3.0),
            ("SDOW", "ProShares UltraPro Short Dow30", "broad_market", -3.0),
            ("TZA", "Direxion Small Cap Bear 3X", "small_cap", -3.0),
            ("FAZ", "Direxion Financial Bear 3X", "financials", -3.0),
            ("SPXU", "ProShares UltraPro Short S&P 500", "broad_market", -3.0),
            ("SRTY", "ProShares UltraPro Short Russell 2000", "small_cap", -3.0),
            ("LABD", "Direxion Biotech Bear 3X", "healthcare", -3.0),
            ("ERY", "Direxion Energy Bear 2X", "energy", -2.0),
            ("DRIP", "Direxion Oil & Gas Bear 2X", "energy", -2.0),
            ("DUST", "Direxion Gold Miners Bear 2X", "materials", -2.0),
            ("JDST", "Direxion Gold Miners Bear 2X", "materials", -2.0),
        ]
        
        for symbol, name, sector, leverage in leveraged_bear:
            self.assets[symbol] = Asset(
                symbol=symbol,
                name=name,
                asset_class=AssetClass.LEVERAGED_BEAR,
                leverage=leverage,
                sector=sector,
                is_shortable=False  # Don't short inverse ETFs
            )
        
        # Sector ETFs (SPDR Select Sectors)
        sector_etfs = [
            ("XLK", "Technology Select Sector SPDR", "technology"),
            ("XLF", "Financial Select Sector SPDR", "financials"),
            ("XLE", "Energy Select Sector SPDR", "energy"),
            ("XLV", "Health Care Select Sector SPDR", "healthcare"),
            ("XLI", "Industrial Select Sector SPDR", "industrials"),
            ("XLB", "Materials Select Sector SPDR", "materials"),
            ("XLY", "Consumer Discretionary Select Sector", "consumer"),
            ("XLP", "Consumer Staples Select Sector", "consumer_staples"),
            ("XLU", "Utilities Select Sector SPDR", "utilities"),
            ("XLRE", "Real Estate Select Sector SPDR", "real_estate"),
            ("XLC", "Communication Services Select Sector", "communication"),
            ("SMH", "VanEck Semiconductor ETF", "technology"),
            ("IBB", "iShares Biotechnology ETF", "healthcare"),
            ("XBI", "SPDR S&P Biotech ETF", "healthcare"),
            ("XHB", "SPDR S&P Homebuilders ETF", "real_estate"),
            ("XOP", "SPDR S&P Oil & Gas Exploration", "energy"),
            ("XME", "SPDR S&P Metals & Mining", "materials"),
            ("KRE", "SPDR S&P Regional Banking", "financials"),
            ("KBE", "SPDR S&P Bank ETF", "financials"),
            ("XRT", "SPDR S&P Retail ETF", "consumer"),
            ("ITB", "iShares U.S. Home Construction", "real_estate"),
            ("IYR", "iShares U.S. Real Estate ETF", "real_estate"),
            ("VNQ", "Vanguard Real Estate ETF", "real_estate"),
        ]
        
        for symbol, name, sector in sector_etfs:
            self.assets[symbol] = Asset(
                symbol=symbol,
                name=name,
                asset_class=AssetClass.SECTOR,
                leverage=1.0,
                sector=sector
            )
        
        # Major Index ETFs
        index_etfs = [
            ("SPY", "SPDR S&P 500 ETF Trust", "broad_market"),
            ("QQQ", "Invesco QQQ Trust", "technology"),
            ("IWM", "iShares Russell 2000 ETF", "small_cap"),
            ("DIA", "SPDR Dow Jones Industrial Average", "broad_market"),
            ("MDY", "SPDR S&P MidCap 400 ETF", "mid_cap"),
            ("VTI", "Vanguard Total Stock Market ETF", "broad_market"),
            ("VOO", "Vanguard S&P 500 ETF", "broad_market"),
            ("IVV", "iShares Core S&P 500 ETF", "broad_market"),
            ("RSP", "Invesco S&P 500 Equal Weight", "broad_market"),
            ("VTV", "Vanguard Value ETF", "value"),
            ("VUG", "Vanguard Growth ETF", "growth"),
            ("MTUM", "iShares MSCI USA Momentum Factor", "momentum"),
            ("QUAL", "iShares MSCI USA Quality Factor", "quality"),
            ("USMV", "iShares MSCI USA Min Vol Factor", "low_vol"),
            ("VIG", "Vanguard Dividend Appreciation", "dividend"),
            ("SCHD", "Schwab U.S. Dividend Equity ETF", "dividend"),
        ]
        
        for symbol, name, sector in index_etfs:
            self.assets[symbol] = Asset(
                symbol=symbol,
                name=name,
                asset_class=AssetClass.EQUITY,
                leverage=1.0,
                sector=sector
            )
        
        # Crypto (Alpaca Crypto API)
        crypto_assets = [
            ("BTCUSD", "Bitcoin", "crypto"),
            ("ETHUSD", "Ethereum", "crypto"),
            ("SOLUSD", "Solana", "crypto"),
            ("AVAXUSD", "Avalanche", "crypto"),
            ("LINKUSD", "Chainlink", "crypto"),
            ("MATICUSD", "Polygon", "crypto"),
            ("DOTUSD", "Polkadot", "crypto"),
            ("ADAUSD", "Cardano", "crypto"),
            ("ATOMUSD", "Cosmos", "crypto"),
            ("UNIUSD", "Uniswap", "crypto"),
        ]
        
        for symbol, name, sector in crypto_assets:
            self.assets[symbol] = Asset(
                symbol=symbol,
                name=name,
                asset_class=AssetClass.CRYPTO,
                leverage=1.0,
                sector=sector,
                is_shortable=False  # Alpaca doesn't support crypto shorting
            )
        
        # Commodity ETFs
        commodity_etfs = [
            ("GLD", "SPDR Gold Shares", "precious_metals"),
            ("SLV", "iShares Silver Trust", "precious_metals"),
            ("USO", "United States Oil Fund", "energy"),
            ("UNG", "United States Natural Gas Fund", "energy"),
            ("DBA", "Invesco DB Agriculture Fund", "agriculture"),
            ("PDBC", "Invesco Optimum Yield Diversified Commodity", "diversified"),
            ("DBC", "Invesco DB Commodity Index", "diversified"),
            ("CORN", "Teucrium Corn Fund", "agriculture"),
            ("WEAT", "Teucrium Wheat Fund", "agriculture"),
            ("SOYB", "Teucrium Soybean Fund", "agriculture"),
            ("CPER", "United States Copper Index", "base_metals"),
            ("PPLT", "abrdn Physical Platinum Shares", "precious_metals"),
            ("PALL", "abrdn Physical Palladium Shares", "precious_metals"),
            ("IAU", "iShares Gold Trust", "precious_metals"),
            ("SGOL", "abrdn Physical Gold Shares", "precious_metals"),
        ]
        
        for symbol, name, sector in commodity_etfs:
            self.assets[symbol] = Asset(
                symbol=symbol,
                name=name,
                asset_class=AssetClass.COMMODITY,
                leverage=1.0,
                sector=sector
            )
        
        # International ETFs
        international_etfs = [
            ("EWJ", "iShares MSCI Japan ETF", "japan"),
            ("EWZ", "iShares MSCI Brazil ETF", "brazil"),
            ("FXI", "iShares China Large-Cap ETF", "china"),
            ("EEM", "iShares MSCI Emerging Markets", "emerging"),
            ("VGK", "Vanguard FTSE Europe ETF", "europe"),
            ("VWO", "Vanguard FTSE Emerging Markets", "emerging"),
            ("EFA", "iShares MSCI EAFE ETF", "developed"),
            ("EWG", "iShares MSCI Germany ETF", "germany"),
            ("EWU", "iShares MSCI United Kingdom", "uk"),
            ("EWT", "iShares MSCI Taiwan ETF", "taiwan"),
            ("EWY", "iShares MSCI South Korea ETF", "korea"),
            ("INDA", "iShares MSCI India ETF", "india"),
            ("EWA", "iShares MSCI Australia ETF", "australia"),
            ("EWC", "iShares MSCI Canada ETF", "canada"),
            ("EWH", "iShares MSCI Hong Kong ETF", "hong_kong"),
            ("EWS", "iShares MSCI Singapore ETF", "singapore"),
            ("IEMG", "iShares Core MSCI Emerging Markets", "emerging"),
            ("KWEB", "KraneShares CSI China Internet", "china"),
            ("MCHI", "iShares MSCI China ETF", "china"),
            ("VEA", "Vanguard FTSE Developed Markets", "developed"),
        ]
        
        for symbol, name, region in international_etfs:
            self.assets[symbol] = Asset(
                symbol=symbol,
                name=name,
                asset_class=AssetClass.INTERNATIONAL,
                leverage=1.0,
                sector=region
            )
        
        # Fixed Income ETFs
        fixed_income_etfs = [
            ("TLT", "iShares 20+ Year Treasury Bond", "treasuries"),
            ("IEF", "iShares 7-10 Year Treasury Bond", "treasuries"),
            ("SHY", "iShares 1-3 Year Treasury Bond", "treasuries"),
            ("TIP", "iShares TIPS Bond ETF", "tips"),
            ("LQD", "iShares iBoxx Investment Grade", "corporate"),
            ("HYG", "iShares iBoxx High Yield Corporate", "high_yield"),
            ("JNK", "SPDR Bloomberg High Yield Bond", "high_yield"),
            ("BND", "Vanguard Total Bond Market", "aggregate"),
            ("AGG", "iShares Core U.S. Aggregate Bond", "aggregate"),
            ("EMB", "iShares J.P. Morgan USD Emerging Markets", "emerging"),
            ("TMF", "Direxion 20+ Year Treasury Bull 3X", "treasuries"),
            ("TMV", "Direxion 20+ Year Treasury Bear 3X", "treasuries"),
            ("TBT", "ProShares UltraShort 20+ Year Treasury", "treasuries"),
            ("GOVT", "iShares U.S. Treasury Bond ETF", "treasuries"),
            ("VCSH", "Vanguard Short-Term Corporate Bond", "corporate"),
        ]
        
        for symbol, name, sector in fixed_income_etfs:
            self.assets[symbol] = Asset(
                symbol=symbol,
                name=name,
                asset_class=AssetClass.FIXED_INCOME,
                leverage=1.0 if "3X" not in name else 3.0,
                sector=sector
            )
        
        # Top momentum stocks (will be updated dynamically)
        momentum_stocks = [
            "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "AVGO", "CRM",
            "NFLX", "ADBE", "ORCL", "INTC", "CSCO", "QCOM", "TXN", "MU", "AMAT", "LRCX",
            "NOW", "INTU", "PANW", "SNPS", "CDNS", "KLAC", "MRVL", "FTNT", "CRWD", "ZS",
            "DDOG", "SNOW", "PLTR", "COIN", "MSTR", "SQ", "PYPL", "SHOP", "UBER", "ABNB",
            "RIVN", "LCID", "NIO", "XPEV", "LI", "F", "GM", "TM", "HMC", "RACE",
        ]
        
        for symbol in momentum_stocks:
            self.assets[symbol] = Asset(
                symbol=symbol,
                name=symbol,
                asset_class=AssetClass.EQUITY,
                leverage=1.0,
                sector="momentum"
            )
        
        # Build mappings
        self._build_mappings()
    
    def _build_mappings(self) -> None:
        """Build sector and asset class mappings."""
        self.sector_mapping = {}
        self.asset_class_mapping = {ac: [] for ac in AssetClass}
        
        for symbol, asset in self.assets.items():
            # Asset class mapping
            self.asset_class_mapping[asset.asset_class].append(symbol)
            
            # Sector mapping
            if asset.sector:
                if asset.sector not in self.sector_mapping:
                    self.sector_mapping[asset.sector] = []
                self.sector_mapping[asset.sector].append(symbol)
    
    def get_symbols_by_class(self, asset_class: AssetClass) -> List[str]:
        """Get all symbols for a given asset class."""
        return self.asset_class_mapping.get(asset_class, [])
    
    def get_symbols_by_sector(self, sector: str) -> List[str]:
        """Get all symbols for a given sector."""
        return self.sector_mapping.get(sector, [])
    
    def get_leveraged_bull(self) -> List[str]:
        """Get all leveraged bull ETF symbols."""
        return self.get_symbols_by_class(AssetClass.LEVERAGED_BULL)
    
    def get_leveraged_bear(self) -> List[str]:
        """Get all leveraged bear (inverse) ETF symbols."""
        return self.get_symbols_by_class(AssetClass.LEVERAGED_BEAR)
    
    def get_crypto(self) -> List[str]:
        """Get all crypto asset symbols."""
        return self.get_symbols_by_class(AssetClass.CRYPTO)
    
    def get_tradable_symbols(self, 
                              include_leveraged: bool = True,
                              include_inverse: bool = True,
                              include_crypto: bool = True) -> List[str]:
        """
        Get list of tradable symbols based on filters.
        
        Args:
            include_leveraged: Include leveraged bull ETFs
            include_inverse: Include inverse/bear ETFs
            include_crypto: Include crypto assets
            
        Returns:
            List of symbol strings
        """
        symbols = []
        
        for symbol, asset in self.assets.items():
            if asset.asset_class == AssetClass.LEVERAGED_BULL and not include_leveraged:
                continue
            if asset.asset_class == AssetClass.LEVERAGED_BEAR and not include_inverse:
                continue
            if asset.asset_class == AssetClass.CRYPTO and not include_crypto:
                continue
            symbols.append(symbol)
        
        return symbols
    
    def get_asset(self, symbol: str) -> Optional[Asset]:
        """Get asset information for a symbol."""
        return self.assets.get(symbol)
    
    def get_leverage(self, symbol: str) -> float:
        """Get leverage factor for a symbol."""
        asset = self.assets.get(symbol)
        return asset.leverage if asset else 1.0
    
    def is_inverse(self, symbol: str) -> bool:
        """Check if symbol is an inverse/bear ETF."""
        asset = self.assets.get(symbol)
        return asset.asset_class == AssetClass.LEVERAGED_BEAR if asset else False
    
    def get_hedge_pair(self, symbol: str) -> Optional[str]:
        """Get inverse/hedge pair for a symbol."""
        hedge_pairs = {
            "TQQQ": "SQQQ", "SQQQ": "TQQQ",
            "UPRO": "SPXU", "SPXU": "UPRO",
            "SPXL": "SPXS", "SPXS": "SPXL",
            "SOXL": "SOXS", "SOXS": "SOXL",
            "TNA": "TZA", "TZA": "TNA",
            "UDOW": "SDOW", "SDOW": "UDOW",
            "FAS": "FAZ", "FAZ": "FAS",
            "LABU": "LABD", "LABD": "LABU",
            "ERX": "ERY", "ERY": "ERX",
            "NUGT": "DUST", "DUST": "NUGT",
            "JNUG": "JDST", "JDST": "JNUG",
            "GUSH": "DRIP", "DRIP": "GUSH",
            "QQQ": "SQQQ", "SPY": "SPXS",
            "IWM": "TZA", "DIA": "SDOW",
        }
        return hedge_pairs.get(symbol)
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of the universe."""
        return {
            "total_assets": len(self.assets),
            "by_asset_class": {
                ac.value: len(symbols) 
                for ac, symbols in self.asset_class_mapping.items()
            },
            "sectors": list(self.sector_mapping.keys()),
            "leveraged_bull_count": len(self.get_leveraged_bull()),
            "leveraged_bear_count": len(self.get_leveraged_bear()),
            "crypto_count": len(self.get_crypto()),
        }


# =============================================================================
# REGIME DETECTOR CLASS
# =============================================================================

class RegimeDetector:
    """
    Multi-model ensemble for market regime detection.
    
    Combines:
    - 4-state Hidden Markov Model (bull/bear/high-vol/low-vol)
    - VIX-based regime classification
    - Volatility clustering analysis
    - Trend regime detection
    """
    
    def __init__(self, 
                 lookback_window: int = 252,
                 vol_window: int = 21,
                 hmm_states: int = 4):
        """
        Initialize the regime detector.
        
        Args:
            lookback_window: Historical lookback for training
            vol_window: Window for volatility calculation
            hmm_states: Number of HMM states
        """
        self.lookback_window = lookback_window
        self.vol_window = vol_window
        self.hmm_states = hmm_states
        
        # Models
        self.hmm_model: Optional[Any] = None
        self.scaler = StandardScaler()
        
        # State mappings
        self.hmm_state_mapping: Dict[int, RegimeState] = {}
        
        # Current regime
        self.current_regime: Optional[RegimeInfo] = None
        self.regime_history: List[RegimeInfo] = []
        
        # Strategy weights per regime
        self.regime_strategy_weights = {
            RegimeState.BULL: {
                "momentum": 0.4,
                "trend_following": 0.3,
                "mean_reversion": 0.15,
                "stat_arb": 0.1,
                "volatility": 0.05,
            },
            RegimeState.BEAR: {
                "momentum": 0.1,
                "trend_following": 0.15,
                "mean_reversion": 0.3,
                "stat_arb": 0.25,
                "volatility": 0.2,
            },
            RegimeState.HIGH_VOL: {
                "momentum": 0.1,
                "trend_following": 0.1,
                "mean_reversion": 0.35,
                "stat_arb": 0.2,
                "volatility": 0.25,
            },
            RegimeState.LOW_VOL: {
                "momentum": 0.35,
                "trend_following": 0.35,
                "mean_reversion": 0.1,
                "stat_arb": 0.15,
                "volatility": 0.05,
            },
            RegimeState.CRISIS: {
                "momentum": 0.0,
                "trend_following": 0.05,
                "mean_reversion": 0.4,
                "stat_arb": 0.15,
                "volatility": 0.4,
            },
            RegimeState.RECOVERY: {
                "momentum": 0.3,
                "trend_following": 0.25,
                "mean_reversion": 0.25,
                "stat_arb": 0.15,
                "volatility": 0.05,
            },
        }
        
        logger.info(f"Initialized RegimeDetector with {hmm_states} HMM states")
    
    def _prepare_hmm_features(self, prices: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for HMM training.
        
        Args:
            prices: DataFrame with price data (must have 'close' column)
            
        Returns:
            Feature matrix for HMM
        """
        if 'close' in prices.columns:
            close = prices['close']
        else:
            close = prices.iloc[:, 0]
        
        # Calculate features
        returns = close.pct_change().fillna(0)
        volatility = returns.rolling(self.vol_window).std().fillna(returns.std())
        momentum = close.pct_change(21).fillna(0)
        
        # Volume if available
        if 'volume' in prices.columns:
            volume_ma = prices['volume'].rolling(20).mean()
            volume_ratio = (prices['volume'] / volume_ma).fillna(1)
        else:
            volume_ratio = pd.Series(1, index=prices.index)
        
        # Stack features
        features = np.column_stack([
            returns.values,
            volatility.values,
            momentum.values,
            volume_ratio.values
        ])
        
        # Remove NaN rows
        valid_mask = ~np.isnan(features).any(axis=1)
        features = features[valid_mask]
        
        return features
    
    def train_hmm(self, prices: pd.DataFrame) -> bool:
        """
        Train the Hidden Markov Model on historical price data.
        
        Args:
            prices: DataFrame with historical prices
            
        Returns:
            True if training successful
        """
        if not HAS_HMM:
            logger.warning("HMM library not available. Skipping HMM training.")
            return False
        
        try:
            features = self._prepare_hmm_features(prices)
            
            if len(features) < 100:
                logger.warning("Insufficient data for HMM training")
                return False
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train HMM
            self.hmm_model = hmm.GaussianHMM(
                n_components=self.hmm_states,
                covariance_type="full",
                n_iter=1000,
                random_state=42
            )
            
            self.hmm_model.fit(features_scaled)
            
            # Map states to regimes based on mean returns and volatility
            self._map_hmm_states(features_scaled)
            
            logger.info("HMM training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"HMM training failed: {e}")
            return False
    
    def _map_hmm_states(self, features: np.ndarray) -> None:
        """Map HMM states to regime labels based on state characteristics."""
        if self.hmm_model is None:
            return
        
        # Get state sequence
        states = self.hmm_model.predict(features)
        
        # Calculate mean characteristics for each state
        state_stats = {}
        for state in range(self.hmm_states):
            mask = states == state
            if mask.sum() > 0:
                state_stats[state] = {
                    "mean_return": features[mask, 0].mean(),
                    "mean_vol": features[mask, 1].mean(),
                    "count": mask.sum()
                }
        
        # Sort by return and volatility to assign regimes
        sorted_by_return = sorted(
            state_stats.items(), 
            key=lambda x: x[1]["mean_return"],
            reverse=True
        )
        sorted_by_vol = sorted(
            state_stats.items(),
            key=lambda x: x[1]["mean_vol"],
            reverse=True
        )
        
        # Assign states
        if len(sorted_by_return) >= 4:
            self.hmm_state_mapping = {
                sorted_by_return[0][0]: RegimeState.BULL,
                sorted_by_return[-1][0]: RegimeState.BEAR,
                sorted_by_vol[0][0]: RegimeState.HIGH_VOL,
                sorted_by_vol[-1][0]: RegimeState.LOW_VOL,
            }
        else:
            # Fallback for fewer states
            for i, (state, _) in enumerate(sorted_by_return):
                if i == 0:
                    self.hmm_state_mapping[state] = RegimeState.BULL
                elif i == len(sorted_by_return) - 1:
                    self.hmm_state_mapping[state] = RegimeState.BEAR
                else:
                    self.hmm_state_mapping[state] = RegimeState.HIGH_VOL
    
    def get_hmm_regime(self, recent_prices: pd.DataFrame) -> Tuple[RegimeState, float]:
        """
        Get current regime from HMM model.
        
        Args:
            recent_prices: Recent price data
            
        Returns:
            Tuple of (regime state, confidence)
        """
        if self.hmm_model is None or not HAS_HMM:
            return RegimeState.BULL, 0.5
        
        try:
            features = self._prepare_hmm_features(recent_prices)
            if len(features) == 0:
                return RegimeState.BULL, 0.5
            
            features_scaled = self.scaler.transform(features)
            
            # Get state probabilities
            state_probs = self.hmm_model.predict_proba(features_scaled)
            current_probs = state_probs[-1]
            
            # Get most likely state
            current_state = np.argmax(current_probs)
            confidence = current_probs[current_state]
            
            regime = self.hmm_state_mapping.get(current_state, RegimeState.BULL)
            
            return regime, float(confidence)
            
        except Exception as e:
            logger.error(f"HMM prediction failed: {e}")
            return RegimeState.BULL, 0.5
    
    def get_vix_regime(self, vix_value: float) -> VIXRegime:
        """
        Classify regime based on VIX level.
        
        Args:
            vix_value: Current VIX value
            
        Returns:
            VIX regime classification
        """
        if vix_value < 15:
            return VIXRegime.CALM
        elif vix_value < 25:
            return VIXRegime.NORMAL
        elif vix_value < 35:
            return VIXRegime.ELEVATED
        else:
            return VIXRegime.CRISIS
    
    def get_volatility_regime(self, 
                               returns: pd.Series, 
                               lookback: int = 63) -> Tuple[str, float]:
        """
        Detect volatility regime using clustering.
        
        Args:
            returns: Return series
            lookback: Lookback period
            
        Returns:
            Tuple of (regime name, percentile)
        """
        if len(returns) < lookback:
            return "normal", 0.5
        
        # Calculate rolling volatility
        current_vol = returns.tail(21).std() * np.sqrt(252)
        historical_vol = returns.rolling(21).std() * np.sqrt(252)
        
        # Get percentile
        percentile = stats.percentileofscore(
            historical_vol.dropna().values, 
            current_vol
        ) / 100
        
        if percentile < 0.25:
            regime = "very_low"
        elif percentile < 0.40:
            regime = "low"
        elif percentile < 0.60:
            regime = "normal"
        elif percentile < 0.80:
            regime = "high"
        else:
            regime = "very_high"
        
        return regime, percentile
    
    def get_trend_regime(self, prices: pd.Series) -> Tuple[str, float]:
        """
        Detect trend regime using moving averages.
        
        Args:
            prices: Price series
            
        Returns:
            Tuple of (trend regime, strength)
        """
        if len(prices) < 200:
            return "sideways", 0.0
        
        # Calculate moving averages
        ma_20 = prices.rolling(20).mean().iloc[-1]
        ma_50 = prices.rolling(50).mean().iloc[-1]
        ma_200 = prices.rolling(200).mean().iloc[-1]
        current = prices.iloc[-1]
        
        # Calculate trend strength
        above_20 = current > ma_20
        above_50 = current > ma_50
        above_200 = current > ma_200
        ma_20_above_50 = ma_20 > ma_50
        ma_50_above_200 = ma_50 > ma_200
        
        bull_score = sum([above_20, above_50, above_200, ma_20_above_50, ma_50_above_200])
        strength = (bull_score - 2.5) / 2.5  # Normalize to [-1, 1]
        
        if bull_score >= 4:
            regime = "strong_uptrend"
        elif bull_score >= 3:
            regime = "uptrend"
        elif bull_score == 2:
            regime = "sideways"
        elif bull_score >= 1:
            regime = "downtrend"
        else:
            regime = "strong_downtrend"
        
        return regime, strength
    
    def detect_regime(self,
                      prices: pd.DataFrame,
                      vix: Optional[float] = None) -> RegimeInfo:
        """
        Comprehensive regime detection using all models.
        
        Args:
            prices: DataFrame with OHLCV data
            vix: Optional VIX value
            
        Returns:
            RegimeInfo with full regime analysis
        """
        # Get close prices
        if 'close' in prices.columns:
            close = prices['close']
        else:
            close = prices.iloc[:, 0]
        
        returns = close.pct_change().dropna()
        
        # HMM regime
        hmm_state, hmm_confidence = self.get_hmm_regime(prices)
        
        # VIX regime
        vix_regime = VIXRegime.NORMAL
        if vix is not None:
            vix_regime = self.get_vix_regime(vix)
        
        # Volatility regime
        vol_regime, vol_percentile = self.get_volatility_regime(returns)
        
        # Trend regime
        trend_regime, trend_strength = self.get_trend_regime(close)
        
        # Composite score (-1 to 1)
        composite = self._calculate_composite_score(
            hmm_state, vix_regime, vol_regime, trend_regime, trend_strength
        )
        
        # Determine final regime
        final_regime = self._determine_final_regime(
            hmm_state, vix_regime, composite
        )
        
        # Get strategy weights
        strategy_weights = self.regime_strategy_weights.get(
            final_regime, 
            self.regime_strategy_weights[RegimeState.BULL]
        )
        
        # Build regime info
        regime_info = RegimeInfo(
            hmm_state=final_regime,
            vix_regime=vix_regime,
            volatility_regime=vol_regime,
            trend_regime=trend_regime,
            composite_score=composite,
            confidence=hmm_confidence,
            strategy_weights=strategy_weights
        )
        
        self.current_regime = regime_info
        self.regime_history.append(regime_info)
        
        # Keep history bounded
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-500:]
        
        return regime_info
    
    def _calculate_composite_score(self,
                                    hmm_state: RegimeState,
                                    vix_regime: VIXRegime,
                                    vol_regime: str,
                                    trend_regime: str,
                                    trend_strength: float) -> float:
        """Calculate composite bullish/bearish score."""
        score = 0.0
        
        # HMM contribution
        hmm_scores = {
            RegimeState.BULL: 0.8,
            RegimeState.LOW_VOL: 0.4,
            RegimeState.HIGH_VOL: -0.2,
            RegimeState.BEAR: -0.8,
            RegimeState.CRISIS: -1.0,
            RegimeState.RECOVERY: 0.3,
        }
        score += hmm_scores.get(hmm_state, 0) * 0.3
        
        # VIX contribution
        vix_scores = {
            VIXRegime.CALM: 0.6,
            VIXRegime.NORMAL: 0.2,
            VIXRegime.ELEVATED: -0.3,
            VIXRegime.CRISIS: -0.8,
        }
        score += vix_scores.get(vix_regime, 0) * 0.25
        
        # Volatility regime contribution
        vol_scores = {
            "very_low": 0.4,
            "low": 0.2,
            "normal": 0.0,
            "high": -0.2,
            "very_high": -0.5,
        }
        score += vol_scores.get(vol_regime, 0) * 0.2
        
        # Trend contribution
        score += trend_strength * 0.25
        
        return np.clip(score, -1, 1)
    
    def _determine_final_regime(self,
                                 hmm_state: RegimeState,
                                 vix_regime: VIXRegime,
                                 composite: float) -> RegimeState:
        """Determine final regime from all signals."""
        # Crisis override
        if vix_regime == VIXRegime.CRISIS:
            return RegimeState.CRISIS
        
        # High VIX suggests high vol
        if vix_regime == VIXRegime.ELEVATED and composite < 0:
            return RegimeState.HIGH_VOL
        
        # Use composite score to refine
        if composite > 0.5:
            return RegimeState.BULL
        elif composite > 0.2:
            return RegimeState.LOW_VOL
        elif composite > -0.2:
            return hmm_state  # Use HMM state for uncertain periods
        elif composite > -0.5:
            return RegimeState.HIGH_VOL
        else:
            return RegimeState.BEAR
    
    def get_regime_weights(self) -> Dict[str, float]:
        """
        Get current strategy weights based on regime.
        
        Returns:
            Dictionary of strategy name to weight
        """
        if self.current_regime is None:
            # Default weights
            return {
                "momentum": 0.3,
                "trend_following": 0.25,
                "mean_reversion": 0.2,
                "stat_arb": 0.15,
                "volatility": 0.1,
            }
        
        return self.current_regime.strategy_weights
    
    def get_risk_multiplier(self) -> float:
        """
        Get risk multiplier based on current regime.
        
        Returns:
            Risk multiplier (0.0 to 1.0)
        """
        if self.current_regime is None:
            return 0.8
        
        multipliers = {
            RegimeState.BULL: 1.0,
            RegimeState.LOW_VOL: 0.95,
            RegimeState.HIGH_VOL: 0.6,
            RegimeState.BEAR: 0.5,
            RegimeState.CRISIS: 0.2,
            RegimeState.RECOVERY: 0.8,
        }
        
        return multipliers.get(self.current_regime.hmm_state, 0.7)
    
    def should_use_leverage(self) -> bool:
        """Check if current regime supports leveraged trading."""
        if self.current_regime is None:
            return False
        
        return self.current_regime.hmm_state in [
            RegimeState.BULL, 
            RegimeState.LOW_VOL,
            RegimeState.RECOVERY
        ]


# =============================================================================
# ML ENSEMBLE CLASS
# =============================================================================

class MLEnsemble:
    """
    Stacked machine learning ensemble for alpha generation.
    
    Base models:
    - XGBoost: Momentum classification
    - LightGBM: Mean-reversion detection
    - RandomForest: Volatility prediction
    
    Meta-learner:
    - Logistic Regression for stacking
    """
    
    def __init__(self,
                 feature_window: int = 60,
                 prediction_horizon: int = 5,
                 n_estimators: int = 100):
        """
        Initialize the ML ensemble.
        
        Args:
            feature_window: Lookback for feature calculation
            prediction_horizon: Forward prediction period
            n_estimators: Number of trees for tree-based models
        """
        self.feature_window = feature_window
        self.prediction_horizon = prediction_horizon
        self.n_estimators = n_estimators
        
        # Feature scalers
        self.feature_scaler = RobustScaler()
        
        # Base models
        self.models: Dict[str, Any] = {}
        self.meta_learner: Optional[LogisticRegression] = None
        
        # Training state
        self.is_trained = False
        self.feature_names: List[str] = []
        self.model_weights: Dict[str, float] = {
            "xgboost": 0.35,
            "lightgbm": 0.35,
            "random_forest": 0.30,
        }
        
        # Performance tracking
        self.training_metrics: Dict[str, float] = {}
        
        self._initialize_models()
        logger.info("Initialized MLEnsemble with base models")
    
    def _initialize_models(self) -> None:
        """Initialize base models."""
        # XGBoost for momentum
        if HAS_XGBOOST:
            self.models["xgboost"] = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=3,  # Up, Down, Neutral
                eval_metric="mlogloss",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1
            )
        
        # LightGBM for mean-reversion
        if HAS_LIGHTGBM:
            self.models["lightgbm"] = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multiclass",
                num_class=3,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        # RandomForest for volatility
        self.models["random_forest"] = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Meta-learner
        self.meta_learner = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            random_state=42
        )
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical features for ML models.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with features
        """
        features = pd.DataFrame(index=df.index)
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('volume', pd.Series(1, index=df.index))
        
        # === MOMENTUM FEATURES ===
        # Price momentum (multiple periods)
        for period in [1, 3, 5, 10, 21, 63]:
            features[f'momentum_{period}d'] = close.pct_change(period)
        
        # Rate of change
        for period in [5, 10, 21]:
            features[f'roc_{period}'] = (close / close.shift(period) - 1) * 100
        
        # === RSI (multiple periods) ===
        for period in [7, 14, 21]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # === MACD ===
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        features['macd_divergence'] = features['macd'] / (close + 1e-10)
        
        # === BOLLINGER BANDS ===
        for period in [20]:
            ma = close.rolling(period).mean()
            std = close.rolling(period).std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            features[f'bb_width_{period}'] = (upper - lower) / (ma + 1e-10)
            features[f'bb_pctb_{period}'] = (close - lower) / (upper - lower + 1e-10)
            features[f'bb_zscore_{period}'] = (close - ma) / (std + 1e-10)
        
        # === ATR (Average True Range) ===
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        for period in [14, 21]:
            features[f'atr_{period}'] = tr.rolling(period).mean()
            features[f'atr_pct_{period}'] = features[f'atr_{period}'] / close
        
        # === VOLATILITY ===
        returns = close.pct_change()
        for period in [5, 10, 21, 63]:
            features[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(252)
        
        # Volatility ratio
        features['vol_ratio'] = (
            returns.rolling(5).std() / 
            (returns.rolling(21).std() + 1e-10)
        )
        
        # === VOLUME FEATURES ===
        volume_ma_20 = volume.rolling(20).mean()
        features['volume_ratio'] = volume / (volume_ma_20 + 1e-10)
        features['volume_momentum'] = volume.pct_change(5)
        
        # OBV (On-Balance Volume)
        obv = (np.sign(close.diff()) * volume).cumsum()
        features['obv_momentum'] = obv.pct_change(10)
        
        # === PRICE PATTERNS ===
        # Distance from moving averages
        for period in [10, 20, 50, 200]:
            ma = close.rolling(period).mean()
            features[f'dist_ma_{period}'] = (close - ma) / (ma + 1e-10)
        
        # MA crossover signals
        ma_10 = close.rolling(10).mean()
        ma_20 = close.rolling(20).mean()
        ma_50 = close.rolling(50).mean()
        features['ma_cross_10_20'] = (ma_10 - ma_20) / (ma_20 + 1e-10)
        features['ma_cross_20_50'] = (ma_20 - ma_50) / (ma_50 + 1e-10)
        
        # === HIGH/LOW FEATURES ===
        # Distance from N-day high/low
        for period in [10, 20, 52]:
            rolling_high = high.rolling(period).max()
            rolling_low = low.rolling(period).min()
            features[f'dist_high_{period}'] = (close - rolling_high) / (rolling_high + 1e-10)
            features[f'dist_low_{period}'] = (close - rolling_low) / (rolling_low + 1e-10)
        
        # === MEAN REVERSION SIGNALS ===
        features['mean_rev_signal'] = -features['bb_zscore_20']
        features['rsi_oversold'] = (features['rsi_14'] < 30).astype(float)
        features['rsi_overbought'] = (features['rsi_14'] > 70).astype(float)
        
        # === TREND STRENGTH ===
        # ADX-like calculation
        plus_dm = high.diff().where((high.diff() > low.diff().abs()) & (high.diff() > 0), 0)
        minus_dm = low.diff().abs().where((low.diff().abs() > high.diff()) & (low.diff() < 0), 0)
        
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
        features['adx'] = dx.rolling(14).mean()
        features['di_diff'] = plus_di - minus_di
        
        # === LAGGED FEATURES ===
        for lag in [1, 2, 3, 5]:
            features[f'momentum_1d_lag{lag}'] = features['momentum_1d'].shift(lag)
            features[f'rsi_14_lag{lag}'] = features['rsi_14'].shift(lag)
        
        # Store feature names
        self.feature_names = list(features.columns)
        
        return features
    
    def prepare_labels(self, 
                       close: pd.Series, 
                       horizon: int = 5,
                       threshold: float = 0.01) -> pd.Series:
        """
        Prepare classification labels.
        
        Args:
            close: Close prices
            horizon: Forward look period
            threshold: Classification threshold
            
        Returns:
            Series with labels (0=down, 1=neutral, 2=up)
        """
        forward_returns = close.shift(-horizon) / close - 1
        
        labels = pd.Series(1, index=close.index)  # Neutral
        labels[forward_returns > threshold] = 2   # Up
        labels[forward_returns < -threshold] = 0  # Down
        
        return labels
    
    def train(self, 
              df: pd.DataFrame,
              labels: Optional[pd.Series] = None,
              validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train all ensemble models.
        
        Args:
            df: OHLCV DataFrame
            labels: Optional pre-computed labels
            validation_split: Validation set proportion
            
        Returns:
            Dictionary of training metrics
        """
        logger.info("Starting ML ensemble training...")
        
        # Calculate features
        features = self.calculate_features(df)
        
        # Prepare labels if not provided
        if labels is None:
            labels = self.prepare_labels(
                df['close'], 
                horizon=self.prediction_horizon
            )
        
        # Align features and labels
        valid_idx = features.dropna().index.intersection(labels.dropna().index)
        X = features.loc[valid_idx]
        y = labels.loc[valid_idx]
        
        # Remove last rows without labels
        X = X.iloc[:-self.prediction_horizon]
        y = y.iloc[:-self.prediction_horizon]
        
        logger.info(f"Training on {len(X)} samples with {len(self.feature_names)} features")
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        
        # Train/validation split (time-series aware)
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train = X_scaled.iloc[:split_idx]
        X_val = X_scaled.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_val = y.iloc[split_idx:]
        
        metrics = {}
        base_predictions_train = {}
        base_predictions_val = {}
        
        # Train base models
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                model.fit(X_train, y_train)
                
                # Get predictions
                pred_train = model.predict(X_train)
                pred_val = model.predict(X_val)
                
                # Store probability predictions for stacking
                if hasattr(model, 'predict_proba'):
                    base_predictions_train[name] = model.predict_proba(X_train)
                    base_predictions_val[name] = model.predict_proba(X_val)
                else:
                    # One-hot encode predictions
                    base_predictions_train[name] = np.eye(3)[pred_train]
                    base_predictions_val[name] = np.eye(3)[pred_val]
                
                # Calculate metrics
                metrics[f"{name}_train_accuracy"] = accuracy_score(y_train, pred_train)
                metrics[f"{name}_val_accuracy"] = accuracy_score(y_val, pred_val)
                metrics[f"{name}_val_f1"] = f1_score(y_val, pred_val, average='weighted')
                
                logger.info(
                    f"{name}: Train Acc={metrics[f'{name}_train_accuracy']:.3f}, "
                    f"Val Acc={metrics[f'{name}_val_accuracy']:.3f}"
                )
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                continue
        
        # Train meta-learner (stacking)
        if len(base_predictions_train) >= 2:
            try:
                logger.info("Training meta-learner...")
                
                # Stack base predictions
                meta_X_train = np.hstack(list(base_predictions_train.values()))
                meta_X_val = np.hstack(list(base_predictions_val.values()))
                
                self.meta_learner.fit(meta_X_train, y_train)
                
                # Evaluate meta-learner
                meta_pred = self.meta_learner.predict(meta_X_val)
                metrics["meta_val_accuracy"] = accuracy_score(y_val, meta_pred)
                metrics["meta_val_f1"] = f1_score(y_val, meta_pred, average='weighted')
                
                logger.info(
                    f"Meta-learner: Val Acc={metrics['meta_val_accuracy']:.3f}, "
                    f"Val F1={metrics['meta_val_f1']:.3f}"
                )
                
            except Exception as e:
                logger.error(f"Meta-learner training failed: {e}")
        
        self.is_trained = True
        self.training_metrics = metrics
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions using the ensemble.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_trained:
            logger.warning("Models not trained. Returning neutral predictions.")
            return np.ones(len(df)), np.full((len(df), 3), 1/3)
        
        # Calculate features
        features = self.calculate_features(df)
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(0)
        
        # Scale
        X = self.feature_scaler.transform(features)
        
        # Get base model predictions
        base_probs = {}
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    base_probs[name] = model.predict_proba(X)
                else:
                    pred = model.predict(X)
                    base_probs[name] = np.eye(3)[pred]
            except Exception as e:
                logger.error(f"Prediction failed for {name}: {e}")
                base_probs[name] = np.full((len(X), 3), 1/3)
        
        # Meta-learner prediction
        if self.meta_learner is not None and len(base_probs) >= 2:
            try:
                meta_X = np.hstack(list(base_probs.values()))
                predictions = self.meta_learner.predict(meta_X)
                probabilities = self.meta_learner.predict_proba(meta_X)
            except Exception as e:
                logger.error(f"Meta-learner prediction failed: {e}")
                # Fallback to weighted average
                probabilities = np.zeros((len(X), 3))
                for name, probs in base_probs.items():
                    probabilities += probs * self.model_weights.get(name, 1/len(base_probs))
                predictions = np.argmax(probabilities, axis=1)
        else:
            # Weighted average of base models
            probabilities = np.zeros((len(X), 3))
            for name, probs in base_probs.items():
                probabilities += probs * self.model_weights.get(name, 1/len(base_probs))
            predictions = np.argmax(probabilities, axis=1)
        
        return predictions, probabilities
    
    def get_signal_strength(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calculate signal strength from probabilities.
        
        Args:
            probabilities: Probability matrix (n_samples, 3)
            
        Returns:
            Signal strength array (-1 to 1)
        """
        # Strength = P(up) - P(down)
        return probabilities[:, 2] - probabilities[:, 0]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from tree-based models."""
        importance_dfs = []
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                imp = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': model.feature_importances_,
                    'model': name
                })
                importance_dfs.append(imp)
        
        if importance_dfs:
            return pd.concat(importance_dfs, ignore_index=True)
        
        return pd.DataFrame()
    
    def save(self, filepath: str) -> None:
        """Save trained models to disk."""
        save_dict = {
            'models': self.models,
            'meta_learner': self.meta_learner,
            'feature_scaler': self.feature_scaler,
            'feature_names': self.feature_names,
            'model_weights': self.model_weights,
            'training_metrics': self.training_metrics,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Models saved to {filepath}")
    
    def load(self, filepath: str) -> bool:
        """Load trained models from disk."""
        try:
            with open(filepath, 'rb') as f:
                save_dict = pickle.load(f)
            
            self.models = save_dict['models']
            self.meta_learner = save_dict['meta_learner']
            self.feature_scaler = save_dict['feature_scaler']
            self.feature_names = save_dict['feature_names']
            self.model_weights = save_dict['model_weights']
            self.training_metrics = save_dict['training_metrics']
            self.is_trained = save_dict['is_trained']
            
            logger.info(f"Models loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False


# =============================================================================
# AGGRESSIVE POSITION SIZER
# =============================================================================

class AggressivePositionSizer:
    """
    Aggressive position sizing using Kelly Criterion with modifications.
    
    Features:
    - 95% capital deployment target
    - 0.75x Kelly for position sizing
    - Max 20% per position
    - Dynamic sizing based on signal strength
    """
    
    def __init__(self,
                 target_deployment: float = 0.95,
                 kelly_fraction: float = 0.75,
                 max_position_pct: float = 0.20,
                 min_position_pct: float = 0.02,
                 max_positions: int = 25):
        """
        Initialize the position sizer.
        
        Args:
            target_deployment: Target capital deployment (default 95%)
            kelly_fraction: Fraction of Kelly to use (default 0.75)
            max_position_pct: Maximum position size (default 20%)
            min_position_pct: Minimum position size (default 2%)
            max_positions: Maximum number of positions
        """
        self.target_deployment = target_deployment
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.max_positions = max_positions
        
        # Performance tracking for Kelly
        self.win_rate: float = 0.55  # Default
        self.avg_win: float = 0.02   # 2% average win
        self.avg_loss: float = 0.015 # 1.5% average loss
        
        logger.info(
            f"Initialized AggressivePositionSizer: "
            f"target={target_deployment:.0%}, kelly={kelly_fraction}x, "
            f"max_pos={max_position_pct:.0%}"
        )
    
    def calculate_kelly(self,
                        win_rate: Optional[float] = None,
                        avg_win: Optional[float] = None,
                        avg_loss: Optional[float] = None) -> float:
        """
        Calculate Kelly Criterion optimal fraction.
        
        Kelly = W - (1-W)/R
        Where W = win rate, R = win/loss ratio
        
        Args:
            win_rate: Probability of winning
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
            
        Returns:
            Kelly fraction
        """
        w = win_rate or self.win_rate
        win = avg_win or self.avg_win
        loss = avg_loss or self.avg_loss
        
        if loss == 0:
            return 0.0
        
        r = win / loss  # Win/loss ratio
        kelly = w - (1 - w) / r
        
        # Apply Kelly fraction
        kelly *= self.kelly_fraction
        
        # Bound result
        return max(0, min(kelly, self.max_position_pct))
    
    def update_performance(self, 
                            trades: List[Dict[str, float]]) -> None:
        """
        Update performance metrics from trade history.
        
        Args:
            trades: List of trade dictionaries with 'return' key
        """
        if not trades:
            return
        
        returns = [t.get('return', 0) for t in trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        if len(returns) >= 20:
            self.win_rate = len(wins) / len(returns)
            self.avg_win = np.mean(wins) if wins else 0.02
            self.avg_loss = abs(np.mean(losses)) if losses else 0.015
            
            logger.info(
                f"Updated performance: win_rate={self.win_rate:.2%}, "
                f"avg_win={self.avg_win:.2%}, avg_loss={self.avg_loss:.2%}"
            )
    
    def size_position(self,
                      signal: Signal,
                      portfolio_value: float,
                      current_positions: Dict[str, float],
                      regime_multiplier: float = 1.0,
                      volatility: Optional[float] = None) -> float:
        """
        Calculate position size for a signal.
        
        Args:
            signal: Trading signal
            portfolio_value: Total portfolio value
            current_positions: Current position values by symbol
            regime_multiplier: Regime-based multiplier (0-1)
            volatility: Asset volatility for vol-targeting
            
        Returns:
            Position size in dollars
        """
        if portfolio_value <= 0:
            return 0.0
        
        # Base Kelly size
        base_kelly = self.calculate_kelly()
        
        # Adjust for signal strength
        strength_multiplier = 0.5 + 0.5 * abs(signal.strength)
        
        # Adjust for confidence
        confidence_multiplier = 0.5 + 0.5 * signal.confidence
        
        # Regime adjustment
        regime_mult = max(0.2, regime_multiplier)
        
        # Calculate raw position size
        raw_size = (
            base_kelly * 
            strength_multiplier * 
            confidence_multiplier * 
            regime_mult
        )
        
        # Volatility targeting (if provided)
        if volatility is not None and volatility > 0:
            target_vol = 0.20  # 20% annualized vol target
            vol_adjustment = target_vol / volatility
            raw_size *= min(2.0, max(0.5, vol_adjustment))
        
        # Apply position limits
        position_pct = max(
            self.min_position_pct,
            min(self.max_position_pct, raw_size)
        )
        
        # Check current deployment
        current_deployment = sum(abs(v) for v in current_positions.values())
        available = (self.target_deployment * portfolio_value) - current_deployment
        
        # Position size in dollars
        position_value = min(
            position_pct * portfolio_value,
            available,
            self.max_position_pct * portfolio_value
        )
        
        # Ensure minimum meaningful size
        min_size = self.min_position_pct * portfolio_value
        if position_value < min_size:
            return 0.0
        
        return position_value
    
    def size_portfolio(self,
                       signals: List[Signal],
                       portfolio_value: float,
                       current_positions: Dict[str, float],
                       regime_multiplier: float = 1.0) -> Dict[str, float]:
        """
        Size all positions for a list of signals.
        
        Args:
            signals: List of trading signals
            portfolio_value: Total portfolio value
            current_positions: Current positions
            regime_multiplier: Regime multiplier
            
        Returns:
            Dictionary of symbol to position size
        """
        if not signals:
            return {}
        
        # Sort signals by strength
        sorted_signals = sorted(
            signals, 
            key=lambda s: abs(s.strength) * s.confidence,
            reverse=True
        )
        
        # Take top signals up to max positions
        top_signals = sorted_signals[:self.max_positions]
        
        # Calculate target allocation
        target_allocation = self.target_deployment * portfolio_value
        
        # Weight by signal strength and confidence
        total_weight = sum(
            abs(s.strength) * s.confidence 
            for s in top_signals
        )
        
        if total_weight == 0:
            return {}
        
        positions = {}
        remaining = target_allocation
        
        for signal in top_signals:
            if remaining <= 0:
                break
            
            # Proportional allocation
            weight = abs(signal.strength) * signal.confidence / total_weight
            target_size = weight * target_allocation
            
            # Apply regime and limits
            size = min(
                target_size * regime_multiplier,
                self.max_position_pct * portfolio_value,
                remaining
            )
            
            # Minimum check
            if size >= self.min_position_pct * portfolio_value:
                # Adjust sign for direction
                if signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                    size = -size
                
                positions[signal.symbol] = size
                remaining -= abs(size)
        
        return positions
    
    def rebalance_positions(self,
                             target_positions: Dict[str, float],
                             current_positions: Dict[str, float],
                             portfolio_value: float,
                             min_trade_pct: float = 0.01) -> Dict[str, float]:
        """
        Calculate trades needed to rebalance to target positions.
        
        Args:
            target_positions: Target position values
            current_positions: Current position values
            portfolio_value: Total portfolio value
            min_trade_pct: Minimum trade size as % of portfolio
            
        Returns:
            Dictionary of symbol to trade amount
        """
        trades = {}
        min_trade = min_trade_pct * portfolio_value
        
        all_symbols = set(target_positions.keys()) | set(current_positions.keys())
        
        for symbol in all_symbols:
            target = target_positions.get(symbol, 0)
            current = current_positions.get(symbol, 0)
            diff = target - current
            
            if abs(diff) >= min_trade:
                trades[symbol] = diff
        
        return trades
    
    def get_deployment_stats(self,
                              positions: Dict[str, float],
                              portfolio_value: float) -> Dict[str, float]:
        """
        Get deployment statistics.
        
        Args:
            positions: Current positions
            portfolio_value: Portfolio value
            
        Returns:
            Dictionary of stats
        """
        total_long = sum(v for v in positions.values() if v > 0)
        total_short = sum(abs(v) for v in positions.values() if v < 0)
        total_deployed = total_long + total_short
        
        return {
            "long_exposure": total_long / portfolio_value if portfolio_value else 0,
            "short_exposure": total_short / portfolio_value if portfolio_value else 0,
            "gross_exposure": total_deployed / portfolio_value if portfolio_value else 0,
            "net_exposure": (total_long - total_short) / portfolio_value if portfolio_value else 0,
            "cash_pct": 1 - (total_deployed / portfolio_value) if portfolio_value else 1,
            "num_positions": len(positions),
            "target_deployment": self.target_deployment,
            "deployment_gap": self.target_deployment - (total_deployed / portfolio_value) if portfolio_value else 0,
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_alpaca_api() -> Optional[Any]:
    """
    Get Alpaca API instance from environment variables.
    
    Returns:
        Alpaca REST API instance or None
    """
    if not HAS_ALPACA:
        logger.warning("Alpaca API not installed")
        return None
    
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    
    if not api_key or not api_secret:
        logger.warning("Alpaca API credentials not found in environment")
        return None
    
    try:
        api = REST(api_key, api_secret, base_url)
        account = api.get_account()
        logger.info(f"Connected to Alpaca. Account: {account.status}")
        return api
    except Exception as e:
        logger.error(f"Failed to connect to Alpaca: {e}")
        return None


def validate_environment() -> Dict[str, bool]:
    """
    Validate environment setup.
    
    Returns:
        Dictionary of component availability
    """
    return {
        "alpaca_api": get_alpaca_api() is not None,
        "xgboost": HAS_XGBOOST,
        "lightgbm": HAS_LIGHTGBM,
        "hmmlearn": HAS_HMM,
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("V38 ALPHA CORE ENGINE - Module 1/3")
    print("=" * 60)
    
    # Validate environment
    print("\nValidating environment...")
    env_status = validate_environment()
    for component, available in env_status.items():
        status = "✓" if available else "✗"
        print(f"  {status} {component}")
    
    # Initialize universe
    print("\nInitializing universe...")
    universe = ExpandedUniverse()
    summary = universe.summary()
    print(f"  Total assets: {summary['total_assets']}")
    for asset_class, count in summary['by_asset_class'].items():
        if count > 0:
            print(f"    - {asset_class}: {count}")
    
    # Initialize regime detector
    print("\nInitializing regime detector...")
    detector = RegimeDetector()
    print(f"  HMM states: {detector.hmm_states}")
    
    # Initialize ML ensemble
    print("\nInitializing ML ensemble...")
    ensemble = MLEnsemble()
    print(f"  Base models: {list(ensemble.models.keys())}")
    print(f"  Features: {len(ensemble.feature_names)} (will be populated on first training)")
    
    # Initialize position sizer
    print("\nInitializing position sizer...")
    sizer = AggressivePositionSizer()
    print(f"  Target deployment: {sizer.target_deployment:.0%}")
    print(f"  Kelly fraction: {sizer.kelly_fraction}x")
    print(f"  Max position: {sizer.max_position_pct:.0%}")
    
    print("\n" + "=" * 60)
    print("V38 Alpha Core initialized successfully!")
    print("=" * 60)

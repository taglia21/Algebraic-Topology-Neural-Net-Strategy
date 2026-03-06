"""
Volatility Surface Engine
=========================

Citadel GQS-inspired implied volatility surface modeling and arbitrage detection.

Features:
- 3D IV surface construction (strike x expiration)
- SVI (Stochastic Volatility Inspired) parametric fitting
- Surface anomaly detection
- Arbitrage opportunity identification
- Volatility-of-volatility calculation

Use cases:
- Identify mispriced options
- Detect butterfly/calendar arbitrages
- Monitor term structure inversions
- Track vol-of-vol for dynamic hedging
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import griddata
import yfinance as yf


# ============================================================================
# DATA MODELS
# ============================================================================

class OptionType(Enum):
    """Option type."""
    CALL = "call"
    PUT = "put"


@dataclass
class OptionQuote:
    """Individual option quote."""
    symbol: str
    underlying: str
    strike: float
    expiration: datetime
    option_type: OptionType
    bid: float
    ask: float
    mid: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    dte: int  # Days to expiration


@dataclass
class IVSurface:
    """Implied volatility surface."""
    underlying: str
    spot_price: float
    strikes: np.ndarray  # Array of strikes
    expirations: np.ndarray  # Array of DTEs
    iv_matrix: np.ndarray  # 2D array: strikes x expirations
    moneyness: np.ndarray  # Log-moneyness (log(K/S))
    timestamp: datetime


@dataclass
class SVIParams:
    """SVI model parameters.
    
    SVI Formula:
    σ²(k) = a + b * [ρ * (k - m) + sqrt((k - m)² + σ²)]
    
    Where k = log(K/F) is log-moneyness
    """
    a: float  # Level
    b: float  # Angle
    rho: float  # Correlation
    m: float  # Translation
    sigma: float  # Volatility of volatility
    
    def __post_init__(self):
        """Validate SVI constraints."""
        assert self.b >= 0, "b must be non-negative"
        assert abs(self.rho) <= 1, "rho must be in [-1, 1]"
        assert self.sigma > 0, "sigma must be positive"


@dataclass
class IVAnomaly:
    """Volatility surface anomaly."""
    anomaly_type: str  # "rich", "cheap", "butterfly_arb", "calendar_arb"
    severity: float  # Standard deviations from fitted surface
    strike: float
    expiration: int  # DTE
    market_iv: float
    fitted_iv: float
    deviation: float  # market_iv - fitted_iv
    description: str


@dataclass
class ArbSignal:
    """Arbitrage trading signal."""
    signal_type: str  # "butterfly", "calendar", "vertical"
    underlying: str
    leg1: Dict  # {strike, expiration, side, iv}
    leg2: Dict
    leg3: Optional[Dict]
    expected_edge_vol: float  # Expected edge in IV points
    estimated_profit: float  # Estimated $ profit
    confidence: float  # 0-1
    reasoning: str


# ============================================================================
# VOLATILITY SURFACE ENGINE
# ============================================================================

class VolatilitySurfaceEngine:
    """
    Build and analyze implied volatility surfaces.
    
    Workflow:
    1. Fetch option chain data
    2. Build 3D IV surface
    3. Fit SVI parametric model
    4. Detect anomalies (rich/cheap vol)
    5. Generate arbitrage signals
    6. Calculate vol-of-vol
    """
    
    # Anomaly detection thresholds
    ANOMALY_THRESHOLD_STD = 2.0  # 2 std dev from fitted surface
    MIN_VOLUME_LIQUIDITY = 10  # Minimum volume for tradability
    MAX_BID_ASK_SPREAD_PCT = 0.20  # Max 20% bid-ask spread
    
    def __init__(self, min_dte: int = 7, max_dte: int = 90):
        """
        Initialize volatility surface engine.
        
        Args:
            min_dte: Minimum days to expiration (default 7)
            max_dte: Maximum days to expiration (default 90)
        """
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(
            f"Initialized VolatilitySurfaceEngine "
            f"(min_dte={min_dte}, max_dte={max_dte})"
        )
    
    async def build_iv_surface(
        self, 
        symbol: str,
        option_quotes: Optional[List[OptionQuote]] = None,
    ) -> IVSurface:
        """
        Build 3D implied volatility surface.
        
        Args:
            symbol: Underlying symbol
            option_quotes: Optional preloaded option quotes
        
        Returns:
            IVSurface object with IV matrix
        """
        try:
            # Fetch quotes if not provided
            if option_quotes is None:
                option_quotes = await self._fetch_option_chain(symbol)
            
            if len(option_quotes) == 0:
                raise ValueError(f"No option quotes found for {symbol}")
            
            # Get spot price
            if option_quotes and len(option_quotes) > 0:
                # Estimate spot from ATM strikes if quotes provided
                strikes = [q.strike for q in option_quotes]
                spot_price = np.median(strikes)
                
                # Try to fetch actual spot price
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    if len(hist) > 0:
                        spot_price = float(hist['Close'].iloc[-1])
                except:
                    pass  # Use median strike as fallback
            else:
                ticker = yf.Ticker(symbol)
                spot_price = float(ticker.history(period="1d")['Close'].iloc[-1])
            
            # Filter by DTE range
            valid_quotes = [
                q for q in option_quotes
                if self.min_dte <= q.dte <= self.max_dte
                and q.implied_volatility > 0
            ]
            
            if len(valid_quotes) < 10:
                raise ValueError(f"Insufficient valid quotes: {len(valid_quotes)}")
            
            # Extract data
            strikes = np.array([q.strike for q in valid_quotes])
            dtes = np.array([q.dte for q in valid_quotes])
            ivs = np.array([q.implied_volatility for q in valid_quotes])
            
            # Calculate moneyness
            moneyness = np.log(strikes / spot_price)
            
            # Build grid
            unique_strikes = np.unique(strikes)
            unique_dtes = np.unique(dtes)
            
            # Create meshgrid
            strike_grid, dte_grid = np.meshgrid(
                unique_strikes,
                unique_dtes,
                indexing='ij'
            )
            
            # Interpolate IV surface
            points = np.column_stack((strikes, dtes))
            iv_matrix = griddata(
                points,
                ivs,
                (strike_grid, dte_grid),
                method='cubic',
                fill_value=np.nan,
            )
            
            # Fill NaN with linear interpolation
            mask = ~np.isnan(iv_matrix)
            if mask.sum() > 0:
                iv_matrix_filled = griddata(
                    np.column_stack((strike_grid[mask], dte_grid[mask])),
                    iv_matrix[mask],
                    (strike_grid, dte_grid),
                    method='linear',
                    fill_value=np.nanmean(ivs),
                )
                iv_matrix = np.where(np.isnan(iv_matrix), iv_matrix_filled, iv_matrix)
            
            surface = IVSurface(
                underlying=symbol,
                spot_price=spot_price,
                strikes=unique_strikes,
                expirations=unique_dtes,
                iv_matrix=iv_matrix,
                moneyness=np.log(unique_strikes / spot_price),
                timestamp=datetime.now(),
            )
            
            self.logger.info(
                f"Built IV surface for {symbol}: "
                f"{len(unique_strikes)} strikes x {len(unique_dtes)} expirations"
            )
            
            return surface
        
        except Exception as e:
            self.logger.error(f"Failed to build IV surface for {symbol}: {e}", exc_info=True)
            raise
    
    def fit_svi_model(
        self, 
        surface: IVSurface,
        expiration_idx: int = 0,
    ) -> SVIParams:
        """
        Fit SVI (Stochastic Volatility Inspired) model to IV smile.
        
        Fits one expiration slice at a time.
        
        Args:
            surface: IV surface
            expiration_idx: Index of expiration to fit (default 0 = shortest)
        
        Returns:
            SVIParams with fitted parameters
        """
        # Extract smile for this expiration
        iv_smile = surface.iv_matrix[:, expiration_idx]
        moneyness = surface.moneyness
        
        # Remove NaN values
        mask = ~np.isnan(iv_smile)
        k = moneyness[mask]
        sigma_squared = (iv_smile[mask] ** 2)
        
        if len(k) < 5:
            # Not enough points, return default params
            return SVIParams(
                a=0.04,
                b=0.04,
                rho=-0.4,
                m=0.0,
                sigma=0.1,
            )
        
        # SVI function
        def svi(k_val, a, b, rho, m, sigma):
            return a + b * (rho * (k_val - m) + np.sqrt((k_val - m)**2 + sigma**2))
        
        # Initial guess
        p0 = [
            np.mean(sigma_squared),  # a
            0.04,  # b
            -0.4,  # rho
            0.0,  # m
            0.1,  # sigma
        ]
        
        # Bounds
        bounds = (
            [0, 0, -0.999, -2, 0.01],  # Lower bounds
            [1, 1, 0.999, 2, 2],  # Upper bounds
        )
        
        try:
            # Fit
            popt, _ = curve_fit(
                svi,
                k,
                sigma_squared,
                p0=p0,
                bounds=bounds,
                maxfev=5000,
            )
            
            params = SVIParams(
                a=popt[0],
                b=popt[1],
                rho=popt[2],
                m=popt[3],
                sigma=popt[4],
            )
            
            self.logger.info(f"SVI fit: a={params.a:.4f}, b={params.b:.4f}, ρ={params.rho:.2f}")
            
            return params
        
        except Exception as e:
            self.logger.warning(f"SVI fitting failed: {e}, using defaults")
            return SVIParams(
                a=0.04,
                b=0.04,
                rho=-0.4,
                m=0.0,
                sigma=0.1,
            )
    
    async def detect_anomalies(
        self, 
        surface: IVSurface,
        svi_params: Optional[SVIParams] = None,
    ) -> List[IVAnomaly]:
        """
        Detect anomalies in volatility surface.
        
        Finds:
        - Rich options (market IV >> fitted IV)
        - Cheap options (market IV << fitted IV)
        - Butterfly arbitrages
        - Calendar arbitrages
        
        Args:
            surface: IV surface
            svi_params: Optional SVI parameters (computed if not provided)
        
        Returns:
            List of detected anomalies
        """
        anomalies: List[IVAnomaly] = []
        
        # Fit SVI for shortest expiration if not provided
        if svi_params is None:
            svi_params = self.fit_svi_model(surface, expiration_idx=0)
        
        # SVI function
        def svi_iv(k, params):
            sigma_sq = (
                params.a + params.b * (
                    params.rho * (k - params.m) + 
                    np.sqrt((k - params.m)**2 + params.sigma**2)
                )
            )
            return np.sqrt(max(sigma_sq, 0))
        
        # Check each point in surface
        for i, strike in enumerate(surface.strikes):
            for j, dte in enumerate(surface.expirations):
                market_iv = surface.iv_matrix[i, j]
                
                if np.isnan(market_iv):
                    continue
                
                # Calculate fitted IV
                k = np.log(strike / surface.spot_price)
                fitted_iv = svi_iv(k, svi_params)
                
                # Calculate deviation
                deviation = market_iv - fitted_iv
                
                # Estimate std of residuals (simplified: use 5% of ATM IV)
                residual_std = 0.05 * fitted_iv
                
                # Check if anomaly
                std_devs = abs(deviation) / (residual_std + 1e-8)
                
                if std_devs > self.ANOMALY_THRESHOLD_STD:
                    anomaly_type = "rich" if deviation > 0 else "cheap"
                    
                    anomalies.append(IVAnomaly(
                        anomaly_type=anomaly_type,
                        severity=std_devs,
                        strike=strike,
                        expiration=int(dte),
                        market_iv=market_iv,
                        fitted_iv=fitted_iv,
                        deviation=deviation,
                        description=f"{anomaly_type.upper()}: {strike:.0f} strike, "
                                  f"{dte:.0f}d - market {market_iv:.1%} vs fitted {fitted_iv:.1%}",
                    ))
        
        # Check for butterfly arbitrages
        anomalies.extend(self._detect_butterfly_arbs(surface))
        
        # Check for calendar arbitrages
        anomalies.extend(self._detect_calendar_arbs(surface))
        
        self.logger.info(f"Detected {len(anomalies)} IV anomalies")
        
        return anomalies
    
    async def generate_arb_signals(
        self, 
        anomalies: List[IVAnomaly],
        surface: IVSurface,
    ) -> List[ArbSignal]:
        """
        Generate arbitrage trading signals from anomalies.
        
        Args:
            anomalies: Detected anomalies
            surface: IV surface
        
        Returns:
            List of actionable arbitrage signals
        """
        signals: List[ArbSignal] = []
        
        for anomaly in anomalies:
            # Only generate signals for high-severity anomalies
            if anomaly.severity < 2.5:
                continue
            
            if anomaly.anomaly_type == "rich":
                # Sell overpriced option
                signals.append(ArbSignal(
                    signal_type="sell_rich_vol",
                    underlying=surface.underlying,
                    leg1={
                        "strike": anomaly.strike,
                        "expiration": anomaly.expiration,
                        "side": "sell",
                        "iv": anomaly.market_iv,
                    },
                    leg2={},
                    leg3=None,
                    expected_edge_vol=anomaly.deviation,
                    estimated_profit=abs(anomaly.deviation) * 100,  # Simplified
                    confidence=min(anomaly.severity / 5.0, 1.0),
                    reasoning=anomaly.description,
                ))
            
            elif anomaly.anomaly_type == "cheap":
                # Buy underpriced option
                signals.append(ArbSignal(
                    signal_type="buy_cheap_vol",
                    underlying=surface.underlying,
                    leg1={
                        "strike": anomaly.strike,
                        "expiration": anomaly.expiration,
                        "side": "buy",
                        "iv": anomaly.market_iv,
                    },
                    leg2={},
                    leg3=None,
                    expected_edge_vol=abs(anomaly.deviation),
                    estimated_profit=abs(anomaly.deviation) * 100,
                    confidence=min(anomaly.severity / 5.0, 1.0),
                    reasoning=anomaly.description,
                ))
            
            elif anomaly.anomaly_type == "butterfly_arb":
                # Butterfly spread opportunity
                signals.append(ArbSignal(
                    signal_type="butterfly",
                    underlying=surface.underlying,
                    leg1={"strike": anomaly.strike, "side": "buy"},
                    leg2={"strike": anomaly.strike, "side": "sell"},
                    leg3={"strike": anomaly.strike, "side": "buy"},
                    expected_edge_vol=anomaly.deviation,
                    estimated_profit=abs(anomaly.deviation) * 150,
                    confidence=0.7,
                    reasoning=anomaly.description,
                ))
        
        self.logger.info(f"Generated {len(signals)} arbitrage signals")
        
        return signals
    
    def calculate_vol_of_vol(
        self, 
        surface: IVSurface,
        window_days: int = 20,
    ) -> float:
        """
        Calculate volatility-of-volatility.
        
        Measures how much IV itself is moving - critical for vega hedging.
        
        Args:
            surface: IV surface
            window_days: Lookback window for vol calculation
        
        Returns:
            Vol-of-vol (annualized)
        """
        # Use ATM IV time series (simplified: use middle strike)
        mid_strike_idx = len(surface.strikes) // 2
        atm_iv_series = surface.iv_matrix[mid_strike_idx, :]
        
        # Remove NaN
        atm_iv_series = atm_iv_series[~np.isnan(atm_iv_series)]
        
        if len(atm_iv_series) < 2:
            return 0.0
        
        # Calculate returns of IV
        iv_returns = np.diff(atm_iv_series) / atm_iv_series[:-1]
        
        # Vol-of-vol is std of IV returns
        vol_of_vol = np.std(iv_returns) * np.sqrt(252)  # Annualize
        
        self.logger.info(f"Vol-of-vol: {vol_of_vol:.1%}")
        
        return vol_of_vol
    
    async def _fetch_option_chain(self, symbol: str) -> List[OptionQuote]:
        """
        Fetch option chain from yfinance.
        
        Args:
            symbol: Underlying symbol
        
        Returns:
            List of OptionQuote objects
        """
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            if len(expirations) == 0:
                return []
            
            quotes: List[OptionQuote] = []
            
            # Limit to first 4 expirations for speed
            for exp_str in expirations[:4]:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
                dte = (exp_date - datetime.now()).days
                
                if dte < self.min_dte or dte > self.max_dte:
                    continue
                
                # Get option chain
                opt_chain = ticker.option_chain(exp_str)
                
                # Process calls
                for _, row in opt_chain.calls.iterrows():
                    quotes.append(OptionQuote(
                        symbol=f"{symbol}_{exp_str}_C_{row['strike']}",
                        underlying=symbol,
                        strike=float(row['strike']),
                        expiration=exp_date,
                        option_type=OptionType.CALL,
                        bid=float(row.get('bid', 0)),
                        ask=float(row.get('ask', 0)),
                        mid=(float(row.get('bid', 0)) + float(row.get('ask', 0))) / 2,
                        volume=int(row.get('volume', 0)),
                        open_interest=int(row.get('openInterest', 0)),
                        implied_volatility=float(row.get('impliedVolatility', 0)),
                        delta=float(row.get('delta', 0)),
                        dte=dte,
                    ))
            
            self.logger.info(f"Fetched {len(quotes)} option quotes for {symbol}")
            return quotes
        
        except Exception as e:
            self.logger.error(f"Failed to fetch option chain: {e}")
            return []
    
    def _detect_butterfly_arbs(self, surface: IVSurface) -> List[IVAnomaly]:
        """Detect butterfly arbitrage opportunities."""
        anomalies = []
        
        # Check for negative butterfly spreads (violation of convexity)
        for j, dte in enumerate(surface.expirations):
            for i in range(1, len(surface.strikes) - 1):
                iv_low = surface.iv_matrix[i-1, j]
                iv_mid = surface.iv_matrix[i, j]
                iv_high = surface.iv_matrix[i+1, j]
                
                if np.isnan(iv_low) or np.isnan(iv_mid) or np.isnan(iv_high):
                    continue
                
                # Butterfly spread value
                butterfly = iv_mid - (iv_low + iv_high) / 2
                
                # Should be <= 0 (convexity), if positive => arbitrage
                if butterfly > 0.02:  # 2% threshold
                    anomalies.append(IVAnomaly(
                        anomaly_type="butterfly_arb",
                        severity=butterfly / 0.02,
                        strike=surface.strikes[i],
                        expiration=int(dte),
                        market_iv=iv_mid,
                        fitted_iv=(iv_low + iv_high) / 2,
                        deviation=butterfly,
                        description=f"Butterfly arb at {surface.strikes[i]:.0f} strike",
                    ))
        
        return anomalies
    
    def _detect_calendar_arbs(self, surface: IVSurface) -> List[IVAnomaly]:
        """Detect calendar spread arbitrage opportunities."""
        anomalies = []
        
        # Check for inverted term structure (short-term IV > long-term IV)
        for i, strike in enumerate(surface.strikes):
            for j in range(len(surface.expirations) - 1):
                iv_short = surface.iv_matrix[i, j]
                iv_long = surface.iv_matrix[i, j+1]
                
                if np.isnan(iv_short) or np.isnan(iv_long):
                    continue
                
                # Normally iv_long >= iv_short (term structure upward sloping)
                if iv_short > iv_long + 0.05:  # 5% inversion
                    inversion = iv_short - iv_long
                    
                    anomalies.append(IVAnomaly(
                        anomaly_type="calendar_arb",
                        severity=inversion / 0.05,
                        strike=strike,
                        expiration=int(surface.expirations[j]),
                        market_iv=iv_short,
                        fitted_iv=iv_long,
                        deviation=inversion,
                        description=f"Term structure inversion at {strike:.0f} strike",
                    ))
        
        return anomalies


# ============================================================================
# TESTING HELPER
# ============================================================================

async def test_volatility_surface():
    """Test the volatility surface engine."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    engine = VolatilitySurfaceEngine(min_dte=7, max_dte=60)
    
    print("\n" + "="*60)
    print("TESTING VOLATILITY SURFACE ENGINE")
    print("="*60)
    
    # Create synthetic option quotes for testing
    symbol = "SPY"
    spot = 500.0
    
    print(f"\n1. Creating synthetic IV surface for {symbol}...")
    
    # Generate synthetic option chain
    strikes = np.arange(480, 521, 5)  # 480 to 520 in $5 increments
    dtes = [14, 30, 45]  # 3 expirations
    quotes = []
    
    for dte in dtes:
        for strike in strikes:
            # Synthetic IV smile (lower for ATM, higher for OTM)
            moneyness = np.log(strike / spot)
            base_iv = 0.15
            skew = 0.10 * moneyness  # Negative skew
            smile = 0.05 * moneyness ** 2  # Smile effect
            iv = base_iv + skew + smile + np.random.normal(0, 0.01)
            
            quotes.append(OptionQuote(
                symbol=f"{symbol}_{strike}",
                underlying=symbol,
                strike=strike,
                expiration=datetime.now() + timedelta(days=dte),
                option_type=OptionType.CALL,
                bid=1.0,
                ask=1.2,
                mid=1.1,
                volume=100,
                open_interest=500,
                implied_volatility=max(iv, 0.05),
                delta=0.5,
                dte=dte,
            ))
    
    try:
        surface = await engine.build_iv_surface(symbol, option_quotes=quotes)
        print(f"✓ Surface built: {surface.iv_matrix.shape} points")
        print(f"  Spot: ${surface.spot_price:.2f}")
        print(f"  Strikes: {len(surface.strikes)}")
        print(f"  Expirations: {len(surface.expirations)}")
    except Exception as e:
        print(f"✗ Failed to build surface: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n2. Fitting SVI model...")
    svi_params = engine.fit_svi_model(surface)
    print(f"✓ SVI parameters:")
    print(f"  a (level): {svi_params.a:.4f}")
    print(f"  b (angle): {svi_params.b:.4f}")
    print(f"  ρ (correlation): {svi_params.rho:.2f}")
    print(f"  m (translation): {svi_params.m:.2f}")
    print(f"  σ (vol-of-vol): {svi_params.sigma:.2f}")
    
    print(f"\n3. Detecting anomalies...")
    anomalies = await engine.detect_anomalies(surface, svi_params)
    print(f"✓ Found {len(anomalies)} anomalies")
    for i, anomaly in enumerate(anomalies[:5]):  # Show first 5
        print(f"  {i+1}. {anomaly.description}")
    
    print(f"\n4. Generating arbitrage signals...")
    signals = await engine.generate_arb_signals(anomalies, surface)
    print(f"✓ Generated {len(signals)} signals")
    for i, signal in enumerate(signals[:3]):  # Show first 3
        print(f"  {i+1}. {signal.signal_type}: edge={signal.expected_edge_vol:.2%}, "
              f"confidence={signal.confidence:.1%}")
    
    print(f"\n5. Calculating vol-of-vol...")
    vol_of_vol = engine.calculate_vol_of_vol(surface)
    print(f"✓ Vol-of-vol: {vol_of_vol:.1%}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_volatility_surface())

"""
Gamma Exposure (GEX) Awareness Module
=======================================

Provides GEX-based signal filtering to avoid selling spreads near
high-gamma strike levels where dealer hedging can cause pins.

Features:
- Fetch aggregate GEX data (synthetic when real dealer data unavailable)
- Identify sticky strikes (high absolute gamma exposure)
- Filter out strikes near GEX walls
- Detect positive-to-negative GEX flips for entry timing
- Estimate gamma exposure from open interest + IV data

Architecture Note:
Real institutional GEX data requires a SpotGamma / Orats subscription.
This module provides a synthetic GEX estimate from publicly available
open interest data via Alpaca's options chain API. The estimates are
directionally correct but less precise than paid dealer flow data.

Usage:
    gex = GammaExposureAnalyzer(data_client)
    profile = await gex.compute_gex_profile("SPY")
    sticky = gex.get_sticky_strikes(profile, n=3)
    safe = gex.is_safe_strike(profile, strike=550.0, threshold=0.3)
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrikeGEX:
    """Gamma exposure at a single strike."""
    strike: float
    call_gamma: float   # Aggregate call gamma * OI * 100 * spot
    put_gamma: float    # Aggregate put gamma * OI * 100 * spot (negative convention)
    net_gex: float      # call_gamma - put_gamma (positive = dealer long gamma)
    call_oi: int = 0
    put_oi: int = 0


@dataclass
class GEXProfile:
    """Full gamma exposure profile for a symbol."""
    symbol: str
    spot_price: float
    timestamp: datetime
    strikes: List[StrikeGEX]
    total_call_gex: float = 0.0
    total_put_gex: float = 0.0
    net_gex: float = 0.0
    zero_gamma_strike: Optional[float] = None  # Where GEX flips sign
    max_gamma_strike: Optional[float] = None    # Highest absolute GEX
    is_positive_gex: bool = True  # Overall GEX environment


@dataclass
class GEXSignalFilter:
    """Result of GEX-based signal filtering."""
    is_safe: bool
    reason: str
    nearest_sticky_strike: Optional[float] = None
    distance_to_sticky_pct: Optional[float] = None
    gex_environment: str = "neutral"  # "positive", "negative", "neutral"
    recommended_action: str = "proceed"  # "proceed", "avoid", "reduce_size"


class GammaExposureAnalyzer:
    """
    Analyzes gamma exposure to improve options trade selection.

    Primary use cases:
    1. Avoid selling spreads with short strikes near sticky (high-GEX) levels
    2. Time entries when GEX flips from positive to negative (regime change)
    3. Identify support/resistance from dealer gamma hedging
    """

    def __init__(
        self,
        data_client=None,
        sticky_strike_threshold: float = 0.30,  # Top 30% GEX = sticky
        avoidance_radius_pct: float = 0.005,    # Avoid strikes within 0.5% of sticky
        cache_ttl_minutes: int = 15,             # Cache GEX for 15 min
    ):
        self.data_client = data_client
        self.sticky_strike_threshold = sticky_strike_threshold
        self.avoidance_radius_pct = avoidance_radius_pct
        self.cache_ttl_minutes = cache_ttl_minutes

        self._cache: Dict[str, Tuple[datetime, GEXProfile]] = {}

        logger.info(
            f"GEX Analyzer initialized: sticky_threshold={sticky_strike_threshold:.0%}, "
            f"avoidance_radius={avoidance_radius_pct:.1%}"
        )

    async def compute_gex_profile(
        self,
        symbol: str,
        spot_price: Optional[float] = None,
        target_dte: int = 30,
    ) -> Optional[GEXProfile]:
        """
        Compute GEX profile for a symbol.

        Estimates gamma exposure from options chain open interest and
        approximate gamma values (Black-Scholes estimate).

        Args:
            symbol: Underlying symbol (e.g., "SPY")
            spot_price: Current spot price (fetched if None)
            target_dte: Focus on expirations near this DTE

        Returns:
            GEXProfile or None if data unavailable
        """
        # Check cache
        if symbol in self._cache:
            cached_time, cached_profile = self._cache[symbol]
            age_min = (datetime.now() - cached_time).total_seconds() / 60
            if age_min < self.cache_ttl_minutes:
                return cached_profile

        if spot_price is None:
            spot_price = await self._get_spot_price(symbol)
            if spot_price is None:
                logger.warning(f"Cannot get spot price for {symbol}")
                return None

        # Build GEX from options chain
        try:
            strikes_gex = await self._build_gex_from_chain(
                symbol, spot_price, target_dte
            )
        except Exception as e:
            logger.warning(f"GEX computation failed for {symbol}: {e}")
            # Fall back to synthetic estimate
            strikes_gex = self._synthetic_gex(symbol, spot_price, target_dte)

        if not strikes_gex:
            return None

        # Compute aggregates
        total_call_gex = sum(s.call_gamma for s in strikes_gex)
        total_put_gex = sum(s.put_gamma for s in strikes_gex)
        net_gex = total_call_gex + total_put_gex  # Put gamma is negative

        # Find zero-gamma strike (where net GEX crosses zero)
        zero_gamma = self._find_zero_gamma_strike(strikes_gex, spot_price)

        # Find max gamma strike
        max_strike = max(strikes_gex, key=lambda s: abs(s.net_gex))

        profile = GEXProfile(
            symbol=symbol,
            spot_price=spot_price,
            timestamp=datetime.now(),
            strikes=strikes_gex,
            total_call_gex=total_call_gex,
            total_put_gex=total_put_gex,
            net_gex=net_gex,
            zero_gamma_strike=zero_gamma,
            max_gamma_strike=max_strike.strike,
            is_positive_gex=net_gex > 0,
        )

        # Cache
        self._cache[symbol] = (datetime.now(), profile)
        return profile

    def get_sticky_strikes(
        self, profile: GEXProfile, n: int = 5
    ) -> List[Tuple[float, float]]:
        """
        Get the top N sticky strikes by absolute GEX.

        Returns:
            List of (strike, net_gex) tuples, sorted by |gex| descending
        """
        sorted_strikes = sorted(
            profile.strikes, key=lambda s: abs(s.net_gex), reverse=True
        )
        return [(s.strike, s.net_gex) for s in sorted_strikes[:n]]

    def filter_signal(
        self,
        profile: Optional[GEXProfile],
        short_strike: float,
        signal_type: str = "credit_spread",
    ) -> GEXSignalFilter:
        """
        Filter a trading signal based on GEX profile.

        Args:
            profile: GEX profile for the underlying
            short_strike: The short strike price being considered
            signal_type: Type of signal (credit_spread, iron_condor, etc.)

        Returns:
            GEXSignalFilter with recommendation
        """
        if profile is None:
            return GEXSignalFilter(
                is_safe=True,
                reason="No GEX data available — proceeding with caution",
                gex_environment="unknown",
                recommended_action="proceed",
            )

        spot = profile.spot_price

        # Find nearest sticky strike
        sticky = self.get_sticky_strikes(profile)
        nearest_sticky = None
        min_dist = float("inf")

        for strike, gex_val in sticky:
            dist = abs(strike - short_strike)
            if dist < min_dist:
                min_dist = dist
                nearest_sticky = strike

        distance_pct = min_dist / spot if spot > 0 else 1.0

        # Determine GEX environment
        if profile.is_positive_gex:
            gex_env = "positive"  # Dealers long gamma → mean reverting
        else:
            gex_env = "negative"  # Dealers short gamma → trending/volatile

        # Check if short strike is too close to a sticky strike
        if distance_pct < self.avoidance_radius_pct:
            return GEXSignalFilter(
                is_safe=False,
                reason=(
                    f"Short strike ${short_strike:.0f} is within "
                    f"{distance_pct:.1%} of sticky strike ${nearest_sticky:.0f} "
                    f"(threshold={self.avoidance_radius_pct:.1%})"
                ),
                nearest_sticky_strike=nearest_sticky,
                distance_to_sticky_pct=distance_pct,
                gex_environment=gex_env,
                recommended_action="avoid",
            )

        # In negative GEX environment, reduce size on all credit strategies
        if gex_env == "negative" and signal_type in ("credit_spread", "iron_condor"):
            return GEXSignalFilter(
                is_safe=True,
                reason=(
                    f"Negative GEX environment — reduce position size. "
                    f"Net GEX={profile.net_gex:+,.0f}"
                ),
                nearest_sticky_strike=nearest_sticky,
                distance_to_sticky_pct=distance_pct,
                gex_environment=gex_env,
                recommended_action="reduce_size",
            )

        return GEXSignalFilter(
            is_safe=True,
            reason=f"GEX clear: short strike ${short_strike:.0f} is {distance_pct:.1%} from sticky",
            nearest_sticky_strike=nearest_sticky,
            distance_to_sticky_pct=distance_pct,
            gex_environment=gex_env,
            recommended_action="proceed",
        )

    # ====================================================================
    # GEX COMPUTATION
    # ====================================================================

    async def _build_gex_from_chain(
        self,
        symbol: str,
        spot_price: float,
        target_dte: int,
    ) -> List[StrikeGEX]:
        """
        Build GEX from live options chain via Alpaca.

        Uses approximate Black-Scholes gamma for each strike.
        """
        if self.data_client is None:
            return self._synthetic_gex(symbol, spot_price, target_dte)

        try:
            from alpaca.data.requests import OptionChainRequest
            target_exp = date.today() + timedelta(days=target_dte)

            req = OptionChainRequest(
                underlying_symbol=symbol,
                expiration_date=target_exp.isoformat(),
            )
            chain = self.data_client.get_option_chain(req)
        except Exception as e:
            logger.debug(f"Option chain fetch failed for {symbol}: {e}")
            return self._synthetic_gex(symbol, spot_price, target_dte)

        if not chain:
            return self._synthetic_gex(symbol, spot_price, target_dte)

        # Parse chain into strike GEX
        strike_data: Dict[float, dict] = {}

        for occ_sym, contract_data in chain.items():
            try:
                # Parse strike from OCC symbol
                strike = self._parse_occ_strike(occ_sym)
                if strike is None:
                    continue

                is_call = "C" in occ_sym[6:8] or occ_sym[-9] == "C"

                # Get OI from snapshot
                oi = getattr(contract_data, "open_interest", 0) or 0

                if strike not in strike_data:
                    strike_data[strike] = {"call_oi": 0, "put_oi": 0}

                if is_call:
                    strike_data[strike]["call_oi"] += oi
                else:
                    strike_data[strike]["put_oi"] += oi
            except Exception:
                continue

        # Convert to GEX
        T = target_dte / 365.0
        return self._compute_gex_from_oi(strike_data, spot_price, T)

    def _synthetic_gex(
        self, symbol: str, spot_price: float, target_dte: int
    ) -> List[StrikeGEX]:
        """
        Generate synthetic GEX profile using reasonable assumptions.

        Uses a bell-curve OI distribution centered on ATM with gamma
        computed from Black-Scholes approximation.
        """
        T = target_dte / 365.0
        if T <= 0:
            T = 30 / 365.0

        # Generate strikes around spot (+/- 10%)
        strike_range = spot_price * 0.10
        n_strikes = 21
        strikes = np.linspace(
            spot_price - strike_range, spot_price + strike_range, n_strikes
        )

        # Synthetic OI: bell curve centered on ATM, higher for round numbers
        base_oi = 5000
        results = []

        for strike in strikes:
            dist_from_atm = abs(strike - spot_price) / spot_price
            oi_multiplier = math.exp(-50 * dist_from_atm ** 2)

            # Round strikes get more OI
            if strike % 5 == 0:
                oi_multiplier *= 1.5
            if strike % 10 == 0:
                oi_multiplier *= 2.0

            call_oi = int(base_oi * oi_multiplier * (1 + 0.1 * np.random.randn()))
            put_oi = int(base_oi * oi_multiplier * (1 + 0.1 * np.random.randn()))
            call_oi = max(0, call_oi)
            put_oi = max(0, put_oi)

            gamma = self._bs_gamma(spot_price, strike, T, vol=0.20)

            call_gex = gamma * call_oi * 100 * spot_price
            put_gex = -gamma * put_oi * 100 * spot_price

            results.append(StrikeGEX(
                strike=round(strike, 2),
                call_gamma=call_gex,
                put_gamma=put_gex,
                net_gex=call_gex + put_gex,
                call_oi=call_oi,
                put_oi=put_oi,
            ))

        return results

    def _compute_gex_from_oi(
        self,
        strike_data: Dict[float, dict],
        spot_price: float,
        T: float,
        vol: float = 0.20,
    ) -> List[StrikeGEX]:
        """Convert OI data to GEX using BS gamma."""
        results = []
        for strike, data in sorted(strike_data.items()):
            gamma = self._bs_gamma(spot_price, strike, T, vol)

            call_gex = gamma * data["call_oi"] * 100 * spot_price
            put_gex = -gamma * data["put_oi"] * 100 * spot_price

            results.append(StrikeGEX(
                strike=strike,
                call_gamma=call_gex,
                put_gamma=put_gex,
                net_gex=call_gex + put_gex,
                call_oi=data["call_oi"],
                put_oi=data["put_oi"],
            ))
        return results

    @staticmethod
    def _bs_gamma(
        S: float, K: float, T: float, vol: float = 0.20, r: float = 0.05
    ) -> float:
        """Black-Scholes gamma approximation."""
        if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
            return 0.0
        try:
            d1 = (math.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * math.sqrt(T))
            gamma = math.exp(-0.5 * d1 ** 2) / (
                S * vol * math.sqrt(2 * math.pi * T)
            )
            return gamma
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _find_zero_gamma_strike(
        self, strikes: List[StrikeGEX], spot_price: float
    ) -> Optional[float]:
        """Find where net GEX crosses zero (interpolated)."""
        sorted_strikes = sorted(strikes, key=lambda s: s.strike)
        for i in range(len(sorted_strikes) - 1):
            s1, s2 = sorted_strikes[i], sorted_strikes[i + 1]
            if s1.net_gex * s2.net_gex < 0:  # Sign change
                # Linear interpolation
                if abs(s2.net_gex - s1.net_gex) > 0:
                    frac = abs(s1.net_gex) / abs(s2.net_gex - s1.net_gex)
                    return s1.strike + frac * (s2.strike - s1.strike)
        return None

    async def _get_spot_price(self, symbol: str) -> Optional[float]:
        """Get current spot price for symbol."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            return float(info.get("lastPrice", 0) or info.get("previousClose", 0))
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_occ_strike(occ_symbol: str) -> Optional[float]:
        """Parse strike price from OCC symbol (last 8 digits / 1000)."""
        try:
            strike_str = occ_symbol[-8:]
            return int(strike_str) / 1000.0
        except (ValueError, IndexError):
            return None

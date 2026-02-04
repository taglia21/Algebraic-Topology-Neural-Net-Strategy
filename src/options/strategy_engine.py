"""
Strategy Engine
===============

Premium selling strategies for options trading.

Strategies:
1. Wheel Strategy: Sell cash-secured puts, convert to covered calls if assigned
2. Credit Spreads: Bull put spreads and bear call spreads
3. Iron Condors: Delta-neutral range-bound strategy

All strategies optimized for theta decay and high win rates.
"""

import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .utils.black_scholes import BlackScholes, OptionType, Greeks
from .utils.constants import (
    WHEEL_DTE_RANGE,
    SPREAD_DTE_RANGE,
    IRON_CONDOR_DTE_RANGE,
    WHEEL_DELTA_TARGET,
    SPREAD_DELTA_TARGET,
    IRON_CONDOR_WING_DELTA,
    PROFIT_TARGET_PCT,
    STOP_LOSS_MULTIPLIER,
    THETA_ACCELERATION_DTE,
)
from .theta_decay_engine import ThetaDecayEngine, IVRegime, TrendDirection
from .iv_analyzer import IVAnalyzer

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Available trading strategies."""
    WHEEL = "wheel"
    CREDIT_SPREAD = "credit_spread"
    IRON_CONDOR = "iron_condor"


class SpreadType(Enum):
    """Type of credit spread."""
    BULL_PUT = "bull_put"  # Bullish: sell put spread below market
    BEAR_CALL = "bear_call"  # Bearish: sell call spread above market


@dataclass
class OptionCandidate:
    """Potential option to trade."""
    symbol: str
    underlying_price: float
    strike: float
    expiration: datetime
    dte: int
    option_type: OptionType
    
    bid: float
    ask: float
    mid: float
    
    iv: float
    greeks: Greeks
    
    delta: float
    theta: float
    vega: float
    
    score: float  # Overall attractiveness (0-100)
    reasoning: str
    
    def __str__(self) -> str:
        return (
            f"{self.symbol} ${self.strike} {self.option_type.value} "
            f"DTE={self.dte} IV={self.iv:.1%} Δ={self.delta:.2f} "
            f"θ={self.theta:.2f} ${self.mid:.2f} Score={self.score:.0f}"
        )


@dataclass
class SpreadCandidate:
    """Potential spread to trade."""
    symbol: str
    underlying_price: float
    spread_type: SpreadType
    
    short_strike: float
    long_strike: float
    expiration: datetime
    dte: int
    
    net_credit: float
    max_profit: float
    max_loss: float
    breakeven: float
    
    pop: float  # Probability of profit (%)
    expected_value: float
    
    short_greeks: Greeks
    long_greeks: Greeks
    net_greeks: Greeks
    
    score: float
    reasoning: str
    
    def __str__(self) -> str:
        return (
            f"{self.symbol} {self.spread_type.value.upper()} "
            f"${self.short_strike}/{self.long_strike} DTE={self.dte} "
            f"Credit=${self.net_credit:.2f} PoP={self.pop:.0f}% "
            f"Score={self.score:.0f}"
        )


@dataclass
class IronCondorCandidate:
    """Potential iron condor to trade."""
    symbol: str
    underlying_price: float
    expiration: datetime
    dte: int
    
    # Put side
    put_short_strike: float
    put_long_strike: float
    put_credit: float
    
    # Call side
    call_short_strike: float
    call_long_strike: float
    call_credit: float
    
    # Combined metrics
    total_credit: float
    max_profit: float
    max_loss: float
    breakeven_lower: float
    breakeven_upper: float
    profit_range: float
    
    pop: float
    expected_value: float
    
    net_greeks: Greeks
    
    score: float
    reasoning: str
    
    def __str__(self) -> str:
        return (
            f"{self.symbol} IC ${self.put_short_strike}/{self.call_short_strike} "
            f"(±${self.profit_range:.0f} range) DTE={self.dte} "
            f"Credit=${self.total_credit:.2f} PoP={self.pop:.0f}% "
            f"Score={self.score:.0f}"
        )


class StrategyEngine:
    """
    Options Strategy Engine.
    
    Generates trade candidates for premium selling strategies.
    
    Usage:
        engine = StrategyEngine()
        candidates = engine.find_wheel_candidates('SPY', 450, ...)
        spreads = engine.find_credit_spread_candidates('SPY', 450, ...)
        condors = engine.find_iron_condor_candidates('SPY', 450, ...)
    """
    
    def __init__(
        self,
        theta_engine: Optional[ThetaDecayEngine] = None,
        iv_analyzer: Optional[IVAnalyzer] = None,
        risk_free_rate: float = 0.05
    ):
        """
        Initialize strategy engine.
        
        Args:
            theta_engine: Theta decay analyzer (creates if None)
            iv_analyzer: IV analyzer (creates if None)
            risk_free_rate: Risk-free interest rate
        """
        self.theta_engine = theta_engine or ThetaDecayEngine()
        self.iv_analyzer = iv_analyzer or IVAnalyzer()
        self.risk_free_rate = risk_free_rate
        
        self.bs = BlackScholes(risk_free_rate=risk_free_rate)
        
        logger.info("Strategy Engine initialized")
    
    # ============ WHEEL STRATEGY ============
    
    def find_wheel_candidates(
        self,
        symbol: str,
        underlying_price: float,
        current_iv: float,
        historical_vol: Optional[float] = None,
        iv_rank: Optional[float] = None,
        trend: TrendDirection = TrendDirection.NEUTRAL,
        chain_data: Optional[List[Dict]] = None,
        top_n: int = 5
    ) -> List[OptionCandidate]:
        """
        Find best cash-secured put candidates for Wheel strategy.
        
        Wheel Strategy:
        - Sell cash-secured puts at 0.25-0.30 delta
        - Target 30-45 DTE
        - Collect premium, buy stock if assigned
        - Sell covered calls if assigned
        
        Args:
            symbol: Underlying symbol
            underlying_price: Current stock price
            current_iv: Current IV
            historical_vol: Historical volatility (optional)
            iv_rank: IV rank (will calculate if None)
            trend: Market trend
            chain_data: Option chain data (will generate if None)
            top_n: Number of candidates to return
            
        Returns:
            List of top OptionCandidates for CSP entry
        """
        # Analyze IV environment
        if iv_rank is None:
            metrics = self.iv_analyzer.analyze(symbol, current_iv, historical_vol)
            iv_rank = metrics.iv_rank
        
        # Check if IV is favorable for selling
        should_sell, reason = self.iv_analyzer.should_sell_premium(symbol, current_iv)
        if not should_sell:
            logger.warning(f"Wheel: Unfavorable IV environment - {reason}")
            return []
        
        # Get optimal DTE recommendation
        dte_rec = self.theta_engine.calculate_optimal_dte(
            iv_rank=iv_rank,
            trend=trend,
            volatility_regime=IVRegime.NORMAL,  # Will be classified by theta engine
            strategy_type=StrategyType.WHEEL.value
        )
        
        # Generate option candidates if not provided
        if chain_data is None:
            chain_data = self._generate_option_chain(
                underlying_price=underlying_price,
                current_iv=current_iv,
                dte_range=(dte_rec.entry_dte_min, dte_rec.entry_dte_max),
                option_type=OptionType.PUT
            )
        
        candidates = []
        
        for opt in chain_data:
            strike = opt['strike']
            dte = opt['dte']
            expiration = opt['expiration']
            
            # Calculate option price and Greeks
            time_to_expiration = dte / 365.0
            price = self.bs.put_price(
                S=underlying_price,
                K=strike,
                T=time_to_expiration,
                sigma=current_iv
            )
            
            greeks = self.bs.calculate_all_greeks(
                S=underlying_price,
                K=strike,
                T=time_to_expiration,
                sigma=current_iv,
                option_type=OptionType.PUT
            )
            
            delta = abs(greeks.delta)  # Put delta is negative
            
            # Filter: target 0.25-0.30 delta
            target_delta = WHEEL_DELTA_TARGET
            if not (target_delta - 0.05 <= delta <= target_delta + 0.05):
                continue
            
            # Score the candidate
            score = self._score_wheel_candidate(
                delta=delta,
                theta=greeks.theta,
                vega=greeks.vega,
                premium=price,
                iv_rank=iv_rank,
                dte=dte,
                target_dte=dte_rec.entry_dte_min
            )
            
            # Build candidate
            candidate = OptionCandidate(
                symbol=symbol,
                underlying_price=underlying_price,
                strike=strike,
                expiration=expiration,
                dte=dte,
                option_type=OptionType.PUT,
                bid=price * 0.98,  # Approximate bid/ask
                ask=price * 1.02,
                mid=price,
                iv=current_iv,
                greeks=greeks,
                delta=delta,
                theta=greeks.theta,
                vega=greeks.vega,
                score=score,
                reasoning=f"Wheel CSP: Δ{delta:.2f}, θ{greeks.theta:.2f}, IV rank {iv_rank:.0f}"
            )
            
            candidates.append(candidate)
        
        # Sort by score and return top N
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Wheel: Found {len(candidates)} candidates, returning top {top_n}")
        for i, c in enumerate(candidates[:top_n], 1):
            logger.info(f"  {i}. {c}")
        
        return candidates[:top_n]
    
    def _score_wheel_candidate(
        self,
        delta: float,
        theta: float,
        vega: float,
        premium: float,
        iv_rank: float,
        dte: int,
        target_dte: int
    ) -> float:
        """
        Score a Wheel strategy candidate (0-100).
        
        Prefer:
        - Delta close to 0.30
        - High theta (premium decay)
        - Moderate DTE (30-45 days)
        - High IV rank
        - Good premium (> 1% of strike weekly)
        """
        score = 50.0  # Base score
        
        # Delta score: prefer 0.30 ± 0.02
        delta_diff = abs(delta - 0.30)
        score += (0.05 - delta_diff) * 200  # Max +10 if perfect delta
        
        # Theta score: higher is better
        score += min(abs(theta) * 2, 15)  # Max +15
        
        # DTE score: prefer target DTE ± 5 days
        dte_diff = abs(dte - target_dte)
        score += max(10 - dte_diff, 0)  # Max +10
        
        # IV rank score: prefer high IV
        score += (iv_rank / 100) * 15  # Max +15
        
        # Premium score: prefer higher premium
        # Weekly premium should be ~1% of strike or more
        weekly_premium = premium * (7 / max(dte, 1))
        if weekly_premium > 1.0:
            score += min(weekly_premium, 10)
        
        return np.clip(score, 0, 100)
    
    # ============ CREDIT SPREADS ============
    
    def find_credit_spread_candidates(
        self,
        symbol: str,
        underlying_price: float,
        current_iv: float,
        spread_type: SpreadType,
        historical_vol: Optional[float] = None,
        iv_rank: Optional[float] = None,
        top_n: int = 5
    ) -> List[SpreadCandidate]:
        """
        Find credit spread candidates.
        
        Args:
            symbol: Underlying symbol
            underlying_price: Current price
            current_iv: Current IV
            spread_type: Bull put or bear call
            historical_vol: HV (optional)
            iv_rank: IV rank (optional)
            top_n: Number to return
            
        Returns:
            List of SpreadCandidates
        """
        # Analyze IV
        if iv_rank is None:
            metrics = self.iv_analyzer.analyze(symbol, current_iv, historical_vol)
            iv_rank = metrics.iv_rank
        
        # Get optimal DTE
        dte_rec = self.theta_engine.calculate_optimal_dte(
            iv_rank=iv_rank,
            trend=TrendDirection.NEUTRAL,
            volatility_regime=IVRegime.NORMAL,
            strategy_type=StrategyType.CREDIT_SPREAD.value
        )
        
        # Generate spread candidates
        candidates = []
        target_dte = (dte_rec.entry_dte_min + dte_rec.entry_dte_max) // 2
        
        # Spread width: $5 for stocks under $200, $10 for higher
        spread_width = 5 if underlying_price < 200 else 10
        
        # Generate strikes
        if spread_type == SpreadType.BULL_PUT:
            # Sell put below current price
            option_type = OptionType.PUT
            short_deltas = [0.25, 0.30, 0.35]  # Try different deltas
        else:
            # Sell call above current price
            option_type = OptionType.CALL
            short_deltas = [0.25, 0.30, 0.35]
        
        for short_delta_target in short_deltas:
            # Find strike closest to target delta
            strike = self._find_strike_for_delta(
                underlying_price=underlying_price,
                target_delta=short_delta_target,
                current_iv=current_iv,
                dte=target_dte,
                option_type=option_type
            )
            
            if spread_type == SpreadType.BULL_PUT:
                short_strike = strike
                long_strike = strike - spread_width
            else:
                short_strike = strike
                long_strike = strike + spread_width
            
            # Calculate spread value
            T = target_dte / 365.0
            
            if option_type == OptionType.PUT:
                short_price = self.bs.put_price(underlying_price, short_strike, T, current_iv)
                long_price = self.bs.put_price(underlying_price, long_strike, T, current_iv)
                short_greeks = self.bs.calculate_all_greeks(underlying_price, short_strike, T, current_iv, OptionType.PUT)
                long_greeks = self.bs.calculate_all_greeks(underlying_price, long_strike, T, current_iv, OptionType.PUT)
            else:
                short_price = self.bs.call_price(underlying_price, short_strike, T, current_iv)
                long_price = self.bs.call_price(underlying_price, long_strike, T, current_iv)
                short_greeks = self.bs.calculate_all_greeks(underlying_price, short_strike, T, current_iv, OptionType.CALL)
                long_greeks = self.bs.calculate_all_greeks(underlying_price, long_strike, T, current_iv, OptionType.CALL)
            
            net_credit = short_price - long_price
            max_profit = net_credit * 100  # Per contract
            max_loss = (spread_width - net_credit) * 100
            
            # Calculate breakeven
            if spread_type == SpreadType.BULL_PUT:
                breakeven = short_strike - net_credit
            else:
                breakeven = short_strike + net_credit
            
            # Estimate probability of profit (simplified)
            if spread_type == SpreadType.BULL_PUT:
                # Profit if closes above short strike
                pop = self._estimate_pop(underlying_price, short_strike, current_iv, target_dte, is_above=True)
            else:
                # Profit if closes below short strike
                pop = self._estimate_pop(underlying_price, short_strike, current_iv, target_dte, is_above=False)
            
            expected_value = (max_profit * (pop / 100)) - (max_loss * ((100 - pop) / 100))
            
            # Net Greeks (short - long)
            net_greeks = short_greeks - long_greeks
            
            # Score
            score = self._score_spread_candidate(
                net_credit=net_credit,
                max_loss=max_loss,
                pop=pop,
                iv_rank=iv_rank,
                net_theta=net_greeks.theta
            )
            
            candidate = SpreadCandidate(
                symbol=symbol,
                underlying_price=underlying_price,
                spread_type=spread_type,
                short_strike=short_strike,
                long_strike=long_strike,
                expiration=datetime.now() + timedelta(days=target_dte),
                dte=target_dte,
                net_credit=net_credit,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven=breakeven,
                pop=pop,
                expected_value=expected_value,
                short_greeks=short_greeks,
                long_greeks=long_greeks,
                net_greeks=net_greeks,
                score=score,
                reasoning=f"{spread_type.value}: Credit ${net_credit:.2f}, PoP {pop:.0f}%, EV ${expected_value:.0f}"
            )
            
            candidates.append(candidate)
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Credit Spread: Found {len(candidates)} candidates")
        return candidates[:top_n]
    
    def _score_spread_candidate(
        self,
        net_credit: float,
        max_loss: float,
        pop: float,
        iv_rank: float,
        net_theta: float
    ) -> float:
        """Score credit spread (0-100)."""
        score = 50.0
        
        # Risk/reward ratio
        if max_loss > 0:
            risk_reward = net_credit * 100 / max_loss
            score += min(risk_reward * 20, 20)  # Max +20
        
        # PoP score
        score += (pop / 100) * 20  # Max +20
        
        # IV rank
        score += (iv_rank / 100) * 10  # Max +10
        
        # Theta
        score += min(abs(net_theta), 10)
        
        return np.clip(score, 0, 100)
    
    # ============ IRON CONDOR ============
    
    def find_iron_condor_candidates(
        self,
        symbol: str,
        underlying_price: float,
        current_iv: float,
        historical_vol: Optional[float] = None,
        iv_rank: Optional[float] = None,
        top_n: int = 3
    ) -> List[IronCondorCandidate]:
        """
        Find iron condor candidates.
        
        Iron Condor:
        - Sell OTM put + buy further OTM put
        - Sell OTM call + buy further OTM call
        - Delta-neutral, profit from theta decay
        - Target 45 DTE, exit at 21 DTE or 50% profit
        
        Returns:
            List of IronCondorCandidates
        """
        # Analyze IV
        if iv_rank is None:
            metrics = self.iv_analyzer.analyze(symbol, current_iv, historical_vol)
            iv_rank = metrics.iv_rank
        
        # Iron condors work best in normal to high IV
        if iv_rank < 40:
            logger.warning(f"IC: Low IV rank ({iv_rank:.0f}), not ideal for iron condors")
        
        # Target DTE
        dte_range = IRON_CONDOR_DTE_RANGE
        target_dte = (dte_range[0] + dte_range[1]) // 2
        
        candidates = []
        
        # Wing width: $5 or $10
        wing_width = 5 if underlying_price < 200 else 10
        
        # Wing delta target
        wing_delta = IRON_CONDOR_WING_DELTA  # ~0.16
        
        T = target_dte / 365.0
        
        # Find put strikes
        put_short_strike = self._find_strike_for_delta(
            underlying_price, wing_delta, current_iv, target_dte, OptionType.PUT
        )
        put_long_strike = put_short_strike - wing_width
        
        # Find call strikes (symmetric)
        call_short_strike = self._find_strike_for_delta(
            underlying_price, wing_delta, current_iv, target_dte, OptionType.CALL
        )
        call_long_strike = call_short_strike + wing_width
        
        # Calculate put side
        put_short_price = self.bs.put_price(underlying_price, put_short_strike, T, current_iv)
        put_long_price = self.bs.put_price(underlying_price, put_long_strike, T, current_iv)
        put_credit = put_short_price - put_long_price
        
        # Calculate call side
        call_short_price = self.bs.call_price(underlying_price, call_short_strike, T, current_iv)
        call_long_price = self.bs.call_price(underlying_price, call_long_strike, T, current_iv)
        call_credit = call_short_price - call_long_price
        
        # Combined
        total_credit = put_credit + call_credit
        max_profit = total_credit * 100
        max_loss = (wing_width - total_credit) * 100
        
        breakeven_lower = put_short_strike - total_credit
        breakeven_upper = call_short_strike + total_credit
        profit_range = call_short_strike - put_short_strike
        
        # PoP: stock stays between short strikes
        pop = self._estimate_range_pop(
            underlying_price, put_short_strike, call_short_strike, current_iv, target_dte
        )
        
        expected_value = (max_profit * (pop / 100)) - (max_loss * ((100 - pop) / 100))
        
        # Net Greeks
        put_short_greeks = self.bs.calculate_all_greeks(underlying_price, put_short_strike, T, current_iv, OptionType.PUT)
        put_long_greeks = self.bs.calculate_all_greeks(underlying_price, put_long_strike, T, current_iv, OptionType.PUT)
        call_short_greeks = self.bs.calculate_all_greeks(underlying_price, call_short_strike, T, current_iv, OptionType.CALL)
        call_long_greeks = self.bs.calculate_all_greeks(underlying_price, call_long_strike, T, current_iv, OptionType.CALL)
        
        net_greeks = (put_short_greeks - put_long_greeks) + (call_short_greeks - call_long_greeks)
        
        # Score
        score = self._score_iron_condor(total_credit, max_loss, pop, iv_rank, profit_range, underlying_price)
        
        candidate = IronCondorCandidate(
            symbol=symbol,
            underlying_price=underlying_price,
            expiration=datetime.now() + timedelta(days=target_dte),
            dte=target_dte,
            put_short_strike=put_short_strike,
            put_long_strike=put_long_strike,
            put_credit=put_credit,
            call_short_strike=call_short_strike,
            call_long_strike=call_long_strike,
            call_credit=call_credit,
            total_credit=total_credit,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_lower=breakeven_lower,
            breakeven_upper=breakeven_upper,
            profit_range=profit_range,
            pop=pop,
            expected_value=expected_value,
            net_greeks=net_greeks,
            score=score,
            reasoning=f"IC: {profit_range:.0f} pt range, Credit ${total_credit:.2f}, PoP {pop:.0f}%"
        )
        
        candidates.append(candidate)
        
        logger.info(f"Iron Condor: Generated {len(candidates)} candidates")
        for c in candidates:
            logger.info(f"  {c}")
        
        return candidates[:top_n]
    
    def _score_iron_condor(
        self,
        total_credit: float,
        max_loss: float,
        pop: float,
        iv_rank: float,
        profit_range: float,
        underlying_price: float
    ) -> float:
        """Score iron condor (0-100)."""
        score = 50.0
        
        # Risk/reward
        if max_loss > 0:
            rr = (total_credit * 100) / max_loss
            score += min(rr * 20, 20)
        
        # PoP
        score += (pop / 100) * 25  # Max +25
        
        # IV rank
        score += (iv_rank / 100) * 15  # Max +15
        
        # Profit range as % of price
        range_pct = (profit_range / underlying_price) * 100
        score += min(range_pct, 10)  # Max +10
        
        return np.clip(score, 0, 100)
    
    # ============ HELPER METHODS ============
    
    def _generate_option_chain(
        self,
        underlying_price: float,
        current_iv: float,
        dte_range: Tuple[int, int],
        option_type: OptionType
    ) -> List[Dict]:
        """Generate synthetic option chain for testing."""
        chain = []
        
        # Generate strikes around current price
        for pct in np.arange(0.85, 1.15, 0.01):
            strike = round(underlying_price * pct)
            
            for dte in range(dte_range[0], dte_range[1] + 1, 7):
                expiration = datetime.now() + timedelta(days=dte)
                
                chain.append({
                    'strike': strike,
                    'dte': dte,
                    'expiration': expiration,
                    'option_type': option_type
                })
        
        return chain
    
    def _find_strike_for_delta(
        self,
        underlying_price: float,
        target_delta: float,
        current_iv: float,
        dte: int,
        option_type: OptionType
    ) -> float:
        """Find strike price that gives target delta."""
        T = dte / 365.0
        
        # Use Newton's method to find strike
        # Start with educated guess based on delta
        if option_type == OptionType.PUT:
            # Put delta is negative, we want abs(delta) = target
            # Lower strikes have lower absolute delta
            strike = underlying_price * (1 - target_delta)
        else:
            # Call delta is positive
            # Higher strikes have lower delta
            strike = underlying_price * (1 + target_delta)
        
        # Refine with a few iterations
        for _ in range(5):
            greeks = self.bs.calculate_all_greeks(
                underlying_price, strike, T, current_iv, option_type
            )
            current_delta = abs(greeks.delta)
            
            # Adjust strike based on delta error
            error = target_delta - current_delta
            strike += error * underlying_price * 0.5
        
        # Round to nearest dollar (or $0.50 for lower prices)
        if underlying_price > 100:
            strike = round(strike)
        else:
            strike = round(strike * 2) / 2
        
        return strike
    
    def _estimate_pop(
        self,
        current_price: float,
        target_strike: float,
        iv: float,
        dte: int,
        is_above: bool
    ) -> float:
        """
        Estimate probability of profit (simplified).
        
        Uses lognormal distribution.
        """
        T = dte / 365.0
        
        # Standard deviation of log returns
        std_dev = iv * np.sqrt(T)
        
        # Log of strike/spot ratio
        log_ratio = np.log(target_strike / current_price)
        
        # Z-score
        z = log_ratio / std_dev
        
        # Probability using normal CDF approximation
        from scipy.stats import norm
        
        if is_above:
            # Probability price stays above strike
            prob = 1 - norm.cdf(z)
        else:
            # Probability price stays below strike
            prob = norm.cdf(z)
        
        return prob * 100  # Return as percentage
    
    def _estimate_range_pop(
        self,
        current_price: float,
        lower_strike: float,
        upper_strike: float,
        iv: float,
        dte: int
    ) -> float:
        """Estimate probability price stays in range."""
        pop_above_lower = self._estimate_pop(current_price, lower_strike, iv, dte, is_above=True)
        pop_below_upper = self._estimate_pop(current_price, upper_strike, iv, dte, is_above=False)
        
        # Probability of being in range = both conditions true
        # Approximation: assume independence
        return (pop_above_lower / 100) * (pop_below_upper / 100) * 100

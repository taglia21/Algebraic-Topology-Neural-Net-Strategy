"""
Options Overlay Strategy for Phase 13
======================================

Selective options overlay on top of Phase 12 core strategy:
- Covered calls for income in strong bull
- Protective puts during transitions
- Long calls/puts for directional bets
- Straddles for volatility expansion
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .options_pricer import (
    BlackScholes, OptionType, OptionContract, OptionPrice,
    estimate_implied_vol, get_atm_strike, get_otm_call_strike, get_otm_put_strike
)

logger = logging.getLogger(__name__)


class OptionsStrategy(Enum):
    """Types of options strategies."""
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    CALL_SPREAD = "call_spread"
    PUT_SPREAD = "put_spread"


@dataclass
class OptionsSignal:
    """Signal for options trade."""
    strategy: OptionsStrategy
    underlying: str
    direction: str  # 'buy', 'sell'
    contracts: int
    strike: float
    expiry_days: int
    entry_price: float
    max_loss: float
    target_profit: float
    confidence: float
    reason: str


@dataclass
class OptionsPosition:
    """Active options position."""
    id: str
    strategy: OptionsStrategy
    underlying: str
    option_type: OptionType
    strike: float
    expiry_days: int
    contracts: int
    entry_price: float
    entry_date: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    days_held: int = 0
    
    @property
    def contract_value(self) -> float:
        """Value per contract (100 shares per contract)."""
        return self.current_price * 100 * self.contracts
    
    @property
    def pnl_pct(self) -> float:
        """Unrealized P&L percentage."""
        if self.entry_price > 0:
            return (self.current_price - self.entry_price) / self.entry_price
        return 0.0


@dataclass
class OptionsConfig:
    """Configuration for options overlay."""
    # Allocation limits
    max_options_pct: float = 0.30  # Max 30% in options
    max_single_position_pct: float = 0.05  # Max 5% per position
    
    # Strategy-specific allocation
    covered_call_pct: float = 0.15  # % of stock position for covered calls
    protective_put_pct: float = 0.05  # % for protective puts
    directional_pct: float = 0.20  # % for long calls/puts
    volatility_pct: float = 0.10  # % for straddles
    
    # Exit rules
    profit_target: float = 0.50  # Exit at 50% profit
    stop_loss: float = -0.30  # Exit at 30% loss
    min_days_to_expiry: int = 7  # Roll or exit before this
    
    # Entry rules
    min_days_to_expiry_entry: int = 30
    max_days_to_expiry_entry: int = 60
    min_iv_percentile: float = 0.20  # For selling premium
    max_iv_percentile: float = 0.80  # For buying premium


class OptionsOverlay:
    """
    Options overlay strategy integrated with Phase 12 core.
    
    Generates options signals based on regime and market conditions.
    """
    
    def __init__(self, config: OptionsConfig = None):
        self.config = config or OptionsConfig()
        self.positions: List[OptionsPosition] = []
        self.position_counter = 0
        self.closed_positions: List[Dict] = []
    
    def generate_signals(
        self,
        regime: str,
        underlying_prices: Dict[str, float],
        vix_level: float,
        portfolio_value: float,
        current_holdings: Dict[str, float],
        date: datetime = None,
    ) -> List[OptionsSignal]:
        """
        Generate options signals based on regime and market conditions.
        
        Args:
            regime: Current market regime
            underlying_prices: Dict of ticker -> current price
            vix_level: Current VIX level
            portfolio_value: Total portfolio value
            current_holdings: Dict of ticker -> position value
            date: Current date
            
        Returns:
            List of OptionsSignal to execute
        """
        signals = []
        
        # Calculate current options exposure
        options_exposure = sum(p.contract_value for p in self.positions)
        options_pct = options_exposure / portfolio_value if portfolio_value > 0 else 0
        
        # Check if we have room for more options
        available_allocation = self.config.max_options_pct - options_pct
        if available_allocation <= 0.02:
            return signals
        
        # Generate signals based on regime
        if regime == 'strong_bull':
            signals.extend(self._generate_bull_signals(
                underlying_prices, vix_level, portfolio_value, 
                current_holdings, available_allocation
            ))
        elif regime == 'mild_bull':
            signals.extend(self._generate_mild_bull_signals(
                underlying_prices, vix_level, portfolio_value,
                current_holdings, available_allocation
            ))
        elif regime == 'strong_bear':
            signals.extend(self._generate_bear_signals(
                underlying_prices, vix_level, portfolio_value,
                available_allocation
            ))
        elif regime == 'mild_bear':
            signals.extend(self._generate_mild_bear_signals(
                underlying_prices, vix_level, portfolio_value,
                available_allocation
            ))
        else:  # neutral
            signals.extend(self._generate_neutral_signals(
                underlying_prices, vix_level, portfolio_value,
                available_allocation
            ))
        
        return signals
    
    def _generate_bull_signals(
        self,
        prices: Dict[str, float],
        vix: float,
        portfolio: float,
        holdings: Dict[str, float],
        available: float,
    ) -> List[OptionsSignal]:
        """Generate signals for strong bull regime."""
        signals = []
        
        # 1. Covered calls on long positions (income generation)
        for ticker, holding_value in holdings.items():
            if holding_value > 0 and ticker in prices:
                price = prices[ticker]
                strike = get_otm_call_strike(price, pct_otm=0.08)  # 8% OTM
                iv = estimate_implied_vol(ticker, vix)
                
                # Price the call we'd sell
                contract = OptionContract(
                    underlying=ticker,
                    option_type=OptionType.CALL,
                    strike=strike,
                    expiry_days=30,
                    underlying_price=price,
                    implied_vol=iv,
                )
                option_price = BlackScholes.price_option(contract)
                
                # Calculate contracts (1 contract = 100 shares)
                shares_held = int(holding_value / price)
                contracts = shares_held // 100
                
                if contracts > 0:
                    signals.append(OptionsSignal(
                        strategy=OptionsStrategy.COVERED_CALL,
                        underlying=ticker,
                        direction='sell',
                        contracts=contracts,
                        strike=strike,
                        expiry_days=30,
                        entry_price=option_price.price,
                        max_loss=0,  # Covered, no additional risk
                        target_profit=option_price.price * contracts * 100,
                        confidence=0.75,
                        reason="Income generation in strong bull",
                    ))
        
        # 2. Long calls for additional upside (aggressive)
        if 'QQQ' in prices:
            qqq_price = prices['QQQ']
            strike = get_atm_strike(qqq_price)
            iv = estimate_implied_vol('QQQ', vix)
            
            contract = OptionContract(
                underlying='QQQ',
                option_type=OptionType.CALL,
                strike=strike,
                expiry_days=45,
                underlying_price=qqq_price,
                implied_vol=iv,
            )
            option_price = BlackScholes.price_option(contract)
            
            # Allocate up to 10% for long calls in strong bull
            allocation = min(available * 0.5, self.config.directional_pct) * portfolio
            contracts = int(allocation / (option_price.price * 100))
            
            if contracts > 0:
                signals.append(OptionsSignal(
                    strategy=OptionsStrategy.LONG_CALL,
                    underlying='QQQ',
                    direction='buy',
                    contracts=contracts,
                    strike=strike,
                    expiry_days=45,
                    entry_price=option_price.price,
                    max_loss=option_price.price * contracts * 100,
                    target_profit=option_price.price * contracts * 100 * 2,  # 2x target
                    confidence=0.70,
                    reason="Amplify bull gains with long calls",
                ))
        
        return signals
    
    def _generate_mild_bull_signals(
        self,
        prices: Dict[str, float],
        vix: float,
        portfolio: float,
        holdings: Dict[str, float],
        available: float,
    ) -> List[OptionsSignal]:
        """Generate signals for mild bull regime."""
        signals = []
        
        # Protective puts on leveraged holdings
        for ticker in ['TQQQ', 'SPXL', 'SOXL']:
            if ticker in holdings and holdings[ticker] > 0 and ticker in prices:
                price = prices[ticker]
                strike = get_otm_put_strike(price, pct_otm=0.10)  # 10% OTM
                iv = estimate_implied_vol(ticker, vix)
                
                contract = OptionContract(
                    underlying=ticker,
                    option_type=OptionType.PUT,
                    strike=strike,
                    expiry_days=30,
                    underlying_price=price,
                    implied_vol=iv,
                )
                option_price = BlackScholes.price_option(contract)
                
                # Protect a portion of the position
                protect_value = holdings[ticker] * 0.5  # Protect 50%
                contracts = int(protect_value / (price * 100))
                
                if contracts > 0:
                    signals.append(OptionsSignal(
                        strategy=OptionsStrategy.PROTECTIVE_PUT,
                        underlying=ticker,
                        direction='buy',
                        contracts=contracts,
                        strike=strike,
                        expiry_days=30,
                        entry_price=option_price.price,
                        max_loss=option_price.price * contracts * 100,
                        target_profit=protect_value * 0.10,  # Offset 10% drop
                        confidence=0.65,
                        reason="Protect leveraged position in mild bull",
                    ))
        
        return signals
    
    def _generate_bear_signals(
        self,
        prices: Dict[str, float],
        vix: float,
        portfolio: float,
        available: float,
    ) -> List[OptionsSignal]:
        """Generate signals for strong bear regime."""
        signals = []
        
        # Long puts as alternative/complement to inverse ETFs
        for ticker in ['QQQ', 'SPY']:
            if ticker in prices:
                price = prices[ticker]
                strike = get_atm_strike(price)  # ATM for maximum delta
                iv = estimate_implied_vol(ticker, vix)
                
                contract = OptionContract(
                    underlying=ticker,
                    option_type=OptionType.PUT,
                    strike=strike,
                    expiry_days=45,
                    underlying_price=price,
                    implied_vol=iv,
                )
                option_price = BlackScholes.price_option(contract)
                
                # Allocate 15% for put options in bear
                allocation = min(available * 0.5, 0.15) * portfolio
                contracts = int(allocation / (option_price.price * 100))
                
                if contracts > 0:
                    signals.append(OptionsSignal(
                        strategy=OptionsStrategy.LONG_PUT,
                        underlying=ticker,
                        direction='buy',
                        contracts=contracts,
                        strike=strike,
                        expiry_days=45,
                        entry_price=option_price.price,
                        max_loss=option_price.price * contracts * 100,
                        target_profit=option_price.price * contracts * 100 * 3,  # 3x in bear
                        confidence=0.72,
                        reason="Capture downside with long puts in bear",
                    ))
        
        return signals
    
    def _generate_mild_bear_signals(
        self,
        prices: Dict[str, float],
        vix: float,
        portfolio: float,
        available: float,
    ) -> List[OptionsSignal]:
        """Generate signals for mild bear regime."""
        signals = []
        
        # Put spreads for defined risk bearish bet
        if 'SPY' in prices:
            price = prices['SPY']
            
            # Buy ATM put, sell OTM put (bear put spread)
            buy_strike = get_atm_strike(price)
            sell_strike = get_otm_put_strike(price, pct_otm=0.08)
            iv = estimate_implied_vol('SPY', vix)
            
            buy_contract = OptionContract('SPY', OptionType.PUT, buy_strike, 30, price, iv)
            sell_contract = OptionContract('SPY', OptionType.PUT, sell_strike, 30, price, iv)
            
            buy_price = BlackScholes.price_option(buy_contract).price
            sell_price = BlackScholes.price_option(sell_contract).price
            net_debit = buy_price - sell_price
            
            allocation = min(available * 0.3, 0.08) * portfolio
            contracts = int(allocation / (net_debit * 100)) if net_debit > 0 else 0
            
            if contracts > 0:
                signals.append(OptionsSignal(
                    strategy=OptionsStrategy.PUT_SPREAD,
                    underlying='SPY',
                    direction='buy',
                    contracts=contracts,
                    strike=buy_strike,  # Long strike
                    expiry_days=30,
                    entry_price=net_debit,
                    max_loss=net_debit * contracts * 100,
                    target_profit=(buy_strike - sell_strike) * contracts * 100 - net_debit * contracts * 100,
                    confidence=0.60,
                    reason="Bear put spread in mild bear",
                ))
        
        return signals
    
    def _generate_neutral_signals(
        self,
        prices: Dict[str, float],
        vix: float,
        portfolio: float,
        available: float,
    ) -> List[OptionsSignal]:
        """Generate signals for neutral regime."""
        signals = []
        
        # If VIX is elevated, consider straddles
        if vix > 25 and 'SPY' in prices:
            price = prices['SPY']
            strike = get_atm_strike(price)
            iv = estimate_implied_vol('SPY', vix)
            
            call_contract = OptionContract('SPY', OptionType.CALL, strike, 30, price, iv)
            put_contract = OptionContract('SPY', OptionType.PUT, strike, 30, price, iv)
            
            call_price = BlackScholes.price_option(call_contract).price
            put_price = BlackScholes.price_option(put_contract).price
            straddle_cost = call_price + put_price
            
            allocation = min(available * 0.3, self.config.volatility_pct) * portfolio
            contracts = int(allocation / (straddle_cost * 100))
            
            if contracts > 0:
                signals.append(OptionsSignal(
                    strategy=OptionsStrategy.STRADDLE,
                    underlying='SPY',
                    direction='buy',
                    contracts=contracts,
                    strike=strike,
                    expiry_days=30,
                    entry_price=straddle_cost,
                    max_loss=straddle_cost * contracts * 100,
                    target_profit=straddle_cost * contracts * 100,  # 100% target
                    confidence=0.55,
                    reason="Straddle for volatility expansion in neutral",
                ))
        
        return signals
    
    def update_positions(
        self,
        prices: Dict[str, float],
        vix: float,
        date: datetime,
    ) -> Tuple[List[Dict], float]:
        """
        Update existing positions and check exit conditions.
        
        Returns:
            Tuple of (closed positions, total realized P&L)
        """
        closed = []
        realized_pnl = 0.0
        remaining_positions = []
        
        for position in self.positions:
            position.days_held += 1
            
            # Update current price
            if position.underlying in prices:
                price = prices[position.underlying]
                iv = estimate_implied_vol(position.underlying, vix)
                
                remaining_days = max(1, position.expiry_days - position.days_held)
                
                contract = OptionContract(
                    underlying=position.underlying,
                    option_type=position.option_type,
                    strike=position.strike,
                    expiry_days=remaining_days,
                    underlying_price=price,
                    implied_vol=iv,
                )
                position.current_price = BlackScholes.price_option(contract).price
                position.unrealized_pnl = (position.current_price - position.entry_price) * position.contracts * 100
            
            # Check exit conditions
            pnl_pct = position.pnl_pct
            days_remaining = position.expiry_days - position.days_held
            
            should_close = False
            close_reason = ""
            
            # Profit target
            if pnl_pct >= self.config.profit_target:
                should_close = True
                close_reason = f"Profit target hit ({pnl_pct:.1%})"
            
            # Stop loss
            elif pnl_pct <= self.config.stop_loss:
                should_close = True
                close_reason = f"Stop loss hit ({pnl_pct:.1%})"
            
            # Expiration approaching
            elif days_remaining <= self.config.min_days_to_expiry:
                should_close = True
                close_reason = f"Expiration approaching ({days_remaining} days)"
            
            if should_close:
                realized = position.unrealized_pnl
                realized_pnl += realized
                closed.append({
                    'id': position.id,
                    'underlying': position.underlying,
                    'strategy': position.strategy.value,
                    'entry_price': position.entry_price,
                    'exit_price': position.current_price,
                    'pnl': realized,
                    'pnl_pct': pnl_pct,
                    'days_held': position.days_held,
                    'close_reason': close_reason,
                    'date': date,
                })
                self.closed_positions.append(closed[-1])
            else:
                remaining_positions.append(position)
        
        self.positions = remaining_positions
        return closed, realized_pnl
    
    def open_position(
        self,
        signal: OptionsSignal,
        date: datetime,
    ) -> OptionsPosition:
        """Open a new options position from a signal."""
        self.position_counter += 1
        position_id = f"OPT-{self.position_counter:04d}"
        
        option_type = OptionType.CALL if 'call' in signal.strategy.value.lower() else OptionType.PUT
        
        position = OptionsPosition(
            id=position_id,
            strategy=signal.strategy,
            underlying=signal.underlying,
            option_type=option_type,
            strike=signal.strike,
            expiry_days=signal.expiry_days,
            contracts=signal.contracts,
            entry_price=signal.entry_price,
            entry_date=date,
            current_price=signal.entry_price,
        )
        
        self.positions.append(position)
        return position
    
    def get_total_exposure(self) -> float:
        """Get total options exposure."""
        return sum(p.contract_value for p in self.positions)
    
    def get_position_summary(self) -> Dict:
        """Get summary of current positions."""
        return {
            'n_positions': len(self.positions),
            'total_exposure': self.get_total_exposure(),
            'unrealized_pnl': sum(p.unrealized_pnl for p in self.positions),
            'by_strategy': {
                strategy.value: sum(
                    p.contract_value for p in self.positions 
                    if p.strategy == strategy
                )
                for strategy in OptionsStrategy
            },
        }
    
    def reset(self):
        """Reset overlay state."""
        self.positions = []
        self.position_counter = 0
        self.closed_positions = []

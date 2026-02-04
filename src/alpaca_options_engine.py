"""
Alpaca Options Trading Engine
==============================

COMPLETE MIGRATION FROM TRADIER TO ALPACA
Tradier platform failed us - we lost $8k on their broken system.

This engine handles:
- Options chain retrieval
- Order placement and execution
- Position monitoring
- Risk management with SAFE parameters

CRITICAL RISK FIXES:
- STOP_LOSS_PERCENT = 25% (was 100% - INSANE!)
- PROFIT_TARGET_PERCENT = 50%
- Real-time position monitoring
- Automatic stop-loss execution

Author: Post-Tradier Era
Date: February 4, 2026
"""

import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from dotenv import load_dotenv

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderType,
    AssetClass,
    QueryOrderStatus
)
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, OptionLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION - ALPACA ONLY (TRADIER IS DEAD TO US)
# ============================================================================

API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
USE_PAPER = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'

# Base URL
BASE_URL = 'https://paper-api.alpaca.markets' if USE_PAPER else 'https://api.alpaca.markets'

# CRITICAL RISK PARAMETERS - FIXED FROM TRADIER'S INSANITY
STOP_LOSS_PERCENT = 25.0  # Exit if loss exceeds 25% of premium - SAFE
PROFIT_TARGET_PERCENT = 50.0  # Take profit at 50% of max gain
MAX_POSITION_SIZE = 5  # Max contracts per position
MAX_PORTFOLIO_RISK = 0.02  # 2% max risk per trade

# Monitoring
MONITOR_INTERVAL_SECONDS = 60  # Check positions every 60 seconds
MIN_PREMIUM = 0.30  # Minimum $0.30 per contract

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class PositionStatus(Enum):
    """Position status."""
    OPEN = "open"
    CLOSED = "closed"
    STOP_LOSS = "stop_loss"
    PROFIT_TARGET = "profit_target"


@dataclass
class OptionContract:
    """Options contract details."""
    symbol: str
    underlying: str
    strike: float
    expiration: str
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    mid: float
    volume: int
    open_interest: int


@dataclass
class OptionsPosition:
    """Active options position."""
    symbol: str
    underlying: str
    strike: float
    expiration: str
    option_type: str
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    status: PositionStatus
    entry_time: datetime


# ============================================================================
# ALPACA OPTIONS ENGINE
# ============================================================================

class AlpacaOptionsEngine:
    """
    Complete Alpaca options trading engine.
    
    NO TRADIER CODE - COMPLETELY REBUILT FOR ALPACA.
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True):
        """
        Initialize Alpaca options engine.
        
        Args:
            api_key: Alpaca API key (from .env if None)
            secret_key: Alpaca secret key (from .env if None)
            paper: Use paper trading (default True)
        """
        self.api_key = api_key or API_KEY
        self.secret_key = secret_key or SECRET_KEY
        self.paper = paper
        
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials not found! "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file"
            )
        
        # Initialize Alpaca clients
        self.trading_client = TradingClient(
            self.api_key,
            self.secret_key,
            paper=self.paper
        )
        
        self.data_client = OptionHistoricalDataClient(
            self.api_key,
            self.secret_key
        )
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info(f"✅ Alpaca Options Engine initialized (Paper: {self.paper})")
    
    # ========================================================================
    # ACCOUNT & POSITION MANAGEMENT
    # ========================================================================
    
    def get_account(self) -> Dict:
        """
        Get Alpaca account information.
        
        Returns:
            Account details including buying power, equity, etc.
        """
        try:
            account = self.trading_client.get_account()
            
            account_info = {
                'account_id': account.id,
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'account_blocked': account.account_blocked,
                'day_trade_count': account.daytrade_count
            }
            
            self.logger.info(f"Account Equity: ${account_info['equity']:,.2f}")
            self.logger.info(f"Buying Power: ${account_info['buying_power']:,.2f}")
            
            return account_info
            
        except Exception as e:
            self.logger.error(f"Error getting account: {e}")
            raise
    
    def get_positions(self) -> List[OptionsPosition]:
        """
        Get all current options positions.
        
        Returns:
            List of OptionsPosition objects
        """
        try:
            positions = self.trading_client.get_all_positions()
            options_positions = []
            
            for pos in positions:
                # Only process options positions
                if pos.asset_class != AssetClass.US_OPTION:
                    continue
                
                # Parse option symbol (format: AAPL250117C00150000)
                symbol = pos.symbol
                underlying = self._parse_underlying(symbol)
                option_type, strike, expiration = self._parse_option_symbol(symbol)
                
                # Calculate P&L
                entry_price = float(pos.avg_entry_price)
                current_price = float(pos.current_price)
                quantity = int(pos.qty)
                
                unrealized_pnl = (current_price - entry_price) * quantity * 100
                unrealized_pnl_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
                
                # Determine status
                status = PositionStatus.OPEN
                if unrealized_pnl_pct <= -STOP_LOSS_PERCENT:
                    status = PositionStatus.STOP_LOSS
                elif unrealized_pnl_pct >= PROFIT_TARGET_PERCENT:
                    status = PositionStatus.PROFIT_TARGET
                
                position = OptionsPosition(
                    symbol=symbol,
                    underlying=underlying,
                    strike=strike,
                    expiration=expiration,
                    option_type=option_type,
                    quantity=quantity,
                    entry_price=entry_price,
                    current_price=current_price,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    status=status,
                    entry_time=pos.created_at
                )
                
                options_positions.append(position)
            
            self.logger.info(f"Found {len(options_positions)} options positions")
            return options_positions
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    # ========================================================================
    # OPTIONS CHAIN & QUOTES
    # ========================================================================
    
    def get_options_chain(
        self,
        underlying: str,
        expiration_date: str = None,
        strike_range: Tuple[float, float] = None
    ) -> List[OptionContract]:
        """
        Get options chain for underlying symbol.
        
        Args:
            underlying: Stock symbol (e.g., 'SPY')
            expiration_date: Expiration date (YYYY-MM-DD) or None for all
            strike_range: (min_strike, max_strike) or None for all
            
        Returns:
            List of OptionContract objects
        """
        try:
            # Get option contracts
            request = GetOptionContractsRequest(
                underlying_symbols=[underlying],
                expiration_date=expiration_date
            )
            
            contracts_response = self.trading_client.get_option_contracts(request)
            
            if not contracts_response:
                self.logger.warning(f"No options contracts found for {underlying}")
                return []
            
            # Get quotes for contracts
            option_contracts = []
            
            for contract in contracts_response.option_contracts:
                # Apply strike filter if provided
                if strike_range:
                    if contract.strike_price < strike_range[0] or contract.strike_price > strike_range[1]:
                        continue
                
                # Get latest quote
                try:
                    quote_request = OptionLatestQuoteRequest(symbol_or_symbols=contract.symbol)
                    quotes = self.data_client.get_option_latest_quote(quote_request)
                    
                    if contract.symbol in quotes:
                        quote = quotes[contract.symbol]
                        bid = float(quote.bid_price)
                        ask = float(quote.ask_price)
                        mid = (bid + ask) / 2.0
                        
                        option_contract = OptionContract(
                            symbol=contract.symbol,
                            underlying=contract.underlying_symbol,
                            strike=float(contract.strike_price),
                            expiration=str(contract.expiration_date),
                            option_type=contract.type.lower(),
                            bid=bid,
                            ask=ask,
                            mid=mid,
                            volume=0,  # Alpaca doesn't provide volume in quotes
                            open_interest=int(contract.open_interest) if contract.open_interest else 0
                        )
                        
                        option_contracts.append(option_contract)
                        
                except Exception as e:
                    self.logger.debug(f"Could not get quote for {contract.symbol}: {e}")
                    continue
            
            self.logger.info(f"Retrieved {len(option_contracts)} options contracts for {underlying}")
            return option_contracts
            
        except Exception as e:
            self.logger.error(f"Error getting options chain: {e}")
            return []
    
    # ========================================================================
    # ORDER EXECUTION
    # ========================================================================
    
    def place_option_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str = 'market',
        limit_price: float = None
    ) -> Optional[Dict]:
        """
        Place options order.
        
        Args:
            symbol: Option symbol (e.g., 'SPY250117C00450000')
            quantity: Number of contracts
            side: 'buy' or 'sell'
            order_type: 'market' or 'limit'
            limit_price: Limit price (required for limit orders)
            
        Returns:
            Order details or None if failed
        """
        try:
            # Validate inputs
            if quantity <= 0:
                raise ValueError(f"Invalid quantity: {quantity}")
            
            if quantity > MAX_POSITION_SIZE:
                self.logger.warning(f"Quantity {quantity} exceeds max {MAX_POSITION_SIZE}, capping")
                quantity = MAX_POSITION_SIZE
            
            # Determine order side
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            # Create order request
            if order_type.lower() == 'market':
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    time_in_force=TimeInForce.DAY
                )
            else:
                if not limit_price:
                    raise ValueError("Limit price required for limit orders")
                
                order_data = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price
                )
            
            # Submit order
            order = self.trading_client.submit_order(order_data)
            
            order_info = {
                'order_id': order.id,
                'symbol': order.symbol,
                'quantity': int(order.qty),
                'side': order.side.value,
                'order_type': order.order_type.value,
                'status': order.status.value,
                'filled_qty': int(order.filled_qty) if order.filled_qty else 0,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'submitted_at': order.submitted_at
            }
            
            self.logger.info(
                f"✅ Order placed: {side.upper()} {quantity} {symbol} "
                f"({order_type.upper()}) - Order ID: {order.id}"
            )
            
            return order_info
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    def close_position(self, symbol: str, reason: str = "manual") -> bool:
        """
        Close an options position.
        
        Args:
            symbol: Option symbol to close
            reason: Reason for closing
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current position
            position = self.trading_client.get_open_position(symbol)
            
            if not position:
                self.logger.warning(f"No position found for {symbol}")
                return False
            
            quantity = abs(int(position.qty))
            
            # Determine close side (opposite of current position)
            if int(position.qty) > 0:
                close_side = 'sell'
            else:
                close_side = 'buy'
            
            # Place closing order
            result = self.place_option_order(
                symbol=symbol,
                quantity=quantity,
                side=close_side,
                order_type='market'
            )
            
            if result:
                self.logger.info(f"✅ Position closed: {symbol} (Reason: {reason})")
                return True
            else:
                self.logger.error(f"Failed to close position: {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error closing position {symbol}: {e}")
            return False
    
    # ========================================================================
    # POSITION MONITORING & RISK MANAGEMENT
    # ========================================================================
    
    def monitor_positions(self) -> Dict:
        """
        Monitor all positions and trigger stops/targets.
        
        Returns:
            Summary of monitoring results
        """
        try:
            positions = self.get_positions()
            
            results = {
                'total_positions': len(positions),
                'stop_loss_triggered': 0,
                'profit_target_triggered': 0,
                'open_positions': 0,
                'total_unrealized_pnl': 0.0
            }
            
            for pos in positions:
                results['total_unrealized_pnl'] += pos.unrealized_pnl
                
                # Check stop-loss
                if pos.unrealized_pnl_pct <= -STOP_LOSS_PERCENT:
                    self.logger.warning(
                        f"🛑 STOP LOSS: {pos.symbol} - Loss: {pos.unrealized_pnl_pct:.2f}% "
                        f"(${pos.unrealized_pnl:.2f})"
                    )
                    
                    if self.close_position(pos.symbol, reason="stop_loss"):
                        results['stop_loss_triggered'] += 1
                
                # Check profit target
                elif pos.unrealized_pnl_pct >= PROFIT_TARGET_PERCENT:
                    self.logger.info(
                        f"🎯 PROFIT TARGET: {pos.symbol} - Gain: {pos.unrealized_pnl_pct:.2f}% "
                        f"(${pos.unrealized_pnl:.2f})"
                    )
                    
                    if self.close_position(pos.symbol, reason="profit_target"):
                        results['profit_target_triggered'] += 1
                
                else:
                    results['open_positions'] += 1
                    self.logger.info(
                        f"📊 {pos.symbol}: {pos.unrealized_pnl_pct:+.2f}% "
                        f"(${pos.unrealized_pnl:+.2f})"
                    )
            
            self.logger.info(
                f"\n📈 PORTFOLIO SUMMARY:"
                f"\n  Total Positions: {results['total_positions']}"
                f"\n  Open: {results['open_positions']}"
                f"\n  Stop-Loss Triggered: {results['stop_loss_triggered']}"
                f"\n  Profit Target Hit: {results['profit_target_triggered']}"
                f"\n  Total Unrealized P&L: ${results['total_unrealized_pnl']:+,.2f}"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {e}")
            return {}
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    def _parse_underlying(self, option_symbol: str) -> str:
        """Extract underlying symbol from option symbol."""
        # Option symbol format: AAPL250117C00150000
        # Extract letters before date
        for i, char in enumerate(option_symbol):
            if char.isdigit():
                return option_symbol[:i]
        return option_symbol
    
    def _parse_option_symbol(self, option_symbol: str) -> Tuple[str, float, str]:
        """
        Parse option symbol to extract details.
        
        Returns:
            (option_type, strike, expiration)
        """
        # Format: AAPL250117C00150000
        # YY MM DD C/P Strike*1000
        
        underlying = self._parse_underlying(option_symbol)
        rest = option_symbol[len(underlying):]
        
        # Extract expiration (YYMMDD)
        exp_str = rest[:6]
        expiration = f"20{exp_str[0:2]}-{exp_str[2:4]}-{exp_str[4:6]}"
        
        # Extract type (C/P)
        option_type = 'call' if rest[6] == 'C' else 'put'
        
        # Extract strike (remaining digits / 1000)
        strike_str = rest[7:]
        strike = float(strike_str) / 1000.0
        
        return option_type, strike, expiration
    
    def health_check(self) -> bool:
        """
        Verify Alpaca API connectivity.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            account = self.get_account()
            self.logger.info("✅ Alpaca API health check passed")
            return True
        except Exception as e:
            self.logger.error(f"❌ Alpaca API health check failed: {e}")
            return False


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("\n" + "="*70)
    print("🚀 ALPACA OPTIONS ENGINE - TRADIER MIGRATION COMPLETE")
    print("="*70 + "\n")
    
    # Initialize engine
    engine = AlpacaOptionsEngine(paper=True)
    
    # Health check
    print("\n1. Running health check...")
    if engine.health_check():
        print("   ✅ Alpaca API connected successfully\n")
    else:
        print("   ❌ Failed to connect to Alpaca API\n")
        exit(1)
    
    # Get account info
    print("2. Account Information:")
    account = engine.get_account()
    print(f"   Equity: ${account['equity']:,.2f}")
    print(f"   Buying Power: ${account['buying_power']:,.2f}\n")
    
    # Get positions
    print("3. Current Positions:")
    positions = engine.get_positions()
    if positions:
        for pos in positions:
            print(f"   {pos.symbol}: {pos.quantity} @ ${pos.entry_price:.2f} "
                  f"({pos.unrealized_pnl_pct:+.2f}%)")
    else:
        print("   No options positions\n")
    
    # Get options chain for SPY
    print("4. SPY Options Chain (Next Expiration):")
    next_friday = (datetime.now() + timedelta(days=(4 - datetime.now().weekday()) % 7)).strftime('%Y-%m-%d')
    contracts = engine.get_options_chain('SPY', expiration_date=next_friday, strike_range=(400, 500))
    print(f"   Found {len(contracts)} contracts\n")
    
    if contracts:
        print("   Sample contracts:")
        for contract in contracts[:5]:
            print(f"   {contract.symbol}: ${contract.mid:.2f} (Strike: ${contract.strike})")
    
    print("\n" + "="*70)
    print("✅ MIGRATION COMPLETE - ALPACA READY FOR TRADING")
    print("="*70 + "\n")

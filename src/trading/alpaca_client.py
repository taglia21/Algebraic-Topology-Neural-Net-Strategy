"""
Alpaca Trading Client - Secure API Integration
================================================

Provides secure connection to Alpaca API with:
- Environment-based credential loading
- Automatic paper/live detection
- Connection health checks
- Rate limiting and retry logic
- Comprehensive error handling

NEVER hardcode credentials - always use .env file
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if env vars set externally

import requests

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


@dataclass
class Position:
    """Current position in a security."""
    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    current_price: float
    unrealized_pl: float
    unrealized_plpc: float
    side: str  # 'long' or 'short'


@dataclass
class Order:
    """Order details."""
    id: str
    symbol: str
    qty: float
    side: str
    type: str
    status: str
    filled_qty: float
    filled_avg_price: Optional[float]
    submitted_at: str
    filled_at: Optional[str]


@dataclass
class Account:
    """Account information."""
    id: str
    cash: float
    portfolio_value: float
    buying_power: float
    equity: float
    last_equity: float
    daytrading_buying_power: float
    pattern_day_trader: bool
    trading_blocked: bool
    account_blocked: bool


class AlpacaClient:
    """
    Secure Alpaca API client with comprehensive error handling.
    
    Usage:
        client = AlpacaClient()
        account = client.get_account()
        client.submit_order("AAPL", 10, OrderSide.BUY)
    """
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = 200
    REQUEST_DELAY = 0.3  # seconds between requests
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    
    def __init__(self):
        """Initialize client from environment variables."""
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self.is_paper = os.getenv("PAPER_TRADING", "true").lower() == "true"
        
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "in .env file or environment variables."
            )
        
        # Verify paper trading for safety
        if "paper" not in self.base_url.lower() and self.is_paper:
            logger.warning("⚠️  PAPER_TRADING=true but base URL doesn't contain 'paper'. "
                          "Forcing paper URL for safety.")
            self.base_url = "https://paper-api.alpaca.markets"
        
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json"
        }
        
        self._last_request_time = 0
        self._request_count = 0
        
        logger.info(f"AlpacaClient initialized - {'PAPER' if self.is_paper else 'LIVE'} trading")
        logger.info(f"Base URL: {self.base_url}")
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        now = time.time()
        elapsed = now - self._last_request_time
        
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        
        self._last_request_time = time.time()
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict:
        """Make API request with retry logic."""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.MAX_RETRIES):
            try:
                self._rate_limit()
                
                response = requests.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    json=data,
                    params=params,
                    timeout=30,
                )
                
                if response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                
                if response.text:
                    return response.json()
                return {}
                
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    raise
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error on attempt {attempt + 1}: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise
        
        raise RuntimeError(f"Failed after {self.MAX_RETRIES} attempts")
    
    # ========== Account Methods ==========
    
    def get_account(self) -> Account:
        """Get account information."""
        data = self._request("GET", "/v2/account")
        
        return Account(
            id=data.get("id", ""),
            cash=float(data.get("cash", 0)),
            portfolio_value=float(data.get("portfolio_value", 0)),
            buying_power=float(data.get("buying_power", 0)),
            equity=float(data.get("equity", 0)),
            last_equity=float(data.get("last_equity", 0)),
            daytrading_buying_power=float(data.get("daytrading_buying_power", 0)),
            pattern_day_trader=data.get("pattern_day_trader", False),
            trading_blocked=data.get("trading_blocked", False),
            account_blocked=data.get("account_blocked", False),
        )
    
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        data = self._request("GET", "/v2/clock")
        return data.get("is_open", False)
    
    def get_market_hours(self) -> Dict[str, str]:
        """Get today's market hours."""
        data = self._request("GET", "/v2/clock")
        return {
            "is_open": data.get("is_open", False),
            "next_open": data.get("next_open", ""),
            "next_close": data.get("next_close", ""),
            "timestamp": data.get("timestamp", ""),
        }
    
    # ========== Position Methods ==========
    
    def get_positions(self) -> List[Position]:
        """Get all current positions."""
        data = self._request("GET", "/v2/positions")
        
        positions = []
        for item in data:
            positions.append(Position(
                symbol=item.get("symbol", ""),
                qty=float(item.get("qty", 0)),
                market_value=float(item.get("market_value", 0)),
                avg_entry_price=float(item.get("avg_entry_price", 0)),
                current_price=float(item.get("current_price", 0)),
                unrealized_pl=float(item.get("unrealized_pl", 0)),
                unrealized_plpc=float(item.get("unrealized_plpc", 0)),
                side=item.get("side", "long"),
            ))
        
        return positions
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        try:
            data = self._request("GET", f"/v2/positions/{symbol}")
            return Position(
                symbol=data.get("symbol", ""),
                qty=float(data.get("qty", 0)),
                market_value=float(data.get("market_value", 0)),
                avg_entry_price=float(data.get("avg_entry_price", 0)),
                current_price=float(data.get("current_price", 0)),
                unrealized_pl=float(data.get("unrealized_pl", 0)),
                unrealized_plpc=float(data.get("unrealized_plpc", 0)),
                side=data.get("side", "long"),
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def close_position(self, symbol: str) -> Optional[Order]:
        """Close entire position for a symbol."""
        try:
            data = self._request("DELETE", f"/v2/positions/{symbol}")
            return Order(
                id=data.get("id", ""),
                symbol=data.get("symbol", ""),
                qty=float(data.get("qty", 0)),
                side=data.get("side", ""),
                type=data.get("type", ""),
                status=data.get("status", ""),
                filled_qty=float(data.get("filled_qty", 0)),
                filled_avg_price=float(data.get("filled_avg_price", 0)) if data.get("filled_avg_price") else None,
                submitted_at=data.get("submitted_at", ""),
                filled_at=data.get("filled_at"),
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def close_all_positions(self) -> List[Order]:
        """Close all positions (liquidate portfolio)."""
        data = self._request("DELETE", "/v2/positions")
        
        orders = []
        for item in data:
            if item.get("status") == 200:
                body = item.get("body", {})
                orders.append(Order(
                    id=body.get("id", ""),
                    symbol=body.get("symbol", ""),
                    qty=float(body.get("qty", 0)),
                    side=body.get("side", ""),
                    type=body.get("type", ""),
                    status=body.get("status", ""),
                    filled_qty=float(body.get("filled_qty", 0)),
                    filled_avg_price=None,
                    submitted_at=body.get("submitted_at", ""),
                    filled_at=None,
                ))
        
        return orders
    
    # ========== Order Methods ==========
    
    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        time_in_force: TimeInForce = TimeInForce.DAY,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """Submit a new order."""
        order_data = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side.value,
            "type": order_type.value,
            "time_in_force": time_in_force.value,
        }
        
        if limit_price:
            order_data["limit_price"] = str(limit_price)
        if stop_price:
            order_data["stop_price"] = str(stop_price)
        
        data = self._request("POST", "/v2/orders", data=order_data)
        
        return Order(
            id=data.get("id", ""),
            symbol=data.get("symbol", ""),
            qty=float(data.get("qty", 0)),
            side=data.get("side", ""),
            type=data.get("type", ""),
            status=data.get("status", ""),
            filled_qty=float(data.get("filled_qty", 0)),
            filled_avg_price=float(data.get("filled_avg_price", 0)) if data.get("filled_avg_price") else None,
            submitted_at=data.get("submitted_at", ""),
            filled_at=data.get("filled_at"),
        )
    
    def submit_notional_order(
        self,
        symbol: str,
        notional: float,  # Dollar amount
        side: OrderSide,
    ) -> Order:
        """Submit order by dollar amount (fractional shares)."""
        order_data = {
            "symbol": symbol,
            "notional": str(notional),
            "side": side.value,
            "type": "market",
            "time_in_force": "day",
        }
        
        data = self._request("POST", "/v2/orders", data=order_data)
        
        return Order(
            id=data.get("id", ""),
            symbol=data.get("symbol", ""),
            qty=float(data.get("qty", 0)) if data.get("qty") else 0,
            side=data.get("side", ""),
            type=data.get("type", ""),
            status=data.get("status", ""),
            filled_qty=float(data.get("filled_qty", 0)),
            filled_avg_price=float(data.get("filled_avg_price", 0)) if data.get("filled_avg_price") else None,
            submitted_at=data.get("submitted_at", ""),
            filled_at=data.get("filled_at"),
        )
    
    def get_order(self, order_id: str) -> Order:
        """Get order by ID."""
        data = self._request("GET", f"/v2/orders/{order_id}")
        
        return Order(
            id=data.get("id", ""),
            symbol=data.get("symbol", ""),
            qty=float(data.get("qty", 0)),
            side=data.get("side", ""),
            type=data.get("type", ""),
            status=data.get("status", ""),
            filled_qty=float(data.get("filled_qty", 0)),
            filled_avg_price=float(data.get("filled_avg_price", 0)) if data.get("filled_avg_price") else None,
            submitted_at=data.get("submitted_at", ""),
            filled_at=data.get("filled_at"),
        )
    
    def get_orders(
        self,
        status: str = "open",
        limit: int = 50,
    ) -> List[Order]:
        """Get list of orders."""
        data = self._request("GET", "/v2/orders", params={"status": status, "limit": limit})
        
        orders = []
        for item in data:
            orders.append(Order(
                id=item.get("id", ""),
                symbol=item.get("symbol", ""),
                qty=float(item.get("qty", 0)),
                side=item.get("side", ""),
                type=item.get("type", ""),
                status=item.get("status", ""),
                filled_qty=float(item.get("filled_qty", 0)),
                filled_avg_price=float(item.get("filled_avg_price", 0)) if item.get("filled_avg_price") else None,
                submitted_at=item.get("submitted_at", ""),
                filled_at=item.get("filled_at"),
            ))
        
        return orders
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            self._request("DELETE", f"/v2/orders/{order_id}")
            return True
        except:
            return False
    
    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        data = self._request("DELETE", "/v2/orders")
        return len(data) if isinstance(data, list) else 0
    
    # ========== Market Data Methods ==========
    
    def get_latest_quote(self, symbol: str) -> Dict[str, float]:
        """Get latest quote for a symbol."""
        # Use data API endpoint
        data_url = "https://data.alpaca.markets"
        url = f"{data_url}/v2/stocks/{symbol}/quotes/latest"
        
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        
        quote = response.json().get("quote", {})
        return {
            "bid": float(quote.get("bp", 0)),
            "ask": float(quote.get("ap", 0)),
            "bid_size": int(quote.get("bs", 0)),
            "ask_size": int(quote.get("as", 0)),
            "timestamp": quote.get("t", ""),
        }
    
    def get_latest_bar(self, symbol: str) -> Dict[str, float]:
        """Get latest bar for a symbol."""
        data_url = "https://data.alpaca.markets"
        url = f"{data_url}/v2/stocks/{symbol}/bars/latest"
        
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        
        bar = response.json().get("bar", {})
        return {
            "open": float(bar.get("o", 0)),
            "high": float(bar.get("h", 0)),
            "low": float(bar.get("l", 0)),
            "close": float(bar.get("c", 0)),
            "volume": int(bar.get("v", 0)),
            "timestamp": bar.get("t", ""),
        }
    
    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get historical bars."""
        data_url = "https://data.alpaca.markets"
        url = f"{data_url}/v2/stocks/{symbol}/bars"
        
        params = {"timeframe": timeframe, "limit": limit}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        
        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()
        
        bars = response.json().get("bars", [])
        return [{
            "timestamp": b.get("t", ""),
            "open": float(b.get("o", 0)),
            "high": float(b.get("h", 0)),
            "low": float(b.get("l", 0)),
            "close": float(b.get("c", 0)),
            "volume": int(b.get("v", 0)),
        } for b in bars]
    
    # ========== Health Check ==========
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        try:
            account = self.get_account()
            clock = self.get_market_hours()
            positions = self.get_positions()
            
            return {
                "status": "healthy",
                "account_id": account.id,
                "cash": account.cash,
                "equity": account.equity,
                "portfolio_value": account.portfolio_value,
                "trading_blocked": account.trading_blocked,
                "market_open": clock["is_open"],
                "next_open": clock["next_open"],
                "next_close": clock["next_close"],
                "position_count": len(positions),
                "is_paper": self.is_paper,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


def test_connection():
    """Test Alpaca connection."""
    print("=" * 60)
    print("ALPACA CONNECTION TEST")
    print("=" * 60)
    
    try:
        client = AlpacaClient()
        health = client.health_check()
        
        if health["status"] == "healthy":
            print(f"\n✅ Connection successful!")
            print(f"\nAccount Details:")
            print(f"  Account ID: {health['account_id']}")
            print(f"  Mode: {'PAPER' if health['is_paper'] else 'LIVE'}")
            print(f"  Cash: ${health['cash']:,.2f}")
            print(f"  Equity: ${health['equity']:,.2f}")
            print(f"  Portfolio Value: ${health['portfolio_value']:,.2f}")
            print(f"  Trading Blocked: {health['trading_blocked']}")
            print(f"  Positions: {health['position_count']}")
            print(f"\nMarket Status:")
            print(f"  Currently Open: {health['market_open']}")
            print(f"  Next Open: {health['next_open']}")
            print(f"  Next Close: {health['next_close']}")
        else:
            print(f"\n❌ Connection failed: {health['error']}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    test_connection()

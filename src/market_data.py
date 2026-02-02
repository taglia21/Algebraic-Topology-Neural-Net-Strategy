"""Real-time market data fetcher using yfinance."""

import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class MarketDataFetcher:
    """Fetches real market prices."""
    
    def __init__(self):
        self.cache = {}  # Symbol -> (price, timestamp)
        self.cache_duration = 60  # Cache for 60 seconds
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol."""
        try:
            # Check cache first
            if symbol in self.cache:
                price, timestamp = self.cache[symbol]
                age = (datetime.now() - timestamp).total_seconds()
                if age < self.cache_duration:
                    return price
            
            # Fetch fresh data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            
            if data.empty:
                logger.warning(f"No data for {symbol}")
                return None
            
            # Get latest close price
            price = float(data['Close'].iloc[-1])
            self.cache[symbol] = (price, datetime.now())
            
            return price
            
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None
    
    def get_multiple_prices(self, symbols: list) -> Dict[str, float]:
        """Get prices for multiple symbols."""
        prices = {}
        for symbol in symbols:
            price = self.get_current_price(symbol)
            if price:
                prices[symbol] = price
        return prices

if __name__ == "__main__":
    fetcher = MarketDataFetcher()
    price = fetcher.get_current_price("AAPL")
    print(f"AAPL price: ${price:.2f}" if price else "Failed to fetch")

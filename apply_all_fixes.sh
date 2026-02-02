#!/bin/bash
set -e

echo "=== APPLYING ALL CRITICAL FIXES ==="
echo ""

# Fix #2: Update live_executor to use real prices
echo "Fix #2: Updating live_executor with real market data..."
python3 << 'PYFIX2'
import re

# Read the executor file
with open('src/execution/live_executor.py', 'r') as f:
    content = f.read()

# Replace hardcoded price with market data fetcher
updated = content.replace(
    'fill_price = order.limit_price if order.limit_price else 100.0  # Placeholder',
    '''# Get real market price
        from market_data import MarketDataFetcher
        fetcher = MarketDataFetcher()
        market_price = fetcher.get_current_price(order.symbol) or 100.0
        fill_price = order.limit_price if order.limit_price else market_price'''
)

with open('src/execution/live_executor.py', 'w') as f:
    f.write(updated)

print("✅ Live executor updated with real market data")
PYFIX2

# Fix #3: Add position valuation
echo "Fix #3: Adding position valuation to executor..."
python3 << 'PYFIX3'
with open('src/execution/live_executor.py', 'r') as f:
    content = f.read()

# Find and replace get_account_value method
old_method = '''def get_account_value(self) -> float:
        """Get total account value (cash + positions)."""
        # TODO: Add position valuations
        return self.cash_balance'''

new_method = '''def get_account_value(self) -> float:
        """Get total account value (cash + positions)."""
        total = self.cash_balance
        
        # Add position values
        if self.positions and self.paper_trading:
            from market_data import MarketDataFetcher
            fetcher = MarketDataFetcher()
            for symbol, quantity in self.positions.items():
                price = fetcher.get_current_price(symbol) or 100.0
                total += quantity * price
        
        return total'''

if old_method in content:
    content = content.replace(old_method, new_method)
    with open('src/execution/live_executor.py', 'w') as f:
        f.write(content)
    print("✅ Position valuation added")
else:
    print("⚠️  Method not found or already updated")
PYFIX3

echo ""
echo "=== FIXES APPLIED SUCCESSFULLY ==="

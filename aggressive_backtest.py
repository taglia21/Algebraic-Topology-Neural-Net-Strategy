import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Download SPY data
end = datetime.now()
start = end - timedelta(days=365)
df = yf.download('SPY', start=start, end=end, progress=False)
print(f"Downloaded {len(df)} bars")

# Simple momentum strategy
df['returns'] = df['Close'].pct_change()
df['sma_20'] = df['Close'].rolling(20).mean()
df['sma_50'] = df['Close'].rolling(50).mean()
df['signal'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)

# Backtest
capital = 100000
position = 0
trades = []
for i in range(50, len(df)):
    price = df['Close'].iloc[i]
    signal = df['signal'].iloc[i]
    
    if position == 0 and signal == 1:
        shares = capital * 0.1 / price
        position = shares
        entry = price
        trades.append({'type': 'buy', 'price': price})
    elif position > 0 and signal == -1:
        pnl = (price - entry) / entry
        trades.append({'type': 'sell', 'price': price, 'pnl': pnl})
        position = 0

# Results
wins = [t for t in trades if t.get('pnl', 0) > 0]
losses = [t for t in trades if t.get('pnl', 0) < 0]
total_trades = len([t for t in trades if 'pnl' in t])
win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
avg_win = np.mean([t['pnl'] for t in wins]) * 100 if wins else 0
avg_loss = np.mean([t['pnl'] for t in losses]) * 100 if losses else 0
total_return = sum([t.get('pnl', 0) for t in trades]) * 100

print(f"\nAGGRESSIVE BACKTEST RESULTS")
print(f"Total Trades: {total_trades}")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Avg Win: {avg_win:.2f}%")
print(f"Avg Loss: {avg_loss:.2f}%")
print(f"Total Return: {total_return:.2f}%")

#!/usr/bin/env python3
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys
sys.path.insert(0, 'src/strategy')
from tda_neural_strategy import TDANeuralStrategy

def download_data(symbol, start, end):
    print(f"Downloading {symbol} data...")
    df = yf.download(symbol, start=start, end=end, progress=False)
    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    return df

def backtest(strategy, df, initial_capital=100000):
    print("Running backtest...")
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [capital]
    prices = df['close'].values
    
    for i in range(strategy.lookback, len(prices)):
        current_price = prices[i]
        lookback_prices = prices[max(0, i-strategy.lookback):i]
        signal = strategy.generate_signal(lookback_prices, current_price)
        
        if position != 0:
            if position > 0:
                if current_price <= signal.stop_loss or current_price >= signal.take_profit or signal.direction != 1:
                    pnl = position * (current_price - entry_price)
                    capital += pnl
                    trades.append({'pnl': pnl})
                    position = 0
            elif position < 0:
                if current_price >= signal.stop_loss or current_price <= signal.take_profit or signal.direction != -1:
                    pnl = abs(position) * (entry_price - current_price)
                    capital += pnl
                    trades.append({'pnl': pnl})
                    position = 0
        
        if position == 0 and True and signal.confidence > 0.10:
            position_value = capital * signal.position_size
            shares = position_value / current_price
            if signal.direction == 1:
                position = shares
            else:
                position = -shares
            entry_price = current_price
        
        if position > 0:
            equity = capital + position * (current_price - entry_price)
        elif position < 0:
            equity = capital + abs(position) * (entry_price - current_price)
        else:
            equity = capital
        equity_curve.append(equity)
    
    equity_curve = np.array(equity_curve)
    returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-10)
    total_return = (equity_curve[-1] - initial_capital) / initial_capital
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
    max_dd = np.min(equity_curve / np.maximum.accumulate(equity_curve) - 1)
    winning = [t for t in trades if t.get('pnl', 0) > 0]
    win_rate = len(winning) / max(1, len(trades))
    
    return {'total_return': total_return, 'sharpe_ratio': sharpe, 'max_drawdown': max_dd,
            'num_trades': len(trades), 'win_rate': win_rate, 'final_capital': equity_curve[-1]}

def main():
    print("="*60)
    print("TDA + NEURAL NETWORK TRADING STRATEGY")
    print("="*60)
    
    symbol = "SPY"
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=3*365)).strftime("%Y-%m-%d")
    df = download_data(symbol, start_date, end_date)
    print(f"Downloaded {len(df)} days of data")
    
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    print(f"Training: {len(train_df)} days, Testing: {len(test_df)} days")
    
    strategy = TDANeuralStrategy(lookback=50)
    train_returns = train_df['close'].pct_change(5).shift(-5).fillna(0).values
    
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    strategy.train(train_df, train_returns, epochs=50)
    strategy.save()
    
    print("\n" + "="*60)
    print("OUT-OF-SAMPLE BACKTEST")
    print("="*60)
    results = backtest(strategy, test_df)
    print(f"Total Return: {results['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
    print(f"Win Rate: {results['win_rate']*100:.1f}%")
    print(f"Trades: {results['num_trades']}")
    
    buy_hold = (test_df['close'].iloc[-1] - test_df['close'].iloc[0]) / test_df['close'].iloc[0]
    print(f"\nBuy & Hold: {buy_hold*100:.2f}%")
    print(f"Alpha: {(results['total_return'] - buy_hold)*100:.2f}%")
    print("\nModel saved to: models/tda_neural_model.pt")

if __name__ == "__main__":
    main()

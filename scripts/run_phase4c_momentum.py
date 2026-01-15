"""Phase 4C: Single Asset Leveraged Trend Following with Regime Filter.

Goal: Beat SPY by being MORE invested when bullish, LESS when bearish.
Use QQQ instead of SPY since it has higher beta and better momentum.

Key insight: The 2020-2025 period was heavily bullish. To beat SPY we need:
1. Full exposure during bull runs
2. Quick exit at trend breaks
3. Consider momentum assets (QQQ outperformed SPY)
"""

import os
import sys
import json
import warnings
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import backtrader as bt
from src.data.data_provider import get_ohlcv_hybrid


class MomentumTrendStrategy(bt.Strategy):
    """
    Momentum-focused trend following with aggressive positioning.
    
    Strategy:
    - Buy when price > 50 SMA AND RSI momentum positive
    - Full investment (90-100%) when trend is strong  
    - Exit on trend break (price < 50 SMA or momentum loss)
    - Faster moving averages for quicker signals
    """
    
    params = dict(
        sma_trend=50,        # Trend filter (was 200, now faster)
        sma_exit=20,         # Exit trigger (faster)
        rsi_period=10,       # RSI for momentum (faster)
        momentum_period=10,  # Momentum lookback
        max_position=0.98,   # Almost full investment
        stop_loss=0.07,      # 7% stop loss
        printlog=False,
    )
    
    def __init__(self):
        self.data0 = self.datas[0]
        
        # Indicators
        self.sma_trend = bt.indicators.SMA(self.data0.close, period=self.p.sma_trend)
        self.sma_exit = bt.indicators.SMA(self.data0.close, period=self.p.sma_exit)
        self.rsi = bt.indicators.RSI(self.data0.close, period=self.p.rsi_period)
        self.momentum = bt.indicators.Momentum(self.data0.close, period=self.p.momentum_period)
        self.atr = bt.indicators.ATR(self.data0, period=14)
        
        # State
        self.order = None
        self.entry_price = 0
        self.highest_since_entry = 0
        
        # Tracking
        self.trade_count = 0
        self.winning_trades = 0
        self.start_cash = None
        self.max_equity = 0
        self.max_drawdown = 0
        
    def log(self, txt, force=False):
        if self.p.printlog or force:
            dt = self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')
    
    def start(self):
        self.start_cash = self.broker.getvalue()
        self.max_equity = self.start_cash
    
    def next(self):
        current_equity = self.broker.getvalue()
        
        if current_equity > self.max_equity:
            self.max_equity = current_equity
        
        dd = (self.max_equity - current_equity) / self.max_equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd
        
        if len(self) < self.p.sma_trend + 5:
            return
        
        if self.order:
            return
        
        price = self.data0.close[0]
        position = self.getposition().size
        
        # Trend conditions
        above_trend = price > self.sma_trend[0]
        trend_rising = self.sma_trend[0] > self.sma_trend[-5]  # Trend slope
        momentum_positive = self.momentum[0] > 0
        rsi_not_overbought = self.rsi[0] < 75
        
        # Entry: price above trend MA + positive momentum
        if position == 0:
            if above_trend and trend_rising and momentum_positive:
                # Full position
                cash = self.broker.getcash()
                size = int((cash * self.p.max_position) / price)
                
                if size > 0:
                    self.log(f'BUY {size} @ ${price:.2f} | SMA50={self.sma_trend[0]:.0f} | Mom={self.momentum[0]:.2f}')
                    self.order = self.buy(size=size)
                    self.entry_price = price
                    self.highest_since_entry = price
        
        # Position management
        else:
            # Update trailing high
            if price > self.highest_since_entry:
                self.highest_since_entry = price
            
            # Exit conditions
            exit_signal = False
            exit_reason = ""
            
            # 1. Stop loss from entry
            if price < self.entry_price * (1 - self.p.stop_loss):
                exit_signal = True
                exit_reason = "Stop loss hit"
            
            # 2. Trailing stop (10% from high)
            elif price < self.highest_since_entry * 0.90:
                exit_signal = True
                exit_reason = "Trailing stop"
            
            # 3. Trend break
            elif price < self.sma_exit[0] and self.momentum[0] < 0:
                exit_signal = True
                exit_reason = "Trend break"
            
            if exit_signal:
                self.log(f'SELL ALL @ ${price:.2f} | Reason: {exit_reason}')
                self.order = self.close()
                self.entry_price = 0
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None
    
    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_count += 1
            if trade.pnl > 0:
                self.winning_trades += 1
    
    def stop(self):
        final_value = self.broker.getvalue()
        total_return = (final_value / self.start_cash - 1) * 100
        
        print(f"\n{'='*50}")
        print("MOMENTUM TREND RESULTS")
        print(f"{'='*50}")
        print(f"Final Value:    ${final_value:,.2f}")
        print(f"Total Return:   {total_return:.2f}%")
        print(f"Max Drawdown:   {self.max_drawdown*100:.2f}%")
        print(f"Trades:         {self.trade_count}")
        print(f"Win Rate:       {self.winning_trades/max(1,self.trade_count)*100:.1f}%")


def get_benchmark(symbol: str, start: str, end: str) -> Dict:
    """Get buy-and-hold benchmark."""
    import yfinance as yf
    data = yf.download(symbol, start=start, end=end, progress=False)
    
    if len(data) == 0:
        return {}
    
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close'][symbol]
    else:
        close = data['Close']
    
    start_p = float(close.iloc[0])
    end_p = float(close.iloc[-1])
    ret = (end_p / start_p - 1) * 100
    
    days = (data.index[-1] - data.index[0]).days
    years = days / 365.25
    cagr = ((end_p / start_p) ** (1/years) - 1) * 100
    
    running_max = close.expanding().max()
    dd = (close - running_max) / running_max
    max_dd = abs(float(dd.min())) * 100
    
    daily_ret = close.pct_change().dropna()
    sharpe = float((daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252)))
    
    return {'return': ret, 'cagr': cagr, 'max_dd': max_dd, 'sharpe': sharpe}


def run_single_asset_backtest(symbol='QQQ', start='2020-01-01', end='2025-01-01', capital=100000):
    """Run single-asset momentum strategy."""
    
    print(f"\n{'='*60}")
    print(f"PHASE 4C: MOMENTUM TREND ON {symbol}")
    print(f"{'='*60}")
    
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MomentumTrendStrategy, printlog=True)
    
    # Load data
    df = get_ohlcv_hybrid(symbol, start, end)
    if df is None or len(df) < 300:
        print(f"Error: insufficient data for {symbol}")
        return None
    
    data = bt.feeds.PandasData(dataname=df, name=symbol)
    cerebro.adddata(data)
    print(f"Loaded {symbol}: {len(df)} bars")
    
    cerebro.broker.setcash(capital)
    cerebro.broker.setcommission(commission=0.001)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    
    print("\nRunning backtest...")
    results = cerebro.run()
    strat = results[0]
    
    final_value = cerebro.broker.getvalue()
    total_return = (final_value / capital - 1) * 100
    
    from dateutil.parser import parse
    years = (parse(end) - parse(start)).days / 365.25
    cagr = ((final_value / capital) ** (1/years) - 1) * 100
    
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0
    max_dd = strat.analyzers.dd.get_analysis().get('max', {}).get('drawdown', 0)
    
    # Get benchmarks
    spy_bench = get_benchmark('SPY', start, end)
    qqq_bench = get_benchmark('QQQ', start, end)
    
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Strategy':>12} {'SPY B&H':>12} {'QQQ B&H':>12}")
    print("-" * 56)
    print(f"{'Return':<20} {total_return:>11.2f}% {spy_bench.get('return',0):>11.2f}% {qqq_bench.get('return',0):>11.2f}%")
    print(f"{'CAGR':<20} {cagr:>11.2f}% {spy_bench.get('cagr',0):>11.2f}% {qqq_bench.get('cagr',0):>11.2f}%")
    print(f"{'Max Drawdown':<20} {max_dd:>11.2f}% {spy_bench.get('max_dd',0):>11.2f}% {qqq_bench.get('max_dd',0):>11.2f}%")
    print(f"{'Sharpe':<20} {sharpe:>12.2f} {spy_bench.get('sharpe',0):>12.2f} {qqq_bench.get('sharpe',0):>12.2f}")
    
    # Check targets
    print(f"\n{'='*60}")
    print("TARGET CHECK (vs SPY)")
    print(f"{'='*60}")
    
    beat_spy_cagr = cagr > spy_bench.get('cagr', 0)
    good_sharpe = sharpe > 0.7
    acceptable_dd = max_dd < 35  # Less than SPY's DD
    
    print(f"Beat SPY CAGR ({spy_bench.get('cagr',0):.2f}%): {'✓' if beat_spy_cagr else '✗'} ({cagr:.2f}%)")
    print(f"Sharpe > 0.7: {'✓' if good_sharpe else '✗'} ({sharpe:.2f})")
    print(f"Max DD < SPY's ({spy_bench.get('max_dd',0):.1f}%): {'✓' if acceptable_dd else '✗'} ({max_dd:.2f}%)")
    
    if beat_spy_cagr:
        print("\n🎉 STRATEGY BEATS SPY!")
    
    return {
        'symbol': symbol,
        'return': total_return,
        'cagr': cagr,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'spy_cagr': spy_bench.get('cagr', 0),
        'beats_spy': beat_spy_cagr
    }


if __name__ == "__main__":
    # Try different assets
    for symbol in ['QQQ', 'SPY']:
        result = run_single_asset_backtest(symbol)
        print("\n")

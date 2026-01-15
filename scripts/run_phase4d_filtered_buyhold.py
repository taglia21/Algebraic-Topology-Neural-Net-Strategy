"""Phase 4D: Buy & Hold with Downside Protection (200 SMA Filter).

Key insight from Phase 4C: QQQ B&H returned 143.86% vs our 16%.
The over-trading is the problem. 

New strategy: STAY INVESTED unless clear bear market signal.
- Hold QQQ (highest momentum) as long as price > 200 SMA
- Only exit when price closes below 200 SMA for 3 days
- Re-enter when price closes above 200 SMA

This is a simple "Trend Following" / "Time in Market" strategy.
Should capture most of the upside while avoiding major drawdowns.
"""

import os
import sys
import json
import warnings
from typing import Dict, List
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import backtrader as bt
from src.data.data_provider import get_ohlcv_hybrid


class BuyHoldWithFilter(bt.Strategy):
    """
    Buy & Hold with 200 SMA filter.
    
    - Enter when price > 200 SMA
    - Exit when price < 200 SMA for 3 consecutive days
    - This simple filter avoided the worst of 2022 drawdown
    """
    
    params = dict(
        sma_period=200,
        exit_days=3,  # Days below SMA to trigger exit
        entry_days=1, # Days above SMA to re-enter
        position_pct=0.98,
        printlog=True,
    )
    
    def __init__(self):
        self.sma = bt.indicators.SMA(self.datas[0].close, period=self.p.sma_period)
        
        self.order = None
        self.days_below_sma = 0
        self.days_above_sma = 0
        
        self.start_cash = None
        self.max_equity = 0
        self.max_drawdown = 0
        self.trade_count = 0
        
    def log(self, txt, force=False):
        if self.p.printlog or force:
            print(f'{self.datas[0].datetime.date(0)} {txt}')
    
    def start(self):
        self.start_cash = self.broker.getvalue()
        self.max_equity = self.start_cash
        self.log(f'Starting: ${self.start_cash:,.0f}', force=True)
    
    def next(self):
        equity = self.broker.getvalue()
        if equity > self.max_equity:
            self.max_equity = equity
        
        dd = (self.max_equity - equity) / self.max_equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd
        
        if len(self) < self.p.sma_period + 5:
            return
        
        if self.order:
            return
        
        price = self.datas[0].close[0]
        sma = self.sma[0]
        position = self.getposition().size
        
        # Track days above/below SMA
        if price > sma:
            self.days_above_sma += 1
            self.days_below_sma = 0
        else:
            self.days_below_sma += 1
            self.days_above_sma = 0
        
        # ENTRY: Price above SMA
        if position == 0 and self.days_above_sma >= self.p.entry_days:
            cash = self.broker.getcash()
            size = int((cash * self.p.position_pct) / price)
            if size > 0:
                self.log(f'BUY {size} @ ${price:.2f} | SMA200=${sma:.2f}')
                self.order = self.buy(size=size)
                self.trade_count += 1
        
        # EXIT: Price below SMA for N days
        elif position > 0 and self.days_below_sma >= self.p.exit_days:
            self.log(f'SELL ALL @ ${price:.2f} | Below SMA for {self.days_below_sma} days')
            self.order = self.close()
            self.trade_count += 1
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None
    
    def stop(self):
        final = self.broker.getvalue()
        ret = (final / self.start_cash - 1) * 100
        print(f"\n{'='*50}")
        print("BUY & HOLD WITH 200 SMA FILTER")
        print(f"{'='*50}")
        print(f"Final: ${final:,.2f}")
        print(f"Return: {ret:.2f}%")
        print(f"Max DD: {self.max_drawdown*100:.2f}%")
        print(f"Trades: {self.trade_count}")


def get_benchmark(symbol: str, start: str, end: str) -> Dict:
    """Get benchmark stats."""
    import yfinance as yf
    data = yf.download(symbol, start=start, end=end, progress=False)
    if len(data) == 0:
        return {}
    
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close'][symbol]
    else:
        close = data['Close']
    
    start_p, end_p = float(close.iloc[0]), float(close.iloc[-1])
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


def run_filtered_buyhold(symbol='QQQ', start='2020-01-01', end='2025-01-01', capital=100000):
    """Run Buy & Hold with SMA filter."""
    
    print(f"\n{'='*60}")
    print(f"PHASE 4D: BUY & HOLD WITH 200 SMA FILTER - {symbol}")
    print(f"{'='*60}")
    
    cerebro = bt.Cerebro()
    cerebro.addstrategy(BuyHoldWithFilter, printlog=True)
    
    df = get_ohlcv_hybrid(symbol, start, end)
    if df is None or len(df) < 250:
        print(f"Error: insufficient data")
        return None
    
    data = bt.feeds.PandasData(dataname=df, name=symbol)
    cerebro.adddata(data)
    print(f"Loaded: {len(df)} bars")
    
    cerebro.broker.setcash(capital)
    cerebro.broker.setcommission(commission=0.001)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    
    print("\nRunning...")
    results = cerebro.run()
    strat = results[0]
    
    final = cerebro.broker.getvalue()
    ret = (final / capital - 1) * 100
    
    from dateutil.parser import parse
    years = (parse(end) - parse(start)).days / 365.25
    cagr = ((final / capital) ** (1/years) - 1) * 100
    
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0
    max_dd = strat.analyzers.dd.get_analysis().get('max', {}).get('drawdown', 0)
    
    # Benchmarks
    spy = get_benchmark('SPY', start, end)
    qqq = get_benchmark(symbol, start, end)
    
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Strategy':>12} {'SPY B&H':>12} {f'{symbol} B&H':>12}")
    print("-" * 56)
    print(f"{'Return':<20} {ret:>11.2f}% {spy.get('return',0):>11.2f}% {qqq.get('return',0):>11.2f}%")
    print(f"{'CAGR':<20} {cagr:>11.2f}% {spy.get('cagr',0):>11.2f}% {qqq.get('cagr',0):>11.2f}%")
    print(f"{'Max Drawdown':<20} {max_dd:>11.2f}% {spy.get('max_dd',0):>11.2f}% {qqq.get('max_dd',0):>11.2f}%")
    print(f"{'Sharpe':<20} {sharpe:>12.2f} {spy.get('sharpe',0):>12.2f} {qqq.get('sharpe',0):>12.2f}")
    
    print(f"\n{'='*60}")
    print("TARGETS")
    print(f"{'='*60}")
    
    beat_spy = cagr > spy.get('cagr', 0)
    good_sharpe = sharpe > 0.7
    better_dd = max_dd < qqq.get('max_dd', 100)  # Less DD than underlying
    
    print(f"Beat SPY CAGR: {'✓' if beat_spy else '✗'} ({cagr:.2f}% vs {spy.get('cagr',0):.2f}%)")
    print(f"Sharpe > 0.7: {'✓' if good_sharpe else '✗'} ({sharpe:.2f})")
    print(f"DD < {symbol} B&H: {'✓' if better_dd else '✗'} ({max_dd:.1f}% vs {qqq.get('max_dd',0):.1f}%)")
    
    if beat_spy:
        print("\n🎉 BEATS SPY!")
    elif cagr > 14.83:
        print(f"\n✓ Exceeds 14.83% target CAGR")
    else:
        print(f"\n📉 Below targets")
    
    return {
        'symbol': symbol,
        'return': ret,
        'cagr': cagr,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'spy_cagr': spy.get('cagr', 0),
        'beats_spy': beat_spy
    }


if __name__ == "__main__":
    # Test on QQQ (best momentum) and SPY
    results = {}
    
    for symbol in ['QQQ', 'SPY']:
        r = run_filtered_buyhold(symbol)
        if r:
            results[symbol] = r
        print("\n")
    
    # Also test TQQQ (3x leveraged) for aggressive growth
    print("\n" + "="*60)
    print("BONUS: Testing 3x Leveraged TQQQ")
    print("="*60)
    r = run_filtered_buyhold('TQQQ')
    if r:
        results['TQQQ'] = r
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for sym, r in results.items():
        status = "✓ BEATS SPY" if r['beats_spy'] else "✗"
        print(f"{sym}: CAGR {r['cagr']:.2f}%, DD {r['max_dd']:.1f}%, Sharpe {r['sharpe']:.2f} {status}")
    
    # Save results
    with open('results/phase4d_results.json', 'w') as f:
        json.dump(results, f, indent=2)

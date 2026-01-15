"""Phase 4B: Simplified Aggressive Trend-Following Strategy.

Previous Phase 4 was over-engineered. This version is simpler and more aggressive:
1. Follow the trend (price > 200 SMA = long)
2. Use momentum for timing (ROC, RSI)
3. Fixed position sizing (no complex Kelly until we have history)
4. Full investment when bullish, cash when bearish

Target: Beat SPY's 14.83% CAGR.
"""

import os
import sys
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import backtrader as bt
from src.data.data_provider import get_ohlcv_hybrid


class AggressiveTrendStrategy(bt.Strategy):
    """
    Simplified Aggressive Trend-Following Strategy.
    
    Rules:
    - LONG when price > 200 SMA AND 50 SMA > 200 SMA
    - Add to position when RSI < 40 in uptrend (buy dips)
    - Exit to cash when price < 200 SMA or 50 < 200 SMA (death cross)
    - Use 80-100% of capital when bullish
    """
    
    params = dict(
        # Trend parameters
        sma_fast=50,
        sma_slow=200,
        # Momentum parameters  
        rsi_period=14,
        rsi_buy_threshold=45,
        rsi_sell_threshold=70,
        roc_period=20,
        # Position sizing
        max_position_pct=0.95,  # Use up to 95% of capital
        min_position_pct=0.30,  # Minimum 30% position when signal
        # Stop loss
        stop_loss_pct=0.08,     # 8% trailing stop
        printlog=False,
    )
    
    def __init__(self):
        # Track indicators for each data feed
        self.indicators = {}
        
        for i, data in enumerate(self.datas):
            self.indicators[data._name] = {
                'sma_fast': bt.indicators.SMA(data.close, period=self.p.sma_fast),
                'sma_slow': bt.indicators.SMA(data.close, period=self.p.sma_slow),
                'rsi': bt.indicators.RSI(data.close, period=self.p.rsi_period),
                'roc': bt.indicators.ROC(data.close, period=self.p.roc_period),
                'atr': bt.indicators.ATR(data, period=14),
            }
        
        # Track state
        self.order_dict = {d._name: None for d in self.datas}
        self.entry_prices = {d._name: 0 for d in self.datas}
        self.highest_prices = {d._name: 0 for d in self.datas}
        
        # Performance tracking
        self.trade_count = 0
        self.winning_trades = 0
        self.total_pnl = 0
        self.start_cash = None
        self.max_equity = 0
        self.max_drawdown = 0
        self.equity_curve = []
        
    def log(self, txt, dt=None, force=False):
        if self.p.printlog or force:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')
    
    def start(self):
        self.start_cash = self.broker.getvalue()
        self.max_equity = self.start_cash
        self.log(f'Starting cash: ${self.start_cash:,.2f}', force=True)
    
    def next(self):
        current_equity = self.broker.getvalue()
        self.equity_curve.append(current_equity)
        
        if current_equity > self.max_equity:
            self.max_equity = current_equity
        
        current_dd = (self.max_equity - current_equity) / self.max_equity
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
        
        # Need enough data for slow SMA
        if len(self) < self.p.sma_slow + 5:
            return
        
        # Process each asset
        for data in self.datas:
            self._process_asset(data)
    
    def _process_asset(self, data):
        """Process signals for single asset."""
        symbol = data._name
        ind = self.indicators[symbol]
        
        # Skip if pending order
        if self.order_dict[symbol]:
            return
        
        # Current values
        price = data.close[0]
        sma_fast = ind['sma_fast'][0]
        sma_slow = ind['sma_slow'][0]
        rsi = ind['rsi'][0]
        roc = ind['roc'][0]
        
        # Trend conditions
        price_above_slow = price > sma_slow
        golden_cross = sma_fast > sma_slow
        trend_bullish = price_above_slow and golden_cross
        
        # Death cross or price breakdown
        trend_bearish = (sma_fast < sma_slow) or (price < sma_slow * 0.97)
        
        # Current position
        current_pos = self.getposition(data).size
        current_value = self.broker.getvalue()
        
        # Update trailing stop tracking
        if current_pos > 0:
            if price > self.highest_prices[symbol]:
                self.highest_prices[symbol] = price
            
            # Check trailing stop
            stop_price = self.highest_prices[symbol] * (1 - self.p.stop_loss_pct)
            if price < stop_price:
                self.log(f'{symbol} TRAILING STOP triggered at ${price:.2f}')
                self.order_dict[symbol] = self.close(data)
                return
        
        # === ENTRY LOGIC ===
        if trend_bullish and current_pos == 0:
            # Strong trend entry
            position_pct = self.p.max_position_pct
            
            # Size down slightly if RSI is high
            if rsi > 60:
                position_pct *= 0.7
            
            # Calculate shares
            target_value = current_value * position_pct / len(self.datas)  # Split across assets
            shares = int(target_value / price)
            
            if shares > 0:
                self.log(f'{symbol} BUY {shares} @ ${price:.2f} | SMA50={sma_fast:.0f} > SMA200={sma_slow:.0f} | RSI={rsi:.0f}')
                self.order_dict[symbol] = self.buy(data=data, size=shares)
                self.entry_prices[symbol] = price
                self.highest_prices[symbol] = price
        
        # === ADD TO POSITION (buy dip in uptrend) ===
        elif trend_bullish and current_pos > 0:
            # Buy dip: RSI oversold in uptrend
            if rsi < self.p.rsi_buy_threshold and roc > -5:
                # Add 30% more if dip opportunity
                pos_value = current_pos * price
                available = current_value - pos_value
                add_value = available * 0.5  # Use 50% of remaining
                add_shares = int(add_value / price)
                
                if add_shares > 0:
                    self.log(f'{symbol} ADD {add_shares} @ ${price:.2f} | RSI dip={rsi:.0f}')
                    self.order_dict[symbol] = self.buy(data=data, size=add_shares)
        
        # === EXIT LOGIC ===
        elif trend_bearish and current_pos > 0:
            self.log(f'{symbol} SELL ALL @ ${price:.2f} | Trend turned bearish')
            self.order_dict[symbol] = self.close(data)
            self.entry_prices[symbol] = 0
            self.highest_prices[symbol] = 0
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            symbol = order.data._name
            self.order_dict[symbol] = None
    
    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_count += 1
            if trade.pnl > 0:
                self.winning_trades += 1
            self.total_pnl += trade.pnl
    
    def stop(self):
        final_value = self.broker.getvalue()
        total_return = (final_value / self.start_cash - 1) * 100
        
        if len(self.equity_curve) > 252:
            years = len(self.equity_curve) / 252
            cagr = ((final_value / self.start_cash) ** (1/years) - 1) * 100
        else:
            cagr = total_return
        
        trades_per_year = self.trade_count / max(1, len(self.equity_curve) / 252)
        win_rate = self.winning_trades / max(1, self.trade_count) * 100
        
        print("\n" + "=" * 60)
        print("PHASE 4B AGGRESSIVE TREND RESULTS")
        print("=" * 60)
        print(f"Starting Capital:    ${self.start_cash:,.2f}")
        print(f"Final Value:         ${final_value:,.2f}")
        print(f"Total Return:        {total_return:.2f}%")
        print(f"CAGR:                {cagr:.2f}%")
        print(f"Max Drawdown:        {self.max_drawdown*100:.2f}%")
        print(f"Total Trades:        {self.trade_count}")
        print(f"Trades/Year:         {trades_per_year:.0f}")
        print(f"Win Rate:            {win_rate:.1f}%")
        print(f"Total PnL:           ${self.total_pnl:,.2f}")


def get_spy_benchmark(start_date: str, end_date: str) -> Dict:
    """Get SPY buy-and-hold benchmark."""
    import yfinance as yf
    
    spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
    
    if len(spy) == 0:
        return {}
    
    if isinstance(spy.columns, pd.MultiIndex):
        close = spy['Close']['SPY']
    else:
        close = spy['Close']
    
    start_price = float(close.iloc[0])
    end_price = float(close.iloc[-1])
    total_return = (end_price / start_price - 1) * 100
    
    days = (spy.index[-1] - spy.index[0]).days
    years = days / 365.25
    cagr = ((end_price / start_price) ** (1/years) - 1) * 100
    
    running_max = close.expanding().max()
    drawdowns = (close - running_max) / running_max
    max_dd = abs(float(drawdowns.min())) * 100
    
    daily_returns = close.pct_change().dropna()
    sharpe = float((daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)))
    
    return {
        'total_return': float(total_return),
        'cagr': float(cagr),
        'max_drawdown': float(max_dd),
        'sharpe': float(sharpe)
    }


def run_phase4b_backtest(
    symbols: List[str] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2025-01-01",
    initial_capital: float = 100000
) -> Dict:
    """Run Phase 4B simplified aggressive backtest."""
    
    if symbols is None:
        symbols = ['SPY', 'QQQ', 'IWM']
    
    print(f"\n{'='*60}")
    print("PHASE 4B: SIMPLIFIED AGGRESSIVE TREND-FOLLOWING")
    print(f"{'='*60}")
    print(f"Symbols: {symbols}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Capital: ${initial_capital:,.2f}")
    print(f"{'='*60}\n")
    
    cerebro = bt.Cerebro()
    cerebro.addstrategy(AggressiveTrendStrategy, printlog=False)
    
    # Load data
    for symbol in symbols:
        try:
            df = get_ohlcv_hybrid(symbol, start_date, end_date)
            if df is None or len(df) < 300:
                print(f"Skipping {symbol}: insufficient data")
                continue
            
            data = bt.feeds.PandasData(
                dataname=df,
                name=symbol
            )
            cerebro.adddata(data)
            print(f"Loaded {symbol}: {len(df)} bars")
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
    
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=0.001)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    print("\nRunning backtest...")
    results = cerebro.run()
    strat = results[0]
    
    # Get results
    final_value = cerebro.broker.getvalue()
    total_return = (final_value / initial_capital - 1) * 100
    
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0
    max_dd = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0) / 100
    
    from dateutil.parser import parse
    years = (parse(end_date) - parse(start_date)).days / 365.25
    cagr = ((final_value / initial_capital) ** (1/years) - 1) * 100
    
    # Get SPY benchmark
    print("\nFetching SPY benchmark...")
    spy = get_spy_benchmark(start_date, end_date)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("PHASE 4B vs SPY COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'Strategy':>15} {'SPY':>15}")
    print("-" * 55)
    print(f"{'Total Return':<25} {total_return:>14.2f}% {spy.get('total_return', 0):>14.2f}%")
    print(f"{'CAGR':<25} {cagr:>14.2f}% {spy.get('cagr', 0):>14.2f}%")
    print(f"{'Max Drawdown':<25} {max_dd*100:>14.2f}% {spy.get('max_drawdown', 0):>14.2f}%")
    print(f"{'Sharpe Ratio':<25} {sharpe:>15.2f} {spy.get('sharpe', 0):>15.2f}")
    
    # Target check
    print("\n" + "-" * 55)
    print("TARGET CHECK:")
    targets_met = 0
    
    if cagr > 14.83:
        print(f"  ✓ CAGR > 14.83%: {cagr:.2f}%")
        targets_met += 1
    else:
        print(f"  ✗ CAGR > 14.83%: {cagr:.2f}%")
    
    if sharpe > 1.0:
        print(f"  ✓ Sharpe > 1.0: {sharpe:.2f}")
        targets_met += 1
    else:
        print(f"  ✗ Sharpe > 1.0: {sharpe:.2f}")
    
    if max_dd * 100 < 8:
        print(f"  ✓ Max DD < 8%: {max_dd*100:.2f}%")
        targets_met += 1
    else:
        print(f"  ✗ Max DD < 8%: {max_dd*100:.2f}%")
    
    print(f"\nTargets met: {targets_met}/3")
    
    if cagr > spy.get('cagr', 0):
        print("🎉 Strategy BEATS SPY!")
    else:
        print("📉 Strategy underperforms SPY")
    
    # Save results
    results_dict = {
        'phase': '4B',
        'final_value': final_value,
        'total_return': total_return,
        'cagr': cagr,
        'max_drawdown': max_dd * 100,
        'sharpe': sharpe,
        'spy_cagr': spy.get('cagr', 0),
        'beats_spy': cagr > spy.get('cagr', 0)
    }
    
    with open('results/phase4b_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    return results_dict


if __name__ == "__main__":
    run_phase4b_backtest()

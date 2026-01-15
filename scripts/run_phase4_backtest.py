"""Phase 4 Aggressive Optimization Backtest.

Goal: Beat SPY's 14.83% CAGR with max 8% drawdown.

Changes from Phase 3:
1. Kelly Criterion position sizing (Half-Kelly)
2. Trend-following overlay (follow strong trends)
3. Momentum features (ROC, MACD, volume)
4. Dynamic cash allocation (max 20% cash normally)
5. Volatility targeting (12% target annual vol)

Target metrics:
- CAGR > 14.83%
- Sharpe > 1.0
- Max DD < 8%
- 50+ trades/year
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

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import backtrader as bt

from src.data.data_provider import get_ohlcv_hybrid
from src.tda_features import TDAFeatureGenerator
from src.regime_detector import MarketRegimeDetector
from src.kelly_position_sizer import (
    KellyPositionSizer, 
    DynamicCashAllocator,
    VolatilityTargeting
)
from src.trend_following import (
    TrendFollowingOverlay,
    MomentumFeatureGenerator,
    DualModeStrategy,
    TrendMode
)


class Phase4Strategy(bt.Strategy):
    """
    Phase 4 Aggressive TDA+NN Strategy with Trend Following.
    
    Key differences from Phase 3:
    - Kelly position sizing instead of fixed fractional
    - Trend-following mode when ADX > 25
    - Maximum 20% cash in normal conditions
    - Volatility targeting for consistent risk
    """
    
    params = dict(
        lookback=50,
        min_signal_threshold=0.15,  # Lowered to take more trades
        max_position_pct=0.70,     # Increased to 70% max per asset
        min_position_pct=0.20,     # Minimum 20% position
        target_volatility=0.15,    # 15% annual volatility target (more aggressive)
        max_cash_pct=0.15,         # Maximum 15% cash normally
        kelly_fraction=0.6,        # 60% Kelly (slightly higher than half)
        printlog=True,
        # Trade cost params
        commission_pct=0.001,
        slippage_pct=0.0005,
    )
    
    def __init__(self):
        # Initialize components
        self.tda_extractor = TDAFeatureGenerator()
        self.regime_detector = MarketRegimeDetector()
        self.trend_overlay = TrendFollowingOverlay()
        self.momentum_gen = MomentumFeatureGenerator()
        self.dual_mode = DualModeStrategy()
        
        # Position sizing
        self.kelly_sizer = KellyPositionSizer(
            min_position_pct=self.p.min_position_pct,
            max_position_pct=self.p.max_position_pct,
            kelly_fraction=self.p.kelly_fraction,
            target_volatility=self.p.target_volatility
        )
        self.cash_allocator = DynamicCashAllocator(
            max_cash_normal=self.p.max_cash_pct
        )
        self.vol_targeting = VolatilityTargeting(
            target_volatility=self.p.target_volatility
        )
        
        # Track signals for each data feed
        self.signals = {d._name: 0 for d in self.datas}
        self.positions_history = []
        
        # Trade tracking
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.trade_returns = []
        
        # Performance tracking
        self.equity_curve = []
        self.start_cash = None
        self.max_equity = 0
        self.max_drawdown = 0
        
    def log(self, txt, dt=None, force=False):
        """Logging function."""
        if self.p.printlog or force:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')
    
    def start(self):
        """Called at strategy start."""
        self.start_cash = self.broker.getvalue()
        self.max_equity = self.start_cash
        self.log(f'Starting cash: ${self.start_cash:,.2f}', force=True)
    
    def next(self):
        """Main strategy logic - called on each bar."""
        current_equity = self.broker.getvalue()
        self.equity_curve.append(current_equity)
        
        # Update max equity and drawdown
        if current_equity > self.max_equity:
            self.max_equity = current_equity
        
        current_dd = (self.max_equity - current_equity) / self.max_equity
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
        
        # Need enough data for analysis
        if len(self) < self.p.lookback + 200:  # Need 200 for long MA
            return
        
        # Process each asset
        for data in self.datas:
            self._process_asset(data)
    
    def _process_asset(self, data):
        """Process signals and manage position for single asset."""
        symbol = data._name
        
        # Build price DataFrame for analysis
        df = self._build_dataframe(data)
        if df is None or len(df) < 250:
            return
        
        # Get trend analysis
        try:
            trend = self.trend_overlay.analyze_trend(df)
        except Exception as e:
            self.log(f"{symbol} trend analysis error: {e}")
            return
        
        # Generate TDA features
        try:
            close_values = df['Close'].values[-self.p.lookback:]
            log_returns = np.diff(np.log(close_values + 1e-10))
            tda_features = self.tda_extractor.compute_persistence_features(log_returns)
        except Exception as e:
            self.log(f"{symbol} TDA error: {e}")
            return
        
        # Get regime
        try:
            regime = self.regime_detector.detect_regime(df)
        except:
            regime = 'normal'
        
        # Calculate current volatility
        returns = df['Close'].pct_change().dropna()
        current_vol = returns.iloc[-20:].std() * np.sqrt(252)  # Annualized
        
        # Generate base signal from TDA features
        base_signal = self._calculate_tda_signal(tda_features, df)
        
        # Combine with trend overlay
        combined_signal, signal_reason = self.dual_mode.combine_signals(base_signal, df)
        
        # Store signal
        self.signals[symbol] = combined_signal
        
        # Get position sizing
        position_size = self._calculate_position_size(
            combined_signal, current_vol, trend, regime
        )
        
        # Execute trade logic
        self._execute_trade(data, symbol, combined_signal, position_size, signal_reason, trend)
    
    def _build_dataframe(self, data) -> Optional[pd.DataFrame]:
        """Build DataFrame from backtrader data feed."""
        try:
            length = min(len(data), 500)  # Last 500 bars
            
            df = pd.DataFrame({
                'Open': [data.open[-i] for i in range(length-1, -1, -1)],
                'High': [data.high[-i] for i in range(length-1, -1, -1)],
                'Low': [data.low[-i] for i in range(length-1, -1, -1)],
                'Close': [data.close[-i] for i in range(length-1, -1, -1)],
                'Volume': [data.volume[-i] for i in range(length-1, -1, -1)],
            })
            
            return df
        except Exception as e:
            return None
    
    def _calculate_tda_signal(self, tda_features: Dict, df: pd.DataFrame) -> float:
        """Calculate signal from TDA features and price action."""
        signal = 0.0
        
        # Use actual TDA v1.3 features
        persistence_l0 = tda_features.get('persistence_l0', 0)
        persistence_l1 = tda_features.get('persistence_l1', 0)
        entropy_l0 = tda_features.get('entropy_l0', 1)
        entropy_l1 = tda_features.get('entropy_l1', 1)
        betti_0 = tda_features.get('betti_0', 0)
        betti_1 = tda_features.get('betti_1', 0)
        
        # Persistence strength indicates stable patterns
        persistence_strength = np.sqrt(persistence_l0**2 + persistence_l1**2)
        if persistence_strength > 0.5:
            signal += 0.15
        
        # Low entropy = more predictable = stronger signal
        avg_entropy = (entropy_l0 + entropy_l1) / 2
        if avg_entropy < 1.0:
            signal += 0.1
        
        # NOW THE KEY PART: Aggressive momentum-based signal generation
        close = df['Close']
        
        # Rate of change signals
        roc_5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100
        roc_10 = (close.iloc[-1] / close.iloc[-11] - 1) * 100
        roc_20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100
        
        # Strong bullish momentum
        if roc_5 > 1 and roc_10 > 0:
            signal += 0.35
        elif roc_5 > 0 and roc_10 > 0 and roc_20 > 0:
            signal += 0.25
        # Strong bearish momentum  
        elif roc_5 < -1 and roc_10 < 0:
            signal -= 0.25
        
        # Moving average trend (simple but effective)
        ma_20 = close.rolling(20).mean().iloc[-1]
        ma_50 = close.rolling(50).mean().iloc[-1]
        ma_200 = close.rolling(200).mean().iloc[-1]
        current_price = close.iloc[-1]
        
        # Price above all MAs = strong bull
        if current_price > ma_20 > ma_50 > ma_200:
            signal += 0.3
        # Price above 200 MA = medium bull
        elif current_price > ma_200:
            signal += 0.15
        # Price below all MAs = defensive
        elif current_price < ma_20 < ma_50 < ma_200:
            signal -= 0.3
        
        # RSI mean reversion layer
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta).where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Oversold in uptrend = buy
        if current_rsi < 35 and current_price > ma_200:
            signal += 0.2
        # Overbought in downtrend = reduce
        elif current_rsi > 75 and current_price < ma_200:
            signal -= 0.15
        
        # Volume confirmation
        volume = df['Volume']
        avg_volume = volume.rolling(20).mean().iloc[-1]
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # High volume confirms direction
        if volume_ratio > 1.5 and signal > 0:
            signal *= 1.2
        
        # Normalize to -1 to 1
        signal = np.clip(signal, -1.0, 1.0)
        
        return signal
    
    def _calculate_position_size(
        self, 
        signal: float, 
        volatility: float,
        trend,  # TrendSignal
        regime: str
    ) -> float:
        """Calculate position size using Kelly and volatility targeting."""
        # Get Kelly-based position
        kelly_position, kelly_reason = self.kelly_sizer.get_position_size(
            current_volatility=volatility,
            signal_strength=abs(signal),
            regime=regime
        )
        
        # Get volatility scalar
        vol_scalar, vol_reason = self.vol_targeting.calculate_exposure_scalar(volatility)
        
        # Get trend bias
        trend_bias, trend_reason = self.trend_overlay.get_trend_position_bias(trend)
        
        # Combine
        final_position = kelly_position * vol_scalar * trend_bias
        
        # Scale by signal strength
        final_position *= abs(signal)
        
        # Constrain
        final_position = np.clip(
            final_position, 
            0, 
            self.p.max_position_pct
        )
        
        return final_position
    
    def _execute_trade(
        self, 
        data, 
        symbol: str, 
        signal: float, 
        position_size: float,
        signal_reason: str,
        trend
    ):
        """Execute trade based on signal and position size."""
        current_position = self.getposition(data).size
        current_value = self.broker.getvalue()
        
        # Calculate target position
        if abs(signal) < self.p.min_signal_threshold:
            target_position = 0
        else:
            # Target in shares
            price = data.close[0]
            target_value = current_value * position_size
            
            if signal > 0:
                target_position = int(target_value / price)
            else:
                target_position = 0  # No shorting for now
        
        # Check if we need to trade
        position_diff = target_position - current_position
        
        if abs(position_diff) > 0:
            if position_diff > 0:
                # Buy
                self.buy(data=data, size=position_diff)
                action = "BUY"
            else:
                # Sell
                self.sell(data=data, size=abs(position_diff))
                action = "SELL"
            
            self.log(
                f'{symbol} {action} {abs(position_diff)} shares @ ${data.close[0]:.2f} | '
                f'Signal: {signal:.2f} | Trend: {trend.mode.value} | '
                f'Position size: {position_size:.1%}'
            )
    
    def notify_trade(self, trade):
        """Called when a trade is closed."""
        if trade.isclosed:
            self.trade_count += 1
            
            # Calculate return safely
            try:
                if trade.price > 0 and abs(trade.size) > 0:
                    trade_return = trade.pnl / (trade.price * abs(trade.size))
                else:
                    trade_return = 0
            except (ZeroDivisionError, ValueError):
                trade_return = 0
            
            self.trade_returns.append(trade_return)
            self.kelly_sizer.add_trade_result(trade_return)
            
            if trade.pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            self.total_pnl += trade.pnl
            
            self.log(
                f'Trade closed: PnL ${trade.pnl:.2f} | '
                f'Total trades: {self.trade_count} | '
                f'Win rate: {self.winning_trades/max(1,self.trade_count):.1%}'
            )
    
    def stop(self):
        """Called at end of backtest."""
        final_value = self.broker.getvalue()
        total_return = (final_value / self.start_cash - 1) * 100
        
        # Calculate metrics
        if len(self.equity_curve) > 252:
            years = len(self.equity_curve) / 252
            cagr = ((final_value / self.start_cash) ** (1/years) - 1) * 100
        else:
            cagr = total_return
        
        # Sharpe ratio
        if len(self.trade_returns) > 0:
            avg_return = np.mean(self.trade_returns)
            std_return = np.std(self.trade_returns) if len(self.trade_returns) > 1 else 0.01
            sharpe = (avg_return * np.sqrt(252)) / (std_return + 1e-10)
        else:
            sharpe = 0
        
        trades_per_year = self.trade_count / max(1, len(self.equity_curve) / 252)
        
        print("\n" + "=" * 60)
        print("PHASE 4 STRATEGY RESULTS")
        print("=" * 60)
        print(f"Starting Capital:    ${self.start_cash:,.2f}")
        print(f"Final Value:         ${final_value:,.2f}")
        print(f"Total Return:        {total_return:.2f}%")
        print(f"CAGR:                {cagr:.2f}%")
        print(f"Max Drawdown:        {self.max_drawdown*100:.2f}%")
        print(f"Sharpe Ratio:        {sharpe:.2f}")
        print(f"Total Trades:        {self.trade_count}")
        print(f"Trades/Year:         {trades_per_year:.0f}")
        print(f"Win Rate:            {self.winning_trades/max(1,self.trade_count)*100:.1f}%")
        print(f"Total PnL:           ${self.total_pnl:,.2f}")
        
        # Compare to targets
        print("\n" + "-" * 60)
        print("PHASE 4 TARGET COMPARISON")
        print("-" * 60)
        print(f"CAGR Target:         > 14.83% (SPY) → {'✓ PASS' if cagr > 14.83 else '✗ FAIL'} ({cagr:.2f}%)")
        print(f"Sharpe Target:       > 1.0 → {'✓ PASS' if sharpe > 1.0 else '✗ FAIL'} ({sharpe:.2f})")
        print(f"Max DD Target:       < 8% → {'✓ PASS' if self.max_drawdown < 0.08 else '✗ FAIL'} ({self.max_drawdown*100:.2f}%)")
        print(f"Trades/Year Target:  > 50 → {'✓ PASS' if trades_per_year > 50 else '✗ FAIL'} ({trades_per_year:.0f})")


def get_spy_benchmark(start_date: str, end_date: str) -> Dict:
    """Get SPY buy-and-hold benchmark performance."""
    import yfinance as yf
    
    spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
    
    if len(spy) == 0:
        return {'error': 'Could not download SPY data'}
    
    # Handle both single ticker and multi-ticker column formats
    if isinstance(spy.columns, pd.MultiIndex):
        close = spy['Close']['SPY']
    else:
        close = spy['Close']
    
    start_price = float(close.iloc[0])
    end_price = float(close.iloc[-1])
    total_return = (end_price / start_price - 1) * 100
    
    # Calculate years
    days = (spy.index[-1] - spy.index[0]).days
    years = days / 365.25
    
    cagr = ((end_price / start_price) ** (1/years) - 1) * 100
    
    # Max drawdown
    running_max = close.expanding().max()
    drawdowns = (close - running_max) / running_max
    max_dd = abs(float(drawdowns.min())) * 100
    
    # Sharpe
    daily_returns = close.pct_change().dropna()
    sharpe = float((daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)))
    
    return {
        'total_return': float(total_return),
        'cagr': float(cagr),
        'max_drawdown': float(max_dd),
        'sharpe': float(sharpe),
        'years': float(years)
    }


def run_phase4_backtest(
    symbols: List[str] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2025-01-01",
    initial_capital: float = 100000,
    printlog: bool = True
) -> Dict:
    """
    Run Phase 4 aggressive backtest.
    
    Args:
        symbols: List of symbols to trade
        start_date: Start date
        end_date: End date
        initial_capital: Starting capital
        printlog: Whether to print trade logs
        
    Returns:
        Dict with results
    """
    if symbols is None:
        symbols = ['SPY', 'QQQ', 'IWM']
    
    print(f"\n{'='*60}")
    print("PHASE 4 AGGRESSIVE OPTIMIZATION BACKTEST")
    print(f"{'='*60}")
    print(f"Symbols: {symbols}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"{'='*60}\n")
    
    # Create Cerebro
    cerebro = bt.Cerebro()
    
    # Add strategy
    cerebro.addstrategy(
        Phase4Strategy,
        printlog=printlog
    )
    
    # Add data feeds
    for symbol in symbols:
        try:
            df = get_ohlcv_hybrid(symbol, start_date, end_date)
            
            if df is None or len(df) < 252:
                print(f"Warning: Insufficient data for {symbol}, skipping")
                continue
            
            # Convert to backtrader format
            data = bt.feeds.PandasData(
                dataname=df,
                name=symbol,
                datetime=None,  # Use index
                open='Open',
                high='High',
                low='Low',
                close='Close',
                volume='Volume',
                openinterest=-1
            )
            
            cerebro.adddata(data)
            print(f"Loaded {symbol}: {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
            
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
    
    # Set broker parameters
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=0.001)  # 0.1%
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # Run backtest
    print("\nRunning Phase 4 backtest...")
    results = cerebro.run()
    strat = results[0]
    
    # Get analyzer results
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    drawdown_analysis = strat.analyzers.drawdown.get_analysis()
    trades_analysis = strat.analyzers.trades.get_analysis()
    returns_analysis = strat.analyzers.returns.get_analysis()
    
    # Extract metrics
    sharpe = sharpe_analysis.get('sharperatio', 0) or 0
    max_dd = drawdown_analysis.get('max', {}).get('drawdown', 0) / 100
    total_trades = trades_analysis.get('total', {}).get('total', 0)
    
    final_value = cerebro.broker.getvalue()
    total_return = (final_value / initial_capital - 1) * 100
    
    # Calculate CAGR
    from dateutil.parser import parse
    start_dt = parse(start_date)
    end_dt = parse(end_date)
    years = (end_dt - start_dt).days / 365.25
    cagr = ((final_value / initial_capital) ** (1/years) - 1) * 100
    
    trades_per_year = total_trades / years
    
    # Get SPY benchmark
    print("\nFetching SPY benchmark...")
    spy_benchmark = get_spy_benchmark(start_date, end_date)
    
    # Build results
    results_dict = {
        'phase': 'Phase 4 - Aggressive Optimization',
        'symbols': symbols,
        'start_date': start_date,
        'end_date': end_date,
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return_pct': total_return,
        'cagr_pct': cagr,
        'max_drawdown_pct': max_dd * 100,
        'sharpe_ratio': sharpe,
        'total_trades': total_trades,
        'trades_per_year': trades_per_year,
        'spy_benchmark': spy_benchmark,
        'targets': {
            'cagr_target': 14.83,
            'sharpe_target': 1.0,
            'max_dd_target': 8.0,
            'trades_per_year_target': 50
        },
        'passes': {
            'cagr': cagr > 14.83,
            'sharpe': sharpe > 1.0,
            'max_dd': max_dd * 100 < 8.0,
            'trades_per_year': trades_per_year > 50
        }
    }
    
    # Print comparison
    print("\n" + "=" * 60)
    print("PHASE 4 vs SPY BENCHMARK")
    print("=" * 60)
    print(f"{'Metric':<25} {'Phase 4':>15} {'SPY':>15} {'Target':>15}")
    print("-" * 70)
    print(f"{'Total Return':<25} {total_return:>14.2f}% {spy_benchmark.get('total_return', 0):>14.2f}%")
    print(f"{'CAGR':<25} {cagr:>14.2f}% {spy_benchmark.get('cagr', 0):>14.2f}% {'>14.83%':>15}")
    print(f"{'Sharpe Ratio':<25} {sharpe:>15.2f} {spy_benchmark.get('sharpe', 0):>15.2f} {'>1.0':>15}")
    print(f"{'Max Drawdown':<25} {max_dd*100:>14.2f}% {spy_benchmark.get('max_drawdown', 0):>14.2f}% {'<8%':>15}")
    print(f"{'Trades/Year':<25} {trades_per_year:>15.0f} {'N/A':>15} {'>50':>15}")
    
    # Save results
    results_path = 'results/phase4_backtest_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    
    return results_dict


if __name__ == "__main__":
    # Run Phase 4 backtest
    results = run_phase4_backtest(
        symbols=['SPY', 'QQQ', 'IWM'],
        start_date="2020-01-01",
        end_date="2025-01-01",
        initial_capital=100000,
        printlog=False  # Reduce output noise
    )
    
    print("\n" + "=" * 60)
    print("PHASE 4 OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    passes = results['passes']
    all_pass = all(passes.values())
    
    for metric, passed in passes.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {metric}: {status}")
    
    print("\n" + "-" * 60)
    if all_pass:
        print("🎉 PHASE 4 TARGETS MET - Strategy beats SPY!")
        print("   Recommended for deployment with $2K minimum.")
    else:
        print("⚠️  PHASE 4 TARGETS NOT MET - Further optimization needed.")
        print("   Consider: increasing aggressiveness, adding more assets,")
        print("   or relaxing position sizing constraints.")

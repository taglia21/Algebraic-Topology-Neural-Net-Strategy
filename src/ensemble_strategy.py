"""Ensemble Strategy combining TDA regime filter with LSTM predictor for Backtrader."""

import numpy as np
import pandas as pd
import backtrader as bt


class EnsembleStrategy(bt.Strategy):
    """Long-only strategy using TDA turbulence gating and NN signal threshold."""

    params = (
        ('nn_model', None),
        ('tda_generator', None),
        ('preprocessor', None),
        ('sequence_length', 20),
        ('buy_threshold', 0.52),
        ('sell_threshold', 0.48),
        ('max_position_pct', 0.25),
        ('tda_scale_min', 0.3),
        ('tda_scale_max', 1.0),
        ('verbose', False),
    )

    def __init__(self):
        """Initialize strategy with data buffers and tracking variables."""
        self.order = None
        self.price_history = []
        self.volume_history = []
        self.tda_history = []
        self.trade_log = []
        self.bar_count = 0

    def log(self, msg: str):
        """Log message with current datetime if verbose mode enabled."""
        if self.params.verbose:
            dt = self.datas[0].datetime.date(0)
            print(f'{dt}: {msg}')

    def notify_order(self, order):
        """Handle order status notifications."""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY @ {order.executed.price:.2f}')
            else:
                self.log(f'SELL @ {order.executed.price:.2f}')
        self.order = None

    def notify_trade(self, trade):
        """Track completed trades for performance metrics."""
        if trade.isclosed:
            self.trade_log.append({
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,
                'size': trade.size,
                'price': trade.price
            })

    def next(self):
        """Execute strategy logic on each bar."""
        self.bar_count += 1
        
        self._update_history()
        
        if len(self.price_history) < self.params.sequence_length + 25:
            return
        
        if self.order:
            return
        
        nn_signal = self._get_nn_signal()
        turbulence = self._get_turbulence_index()
        position_scale = self._calculate_position_scale(turbulence)
        
        self._execute_trading_logic(nn_signal, position_scale)

    def _update_history(self):
        """Update price and volume history buffers."""
        self.price_history.append({
            'open': self.data.open[0],
            'high': self.data.high[0],
            'low': self.data.low[0],
            'close': self.data.close[0],
            'volume': self.data.volume[0]
        })

    def _get_nn_signal(self) -> float:
        """Get neural network prediction signal."""
        if self.params.nn_model is None:
            return 0.5
        
        try:
            import pandas as pd
            import tensorflow as tf
            
            recent = self.price_history[-(self.params.sequence_length + 25):]
            ohlcv_df = pd.DataFrame(recent)
            
            tda_features = self.params.tda_generator.generate_features(ohlcv_df)
            
            if len(tda_features) < self.params.sequence_length:
                return 0.5
            
            X, _ = self.params.preprocessor.prepare_sequences(ohlcv_df, tda_features)
            
            if len(X) == 0:
                return 0.5
            
            prediction = self.params.nn_model(X[-1:], training=False)
            return float(prediction[0, 0])
            
        except Exception:
            return 0.5

    def _get_turbulence_index(self) -> float:
        """Calculate current TDA turbulence index."""
        if self.params.tda_generator is None or len(self.price_history) < 35:
            return 0.5
        
        try:
            import pandas as pd
            
            recent = self.price_history[-35:]
            ohlcv_df = pd.DataFrame(recent)
            
            close = ohlcv_df['close'].values
            returns = np.diff(np.log(close + 1e-10))
            
            features = self.params.tda_generator.compute_persistence_features(returns)
            
            combined = np.sqrt(features['persistence_l0']**2 + features['persistence_l1']**2)
            return min(combined / 2.0, 1.0)
            
        except Exception:
            return 0.5

    def _calculate_position_scale(self, turbulence: float) -> float:
        """Scale position size inversely to turbulence (high turbulence = smaller position)."""
        scale = self.params.tda_scale_max - turbulence * (self.params.tda_scale_max - self.params.tda_scale_min)
        return max(self.params.tda_scale_min, min(self.params.tda_scale_max, scale))

    def _execute_trading_logic(self, nn_signal: float, position_scale: float):
        """Execute buy/sell logic based on signals."""
        current_position = self.position.size
        
        if nn_signal > self.params.buy_threshold and current_position == 0:
            cash = self.broker.getcash()
            price = self.data.close[0]
            max_spend = cash * self.params.max_position_pct * position_scale
            size = int(max_spend / price)
            
            if size > 0:
                self.order = self.buy(size=size)
                self.log(f'BUY SIGNAL: {nn_signal:.3f}, scale: {position_scale:.2f}, size: {size}')
        
        elif nn_signal < self.params.sell_threshold and current_position > 0:
            self.order = self.close()
            self.log(f'SELL SIGNAL: {nn_signal:.3f}')


class PerformanceAnalyzer(bt.Analyzer):
    """Custom analyzer to compute trading performance metrics."""

    def __init__(self):
        """Initialize performance tracking."""
        self.trades = []
        self.returns = []

    def notify_trade(self, trade):
        """Record completed trade information."""
        if trade.isclosed:
            self.trades.append({
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm
            })

    def get_analysis(self):
        """Compute and return performance metrics."""
        if not self.trades:
            return self._empty_analysis()
        
        pnls = [t['pnlcomm'] for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        return {
            'num_trades': len(self.trades),
            'win_rate': len(wins) / len(self.trades) if self.trades else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'total_pnl': sum(pnls)
        }

    def _empty_analysis(self):
        """Return empty analysis when no trades."""
        return {
            'num_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'total_pnl': 0
        }


def test():
    """Test ensemble strategy instantiation and basic functionality."""
    cerebro = bt.Cerebro()
    
    np.random.seed(42)
    n_bars = 200
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='D')
    
    base_price = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    data_df = pd.DataFrame({
        'open': base_price + np.random.randn(n_bars) * 0.1,
        'high': base_price + np.abs(np.random.randn(n_bars) * 0.5),
        'low': base_price - np.abs(np.random.randn(n_bars) * 0.5),
        'close': base_price,
        'volume': np.random.randint(1000, 10000, n_bars)
    }, index=dates)
    
    data = bt.feeds.PandasData(dataname=data_df)
    cerebro.adddata(data)
    
    cerebro.addstrategy(EnsembleStrategy, verbose=False)
    cerebro.addanalyzer(PerformanceAnalyzer, _name='performance')
    
    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.001)
    
    results = cerebro.run()
    
    strategy = results[0]
    assert hasattr(strategy, 'trade_log'), "Strategy missing trade_log"
    assert strategy.bar_count > 0, "Strategy did not process any bars"
    
    perf = results[0].analyzers.performance.get_analysis()
    assert 'num_trades' in perf, "Missing num_trades in analysis"
    
    return True


if __name__ == "__main__":
    success = test()
    if success:
        import sys
        sys.stdout.write("Ensemble Strategy: All tests passed\n")

"""Ensemble Strategy combining TDA regime filter with LSTM predictor for Backtrader.

Version 3.0: Multi-asset support with confidence-based position sizing.
"""

import csv
import os
import numpy as np
import pandas as pd
import backtrader as bt


class EnsembleStrategy(bt.Strategy):
    """Long-only strategy using TDA turbulence gating and NN signal threshold.
    
    Features:
    - Diagnostic logging to CSV for threshold tuning
    - Multi-asset support via ticker parameter
    - Optional confidence-based position sizing
    - Pre-computed TDA features support for efficiency
    """

    params = (
        ('nn_model', None),
        ('tda_generator', None),
        ('preprocessor', None),
        ('sequence_length', 20),
        # Thresholds
        ('nn_buy_threshold', 0.52),
        ('nn_sell_threshold', 0.48),
        ('position_size_pct', 0.15),
        ('tda_regime_min_multiplier', 0.4),
        ('tda_scale_max', 1.0),
        # Multi-asset support
        ('ticker', 'UNKNOWN'),  # Ticker symbol for logging
        ('precomputed_features', None),  # Pre-computed TDA features DataFrame
        ('precomputed_labels', None),  # Pre-computed labels (for alignment)
        # Confidence-based sizing (Phase C)
        ('use_confidence_sizing', False),
        ('min_position_pct', 0.05),
        ('max_position_pct', 0.20),
        # Logging
        ('verbose', False),
        ('diagnostic_csv_path', '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results/signal_diagnostics.csv'),
        ('enable_diagnostics', True),
    )

    def __init__(self):
        """Initialize strategy with data buffers and tracking variables."""
        self.order = None
        self.price_history = []
        self.volume_history = []
        self.tda_history = []
        self.trade_log = []
        self.bar_count = 0
        self.num_trades = 0
        
        # Multi-asset: track which ticker this strategy is for
        self.ticker = self.params.ticker
        
        # Pre-computed features index (for aligning with data bars)
        self.feature_index = 0
        self.use_precomputed = self.params.precomputed_features is not None
        
        # Diagnostic tracking
        self.diagnostic_counts = {
            'ALREADY_POSITIONED': 0,
            'NN_SIGNAL_TOO_LOW_BUY': 0,
            'NN_SIGNAL_TOO_HIGH_SELL': 0,
            'TURBULENCE_BLOCKING': 0,
            'INSUFFICIENT_CASH': 0,
            'TRADE_EXECUTED_BUY': 0,
            'TRADE_EXECUTED_SELL': 0,
            'NO_SIGNAL': 0,
            'WARMING_UP': 0,
            'ORDER_PENDING': 0,
        }
        
        # CSV logging setup
        self.csv_file = None
        self.csv_writer = None
        if self.params.enable_diagnostics:
            self._init_csv_logging()

    def _init_csv_logging(self):
        """Initialize CSV file for diagnostic logging."""
        csv_path = self.params.diagnostic_csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.csv_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'date', 'nn_signal', 'tda_turbulence', 'tda_regime_multiplier',
            'has_position', 'blocked_reason', 'position_size_if_traded',
            'portfolio_value', 'num_trades_so_far'
        ])

    def _log_diagnostic(self, date, nn_signal, turbulence, regime_multiplier,
                        has_position, blocked_reason, position_size, portfolio_value):
        """Log a single bar's diagnostic data to CSV."""
        if self.csv_writer:
            self.csv_writer.writerow([
                date,
                f'{nn_signal:.6f}',
                f'{turbulence:.6f}',
                f'{regime_multiplier:.6f}',
                has_position,
                blocked_reason,
                f'{position_size:.2f}',
                f'{portfolio_value:.2f}',
                self.num_trades
            ])
        self.diagnostic_counts[blocked_reason] = self.diagnostic_counts.get(blocked_reason, 0) + 1

    def log(self, msg: str):
        """Log message with current datetime and ticker if verbose mode enabled."""
        if self.params.verbose:
            dt = self.datas[0].datetime.date(0)
            print(f'[{self.ticker}] {dt}: {msg}')

    def _compute_position_size_pct(self, nn_signal: float) -> float:
        """Compute position size percentage, optionally based on signal confidence.
        
        Confidence = |nn_signal - 0.5| ranges from 0 (neutral) to 0.5 (max confidence).
        Position size scales linearly between min_position_pct and max_position_pct.
        """
        if not self.params.use_confidence_sizing:
            return self.params.position_size_pct
        
        confidence = abs(nn_signal - 0.5)  # 0 to 0.5
        min_pct = self.params.min_position_pct
        max_pct = self.params.max_position_pct
        
        # Linear interpolation: low confidence → min_pct, high confidence → max_pct
        position_pct = min_pct + (max_pct - min_pct) * (confidence / 0.5)
        return position_pct

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
        current_date = self.datas[0].datetime.date(0)
        portfolio_value = self.broker.getvalue()
        has_position = self.position.size > 0
        
        self._update_history()
        
        # Warmup period - not enough data for TDA/NN
        if len(self.price_history) < self.params.sequence_length + 25:
            self._log_diagnostic(
                current_date, 0.5, 0.5, 1.0, has_position,
                'WARMING_UP', 0.0, portfolio_value
            )
            return
        
        # Order pending - skip this bar
        if self.order:
            self._log_diagnostic(
                current_date, 0.5, 0.5, 1.0, has_position,
                'ORDER_PENDING', 0.0, portfolio_value
            )
            return
        
        # Compute signals
        nn_signal = self._get_nn_signal()
        turbulence = self._get_turbulence_index()
        regime_multiplier = self._calculate_position_scale(turbulence)
        
        # Compute potential position size (with optional confidence-based sizing)
        cash = self.broker.getcash()
        price = self.data.close[0]
        position_pct = self._compute_position_size_pct(nn_signal)
        potential_position_value = cash * position_pct * regime_multiplier
        potential_size = int(potential_position_value / price) if price > 0 else 0
        
        # Execute trading logic with diagnostic logging
        self._execute_trading_logic_with_diagnostics(
            nn_signal, turbulence, regime_multiplier,
            has_position, current_date, portfolio_value,
            potential_size, cash, price
        )

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
            import numpy as np
            
            recent = self.price_history[-(self.params.sequence_length + 25):]
            ohlcv_df = pd.DataFrame(recent)
            
            tda_features = self.params.tda_generator.generate_features(ohlcv_df)
            
            if len(tda_features) < self.params.sequence_length:
                return 0.5
            
            X, _ = self.params.preprocessor.prepare_sequences(ohlcv_df, tda_features)
            
            if len(X) == 0:
                return 0.5
            
            # Ensure proper float32 dtype for TensorFlow
            X_input = np.array(X[-1:], dtype=np.float32)
            prediction = self.params.nn_model(X_input, training=False)
            return float(prediction[0, 0])
            
        except Exception as e:
            # Silently return neutral signal on any error
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
        scale = self.params.tda_scale_max - turbulence * (self.params.tda_scale_max - self.params.tda_regime_min_multiplier)
        return max(self.params.tda_regime_min_multiplier, min(self.params.tda_scale_max, scale))

    def _execute_trading_logic_with_diagnostics(self, nn_signal: float, turbulence: float,
                                                  regime_multiplier: float, has_position: bool,
                                                  current_date, portfolio_value: float,
                                                  potential_size: int, cash: float, price: float):
        """Execute buy/sell logic with full diagnostic logging."""
        
        position_value = potential_size * price if potential_size > 0 else 0.0
        
        if has_position:
            # Already in position - check for sell signal
            if nn_signal < self.params.nn_sell_threshold:
                # SELL signal triggered
                self.order = self.close()
                self.num_trades += 1
                self.log(f'SELL SIGNAL: {nn_signal:.3f}')
                self._log_diagnostic(
                    current_date, nn_signal, turbulence, regime_multiplier,
                    True, 'TRADE_EXECUTED_SELL', position_value, portfolio_value
                )
            else:
                # Holding position, no sell signal
                self._log_diagnostic(
                    current_date, nn_signal, turbulence, regime_multiplier,
                    True, 'ALREADY_POSITIONED', position_value, portfolio_value
                )
        else:
            # Not in position - check for buy signal
            if nn_signal > self.params.nn_buy_threshold:
                # Buy signal present
                if regime_multiplier < self.params.tda_regime_min_multiplier:
                    # Turbulence too high
                    self._log_diagnostic(
                        current_date, nn_signal, turbulence, regime_multiplier,
                        False, 'TURBULENCE_BLOCKING', position_value, portfolio_value
                    )
                elif potential_size <= 0:
                    # Not enough cash
                    self._log_diagnostic(
                        current_date, nn_signal, turbulence, regime_multiplier,
                        False, 'INSUFFICIENT_CASH', position_value, portfolio_value
                    )
                else:
                    # Execute BUY
                    self.order = self.buy(size=potential_size)
                    self.num_trades += 1
                    self.log(f'BUY SIGNAL: {nn_signal:.3f}, scale: {regime_multiplier:.2f}, size: {potential_size}')
                    self._log_diagnostic(
                        current_date, nn_signal, turbulence, regime_multiplier,
                        False, 'TRADE_EXECUTED_BUY', position_value, portfolio_value
                    )
            elif nn_signal < self.params.nn_sell_threshold:
                # Signal indicates sell but we're not holding - no action
                self._log_diagnostic(
                    current_date, nn_signal, turbulence, regime_multiplier,
                    False, 'NN_SIGNAL_TOO_HIGH_SELL', position_value, portfolio_value
                )
            else:
                # Signal in neutral zone (between sell and buy thresholds)
                self._log_diagnostic(
                    current_date, nn_signal, turbulence, regime_multiplier,
                    False, 'NO_SIGNAL', position_value, portfolio_value
                )

    def stop(self):
        """Called at end of backtest - print diagnostic summary and close CSV."""
        if self.csv_file:
            self.csv_file.close()
        
        if self.params.enable_diagnostics:
            self._print_diagnostic_summary()

    def _print_diagnostic_summary(self):
        """Print summary of signal diagnostics."""
        total_bars = sum(self.diagnostic_counts.values())
        
        if total_bars == 0:
            return
        
        print("\n")
        print("═" * 60)
        print(f"SIGNAL DIAGNOSTIC SUMMARY [{self.ticker}]")
        print("═" * 60)
        print(f"Ticker: {self.ticker}")
        print(f"Total bars analyzed: {total_bars}")
        print("-" * 60)
        
        # Define display order and labels
        reasons = [
            ('WARMING_UP', 'Warming up (insufficient data)'),
            ('ORDER_PENDING', 'Order pending'),
            ('ALREADY_POSITIONED', 'Already positioned (holding)'),
            ('NN_SIGNAL_TOO_LOW_BUY', 'NN signal too low for buy'),
            ('NN_SIGNAL_TOO_HIGH_SELL', 'NN signal too high for sell (no pos)'),
            ('TURBULENCE_BLOCKING', 'Turbulence blocking'),
            ('INSUFFICIENT_CASH', 'Insufficient cash'),
            ('TRADE_EXECUTED_BUY', 'Trade executed (BUY)'),
            ('TRADE_EXECUTED_SELL', 'Trade executed (SELL)'),
            ('NO_SIGNAL', 'No signal (neutral zone)'),
        ]
        
        for key, label in reasons:
            count = self.diagnostic_counts.get(key, 0)
            pct = (count / total_bars * 100) if total_bars > 0 else 0
            print(f"  {label:<40} {count:>6} ({pct:>5.1f}%)")
        
        print("-" * 60)
        total_trades = self.diagnostic_counts.get('TRADE_EXECUTED_BUY', 0) + \
                       self.diagnostic_counts.get('TRADE_EXECUTED_SELL', 0)
        print(f"  {'TOTAL TRADES':<40} {total_trades:>6}")
        print("═" * 60)
        
        # Threshold info
        print(f"\nCurrent Thresholds:")
        print(f"  NN Buy Threshold:        {self.params.nn_buy_threshold}")
        print(f"  NN Sell Threshold:       {self.params.nn_sell_threshold}")
        print(f"  Position Size Pct:       {self.params.position_size_pct}")
        print(f"  TDA Regime Min Mult:     {self.params.tda_regime_min_multiplier}")
        print("═" * 60)


class PerformanceAnalyzer(bt.Analyzer):
    """Custom analyzer to compute trading performance metrics including turnover.
    
    V1.1: Added total_notional_traded and turnover tracking for cost-aware analysis.
    """

    def __init__(self):
        """Initialize performance tracking."""
        self.trades = []
        self.returns = []
        self.total_notional_traded = 0.0  # Sum of |price * size| across all orders
        self.initial_cash = 0.0

    def start(self):
        """Called when the analyzer starts - capture initial cash."""
        self.initial_cash = self.strategy.broker.getvalue()

    def notify_order(self, order):
        """Track notional value of executed orders for turnover calculation."""
        if order.status == order.Completed:
            # Accumulate absolute notional value traded
            notional = abs(order.executed.price * order.executed.size)
            self.total_notional_traded += notional

    def notify_trade(self, trade):
        """Record completed trade information."""
        if trade.isclosed:
            self.trades.append({
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm
            })

    def get_analysis(self):
        """Compute and return performance metrics including turnover."""
        if not self.trades:
            return self._empty_analysis()
        
        pnls = [t['pnlcomm'] for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        # Compute turnover: total notional traded / initial cash
        turnover = self.total_notional_traded / self.initial_cash if self.initial_cash > 0 else 0.0
        
        return {
            'num_trades': len(self.trades),
            'win_rate': len(wins) / len(self.trades) if self.trades else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'total_pnl': sum(pnls),
            'total_notional_traded': self.total_notional_traded,
            'turnover': turnover,
        }

    def _empty_analysis(self):
        """Return empty analysis when no trades."""
        return {
            'num_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'total_pnl': 0,
            'total_notional_traded': 0.0,
            'turnover': 0.0,
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
    
    cerebro.addstrategy(EnsembleStrategy, verbose=False, enable_diagnostics=False)
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

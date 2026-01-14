"""Ensemble Strategy combining TDA regime filter with LSTM predictor for Backtrader.

Version 4.0: Multi-asset support with risk management framework.
- Fractional Kelly position sizing
- ATR-based stop-losses with min/max bounds  
- Take-profit targets based on risk-reward ratio
- Portfolio heat limits
"""

import csv
import os
import numpy as np
import pandas as pd
import backtrader as bt

from src.risk_management import RiskManager, TradeJournal, calculate_atr
from src.signal_filters import SignalFilter


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
        # V4.0: Risk Management Parameters
        ('use_risk_management', True),  # Enable risk management framework
        ('initial_capital', 100000.0),   # Starting capital for risk calcs
        ('risk_per_trade', 0.02),        # 2% risk per trade (optimized)
        ('stop_atr_multiplier', 2.0),    # ATR multiplier for stops
        ('risk_reward_ratio', 2.0),      # Take-profit R:R ratio
        ('max_portfolio_heat', 0.35),    # Max 35% portfolio heat (optimized)
        ('risk_manager', None),          # External RiskManager instance
        ('trade_journal', None),         # External TradeJournal instance
        # V4.1: Signal Quality Filters
        ('use_signal_filter', True),     # Enable RSI/volatility filtering
        ('rsi_period', 14),              # RSI calculation period
        ('rsi_oversold', 45),            # Buy below this RSI (relaxed from 35)
        ('rsi_overbought', 55),          # Sell above this RSI (relaxed from 65)
        ('vol_threshold', 0.35),         # Pause if vol > 35% (relaxed from 30%)
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
        
        # V4.0: Risk Management Initialization
        self.open_positions = {}  # {ticker: {entry, stop, target, size, date}}
        self.account_balance = self.params.initial_capital
        
        # Use external risk manager if provided, otherwise create new one
        if self.params.risk_manager is not None:
            self.risk_manager = self.params.risk_manager
        elif self.params.use_risk_management:
            self.risk_manager = RiskManager(
                initial_capital=self.params.initial_capital,
                risk_per_trade=self.params.risk_per_trade
            )
        else:
            self.risk_manager = None
        
        # Use external trade journal if provided, otherwise create new one
        if self.params.trade_journal is not None:
            self.trade_journal = self.params.trade_journal
        elif self.params.use_risk_management:
            self.trade_journal = TradeJournal()
        else:
            self.trade_journal = None
        
        # V4.0: Risk metrics tracking
        self.num_stopped_out = 0
        self.num_take_profit_hits = 0
        self.max_portfolio_heat_reached = 0.0
        
        # V4.1: Signal Filter Initialization
        if self.params.use_signal_filter:
            self.signal_filter = SignalFilter(
                rsi_period=self.params.rsi_period,
                rsi_oversold=self.params.rsi_oversold,
                rsi_overbought=self.params.rsi_overbought,
                vol_threshold=self.params.vol_threshold
            )
        else:
            self.signal_filter = None
        
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
            'RSI_FILTERED': 0,           # V4.1: RSI filter blocked
            'VOLATILITY_FILTERED': 0,    # V4.1: Volatility filter blocked
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

    def _calculate_current_atr(self) -> float:
        """Calculate current ATR from price history."""
        if len(self.price_history) < 14:
            return 0.0
        
        recent = self.price_history[-14:]
        df = pd.DataFrame(recent)
        
        if 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
            return 0.0
        
        atr_series = calculate_atr(df, period=14)
        return float(atr_series.iloc[-1]) if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1]) else 0.0

    def generate_signal(self) -> dict:
        """
        Generate trading signal with full risk management parameters.
        
        V4.0 Enhanced Return:
        {
            'signal': 'buy'/'sell'/'neutral',
            'confidence': float (0-1),
            'position_size': int,
            'stop_loss': float,
            'take_profit': float,
            'atr': float,
            'reason': str
        }
        """
        result = {
            'signal': 'neutral',
            'confidence': 0.0,
            'position_size': 0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'atr': 0.0,
            'reason': 'Insufficient data'
        }
        
        # Check warmup
        if len(self.price_history) < self.params.sequence_length + 25:
            return result
        
        # Get current price and signals
        current_price = self.data.close[0]
        nn_signal = self._get_nn_signal()
        turbulence = self._get_turbulence_index()
        regime_multiplier = self._calculate_position_scale(turbulence)
        
        # Calculate ATR for stops
        atr = self._calculate_current_atr()
        if atr <= 0:
            atr = current_price * 0.02  # Default 2% if ATR unavailable
        
        result['atr'] = atr
        result['confidence'] = abs(nn_signal - 0.5) * 2  # Scale to 0-1
        
        # Determine signal direction
        if nn_signal > self.params.nn_buy_threshold:
            result['signal'] = 'buy'
            result['reason'] = f'NN signal {nn_signal:.4f} > buy threshold {self.params.nn_buy_threshold}'
            
            # Check turbulence gating
            if regime_multiplier < self.params.tda_regime_min_multiplier:
                result['signal'] = 'neutral'
                result['reason'] = f'Turbulence blocking: multiplier {regime_multiplier:.3f}'
                return result
            
            # Use risk manager for position sizing if available
            if self.risk_manager is not None and self.params.use_risk_management:
                # Calculate stop and target
                stop_loss = self.risk_manager.set_stop_loss(
                    entry_price=current_price,
                    direction='long',
                    atr_value=atr,
                    multiplier=self.params.stop_atr_multiplier
                )
                take_profit = self.risk_manager.set_take_profit(
                    entry_price=current_price,
                    stop_price=stop_loss,
                    risk_reward_ratio=self.params.risk_reward_ratio
                )
                
                # Check portfolio heat
                can_open, current_heat = self.risk_manager.check_portfolio_heat(
                    self.open_positions,
                    max_heat=self.params.max_portfolio_heat
                )
                self.max_portfolio_heat_reached = max(self.max_portfolio_heat_reached, current_heat)
                
                if not can_open:
                    result['signal'] = 'neutral'
                    result['reason'] = f'Portfolio heat {current_heat:.1%} exceeds max {self.params.max_portfolio_heat:.1%}'
                    return result
                
                # Calculate position size
                position_size = self.risk_manager.calculate_position_size(
                    account_balance=self.account_balance,
                    risk_per_trade=self.params.risk_per_trade,
                    entry_price=current_price,
                    stop_price=stop_loss,
                    volatility=atr,
                    ticker=self.ticker
                )
                
                result['position_size'] = position_size
                result['stop_loss'] = stop_loss
                result['take_profit'] = take_profit
            else:
                # Fallback to simple position sizing
                cash = self.broker.getcash()
                position_pct = self._compute_position_size_pct(nn_signal)
                potential_value = cash * position_pct * regime_multiplier
                result['position_size'] = int(potential_value / current_price) if current_price > 0 else 0
                result['stop_loss'] = current_price * 0.97  # Default 3% stop
                result['take_profit'] = current_price * 1.06  # Default 6% target
                
        elif nn_signal < self.params.nn_sell_threshold:
            result['signal'] = 'sell'
            result['reason'] = f'NN signal {nn_signal:.4f} < sell threshold {self.params.nn_sell_threshold}'
        else:
            result['signal'] = 'neutral'
            result['reason'] = f'NN signal {nn_signal:.4f} in neutral zone'
        
        return result

    def check_exits(self, current_prices: dict, current_date: str) -> list:
        """
        Check if any open positions hit stop-loss or take-profit.
        
        Args:
            current_prices: Dict of {ticker: current_price}
            current_date: Current date string
            
        Returns:
            List of closed position dicts
        """
        closed_positions = []
        
        if not self.open_positions:
            return closed_positions
        
        positions_to_close = []
        
        for ticker, pos in self.open_positions.items():
            current_price = current_prices.get(ticker, pos.get('entry', 0))
            stop_loss = pos.get('stop', 0)
            take_profit = pos.get('target', float('inf'))
            entry_price = pos.get('entry', 0)
            size = pos.get('size', 0)
            entry_date = pos.get('date', '')
            
            exit_reason = None
            exit_price = current_price
            
            # Check stop-loss (long position)
            if current_price <= stop_loss and stop_loss > 0:
                exit_reason = 'stop_loss'
                exit_price = stop_loss
                self.num_stopped_out += 1
            
            # Check take-profit (long position)
            elif current_price >= take_profit and take_profit > 0:
                exit_reason = 'take_profit'
                exit_price = take_profit
                self.num_take_profit_hits += 1
            
            if exit_reason:
                positions_to_close.append({
                    'ticker': ticker,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'size': size,
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'exit_reason': exit_reason
                })
        
        # Close positions and update tracking
        for close_info in positions_to_close:
            ticker = close_info['ticker']
            
            # Record trade in journal
            if self.trade_journal is not None:
                self.trade_journal.log_trade(
                    ticker=ticker,
                    direction='long',
                    entry_date=close_info['entry_date'],
                    entry_price=close_info['entry_price'],
                    exit_date=close_info['exit_date'],
                    exit_price=close_info['exit_price'],
                    size=close_info['size'],
                    stop_loss=close_info['stop_loss'],
                    take_profit=close_info['take_profit'],
                    exit_reason=close_info['exit_reason']
                )
            
            # Update account balance
            pnl = (close_info['exit_price'] - close_info['entry_price']) * close_info['size']
            self.account_balance += pnl
            
            # Record in risk manager
            if self.risk_manager is not None:
                self.risk_manager.record_trade(
                    ticker=ticker,
                    entry_price=close_info['entry_price'],
                    exit_price=close_info['exit_price'],
                    size=close_info['size'],
                    direction='long',
                    entry_date=close_info['entry_date'],
                    exit_date=close_info['exit_date'],
                    exit_reason=close_info['exit_reason']
                )
            
            # Remove from open positions
            del self.open_positions[ticker]
            closed_positions.append(close_info)
            
            self.log(f'EXIT {close_info["exit_reason"].upper()}: {ticker} @ {close_info["exit_price"]:.2f}, PnL: ${pnl:.2f}')
        
        return closed_positions

    def register_position(self, ticker: str, entry_price: float, stop_loss: float,
                         take_profit: float, size: int, entry_date: str):
        """
        Register a new open position for exit monitoring.
        
        Args:
            ticker: Symbol
            entry_price: Entry price
            stop_loss: Stop-loss price
            take_profit: Take-profit target
            size: Position size (shares)
            entry_date: Entry date string
        """
        self.open_positions[ticker] = {
            'entry': entry_price,
            'stop': stop_loss,
            'target': take_profit,
            'size': size,
            'date': entry_date
        }
        self.log(f'REGISTERED: {ticker} @ {entry_price:.2f}, Stop: {stop_loss:.2f}, Target: {take_profit:.2f}')

    def get_risk_metrics(self) -> dict:
        """
        Get current risk management metrics.
        
        Returns:
            Dict with risk metrics for reporting
        """
        journal_stats = {}
        if self.trade_journal:
            journal_stats = self.trade_journal.get_summary_stats()
        
        risk_stats = {}
        if self.risk_manager:
            risk_stats = self.risk_manager.get_risk_metrics(self.open_positions)
        
        return {
            'num_stopped_out': self.num_stopped_out,
            'num_take_profit_hits': self.num_take_profit_hits,
            'max_portfolio_heat_reached': round(self.max_portfolio_heat_reached, 4),
            'open_positions_count': len(self.open_positions),
            'account_balance': self.account_balance,
            **journal_stats,
            **risk_stats
        }

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
        
        cash = self.broker.getcash()
        price = self.data.close[0]
        
        # V4.0: Use risk-aware position sizing if enabled
        if self.params.use_risk_management and self.risk_manager is not None:
            # Use generate_signal for risk-aware sizing
            signal_result = self.generate_signal()
            potential_size = signal_result.get('position_size', 0)
            stop_loss = signal_result.get('stop_loss', price * 0.97)
            take_profit = signal_result.get('take_profit', price * 1.06)
        else:
            # Fallback to simple position sizing
            position_pct = self._compute_position_size_pct(nn_signal)
            potential_position_value = cash * position_pct * regime_multiplier
            potential_size = int(potential_position_value / price) if price > 0 else 0
            stop_loss = price * 0.97
            take_profit = price * 1.06
        
        # Execute trading logic with diagnostic logging
        self._execute_trading_logic_with_risk(
            nn_signal, turbulence, regime_multiplier,
            has_position, current_date, portfolio_value,
            potential_size, cash, price, stop_loss, take_profit
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

    def _execute_trading_logic_with_risk(self, nn_signal: float, turbulence: float,
                                          regime_multiplier: float, has_position: bool,
                                          current_date, portfolio_value: float,
                                          potential_size: int, cash: float, price: float,
                                          stop_loss: float, take_profit: float):
        """Execute buy/sell logic with risk management integration."""
        
        position_value = potential_size * price if potential_size > 0 else 0.0
        
        if has_position:
            # Check stop-loss and take-profit exits first
            current_pos = self.open_positions.get(self.ticker, {})
            pos_stop = current_pos.get('stop', 0)
            pos_target = current_pos.get('target', float('inf'))
            
            # Stop-loss hit
            if pos_stop > 0 and price <= pos_stop:
                self.order = self.close()
                self.num_trades += 1
                self.log(f'STOP-LOSS HIT: price {price:.2f} <= stop {pos_stop:.2f}')
                self._log_diagnostic(
                    current_date, nn_signal, turbulence, regime_multiplier,
                    True, 'STOP_LOSS_EXIT', position_value, portfolio_value
                )
                # Clear tracked position
                if self.ticker in self.open_positions:
                    del self.open_positions[self.ticker]
                return
            
            # Take-profit hit
            if pos_target < float('inf') and price >= pos_target:
                self.order = self.close()
                self.num_trades += 1
                self.log(f'TAKE-PROFIT HIT: price {price:.2f} >= target {pos_target:.2f}')
                self._log_diagnostic(
                    current_date, nn_signal, turbulence, regime_multiplier,
                    True, 'TAKE_PROFIT_EXIT', position_value, portfolio_value
                )
                # Clear tracked position
                if self.ticker in self.open_positions:
                    del self.open_positions[self.ticker]
                return
            
            # Check for NN sell signal
            if nn_signal < self.params.nn_sell_threshold:
                self.order = self.close()
                self.num_trades += 1
                self.log(f'SELL SIGNAL: {nn_signal:.3f}')
                self._log_diagnostic(
                    current_date, nn_signal, turbulence, regime_multiplier,
                    True, 'TRADE_EXECUTED_SELL', position_value, portfolio_value
                )
                # Clear tracked position
                if self.ticker in self.open_positions:
                    del self.open_positions[self.ticker]
            else:
                # Holding position
                self._log_diagnostic(
                    current_date, nn_signal, turbulence, regime_multiplier,
                    True, 'ALREADY_POSITIONED', position_value, portfolio_value
                )
        else:
            # Not in position - check for buy signal
            if nn_signal > self.params.nn_buy_threshold:
                # V4.1: Apply signal quality filter before buying
                signal_passes_filter = True
                filter_reason = None
                
                if self.signal_filter is not None and len(self.price_history) >= 20:
                    # Build price dataframe for filter
                    price_df = pd.DataFrame(self.price_history[-50:])
                    filter_result = self.signal_filter.filter_signal('buy', price_df)
                    
                    if filter_result['filtered']:
                        signal_passes_filter = False
                        filter_reason = filter_result['filter_reason']
                        
                        # Track filtered signals
                        if 'RSI' in filter_reason:
                            self.diagnostic_counts['RSI_FILTERED'] += 1
                        elif 'VOL' in filter_reason:
                            self.diagnostic_counts['VOLATILITY_FILTERED'] += 1
                        
                        self.log(f'SIGNAL FILTERED: {filter_reason}')
                        self._log_diagnostic(
                            current_date, nn_signal, turbulence, regime_multiplier,
                            False, 'SIGNAL_FILTERED', position_value, portfolio_value
                        )
                
                if not signal_passes_filter:
                    pass  # Already logged above
                elif regime_multiplier < self.params.tda_regime_min_multiplier:
                    self._log_diagnostic(
                        current_date, nn_signal, turbulence, regime_multiplier,
                        False, 'TURBULENCE_BLOCKING', position_value, portfolio_value
                    )
                elif potential_size <= 0:
                    self._log_diagnostic(
                        current_date, nn_signal, turbulence, regime_multiplier,
                        False, 'INSUFFICIENT_CASH', position_value, portfolio_value
                    )
                else:
                    # Execute BUY with risk-managed size
                    self.order = self.buy(size=potential_size)
                    self.num_trades += 1
                    
                    # Track position for stop/target management
                    self.open_positions[self.ticker] = {
                        'entry': price,
                        'stop': stop_loss,
                        'target': take_profit,
                        'size': potential_size,
                        'date': str(current_date)
                    }
                    
                    self.log(f'BUY SIGNAL: {nn_signal:.3f}, size: {potential_size}, '
                            f'stop: {stop_loss:.2f}, target: {take_profit:.2f}')
                    self._log_diagnostic(
                        current_date, nn_signal, turbulence, regime_multiplier,
                        False, 'TRADE_EXECUTED_BUY', position_value, portfolio_value
                    )
            elif nn_signal < self.params.nn_sell_threshold:
                self._log_diagnostic(
                    current_date, nn_signal, turbulence, regime_multiplier,
                    False, 'NN_SIGNAL_TOO_HIGH_SELL', position_value, portfolio_value
                )
            else:
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
            ('RSI_FILTERED', 'RSI filter blocked signal'),
            ('VOLATILITY_FILTERED', 'Volatility filter blocked signal'),
            ('INSUFFICIENT_CASH', 'Insufficient cash'),
            ('TRADE_EXECUTED_BUY', 'Trade executed (BUY)'),
            ('TRADE_EXECUTED_SELL', 'Trade executed (SELL)'),
            ('STOP_LOSS_EXIT', 'Stop-loss exit'),
            ('TAKE_PROFIT_EXIT', 'Take-profit exit'),
            ('NO_SIGNAL', 'No signal (neutral zone)'),
        ]
        
        for key, label in reasons:
            count = self.diagnostic_counts.get(key, 0)
            pct = (count / total_bars * 100) if total_bars > 0 else 0
            print(f"  {label:<40} {count:>6} ({pct:>5.1f}%)")
        
        print("-" * 60)
        total_trades = self.diagnostic_counts.get('TRADE_EXECUTED_BUY', 0) + \
                       self.diagnostic_counts.get('TRADE_EXECUTED_SELL', 0) + \
                       self.diagnostic_counts.get('STOP_LOSS_EXIT', 0) + \
                       self.diagnostic_counts.get('TAKE_PROFIT_EXIT', 0)
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

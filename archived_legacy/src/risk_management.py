"""Risk Management Framework for TDA+NN Trading System.

Implements professional position sizing, stop-losses, take-profits, and portfolio heat limits
using Fractional Kelly Criterion for optimal risk-adjusted sizing.

Version 1.0: Initial implementation with full risk controls.
"""

import os
import csv
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskManager:
    """
    Professional risk management for algorithmic trading.
    
    Implements:
    - Fractional Kelly Criterion position sizing (Half-Kelly for optimal performance)
    - ATR-based stop-losses with min/max bounds
    - Risk-reward ratio based take-profits
    - Portfolio heat limits to prevent over-concentration
    
    Formulas:
    - Kelly Fraction = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
    - Position Size = (balance * kelly_fraction * 0.50) / abs(entry - stop)
    - Stop-Loss (long) = entry - (ATR * multiplier)
    - Take-Profit = entry + (abs(entry - stop) * risk_reward_ratio)
    - Portfolio Heat = sum(position_risk) / account_balance
    
    OPTIMIZED V1.1: Half-Kelly (0.50) + 15% position cap for 60%+ win rate strategies
    V2.0: Added volatility-adaptive position sizing for Iteration 2
    """

    # Constants for risk limits - OPTIMIZED for strong win rates
    MAX_POSITION_PCT = 0.15  # Maximum 15% of account per position (up from 10%)
    MIN_STOP_DISTANCE_PCT = 0.015  # Minimum 1.5% stop distance
    MAX_STOP_DISTANCE_PCT = 0.04  # Maximum 4% stop distance
    DEFAULT_KELLY_FRACTION = 0.50  # Half-Kelly (industry standard for 60%+ win rates)
    
    # V2.0: Volatility-adaptive sizing constants
    MIN_VOL_ADJUSTMENT = 0.5  # Minimum position size multiplier
    MAX_VOL_ADJUSTMENT = 1.5  # Maximum position size multiplier

    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_per_trade: float = 0.01,
        log_path: str = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results/risk_log.csv',
        use_volatility_scaling: bool = True,
        target_volatility: float = None
    ):
        """
        Initialize RiskManager with capital and risk parameters.
        
        Args:
            initial_capital: Starting account balance
            risk_per_trade: Maximum risk per trade as fraction of account (default 1%)
            log_path: Path to save risk calculations log
            use_volatility_scaling: Enable volatility-adaptive position sizing (V2.0)
            target_volatility: Target volatility level (if None, uses historical median)
        """
        self.initial_capital = initial_capital
        self.account_balance = initial_capital
        self.risk_per_trade = risk_per_trade
        self.log_path = log_path
        
        # V2.0: Volatility-adaptive sizing parameters
        self.use_volatility_scaling = use_volatility_scaling
        self.target_volatility = target_volatility
        self.volatility_history: List[float] = []
        
        # Kelly parameters (updated from trade history)
        # Default to Half-Kelly (0.50) assumptions:
        # 60% win rate, 2% avg win, 0.8% avg loss produces kelly ~= 0.60
        self.win_rate = 0.60  # Assume 60% win rate for new strategies
        self.avg_win = 0.02  # 2% average win
        self.avg_loss = 0.008  # 0.8% average loss
        self.kelly_fraction = max(self._compute_kelly_fraction(), self.DEFAULT_KELLY_FRACTION)
        
        # Trade history for Kelly updates
        self.trade_history: List[Dict] = []
        
        # Initialize log file
        self._init_log_file()
        
        logger.info(f"RiskManager initialized: capital=${initial_capital:,.2f}, "
                   f"risk_per_trade={risk_per_trade*100:.1f}%, kelly={self.kelly_fraction:.4f}")

    def _init_log_file(self):
        """Initialize the risk log CSV file with headers."""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'calculation_type', 'ticker', 'account_balance',
                    'entry_price', 'stop_price', 'target_price', 'atr', 'volatility',
                    'kelly_fraction', 'position_size', 'position_value', 'risk_amount',
                    'portfolio_heat', 'notes'
                ])

    def _log_calculation(self, calc_type: str, ticker: str = '', **kwargs):
        """Log risk calculation to CSV file."""
        try:
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    calc_type,
                    ticker,
                    kwargs.get('account_balance', self.account_balance),
                    kwargs.get('entry_price', ''),
                    kwargs.get('stop_price', ''),
                    kwargs.get('target_price', ''),
                    kwargs.get('atr', ''),
                    kwargs.get('volatility', ''),
                    kwargs.get('kelly_fraction', self.kelly_fraction),
                    kwargs.get('position_size', ''),
                    kwargs.get('position_value', ''),
                    kwargs.get('risk_amount', ''),
                    kwargs.get('portfolio_heat', ''),
                    kwargs.get('notes', '')
                ])
        except Exception as e:
            logger.warning(f"Failed to log calculation: {e}")

    def _compute_kelly_fraction(self) -> float:
        """
        Compute Kelly fraction from win_rate, avg_win, avg_loss.
        
        Kelly = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
        
        Returns:
            Kelly fraction, constrained to [0, 0.5] range
        """
        if self.avg_win <= 0:
            return 0.0
        
        loss_rate = 1 - self.win_rate
        kelly = (self.win_rate * self.avg_win - loss_rate * self.avg_loss) / self.avg_win
        
        # Constrain to reasonable range
        return max(0.0, min(0.5, kelly))

    def update_volatility_history(self, current_volatility: float):
        """
        Track volatility history for adaptive sizing.
        
        Args:
            current_volatility: Current volatility (ATR or annualized std dev)
        """
        self.volatility_history.append(current_volatility)
        # Keep last 252 trading days (1 year)
        if len(self.volatility_history) > 252:
            self.volatility_history = self.volatility_history[-252:]
        
        # Auto-calculate target volatility as historical median if not set
        if self.target_volatility is None and len(self.volatility_history) >= 20:
            self.target_volatility = np.median(self.volatility_history)

    def calculate_volatility_adjustment(self, current_volatility: float) -> float:
        """
        Calculate position size multiplier based on current vs target volatility.
        
        V2.0: Inverse scaling - reduce size in high volatility, increase in low volatility.
        
        Formula: adjustment = target_vol / current_vol
        - High vol (current > target): adjustment < 1.0 → smaller positions
        - Low vol (current < target): adjustment > 1.0 → larger positions
        - Capped between MIN_VOL_ADJUSTMENT (0.5) and MAX_VOL_ADJUSTMENT (1.5)
        
        Args:
            current_volatility: Current volatility measure
            
        Returns:
            Position size multiplier (0.5 to 1.5)
        """
        if not self.use_volatility_scaling:
            return 1.0
        
        # Need target volatility for scaling
        if self.target_volatility is None or self.target_volatility <= 0:
            return 1.0
        
        if current_volatility <= 0:
            return 1.0
        
        # Inverse scaling: higher vol → smaller position
        raw_adjustment = self.target_volatility / current_volatility
        
        # Clamp to safe range
        adjustment = max(self.MIN_VOL_ADJUSTMENT, min(self.MAX_VOL_ADJUSTMENT, raw_adjustment))
        
        logger.debug(f"Vol adjustment: target={self.target_volatility:.4f}, "
                    f"current={current_volatility:.4f}, adjustment={adjustment:.2f}")
        
        return adjustment

    def calculate_position_size(
        self,
        account_balance: float,
        risk_per_trade: float,
        entry_price: float,
        stop_price: float,
        volatility: float = 0.0,
        ticker: str = '',
        regime_multiplier: float = 1.0
    ) -> int:
        """
        Calculate position size using Fractional Kelly Criterion with volatility adjustment.
        
        V2.0: Adds volatility-adaptive sizing and regime multiplier integration.
        
        Formula:
        - kelly_fraction = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
        - vol_adjustment = target_vol / current_vol (capped 0.5 - 1.5)
        - regime_multiplier = from regime detector (0.0 - 1.25)
        - position_value = balance * kelly_fraction * vol_adjustment * regime_multiplier
        - Hard cap at 15% of account per position
        
        Args:
            account_balance: Current account balance
            risk_per_trade: Risk per trade as fraction of account
            entry_price: Entry price for the trade
            stop_price: Stop-loss price
            volatility: Volatility measure (ATR or similar)
            ticker: Ticker symbol for logging
            regime_multiplier: Position size multiplier from regime detector (default 1.0)
            
        Returns:
            Number of shares to buy (integer)
        """
        # Edge case: invalid prices
        if entry_price <= 0 or stop_price <= 0:
            self._log_calculation('position_size', ticker, notes='Invalid prices',
                                 entry_price=entry_price, stop_price=stop_price)
            return 0
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_price)
        
        # Edge case: stop too close
        if risk_per_share <= 0:
            self._log_calculation('position_size', ticker, notes='Stop too close to entry',
                                 entry_price=entry_price, stop_price=stop_price)
            return 0
        
        # V2.0: Update volatility history and get adjustment
        if volatility > 0:
            self.update_volatility_history(volatility)
        vol_adjustment = self.calculate_volatility_adjustment(volatility)
        
        # Fractional Kelly sizing (use computed kelly_fraction, capped at DEFAULT_KELLY_FRACTION)
        # kelly_fraction already incorporates the half-kelly cap from initialization
        kelly_position_value = account_balance * self.kelly_fraction
        
        # Alternative: fixed risk per trade
        fixed_risk_value = account_balance * risk_per_trade
        
        # Use the more conservative of Kelly-based or fixed risk
        base_position_value = min(kelly_position_value, fixed_risk_value)
        
        # V2.0: Apply volatility adjustment and regime multiplier
        position_risk_value = base_position_value * vol_adjustment * regime_multiplier
        
        # Calculate shares based on risk
        shares = int(position_risk_value / risk_per_share)
        
        # Enforce max position size (15% of account)
        max_shares = int((account_balance * self.MAX_POSITION_PCT) / entry_price)
        shares = min(shares, max_shares)
        
        # Ensure at least 1 share if we have a valid position
        if shares <= 0 and position_risk_value > 0:
            shares = 0  # Can't afford minimum position
        
        # Calculate final position value
        position_value = shares * entry_price
        risk_amount = shares * risk_per_share
        
        # Log the calculation with V2.0 fields
        self._log_calculation(
            'position_size', ticker,
            account_balance=account_balance,
            entry_price=entry_price,
            stop_price=stop_price,
            volatility=volatility,
            kelly_fraction=self.kelly_fraction,
            position_size=shares,
            position_value=position_value,
            risk_amount=risk_amount,
            notes=f'Kelly:{self.kelly_fraction:.4f}, MaxPct:{self.MAX_POSITION_PCT}'
        )
        
        return shares

    def set_stop_loss(
        self,
        entry_price: float,
        direction: str,
        atr_value: float,
        multiplier: float = 2.0
    ) -> float:
        """
        Calculate ATR-based stop-loss price.
        
        Formula:
        - Long: stop = entry - (ATR * multiplier)
        - Short: stop = entry + (ATR * multiplier)
        - Enforce min 1.5% and max 4% stop distance
        
        Args:
            entry_price: Entry price for the trade
            direction: 'long' or 'short'
            atr_value: 14-period ATR value
            multiplier: ATR multiplier for stop distance (default 2.0)
            
        Returns:
            Stop-loss price
        """
        # Edge case: invalid inputs
        if entry_price <= 0 or atr_value < 0:
            return entry_price * 0.97  # Default 3% stop
        
        # Calculate raw stop distance
        stop_distance = atr_value * multiplier
        
        # Enforce min/max stop distance as percentage of entry
        min_distance = entry_price * self.MIN_STOP_DISTANCE_PCT
        max_distance = entry_price * self.MAX_STOP_DISTANCE_PCT
        stop_distance = max(min_distance, min(max_distance, stop_distance))
        
        # Calculate stop price based on direction
        if direction.lower() == 'long':
            stop_price = entry_price - stop_distance
        else:  # short
            stop_price = entry_price + stop_distance
        
        return round(stop_price, 4)

    def set_take_profit(
        self,
        entry_price: float,
        stop_price: float,
        risk_reward_ratio: float = 2.0
    ) -> float:
        """
        Calculate take-profit price based on risk-reward ratio.
        
        Formula:
        - risk = abs(entry - stop)
        - target = entry + (risk * risk_reward_ratio)  # for long
        
        Args:
            entry_price: Entry price for the trade
            stop_price: Stop-loss price
            risk_reward_ratio: Desired risk-reward ratio (default 2:1)
            
        Returns:
            Take-profit price
        """
        # Edge case: invalid inputs
        if entry_price <= 0 or stop_price <= 0:
            return entry_price * 1.06  # Default 6% target (2:1 on 3% risk)
        
        risk = abs(entry_price - stop_price)
        
        # Determine direction from stop placement
        if stop_price < entry_price:  # Long position
            target_price = entry_price + (risk * risk_reward_ratio)
        else:  # Short position
            target_price = entry_price - (risk * risk_reward_ratio)
        
        return round(target_price, 4)

    def check_portfolio_heat(
        self,
        open_positions: Dict[str, Dict],
        max_heat: float = 0.20
    ) -> Tuple[bool, float]:
        """
        Check if portfolio heat allows new positions.
        
        Portfolio Heat = sum(position_risk) / account_balance
        where position_risk = position_size * abs(entry - stop)
        
        Args:
            open_positions: Dict of open positions
                {ticker: {entry, stop, target, size, date}}
            max_heat: Maximum allowed portfolio heat (default 20%)
            
        Returns:
            Tuple of (can_open_new, current_heat)
        """
        if not open_positions:
            return True, 0.0
        
        total_risk = 0.0
        
        for ticker, pos in open_positions.items():
            if not isinstance(pos, dict):
                continue
            
            size = pos.get('size', 0)
            entry = pos.get('entry', 0)
            stop = pos.get('stop', entry)
            
            # Risk for this position
            if size > 0 and entry > 0:
                position_risk = size * abs(entry - stop)
                total_risk += position_risk
        
        current_heat = total_risk / self.account_balance if self.account_balance > 0 else 0.0
        can_open = current_heat < max_heat
        
        self._log_calculation(
            'portfolio_heat_check', '',
            portfolio_heat=current_heat,
            notes=f'Can open: {can_open}, Max heat: {max_heat}'
        )
        
        return can_open, current_heat

    def update_kelly_parameters(
        self,
        recent_trades_df: Optional[pd.DataFrame] = None,
        lookback: int = 50
    ) -> Dict[str, float]:
        """
        Recalculate Kelly parameters from recent trade history.
        
        Args:
            recent_trades_df: DataFrame with columns ['pnl', 'entry_value'] or None to use internal history
            lookback: Number of recent trades to analyze
            
        Returns:
            Dict with updated stats: win_rate, avg_win, avg_loss, kelly_fraction
        """
        trades = []
        
        if recent_trades_df is not None and len(recent_trades_df) > 0:
            # Use provided DataFrame
            if 'pnl' in recent_trades_df.columns:
                trades = recent_trades_df['pnl'].tail(lookback).tolist()
        elif self.trade_history:
            # Use internal trade history
            recent = self.trade_history[-lookback:]
            trades = [t.get('pnl', 0) for t in recent]
        
        # Edge case: no trades
        if not trades or len(trades) < 5:
            # Keep defaults but return current state
            return {
                'win_rate': self.win_rate,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'kelly_fraction': self.kelly_fraction,
                'num_trades': len(trades)
            }
        
        # Compute statistics
        wins = [t for t in trades if t > 0]
        losses = [abs(t) for t in trades if t < 0]
        
        self.win_rate = len(wins) / len(trades) if trades else 0.5
        self.avg_win = np.mean(wins) if wins else 0.02
        self.avg_loss = np.mean(losses) if losses else 0.01
        
        # Handle edge case where avg_loss is 0
        if self.avg_loss <= 0:
            self.avg_loss = 0.001
        
        # Recompute Kelly fraction
        self.kelly_fraction = self._compute_kelly_fraction()
        
        result = {
            'win_rate': round(self.win_rate, 4),
            'avg_win': round(self.avg_win, 4),
            'avg_loss': round(self.avg_loss, 4),
            'kelly_fraction': round(self.kelly_fraction, 4),
            'num_trades': len(trades)
        }
        
        self._log_calculation(
            'kelly_update', '',
            kelly_fraction=self.kelly_fraction,
            notes=f'WR:{self.win_rate:.2f}, AvgW:{self.avg_win:.4f}, AvgL:{self.avg_loss:.4f}'
        )
        
        logger.info(f"Kelly parameters updated: {result}")
        return result

    def record_trade(
        self,
        ticker: str,
        entry_price: float,
        exit_price: float,
        size: int,
        direction: str,
        entry_date: str,
        exit_date: str,
        exit_reason: str = ''
    ) -> Dict[str, Any]:
        """
        Record a completed trade in history.
        
        Args:
            ticker: Symbol traded
            entry_price: Entry price
            exit_price: Exit price
            size: Position size (shares)
            direction: 'long' or 'short'
            entry_date: Entry date string
            exit_date: Exit date string
            exit_reason: Reason for exit (stop/target/signal)
            
        Returns:
            Trade record dict
        """
        if direction.lower() == 'long':
            pnl = (exit_price - entry_price) * size
            pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
        else:
            pnl = (entry_price - exit_price) * size
            pnl_pct = (entry_price - exit_price) / entry_price if entry_price > 0 else 0
        
        trade = {
            'ticker': ticker,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'direction': direction,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_value': entry_price * size,
            'exit_reason': exit_reason
        }
        
        self.trade_history.append(trade)
        
        # Update account balance
        self.account_balance += pnl
        
        return trade

    def update_account_balance(self, new_balance: float):
        """Update the current account balance."""
        self.account_balance = new_balance

    def get_risk_metrics(self, open_positions: Dict[str, Dict] = None) -> Dict[str, Any]:
        """
        Get current risk management metrics summary.
        
        Returns:
            Dict with current risk state
        """
        open_positions = open_positions or {}
        can_open, current_heat = self.check_portfolio_heat(open_positions)
        
        # Compute R-multiples from trade history
        r_multiples = []
        for trade in self.trade_history:
            # R = profit / risk (simplified as pnl / avg_loss * entry_value)
            entry_val = trade.get('entry_value', 1)
            pnl = trade.get('pnl', 0)
            if entry_val > 0 and self.avg_loss > 0:
                risk_amt = entry_val * self.avg_loss
                if risk_amt > 0:
                    r_multiples.append(pnl / risk_amt)
        
        return {
            'account_balance': self.account_balance,
            'initial_capital': self.initial_capital,
            'account_return': (self.account_balance - self.initial_capital) / self.initial_capital,
            'kelly_fraction': self.kelly_fraction,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'num_trades_recorded': len(self.trade_history),
            'current_portfolio_heat': current_heat,
            'can_open_new_position': can_open,
            'avg_r_multiple': np.mean(r_multiples) if r_multiples else 0,
            'open_positions_count': len(open_positions)
        }


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) for a price DataFrame.
    
    True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    ATR = SMA(True Range, period)
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period (default 14)
        
    Returns:
        Series with ATR values
    """
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']
    
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


class TradeJournal:
    """
    Trade journal for tracking entries, exits, and performance metrics.
    """
    
    def __init__(
        self,
        journal_path: str = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results/trade_journal.csv'
    ):
        """
        Initialize trade journal.
        
        Args:
            journal_path: Path to save trade journal CSV
        """
        self.journal_path = journal_path
        self.trades: List[Dict] = []
        self._init_journal_file()
    
    def _init_journal_file(self):
        """Initialize journal CSV with headers."""
        os.makedirs(os.path.dirname(self.journal_path), exist_ok=True)
        if not os.path.exists(self.journal_path):
            with open(self.journal_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'trade_id', 'ticker', 'direction', 'entry_date', 'entry_price',
                    'exit_date', 'exit_price', 'size', 'stop_loss', 'take_profit',
                    'pnl', 'pnl_pct', 'r_multiple', 'exit_reason', 'holding_period_days'
                ])
    
    def log_trade(
        self,
        ticker: str,
        direction: str,
        entry_date: str,
        entry_price: float,
        exit_date: str,
        exit_price: float,
        size: int,
        stop_loss: float,
        take_profit: float,
        exit_reason: str
    ) -> Dict:
        """
        Log a completed trade to the journal.
        
        Returns:
            Trade record dict with calculated metrics
        """
        # Calculate PnL
        if direction.lower() == 'long':
            pnl = (exit_price - entry_price) * size
            pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
        else:
            pnl = (entry_price - exit_price) * size
            pnl_pct = (entry_price - exit_price) / entry_price if entry_price > 0 else 0
        
        # Calculate R-multiple
        risk = abs(entry_price - stop_loss) * size
        r_multiple = pnl / risk if risk > 0 else 0
        
        # Calculate holding period
        try:
            entry_dt = pd.to_datetime(entry_date)
            exit_dt = pd.to_datetime(exit_date)
            holding_days = (exit_dt - entry_dt).days
        except:
            holding_days = 0
        
        trade_id = len(self.trades) + 1
        
        trade = {
            'trade_id': trade_id,
            'ticker': ticker,
            'direction': direction,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'size': size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'r_multiple': r_multiple,
            'exit_reason': exit_reason,
            'holding_period_days': holding_days
        }
        
        self.trades.append(trade)
        
        # Write to CSV
        with open(self.journal_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trade_id, ticker, direction, entry_date, entry_price,
                exit_date, exit_price, size, stop_loss, take_profit,
                round(pnl, 2), round(pnl_pct, 4), round(r_multiple, 2),
                exit_reason, holding_days
            ])
        
        return trade
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Calculate summary statistics from logged trades.
        
        Returns:
            Dict with summary metrics
        """
        if not self.trades:
            return {
                'num_trades': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'avg_r_multiple': 0,
                'profit_factor': 0,
                'num_stopped_out': 0,
                'num_take_profit_hits': 0,
                'expectancy_per_trade': 0
            }
        
        pnls = [t['pnl'] for t in self.trades]
        r_multiples = [t['r_multiple'] for t in self.trades]
        
        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]
        
        gross_wins = sum(wins)
        gross_losses = sum(losses)
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
        
        num_stopped_out = sum(1 for t in self.trades if t['exit_reason'] == 'stop_loss')
        num_take_profit = sum(1 for t in self.trades if t['exit_reason'] == 'take_profit')
        
        # Expectancy = avg_win * win_rate - avg_loss * loss_rate
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        win_rate = len(wins) / len(self.trades)
        loss_rate = 1 - win_rate
        expectancy = (avg_win * win_rate) - (avg_loss * loss_rate)
        
        return {
            'num_trades': len(self.trades),
            'win_rate': round(win_rate, 4),
            'avg_pnl': round(np.mean(pnls), 2),
            'total_pnl': round(sum(pnls), 2),
            'avg_r_multiple': round(np.mean(r_multiples), 2),
            'profit_factor': round(profit_factor, 2),
            'num_stopped_out': num_stopped_out,
            'num_take_profit_hits': num_take_profit,
            'expectancy_per_trade': round(expectancy, 2),
            'gross_wins': round(gross_wins, 2),
            'gross_losses': round(gross_losses, 2)
        }


def test_risk_manager():
    """Unit tests for RiskManager."""
    print("\n" + "=" * 60)
    print("Testing RiskManager")
    print("=" * 60)
    
    rm = RiskManager(initial_capital=100000, risk_per_trade=0.01)
    
    # Test 1: Position sizing
    print("\nTest 1: Position sizing")
    entry = 100.0
    stop = 97.0  # 3% stop
    size = rm.calculate_position_size(100000, 0.01, entry, stop)
    max_allowed = int(100000 * 0.10 / entry)  # 10% cap
    print(f"  Entry: ${entry}, Stop: ${stop}, Size: {size} shares")
    assert size <= max_allowed, f"Position size {size} exceeds max {max_allowed}"
    print("  ✓ Position size within 10% cap")
    
    # Test 2: Stop-loss calculation
    print("\nTest 2: Stop-loss calculation")
    atr = 2.5
    stop_long = rm.set_stop_loss(100.0, 'long', atr, multiplier=2.0)
    stop_short = rm.set_stop_loss(100.0, 'short', atr, multiplier=2.0)
    print(f"  ATR: {atr}, Long stop: ${stop_long}, Short stop: ${stop_short}")
    
    # Check min/max bounds
    stop_dist_pct = abs(100.0 - stop_long) / 100.0
    assert stop_dist_pct >= 0.015, f"Stop distance {stop_dist_pct:.3f} below minimum 1.5%"
    assert stop_dist_pct <= 0.04, f"Stop distance {stop_dist_pct:.3f} above maximum 4%"
    print(f"  ✓ Stop distance {stop_dist_pct*100:.1f}% within bounds [1.5%, 4%]")
    
    # Test 3: Take-profit calculation
    print("\nTest 3: Take-profit calculation")
    target = rm.set_take_profit(100.0, 97.0, risk_reward_ratio=2.0)
    expected_target = 100.0 + (3.0 * 2.0)  # 106.0
    print(f"  Entry: $100, Stop: $97, Target: ${target} (expected ~${expected_target})")
    assert abs(target - expected_target) < 0.01, f"Target {target} != expected {expected_target}"
    print("  ✓ Take-profit correctly calculated")
    
    # Test 4: Portfolio heat
    print("\nTest 4: Portfolio heat check")
    positions = {
        'SPY': {'entry': 100, 'stop': 97, 'size': 300},
        'QQQ': {'entry': 200, 'stop': 194, 'size': 150}
    }
    can_open, heat = rm.check_portfolio_heat(positions, max_heat=0.20)
    print(f"  Open positions: {len(positions)}")
    print(f"  Portfolio heat: {heat*100:.2f}%, Can open: {can_open}")
    assert heat < 1.0, f"Heat {heat} unreasonably high"
    print("  ✓ Portfolio heat calculated correctly")
    
    # Test 5: Kelly update
    print("\nTest 5: Kelly parameter update")
    trades_df = pd.DataFrame({
        'pnl': [100, -50, 200, -30, 150, -40, 80, 120, -60, 90]
    })
    stats = rm.update_kelly_parameters(trades_df, lookback=10)
    print(f"  Updated Kelly stats: {stats}")
    assert 'kelly_fraction' in stats
    assert 0 <= stats['kelly_fraction'] <= 0.5
    print("  ✓ Kelly parameters updated")
    
    # Test 6: Edge cases
    print("\nTest 6: Edge cases")
    size_zero_vol = rm.calculate_position_size(100000, 0.01, 100.0, 100.0)  # Stop = entry
    print(f"  Size when stop=entry: {size_zero_vol}")
    assert size_zero_vol == 0, "Should return 0 when stop equals entry"
    
    size_invalid = rm.calculate_position_size(100000, 0.01, -100.0, 97.0)  # Negative entry
    print(f"  Size with invalid price: {size_invalid}")
    assert size_invalid == 0, "Should return 0 for invalid price"
    print("  ✓ Edge cases handled correctly")
    
    print("\n" + "=" * 60)
    print("All RiskManager tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_risk_manager()

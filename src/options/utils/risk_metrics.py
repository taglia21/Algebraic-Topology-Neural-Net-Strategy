"""
Risk Metrics and Performance Calculations
=========================================

Kelly criterion, Sharpe ratio, and other risk-adjusted metrics.
"""

import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    max_fraction: float = 0.25
) -> float:
    """
    Calculate Kelly criterion position size.
    
    Formula: f = (p * b - q) / b
    where:
        p = probability of win
        q = probability of loss = 1 - p
        b = payoff ratio = avg_win / avg_loss
        f = fraction of capital to risk
    
    Args:
        win_rate: Historical win rate (0.0 to 1.0)
        avg_win: Average win amount
        avg_loss: Average loss amount (positive value)
        max_fraction: Maximum Kelly fraction (cap for safety)
        
    Returns:
        Kelly fraction (0.0 to max_fraction)
    """
    # Validate inputs
    if not (0.0 < win_rate < 1.0):
        logger.warning(f"Invalid win_rate: {win_rate}, using default 0.5")
        win_rate = 0.5
    
    if avg_win <= 0 or avg_loss <= 0:
        logger.warning(f"Invalid avg_win={avg_win} or avg_loss={avg_loss}, using min fraction")
        return 0.01
    
    # Calculate payoff ratio
    payoff_ratio = avg_win / avg_loss
    
    if payoff_ratio <= 0:
        logger.warning(f"Invalid payoff_ratio: {payoff_ratio}")
        return 0.01
    
    # Kelly formula
    loss_rate = 1.0 - win_rate
    kelly = (win_rate * payoff_ratio - loss_rate) / payoff_ratio
    
    # Cap Kelly (full Kelly can be too aggressive)
    kelly = max(0.01, min(kelly, max_fraction))
    
    return kelly


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio (risk-adjusted return).
    
    Formula: (Mean Return - Risk Free Rate) / Std Dev of Returns * sqrt(periods)
    
    Args:
        returns: List of period returns (e.g., daily returns)
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    
    # Calculate mean and std dev
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array, ddof=1)  # Sample std dev
    
    if std_return == 0:
        return 0.0
    
    # Annualize
    annualized_return = mean_return * periods_per_year
    annualized_std = std_return * np.sqrt(periods_per_year)
    
    # Sharpe ratio
    sharpe = (annualized_return - risk_free_rate) / annualized_std
    
    return float(sharpe)


def calculate_sortino_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (like Sharpe but only penalizes downside volatility).
    
    Args:
        returns: List of period returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year
        
    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    
    # Calculate mean return
    mean_return = np.mean(returns_array)
    
    # Calculate downside deviation (only negative returns)
    downside_returns = returns_array[returns_array < 0]
    
    if len(downside_returns) == 0:
        return float('inf')  # No downside = infinite Sortino
    
    downside_std = np.std(downside_returns, ddof=1)
    
    if downside_std == 0:
        return 0.0
    
    # Annualize
    annualized_return = mean_return * periods_per_year
    annualized_downside_std = downside_std * np.sqrt(periods_per_year)
    
    # Sortino ratio
    sortino = (annualized_return - risk_free_rate) / annualized_downside_std
    
    return float(sortino)


def calculate_max_drawdown(equity_curve: List[float]) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown from equity curve.
    
    Args:
        equity_curve: List of portfolio values over time
        
    Returns:
        (max_drawdown_pct, start_index, end_index)
    """
    if len(equity_curve) < 2:
        return 0.0, 0, 0
    
    equity_array = np.array(equity_curve)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_array)
    
    # Calculate drawdown at each point
    drawdown = (equity_array - running_max) / running_max
    
    # Find maximum drawdown
    max_dd_idx = np.argmin(drawdown)
    max_dd = drawdown[max_dd_idx]
    
    # Find peak before drawdown
    peak_idx = np.argmax(equity_array[:max_dd_idx + 1])
    
    return float(max_dd), int(peak_idx), int(max_dd_idx)


def calculate_profit_factor(
    winning_trades: List[float],
    losing_trades: List[float]
) -> float:
    """
    Calculate profit factor (gross profits / gross losses).
    
    Args:
        winning_trades: List of winning trade P&Ls
        losing_trades: List of losing trade P&Ls
        
    Returns:
        Profit factor (>1.0 = profitable)
    """
    if not winning_trades:
        return 0.0
    
    gross_profit = sum(winning_trades)
    gross_loss = abs(sum(losing_trades)) if losing_trades else 0.0
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_calmar_ratio(
    annualized_return: float,
    max_drawdown: float
) -> float:
    """
    Calculate Calmar ratio (return / max drawdown).
    
    Args:
        annualized_return: Annual return (e.g., 0.25 for 25%)
        max_drawdown: Maximum drawdown (e.g., -0.15 for -15%)
        
    Returns:
        Calmar ratio
    """
    if max_drawdown >= 0:
        return 0.0
    
    return annualized_return / abs(max_drawdown)


def calculate_win_rate(trades: List[float]) -> float:
    """
    Calculate win rate from list of trade P&Ls.
    
    Args:
        trades: List of trade P&Ls
        
    Returns:
        Win rate (0.0 to 1.0)
    """
    if not trades:
        return 0.0
    
    winning_trades = sum(1 for pnl in trades if pnl > 0)
    return winning_trades / len(trades)


def calculate_expected_value(
    win_rate: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Calculate expected value per trade.
    
    Formula: (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
    
    Args:
        win_rate: Probability of winning (0.0 to 1.0)
        avg_win: Average winning trade
        avg_loss: Average losing trade (positive value)
        
    Returns:
        Expected value per trade
    """
    loss_rate = 1.0 - win_rate
    return (win_rate * avg_win) - (loss_rate * avg_loss)


def risk_of_ruin(
    win_rate: float,
    payoff_ratio: float,
    max_drawdown_pct: float = 0.50
) -> float:
    """
    Calculate risk of ruin (probability of hitting max drawdown).
    
    Simplified formula assuming fixed bet sizing.
    
    Args:
        win_rate: Historical win rate
        payoff_ratio: Avg win / avg loss
        max_drawdown_pct: Maximum acceptable drawdown (e.g., 0.50 for 50%)
        
    Returns:
        Probability of ruin (0.0 to 1.0)
    """
    if win_rate >= 1.0 or win_rate <= 0.0:
        return 0.5
    
    loss_rate = 1.0 - win_rate
    
    # Risk of ruin formula (simplified)
    if payoff_ratio <= 0:
        return 1.0
    
    q_over_p = loss_rate / win_rate
    
    # If payoff ratio >= q/p, risk approaches 0
    if payoff_ratio >= q_over_p:
        return 0.0
    
    # Otherwise calculate ruin probability
    ruin_ratio = (q_over_p / payoff_ratio) ** (1 / max_drawdown_pct)
    
    return min(1.0, ruin_ratio)

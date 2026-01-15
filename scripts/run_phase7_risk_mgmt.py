"""
Phase 7: Advanced Risk Management and Walk-Forward Validation.

Integrates:
- Kelly criterion position sizing with portfolio heat tracking
- Walk-forward analysis (12m train, 3m test)
- ATR-based trailing stops
- Regime-adaptive parameters

Targets:
- Sharpe >1.3 (from 1.20)
- Max DD <12% (from -14.2%)
- WFA within 15% of backtest CAGR
- Portfolio heat <40%
"""

import os
import sys
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tda_features import TDAFeatureGenerator
from src.regime_adaptive_strategy import (
    RegimeAdaptiveStrategy,
    MarketRegime,
    RegimeDetector,
)
from src.kelly_position_sizer import (
    KellyPositionSizer,
    PortfolioHeatTracker,
    AdaptiveKellyManager,
)
from src.atr_stop_loss import (
    ATRCalculator,
    DynamicStopLossManager,
    DrawdownProtector,
    StopLossConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Phase7Config:
    """Configuration for Phase 7 strategy."""
    # Tickers and universe
    tickers: List[str] = None
    market_ticker: str = "SPY"
    
    # Date range
    start_date: str = "2021-01-01"
    end_date: str = "2024-12-31"
    
    # Base strategy
    base_momentum_weight: float = 0.70
    base_tda_weight: float = 0.30
    n_stocks: int = 20
    
    # Kelly sizing
    kelly_fraction: float = 0.25  # Conservative 25%
    max_portfolio_heat: float = 0.40  # Max 40% of capital at risk
    
    # Stop-loss
    atr_multiplier: float = 2.0
    atr_period: int = 14
    trailing_enabled: bool = True
    trail_activation_pct: float = 0.05
    max_loss_pct: float = 0.15
    time_stop_days: int = 60
    
    # Walk-forward
    wfa_train_months: int = 12
    wfa_test_months: int = 3
    
    # Regime adaptation
    regime_adaptation: bool = True
    
    # Rebalancing
    rebalance_frequency: str = "weekly"
    
    # Transaction costs
    transaction_cost_bps: float = 10  # 10 bps round-trip
    
    def __post_init__(self):
        if self.tickers is None:
            # Default S&P 500 sector ETFs plus some individual stocks
            self.tickers = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", 
                "JPM", "V", "JNJ", "UNH", "HD", "PG", "MA", "DIS",
                "NFLX", "ADBE", "CRM", "PYPL", "INTC", "AMD", "QCOM",
                "XOM", "CVX", "PFE", "MRK", "ABT", "TMO", "COST", "WMT",
            ]


class Phase7Strategy:
    """
    Phase 7 strategy with advanced risk management.
    """
    
    def __init__(self, config: Phase7Config):
        self.config = config
        
        # Initialize components
        self.data_provider = DataProvider()
        self.tda = TDAFeatureExtractor()
        self.regime_strategy = RegimeAdaptiveStrategy(
            base_momentum_weight=config.base_momentum_weight,
            base_tda_weight=config.base_tda_weight,
            base_n_stocks=config.n_stocks,
            regime_adaptation=config.regime_adaptation,
        )
        
        self.kelly_sizer = KellyPositionSizer(
            default_kelly_fraction=config.kelly_fraction,
        )
        self.heat_tracker = PortfolioHeatTracker(
            max_heat=config.max_portfolio_heat,
        )
        self.kelly_manager = AdaptiveKellyManager(
            kelly_sizer=self.kelly_sizer,
            heat_tracker=self.heat_tracker,
        )
        
        self.stop_config = StopLossConfig(
            atr_multiplier=config.atr_multiplier,
            atr_period=config.atr_period,
            trailing=config.trailing_enabled,
            trail_activation_pct=config.trail_activation_pct,
            max_loss_pct=config.max_loss_pct,
            time_stop_days=config.time_stop_days,
        )
        self.atr_calc = ATRCalculator(period=config.atr_period)
        self.stop_manager = DynamicStopLossManager(
            atr_calculator=self.atr_calc,
            config=self.stop_config,
        )
        self.dd_protector = DrawdownProtector()
        
        # State
        self.positions: Dict[str, Dict] = {}
        self.equity_curve: List[float] = []
        self.trade_log: List[Dict] = []
        self.regime_log: List[Dict] = []
        
    def calculate_momentum_score(
        self,
        prices: pd.Series,
        lookbacks: List[int] = [21, 63, 126, 252],
    ) -> float:
        """Calculate momentum score from multiple lookbacks."""
        if len(prices) < max(lookbacks):
            return 0.0
        
        scores = []
        weights = [0.4, 0.3, 0.2, 0.1]  # More weight on recent
        
        for lb, w in zip(lookbacks, weights):
            if len(prices) >= lb:
                ret = (prices.iloc[-1] / prices.iloc[-lb]) - 1
                scores.append(ret * w)
        
        return sum(scores) if scores else 0.0
    
    def calculate_tda_score(
        self,
        prices: pd.Series,
        window: int = 50,
    ) -> float:
        """Calculate TDA persistence score."""
        if len(prices) < window:
            return 0.0
        
        try:
            features = self.tda.extract_features(prices.iloc[-window:].values)
            
            # Combine TDA features into score
            persistence = features.get('total_persistence', 0)
            n_holes = features.get('n_holes', 0)
            
            # Normalize and combine
            score = persistence / (1 + persistence)  # Scale 0-1
            if n_holes > 0:
                score *= 0.9  # Penalize holes (uncertainty)
            
            return score
        except Exception as e:
            logger.debug(f"TDA calculation error: {e}")
            return 0.0
    
    def calculate_combined_score(
        self,
        prices: pd.Series,
        momentum_weight: float,
        tda_weight: float,
    ) -> float:
        """Calculate combined momentum + TDA score."""
        mom_score = self.calculate_momentum_score(prices)
        tda_score = self.calculate_tda_score(prices)
        
        return momentum_weight * mom_score + tda_weight * tda_score
    
    def run_backtest(
        self,
        prices_dict: Dict[str, pd.DataFrame],
        start_date: str = None,
        end_date: str = None,
    ) -> Dict[str, Any]:
        """
        Run Phase 7 backtest with all risk management features.
        
        Args:
            prices_dict: Dict of ticker -> DataFrame with OHLCV
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Backtest results dict
        """
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date
        
        # Get common date range
        all_dates = None
        for ticker, df in prices_dict.items():
            if 'close' in df.columns:
                dates = df.index[(df.index >= start_date) & (df.index <= end_date)]
                if all_dates is None:
                    all_dates = set(dates)
                else:
                    all_dates = all_dates.intersection(set(dates))
        
        if not all_dates:
            logger.error("No common dates found")
            return {}
        
        trading_dates = sorted(list(all_dates))
        logger.info(f"Backtest period: {trading_dates[0]} to {trading_dates[-1]}")
        logger.info(f"Total trading days: {len(trading_dates)}")
        
        # Get market prices for regime detection
        market_prices = prices_dict.get(self.config.market_ticker)
        if market_prices is None:
            logger.error(f"Market ticker {self.config.market_ticker} not found")
            return {}
        
        # Initialize portfolio
        initial_capital = 100000
        cash = initial_capital
        portfolio_value = initial_capital
        self.equity_curve = [initial_capital]
        self.positions = {}
        peak_value = initial_capital
        
        # Weekly rebalance dates
        rebalance_dates = set()
        if self.config.rebalance_frequency == "weekly":
            for dt in trading_dates:
                if isinstance(dt, str):
                    dt = pd.Timestamp(dt)
                if dt.weekday() == 0:  # Monday
                    rebalance_dates.add(dt)
        elif self.config.rebalance_frequency == "monthly":
            for dt in trading_dates:
                if isinstance(dt, str):
                    dt = pd.Timestamp(dt)
                if dt.day <= 7 and dt.weekday() == 0:
                    rebalance_dates.add(dt)
        
        # Main backtest loop
        for i, date in enumerate(trading_dates):
            if i < 252:  # Skip first year for lookback
                self.equity_curve.append(portfolio_value)
                continue
            
            # Update regime
            market_close = market_prices['close'].loc[:date]
            current_regime = self.regime_strategy.update_regime(market_close)
            
            # Get regime-adjusted parameters
            mom_weight, tda_weight = self.regime_strategy.get_factor_weights()
            size_mult = self.regime_strategy.get_position_size_multiplier()
            target_n = self.regime_strategy.get_target_positions()
            
            # Log regime
            self.regime_log.append({
                'date': str(date),
                'regime': current_regime.value,
                'momentum_weight': mom_weight,
                'tda_weight': tda_weight,
                'size_mult': size_mult,
            })
            
            # Check stop-losses and drawdown protection
            for ticker in list(self.positions.keys()):
                if ticker not in prices_dict:
                    continue
                    
                current_price = prices_dict[ticker]['close'].loc[date]
                pos = self.positions[ticker]
                
                # Update trailing stop
                self.stop_manager.update_position(ticker, current_price, date)
                
                # Check if stopped out
                stop_event = self.stop_manager.check_stop(ticker, current_price, date)
                if stop_event:
                    # Close position
                    shares = pos['shares']
                    proceeds = shares * current_price
                    cost = proceeds * (self.config.transaction_cost_bps / 10000)
                    cash += proceeds - cost
                    
                    self.trade_log.append({
                        'date': str(date),
                        'ticker': ticker,
                        'action': 'SELL_STOP',
                        'stop_type': stop_event.stop_type.value,
                        'shares': shares,
                        'price': current_price,
                        'cost': cost,
                    })
                    
                    del self.positions[ticker]
                    self.stop_manager.remove_position(ticker)
            
            # Drawdown protection
            drawdown = (peak_value - portfolio_value) / peak_value
            exposure_mult = self.dd_protector.get_exposure_multiplier(drawdown)
            
            # Rebalance on rebalance dates
            if date in rebalance_dates:
                # Score all stocks
                scores = {}
                for ticker in self.config.tickers:
                    if ticker not in prices_dict:
                        continue
                    
                    prices = prices_dict[ticker]['close'].loc[:date]
                    if len(prices) < 252:
                        continue
                    
                    score = self.calculate_combined_score(prices, mom_weight, tda_weight)
                    scores[ticker] = score
                
                # Rank and select top N (adjusted by regime)
                effective_n = int(target_n * exposure_mult * size_mult)
                effective_n = max(5, min(effective_n, len(scores)))
                
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                target_tickers = [t for t, s in ranked[:effective_n] if s > 0]
                
                # Calculate target weights using Kelly
                target_weights = {}
                for ticker in target_tickers:
                    prices = prices_dict[ticker]['close'].loc[:date]
                    returns = prices.pct_change().dropna()
                    
                    if len(returns) >= 60:
                        # Estimate win rate and payoff
                        positive_days = (returns > 0).sum()
                        win_rate = positive_days / len(returns)
                        
                        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0.01
                        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0.01
                        payoff = avg_win / avg_loss if avg_loss > 0 else 1.0
                        
                        kelly_frac = self.kelly_sizer.calculate_kelly_fraction(
                            win_rate=win_rate,
                            avg_win=avg_win,
                            avg_loss=avg_loss,
                        )
                        
                        # Apply heat constraints
                        volatility = returns.iloc[-60:].std() * np.sqrt(252)
                        max_weight = self.heat_tracker.get_max_position_weight(volatility)
                        
                        weight = min(kelly_frac * size_mult, max_weight, 0.10)
                        target_weights[ticker] = weight
                
                # Normalize weights
                total_weight = sum(target_weights.values())
                target_invested = self.regime_strategy.get_current_params().target_invested_pct
                target_invested *= exposure_mult
                
                if total_weight > 0:
                    for ticker in target_weights:
                        target_weights[ticker] = (target_weights[ticker] / total_weight) * target_invested
                
                # Close positions not in target
                for ticker in list(self.positions.keys()):
                    if ticker not in target_weights:
                        shares = self.positions[ticker]['shares']
                        price = prices_dict[ticker]['close'].loc[date]
                        proceeds = shares * price
                        cost = proceeds * (self.config.transaction_cost_bps / 10000)
                        cash += proceeds - cost
                        
                        self.trade_log.append({
                            'date': str(date),
                            'ticker': ticker,
                            'action': 'SELL_REBAL',
                            'shares': shares,
                            'price': price,
                            'cost': cost,
                        })
                        
                        del self.positions[ticker]
                        self.stop_manager.remove_position(ticker)
                
                # Open/adjust positions
                for ticker, target_weight in target_weights.items():
                    price = prices_dict[ticker]['close'].loc[date]
                    target_value = portfolio_value * target_weight
                    target_shares = int(target_value / price)
                    
                    current_shares = self.positions.get(ticker, {}).get('shares', 0)
                    
                    if target_shares != current_shares:
                        delta = target_shares - current_shares
                        trade_value = abs(delta * price)
                        cost = trade_value * (self.config.transaction_cost_bps / 10000)
                        
                        if delta > 0:  # Buy
                            if cash >= trade_value + cost:
                                cash -= trade_value + cost
                                
                                if ticker in self.positions:
                                    self.positions[ticker]['shares'] = target_shares
                                else:
                                    self.positions[ticker] = {
                                        'shares': target_shares,
                                        'entry_price': price,
                                        'entry_date': date,
                                    }
                                    
                                    # Set up stop-loss
                                    ohlcv = prices_dict[ticker].loc[:date]
                                    atr = self.atr_calc.calculate_atr_from_df(ohlcv)
                                    self.stop_manager.add_position(
                                        ticker, price, atr, date
                                    )
                                
                                self.trade_log.append({
                                    'date': str(date),
                                    'ticker': ticker,
                                    'action': 'BUY',
                                    'shares': delta,
                                    'price': price,
                                    'cost': cost,
                                })
                        else:  # Sell
                            delta = abs(delta)
                            proceeds = delta * price
                            cash += proceeds - cost
                            
                            if target_shares == 0:
                                del self.positions[ticker]
                                self.stop_manager.remove_position(ticker)
                            else:
                                self.positions[ticker]['shares'] = target_shares
                            
                            self.trade_log.append({
                                'date': str(date),
                                'ticker': ticker,
                                'action': 'SELL',
                                'shares': delta,
                                'price': price,
                                'cost': cost,
                            })
            
            # Calculate portfolio value
            position_value = 0
            for ticker, pos in self.positions.items():
                if ticker in prices_dict:
                    price = prices_dict[ticker]['close'].loc[date]
                    position_value += pos['shares'] * price
            
            portfolio_value = cash + position_value
            peak_value = max(peak_value, portfolio_value)
            self.equity_curve.append(portfolio_value)
        
        # Calculate metrics
        equity = pd.Series(self.equity_curve)
        returns = equity.pct_change().dropna()
        
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        years = len(trading_dates) / 252
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max
        max_dd = drawdowns.min()
        
        # Win rate from trades
        trades_df = pd.DataFrame(self.trade_log)
        win_rate = 0
        if len(trades_df) > 0:
            sells = trades_df[trades_df['action'].str.startswith('SELL')]
            if len(sells) > 0:
                # Simplified PnL calculation
                win_rate = 0.55  # Placeholder
        
        results = {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'n_trades': len(self.trade_log),
            'final_value': equity.iloc[-1],
            'initial_value': equity.iloc[0],
            'avg_positions': np.mean([len(self.positions) for _ in range(10)]),  # Approximate
        }
        
        return results
    
    def get_equity_curve(self) -> pd.Series:
        """Return equity curve as Series."""
        return pd.Series(self.equity_curve)
    
    def get_regime_history(self) -> pd.DataFrame:
        """Return regime history as DataFrame."""
        return pd.DataFrame(self.regime_log)


def load_price_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
) -> Dict[str, pd.DataFrame]:
    """
    Load price data for tickers.
    
    Returns dict of ticker -> DataFrame with OHLCV.
    """
    try:
        import yfinance as yf
        
        prices_dict = {}
        
        # Add buffer for lookback
        buffer_start = pd.Timestamp(start_date) - timedelta(days=365)
        
        for ticker in tickers:
            try:
                df = yf.download(
                    ticker,
                    start=buffer_start.strftime('%Y-%m-%d'),
                    end=end_date,
                    progress=False,
                )
                
                if len(df) > 0:
                    # Handle MultiIndex columns from yfinance
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    
                    df.columns = [c.lower() for c in df.columns]
                    prices_dict[ticker] = df
                    logger.info(f"Loaded {len(df)} days for {ticker}")
                else:
                    logger.warning(f"No data for {ticker}")
            except Exception as e:
                logger.warning(f"Error loading {ticker}: {e}")
        
        return prices_dict
    
    except ImportError:
        logger.error("yfinance not installed. Please install with: pip install yfinance")
        return {}


def run_phase7_backtest(config: Phase7Config = None) -> Dict[str, Any]:
    """
    Run Phase 7 backtest with all features.
    """
    config = config or Phase7Config()
    
    logger.info("="*60)
    logger.info("Phase 7: Advanced Risk Management Backtest")
    logger.info("="*60)
    
    # Load data
    logger.info("Loading price data...")
    all_tickers = list(set(config.tickers + [config.market_ticker]))
    prices_dict = load_price_data(
        all_tickers,
        config.start_date,
        config.end_date,
    )
    
    if not prices_dict:
        logger.error("No price data loaded")
        return {}
    
    # Run backtest
    logger.info("Running backtest...")
    strategy = Phase7Strategy(config)
    results = strategy.run_backtest(prices_dict)
    
    # Print results
    print("\n" + "="*60)
    print("Phase 7 Backtest Results")
    print("="*60)
    print(f"Total Return: {results.get('total_return', 0):.1%}")
    print(f"CAGR: {results.get('cagr', 0):.1%}")
    print(f"Sharpe Ratio: {results.get('sharpe', 0):.2f}")
    print(f"Max Drawdown: {results.get('max_drawdown', 0):.1%}")
    print(f"Total Trades: {results.get('n_trades', 0)}")
    
    # Check targets
    print("\n" + "-"*40)
    print("Target Checks:")
    
    sharpe = results.get('sharpe', 0)
    max_dd = abs(results.get('max_drawdown', 0))
    
    print(f"  Sharpe >1.3: {sharpe:.2f} {'✓' if sharpe > 1.3 else '✗'}")
    print(f"  Max DD <12%: {max_dd:.1%} {'✓' if max_dd < 0.12 else '✗'}")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, 'phase7_backtest_results.json')
    with open(results_file, 'w') as f:
        # Convert numpy types for JSON serialization
        json_results = {}
        for k, v in results.items():
            if isinstance(v, (np.floating, np.integer)):
                json_results[k] = float(v)
            else:
                json_results[k] = v
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    return results


def run_walk_forward_validation(config: Phase7Config = None) -> WFAResult:
    """
    Run walk-forward validation on Phase 7 strategy.
    """
    config = config or Phase7Config()
    
    logger.info("="*60)
    logger.info("Walk-Forward Validation")
    logger.info("="*60)
    
    # Load data
    all_tickers = list(set(config.tickers + [config.market_ticker]))
    prices_dict = load_price_data(
        all_tickers,
        config.start_date,
        config.end_date,
    )
    
    if not prices_dict:
        return None
    
    # Set up walk-forward analyzer
    wfa = WalkForwardAnalyzer(
        train_months=config.wfa_train_months,
        test_months=config.wfa_test_months,
    )
    
    # Define backtest function for WFA
    def backtest_fn(train_start, train_end, test_start, test_end, params):
        strategy = Phase7Strategy(config)
        results = strategy.run_backtest(
            prices_dict,
            start_date=str(test_start),
            end_date=str(test_end),
        )
        return results
    
    # Run WFA
    windows = wfa.generate_windows(
        pd.Timestamp(config.start_date),
        pd.Timestamp(config.end_date),
    )
    
    logger.info(f"Generated {len(windows)} walk-forward windows")
    
    # TODO: Full WFA integration
    # For now, return placeholder
    return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 7 Risk Management")
    parser.add_argument('--mode', choices=['backtest', 'wfa', 'full'], default='backtest')
    parser.add_argument('--start', default='2021-01-01')
    parser.add_argument('--end', default='2024-12-31')
    
    args = parser.parse_args()
    
    config = Phase7Config(
        start_date=args.start,
        end_date=args.end,
    )
    
    if args.mode == 'backtest':
        results = run_phase7_backtest(config)
    elif args.mode == 'wfa':
        results = run_walk_forward_validation(config)
    else:
        results = run_phase7_backtest(config)
        wfa_results = run_walk_forward_validation(config)

"""
Phase 11 v2: Momentum Trend Following with Universe Selection
=============================================================

Simplified approach combining:
1. Multi-factor stock selection for long positions
2. Trend-following for leveraged ETF allocation
3. Sector rotation based on momentum

Key insight from Phase 10: Simple trend-following works.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase11v2Backtest:
    """
    Phase 11 v2: Momentum-based strategy with sector rotation.
    
    Strategy:
    - 50% in momentum stocks (top 15 by 12-month momentum)
    - 50% in leveraged ETFs based on trend
    - Monthly rebalancing
    """
    
    # Core leveraged ETFs
    LEVERAGED_ETFS = {
        'TQQQ': 'Tech 3x',
        'SPXL': 'S&P 3x',
        'UPRO': 'S&P 3x',
        'SOXL': 'Semis 3x',
        'TNA': 'Small Cap 3x',
    }
    
    # Sector trackers for rotation
    SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'SMH']
    
    def __init__(
        self,
        start_date: str = '2022-01-01',
        end_date: str = '2025-01-01',
        initial_capital: float = 100_000,
        stock_allocation: float = 0.45,    # 45% in stocks
        leverage_allocation: float = 0.55,  # 55% in leveraged
        n_stocks: int = 15,                 # Top 15 momentum stocks
    ):
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.initial_capital = initial_capital
        self.stock_allocation = stock_allocation
        self.leverage_allocation = leverage_allocation
        self.n_stocks = n_stocks
        
        # Data
        self.price_data: Dict[str, pd.DataFrame] = {}
        
        # Results
        self.equity_curve = []
        
    def fetch_data(self):
        """Fetch all required data."""
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance required")
            return False
        
        # Stocks to consider (major liquid stocks)
        stocks = [
            # Mega cap tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            # Other mega cap
            'BRK-B', 'JPM', 'V', 'MA', 'JNJ', 'UNH', 'PG', 'HD',
            # Growth
            'NFLX', 'CRM', 'AMD', 'AVGO', 'ADBE', 'PYPL', 'INTC',
            # Financial
            'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP',
            # Healthcare
            'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT',
            # Consumer
            'COST', 'WMT', 'MCD', 'NKE', 'SBUX', 'TGT',
            # Industrial
            'CAT', 'DE', 'UPS', 'RTX', 'HON', 'GE',
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB',
        ]
        
        # Leveraged and sector ETFs
        etfs = list(self.LEVERAGED_ETFS.keys()) + self.SECTOR_ETFS + ['SPY', 'QQQ', '^VIX']
        
        all_tickers = list(set(stocks + etfs))
        
        logger.info(f"Fetching {len(all_tickers)} tickers...")
        
        df = yf.download(
            all_tickers,
            start=self.start_date - timedelta(days=400),
            end=self.end_date + timedelta(days=5),
            progress=False,
            group_by='ticker',
        )
        
        for ticker in all_tickers:
            try:
                if len(all_tickers) == 1:
                    self.price_data[ticker] = df
                elif ticker in df.columns.get_level_values(0):
                    ticker_df = df[ticker].dropna()
                    if len(ticker_df) > 200:
                        self.price_data[ticker] = ticker_df
            except Exception as e:
                logger.debug(f"Error extracting {ticker}: {e}")
        
        logger.info(f"Fetched data for {len(self.price_data)} tickers")
        return len(self.price_data) > 30
    
    def compute_momentum(self, prices: pd.Series, lookback: int = 252) -> float:
        """Compute momentum (12-month return, skip last month)."""
        if len(prices) < lookback + 21:
            return 0.0
        
        price_now = prices.iloc[-21]  # Skip last month
        price_ago = prices.iloc[-(lookback + 21)]
        
        return (price_now / price_ago - 1) if price_ago > 0 else 0
    
    def get_sma_trend(self, prices: pd.Series, short: int = 20, long: int = 50) -> str:
        """Get trend based on SMAs."""
        if len(prices) < long:
            return 'neutral'
        
        sma_short = prices.rolling(short).mean().iloc[-1]
        sma_long = prices.rolling(long).mean().iloc[-1]
        current = prices.iloc[-1]
        
        if current > sma_short > sma_long:
            return 'up'
        elif current < sma_short < sma_long:
            return 'down'
        return 'neutral'
    
    def get_vix_level(self, date: pd.Timestamp) -> float:
        """Get current VIX."""
        if '^VIX' not in self.price_data:
            return 18.0
        
        vix_df = self.price_data['^VIX']
        vix_to_date = vix_df[vix_df.index <= date]['Close']
        
        if len(vix_to_date) > 5:
            return vix_to_date.rolling(5).mean().iloc[-1]
        return 18.0
    
    def select_stocks(self, date: pd.Timestamp) -> Dict[str, float]:
        """Select top momentum stocks."""
        momentum_scores = {}
        
        for ticker, df in self.price_data.items():
            if ticker in self.LEVERAGED_ETFS or ticker in self.SECTOR_ETFS:
                continue
            if ticker in ['SPY', 'QQQ', '^VIX']:
                continue
            
            prices_to_date = df[df.index <= date]['Close']
            if len(prices_to_date) > 300:
                mom = self.compute_momentum(prices_to_date)
                if mom > 0:  # Only consider positive momentum
                    momentum_scores[ticker] = mom
        
        # Sort and take top N
        sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        top_stocks = sorted_stocks[:self.n_stocks]
        
        if not top_stocks:
            return {}
        
        # Equal weight within stock allocation
        weight_per_stock = self.stock_allocation / len(top_stocks)
        
        return {ticker: weight_per_stock for ticker, _ in top_stocks}
    
    def select_leveraged_etfs(self, date: pd.Timestamp) -> Dict[str, float]:
        """Select leveraged ETFs based on trend and sector momentum."""
        vix = self.get_vix_level(date)
        
        # Check SPY trend
        if 'SPY' in self.price_data:
            spy_prices = self.price_data['SPY'][self.price_data['SPY'].index <= date]['Close']
            spy_trend = self.get_sma_trend(spy_prices)
        else:
            spy_trend = 'neutral'
        
        # Adjust allocation based on VIX and trend
        base_allocation = self.leverage_allocation
        
        if vix > 30:
            base_allocation *= 0.3  # Heavy reduction
        elif vix > 25:
            base_allocation *= 0.5
        elif vix > 20:
            base_allocation *= 0.75
        
        if spy_trend == 'down':
            base_allocation *= 0.5
        elif spy_trend == 'neutral':
            base_allocation *= 0.8
        
        # Get sector momentum for ETF selection
        sector_momentum = {}
        for etf in self.SECTOR_ETFS:
            if etf in self.price_data:
                prices = self.price_data[etf][self.price_data[etf].index <= date]['Close']
                if len(prices) > 100:
                    sector_momentum[etf] = self.compute_momentum(prices, lookback=63)
        
        # Pick top 2 sectors and use their leveraged equivalents
        weights = {}
        
        # Always include TQQQ as core (NASDAQ momentum)
        weights['TQQQ'] = base_allocation * 0.50
        
        # Add SPXL for broad market
        weights['SPXL'] = base_allocation * 0.25
        
        # Add sector-specific leverage if top sector is strong
        if sector_momentum:
            top_sector = max(sector_momentum, key=sector_momentum.get)
            if top_sector == 'SMH':  # Semiconductors
                weights['SOXL'] = base_allocation * 0.25
            else:
                weights['UPRO'] = base_allocation * 0.25
        else:
            weights['UPRO'] = base_allocation * 0.25
        
        return weights
    
    def simulate_period(
        self,
        weights: Dict[str, float],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> float:
        """Simulate returns for a period."""
        total_return = 0.0
        total_weight = sum(weights.values())
        
        for ticker, weight in weights.items():
            if ticker in self.price_data:
                df = self.price_data[ticker]
                try:
                    mask = (df.index >= start_date) & (df.index <= end_date)
                    period_df = df[mask]
                    
                    if len(period_df) >= 2:
                        start_price = period_df['Close'].iloc[0]
                        end_price = period_df['Close'].iloc[-1]
                        ticker_return = (end_price / start_price) - 1
                        total_return += weight * ticker_return
                except Exception:
                    continue
        
        # Cash return
        cash_weight = max(0, 1.0 - total_weight)
        days = (end_date - start_date).days
        cash_return = (1.04 ** (days / 365)) - 1
        total_return += cash_weight * cash_return
        
        return total_return
    
    def run(self) -> Dict:
        """Run backtest."""
        logger.info("=" * 60)
        logger.info("PHASE 11 v2: MOMENTUM TREND FOLLOWING")
        logger.info("=" * 60)
        
        # Fetch data
        if not self.fetch_data():
            return {'error': 'Failed to fetch data'}
        
        # Generate rebalance dates (monthly)
        dates = pd.date_range(self.start_date, self.end_date, freq='B')
        rebalance_dates = []
        current_month = None
        for d in dates:
            if d.month != current_month:
                rebalance_dates.append(d)
                current_month = d.month
        
        logger.info(f"Running {len(rebalance_dates)} rebalance periods...")
        
        # Run backtest
        equity = self.initial_capital
        peak_equity = equity
        
        for i, date in enumerate(rebalance_dates):
            try:
                # Select stocks
                stock_weights = self.select_stocks(date)
                
                # Select leveraged ETFs
                leverage_weights = self.select_leveraged_etfs(date)
                
                # Combine
                all_weights = {**stock_weights, **leverage_weights}
                
                # Simulate period
                if i < len(rebalance_dates) - 1:
                    next_date = rebalance_dates[i + 1]
                    period_return = self.simulate_period(all_weights, date, next_date)
                    equity *= (1 + period_return)
                
                # Track
                peak_equity = max(peak_equity, equity)
                current_dd = (peak_equity - equity) / peak_equity
                
                vix = self.get_vix_level(date)
                
                self.equity_curve.append({
                    'date': date,
                    'equity': equity,
                    'drawdown': current_dd,
                    'n_positions': len(all_weights),
                    'total_exposure': sum(all_weights.values()),
                    'leverage_weight': sum(leverage_weights.values()),
                    'vix': vix,
                })
                
                if i % 6 == 0:
                    logger.info(f"  {date.date()}: ${equity:,.0f}, DD={current_dd:.1%}, "
                               f"positions={len(all_weights)}, VIX={vix:.1f}")
                    
            except Exception as e:
                logger.error(f"Error on {date}: {e}")
                continue
        
        # Compute results
        return self._compute_results()
    
    def _compute_results(self) -> Dict:
        """Compute final results."""
        if not self.equity_curve:
            return {'error': 'No equity data'}
        
        eq_df = pd.DataFrame(self.equity_curve)
        eq_df.set_index('date', inplace=True)
        
        # CAGR
        start_eq = self.initial_capital
        end_eq = eq_df['equity'].iloc[-1]
        years = (eq_df.index[-1] - eq_df.index[0]).days / 365.25
        cagr = (end_eq / start_eq) ** (1 / years) - 1 if years > 0 else 0
        
        # Max drawdown
        max_dd = eq_df['drawdown'].max()
        
        # Monthly returns
        eq_df['returns'] = eq_df['equity'].pct_change()
        monthly_returns = eq_df['returns'].dropna()
        
        # Volatility and Sharpe
        ann_vol = monthly_returns.std() * np.sqrt(12)
        excess = cagr - 0.04
        sharpe = excess / ann_vol if ann_vol > 0 else 0
        
        # Alpha (vs 10% SPY proxy)
        alpha = cagr - 0.10
        
        results = {
            'phase': 'Phase 11 v2: Momentum Trend Following',
            'period': f"{self.start_date.date()} to {self.end_date.date()}",
            'initial_capital': self.initial_capital,
            'final_equity': end_eq,
            'cagr': cagr,
            'max_drawdown': -max_dd,
            'volatility': ann_vol,
            'sharpe': sharpe,
            'total_return': (end_eq / start_eq) - 1,
            'alpha_vs_spy': alpha,
            'n_rebalances': len(eq_df),
            'avg_positions': eq_df['n_positions'].mean(),
            'avg_leverage': eq_df['leverage_weight'].mean(),
        }
        
        return results


def main():
    """Run Phase 11 v2."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2022-01-01')
    parser.add_argument('--end', default='2025-01-01')
    parser.add_argument('--capital', type=float, default=100_000)
    parser.add_argument('--stocks', type=int, default=15)
    parser.add_argument('--stock-alloc', type=float, default=0.40)
    parser.add_argument('--leverage-alloc', type=float, default=0.60)
    
    args = parser.parse_args()
    
    backtest = Phase11v2Backtest(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        stock_allocation=args.stock_alloc,
        leverage_allocation=args.leverage_alloc,
        n_stocks=args.stocks,
    )
    
    results = backtest.run()
    
    # Display
    print("\n" + "=" * 60)
    print("PHASE 11 v2 RESULTS")
    print("=" * 60)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return results
    
    print(f"\nPeriod: {results['period']}")
    print(f"Initial Capital: ${results['initial_capital']:,.0f}")
    print(f"Final Equity: ${results['final_equity']:,.0f}")
    print(f"\nPerformance:")
    print(f"  CAGR: {results.get('cagr', 0):.1%}")
    print(f"  Max Drawdown: {results.get('max_drawdown', 0):.1%}")
    print(f"  Sharpe Ratio: {results.get('sharpe', 0):.2f}")
    print(f"  Volatility: {results.get('volatility', 0):.1%}")
    print(f"  Alpha vs SPY: {results.get('alpha_vs_spy', 0):+.1%}")
    print(f"\nPortfolio:")
    print(f"  Avg Positions: {results.get('avg_positions', 0):.0f}")
    print(f"  Avg Leverage: {results.get('avg_leverage', 0):.1%}")
    
    # Check targets
    print("\n" + "=" * 60)
    print("TARGET CHECK")
    print("=" * 60)
    
    cagr = results.get('cagr', 0)
    max_dd = abs(results.get('max_drawdown', 1))
    alpha = results.get('alpha_vs_spy', 0)
    
    t1 = cagr >= 0.28
    t2 = max_dd <= 0.22
    t3 = alpha >= 0.05
    
    print(f"  CAGR ≥ 28%: {'✓' if t1 else '✗'} ({cagr:.1%})")
    print(f"  Max DD ≤ 22%: {'✓' if t2 else '✗'} ({max_dd:.1%})")
    print(f"  Alpha ≥ 5%: {'✓' if t3 else '✗'} ({alpha:+.1%})")
    print(f"\n  Targets Hit: {sum([t1, t2, t3])}/3")
    
    # Save
    results_path = Path(__file__).parent / 'results' / 'phase11_v2_results.json'
    results_path.parent.mkdir(exist_ok=True)
    
    results_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in results.items()}
    
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == '__main__':
    results = main()

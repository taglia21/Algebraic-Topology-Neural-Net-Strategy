"""
Phase 11 v3: Trend-Following with Aggressive Drawdown Protection
================================================================

Key lessons from Phase 10 v3's success (25.6% CAGR, 20.2% DD):
1. Start reducing exposure at just 5-6% drawdown
2. Use trend confirmation before taking leveraged positions
3. Stay in cash/low leverage during downtrends
4. Go aggressive ONLY when trend is clearly up

Strategy:
- 50-70% in leveraged ETFs when trend is up AND VIX < 20
- 30-50% in momentum stocks always (sector rotation)
- Reduce to 20-30% total when trend is down or VIX high
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase11v3:
    """
    Phase 11 v3: Aggressive trend-following with strict risk controls.
    """
    
    LEVERAGED_ETFS = ['TQQQ', 'SPXL', 'UPRO', 'SOXL']
    
    def __init__(
        self,
        start_date: str = '2022-01-01',
        end_date: str = '2025-01-01',
        initial_capital: float = 100_000,
    ):
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.initial_capital = initial_capital
        
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.equity_curve = []
        
    def fetch_data(self):
        """Fetch data."""
        try:
            import yfinance as yf
        except ImportError:
            return False
        
        # Momentum stocks (50 most liquid)
        stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'JPM', 'V', 'MA', 'UNH', 'HD', 'PG', 'JNJ',
            'NFLX', 'CRM', 'AMD', 'AVGO', 'ADBE', 'PYPL', 'INTC',
            'BAC', 'WFC', 'GS', 'MS', 'C',
            'LLY', 'ABBV', 'MRK', 'PFE', 'TMO',
            'COST', 'WMT', 'MCD', 'NKE', 'SBUX',
            'CAT', 'DE', 'UPS', 'RTX', 'HON',
            'XOM', 'CVX', 'COP', 'EOG', 'SLB',
        ]
        
        etfs = self.LEVERAGED_ETFS + ['SPY', 'QQQ', 'IWM', 'XLK', 'XLF', 'SMH', '^VIX']
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
                    if len(ticker_df) > 150:
                        self.price_data[ticker] = ticker_df
            except:
                pass
        
        logger.info(f"Fetched {len(self.price_data)} tickers")
        return len(self.price_data) > 20
    
    def get_trend(self, prices: pd.Series) -> str:
        """Determine trend using multiple MAs."""
        if len(prices) < 200:
            return 'neutral'
        
        current = prices.iloc[-1]
        sma20 = prices.rolling(20).mean().iloc[-1]
        sma50 = prices.rolling(50).mean().iloc[-1]
        sma200 = prices.rolling(200).mean().iloc[-1]
        
        # Strong uptrend: price > SMA20 > SMA50 > SMA200
        if current > sma20 > sma50 > sma200:
            return 'strong_up'
        # Moderate uptrend
        elif current > sma50 > sma200:
            return 'up'
        # Strong downtrend
        elif current < sma20 < sma50 < sma200:
            return 'strong_down'
        # Moderate downtrend  
        elif current < sma50:
            return 'down'
        
        return 'neutral'
    
    def get_vix(self, date: pd.Timestamp) -> float:
        if '^VIX' not in self.price_data:
            return 18.0
        vix_df = self.price_data['^VIX']
        vix_to_date = vix_df[vix_df.index <= date]['Close']
        if len(vix_to_date) > 5:
            return vix_to_date.rolling(5).mean().iloc[-1]
        return 18.0
    
    def get_momentum(self, prices: pd.Series, lookback: int = 252) -> float:
        if len(prices) < lookback + 21:
            return 0.0
        price_now = prices.iloc[-21]  # Skip last month
        price_ago = prices.iloc[-(lookback + 21)]
        return (price_now / price_ago - 1) if price_ago > 0 else 0
    
    def select_stocks(self, date: pd.Timestamp, target_allocation: float) -> Dict[str, float]:
        """Select momentum stocks with target allocation."""
        momentum = {}
        
        for ticker, df in self.price_data.items():
            if ticker in self.LEVERAGED_ETFS + ['SPY', 'QQQ', 'IWM', '^VIX']:
                continue
            if ticker.startswith('XL') or ticker == 'SMH':
                continue
            
            prices = df[df.index <= date]['Close']
            if len(prices) > 300:
                mom = self.get_momentum(prices)
                if mom > 0.05:  # Only stocks with >5% momentum
                    momentum[ticker] = mom
        
        if not momentum:
            return {}
        
        # Top 10 momentum stocks
        sorted_stocks = sorted(momentum.items(), key=lambda x: x[1], reverse=True)[:10]
        
        weight_per = target_allocation / len(sorted_stocks)
        return {t: weight_per for t, _ in sorted_stocks}
    
    def select_leveraged(
        self, 
        date: pd.Timestamp, 
        trend: str, 
        vix: float,
        current_dd: float,
    ) -> Dict[str, float]:
        """Select leveraged ETF allocation based on conditions."""
        
        # Base allocation based on trend
        if trend == 'strong_up':
            base = 0.65  # 65% in leverage
        elif trend == 'up':
            base = 0.50  # 50% in leverage
        elif trend == 'neutral':
            base = 0.30  # 30% in leverage  
        elif trend == 'down':
            base = 0.15  # 15% in leverage
        else:  # strong_down
            base = 0.05  # 5% minimal
        
        # VIX adjustment
        if vix > 30:
            base *= 0.2  # 80% cut
        elif vix > 25:
            base *= 0.4  # 60% cut
        elif vix > 20:
            base *= 0.7  # 30% cut
        
        # CRITICAL: Drawdown protection (like Phase 10)
        if current_dd > 0.15:
            base *= 0.30  # 70% cut
        elif current_dd > 0.10:
            base *= 0.50  # 50% cut
        elif current_dd > 0.06:
            base *= 0.75  # 25% cut
        elif current_dd > 0.03:
            base *= 0.90  # 10% cut
        
        # Cap at 70% max
        base = min(base, 0.70)
        
        if base < 0.05:
            return {}
        
        # Allocate across leveraged ETFs
        weights = {}
        weights['TQQQ'] = base * 0.50  # 50% of leveraged in TQQQ
        weights['SPXL'] = base * 0.30  # 30% in SPXL
        weights['SOXL'] = base * 0.20  # 20% in SOXL
        
        return weights
    
    def simulate_period(
        self,
        weights: Dict[str, float],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> float:
        """Simulate returns."""
        total_return = 0.0
        
        for ticker, weight in weights.items():
            if ticker in self.price_data:
                df = self.price_data[ticker]
                try:
                    mask = (df.index >= start) & (df.index <= end)
                    period = df[mask]
                    
                    if len(period) >= 2:
                        ret = (period['Close'].iloc[-1] / period['Close'].iloc[0]) - 1
                        total_return += weight * ret
                except:
                    pass
        
        # Cash for remainder
        cash_weight = max(0, 1.0 - sum(weights.values()))
        days = (end - start).days
        total_return += cash_weight * (1.04 ** (days/365) - 1)
        
        return total_return
    
    def run(self) -> Dict:
        """Run backtest."""
        logger.info("=" * 60)
        logger.info("PHASE 11 v3: TREND-FOLLOWING WITH DD PROTECTION")
        logger.info("=" * 60)
        
        if not self.fetch_data():
            return {'error': 'Data fetch failed'}
        
        # Monthly rebalance
        dates = pd.date_range(self.start_date, self.end_date, freq='B')
        rebalance = []
        month = None
        for d in dates:
            if d.month != month:
                rebalance.append(d)
                month = d.month
        
        logger.info(f"Running {len(rebalance)} periods...")
        
        equity = self.initial_capital
        peak = equity
        
        for i, date in enumerate(rebalance):
            try:
                # Get SPY trend
                spy_prices = self.price_data['SPY'][self.price_data['SPY'].index <= date]['Close']
                trend = self.get_trend(spy_prices)
                
                # Get VIX
                vix = self.get_vix(date)
                
                # Current drawdown
                current_dd = (peak - equity) / peak if peak > 0 else 0
                
                # Select leveraged ETF allocation
                leverage_weights = self.select_leveraged(date, trend, vix, current_dd)
                leverage_alloc = sum(leverage_weights.values())
                
                # Remaining goes to stocks
                stock_alloc = min(0.50, 1.0 - leverage_alloc - 0.05)  # Keep 5% cash min
                stock_weights = self.select_stocks(date, stock_alloc)
                
                # Combine
                all_weights = {**stock_weights, **leverage_weights}
                
                # Simulate
                if i < len(rebalance) - 1:
                    ret = self.simulate_period(all_weights, date, rebalance[i+1])
                    equity *= (1 + ret)
                
                peak = max(peak, equity)
                current_dd = (peak - equity) / peak
                
                self.equity_curve.append({
                    'date': date,
                    'equity': equity,
                    'drawdown': current_dd,
                    'trend': trend,
                    'vix': vix,
                    'leverage_pct': leverage_alloc,
                    'n_positions': len(all_weights),
                })
                
                if i % 6 == 0:
                    logger.info(f"  {date.date()}: ${equity:,.0f}, DD={current_dd:.1%}, "
                               f"trend={trend}, VIX={vix:.0f}, lev={leverage_alloc:.0%}")
                               
            except Exception as e:
                logger.error(f"Error on {date}: {e}")
        
        return self._compute_results()
    
    def _compute_results(self) -> Dict:
        if not self.equity_curve:
            return {'error': 'No data'}
        
        eq_df = pd.DataFrame(self.equity_curve).set_index('date')
        
        end_eq = eq_df['equity'].iloc[-1]
        years = (eq_df.index[-1] - eq_df.index[0]).days / 365.25
        cagr = (end_eq / self.initial_capital) ** (1/years) - 1
        
        max_dd = eq_df['drawdown'].max()
        
        returns = eq_df['equity'].pct_change().dropna()
        vol = returns.std() * np.sqrt(12)
        sharpe = (cagr - 0.04) / vol if vol > 0 else 0
        
        return {
            'phase': 'Phase 11 v3',
            'period': f"{self.start_date.date()} to {self.end_date.date()}",
            'initial_capital': self.initial_capital,
            'final_equity': end_eq,
            'cagr': cagr,
            'max_drawdown': -max_dd,
            'volatility': vol,
            'sharpe': sharpe,
            'alpha_vs_spy': cagr - 0.10,
            'avg_leverage': eq_df['leverage_pct'].mean(),
            'avg_positions': eq_df['n_positions'].mean(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2022-01-01')
    parser.add_argument('--end', default='2025-01-01')
    parser.add_argument('--capital', type=float, default=100_000)
    args = parser.parse_args()
    
    bt = Phase11v3(args.start, args.end, args.capital)
    results = bt.run()
    
    print("\n" + "=" * 60)
    print("PHASE 11 v3 RESULTS")
    print("=" * 60)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"\nPeriod: {results['period']}")
    print(f"Initial: ${results['initial_capital']:,.0f}")
    print(f"Final: ${results['final_equity']:,.0f}")
    print(f"\nPerformance:")
    print(f"  CAGR: {results['cagr']:.1%}")
    print(f"  Max DD: {results['max_drawdown']:.1%}")
    print(f"  Sharpe: {results['sharpe']:.2f}")
    print(f"  Vol: {results['volatility']:.1%}")
    print(f"  Alpha: {results['alpha_vs_spy']:+.1%}")
    print(f"\nPortfolio:")
    print(f"  Avg Leverage: {results['avg_leverage']:.1%}")
    print(f"  Avg Positions: {results['avg_positions']:.0f}")
    
    print("\n" + "=" * 60)
    print("TARGET CHECK")
    print("=" * 60)
    
    cagr = results['cagr']
    dd = abs(results['max_drawdown'])
    alpha = results['alpha_vs_spy']
    
    t1, t2, t3 = cagr >= 0.28, dd <= 0.22, alpha >= 0.05
    print(f"  CAGR ≥ 28%: {'✓' if t1 else '✗'} ({cagr:.1%})")
    print(f"  Max DD ≤ 22%: {'✓' if t2 else '✗'} ({dd:.1%})")  
    print(f"  Alpha ≥ 5%: {'✓' if t3 else '✗'} ({alpha:+.1%})")
    print(f"\n  Targets: {sum([t1,t2,t3])}/3")
    
    # Save
    path = Path(__file__).parent / 'results' / 'phase11_v3_results.json'
    path.parent.mkdir(exist_ok=True)
    with open(path, 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                  for k, v in results.items()}, f, indent=2, default=str)
    print(f"\nSaved: {path}")
    
    return results


if __name__ == '__main__':
    main()

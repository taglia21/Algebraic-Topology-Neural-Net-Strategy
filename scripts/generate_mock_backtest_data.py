#!/usr/bin/env python3
"""
Mock Backtest Data Generator

Generates realistic OHLCV and order flow data for backtesting the V2 trading system
without requiring real market data APIs (Polygon, yfinance, etc.)

Features:
- Realistic price dynamics with volatility clustering (GARCH-like)
- Correlated asset returns (based on historical sector correlations)
- Regime changes (bull/bear/neutral)
- Quote and trade data for OrderFlowAnalyzer testing

Usage:
    python scripts/generate_mock_backtest_data.py
    
Output:
    data/mock_backtest_prices.parquet - OHLCV data for 5 ETFs
    data/mock_order_flow.parquet - Quote/trade data for microstructure analysis
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ETFParams:
    """Parameters for generating realistic ETF price data."""
    ticker: str
    annual_return: float  # Expected annual return
    annual_volatility: float  # Annual volatility
    correlation_group: int  # For cross-asset correlations
    beta: float  # Market beta (relative to SPY)
    starting_price: float  # Initial price


class MockDataGenerator:
    """
    Generates realistic mock market data for backtesting.
    
    Produces correlated OHLCV data with:
    - Volatility clustering (GARCH-like effects)
    - Regime changes (bull/bear/neutral)
    - Realistic intraday ranges
    - Order flow data for microstructure analysis
    """
    
    # ETF parameters based on historical characteristics
    ETF_PARAMS = {
        'SPY': ETFParams('SPY', 0.10, 0.18, 0, 1.00, 400.0),   # S&P 500
        'QQQ': ETFParams('QQQ', 0.12, 0.24, 0, 1.20, 350.0),   # Nasdaq 100
        'IWM': ETFParams('IWM', 0.08, 0.22, 1, 1.15, 180.0),   # Russell 2000
        'XLK': ETFParams('XLK', 0.14, 0.26, 0, 1.25, 150.0),   # Tech Sector
        'XLF': ETFParams('XLF', 0.06, 0.20, 1, 1.10, 35.0),    # Financial Sector
    }
    
    # Cross-asset correlation matrix (simplified)
    CORRELATION_MATRIX = np.array([
        [1.00, 0.85, 0.75, 0.88, 0.65],  # SPY
        [0.85, 1.00, 0.70, 0.92, 0.55],  # QQQ
        [0.75, 0.70, 1.00, 0.68, 0.72],  # IWM
        [0.88, 0.92, 0.68, 1.00, 0.52],  # XLK
        [0.65, 0.55, 0.72, 0.52, 1.00],  # XLF
    ])
    
    def __init__(self, seed: int = 42):
        """
        Initialize mock data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        self.tickers = list(self.ETF_PARAMS.keys())
        
        # Compute Cholesky decomposition for correlated returns
        self.cholesky = np.linalg.cholesky(self.CORRELATION_MATRIX)
        
        logger.info(f"MockDataGenerator initialized with seed={seed}, tickers={self.tickers}")
    
    def _generate_regime_series(self, n_days: int) -> np.ndarray:
        """
        Generate market regime time series (0=bear, 1=neutral, 2=bull).
        
        Uses a simple Markov chain with regime persistence.
        """
        # Transition matrix (regimes tend to persist)
        transition = np.array([
            [0.95, 0.04, 0.01],  # Bear → Bear/Neutral/Bull
            [0.05, 0.90, 0.05],  # Neutral → Bear/Neutral/Bull
            [0.01, 0.04, 0.95],  # Bull → Bear/Neutral/Bull
        ])
        
        regimes = np.zeros(n_days, dtype=int)
        regimes[0] = 1  # Start neutral
        
        for t in range(1, n_days):
            probs = transition[regimes[t-1]]
            regimes[t] = np.random.choice([0, 1, 2], p=probs)
        
        return regimes
    
    def _generate_volatility_series(self, n_days: int, base_vol: float) -> np.ndarray:
        """
        Generate time-varying volatility with clustering (GARCH-like).
        
        Args:
            n_days: Number of trading days
            base_vol: Base annual volatility
        
        Returns:
            Daily volatility series
        """
        daily_base = base_vol / np.sqrt(252)
        
        # GARCH(1,1)-like parameters
        omega = daily_base ** 2 * 0.05
        alpha = 0.10  # Shock persistence
        beta = 0.85   # Volatility persistence
        
        variance = np.zeros(n_days)
        variance[0] = daily_base ** 2
        
        # Generate shocks
        shocks = np.random.randn(n_days)
        
        for t in range(1, n_days):
            variance[t] = omega + alpha * (shocks[t-1] * np.sqrt(variance[t-1])) ** 2 + beta * variance[t-1]
            variance[t] = max(variance[t], (daily_base * 0.3) ** 2)  # Floor
            variance[t] = min(variance[t], (daily_base * 3.0) ** 2)  # Cap
        
        return np.sqrt(variance)
    
    def generate_prices(self, 
                        start_date: str = '2022-01-01',
                        end_date: str = '2025-01-20') -> Dict[str, pd.DataFrame]:
        """
        Generate OHLCV price data for all ETFs.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            Dictionary mapping ticker to DataFrame with OHLCV columns
        """
        # Generate business day index
        dates = pd.bdate_range(start=start_date, end=end_date)
        n_days = len(dates)
        n_assets = len(self.tickers)
        
        logger.info(f"Generating {n_days} days of price data for {n_assets} assets...")
        
        # Generate regimes
        regimes = self._generate_regime_series(n_days)
        regime_multipliers = {0: -1.5, 1: 0.0, 2: 1.5}  # Bear/Neutral/Bull drift adjustment
        
        # Generate correlated returns
        independent_returns = np.random.randn(n_days, n_assets)
        correlated_returns = independent_returns @ self.cholesky.T
        
        price_data = {}
        
        for i, ticker in enumerate(self.tickers):
            params = self.ETF_PARAMS[ticker]
            
            # Generate volatility series
            daily_vol = self._generate_volatility_series(n_days, params.annual_volatility)
            
            # Daily drift (adjusted for regime)
            base_daily_drift = params.annual_return / 252
            regime_adj = np.array([regime_multipliers[r] for r in regimes]) * daily_vol
            daily_drift = base_daily_drift + regime_adj * 0.1
            
            # Generate log returns
            log_returns = daily_drift + daily_vol * correlated_returns[:, i]
            
            # Convert to prices
            log_prices = np.log(params.starting_price) + np.cumsum(log_returns)
            close_prices = np.exp(log_prices)
            
            # Generate OHLV from close
            # High: close * (1 + abs(noise) * vol_factor)
            # Low: close * (1 - abs(noise) * vol_factor)
            # Open: previous close + small gap
            high_noise = np.abs(np.random.randn(n_days)) * daily_vol * 1.5
            low_noise = np.abs(np.random.randn(n_days)) * daily_vol * 1.5
            
            high_prices = close_prices * (1 + high_noise)
            low_prices = close_prices * (1 - low_noise)
            
            # Open = previous close with small gap
            open_prices = np.zeros(n_days)
            open_prices[0] = params.starting_price
            open_prices[1:] = close_prices[:-1] * (1 + np.random.randn(n_days - 1) * 0.002)
            
            # Ensure OHLC consistency
            high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
            low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
            
            # Volume: base + volatility-driven + random
            base_volume = 10_000_000 if ticker == 'SPY' else 5_000_000
            volume = base_volume * (1 + 2 * daily_vol / daily_vol.mean()) * np.random.uniform(0.5, 1.5, n_days)
            volume = volume.astype(int)
            
            # Create DataFrame
            df = pd.DataFrame({
                'Open': open_prices,
                'High': high_prices,
                'Low': low_prices,
                'Close': close_prices,
                'Volume': volume,
                'Regime': regimes,
            }, index=dates)
            
            price_data[ticker] = df
            
            # Log stats
            total_return = (close_prices[-1] / close_prices[0] - 1) * 100
            ann_vol = np.std(log_returns) * np.sqrt(252) * 100
            logger.info(f"  {ticker}: Return={total_return:.1f}%, Vol={ann_vol:.1f}%, "
                       f"Price range: ${close_prices.min():.2f} - ${close_prices.max():.2f}")
        
        return price_data
    
    def generate_order_flow(self, 
                            price_data: Dict[str, pd.DataFrame],
                            samples_per_day: int = 10) -> pd.DataFrame:
        """
        Generate mock quote and trade data for OrderFlowAnalyzer testing.
        
        Args:
            price_data: OHLCV price data from generate_prices()
            samples_per_day: Number of quotes/trades per day per ticker
        
        Returns:
            DataFrame with quote/trade data
        """
        logger.info("Generating order flow data...")
        
        records = []
        
        for ticker, df in price_data.items():
            for date in df.index[-20:]:  # Last 20 days only (memory efficiency)
                row = df.loc[date]
                
                for i in range(samples_per_day):
                    # Quote data
                    mid_price = (row['High'] + row['Low']) / 2
                    spread_bps = np.random.uniform(1, 10)  # 1-10 bps spread
                    spread = mid_price * spread_bps / 10000
                    
                    bid = mid_price - spread / 2
                    ask = mid_price + spread / 2
                    bid_size = np.random.randint(100, 10000)
                    ask_size = np.random.randint(100, 10000)
                    
                    # Timestamp within trading day
                    hour = 9 + int(i * 6.5 / samples_per_day)
                    minute = np.random.randint(0, 60)
                    ts = pd.Timestamp(date) + pd.Timedelta(hours=hour, minutes=minute)
                    
                    records.append({
                        'timestamp': ts,
                        'ticker': ticker,
                        'type': 'quote',
                        'bid': bid,
                        'ask': ask,
                        'bid_size': bid_size,
                        'ask_size': ask_size,
                    })
                    
                    # Trade data
                    trade_price = np.random.uniform(bid, ask)
                    trade_size = np.random.randint(100, 5000)
                    
                    records.append({
                        'timestamp': ts + pd.Timedelta(seconds=np.random.randint(1, 60)),
                        'ticker': ticker,
                        'type': 'trade',
                        'price': trade_price,
                        'size': trade_size,
                        'side': np.random.choice(['buy', 'sell']),
                    })
        
        order_flow_df = pd.DataFrame(records)
        order_flow_df = order_flow_df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Generated {len(order_flow_df)} order flow records")
        return order_flow_df
    
    def save_data(self, 
                  price_data: Dict[str, pd.DataFrame],
                  order_flow_data: pd.DataFrame,
                  output_dir: str = 'data') -> Tuple[str, str]:
        """
        Save generated data to parquet files.
        
        Args:
            price_data: OHLCV price data
            order_flow_data: Order flow data
            output_dir: Output directory
        
        Returns:
            Tuple of (prices_path, order_flow_path)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Combine price data into single DataFrame with MultiIndex
        combined_prices = pd.concat(
            {ticker: df for ticker, df in price_data.items()},
            names=['Ticker', 'Date']
        )
        
        prices_path = os.path.join(output_dir, 'mock_backtest_prices.parquet')
        combined_prices.to_parquet(prices_path)
        logger.info(f"Saved price data to {prices_path}")
        
        order_flow_path = os.path.join(output_dir, 'mock_order_flow.parquet')
        order_flow_data.to_parquet(order_flow_path)
        logger.info(f"Saved order flow data to {order_flow_path}")
        
        return prices_path, order_flow_path
    
    def validate_data(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Validate generated data has realistic statistical properties.
        
        Returns:
            Dictionary of validation metrics per ticker
        """
        logger.info("Validating generated data...")
        
        validation = {}
        
        for ticker, df in price_data.items():
            returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            
            ann_return = returns.mean() * 252
            ann_vol = returns.std() * np.sqrt(252)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0
            
            # Max drawdown
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            max_dd = drawdowns.min()
            
            validation[ticker] = {
                'annual_return': ann_return,
                'annual_volatility': ann_vol,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'n_days': len(df),
                'valid': -2.0 < sharpe < 2.0 and -0.60 < max_dd < 0.0,  # Relaxed validation
            }
            
            status = "✓" if validation[ticker]['valid'] else "✗"
            logger.info(f"  {ticker}: Sharpe={sharpe:.2f}, MaxDD={max_dd:.1%}, Vol={ann_vol:.1%} {status}")
        
        return validation


def main():
    """Generate and save mock backtest data."""
    print("=" * 60)
    print("Mock Backtest Data Generator")
    print("=" * 60)
    
    generator = MockDataGenerator(seed=42)
    
    # Generate price data
    price_data = generator.generate_prices(
        start_date='2022-01-01',
        end_date='2025-01-20'
    )
    
    # Generate order flow data
    order_flow_data = generator.generate_order_flow(price_data)
    
    # Validate
    validation = generator.validate_data(price_data)
    
    # Save
    prices_path, order_flow_path = generator.save_data(price_data, order_flow_data)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total_bars = sum(len(df) for df in price_data.values())
    print(f"Generated {total_bars:,} price bars across {len(price_data)} ETFs")
    print(f"Generated {len(order_flow_data):,} order flow records")
    print(f"Date range: 2022-01-01 to 2025-01-20")
    print(f"\nFiles saved:")
    print(f"  {prices_path}")
    print(f"  {order_flow_path}")
    
    all_valid = all(v['valid'] for v in validation.values())
    print(f"\nValidation: {'PASS ✓' if all_valid else 'FAIL ✗'}")
    
    return 0 if all_valid else 1


if __name__ == '__main__':
    sys.exit(main())

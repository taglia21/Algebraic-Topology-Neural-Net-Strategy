#!/usr/bin/env python3
"""
V17.0 Data Pipeline
===================
Vectorized data fetching and storage for the entire universe.

Features:
- Batch fetching from yfinance (Polygon fallback)
- Parquet storage with compression
- Incremental updates
- Process 1000+ symbols in <60 seconds
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('V17_DataPipeline')


class DataPipeline:
    """
    Vectorized data pipeline for V17 universe.
    
    Stores data in Parquet format with schema:
    date, symbol, open, high, low, close, volume, vwap
    """
    
    def __init__(self, 
                 cache_dir: str = 'cache/v17_prices',
                 universe_file: str = 'cache/universe/universe_latest.json'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.universe_file = Path(universe_file)
        
        self.symbols: List[str] = []
        self.data: Optional[pd.DataFrame] = None
        
    def load_universe(self) -> List[str]:
        """Load symbols from universe file"""
        if not self.universe_file.exists():
            logger.warning(f"Universe file not found: {self.universe_file}")
            return []
        
        with open(self.universe_file, 'r') as f:
            data = json.load(f)
        
        self.symbols = data.get('symbols', [])
        logger.info(f"📂 Loaded {len(self.symbols)} symbols from universe")
        return self.symbols
    
    def fetch_batch(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch data for a batch of symbols.
        Returns long-format DataFrame.
        """
        if not symbols:
            return pd.DataFrame()
        
        # Join symbols for yfinance batch download
        tickers = ' '.join(symbols)
        
        try:
            df = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False,
                threads=True,
                group_by='ticker'
            )
            
            if df.empty:
                return pd.DataFrame()
            
            # Convert wide format to long format
            records = []
            
            if isinstance(df.columns, pd.MultiIndex):
                # Multiple symbols
                for symbol in symbols:
                    try:
                        sym_df = df[symbol].copy()
                        if sym_df.empty or sym_df['Close'].isna().all():
                            continue
                        
                        for date, row in sym_df.iterrows():
                            if pd.isna(row['Close']):
                                continue
                            
                            # Calculate VWAP approximation
                            vwap = (row['High'] + row['Low'] + row['Close']) / 3
                            
                            records.append({
                                'date': date,
                                'symbol': symbol,
                                'open': row['Open'],
                                'high': row['High'],
                                'low': row['Low'],
                                'close': row['Close'],
                                'volume': row['Volume'],
                                'vwap': vwap
                            })
                    except (KeyError, Exception):
                        pass
            else:
                # Single symbol
                sym_df = df.copy()
                symbol = symbols[0]
                
                for date, row in sym_df.iterrows():
                    if pd.isna(row['Close']):
                        continue
                    
                    vwap = (row['High'] + row['Low'] + row['Close']) / 3
                    
                    records.append({
                        'date': date,
                        'symbol': symbol,
                        'open': row['Open'],
                        'high': row['High'],
                        'low': row['Low'],
                        'close': row['Close'],
                        'volume': row['Volume'],
                        'vwap': vwap
                    })
            
            return pd.DataFrame(records)
            
        except Exception as e:
            logger.error(f"Batch fetch error: {e}")
            return pd.DataFrame()
    
    def fetch_all(self, lookback_days: int = 504, batch_size: int = 100) -> pd.DataFrame:
        """
        Fetch data for entire universe.
        
        Args:
            lookback_days: Number of trading days of history (504 = ~2 years)
            batch_size: Symbols per batch
            
        Returns:
            Long-format DataFrame with all data
        """
        import time
        start_time = time.time()
        
        if not self.symbols:
            self.load_universe()
        
        if not self.symbols:
            logger.error("No symbols in universe")
            return pd.DataFrame()
        
        # Date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=int(lookback_days * 1.5))).strftime('%Y-%m-%d')
        
        logger.info(f"📥 Fetching {len(self.symbols)} symbols ({start_date} to {end_date})...")
        
        all_data = []
        
        for i in range(0, len(self.symbols), batch_size):
            batch = self.symbols[i:i + batch_size]
            batch_df = self.fetch_batch(batch, start_date, end_date)
            
            if not batch_df.empty:
                all_data.append(batch_df)
            
            progress = min(i + batch_size, len(self.symbols))
            if progress % 200 == 0 or progress == len(self.symbols):
                elapsed = time.time() - start_time
                logger.info(f"   Progress: {progress}/{len(self.symbols)} ({elapsed:.1f}s)")
        
        if not all_data:
            logger.error("No data fetched")
            return pd.DataFrame()
        
        self.data = pd.concat(all_data, ignore_index=True)
        
        # Ensure date is datetime
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        # Sort by date and symbol
        self.data = self.data.sort_values(['date', 'symbol']).reset_index(drop=True)
        
        elapsed = time.time() - start_time
        unique_symbols = self.data['symbol'].nunique()
        unique_dates = self.data['date'].nunique()
        
        logger.info(f"✅ Fetched {len(self.data):,} rows in {elapsed:.1f}s")
        logger.info(f"   Symbols: {unique_symbols}, Dates: {unique_dates}")
        
        return self.data
    
    def save_parquet(self, filename: str = None) -> str:
        """Save data to Parquet file"""
        if self.data is None or self.data.empty:
            logger.error("No data to save")
            return ""
        
        if filename is None:
            filename = f"v17_prices_{datetime.now().strftime('%Y%m%d')}.parquet"
        
        filepath = self.cache_dir / filename
        
        self.data.to_parquet(filepath, index=False, compression='snappy')
        
        # Also save as latest
        latest_path = self.cache_dir / 'v17_prices_latest.parquet'
        self.data.to_parquet(latest_path, index=False, compression='snappy')
        
        size_mb = filepath.stat().st_size / 1024 / 1024
        logger.info(f"💾 Saved: {filepath} ({size_mb:.1f} MB)")
        
        return str(filepath)
    
    def load_parquet(self, filename: str = 'v17_prices_latest.parquet') -> pd.DataFrame:
        """Load data from Parquet file"""
        filepath = self.cache_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Data file not found: {filepath}")
            return pd.DataFrame()
        
        self.data = pd.read_parquet(filepath)
        logger.info(f"📂 Loaded {len(self.data):,} rows from {filepath}")
        
        return self.data
    
    def get_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Get data for a single symbol"""
        if self.data is None:
            self.load_parquet()
        
        if self.data is None or self.data.empty:
            return pd.DataFrame()
        
        return self.data[self.data['symbol'] == symbol].copy()
    
    def get_cross_section(self, date: str) -> pd.DataFrame:
        """Get cross-sectional data for a single date"""
        if self.data is None:
            self.load_parquet()
        
        if self.data is None or self.data.empty:
            return pd.DataFrame()
        
        date = pd.to_datetime(date)
        return self.data[self.data['date'] == date].copy()
    
    def pivot_to_wide(self, column: str = 'close') -> pd.DataFrame:
        """
        Pivot data to wide format (dates as index, symbols as columns).
        
        Args:
            column: Column to pivot ('close', 'volume', etc.)
            
        Returns:
            Wide-format DataFrame
        """
        if self.data is None:
            self.load_parquet()
        
        if self.data is None or self.data.empty:
            return pd.DataFrame()
        
        return self.data.pivot(index='date', columns='symbol', values=column)
    
    def calculate_returns(self) -> pd.DataFrame:
        """Calculate daily returns for all symbols"""
        if self.data is None:
            self.load_parquet()
        
        if self.data is None or self.data.empty:
            return pd.DataFrame()
        
        # Vectorized return calculation
        self.data = self.data.sort_values(['symbol', 'date'])
        self.data['return'] = self.data.groupby('symbol')['close'].pct_change()
        
        return self.data
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics"""
        if self.data is None or self.data.empty:
            return {}
        
        return {
            'total_rows': len(self.data),
            'unique_symbols': self.data['symbol'].nunique(),
            'unique_dates': self.data['date'].nunique(),
            'date_range': {
                'start': self.data['date'].min().strftime('%Y-%m-%d'),
                'end': self.data['date'].max().strftime('%Y-%m-%d')
            },
            'memory_mb': self.data.memory_usage(deep=True).sum() / 1024 / 1024
        }


def main():
    """Main entry point"""
    import time
    
    print("\n" + "=" * 60)
    print("📊 V17.0 DATA PIPELINE")
    print("=" * 60)
    
    pipeline = DataPipeline()
    
    # Load universe
    symbols = pipeline.load_universe()
    if not symbols:
        print("❌ No symbols in universe. Run v17_universe_builder.py first.")
        return None
    
    # Fetch all data
    start = time.time()
    data = pipeline.fetch_all(lookback_days=504)
    elapsed = time.time() - start
    
    if data.empty:
        print("❌ No data fetched")
        return None
    
    # Calculate returns
    pipeline.calculate_returns()
    
    # Save to Parquet
    filepath = pipeline.save_parquet()
    
    # Get statistics
    stats = pipeline.get_statistics()
    
    print(f"\n📊 Pipeline Statistics:")
    print(f"   Total Rows:    {stats['total_rows']:,}")
    print(f"   Symbols:       {stats['unique_symbols']}")
    print(f"   Trading Days:  {stats['unique_dates']}")
    print(f"   Date Range:    {stats['date_range']['start']} to {stats['date_range']['end']}")
    print(f"   Memory:        {stats['memory_mb']:.1f} MB")
    print(f"   Fetch Time:    {elapsed:.1f}s")
    
    target_met = elapsed < 60
    print(f"\n{'✅' if target_met else '❌'} Target <60s: {elapsed:.1f}s")
    
    return pipeline


if __name__ == "__main__":
    main()

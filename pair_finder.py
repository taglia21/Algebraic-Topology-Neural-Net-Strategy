#!/usr/bin/env python3
"""
Pair Finder — Cointegration-Based Statistical Arbitrage Pair Discovery
=======================================================================

Scans a universe of liquid US equities, groups by GICS sector, and tests
all intra-sector pairs for cointegration using the Engle-Granger two-step
method.  Returns ranked, tradeable pairs with hedge ratios.

Based on the approach used by quantitative market-neutral funds:
  1. Pre-filter by sector (cointegration is more likely within sectors)
  2. Engle-Granger test on log-price spreads
  3. Half-life of mean reversion via Ornstein-Uhlenbeck estimation
  4. Spread volatility check (need enough movement to be tradeable)
  5. Cache results for weekly reuse

Usage:
    from pair_finder import PairFinder, PairFinderConfig

    finder = PairFinder(PairFinderConfig())
    pairs = finder.find_pairs(price_data)  # DataFrame of close prices
    for p in pairs[:10]:
        print(f"{p.sym_a}/{p.sym_b}  pval={p.pvalue:.4f}  HL={p.half_life:.1f}d")
"""

import os
import json
import logging
import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from itertools import combinations

import numpy as np
import pandas as pd

logger = logging.getLogger("pair_finder")

# ---------------------------------------------------------------------------
# Optional imports — statsmodels for cointegration, OLS
# ---------------------------------------------------------------------------
try:
    from statsmodels.tsa.stattools import coint, adfuller
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger.warning(
        "statsmodels not installed. Pair finding will use fallback correlation method. "
        "Install with: pip install statsmodels"
    )


# ============================================================================
# UNIVERSE — 100+ liquid US equities grouped by GICS sector
# ============================================================================

SECTOR_UNIVERSE: Dict[str, List[str]] = {
    "technology": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "CRM", "ADBE", "INTC",
        "AMD", "CSCO", "ORCL", "IBM", "TXN", "QCOM", "AVGO", "NOW",
        "AMAT", "MU", "LRCX", "KLAC",
    ],
    "healthcare": [
        "UNH", "JNJ", "LLY", "ABBV", "PFE", "MRK", "TMO", "ABT",
        "DHR", "BMY", "AMGN", "MDT", "ISRG", "GILD", "CVS", "CI",
    ],
    "financials": [
        "JPM", "GS", "MS", "BAC", "WFC", "C", "BLK", "SCHW",
        "V", "MA", "AXP", "COF", "USB", "PNC", "TFC", "BK",
    ],
    "energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
        "OXY", "DVN", "HES", "HAL",
    ],
    "consumer_discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TJX", "LOW",
        "BKNG", "CMG", "ORLY", "ROST",
    ],
    "consumer_staples": [
        "KO", "PG", "PEP", "COST", "WMT", "PM", "MO", "CL",
        "KMB", "GIS", "KHC", "SYY",
    ],
    "industrials": [
        "CAT", "HON", "GE", "DE", "UPS", "RTX", "BA", "LMT",
        "NOC", "MMM", "EMR", "ITW",
    ],
    "utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL",
    ],
    "materials": [
        "LIN", "SHW", "APD", "ECL", "FCX", "NEM", "NUE", "DOW",
    ],
    "reits": [
        "AMT", "PLD", "CCI", "EQIX", "O", "SPG", "PSA", "DLR",
    ],
}

# Flat list of all symbols in the universe
ALL_SYMBOLS = sorted(set(
    sym for syms in SECTOR_UNIVERSE.values() for sym in syms
))


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PairFinderConfig:
    """Configuration for pair discovery."""
    # Cointegration test thresholds
    max_pvalue: float = 0.05            # Engle-Granger p-value cutoff
    max_half_life_days: int = 30        # Mean reversion must be < 30 days
    min_half_life_days: int = 1         # Filter out noise (< 1 day = probably spurious)
    min_spread_vol: float = 0.005       # Min annualized spread vol (need enough movement)
    max_spread_vol: float = 0.15        # Max spread vol (too volatile = unstable)

    # Lookback for estimation
    lookback_days: int = 252            # 1 year of daily data for cointegration test
    hedge_ratio_lookback: int = 60      # 60-day rolling OLS for hedge ratio

    # Pair trading parameters (exported for strategy_engine)
    entry_z: float = 2.0               # Enter when |z-score| > 2.0
    exit_z: float = 0.5                # Exit when |z-score| < 0.5
    stop_z: float = 4.0                # Stop loss at |z-score| > 4.0
    z_score_lookback: int = 60          # Rolling window for z-score calc

    # Caching
    cache_dir: str = "cache"
    cache_ttl_days: int = 7             # Rerun weekly

    # Filtering
    min_price: float = 10.0             # Skip penny stocks
    min_avg_volume: int = 500_000       # Min 500K avg daily volume
    max_pairs_per_sector: int = 5       # Top 5 pairs per sector
    max_total_pairs: int = 25           # Cap total pairs


@dataclass
class CointegrationResult:
    """Result of a cointegration test between two stocks."""
    sym_a: str
    sym_b: str
    sector: str
    pvalue: float                       # Engle-Granger p-value (lower = better)
    hedge_ratio: float                  # β: units of B to short per unit of A
    half_life: float                    # Ornstein-Uhlenbeck half-life in days
    spread_mean: float                  # Long-run mean of the spread
    spread_std: float                   # Std dev of the spread
    spread_vol_annual: float            # Annualized spread volatility
    current_z_score: float              # Current z-score of the spread
    adf_stat: float                     # ADF test statistic on the spread
    correlation: float                  # Price correlation (for comparison)
    last_updated: str = ""
    is_tradeable: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# PAIR FINDER
# ============================================================================

class PairFinder:
    """
    Discovers cointegrated pairs from a universe of stocks.

    Methodology:
      1. Group stocks by GICS sector (cointegration within sectors is more stable)
      2. For each intra-sector pair, run Engle-Granger cointegration test
      3. Filter by p-value, half-life, spread volatility
      4. Rank by combined score: low p-value + fast mean reversion + good vol
      5. Cache results for weekly reuse
    """

    def __init__(self, config: PairFinderConfig = None):
        self.cfg = config or PairFinderConfig()
        self._cache_path = Path(self.cfg.cache_dir) / "cointegrated_pairs.json"
        Path(self.cfg.cache_dir).mkdir(exist_ok=True)
        self._pairs: List[CointegrationResult] = []

    @property
    def pairs(self) -> List[CointegrationResult]:
        """Get current list of tradeable pairs."""
        return self._pairs

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def find_pairs(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        force_refresh: bool = False,
    ) -> List[CointegrationResult]:
        """
        Find cointegrated pairs from a DataFrame of close prices.

        Parameters
        ----------
        price_data : pd.DataFrame
            Columns = ticker symbols, rows = dates, values = adjusted close prices.
            Must have at least `lookback_days` rows.
        volume_data : pd.DataFrame, optional
            Same shape as price_data but with daily volumes for liquidity filtering.
        force_refresh : bool
            If True, ignore cache and recompute.

        Returns
        -------
        List[CointegrationResult]
            Ranked list of tradeable pairs. Best pairs first.
        """
        # Check cache
        if not force_refresh:
            cached = self._load_cache(price_data)
            if cached is not None:
                self._pairs = cached
                logger.info(f"Loaded {len(cached)} pairs from cache")
                return cached

        if not HAS_STATSMODELS:
            logger.warning("Using fallback correlation method (statsmodels not available)")
            return self._fallback_correlation_pairs(price_data)

        # Pre-filter symbols by price and volume
        valid_symbols = self._filter_symbols(price_data, volume_data)
        logger.info(f"Valid symbols after pre-filter: {len(valid_symbols)}")

        # Group by sector
        sector_groups = self._group_by_sector(valid_symbols)

        # Test all intra-sector pairs
        all_results: List[CointegrationResult] = []
        total_tested = 0

        for sector, symbols in sector_groups.items():
            if len(symbols) < 2:
                continue

            sector_pairs = list(combinations(symbols, 2))
            logger.info(f"Testing {len(sector_pairs)} pairs in {sector}")

            sector_results = []
            for sym_a, sym_b in sector_pairs:
                total_tested += 1
                result = self._test_pair(sym_a, sym_b, sector, price_data)
                if result is not None and result.is_tradeable:
                    sector_results.append(result)

            # Keep top N pairs per sector, ranked by score
            sector_results.sort(key=lambda r: self._pair_score(r), reverse=True)
            all_results.extend(sector_results[:self.cfg.max_pairs_per_sector])

        # Global ranking and cap
        all_results.sort(key=lambda r: self._pair_score(r), reverse=True)
        all_results = all_results[:self.cfg.max_total_pairs]

        logger.info(
            f"Found {len(all_results)} tradeable pairs from {total_tested} tested"
        )

        # Cache results
        self._save_cache(all_results, price_data)
        self._pairs = all_results
        return all_results

    # ------------------------------------------------------------------
    # Cointegration test for a single pair
    # ------------------------------------------------------------------

    def _test_pair(
        self,
        sym_a: str,
        sym_b: str,
        sector: str,
        price_data: pd.DataFrame,
    ) -> Optional[CointegrationResult]:
        """
        Run Engle-Granger cointegration test on a pair.

        The Engle-Granger test:
          1. Regress log(A) on log(B) to get hedge ratio β
          2. Compute spread = log(A) - β * log(B)
          3. Test spread for stationarity (ADF test)
          4. If stationary → the pair is cointegrated
        """
        if sym_a not in price_data.columns or sym_b not in price_data.columns:
            return None

        # Extract price series, drop NaN
        pa = price_data[sym_a].dropna()
        pb = price_data[sym_b].dropna()

        # Align on common dates
        common_idx = pa.index.intersection(pb.index)
        if len(common_idx) < self.cfg.lookback_days * 0.8:
            return None  # Not enough overlapping data

        pa = pa.loc[common_idx].values.astype(float)
        pb = pb.loc[common_idx].values.astype(float)

        # Use only the lookback window
        pa = pa[-self.cfg.lookback_days:]
        pb = pb[-self.cfg.lookback_days:]

        if len(pa) < 60 or np.any(pa <= 0) or np.any(pb <= 0):
            return None

        # Log prices for cointegration test
        log_a = np.log(pa)
        log_b = np.log(pb)

        try:
            # Step 1: Engle-Granger cointegration test
            coint_stat, pvalue, crit_values = coint(log_a, log_b)

            if pvalue > self.cfg.max_pvalue:
                return None  # Not cointegrated at our significance level

            # Step 2: OLS regression for hedge ratio
            # spread = log(A) - β * log(B) - α
            X = add_constant(log_b)
            model = OLS(log_a, X).fit()
            alpha = model.params[0]  # Intercept
            beta = model.params[1]   # Hedge ratio

            # Step 3: Compute the spread
            spread = log_a - beta * log_b - alpha

            # Step 4: Half-life via Ornstein-Uhlenbeck
            half_life = self._compute_half_life(spread)
            if half_life < self.cfg.min_half_life_days or half_life > self.cfg.max_half_life_days:
                return None

            # Step 5: Spread statistics
            spread_mean = float(np.mean(spread))
            spread_std = float(np.std(spread))
            if spread_std < 1e-10:
                return None

            # Annualized spread volatility
            spread_returns = np.diff(spread)
            spread_vol_annual = float(np.std(spread_returns) * np.sqrt(252))

            if spread_vol_annual < self.cfg.min_spread_vol:
                return None  # Not enough movement to trade
            if spread_vol_annual > self.cfg.max_spread_vol:
                return None  # Too volatile, relationship is unstable

            # Step 6: Current z-score
            # Use rolling lookback for z-score (more responsive)
            recent_spread = spread[-self.cfg.z_score_lookback:]
            z_mean = float(np.mean(recent_spread))
            z_std = float(np.std(recent_spread))
            current_z = float((spread[-1] - z_mean) / z_std) if z_std > 1e-10 else 0.0

            # Step 7: ADF on spread (redundant with coint, but useful for ranking)
            adf_result = adfuller(spread, maxlag=int(np.sqrt(len(spread))))
            adf_stat = float(adf_result[0])

            # Correlation (informational, not used for trading)
            correlation = float(np.corrcoef(pa, pb)[0, 1])

            return CointegrationResult(
                sym_a=sym_a,
                sym_b=sym_b,
                sector=sector,
                pvalue=float(pvalue),
                hedge_ratio=float(beta),
                half_life=float(half_life),
                spread_mean=spread_mean,
                spread_std=spread_std,
                spread_vol_annual=spread_vol_annual,
                current_z_score=current_z,
                adf_stat=adf_stat,
                correlation=correlation,
                last_updated=datetime.now().isoformat(),
                is_tradeable=True,
            )

        except Exception as e:
            logger.debug(f"Cointegration test failed for {sym_a}/{sym_b}: {e}")
            return None

    # ------------------------------------------------------------------
    # Half-life estimation (Ornstein-Uhlenbeck)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_half_life(spread: np.ndarray) -> float:
        """
        Estimate the half-life of mean reversion using the Ornstein-Uhlenbeck
        process: dS = θ(μ - S)dt + σdW

        We regress Δspread on lagged spread:
          Δspread_t = a + b * spread_{t-1} + ε
          θ = -b  →  half_life = -ln(2) / ln(1 + b) ≈ ln(2) / θ

        Returns half-life in days. Returns inf if not mean-reverting.
        """
        lagged = spread[:-1]
        delta = np.diff(spread)

        if len(lagged) < 20:
            return float('inf')

        # OLS: delta = a + b * lagged
        X = add_constant(lagged)
        model = OLS(delta, X).fit()
        b = model.params[1]

        if b >= 0:
            # Not mean-reverting (spread is a random walk or trending)
            return float('inf')

        # half_life = -ln(2) / ln(1 + b)
        # For small b, this ≈ ln(2) / |b|
        try:
            half_life = -np.log(2) / np.log(1 + b)
        except (ValueError, RuntimeWarning):
            half_life = np.log(2) / abs(b)  # Approximation

        return max(float(half_life), 0.1)

    # ------------------------------------------------------------------
    # Z-score calculation (used by strategy engine at runtime)
    # ------------------------------------------------------------------

    def compute_pair_z_score(
        self,
        pair: CointegrationResult,
        price_a: np.ndarray,
        price_b: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute current z-score for a pair using rolling hedge ratio.

        Parameters
        ----------
        pair : CointegrationResult
            The pair to evaluate.
        price_a, price_b : np.ndarray
            Recent price arrays (at least z_score_lookback days).

        Returns
        -------
        z_score : float
            Current z-score of the spread.
        hedge_ratio : float
            Current rolling hedge ratio.
        """
        if len(price_a) < self.cfg.hedge_ratio_lookback or len(price_b) < self.cfg.hedge_ratio_lookback:
            return 0.0, pair.hedge_ratio

        log_a = np.log(price_a[-self.cfg.hedge_ratio_lookback:])
        log_b = np.log(price_b[-self.cfg.hedge_ratio_lookback:])

        # Rolling OLS for hedge ratio (adapts to changing relationship)
        try:
            X = add_constant(log_b) if HAS_STATSMODELS else np.column_stack([np.ones(len(log_b)), log_b])
            if HAS_STATSMODELS:
                model = OLS(log_a, X).fit()
                alpha = model.params[0]
                beta = model.params[1]
            else:
                # Fallback: numpy least-squares
                beta, alpha = np.polyfit(log_b, log_a, 1)
        except Exception:
            return 0.0, pair.hedge_ratio

        # Spread using rolling hedge ratio
        spread = np.log(price_a) - beta * np.log(price_b) - alpha

        # Z-score over lookback window
        recent = spread[-self.cfg.z_score_lookback:]
        z_mean = float(np.mean(recent))
        z_std = float(np.std(recent))

        if z_std < 1e-10:
            return 0.0, float(beta)

        z_score = float((spread[-1] - z_mean) / z_std)
        return z_score, float(beta)

    # ------------------------------------------------------------------
    # Scoring & ranking
    # ------------------------------------------------------------------

    def _pair_score(self, result: CointegrationResult) -> float:
        """
        Score a pair for ranking. Higher = better.

        Factors:
          - Low p-value (strong cointegration evidence)
          - Fast half-life (quick mean reversion)
          - Moderate spread vol (tradeable but not unstable)
          - Strong ADF statistic (more negative = more stationary)
        """
        # p-value score: 0.01 → 1.0, 0.05 → 0.0
        pval_score = max(0, 1.0 - (result.pvalue / self.cfg.max_pvalue))

        # Half-life score: 5 days → 1.0, 30 days → 0.0
        hl_score = max(0, 1.0 - (result.half_life - 5) / 25)

        # Spread vol score: peaks at 0.05 (5% annualized)
        # Too low = no profit; too high = unstable
        vol_ideal = 0.05
        vol_score = max(0, 1.0 - abs(result.spread_vol_annual - vol_ideal) / vol_ideal)

        # ADF score: more negative = better (use absolute value)
        adf_score = min(1.0, abs(result.adf_stat) / 5.0)

        # Weighted combination
        score = (
            0.35 * pval_score +    # Statistical significance is most important
            0.30 * hl_score +      # Fast reversion = more trades = more edge
            0.20 * vol_score +     # Need the right amount of volatility
            0.15 * adf_score       # ADF confirms stationarity
        )
        return score

    # ------------------------------------------------------------------
    # Symbol filtering
    # ------------------------------------------------------------------

    def _filter_symbols(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
    ) -> List[str]:
        """Filter symbols by price and volume thresholds."""
        valid = []
        for sym in price_data.columns:
            prices = price_data[sym].dropna()
            if len(prices) < self.cfg.lookback_days * 0.5:
                continue  # Not enough history

            last_price = float(prices.iloc[-1])
            if last_price < self.cfg.min_price:
                continue  # Too cheap (penny stock risk, wide spreads)

            # Volume filter (if data is available)
            if volume_data is not None and sym in volume_data.columns:
                avg_vol = float(volume_data[sym].dropna().tail(20).mean())
                if avg_vol < self.cfg.min_avg_volume:
                    continue  # Not liquid enough

            valid.append(sym)
        return valid

    def _group_by_sector(self, symbols: List[str]) -> Dict[str, List[str]]:
        """Group symbols by sector. Only include symbols present in our sector map."""
        groups: Dict[str, List[str]] = {}
        sym_set = set(symbols)

        for sector, sector_syms in SECTOR_UNIVERSE.items():
            in_universe = [s for s in sector_syms if s in sym_set]
            if len(in_universe) >= 2:
                groups[sector] = in_universe

        return groups

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _data_hash(self, price_data: pd.DataFrame) -> str:
        """Create a hash of the price data for cache invalidation."""
        # Hash based on columns, shape, and last 5 rows
        info = f"{sorted(price_data.columns.tolist())}_{price_data.shape}"
        if len(price_data) >= 5:
            info += str(price_data.tail(5).values.tobytes()[:200])
        return hashlib.md5(info.encode()).hexdigest()[:12]

    def _load_cache(self, price_data: pd.DataFrame) -> Optional[List[CointegrationResult]]:
        """Load cached pairs if still valid."""
        if not self._cache_path.exists():
            return None

        try:
            with open(self._cache_path, "r") as f:
                cache = json.load(f)

            # Check TTL
            cached_time = datetime.fromisoformat(cache.get("timestamp", "2000-01-01"))
            if (datetime.now() - cached_time).days > self.cfg.cache_ttl_days:
                logger.info("Pair cache expired, recomputing")
                return None

            # Check data hash (has the universe changed?)
            if cache.get("data_hash") != self._data_hash(price_data):
                logger.info("Price data changed, recomputing pairs")
                return None

            pairs = [CointegrationResult(**p) for p in cache.get("pairs", [])]
            return pairs

        except Exception as e:
            logger.debug(f"Cache load failed: {e}")
            return None

    def _save_cache(self, pairs: List[CointegrationResult], price_data: pd.DataFrame):
        """Save pairs to cache."""
        try:
            cache = {
                "timestamp": datetime.now().isoformat(),
                "data_hash": self._data_hash(price_data),
                "n_pairs": len(pairs),
                "pairs": [p.to_dict() for p in pairs],
            }
            with open(self._cache_path, "w") as f:
                json.dump(cache, f, indent=2)
            logger.info(f"Saved {len(pairs)} pairs to cache")
        except Exception as e:
            logger.debug(f"Cache save failed: {e}")

    # ------------------------------------------------------------------
    # Fallback: correlation-based pairing (when statsmodels unavailable)
    # ------------------------------------------------------------------

    def _fallback_correlation_pairs(
        self, price_data: pd.DataFrame
    ) -> List[CointegrationResult]:
        """
        Fallback pair-finding using correlation + spread analysis.
        Less rigorous than Engle-Granger but still useful.
        """
        valid = self._filter_symbols(price_data)
        groups = self._group_by_sector(valid)
        results = []

        for sector, symbols in groups.items():
            for sym_a, sym_b in combinations(symbols, 2):
                if sym_a not in price_data.columns or sym_b not in price_data.columns:
                    continue

                pa = price_data[sym_a].dropna().values.astype(float)
                pb = price_data[sym_b].dropna().values.astype(float)

                n = min(len(pa), len(pb), self.cfg.lookback_days)
                if n < 60:
                    continue
                pa, pb = pa[-n:], pb[-n:]
                if np.any(pa <= 0) or np.any(pb <= 0):
                    continue

                log_a, log_b = np.log(pa), np.log(pb)

                # Simple OLS hedge ratio
                beta, alpha = np.polyfit(log_b, log_a, 1)
                spread = log_a - beta * log_b - alpha

                # Check if spread looks mean-reverting (heuristic)
                spread_returns = np.diff(spread)
                autocorr = np.corrcoef(spread[:-1], spread[1:])[0, 1]
                if autocorr >= 0:
                    continue  # Not mean-reverting

                half_life = -np.log(2) / np.log(abs(autocorr)) if abs(autocorr) < 1 else float('inf')
                if half_life > self.cfg.max_half_life_days:
                    continue

                spread_std = float(np.std(spread))
                spread_vol = float(np.std(spread_returns) * np.sqrt(252))
                corr = float(np.corrcoef(pa, pb)[0, 1])

                if corr < 0.5:
                    continue  # Not correlated enough for pairs trading

                z_recent = spread[-self.cfg.z_score_lookback:]
                z_mean = float(np.mean(z_recent))
                z_std = float(np.std(z_recent))
                z_score = float((spread[-1] - z_mean) / z_std) if z_std > 1e-10 else 0.0

                results.append(CointegrationResult(
                    sym_a=sym_a, sym_b=sym_b, sector=sector,
                    pvalue=0.05,  # Placeholder since we didn't do formal test
                    hedge_ratio=float(beta),
                    half_life=float(half_life),
                    spread_mean=float(np.mean(spread)),
                    spread_std=spread_std,
                    spread_vol_annual=spread_vol,
                    current_z_score=z_score,
                    adf_stat=-2.0,  # Placeholder
                    correlation=corr,
                    last_updated=datetime.now().isoformat(),
                    is_tradeable=True,
                ))

        results.sort(key=lambda r: self._pair_score(r), reverse=True)
        self._pairs = results[:self.cfg.max_total_pairs]
        return self._pairs

    # ------------------------------------------------------------------
    # Utility: fetch prices from Alpaca and build DataFrame
    # ------------------------------------------------------------------

    @staticmethod
    def fetch_prices_from_alpaca(
        symbols: List[str] = None,
        lookback_days: int = 300,
        headers: dict = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch historical daily prices from Alpaca for pair finding.

        Returns (price_df, volume_df) with dates as index, symbols as columns.
        """
        import requests as _req

        if symbols is None:
            symbols = ALL_SYMBOLS

        if headers is None:
            key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY", "")
            secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY", "")
            headers = {
                "APCA-API-KEY-ID": key,
                "APCA-API-SECRET-KEY": secret,
            }

        data_base = "https://data.alpaca.markets"
        end = datetime.now()
        start = end - timedelta(days=lookback_days + 50)  # Buffer for weekends/holidays

        price_dict: Dict[str, pd.Series] = {}
        volume_dict: Dict[str, pd.Series] = {}

        for sym in symbols:
            try:
                resp = _req.get(
                    f"{data_base}/v2/stocks/{sym}/bars",
                    headers=headers,
                    params={
                        "timeframe": "1Day",
                        "start": start.strftime("%Y-%m-%d"),
                        "end": end.strftime("%Y-%m-%d"),
                        "limit": 10000,
                        "adjustment": "all",  # Handles splits + dividends
                    },
                    timeout=15,
                )
                if resp.status_code != 200:
                    logger.debug(f"Failed to fetch {sym}: {resp.status_code}")
                    continue

                bars = resp.json().get("bars", [])
                if not bars or len(bars) < 60:
                    continue

                dates = [b["t"][:10] for b in bars]  # YYYY-MM-DD
                closes = [float(b["c"]) for b in bars]
                volumes = [int(b["v"]) for b in bars]

                price_dict[sym] = pd.Series(closes, index=pd.to_datetime(dates), name=sym)
                volume_dict[sym] = pd.Series(volumes, index=pd.to_datetime(dates), name=sym)

            except Exception as e:
                logger.debug(f"Error fetching {sym}: {e}")
                continue

            time.sleep(0.12)  # Rate limit: ~8 req/sec

        if not price_dict:
            logger.error("No price data fetched!")
            return pd.DataFrame(), pd.DataFrame()

        price_df = pd.DataFrame(price_dict).sort_index()
        volume_df = pd.DataFrame(volume_dict).sort_index()

        logger.info(
            f"Fetched {len(price_df.columns)} symbols, "
            f"{len(price_df)} trading days"
        )
        return price_df, volume_df


# ============================================================================
# MAIN — standalone CLI for pair discovery
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    print("=" * 70)
    print("PAIR FINDER — Cointegration-Based Statistical Arbitrage")
    print("=" * 70)

    config = PairFinderConfig()
    finder = PairFinder(config)

    print(f"\nFetching prices for {len(ALL_SYMBOLS)} symbols...")
    price_df, volume_df = PairFinder.fetch_prices_from_alpaca(lookback_days=300)

    if price_df.empty:
        print("ERROR: No price data. Check Alpaca API credentials.")
        sys.exit(1)

    print(f"Got {len(price_df.columns)} symbols with {len(price_df)} days of data\n")

    pairs = finder.find_pairs(price_df, volume_df, force_refresh=True)

    print(f"\n{'='*70}")
    print(f"TRADEABLE PAIRS: {len(pairs)}")
    print(f"{'='*70}")
    print(f"{'Pair':<20} {'Sector':<15} {'p-val':>8} {'HL':>6} {'SpreadVol':>10} {'Z-score':>8} {'β':>8}")
    print("-" * 75)

    for p in pairs:
        print(
            f"{p.sym_a}/{p.sym_b:<13} {p.sector:<15} "
            f"{p.pvalue:>8.4f} {p.half_life:>5.1f}d "
            f"{p.spread_vol_annual:>9.4f} {p.current_z_score:>+7.2f} "
            f"{p.hedge_ratio:>8.3f}"
        )

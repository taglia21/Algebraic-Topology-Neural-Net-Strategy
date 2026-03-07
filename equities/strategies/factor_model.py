"""
equities/strategies/factor_model.py
=====================================
Multi-Factor Alpha Model for the ATNN trading system.

Overview
--------
This module implements a four-factor composite score model combining:

    - Quality:   Gross profitability (Novy-Marx) — gross_profits / total_assets
    - Value:     Earnings yield (E/P, inverse P/E)
    - Low-Vol:   Negative 60-day realised volatility (long low, short high)
    - Momentum:  12-1 month total return

Each factor is cross-sectionally z-scored, winsorized at ±3σ, then combined
into a composite score with configurable weights.

Fundamental Data
----------------
Real fundamental data (P/E, gross profit, total assets) requires an external
data provider (SEC EDGAR, Alpaca fundamentals, etc.).  To enable the strategy
to run on pure price data initially, a ``FundamentalDataStub`` class is
provided that synthesises estimated metrics from price behaviour.

⚠️  STUB: The FundamentalDataStub returns synthetic/estimated values.
    Replace with real fundamental data before production use.

Pipeline
--------
1. ``z_score_factors(factor_data)``
   - Winsorize each factor column at ±3σ.
   - Z-score cross-sectionally within each factor.

2. ``composite_score(factors, weights)``
   - Weighted average of z-scored factors.

3. ``generate_signals(price_data, fundamental_data, regime_state)``
   - Long stocks with composite z-score > 1.0.
   - Short stocks with composite z-score < -1.0.
   - Factor timing: widen value weight when value spread is wide.
   - Regime adjustment: in BEAR, overweight Quality and Low-Vol.

References
----------
- Novy-Marx (2013), Journal of Financial Economics — Gross Profitability
- Fama & French (1992, 1993) — Multi-factor model
- Asness (1994), Frazzini (2006) — Quality Minus Junk / Betting Against Beta
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from pathlib import Path

import numpy as np
import pandas as pd

from core.config import FactorModelConfig, get_config
from core.logger import TradeLogger, get_trade_logger
from core.regime_detector import Regime, RegimeState
from equities.models import Signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fundamental Data Stub
# ---------------------------------------------------------------------------

class FundamentalDataProvider:
    """Fundamental data provider using yfinance for real financial data.

    Fetches actual earnings yield (inverse P/E) and gross profitability
    (gross profit / total assets) from yfinance.  Falls back to a
    price-based heuristic for tickers that lack fundamental coverage.

    Data is cached per-symbol to avoid redundant API calls during backtest.
    """

    def __init__(self, lookback: int = 63, cache_dir: str = "data/cache/fundamentals") -> None:
        """
        Parameters
        ----------
        lookback:
            Rolling window for the price-based fallback estimator (days).
        cache_dir:
            Directory to cache fetched fundamental data.
        """
        self._lookback = lookback
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[str, float]] = {}
        self._fetched: bool = False

    def _fetch_fundamentals(self, symbol: str) -> Dict[str, float]:
        """Fetch real fundamental data for a single symbol via yfinance.

        Returns
        -------
        dict with keys: earnings_yield, gross_profitability.
        Values are NaN if data unavailable.
        """
        # Check disk cache first
        cache_file = self._cache_dir / f"{symbol}.json"
        if cache_file.exists():
            import json
            try:
                with open(cache_file) as fh:
                    data = json.load(fh)
                    # Cache is valid for 30 days
                    import time
                    if time.time() - data.get("_ts", 0) < 30 * 86400:
                        return data
            except Exception:
                pass

        result: Dict[str, float] = {
            "earnings_yield": float("nan"),
            "gross_profitability": float("nan"),
        }

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}

            # Earnings yield = 1 / trailing P/E
            trailing_pe = info.get("trailingPE")
            if trailing_pe and isinstance(trailing_pe, (int, float)) and trailing_pe > 0:
                result["earnings_yield"] = 1.0 / trailing_pe

            # Forward earnings yield as supplement
            forward_pe = info.get("forwardPE")
            if np.isnan(result["earnings_yield"]) and forward_pe and isinstance(forward_pe, (int, float)) and forward_pe > 0:
                result["earnings_yield"] = 1.0 / forward_pe

            # Gross profitability = gross profit / total assets (Novy-Marx factor)
            gross_profit = info.get("grossProfits")
            total_assets = info.get("totalAssets")
            if gross_profit and total_assets and total_assets > 0:
                result["gross_profitability"] = gross_profit / total_assets
            else:
                # Alternative: profit margins as proxy
                profit_margin = info.get("profitMargins")
                if profit_margin and isinstance(profit_margin, (int, float)):
                    result["gross_profitability"] = max(profit_margin, 0.0)

            # Save to disk cache
            import json, time
            cache_data = {**result, "_ts": time.time()}
            with open(cache_file, "w") as fh:
                json.dump(cache_data, fh)

        except Exception as exc:
            logger.debug(f"yfinance fetch failed for {symbol}: {exc}")

        return result

    def get_factor_data(
        self,
        price_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return a symbol-indexed DataFrame with fundamental factors.

        First attempts real fundamental data via yfinance.  For any symbol
        where real data is unavailable, falls back to price-based heuristics.

        Parameters
        ----------
        price_data:
            Wide-format close price DataFrame.  Index is DatetimeIndex.

        Returns
        -------
        pd.DataFrame with index = symbol and columns:
            ``earnings_yield``, ``gross_profitability``.
        """
        results: Dict[str, Dict[str, float]] = {}
        returns = np.log(price_data / price_data.shift(1))

        for sym in price_data.columns:
            if sym in ("SPY", "QQQ", "IWM"):
                continue

            sym_prices = price_data[sym].dropna()
            sym_rets = returns[sym].dropna()

            if len(sym_prices) < self._lookback:
                continue

            # Try real fundamentals first (cached)
            if sym not in self._cache:
                self._cache[sym] = self._fetch_fundamentals(sym)

            fund = self._cache[sym]
            ey = fund.get("earnings_yield", float("nan"))
            gp = fund.get("gross_profitability", float("nan"))

            # Fall back to price-based heuristics where real data is missing
            if np.isnan(ey):
                # Value proxy: inverse of price-to-52-week-high ratio.
                # Stocks trading far below their 52-week high are "cheaper".
                # This aligns better with value investing intuition than
                # the old rolling-mean / price approach (which was really
                # just a mean-reversion signal).
                latest_price = float(sym_prices.iloc[-1])
                high_252 = float(sym_prices.iloc[-min(252, len(sym_prices)):].max())
                if high_252 > 0 and latest_price > 0:
                    # Ratio in (0, 1]; closer to 0 = cheaper relative to peak
                    price_to_high = latest_price / high_252
                    # Invert so higher = better value (cheaper stock)
                    ey = (1.0 / max(price_to_high, 0.01)) - 1.0
                else:
                    ey = 0.0

            if np.isnan(gp):
                # Quality proxy: inverse of realised vol. Lower vol implies
                # more stable earnings (Novy-Marx quality is correlated with
                # lower vol in cross-section).
                vol_60 = float(
                    sym_rets.rolling(60, min_periods=30).std().iloc[-1]
                    * np.sqrt(252)
                ) if len(sym_rets) >= 30 else 0.20
                gp = 1.0 / max(1.0 + vol_60, 0.01)

            results[sym] = {
                "earnings_yield": ey,
                "gross_profitability": gp,
            }

        df = pd.DataFrame.from_dict(results, orient="index")
        df.index.name = "symbol"
        return df


# Keep the old name as an alias for backwards compatibility
FundamentalDataStub = FundamentalDataProvider


# ---------------------------------------------------------------------------
# Factor computation helpers
# ---------------------------------------------------------------------------

def _winsorize(series: pd.Series, n_sigma: float = 3.0) -> pd.Series:
    """Winsorize a series at ±n_sigma standard deviations.

    Values outside [μ − n×σ, μ + n×σ] are clipped to the boundary.

    Parameters
    ----------
    series:
        Numeric series to winsorize.
    n_sigma:
        Number of standard deviations for the clipping bounds.

    Returns
    -------
    pd.Series with outliers clipped.
    """
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return series
    lower = mean - n_sigma * std
    upper = mean + n_sigma * std
    return series.clip(lower=lower, upper=upper)


def _cross_section_zscore(series: pd.Series) -> pd.Series:
    """Compute cross-sectional z-score.

    Parameters
    ----------
    series:
        Cross-section of factor values (one value per stock at a given date).

    Returns
    -------
    Z-scored series with mean ≈ 0 and std ≈ 1.
    """
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


# ---------------------------------------------------------------------------
# Main strategy class
# ---------------------------------------------------------------------------

class FactorModelStrategy:
    """Multi-factor alpha model combining Quality, Value, Low-Vol, and Momentum.

    Factors are cross-sectionally z-scored and combined into a composite score.
    Long stocks with composite z > entry_z, short stocks with composite z < -entry_z.

    Parameters
    ----------
    config:
        ``FactorModelConfig`` from the system configuration.
    trade_logger:
        ``TradeLogger`` for audit logging.
    fundamental_provider:
        Provider of fundamental data.  If *None*, uses :class:`FundamentalDataStub`.
        For production, replace with a real fundamental data adapter.

    Usage
    -----
    >>> strategy = FactorModelStrategy()
    >>> signals = strategy.generate_signals(price_data, None, regime_state)
    """

    STRATEGY_NAME: str = "factor_model"

    def __init__(
        self,
        config: Optional[FactorModelConfig] = None,
        trade_logger: Optional[TradeLogger] = None,
        fundamental_provider: Optional[FundamentalDataStub] = None,
    ) -> None:
        cfg = config or get_config().strategy.factor_model
        self._cfg = cfg
        self._log = trade_logger or get_trade_logger()
        self._fund_provider = fundamental_provider or FundamentalDataStub(
            lookback=cfg.lookback_days
        )

    # ------------------------------------------------------------------
    # Factor construction
    # ------------------------------------------------------------------

    def _compute_quality_factor(
        self,
        price_data: pd.DataFrame,
        fundamental_data: Optional[pd.DataFrame],
    ) -> pd.Series:
        """Compute quality factor (gross profitability / total assets).

        Uses real fundamental data if available, otherwise falls back to the
        FundamentalDataStub's ``gross_profitability`` estimate.

        Parameters
        ----------
        price_data:
            Wide-format close price DataFrame.
        fundamental_data:
            Optional DataFrame with ``gross_profitability`` column indexed by symbol.

        Returns
        -------
        pd.Series: quality score indexed by symbol.
        """
        if fundamental_data is not None and "gross_profitability" in fundamental_data.columns:
            return fundamental_data["gross_profitability"].dropna()

        # Use stub
        stub_data = self._fund_provider.get_factor_data(price_data)
        if "gross_profitability" in stub_data.columns:
            return stub_data["gross_profitability"].dropna()

        # Absolute fallback: 1.0 for all (neutral)
        return pd.Series(
            1.0,
            index=[c for c in price_data.columns if c not in ("SPY", "QQQ", "IWM")],
            name="quality",
        )

    def _compute_value_factor(
        self,
        price_data: pd.DataFrame,
        fundamental_data: Optional[pd.DataFrame],
    ) -> pd.Series:
        """Compute value factor (earnings yield = E/P).

        Uses real fundamental data if available, otherwise uses the stub's
        earnings_yield proxy.

        Parameters
        ----------
        price_data:
            Wide-format close price DataFrame.
        fundamental_data:
            Optional DataFrame with ``earnings_yield`` column indexed by symbol.

        Returns
        -------
        pd.Series: value score indexed by symbol.
        """
        if fundamental_data is not None and "earnings_yield" in fundamental_data.columns:
            return fundamental_data["earnings_yield"].dropna()

        # Stub
        stub_data = self._fund_provider.get_factor_data(price_data)
        if "earnings_yield" in stub_data.columns:
            return stub_data["earnings_yield"].dropna()

        return pd.Series(
            1.0,
            index=[c for c in price_data.columns if c not in ("SPY", "QQQ", "IWM")],
            name="value",
        )

    def _compute_low_vol_factor(
        self,
        price_data: pd.DataFrame,
        vol_window: int = 60,
    ) -> pd.Series:
        """Compute low-volatility factor (negative realised volatility).

        Lower realised vol → higher score (long low-vol stocks).

        Parameters
        ----------
        price_data:
            Wide-format close price DataFrame.
        vol_window:
            Rolling window for realised volatility (default 60 days).

        Returns
        -------
        pd.Series: negative realised vol indexed by symbol.
        """
        returns = np.log(price_data / price_data.shift(1))
        vol = returns.rolling(window=vol_window, min_periods=vol_window // 2).std().iloc[-1]
        vol = vol.drop(labels=["SPY", "QQQ", "IWM"], errors="ignore")
        # Low-vol factor: negative vol (higher score = lower vol = preferred long)
        return (-vol).dropna()

    def _compute_momentum_factor(
        self,
        price_data: pd.DataFrame,
        lookback: int = 252,
        skip: int = 21,
    ) -> pd.Series:
        """Compute 12-1 month momentum factor.

        Parameters
        ----------
        price_data:
            Wide-format close price DataFrame.
        lookback:
            12-month lookback in trading days.
        skip:
            1-month skip in trading days.

        Returns
        -------
        pd.Series: total return over [t-lookback, t-skip] indexed by symbol.
        """
        if len(price_data) < lookback + 5:
            return pd.Series(dtype=float)

        momentum: Dict[str, float] = {}
        for sym in price_data.columns:
            if sym in ("SPY", "QQQ", "IWM"):
                continue
            prices = price_data[sym].dropna()
            if len(prices) < lookback + 5:
                continue
            p_start = float(prices.iloc[-lookback])
            p_end = float(prices.iloc[-skip]) if skip > 0 else float(prices.iloc[-1])
            if p_start <= 0:
                continue
            momentum[sym] = (p_end / p_start) - 1.0

        return pd.Series(momentum)

    # ------------------------------------------------------------------
    # Z-scoring and composite score
    # ------------------------------------------------------------------

    def z_score_factors(
        self,
        factor_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Cross-sectionally z-score and winsorize each factor column.

        Parameters
        ----------
        factor_data:
            DataFrame with factor values; columns are factor names, index
            is symbols.

        Returns
        -------
        pd.DataFrame of winsorized, cross-sectionally z-scored factors.
            Same shape as input; NaN values are preserved.
        """
        result = pd.DataFrame(index=factor_data.index)
        for col in factor_data.columns:
            series = factor_data[col].dropna()
            if series.empty:
                result[col] = np.nan
                continue
            winsorized = _winsorize(series, n_sigma=3.0)
            result[col] = _cross_section_zscore(winsorized)
        return result

    def composite_score(
        self,
        factors: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """Compute a weighted composite z-score from factor z-scores.

        Parameters
        ----------
        factors:
            DataFrame of z-scored factors (output of :meth:`z_score_factors`).
        weights:
            Optional mapping of column name → weight.  If *None*, uses equal
            weights from config (or falls back to 0.25 per factor).
            Weights are normalised to sum to 1.0.

        Returns
        -------
        pd.Series: composite score indexed by symbol.
        """
        if weights is None:
            # Use config weights
            weights = {}
            if "quality" in factors.columns:
                weights["quality"] = self._cfg.quality_weight
            if "value" in factors.columns:
                weights["value"] = self._cfg.value_weight
            if "low_vol" in factors.columns:
                weights["low_vol"] = self._cfg.low_vol_weight
            if "momentum" in factors.columns:
                weights["momentum"] = self._cfg.momentum_weight

        # Normalise weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            return pd.Series(0.0, index=factors.index)

        norm_weights = {k: v / total_weight for k, v in weights.items()}

        composite = pd.Series(0.0, index=factors.index)
        for col, w in norm_weights.items():
            if col not in factors.columns:
                continue
            composite = composite.add(factors[col].fillna(0.0) * w, fill_value=0.0)

        return composite

    # ------------------------------------------------------------------
    # Factor timing
    # ------------------------------------------------------------------

    def _compute_value_spread(self, factor_data: pd.DataFrame) -> float:
        """Compute the value spread (top decile E/P minus bottom decile E/P).

        A wide spread (top-decile cheap stocks much cheaper than bottom-decile)
        is a signal to overweight the value factor.

        Parameters
        ----------
        factor_data:
            Raw (un-z-scored) factor data.

        Returns
        -------
        Value spread as a scalar.  Returns 0.0 if value factor is unavailable.
        """
        if "value" not in factor_data.columns:
            return 0.0

        val = factor_data["value"].dropna().sort_values(ascending=False)
        n = len(val)
        if n < 10:
            return 0.0

        top_decile_mean = float(val.iloc[: n // 10].mean())
        bot_decile_mean = float(val.iloc[-(n // 10) :].mean())
        return top_decile_mean - bot_decile_mean

    def _adjust_weights_for_regime(
        self,
        base_weights: Dict[str, float],
        regime_state: RegimeState,
        factor_data: pd.DataFrame,
    ) -> Dict[str, float]:
        """Adjust factor weights based on regime and factor timing.

        Rules:
            - BEAR: overweight Quality (+50%) and Low-Vol (+50%).
            - Wide value spread: overweight Value (+30%).

        Parameters
        ----------
        base_weights:
            Starting weight dictionary.
        regime_state:
            Current market regime.
        factor_data:
            Raw factor data for value spread computation.

        Returns
        -------
        Adjusted (unnormalised) weight dictionary.  Will be normalised in
        :meth:`composite_score`.
        """
        weights = dict(base_weights)

        # BEAR regime: emphasise defensive factors
        if regime_state.regime == Regime.BEAR:
            if "quality" in weights:
                weights["quality"] *= 1.5
            if "low_vol" in weights:
                weights["low_vol"] *= 1.5
            if "momentum" in weights:
                weights["momentum"] *= 0.5

        # Factor timing: if value spread is unusually wide, overweight value
        val_spread = self._compute_value_spread(factor_data)
        # Wide spread threshold: use 80th percentile heuristic (0.2 for stub data)
        if val_spread > 0.2 and "value" in weights:
            weights["value"] *= 1.3
            logger.debug(
                f"FactorModelStrategy: wide value spread ({val_spread:.3f}); "
                "overweighting value factor."
            )

        return weights

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        price_data: pd.DataFrame,
        fundamental_data: Optional[pd.DataFrame],
        regime_state: RegimeState,
    ) -> List[Signal]:
        """Generate long/short signals from the composite factor score.

        Parameters
        ----------
        price_data:
            Wide-format close price DataFrame.
        fundamental_data:
            Optional real fundamental data (earnings, gross profit).  When
            *None*, the ``FundamentalDataStub`` is used.  Must have symbol
            index and columns: ``earnings_yield``, ``gross_profitability``
            (optional: additional columns are ignored).
        regime_state:
            Current market regime from the regime detector.

        Returns
        -------
        List[Signal]:
            Long signals for composite z-score > entry_z, short for < -entry_z.
            Signal strength equals the composite z-score normalised to [0, 1]
            (capped at 3σ → 1.0).

        Notes
        -----
        - CRISIS: emits zero new signals.
        - BEAR: overweights Quality and Low-Vol factors.
        - Wide value spread: overweights Value factor.
        """
        # Crisis: no new signals
        if regime_state.is_crisis:
            logger.info("FactorModelStrategy: blocked — CRISIS regime.")
            return []

        # --- Build raw factor table ---
        quality = self._compute_quality_factor(price_data, fundamental_data)
        value = self._compute_value_factor(price_data, fundamental_data)
        low_vol = self._compute_low_vol_factor(price_data)
        momentum = self._compute_momentum_factor(
            price_data,
            lookback=self._cfg.lookback_days * 4,  # ~12 months
            skip=21,
        )

        # Align all factors to a common symbol index
        all_symbols = list(
            set(quality.index) & set(value.index) & set(low_vol.index)
        )
        if not all_symbols:
            logger.warning(
                "FactorModelStrategy.generate_signals: no symbols with "
                "complete factor data."
            )
            return []

        raw_factors = pd.DataFrame(
            {
                "quality": quality.reindex(all_symbols),
                "value": value.reindex(all_symbols),
                "low_vol": low_vol.reindex(all_symbols),
                "momentum": momentum.reindex(all_symbols),
            }
        )
        raw_factors = raw_factors.dropna(subset=["quality", "value", "low_vol"])

        if raw_factors.empty:
            logger.warning("FactorModelStrategy: no valid factor data.")
            return []

        # --- Base weights ---
        base_weights = {
            "quality": self._cfg.quality_weight,
            "value": self._cfg.value_weight,
            "low_vol": self._cfg.low_vol_weight,
            "momentum": self._cfg.momentum_weight,
        }

        # --- Regime and factor timing adjustments ---
        adj_weights = self._adjust_weights_for_regime(base_weights, regime_state, raw_factors)

        # --- Z-score factors ---
        z_factors = self.z_score_factors(raw_factors)

        # --- Composite score ---
        composite = self.composite_score(z_factors, weights=adj_weights)

        # --- Generate signals ---
        entry_z = self._cfg.entry_z
        signals: List[Signal] = []

        for sym in composite.index:
            score = float(composite[sym])
            if np.isnan(score):
                continue

            # Normalise strength: z-score / 3.0 capped at 1.0
            strength = float(np.clip(abs(score) / 3.0, 0.01, 1.0))

            if score > entry_z:
                factor_scores = {
                    col: float(z_factors.loc[sym, col])
                    if sym in z_factors.index and col in z_factors.columns
                    and not pd.isna(z_factors.loc[sym, col])
                    else None
                    for col in z_factors.columns
                }
                sig = Signal(
                    symbol=sym,
                    direction="long",
                    strength=strength,
                    strategy=FactorModelStrategy.STRATEGY_NAME,
                    metadata={
                        "composite_score": score,
                        "factor_scores": factor_scores,
                        "regime": regime_state.regime.value,
                        "weights_used": adj_weights,
                    },
                )
                signals.append(sig)
                self._log.log_signal(
                    FactorModelStrategy.STRATEGY_NAME, sym, "BUY", strength,
                    {"composite_score": score},
                )

            elif score < -entry_z:
                factor_scores = {
                    col: float(z_factors.loc[sym, col])
                    if sym in z_factors.index and col in z_factors.columns
                    and not pd.isna(z_factors.loc[sym, col])
                    else None
                    for col in z_factors.columns
                }
                sig = Signal(
                    symbol=sym,
                    direction="short",
                    strength=strength,
                    strategy=FactorModelStrategy.STRATEGY_NAME,
                    metadata={
                        "composite_score": score,
                        "factor_scores": factor_scores,
                        "regime": regime_state.regime.value,
                        "weights_used": adj_weights,
                    },
                )
                signals.append(sig)
                self._log.log_signal(
                    FactorModelStrategy.STRATEGY_NAME, sym, "SELL", strength,
                    {"composite_score": score},
                )

        logger.info(
            f"FactorModelStrategy.generate_signals: {len(signals)} signals "
            f"from {len(raw_factors)} stocks "
            f"(regime={regime_state.regime.value})."
        )
        return signals

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def get_factor_exposures(
        self,
        price_data: pd.DataFrame,
        fundamental_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return raw and z-scored factor DataFrames for diagnostic purposes.

        Parameters
        ----------
        price_data:
            Wide-format close price DataFrame.
        fundamental_data:
            Optional real fundamental data.

        Returns
        -------
        (raw_factors, z_scored_factors):
            Both DataFrames indexed by symbol with columns
            [quality, value, low_vol, momentum].
        """
        quality = self._compute_quality_factor(price_data, fundamental_data)
        value = self._compute_value_factor(price_data, fundamental_data)
        low_vol = self._compute_low_vol_factor(price_data)
        momentum = self._compute_momentum_factor(price_data)

        all_syms = list(set(quality.index) & set(value.index) & set(low_vol.index))
        raw = pd.DataFrame(
            {
                "quality": quality.reindex(all_syms),
                "value": value.reindex(all_syms),
                "low_vol": low_vol.reindex(all_syms),
                "momentum": momentum.reindex(all_syms),
            }
        )
        z_scored = self.z_score_factors(raw.dropna(subset=["quality", "value", "low_vol"]))
        return raw, z_scored

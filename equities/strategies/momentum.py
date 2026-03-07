"""
equities/strategies/momentum.py
================================
Cross-Sectional Momentum strategy for the ATNN trading system.

Overview
--------
Implements residual momentum: returns are orthogonalized to the market factor
(SPY) via rolling OLS, then ranked cross-sectionally.  The 12-1 month momentum
signal skips the most recent month to avoid short-term reversal contamination.

Pipeline
--------
1. ``rank_stocks(price_data, lookback=252, skip=21)``
   - Compute 12-month minus 1-month returns for each stock.
   - Orthogonalize to SPY using rolling OLS (residual momentum).
   - Rank stocks into deciles / quintiles.
   - Returns a DataFrame of momentum scores in [−1, 1].

2. ``generate_signals(price_data, regime_state)``
   - Long top quintile / decile, short bottom quintile / decile.
   - Sector-neutral: rank within each sector, then go long/short top/bottom.
   - Volatility-scaled: position strength inversely proportional to 60-day vol.
   - REDUCED allocation (50%) in BEAR regime.
   - ZERO signals in CRISIS regime.
   - Monthly rebalancing (every ``config.rebalance_days`` bars).
   - CLOSE signals emitted for positions that fall out of top/bottom ranks.
   - Stop-loss: CLOSE if stock drops >15% from entry between rebalances.
   - Strategy-level vol scaling (Barroso & Santa-Clara 2015).

References
----------
- Jegadeesh & Titman (1993), Journal of Finance
- Blitz, Huij & Martens (2011) — Residual Momentum (Journal of Empirical Finance)
- Asness, Moskowitz & Pedersen (2013) — Value and Momentum Everywhere
- Barroso & Santa-Clara (2015) — Momentum Has Its Moments (volatility-scaled momentum)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from core.config import MomentumConfig, get_config
from core.logger import TradeLogger, get_trade_logger
from core.regime_detector import Regime, RegimeState
from equities.models import Signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector mapping (default GICS-like sector for S&P 500 names)
# ---------------------------------------------------------------------------

# Minimal built-in sector map for the default universe.  This is overridden
# when the DataManager provides a richer sector mapping.
_DEFAULT_SECTOR_MAP: Dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AMD": "Technology", "INTC": "Technology", "QCOM": "Technology",
    "MU": "Technology", "AMAT": "Technology", "LRCX": "Technology",
    "AVGO": "Technology", "CRM": "Technology", "ORCL": "Technology",
    "ADBE": "Technology", "NOW": "Technology",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "WMT": "Consumer Staples", "PG": "Consumer Staples",
    "KO": "Consumer Staples", "PEP": "Consumer Staples", "COST": "Consumer Staples",
    "GOOGL": "Communication Services", "GOOG": "Communication Services",
    "META": "Communication Services", "NFLX": "Communication Services",
    "DIS": "Communication Services", "CMCSA": "Communication Services",
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "V": "Financials", "MA": "Financials",
    "BRK.B": "Financials",
    "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "RTX": "Industrials", "CAT": "Industrials", "DE": "Industrials",
    "LIN": "Materials",
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF",
}


def _get_sector(symbol: str, sector_map: Optional[Dict[str, str]] = None) -> str:
    """Look up the GICS sector for a symbol.

    Parameters
    ----------
    symbol:
        Ticker.
    sector_map:
        Custom mapping to use in priority over the built-in default.

    Returns
    -------
    Sector string, or ``"Unknown"`` if not found.
    """
    merged = {**_DEFAULT_SECTOR_MAP, **(sector_map or {})}
    return merged.get(symbol, "Unknown")


# ---------------------------------------------------------------------------
# Rolling OLS residualisation
# ---------------------------------------------------------------------------

def _rolling_ols_residual(
    y: pd.Series,
    x: pd.Series,
    window: int,
) -> pd.Series:
    """Compute rolling OLS residuals of y on x (vectorised).

    Uses the rolling covariance / variance formulation instead of looping
    over each bar, reducing complexity from O(n²) to O(n).

    For each rolling window of length ``window``, computes:
        beta = cov(y, x) / var(x)
        alpha = mean(y) - beta * mean(x)
        residual = y - alpha - beta * x

    Parameters
    ----------
    y:
        Dependent return series (individual stock).
    x:
        Independent return series (market factor, e.g. SPY).
    window:
        Rolling regression window in periods.

    Returns
    -------
    pd.Series:
        Residual returns of the same length as y.  The first
        ``window - 1`` values are NaN.
    """
    # Align and handle NaN
    both = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(both) < window:
        return pd.Series(np.nan, index=y.index, dtype=float)

    y_a = both["y"]
    x_a = both["x"]

    # Rolling statistics
    roll_cov = y_a.rolling(window, min_periods=window).cov(x_a)
    roll_var_x = x_a.rolling(window, min_periods=window).var()
    roll_mean_y = y_a.rolling(window, min_periods=window).mean()
    roll_mean_x = x_a.rolling(window, min_periods=window).mean()

    # Avoid division by zero
    roll_var_x = roll_var_x.replace(0.0, np.nan)

    beta = roll_cov / roll_var_x
    alpha = roll_mean_y - beta * roll_mean_x
    residuals = y_a - alpha - beta * x_a

    # Reindex to the original y index
    return residuals.reindex(y.index)


# ---------------------------------------------------------------------------
# Main strategy class
# ---------------------------------------------------------------------------

class MomentumStrategy:
    """Cross-sectional residual momentum strategy.

    Ranks all stocks by their 12-1 month residual return (orthogonalized to
    SPY), then goes long the top decile and short the bottom decile in a
    sector-neutral, volatility-scaled fashion.

    Monthly rebalancing with CLOSE signals for positions that exit the
    top/bottom rank bands.  Between rebalances, stop-loss CLOSE signals are
    emitted for any long that drops >15% from its entry price.

    Parameters
    ----------
    config:
        ``MomentumConfig`` from the system configuration.
    trade_logger:
        ``TradeLogger`` for audit logging.
    sector_map:
        Optional custom symbol → sector mapping (overrides the built-in default).

    Usage
    -----
    >>> strategy = MomentumStrategy()
    >>> scores = strategy.rank_stocks(price_data)
    >>> signals = strategy.generate_signals(price_data, regime_state)
    """

    STRATEGY_NAME: str = "momentum"

    # Stop-loss threshold: close long if price drops this far from entry
    _STOP_LOSS_PCT: float = 0.15

    def __init__(
        self,
        config: Optional[MomentumConfig] = None,
        trade_logger: Optional[TradeLogger] = None,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> None:
        cfg = config or get_config().strategy.momentum
        self._cfg = cfg
        self._log = trade_logger or get_trade_logger()
        self._sector_map: Dict[str, str] = {
            **_DEFAULT_SECTOR_MAP,
            **(sector_map or {}),
        }

        # ---- Rebalance / position tracking state ----
        # Number of times generate_signals() has been called
        self._bar_count: int = 0
        # Bar index of the last rebalance (0 = never rebalanced)
        self._last_rebalance_bar: int = 0
        # Currently held positions: symbol → direction ("long" | "short")
        self._current_longs: Set[str] = set()
        self._current_shorts: Set[str] = set()
        # Entry prices for stop-loss tracking: symbol → entry close price
        self._entry_prices: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Returns and volatility computation
    # ------------------------------------------------------------------

    def _compute_returns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Compute daily log returns for all symbols.

        Parameters
        ----------
        price_data:
            Wide-format close price DataFrame.

        Returns
        -------
        pd.DataFrame of log returns (same shape, first row is NaN).
        """
        return np.log(price_data / price_data.shift(1))

    def _compute_realized_vol(
        self,
        returns: pd.DataFrame,
        window: int = 60,
    ) -> pd.DataFrame:
        """Compute annualised realised volatility for each symbol.

        Parameters
        ----------
        returns:
            Daily log return DataFrame.
        window:
            Rolling window in trading days (default 60).

        Returns
        -------
        pd.DataFrame of annualised vol (same columns as returns).
        """
        return returns.rolling(window=window, min_periods=window // 2).std() * np.sqrt(252)

    # ------------------------------------------------------------------
    # Momentum ranking
    # ------------------------------------------------------------------

    def rank_stocks(
        self,
        price_data: pd.DataFrame,
        lookback: int = 252,
        skip: int = 21,
    ) -> pd.DataFrame:
        """Rank all stocks by residual 12-1 month momentum.

        Computes each stock's total return over [t - lookback, t - skip],
        orthogonalizes to SPY using rolling OLS, then cross-sectionally ranks
        the result.

        Parameters
        ----------
        price_data:
            Wide-format DataFrame; must include ``"SPY"`` as one of the columns.
            Index should be a sorted DatetimeIndex.
        lookback:
            Lookback window in trading days (default 252 ≈ 12 months).
        skip:
            Number of most-recent trading days to skip (default 21 ≈ 1 month)
            to avoid short-term reversal.

        Returns
        -------
        pd.DataFrame with one row per symbol (at the latest date) and columns:
            - ``raw_momentum``:    12-1 month total return
            - ``residual_momentum``: market-orthogonalised momentum
            - ``rank``:            cross-sectional percentile rank in [0, 1]
            - ``score``:           re-scaled to [−1, 1] (1 = top rank)
            - ``realized_vol``:    60-day annualised realised volatility
            - ``sector``:          GICS sector string
        """
        if len(price_data) < lookback + 10:
            logger.warning(
                f"MomentumStrategy.rank_stocks: insufficient history "
                f"({len(price_data)} bars, need {lookback + 10})."
            )
            return pd.DataFrame()

        # Market factor returns
        market_col = "SPY" if "SPY" in price_data.columns else None
        returns = self._compute_returns(price_data)

        # 12-1 month raw momentum: price change from [t-lookback] to [t-skip]
        # Using total return (sum of log returns) over the window
        momentum_start = -(lookback)
        momentum_end = -(skip) if skip > 0 else None

        raw_momentum: Dict[str, float] = {}
        for sym in price_data.columns:
            if sym in ("SPY", "QQQ", "IWM"):
                continue
            prices_sym = price_data[sym].dropna()
            if len(prices_sym) < lookback + 5:
                continue
            # Total return: price at (t-skip) / price at (t-lookback) - 1
            p_start = float(prices_sym.iloc[momentum_start])
            p_end = float(prices_sym.iloc[momentum_end]) if momentum_end is not None else float(prices_sym.iloc[-1])
            if p_start <= 0:
                continue
            raw_momentum[sym] = (p_end / p_start) - 1.0

        if not raw_momentum:
            return pd.DataFrame()

        # Residual momentum: orthogonalize to market
        residual_momentum: Dict[str, float] = {}
        if market_col is not None:
            spy_ret = returns[market_col].dropna()
            ols_window = min(63, lookback // 4)  # ~3-month OLS window

            for sym, raw_mom in raw_momentum.items():
                if sym not in returns.columns:
                    residual_momentum[sym] = raw_mom
                    continue
                stock_ret = returns[sym].reindex(spy_ret.index)

                # Use rolling OLS to get residual returns, then sum over momentum window
                res_series = _rolling_ols_residual(stock_ret, spy_ret, window=ols_window)

                # Aggregate residual over the momentum window (t-lookback to t-skip)
                res_window = res_series.iloc[momentum_start:momentum_end]
                if res_window.dropna().empty:
                    residual_momentum[sym] = raw_mom
                else:
                    residual_momentum[sym] = float(res_window.dropna().sum())
        else:
            residual_momentum = dict(raw_momentum)

        # 60-day realised volatility (latest)
        symbols = list(residual_momentum.keys())
        vol_60 = self._compute_realized_vol(returns, window=60)
        latest_vol: Dict[str, float] = {}
        for sym in symbols:
            if sym in vol_60.columns:
                v = vol_60[sym].dropna()
                latest_vol[sym] = float(v.iloc[-1]) if len(v) > 0 else 0.20
            else:
                latest_vol[sym] = 0.20  # default 20% vol

        # Volatility-adjusted momentum (Barroso & Santa-Clara 2015):
        # Normalise residual return by realised vol to avoid momentum crashes.
        # Stocks with strong momentum AND low vol rank highest.
        vol_adj_momentum: Dict[str, float] = {}
        for sym in symbols:
            vol = latest_vol.get(sym, 0.20)
            vol_adj_momentum[sym] = residual_momentum[sym] / max(vol, 0.05)

        # Cross-sectional percentile rank on vol-adjusted momentum
        values = np.array([vol_adj_momentum[s] for s in symbols])

        # Percentile rank: 0 = worst, 1 = best
        ranks = pd.Series(values, index=symbols).rank(pct=True)

        # Build result DataFrame
        result = pd.DataFrame(
            {
                "raw_momentum": pd.Series(raw_momentum),
                "residual_momentum": pd.Series(residual_momentum),
                "vol_adj_momentum": pd.Series(vol_adj_momentum),
                "rank": ranks,
                "score": ranks * 2.0 - 1.0,   # re-scale to [-1, 1]
                "realized_vol": pd.Series(latest_vol),
                "sector": pd.Series(
                    {s: _get_sector(s, self._sector_map) for s in symbols}
                ),
            }
        )
        result.index.name = "symbol"
        return result.dropna(subset=["residual_momentum"])

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        price_data: pd.DataFrame,
        regime_state: RegimeState,
    ) -> List[Signal]:
        """Generate long/short momentum signals with monthly rebalancing.

        Parameters
        ----------
        price_data:
            Wide-format close price DataFrame.
        regime_state:
            Current market regime.

        Returns
        -------
        List[Signal]:
            On rebalance bars: CLOSE signals for positions leaving the
            rank band, then new LONG/SHORT entries for the updated portfolio.
            Between rebalances: CLOSE signals only for stop-loss triggers
            (long position down >15% from entry).

        Notes
        -----
        - ZERO signals emitted in CRISIS regime (all positions also closed).
        - Allocation halved (signal strength × 0.5) in BEAR regime.
        - Sector-neutral construction: select top/bottom per sector when
          the universe is large enough (≥ 10 non-benchmark stocks).  With
          fewer stocks, falls back to universe-wide absolute momentum so
          that small backtests still produce signals.
        - Strategy-level vol scaling: overall exposure is scaled by
          ``vol_target / trailing_60d_momentum_vol`` to protect against
          momentum crashes (Barroso & Santa-Clara 2015).
        """
        self._bar_count += 1
        signals: List[Signal] = []

        # -----------------------------------------------------------------
        # CRISIS: close everything and emit no new entries
        # -----------------------------------------------------------------
        if regime_state.is_crisis:
            logger.info("MomentumStrategy: blocked — CRISIS regime.")
            close_sigs = self._emit_close_all(reason="crisis_regime")
            # Reset state
            self._current_longs.clear()
            self._current_shorts.clear()
            self._entry_prices.clear()
            return close_sigs

        # Bear regime: reduced allocation
        bear_scalar = 0.5 if regime_state.regime == Regime.BEAR else 1.0

        # -----------------------------------------------------------------
        # Between rebalances: only emit stop-loss CLOSE signals
        # -----------------------------------------------------------------
        rebalance_days = max(1, self._cfg.rebalance_days)
        is_rebalance_bar = (self._bar_count % rebalance_days == 0)

        if not is_rebalance_bar:
            stop_signals = self._check_stop_losses(price_data)
            if stop_signals:
                logger.info(
                    f"MomentumStrategy: {len(stop_signals)} stop-loss CLOSE signals "
                    f"(bar {self._bar_count})."
                )
            return stop_signals

        # -----------------------------------------------------------------
        # Rebalance bar: rank → compare to holdings → emit CLOSE + new entries
        # -----------------------------------------------------------------
        self._last_rebalance_bar = self._bar_count

        rankings = self.rank_stocks(
            price_data,
            lookback=self._cfg.lookback_days,
            skip=self._cfg.skip_days,
        )
        if rankings.empty:
            logger.warning("MomentumStrategy.generate_signals: rankings are empty.")
            return []

        n_ranked = len(rankings)
        use_sector_neutral = self._cfg.sector_neutral and n_ranked >= 10

        if use_sector_neutral:
            new_longs, new_shorts = self._select_sector_neutral(rankings)
        else:
            if self._cfg.sector_neutral and n_ranked < 10:
                logger.info(
                    f"MomentumStrategy: only {n_ranked} ranked stocks — "
                    "falling back from sector-neutral to universe-wide momentum."
                )
            new_longs, new_shorts = self._select_universe_wide(rankings)

        # -----------------------------------------------------------------
        # Strategy-level vol scalar (Barroso & Santa-Clara)
        # Scales overall exposure by inverse trailing momentum strategy vol.
        # -----------------------------------------------------------------
        strategy_scalar = self._compute_strategy_scalar(price_data, rankings)

        # -----------------------------------------------------------------
        # CLOSE signals for positions leaving the rank band
        # -----------------------------------------------------------------
        exiting_longs = self._current_longs - new_longs
        exiting_shorts = self._current_shorts - new_shorts

        for sym in exiting_longs:
            signals.append(Signal(
                symbol=sym,
                direction="close",
                strength=1.0,
                strategy=MomentumStrategy.STRATEGY_NAME,
                metadata={"reason": "rank_exit_long", "bar": self._bar_count},
            ))
            self._log.log_signal(
                MomentumStrategy.STRATEGY_NAME, sym, "CLOSE", 1.0,
                {"reason": "rank_exit_long"},
            )

        for sym in exiting_shorts:
            signals.append(Signal(
                symbol=sym,
                direction="close",
                strength=1.0,
                strategy=MomentumStrategy.STRATEGY_NAME,
                metadata={"reason": "rank_exit_short", "bar": self._bar_count},
            ))
            self._log.log_signal(
                MomentumStrategy.STRATEGY_NAME, sym, "CLOSE", 1.0,
                {"reason": "rank_exit_short"},
            )

        # Remove exiting positions from tracking
        self._current_longs -= exiting_longs
        self._current_shorts -= exiting_shorts
        for sym in exiting_longs | exiting_shorts:
            self._entry_prices.pop(sym, None)

        # -----------------------------------------------------------------
        # New LONG/SHORT entry signals
        # -----------------------------------------------------------------
        entering_longs = new_longs - self._current_longs
        entering_shorts = new_shorts - self._current_shorts

        latest_prices = self._get_latest_prices(price_data)

        for sym in entering_longs:
            if sym not in rankings.index:
                continue
            row = rankings.loc[sym]
            strength = self._percentile_strength(
                float(row["rank"]), float(row["realized_vol"]),
                bear_scalar * strategy_scalar, long_side=True,
            )
            signals.append(Signal(
                symbol=sym,
                direction="long",
                strength=strength,
                strategy=MomentumStrategy.STRATEGY_NAME,
                metadata={
                    "momentum_score": float(row["score"]),
                    "residual_momentum": float(row["residual_momentum"]),
                    "rank": float(row["rank"]),
                    "realized_vol": float(row["realized_vol"]),
                    "sector": row["sector"],
                    "sector_neutral": use_sector_neutral,
                    "strategy_scalar": strategy_scalar,
                    "bar": self._bar_count,
                },
            ))
            self._log.log_signal(
                MomentumStrategy.STRATEGY_NAME, sym, "BUY", strength,
                {"rank": float(row["rank"]), "sector": row["sector"]},
            )
            # Record entry price for stop-loss tracking
            if sym in latest_prices:
                self._entry_prices[sym] = latest_prices[sym]

        for sym in entering_shorts:
            if sym not in rankings.index:
                continue
            row = rankings.loc[sym]
            strength = self._percentile_strength(
                float(row["rank"]), float(row["realized_vol"]),
                bear_scalar * strategy_scalar, long_side=False,
            )
            signals.append(Signal(
                symbol=sym,
                direction="short",
                strength=strength,
                strategy=MomentumStrategy.STRATEGY_NAME,
                metadata={
                    "momentum_score": float(row["score"]),
                    "residual_momentum": float(row["residual_momentum"]),
                    "rank": float(row["rank"]),
                    "realized_vol": float(row["realized_vol"]),
                    "sector": row["sector"],
                    "sector_neutral": use_sector_neutral,
                    "strategy_scalar": strategy_scalar,
                    "bar": self._bar_count,
                },
            ))
            self._log.log_signal(
                MomentumStrategy.STRATEGY_NAME, sym, "SELL", strength,
                {"rank": float(row["rank"]), "sector": row["sector"]},
            )
            if sym in latest_prices:
                self._entry_prices[sym] = latest_prices[sym]

        # Update position tracking
        self._current_longs = new_longs
        self._current_shorts = new_shorts

        logger.info(
            f"MomentumStrategy.generate_signals: rebalance bar {self._bar_count} — "
            f"emitted {len(signals)} signals "
            f"(regime={regime_state.regime.value}, bear_scalar={bear_scalar:.2f}, "
            f"strategy_scalar={strategy_scalar:.2f}, "
            f"n_stocks={n_ranked}, sector_neutral={use_sector_neutral}, "
            f"longs={len(new_longs)}, shorts={len(new_shorts)}, "
            f"closes_long={len(exiting_longs)}, closes_short={len(exiting_shorts)})."
        )
        return signals

    # ------------------------------------------------------------------
    # Position selection helpers
    # ------------------------------------------------------------------

    def _select_universe_wide(
        self,
        rankings: pd.DataFrame,
    ) -> tuple[Set[str], Set[str]]:
        """Select top/bottom ranked stocks across the full universe.

        Returns
        -------
        (longs, shorts) — sets of symbol strings.
        """
        n = len(rankings)
        top_n = max(1, int(n * self._cfg.long_pct))
        bot_n = max(1, int(n * self._cfg.short_pct))

        sorted_df = rankings.sort_values("residual_momentum", ascending=False)
        longs = set(str(s) for s in sorted_df.head(top_n).index)
        shorts = set(str(s) for s in sorted_df.tail(bot_n).index)
        return longs, shorts

    def _select_sector_neutral(
        self,
        rankings: pd.DataFrame,
    ) -> tuple[Set[str], Set[str]]:
        """Select top/bottom stocks sector-by-sector.

        Returns
        -------
        (longs, shorts) — sets of symbol strings.
        """
        longs: Set[str] = set()
        shorts: Set[str] = set()
        sectors = rankings["sector"].unique()

        for sector in sectors:
            if sector in ("ETF", "Unknown"):
                continue
            sector_df = rankings[rankings["sector"] == sector].copy()
            if len(sector_df) < 4:
                continue

            n = len(sector_df)
            top_pct = self._cfg.long_pct if n >= 10 else 0.2
            bot_pct = self._cfg.short_pct if n >= 10 else 0.2

            top_n = max(1, int(n * top_pct))
            bot_n = max(1, int(n * bot_pct))

            sector_df = sector_df.sort_values("residual_momentum", ascending=False)
            longs.update(str(s) for s in sector_df.head(top_n).index)
            shorts.update(str(s) for s in sector_df.tail(bot_n).index)

        return longs, shorts

    # ------------------------------------------------------------------
    # Stop-loss checking
    # ------------------------------------------------------------------

    def _check_stop_losses(self, price_data: pd.DataFrame) -> List[Signal]:
        """Emit CLOSE signals for longs that have dropped below stop-loss.

        Only longs are stop-loss checked (shorts have unlimited upside risk
        but are handled at rebalance; asymmetric treatment avoids
        prematurely covering shorts in normal drawdown noise).

        Returns
        -------
        List of CLOSE signals.
        """
        if not self._current_longs or not self._entry_prices:
            return []

        latest_prices = self._get_latest_prices(price_data)
        signals: List[Signal] = []

        triggered: Set[str] = set()
        for sym in list(self._current_longs):
            entry = self._entry_prices.get(sym)
            current = latest_prices.get(sym)
            if entry is None or current is None or entry <= 0:
                continue
            drawdown = (current - entry) / entry  # negative = loss
            if drawdown < -self._STOP_LOSS_PCT:
                signals.append(Signal(
                    symbol=sym,
                    direction="close",
                    strength=1.0,
                    strategy=MomentumStrategy.STRATEGY_NAME,
                    metadata={
                        "reason": "stop_loss",
                        "drawdown": round(drawdown, 4),
                        "entry_price": entry,
                        "current_price": current,
                        "bar": self._bar_count,
                    },
                ))
                self._log.log_signal(
                    MomentumStrategy.STRATEGY_NAME, sym, "CLOSE", 1.0,
                    {"reason": "stop_loss", "drawdown": round(drawdown, 4)},
                )
                triggered.add(sym)

        # Remove stopped-out positions from tracking
        self._current_longs -= triggered
        for sym in triggered:
            self._entry_prices.pop(sym, None)

        return signals

    def _emit_close_all(self, reason: str = "crisis_regime") -> List[Signal]:
        """Emit CLOSE signals for every tracked position.

        Returns
        -------
        List[Signal]
        """
        signals: List[Signal] = []
        for sym in self._current_longs | self._current_shorts:
            signals.append(Signal(
                symbol=sym,
                direction="close",
                strength=1.0,
                strategy=MomentumStrategy.STRATEGY_NAME,
                metadata={"reason": reason, "bar": self._bar_count},
            ))
            self._log.log_signal(
                MomentumStrategy.STRATEGY_NAME, sym, "CLOSE", 1.0,
                {"reason": reason},
            )
        return signals

    # ------------------------------------------------------------------
    # Strategy-level vol scalar (Barroso & Santa-Clara)
    # ------------------------------------------------------------------

    def _compute_strategy_scalar(
        self,
        price_data: pd.DataFrame,
        rankings: pd.DataFrame,
    ) -> float:
        """Compute strategy-level exposure scalar from trailing momentum vol.

        Scales the strategy down when momentum returns have been volatile
        (i.e., momentum crash risk is elevated).

        ``strategy_scalar = vol_target / max(trailing_60d_mom_vol, 0.05)``

        Parameters
        ----------
        price_data:
            Close price DataFrame.
        rankings:
            Current momentum rankings (used to identify long/short portfolio).

        Returns
        -------
        float in [0.25, 2.0].
        """
        vol_target = self._cfg.vol_target

        try:
            returns = self._compute_returns(price_data)

            # Build a simple equal-weight long-minus-short portfolio return
            # using the top/bottom ranked stocks to estimate momentum factor vol.
            n = len(rankings)
            if n < 4:
                return 1.0

            sorted_df = rankings.sort_values("residual_momentum", ascending=False)
            top_n = max(1, int(n * self._cfg.long_pct))
            bot_n = max(1, int(n * self._cfg.short_pct))

            long_syms = [str(s) for s in sorted_df.head(top_n).index if str(s) in returns.columns]
            short_syms = [str(s) for s in sorted_df.tail(bot_n).index if str(s) in returns.columns]

            if not long_syms or not short_syms:
                return 1.0

            long_ret = returns[long_syms].mean(axis=1)
            short_ret = returns[short_syms].mean(axis=1)
            mom_factor_ret = long_ret - short_ret

            # 60-day trailing vol of the momentum factor
            if len(mom_factor_ret.dropna()) < 30:
                return 1.0

            trailing_vol = float(
                mom_factor_ret.dropna().rolling(60, min_periods=30).std().iloc[-1]
                * np.sqrt(252)
            )
            if np.isnan(trailing_vol) or trailing_vol <= 0:
                return 1.0

            scalar = vol_target / max(trailing_vol, 0.05)
            # Cap to avoid extreme leverage or near-zero allocation
            return float(np.clip(scalar, 0.25, 2.0))

        except Exception as exc:
            logger.warning(f"MomentumStrategy._compute_strategy_scalar failed: {exc}")
            return 1.0

    # ------------------------------------------------------------------
    # Signal strength
    # ------------------------------------------------------------------

    def _percentile_strength(
        self,
        rank: float,
        realized_vol: float,
        combined_scalar: float,
        long_side: bool = True,
    ) -> float:
        """Compute differentiated signal strength based on percentile distance.

        Uses the distance from the median (0.5) to create a well-spread
        strength distribution rather than compressing everything near 1.0.

        Tier mapping (long side):
            rank ≥ 0.90  → base 0.90  (top decile)
            rank ≥ 0.80  → base 0.75  (top quintile)
            rank ≥ 0.70  → base 0.60
            rank ≥ 0.60  → base 0.45
            rank ≥ 0.50  → base 0.30  (above median)
        Short side mirrors (1 - rank):
            1-rank ≥ 0.90 → base 0.90  (bottom decile)
            etc.

        Parameters
        ----------
        rank:
            Cross-sectional percentile rank in [0, 1] (1 = best momentum).
        realized_vol:
            60-day annualised realised volatility.
        combined_scalar:
            Composite multiplier = bear_scalar * strategy_scalar.
        long_side:
            True for long legs, False for short legs.

        Returns
        -------
        Strength in [0.01, 1.0].
        """
        # For shorts, mirror the rank
        effective_rank = rank if long_side else (1.0 - rank)

        # Tiered base strength for clear differentiation
        if effective_rank >= 0.90:
            base = 0.90
        elif effective_rank >= 0.80:
            base = 0.75
        elif effective_rank >= 0.70:
            base = 0.60
        elif effective_rank >= 0.60:
            base = 0.45
        else:
            base = 0.30

        # Inverse-vol scaling: stocks with lower vol get higher weight
        vol_target = self._cfg.vol_target
        vol_scale = vol_target / max(realized_vol, 0.01)
        vol_scale = float(np.clip(vol_scale, 0.2, 3.0))

        raw = base * vol_scale * combined_scalar
        return float(np.clip(raw, 0.01, 1.0))

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _get_latest_prices(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Return the most recent close price for each symbol.

        Parameters
        ----------
        price_data:
            Wide-format close price DataFrame.

        Returns
        -------
        Dict[str, float] of symbol → latest price.
        """
        latest: Dict[str, float] = {}
        for sym in price_data.columns:
            col = price_data[sym].dropna()
            if len(col) > 0:
                latest[sym] = float(col.iloc[-1])
        return latest

    # ------------------------------------------------------------------
    # Deprecated helpers kept for backward compatibility
    # ------------------------------------------------------------------

    def _sector_neutral_signals(
        self,
        rankings: pd.DataFrame,
        regime_scalar: float,
    ) -> List[Signal]:
        """Construct signals sector-by-sector (top/bottom within each sector).

        .. deprecated::
            Internal use only.  ``generate_signals`` now manages CLOSE logic
            separately.  This method is retained for backward compatibility.

        Parameters
        ----------
        rankings:
            Output of :meth:`rank_stocks`.
        regime_scalar:
            Multiplier applied to signal strength (1.0 = normal, 0.5 = bear).

        Returns
        -------
        List[Signal]
        """
        signals: List[Signal] = []
        sectors = rankings["sector"].unique()

        for sector in sectors:
            if sector in ("ETF", "Unknown"):
                continue
            sector_df = rankings[rankings["sector"] == sector].copy()
            if len(sector_df) < 4:
                continue

            n = len(sector_df)
            top_pct = self._cfg.long_pct if n >= 10 else 0.2
            bot_pct = self._cfg.short_pct if n >= 10 else 0.2

            top_n = max(1, int(n * top_pct))
            bot_n = max(1, int(n * bot_pct))

            sector_df = sector_df.sort_values("residual_momentum", ascending=False)

            top_stocks = sector_df.head(top_n)
            bottom_stocks = sector_df.tail(bot_n)

            for sym, row in top_stocks.iterrows():
                strength = self._percentile_strength(
                    float(row["rank"]), float(row["realized_vol"]),
                    regime_scalar, long_side=True,
                )
                sig = Signal(
                    symbol=str(sym),
                    direction="long",
                    strength=strength,
                    strategy=MomentumStrategy.STRATEGY_NAME,
                    metadata={
                        "momentum_score": float(row["score"]),
                        "residual_momentum": float(row["residual_momentum"]),
                        "rank": float(row["rank"]),
                        "realized_vol": float(row["realized_vol"]),
                        "sector": row["sector"],
                        "sector_neutral": True,
                    },
                )
                signals.append(sig)
                self._log.log_signal(
                    MomentumStrategy.STRATEGY_NAME, str(sym), "BUY", strength,
                    {"rank": float(row["rank"]), "sector": row["sector"]},
                )

            for sym, row in bottom_stocks.iterrows():
                strength = self._percentile_strength(
                    float(row["rank"]), float(row["realized_vol"]),
                    regime_scalar, long_side=False,
                )
                sig = Signal(
                    symbol=str(sym),
                    direction="short",
                    strength=strength,
                    strategy=MomentumStrategy.STRATEGY_NAME,
                    metadata={
                        "momentum_score": float(row["score"]),
                        "residual_momentum": float(row["residual_momentum"]),
                        "rank": float(row["rank"]),
                        "realized_vol": float(row["realized_vol"]),
                        "sector": row["sector"],
                        "sector_neutral": True,
                    },
                )
                signals.append(sig)
                self._log.log_signal(
                    MomentumStrategy.STRATEGY_NAME, str(sym), "SELL", strength,
                    {"rank": float(row["rank"]), "sector": row["sector"]},
                )

        return signals

    def _universe_wide_signals(
        self,
        rankings: pd.DataFrame,
        regime_scalar: float,
    ) -> List[Signal]:
        """Construct long/short signals across the entire universe.

        .. deprecated::
            Internal use only.  ``generate_signals`` now manages CLOSE logic
            separately.  This method is retained for backward compatibility.

        Parameters
        ----------
        rankings:
            Output of :meth:`rank_stocks`.
        regime_scalar:
            Regime multiplier for signal strength.

        Returns
        -------
        List[Signal]
        """
        signals: List[Signal] = []
        n = len(rankings)
        top_n = max(1, int(n * self._cfg.long_pct))
        bot_n = max(1, int(n * self._cfg.short_pct))

        sorted_df = rankings.sort_values("residual_momentum", ascending=False)
        top_stocks = sorted_df.head(top_n)
        bottom_stocks = sorted_df.tail(bot_n)

        for sym, row in top_stocks.iterrows():
            strength = self._percentile_strength(
                float(row["rank"]), float(row["realized_vol"]),
                regime_scalar, long_side=True,
            )
            signals.append(Signal(
                symbol=str(sym),
                direction="long",
                strength=strength,
                strategy=MomentumStrategy.STRATEGY_NAME,
                metadata={
                    "momentum_score": float(row["score"]),
                    "rank": float(row["rank"]),
                    "realized_vol": float(row["realized_vol"]),
                    "sector": row["sector"],
                },
            ))

        for sym, row in bottom_stocks.iterrows():
            strength = self._percentile_strength(
                float(row["rank"]), float(row["realized_vol"]),
                regime_scalar, long_side=False,
            )
            signals.append(Signal(
                symbol=str(sym),
                direction="short",
                strength=strength,
                strategy=MomentumStrategy.STRATEGY_NAME,
                metadata={
                    "momentum_score": float(row["score"]),
                    "rank": float(row["rank"]),
                    "realized_vol": float(row["realized_vol"]),
                    "sector": row["sector"],
                },
            ))

        return signals

    def _vol_scaled_strength(
        self,
        rank_score: float,
        realized_vol: float,
        regime_scalar: float,
    ) -> float:
        """Compute volatility-scaled signal strength.

        .. deprecated::
            Replaced by :meth:`_percentile_strength`.  Retained for
            backward compatibility with any external callers.

        Parameters
        ----------
        rank_score:
            Percentile rank in [0, 1]; closer to 1 = stronger signal.
        realized_vol:
            60-day annualised realised volatility.
        regime_scalar:
            Multiplier from regime overlay.

        Returns
        -------
        Strength in [0.01, 1.0].
        """
        return self._percentile_strength(
            rank_score, realized_vol, regime_scalar, long_side=True
        )

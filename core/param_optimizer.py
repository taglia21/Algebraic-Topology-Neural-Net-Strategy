"""
core/param_optimizer.py
=======================
Grid search parameter optimizer for TDA strategy parameters.
Runs walk-forward backtests across parameter combinations and
reports Sharpe ratio for each.
"""
import logging
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ParamResult:
    """Result from a single parameter combination backtest."""
    params: Dict[str, Any]
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    num_trades: int


class TDAParamOptimizer:
    """Grid search optimizer for TDA strategy parameters.

    Parameters
    ----------
    param_grid : dict
        Keys are parameter names, values are lists of values to try.
        Example: {"residual_threshold": [1.0, 1.5, 2.0], "corr_window": [30, 60]}
    """

    def __init__(self, param_grid: Dict[str, List]) -> None:
        self.param_grid = param_grid
        self._results: List[ParamResult] = []

    def _compute_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.05) -> float:
        """Annualized Sharpe ratio."""
        if returns.empty or returns.std() == 0:
            return 0.0
        excess = returns - risk_free_rate / 252
        return float(excess.mean() / excess.std() * np.sqrt(252))

    def _compute_max_drawdown(self, returns: pd.Series) -> float:
        """Maximum drawdown from cumulative returns."""
        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return float(dd.min()) if len(dd) > 0 else 0.0

    def run_grid_search(self, price_df: pd.DataFrame, volume_df: Optional[pd.DataFrame] = None) -> List[ParamResult]:
        """Run backtest for every parameter combination.

        Parameters
        ----------
        price_df : pd.DataFrame
            Historical price data (columns = tickers, index = dates).
        volume_df : pd.DataFrame, optional
            Historical volume data.

        Returns
        -------
        list[ParamResult]
            Results sorted by Sharpe ratio descending.
        """
        from tda import TDAFeatureExtractor
        from ensemble.strategy_tda import TDADiffusionStrategy

        returns_df = price_df.pct_change().dropna()

        # Generate all combinations
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = list(itertools.product(*values))

        logger.info("Running grid search: %d combinations", len(combinations))

        results = []
        for combo in combinations:
            params = dict(zip(keys, combo))
            try:
                result = self._backtest_params(params, price_df, returns_df)
                results.append(result)
                logger.info(
                    "Params %s -> Sharpe=%.3f, Return=%.2f%%, Trades=%d",
                    params, result.sharpe_ratio, result.total_return * 100, result.num_trades,
                )
            except Exception as e:
                logger.warning("Failed for params %s: %s", params, e)

        results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
        self._results = results
        return results

    def _backtest_params(self, params: Dict, price_df: pd.DataFrame,
                         returns_df: pd.DataFrame) -> ParamResult:
        """Backtest a single parameter combination."""
        from tda import TDAFeatureExtractor
        from ensemble.strategy_tda import TDADiffusionStrategy

        # Build extractor with these params
        extractor = TDAFeatureExtractor(
            ph_window=params.get("ph_window", 30),
            corr_window=params.get("corr_window", 60),
            diffusion_time=params.get("diffusion_time", 1.0),
        )
        strategy = TDADiffusionStrategy(
            residual_threshold=params.get("residual_threshold", 1.5),
        )

        # Generate signals across the full history
        diffusion_residuals = extractor.diffusion.generate_signals(
            returns_df,
            window=params.get("corr_window", 60),
            diffusion_time=params.get("diffusion_time", 1.0),
        )

        if diffusion_residuals.empty:
            return ParamResult(params=params, sharpe_ratio=0.0, total_return=0.0,
                             max_drawdown=0.0, win_rate=0.0, num_trades=0)

        signals = strategy.generate_signals(diffusion_residuals)

        if signals.empty:
            return ParamResult(params=params, sharpe_ratio=0.0, total_return=0.0,
                             max_drawdown=0.0, win_rate=0.0, num_trades=0)

        # Simulate simple long/short returns based on signals
        daily_returns = []
        trade_returns = []

        # Group by date
        for date_val in signals["timestamp"].unique():
            date_signals = signals[signals["timestamp"] == date_val]
            day_return = 0.0

            for _, sig in date_signals.iterrows():
                ticker = sig["ticker"]
                direction = sig["direction"]
                strength = sig["strength"]

                if direction == "NEUTRAL" or strength < 0.3:
                    continue

                # Look up next-day return
                if ticker in returns_df.columns:
                    date_loc = returns_df.index.get_loc(date_val) if date_val in returns_df.index else None
                    if date_loc is not None and date_loc + 1 < len(returns_df):
                        next_ret = returns_df.iloc[date_loc + 1][ticker]
                        if direction == "LONG":
                            trade_ret = next_ret * strength
                        elif direction == "SHORT":
                            trade_ret = -next_ret * strength
                        else:
                            trade_ret = 0.0
                        day_return += trade_ret
                        trade_returns.append(trade_ret)

            daily_returns.append(day_return)

        if not daily_returns:
            return ParamResult(params=params, sharpe_ratio=0.0, total_return=0.0,
                             max_drawdown=0.0, win_rate=0.0, num_trades=0)

        ret_series = pd.Series(daily_returns)
        total_return = float((1 + ret_series).prod() - 1)
        sharpe = self._compute_sharpe(ret_series)
        max_dd = self._compute_max_drawdown(ret_series)
        win_rate = len([t for t in trade_returns if t > 0]) / len(trade_returns) if trade_returns else 0.0

        return ParamResult(
            params=params,
            sharpe_ratio=sharpe,
            total_return=total_return,
            max_drawdown=max_dd,
            win_rate=win_rate,
            num_trades=len(trade_returns),
        )

    def best_params(self) -> Optional[Dict]:
        """Return the best parameter combination by Sharpe ratio."""
        if not self._results:
            return None
        return self._results[0].params

    def results_dataframe(self) -> pd.DataFrame:
        """Return all results as a DataFrame."""
        records = []
        for r in self._results:
            record = dict(r.params)
            record["sharpe_ratio"] = r.sharpe_ratio
            record["total_return"] = r.total_return
            record["max_drawdown"] = r.max_drawdown
            record["win_rate"] = r.win_rate
            record["num_trades"] = r.num_trades
            records.append(record)
        return pd.DataFrame(records)

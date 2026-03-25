"""
backtest/walk_forward.py
========================
Walk-forward optimization framework matching Aalampour's methodology.

3-year rolling train window, configurable test window, purge gap,
and embargo period. Collects only out-of-sample predictions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (Aalampour methodology)
# ---------------------------------------------------------------------------

_TRAIN_WINDOW: int = 756       # ~3 years of trading days
_TEST_WINDOW: int = 21         # 1 month
_PURGE_GAP: int = 5            # 5-day gap between train and test
_EMBARGO: int = 5              # 5-day embargo after test set


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class WindowSchedule:
    """A single walk-forward window."""
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class WindowResult:
    """Result from a single walk-forward window."""
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    oos_predictions: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    model: Any = None


@dataclass
class WalkForwardResult:
    """Aggregated results from a full walk-forward run."""
    oos_predictions: pd.DataFrame
    per_window_metrics: List[dict]
    aggregate_metrics: dict
    trained_models: List[Any]
    window_schedule: List[WindowSchedule]


# ---------------------------------------------------------------------------
# WalkForwardOptimizer
# ---------------------------------------------------------------------------

class WalkForwardOptimizer:
    """Walk-forward optimization with purge + embargo.

    Parameters
    ----------
    train_window : int
        Number of trading days in train set (default 756 = ~3 years).
    test_window : int
        Number of trading days in test set (default 21 = ~1 month).
    purge_gap : int
        Days to skip between end of train and start of test (default 5).
    embargo : int
        Days to skip after end of test before next train can use (default 5).
    """

    def __init__(
        self,
        train_window: int = _TRAIN_WINDOW,
        test_window: int = _TEST_WINDOW,
        purge_gap: int = _PURGE_GAP,
        embargo: int = _EMBARGO,
    ) -> None:
        self.train_window = train_window
        self.test_window = test_window
        self.purge_gap = purge_gap
        self.embargo = embargo

    def get_window_schedule(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        trading_dates: Optional[pd.DatetimeIndex] = None,
    ) -> List[WindowSchedule]:
        """Generate the walk-forward window schedule.

        Parameters
        ----------
        start_date : pd.Timestamp
            Earliest date in the data.
        end_date : pd.Timestamp
            Latest date in the data.
        trading_dates : pd.DatetimeIndex, optional
            Actual trading dates. If None, generates business days.

        Returns
        -------
        List of WindowSchedule objects.
        """
        if trading_dates is None:
            trading_dates = pd.bdate_range(start_date, end_date)
        else:
            trading_dates = trading_dates.sort_values()

        total_dates = len(trading_dates)
        min_required = self.train_window + self.purge_gap + 1  # At least 1 test day

        if total_dates < min_required:
            logger.warning(
                "Not enough data for walk-forward: have %d, need %d",
                total_dates, min_required,
            )
            return []

        windows: List[WindowSchedule] = []
        window_id = 0
        cursor = self.train_window  # Index of first test day candidate

        while cursor + self.purge_gap < total_dates:
            train_start_idx = max(0, cursor - self.train_window)
            train_end_idx = cursor - 1

            test_start_idx = cursor + self.purge_gap
            test_end_idx = min(test_start_idx + self.test_window - 1, total_dates - 1)

            if test_start_idx >= total_dates:
                break

            windows.append(WindowSchedule(
                window_id=window_id,
                train_start=trading_dates[train_start_idx],
                train_end=trading_dates[train_end_idx],
                test_start=trading_dates[test_start_idx],
                test_end=trading_dates[test_end_idx],
            ))
            window_id += 1

            # Roll forward by test_window
            actual_test_len = test_end_idx - test_start_idx + 1
            cursor += actual_test_len + self.embargo

        logger.info("Walk-forward schedule: %d windows", len(windows))
        return windows

    def run(
        self,
        price_data: pd.DataFrame,
        features: pd.DataFrame,
        model_factory: Callable[..., Any],
        predict_fn: Callable[[Any, pd.DataFrame], pd.DataFrame],
        train_fn: Callable[[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame], Any],
        param_grid: Optional[Dict[str, List[Any]]] = None,
        metric_fn: Optional[Callable[[pd.DataFrame, pd.DataFrame], dict]] = None,
        **kwargs: Any,
    ) -> WalkForwardResult:
        """Run walk-forward optimization.

        Parameters
        ----------
        price_data : pd.DataFrame
            Price data indexed by date. Must have 'close' column (or symbol-level).
        features : pd.DataFrame
            Feature matrix indexed by date, aligned with price_data.
        model_factory : callable
            Creates a new model instance: model_factory(**params) → model.
        predict_fn : callable
            Generates predictions: predict_fn(model, features) → pd.DataFrame.
        train_fn : callable
            Trains a model: train_fn(model, features, prices, targets) → model.
        param_grid : dict, optional
            Parameter grid for optimization within each window.
            {param_name: [values]}.  If None, uses defaults.
        metric_fn : callable, optional
            Evaluates predictions: metric_fn(predictions, actuals) → dict.
        **kwargs :
            Extra args passed to model_factory.

        Returns
        -------
        WalkForwardResult
        """
        trading_dates = features.index.sort_values()
        schedule = self.get_window_schedule(
            trading_dates[0], trading_dates[-1], trading_dates,
        )

        if not schedule:
            return WalkForwardResult(
                oos_predictions=pd.DataFrame(),
                per_window_metrics=[],
                aggregate_metrics={},
                trained_models=[],
                window_schedule=[],
            )

        all_oos: List[pd.DataFrame] = []
        per_window: List[dict] = []
        models: List[Any] = []

        for win in schedule:
            logger.info(
                "Window %d: train [%s → %s], test [%s → %s]",
                win.window_id, win.train_start, win.train_end,
                win.test_start, win.test_end,
            )

            # Slice data
            train_mask = (features.index >= win.train_start) & (features.index <= win.train_end)
            test_mask = (features.index >= win.test_start) & (features.index <= win.test_end)

            train_features = features.loc[train_mask]
            test_features = features.loc[test_mask]

            train_prices = price_data.loc[price_data.index.isin(train_features.index)]
            test_prices = price_data.loc[price_data.index.isin(test_features.index)]

            if len(train_features) == 0 or len(test_features) == 0:
                logger.warning("Window %d: empty train or test set, skipping", win.window_id)
                continue

            # Compute targets from prices (next-day returns)
            if "close" in train_prices.columns:
                train_targets = train_prices["close"].pct_change().shift(-1).dropna()
            elif "Close" in train_prices.columns:
                train_targets = train_prices["Close"].pct_change().shift(-1).dropna()
            else:
                train_targets = pd.Series(dtype=float, index=train_features.index)

            # Parameter optimization
            best_model = None
            best_score = -np.inf
            best_params: dict = {}

            if param_grid:
                param_combos = _grid_search_combos(param_grid)
            else:
                param_combos = [kwargs]

            for params in param_combos:
                merged = {**kwargs, **params}
                model = model_factory(**merged)
                # Align features and targets
                common_idx = train_features.index.intersection(train_targets.index)
                if len(common_idx) == 0:
                    continue
                model = train_fn(
                    model,
                    train_features.loc[common_idx],
                    train_prices.loc[train_prices.index.isin(common_idx)],
                    train_targets.loc[common_idx],
                )
                if metric_fn is not None:
                    preds = predict_fn(model, train_features.loc[common_idx])
                    score_dict = metric_fn(preds, train_targets.loc[common_idx].to_frame())
                    score = score_dict.get("score", 0.0)
                else:
                    score = 0.0

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_params = params

            if best_model is None:
                # Fallback: train with defaults
                model = model_factory(**kwargs)
                common_idx = train_features.index.intersection(train_targets.index)
                if len(common_idx) > 0:
                    best_model = train_fn(
                        model, train_features.loc[common_idx],
                        train_prices.loc[train_prices.index.isin(common_idx)],
                        train_targets.loc[common_idx],
                    )
                else:
                    logger.warning("Window %d: no common dates, skipping", win.window_id)
                    continue

            # Generate OOS predictions
            oos_preds = predict_fn(best_model, test_features)
            if isinstance(oos_preds, pd.Series):
                oos_preds = oos_preds.to_frame(name="prediction")
            oos_preds["window_id"] = win.window_id

            all_oos.append(oos_preds)
            models.append(best_model)

            # Per-window metrics
            win_metrics = {
                "window_id": win.window_id,
                "train_start": str(win.train_start.date()),
                "train_end": str(win.train_end.date()),
                "test_start": str(win.test_start.date()),
                "test_end": str(win.test_end.date()),
                "train_samples": len(train_features),
                "test_samples": len(test_features),
                "best_params": best_params,
            }
            if metric_fn is not None and len(test_features) > 0:
                if "close" in test_prices.columns:
                    test_targets = test_prices["close"].pct_change().shift(-1).dropna()
                elif "Close" in test_prices.columns:
                    test_targets = test_prices["Close"].pct_change().shift(-1).dropna()
                else:
                    test_targets = pd.Series(dtype=float)
                if len(test_targets) > 0:
                    common_test = oos_preds.index.intersection(test_targets.index)
                    if len(common_test) > 0:
                        score_dict = metric_fn(
                            oos_preds.loc[common_test],
                            test_targets.loc[common_test].to_frame(),
                        )
                        win_metrics.update(score_dict)
            per_window.append(win_metrics)

        # Aggregate
        if all_oos:
            oos_combined = pd.concat(all_oos)
        else:
            oos_combined = pd.DataFrame()

        agg_metrics = {
            "total_windows": len(schedule),
            "completed_windows": len(per_window),
            "total_oos_samples": len(oos_combined),
        }

        # Average per-window metrics
        if per_window:
            numeric_keys = [
                k for k in per_window[0]
                if isinstance(per_window[0][k], (int, float))
                and k not in ("window_id",)
            ]
            for k in numeric_keys:
                vals = [w[k] for w in per_window if k in w and not (isinstance(w[k], float) and np.isnan(w[k]))]
                if vals:
                    agg_metrics[f"avg_{k}"] = float(np.mean(vals))

        return WalkForwardResult(
            oos_predictions=oos_combined,
            per_window_metrics=per_window,
            aggregate_metrics=agg_metrics,
            trained_models=models,
            window_schedule=schedule,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_search_combos(param_grid: Dict[str, List[Any]]) -> List[dict]:
    """Expand a parameter grid into a list of parameter combinations."""
    if not param_grid:
        return [{}]

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    combos: List[dict] = [{}]
    for key, vals in zip(keys, values):
        new_combos = []
        for combo in combos:
            for v in vals:
                new_combos.append({**combo, key: v})
        combos = new_combos

    return combos

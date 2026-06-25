"""
ml/validation.py
================
Anti-overfitting validation framework for the ATNN ML trading system.

Implements the Lopez de Prado validation suite (AFML):
  1. CPCV  — Combinatorial Purged Cross-Validation
  2. DSR   — Deflated Sharpe Ratio
  3. WFV   — Walk-Forward Validation
  4. FSC   — Feature Stability Check
  5. Master validate_model orchestrator

References
----------
- Lopez de Prado, M. "Advances in Financial Machine Learning", 2018.
  Chapter 12 (Cross-Validation), Chapter 14 (Backtesting).

Usage
-----
    from ml.validation import validate_model, cpcv_validate

    report = validate_model(model_factory, features, labels)
    print(report["recommendation"])  # DEPLOY / REVIEW / REJECT
"""

from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
# A "model factory" is a zero-argument callable that returns a fresh
# (untrained) model implementing .train(X, y, ...) and .predict(X) -> ndarray.
ModelFactory = Callable[[], Any]


# ===========================================================================
# Utility helpers
# ===========================================================================

def _sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio from a 1-D return series.

    Parameters
    ----------
    returns:
        Array of period returns (e.g., daily P&L).
    periods_per_year:
        Scaling factor for annualisation (default 252 for daily).

    Returns
    -------
    float
        Sharpe ratio; NaN if standard deviation is zero.
    """
    if len(returns) < 2:
        return float("nan")
    mu  = float(np.mean(returns))
    sig = float(np.std(returns, ddof=1))
    if sig == 0:
        return float("nan")
    return mu / sig * math.sqrt(periods_per_year)


def _predictions_to_returns(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert model predictions to long/short returns.

    A prediction > threshold → long (+label_return),
    otherwise → short (−label_return).

    For regression mode, the sign of the prediction determines direction.

    Parameters
    ----------
    predictions:
        1-D array of model scores (probabilities or regression values).
    labels:
        1-D array of actual forward returns (numeric, not binary).
    threshold:
        Decision boundary for classification scores.

    Returns
    -------
    np.ndarray
        1-D strategy return series.
    """
    directions = np.where(predictions >= threshold, 1.0, -1.0)
    return directions * np.asarray(labels, dtype=float)


def _purge_indices(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    purge_window: int,
) -> np.ndarray:
    """Remove training indices within purge_window of the test set.

    Parameters
    ----------
    train_idx:
        Array of integer positional indices for the training set.
    test_idx:
        Array of integer positional indices for the test set.
    purge_window:
        Number of bars to purge on each side of the train/test boundary.

    Returns
    -------
    np.ndarray
        Purged training indices.
    """
    test_min = int(np.min(test_idx))
    test_max = int(np.max(test_idx))
    purge_mask = (
        (train_idx >= test_min - purge_window) &
        (train_idx <= test_max + purge_window)
    )
    return train_idx[~purge_mask]


# ===========================================================================
# 1. CPCV — Combinatorial Purged Cross-Validation
# ===========================================================================

def cpcv_validate(
    model_factory: ModelFactory,
    features: pd.DataFrame,
    labels: pd.Series,
    forward_returns: Optional[pd.Series] = None,
    n_groups: int = 10,
    purge_window: int = 26,
    n_paths: Optional[int] = None,
    threshold: float = 0.5,
    verbose: bool = False,
) -> Dict:
    """Combinatorial Purged Cross-Validation (Lopez de Prado).

    Splits the data into *n_groups* chronological groups.  For every
    combination of 2 test groups (C(n_groups, 2) paths by default),
    trains on the remaining groups (minus purge buffer) and evaluates
    on the test groups.

    Parameters
    ----------
    model_factory:
        Zero-argument callable returning an untrained model.
    features:
        Feature DataFrame (time-indexed, no NaN rows).
    labels:
        Binary or continuous label Series aligned with *features*.
    n_groups:
        Number of chronological data splits.
    purge_window:
        Bars to remove around each train/test boundary (prevents leakage).
    n_paths:
        Maximum number of test combinations to evaluate. If None,
        evaluate all C(n_groups, 2) combinations.
    threshold:
        Prediction threshold for direction mapping.
    verbose:
        Log progress per path if True.

    Returns
    -------
    dict
        Keys:
        - ``sharpe_per_path`` (List[float])
        - ``pct_positive_paths`` (float)   — fraction of paths with SR > 0
        - ``mean_sharpe`` (float)
        - ``std_sharpe`` (float)
        - ``n_paths_evaluated`` (int)
        - ``pass`` (bool)              — True if pct_positive > 0.80

    Raises
    ------
    ValueError
        If fewer than 3 groups or insufficient data for even one path.
    """
    if n_groups < 3:
        raise ValueError(f"cpcv_validate: n_groups must be >= 3; got {n_groups}")

    # Drop rows where labels are NaN (features are imputed inside model.train)
    valid_mask = labels.notna()
    if forward_returns is not None:
        valid_mask &= forward_returns.notna()
    feats = features.loc[valid_mask].copy()
    labs  = labels.loc[valid_mask].copy()
    fwd: Optional[pd.Series] = None
    if forward_returns is not None:
        fwd = forward_returns.loc[valid_mask].copy()

    n = len(feats)
    if n < n_groups * 10:
        raise ValueError(
            f"cpcv_validate: only {n} clean samples for {n_groups} groups. "
            "Need at least n_groups * 10 observations."
        )

    # Assign group indices (chronological)
    group_size = n // n_groups
    group_ids  = np.full(n, -1, dtype=int)
    for g in range(n_groups):
        start = g * group_size
        end   = (g + 1) * group_size if g < n_groups - 1 else n
        group_ids[start:end] = g

    # Enumerate test combinations
    all_test_combos = list(combinations(range(n_groups), 2))
    if n_paths is not None:
        all_test_combos = all_test_combos[:n_paths]

    sharpe_per_path: List[float] = []
    feats_arr = feats.values
    labs_arr  = labs.values
    fwd_arr = fwd.values if fwd is not None else None

    for path_idx, test_groups in enumerate(all_test_combos):
        test_mask  = np.isin(group_ids, test_groups)
        train_mask = ~test_mask

        # Positional indices
        test_pos_idx  = np.where(test_mask)[0]
        train_pos_idx = np.where(train_mask)[0]

        # Purge training indices near test boundaries
        train_pos_idx = _purge_indices(train_pos_idx, test_pos_idx, purge_window)

        if len(train_pos_idx) < 50 or len(test_pos_idx) < 5:
            if verbose:
                logger.debug(f"CPCV path {path_idx}: skipped (insufficient data).")
            continue

        X_train = pd.DataFrame(
            feats_arr[train_pos_idx],
            columns=feats.columns,
        )
        y_train = pd.Series(labs_arr[train_pos_idx])

        X_test  = pd.DataFrame(
            feats_arr[test_pos_idx],
            columns=feats.columns,
        )
        y_test  = pd.Series(labs_arr[test_pos_idx])

        try:
            model = model_factory()
            model.train(X_train, y_train)
            preds = model.predict(X_test)
        except Exception as exc:
            logger.warning(f"CPCV path {path_idx}: training/prediction failed: {exc}")
            continue

        realized = fwd_arr[test_pos_idx] if fwd_arr is not None else y_test.values
        strat_returns = _predictions_to_returns(preds, realized, threshold)
        sr = _sharpe(strat_returns)
        sharpe_per_path.append(sr)

        if verbose:
            logger.debug(
                f"CPCV path {path_idx} (test groups {test_groups}): Sharpe={sr:.3f}"
            )

    if not sharpe_per_path:
        return {
            "sharpe_per_path":       [],
            "pct_positive_paths":    0.0,
            "mean_sharpe":           float("nan"),
            "std_sharpe":            float("nan"),
            "n_paths_evaluated":     0,
            "pass":                  False,
        }

    valid_sharpes = [s for s in sharpe_per_path if not math.isnan(s)]
    pct_positive  = sum(1 for s in valid_sharpes if s > 0) / len(valid_sharpes) if valid_sharpes else 0.0

    return {
        "sharpe_per_path":       sharpe_per_path,
        "pct_positive_paths":    pct_positive,
        "mean_sharpe":           float(np.nanmean(sharpe_per_path)),
        "std_sharpe":            float(np.nanstd(sharpe_per_path, ddof=1)),
        "n_paths_evaluated":     len(sharpe_per_path),
        "pass":                  pct_positive >= 0.80,
    }


# ===========================================================================
# 2. DSR — Deflated Sharpe Ratio
# ===========================================================================

def deflated_sharpe_ratio(
    sharpe_observed: float,
    n_trials: int,
    t_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (Lopez de Prado AFML, Chapter 14).

    Adjusts the observed Sharpe ratio for multiple testing by computing the
    probability that the observed SR exceeds the expected maximum SR under
    the null hypothesis of *n_trials* independent strategies with zero
    expected return.

    Parameters
    ----------
    sharpe_observed:
        Annualised Sharpe ratio observed for the selected strategy.
    n_trials:
        Total number of strategies / configurations tested (including
        discarded ones).
    t_observations:
        Number of in-sample return observations used to compute the SR.
    skewness:
        Skewness of the in-sample return distribution (default 0).
    kurtosis:
        Excess kurtosis of the in-sample return distribution (default 3,
        i.e., normal).

    Returns
    -------
    float
        p-value: probability that the observed SR is NOT due to chance.
        A value < 0.05 indicates statistical significance.
        **PASS condition**: p-value > 0.95 (i.e., DSR p-value = 1 - returned).

    Notes
    -----
    The expected maximum SR under the null (SR*) is approximated via the
    expected maximum of *n_trials* i.i.d. normal draws (Jobson-Korkie):

        E[SR*] ≈ (1 - γ) * Z^{-1}(1 - 1/n) + γ * Z^{-1}(1 - 1/(n*e))

    where γ is the Euler-Mascheroni constant.

    The variance of the SR estimator accounts for non-normality:

        V[SR] = (1 + 0.5 * SR^2 - skew * SR + (kurt - 3)/4 * SR^2) / T

    Reference: Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier".
    """
    if n_trials <= 0:
        raise ValueError(f"n_trials must be > 0; got {n_trials}")
    if t_observations <= 1:
        raise ValueError(f"t_observations must be > 1; got {t_observations}")

    # Euler-Mascheroni constant
    EULER_MASCHERONI = 0.5772156649

    # Work entirely in annualised units. The expected maximum SR under the
    # null of n_trials independent strategies with zero expected return is
    # approximated by the expected maximum of n draws from N(0, V[SR_ann])
    # where V[SR_ann] = (1 + 0.5*SR^2 - skew*SR + (kurt-3)/4*SR^2) / (T-1) * 252.
    #
    # For the *null* (SR=0, normal returns), V[SR_ann|null] = 252 / (T-1).
    # SR* ≈ sqrt(V_null) * E[max of n N(0,1) draws]
    #      = sqrt(252 / (T-1)) * expected_max_norm

    # Expected maximum of n i.i.d. N(0,1) random variables (Jobson-Korkie approx)
    if n_trials == 1:
        expected_max_norm = 0.0
    else:
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        expected_max_norm = (1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2

    # SR* in annualised units
    sr_null_std = math.sqrt(252.0 / (t_observations - 1))
    sr_star = expected_max_norm * sr_null_std

    # Variance of the annualised SR estimator under non-normality
    # V[SR_ann_hat] ≈ (1 + 0.5*SR^2 - skew*SR + (kurt-3)/4*SR^2) / (T-1) * 252
    excess_kurtosis = kurtosis - 3.0
    sr_variance_ann = (
        1.0
        + 0.5 * sharpe_observed ** 2
        - skewness * sharpe_observed
        + excess_kurtosis / 4.0 * sharpe_observed ** 2
    ) / (t_observations - 1) * 252.0
    sr_std = math.sqrt(max(sr_variance_ann, 1e-12))

    # z-statistic: (SR_ann_observed - SR*) / std(SR_ann)
    z_stat = (sharpe_observed - sr_star) / sr_std

    # p-value = P[SR_ann_observed > SR* under null] = Φ(z)
    # PASS condition: p_value > 0.95 (statistically significant after multiple-testing correction)
    p_value = float(stats.norm.cdf(z_stat))

    return p_value


# ===========================================================================
# 3. Walk-Forward Validation
# ===========================================================================

def walk_forward_validate(
    model_factory: ModelFactory,
    features: pd.DataFrame,
    labels: pd.Series,
    forward_returns: Optional[pd.Series] = None,
    train_window: int = 504,
    test_window: int = 21,
    step: int = 21,
    min_windows: int = 12,
    threshold: float = 0.5,
    verbose: bool = False,
) -> Dict:
    """Walk-forward out-of-sample validation.

    Slides a fixed-length training window through the data, training a fresh
    model on each window and collecting out-of-sample predictions.

    Parameters
    ----------
    model_factory:
        Zero-argument callable returning an untrained model.
    features:
        Feature DataFrame, time-indexed.
    labels:
        Label Series aligned with *features*.
    train_window:
        Number of bars in the training window (default 504 ≈ 2 trading years).
    test_window:
        Number of bars in each OOS test window (default 21 ≈ 1 month).
    step:
        Bars to advance the window each iteration (default 21).
    min_windows:
        Minimum number of test windows required for a valid result.
    threshold:
        Decision threshold for converting predictions to returns.
    verbose:
        Log per-window Sharpe if True.

    Returns
    -------
    dict
        Keys:
        - ``oos_sharpe``         (float)   — overall OOS Sharpe ratio
        - ``oos_returns``        (ndarray) — concatenated OOS strategy returns
        - ``per_window_sharpe``  (List[float])
        - ``is_sharpe_mean``     (float)   — average IS Sharpe across windows
        - ``train_val_gap``      (float)   — IS Sharpe / OOS Sharpe
        - ``n_windows``          (int)
        - ``feature_importances``(List[pd.Series]) — per-window importances
        - ``pass_sharpe``        (bool)    — OOS Sharpe > 0.5
        - ``pass_gap``           (bool)    — train_val_gap < 2.0
        - ``pass``               (bool)    — both checks pass

    Raises
    ------
    ValueError
        If insufficient data for even *min_windows* test windows.
    """
    # Drop rows where labels are NaN (features are imputed inside model.train)
    valid_mask = labels.notna()
    if forward_returns is not None:
        valid_mask &= forward_returns.notna()
    feats = features.loc[valid_mask].copy()
    labs  = labels.loc[valid_mask].copy()
    fwd: Optional[pd.Series] = None
    if forward_returns is not None:
        fwd = forward_returns.loc[valid_mask].copy()
    n     = len(feats)

    # Compute how many windows are possible
    total_needed = train_window + test_window
    if n < total_needed:
        raise ValueError(
            f"walk_forward_validate: need at least {total_needed} clean samples; "
            f"got {n}."
        )

    n_possible = (n - train_window - test_window) // step + 1
    if n_possible < min_windows:
        raise ValueError(
            f"walk_forward_validate: only {n_possible} possible windows; "
            f"need at least {min_windows}. Reduce min_windows or provide more data."
        )

    feats_arr = feats.values
    labs_arr  = labs.values
    fwd_arr = fwd.values if fwd is not None else None

    all_oos_returns: List[np.ndarray] = []
    per_window_sharpe: List[float]    = []
    is_sharpes: List[float]           = []
    feature_importances: List[pd.Series] = []

    start = 0
    while start + train_window + test_window <= n:
        train_end  = start + train_window
        test_start = train_end
        test_end   = test_start + test_window

        X_train = pd.DataFrame(feats_arr[start:train_end],  columns=feats.columns)
        y_train = pd.Series(labs_arr[start:train_end])
        X_test  = pd.DataFrame(feats_arr[test_start:test_end], columns=feats.columns)
        y_test  = pd.Series(labs_arr[test_start:test_end])

        realized_train = (
            fwd_arr[start:train_end] if fwd_arr is not None else y_train.values
        )
        realized_test = (
            fwd_arr[test_start:test_end] if fwd_arr is not None else y_test.values
        )

        try:
            model = model_factory()
            model.train(X_train, y_train)
            oos_preds = model.predict(X_test)
            is_preds  = model.predict(X_train)
        except Exception as exc:
            logger.warning(f"WFV window starting at {start}: failed — {exc}")
            start += step
            continue

        oos_ret = _predictions_to_returns(oos_preds, realized_test, threshold)
        is_ret  = _predictions_to_returns(is_preds,  realized_train, threshold)

        oos_sr  = _sharpe(oos_ret)
        is_sr   = _sharpe(is_ret)

        all_oos_returns.append(oos_ret)
        per_window_sharpe.append(oos_sr)
        is_sharpes.append(is_sr)

        # Collect feature importances
        try:
            imp = model.get_feature_importance()
            feature_importances.append(imp)
        except Exception:
            pass

        if verbose:
            logger.debug(
                f"WFV window [{start}:{train_end}] → [{test_start}:{test_end}]: "
                f"IS Sharpe={is_sr:.3f}, OOS Sharpe={oos_sr:.3f}"
            )

        start += step

    if not all_oos_returns:
        return {
            "oos_sharpe":          float("nan"),
            "oos_returns":         np.array([]),
            "per_window_sharpe":   [],
            "is_sharpe_mean":      float("nan"),
            "train_val_gap":       float("nan"),
            "n_windows":           0,
            "feature_importances": [],
            "pass_sharpe":         False,
            "pass_gap":            False,
            "pass":                False,
        }

    combined_oos_ret = np.concatenate(all_oos_returns)
    oos_sharpe       = _sharpe(combined_oos_ret)
    is_sharpe_mean   = float(np.nanmean(is_sharpes))

    # Train / val gap: IS Sharpe / OOS Sharpe
    if oos_sharpe != 0 and not math.isnan(oos_sharpe):
        train_val_gap = is_sharpe_mean / oos_sharpe
    else:
        train_val_gap = float("inf")

    pass_sharpe = (not math.isnan(oos_sharpe)) and oos_sharpe > 0.5
    pass_gap    = (not math.isinf(train_val_gap)) and (not math.isnan(train_val_gap)) and train_val_gap < 2.0

    return {
        "oos_sharpe":          oos_sharpe,
        "oos_returns":         combined_oos_ret,
        "per_window_sharpe":   per_window_sharpe,
        "is_sharpe_mean":      is_sharpe_mean,
        "train_val_gap":       train_val_gap,
        "n_windows":           len(per_window_sharpe),
        "feature_importances": feature_importances,
        "pass_sharpe":         pass_sharpe,
        "pass_gap":            pass_gap,
        "pass":                pass_sharpe and pass_gap,
    }


# ===========================================================================
# 4. Feature Stability Check
# ===========================================================================

def feature_stability_check(
    importance_per_window: List[pd.Series],
    top_n: int = 20,
    threshold: float = 0.7,
) -> Dict:
    """Check if top features are consistent across walk-forward windows.

    Parameters
    ----------
    importance_per_window:
        List of ``pd.Series`` objects, one per walk-forward window, indexed
        by feature name and sorted descending by importance.
    top_n:
        Number of top features to track.
    threshold:
        Fraction of windows in which a feature must appear in the top-n to
        be considered "stable".

    Returns
    -------
    dict
        Keys:
        - ``stable_features``   (List[str]) — features meeting the threshold
        - ``pct_stable``        (float)     — fraction of top_n that are stable
        - ``consistency_matrix``(pd.DataFrame) — feature × window presence
        - ``pass``              (bool)      — pct_stable >= threshold

    Notes
    -----
    If fewer than 2 windows are provided, the check always returns Pass=True
    with a warning.
    """
    if len(importance_per_window) < 2:
        logger.warning(
            "feature_stability_check: fewer than 2 windows provided; "
            "skipping stability check."
        )
        return {
            "stable_features":    [],
            "pct_stable":         1.0,
            "consistency_matrix": pd.DataFrame(),
            "pass":               True,
        }

    # Collect all unique feature names across all windows
    all_features: set = set()
    for imp in importance_per_window:
        all_features.update(imp.index.tolist())
    all_features_list = sorted(all_features)

    n_windows = len(importance_per_window)
    # Binary matrix: feature × window (1 if feature in top_n of that window)
    presence = pd.DataFrame(
        False,
        index=all_features_list,
        columns=range(n_windows),
        dtype=bool,
    )

    for w_idx, imp in enumerate(importance_per_window):
        top_features = imp.nlargest(top_n).index.tolist()
        presence.loc[presence.index.isin(top_features), w_idx] = True

    # Fraction of windows each feature appears in top_n
    freq = presence.mean(axis=1)  # fraction across windows

    stable_features = freq[freq >= threshold].sort_values(ascending=False)
    pct_stable = len(stable_features) / top_n if top_n > 0 else 0.0

    return {
        "stable_features":    stable_features.index.tolist(),
        "feature_frequency":  freq.sort_values(ascending=False).to_dict(),
        "pct_stable":         pct_stable,
        "consistency_matrix": presence,
        "pass":               pct_stable >= threshold,
    }


# ===========================================================================
# 5. Master Validation Orchestrator
# ===========================================================================

def validate_model(
    model_factory: ModelFactory,
    features: pd.DataFrame,
    labels: pd.Series,
    forward_returns: Optional[pd.Series] = None,
    config: Optional[Dict] = None,
) -> Dict:
    """Run ALL validation checks and return a comprehensive report.

    Orchestrates:
      1. Walk-Forward Validation (primary OOS check)
      2. CPCV (combinatorial OOS check)
      3. DSR (multiple-testing correction)
      4. Feature Stability Check

    Parameters
    ----------
    model_factory:
        Zero-argument callable returning an untrained model.
    features:
        Feature DataFrame, time-indexed, NaN rows will be dropped.
    labels:
        Label Series aligned with *features*.
    config:
        Optional override dict with keys:
          ``train_window``, ``test_window``, ``step``, ``min_windows``,
          ``cpcv_n_groups``, ``cpcv_purge_window``,
          ``n_trials``, ``top_n_features``, ``feature_threshold``.

    Returns
    -------
    dict
        Comprehensive validation report including:
        - ``walk_forward``       — WFV sub-report
        - ``cpcv``               — CPCV sub-report
        - ``dsr``                — DSR sub-report (p-value, pass)
        - ``feature_stability``  — FSC sub-report
        - ``overall_pass``       (bool) — True if all checks pass
        - ``recommendation``     (str)  — "DEPLOY" / "REVIEW" / "REJECT"
        - ``summary``            (dict) — key metrics at-a-glance
    """
    cfg = {
        "train_window":        504,
        "test_window":         21,
        "step":                21,
        "min_windows":         12,
        "cpcv_n_groups":       10,
        "cpcv_purge_window":   26,
        "n_trials":            100,
        "top_n_features":      20,
        "feature_threshold":   0.7,
        "threshold":           0.5,
        "verbose":             False,
    }
    if config:
        cfg.update(config)

    logger.info("validate_model: starting Walk-Forward Validation ...")
    wfv_result: Dict = {}
    wfv_error: Optional[str] = None
    try:
        wfv_result = walk_forward_validate(
            model_factory  = model_factory,
            features       = features,
            labels         = labels,
            forward_returns= forward_returns,
            train_window   = cfg["train_window"],
            test_window    = cfg["test_window"],
            step           = cfg["step"],
            min_windows    = cfg["min_windows"],
            threshold      = cfg["threshold"],
            verbose        = cfg["verbose"],
        )
    except Exception as exc:
        wfv_error = str(exc)
        logger.warning(f"validate_model: WFV failed — {exc}")
        wfv_result = {
            "oos_sharpe": float("nan"), "pass": False,
            "feature_importances": [],
        }

    # --- CPCV ---
    logger.info("validate_model: starting CPCV ...")
    cpcv_result: Dict = {}
    cpcv_error: Optional[str] = None
    try:
        cpcv_result = cpcv_validate(
            model_factory  = model_factory,
            features       = features,
            labels         = labels,
            forward_returns= forward_returns,
            n_groups       = cfg["cpcv_n_groups"],
            purge_window   = cfg["cpcv_purge_window"],
            threshold      = cfg["threshold"],
            verbose        = cfg["verbose"],
        )
    except Exception as exc:
        cpcv_error = str(exc)
        logger.warning(f"validate_model: CPCV failed — {exc}")
        cpcv_result = {"pct_positive_paths": float("nan"), "pass": False}

    # --- DSR ---
    logger.info("validate_model: computing DSR ...")
    dsr_result: Dict = {}
    try:
        oos_sr = wfv_result.get("oos_sharpe", float("nan"))
        t_obs  = len(labels.dropna())
        oos_ret = wfv_result.get("oos_returns", np.array([]))

        skew = float(stats.skew(oos_ret)) if len(oos_ret) > 3 else 0.0
        kurt = float(stats.kurtosis(oos_ret, fisher=False)) if len(oos_ret) > 3 else 3.0

        if not math.isnan(oos_sr) and t_obs > 1:
            p_val = deflated_sharpe_ratio(
                sharpe_observed = oos_sr,
                n_trials        = cfg["n_trials"],
                t_observations  = t_obs,
                skewness        = skew,
                kurtosis        = kurt,
            )
        else:
            p_val = float("nan")

        dsr_result = {
            "p_value":           p_val,
            "observed_sharpe":   oos_sr,
            "n_trials":          cfg["n_trials"],
            "t_observations":    t_obs,
            "return_skewness":   skew,
            "return_kurtosis":   kurt,
            # PASS: p_value > 0.95 (SR is statistically significant after deflation)
            "pass":              (not math.isnan(p_val)) and p_val > 0.95,
        }
    except Exception as exc:
        logger.warning(f"validate_model: DSR failed — {exc}")
        dsr_result = {"p_value": float("nan"), "pass": False}

    # --- Feature Stability ---
    logger.info("validate_model: checking feature stability ...")
    fsc_result: Dict = {}
    try:
        imps = wfv_result.get("feature_importances", [])
        if imps:
            fsc_result = feature_stability_check(
                importance_per_window = imps,
                top_n                 = cfg["top_n_features"],
                threshold             = cfg["feature_threshold"],
            )
        else:
            fsc_result = {"pass": True, "pct_stable": float("nan"), "stable_features": []}
    except Exception as exc:
        logger.warning(f"validate_model: FSC failed — {exc}")
        fsc_result = {"pass": False, "pct_stable": float("nan"), "stable_features": []}

    # --- Overall pass / recommendation ---
    checks = {
        "walk_forward":      wfv_result.get("pass", False),
        "cpcv":              cpcv_result.get("pass", False),
        "dsr":               dsr_result.get("pass", False),
        "feature_stability": fsc_result.get("pass", False),
    }

    n_pass = sum(checks.values())

    if n_pass == 4:
        recommendation = "DEPLOY"
    elif n_pass >= 2:
        recommendation = "REVIEW"
    else:
        recommendation = "REJECT"

    overall_pass = (n_pass == 4)

    summary = {
        "oos_sharpe":        wfv_result.get("oos_sharpe", float("nan")),
        "train_val_gap":     wfv_result.get("train_val_gap", float("nan")),
        "cpcv_pct_positive": cpcv_result.get("pct_positive_paths", float("nan")),
        "dsr_p_value":       dsr_result.get("p_value", float("nan")),
        "pct_stable_feats":  fsc_result.get("pct_stable", float("nan")),
        "checks_passed":     n_pass,
        "checks_total":      4,
    }

    report = {
        "walk_forward":      wfv_result,
        "cpcv":              cpcv_result,
        "dsr":               dsr_result,
        "feature_stability": fsc_result,
        "checks":            checks,
        "overall_pass":      overall_pass,
        "recommendation":    recommendation,
        "summary":           summary,
    }

    if wfv_error:
        report["walk_forward_error"] = wfv_error
    if cpcv_error:
        report["cpcv_error"] = cpcv_error

    logger.info(
        f"validate_model: complete. "
        f"Recommendation={recommendation}, checks_passed={n_pass}/4. "
        f"OOS Sharpe={summary['oos_sharpe']:.3f if not math.isnan(summary['oos_sharpe']) else 'N/A'}."
    )

    return report

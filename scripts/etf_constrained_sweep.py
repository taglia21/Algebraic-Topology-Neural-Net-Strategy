"""Constrained ETF parameter sweep with OOS + stress evidence.

This script searches a small, hypothesis-driven neighborhood around the
current 3-sleeve equal-weight baseline and writes a reproducible artifact.
"""

from __future__ import annotations

import json
from copy import deepcopy
from itertools import product
from typing import Dict, Tuple

import numpy as np

from etf.config import ETFConfig, get_default_config
from etf.data import load_price_history
from etf.portfolio import run_combined_backtest
from etf.sleeve_analysis import default_sleeves
from etf.validation import cpcv_oos_sharpes


def run_metrics(cfg: ETFConfig, prices) -> Dict[str, float]:
    sleeves = default_sleeves(cfg)
    res = run_combined_backtest(prices, cfg, sleeves)
    m = res.metrics
    combined_ret = res.returns.loc[res.gross_exposure[res.gross_exposure > 0].index]
    cpcv = cpcv_oos_sharpes(combined_ret, n_groups=8, k_test=2, purge=5)
    cpcv_median = float(np.median(cpcv)) if len(cpcv) else float("nan")
    return {
        "sharpe": float(m.sharpe),
        "sortino": float(m.sortino),
        "cagr": float(m.cagr),
        "vol": float(m.annual_volatility),
        "maxdd": float(m.max_drawdown),
        "calmar": float(m.calmar),
        "pf": float(m.profit_factor),
        "cpcv_median": cpcv_median,
    }


def build_cfg(base: ETFConfig, *, rsi: float, maxp: int, deploy: float, dlook: int) -> ETFConfig:
    cfg = deepcopy(base)
    cfg.portfolio.method = "equal"
    cfg.mean_reversion.rsi_oversold = float(rsi)
    cfg.mean_reversion.max_positions = int(maxp)
    cfg.mean_reversion.deploy_fraction = float(deploy)
    cfg.defensive_carry.momentum_lookback = int(dlook)
    return cfg


def subset_prices(prices, start: str, end: str):
    return prices.loc[(prices.index >= start) & (prices.index <= end)]


def main() -> int:
    base = get_default_config()
    base.portfolio.method = "equal"

    prices = load_price_history(base.all_symbols, base.backtest.start, base.backtest.end, refresh=False)
    baseline = run_metrics(base, prices)

    grid = {
        "rsi": [8.0, 10.0, 12.0],
        "maxp": [3, 4, 5],
        "deploy": [0.8, 1.0],
        "dlook": [84, 126],
    }

    rows = []
    for rsi, maxp, deploy, dlook in product(grid["rsi"], grid["maxp"], grid["deploy"], grid["dlook"]):
        cfg = build_cfg(base, rsi=rsi, maxp=maxp, deploy=deploy, dlook=dlook)
        met = run_metrics(cfg, prices)
        rows.append(
            {
                "params": {
                    "rsi_oversold": rsi,
                    "max_positions": maxp,
                    "deploy_fraction": deploy,
                    "def_momentum_lookback": dlook,
                },
                "metrics": met,
                "delta_vs_baseline": {
                    "sharpe": met["sharpe"] - baseline["sharpe"],
                    "calmar": met["calmar"] - baseline["calmar"],
                    "pf": met["pf"] - baseline["pf"],
                    "maxdd": met["maxdd"] - baseline["maxdd"],
                    "cpcv_median": met["cpcv_median"] - baseline["cpcv_median"],
                },
            }
        )

    rows.sort(key=lambda r: (r["metrics"]["sharpe"], r["metrics"]["calmar"], r["metrics"]["pf"]), reverse=True)

    top3 = rows[:3]
    splits: Dict[str, Tuple[str, str]] = {
        "gfc_2008_2010": ("2008-01-01", "2010-12-31"),
        "bull_2013_2019": ("2013-01-01", "2019-12-31"),
        "covid_2020_2021": ("2020-01-01", "2021-12-31"),
        "inflation_2022_2024": ("2022-01-01", "2024-12-31"),
    }

    stress = []
    for item in top3:
        p = item["params"]
        cfg = build_cfg(
            base,
            rsi=p["rsi_oversold"],
            maxp=p["max_positions"],
            deploy=p["deploy_fraction"],
            dlook=p["def_momentum_lookback"],
        )

        stressed = {}
        for mult, key in ((1.5, "cost_plus_50"), (2.0, "cost_plus_100")):
            c2 = deepcopy(cfg)
            c2.execution.commission_bps *= mult
            c2.execution.slippage_bps *= mult
            stressed[key] = run_metrics(c2, prices)

        regime = {}
        for name, (start, end) in splits.items():
            px = subset_prices(prices, start, end)
            regime[name] = run_metrics(cfg, px)

        stress.append({"params": p, "cost_stress": stressed, "regime": regime})

    payload = {
        "baseline_equal": baseline,
        "grid": grid,
        "num_trials": len(rows),
        "top10_by_sharpe": rows[:10],
        "top3_stress_and_regime": stress,
    }

    out = "artifacts/etf_phase3_constrained_sweep_equal.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("BASELINE", baseline)
    print("TOP3")
    for r in top3:
        print(r["params"], r["metrics"])
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

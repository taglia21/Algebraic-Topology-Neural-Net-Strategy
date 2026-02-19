#!/usr/bin/env python3
"""
Post-Deployment Validation Script
===================================
Verifies that all critical components import, instantiate, and connect
after a deployment.  Prints a PASS/FAIL report.

Usage:
    python scripts/validate_deployment.py
"""

import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Helpers ─────────────────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    """Run fn(); record PASS/FAIL."""
    try:
        msg = fn()
        _results.append((name, True, msg or "OK"))
    except Exception as e:
        _results.append((name, False, f"{e}"))


def report():
    """Print final summary table."""
    print("\n" + "=" * 64)
    print("  POST-DEPLOYMENT VALIDATION REPORT")
    print("=" * 64)
    passed = 0
    for name, ok, msg in _results:
        status = "PASS ✓" if ok else "FAIL ✗"
        passed += int(ok)
        print(f"  [{status}]  {name}")
        if not ok:
            print(f"           └─ {msg}")
    total = len(_results)
    print("-" * 64)
    print(f"  {passed}/{total} checks passed")
    if passed == total:
        print("  🟢 DEPLOYMENT HEALTHY")
    else:
        print("  🔴 DEPLOYMENT HAS ISSUES")
    print("=" * 64)
    return passed == total


# ── Checks ──────────────────────────────────────────────────────────────────

def check_signal_aggregator():
    from src.signal_aggregator import SignalAggregator
    sa = SignalAggregator(min_confidence=0.55, min_models=2)
    sa.initialize()
    return f"initialized, {len(sa.models)} models loaded"


def check_daily_performance_logger():
    from src.metrics.daily_performance import DailyPerformanceLogger
    dpl = DailyPerformanceLogger(log_dir="/tmp/validate_dpl", initial_equity=100_000)
    snap = dpl.log_daily(equity=100_500, daily_pnl=500, n_positions=3)
    return f"logged snap: equity={snap.equity}, ret={snap.daily_return_pct}%"


def check_transaction_cost_model():
    from src.risk.transaction_costs import TransactionCostModel
    tcm = TransactionCostModel()
    cost = tcm.estimate_cost(symbol="AAPL", shares=100, price=185.0, daily_volume=50_000_000)
    return f"AAPL 100sh cost={cost.total_bps:.1f}bps (${cost.total_bps * 185 * 100 / 10000:.2f})"


def check_retraining_scheduler():
    from src.ml.retraining_scheduler import RetrainingScheduler, RetrainingConfig
    rs = RetrainingScheduler(config=RetrainingConfig())
    for i in range(60):
        rs.record_prediction(predicted=1 if i % 2 == 0 else 0, actual=1 if i % 3 == 0 else 0)
    return f"rolling_accuracy={rs.rolling_accuracy:.2f}, needs_retrain={rs.needs_retrain}"


def check_nn_weights_exist():
    wpath = PROJECT_ROOT / "models" / "nn_predictor.weights.h5"
    if not wpath.exists():
        raise FileNotFoundError(f"{wpath} not found")
    size_kb = wpath.stat().st_size / 1024
    return f"exists, {size_kb:.1f} KB"


def check_nn_weights_loadable():
    from src.nn_predictor import NeuralNetPredictor
    model = NeuralNetPredictor(sequence_length=20, n_features=6)
    model.compile_model()
    wpath = str(PROJECT_ROOT / "models" / "nn_predictor.weights.h5")
    model.load_checkpoint(wpath)
    import numpy as np
    dummy = np.random.randn(1, 20, 6).astype("float32")
    pred = float(model(dummy).numpy().flatten()[0])
    return f"loaded & predicted: {pred:.4f}"


def check_alpaca_connectivity():
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID", "")
    secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY", "")
    if not key or not secret:
        raise RuntimeError("Alpaca API keys not found in .env")
    # Try to import and connect
    from src.trading.alpaca_client import AlpacaClient
    client = AlpacaClient()
    acct = client.get_account()
    return f"connected, equity=${acct.equity:,.2f}, status={acct.status}"


def check_equity_engine_import():
    # Import the main engine module (without starting it)
    import run_v28_production  # noqa: F401
    assert hasattr(run_v28_production, "EquityEngine"), "EquityEngine class not found"
    return "EquityEngine class importable"


def check_signal_filter():
    from src.signal_filters import SignalFilter
    sf = SignalFilter(rsi_period=14, vol_threshold=0.30)
    return "instantiated OK"


def check_risk_guardian():
    from risk_guardian import RiskGuardian
    rg = RiskGuardian(initial_equity=100_000)
    return f"initialized, peak=${rg._peak_equity:,.0f}"


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Running post-deployment validation...\n")

    check("SignalAggregator import + init", check_signal_aggregator)
    check("DailyPerformanceLogger", check_daily_performance_logger)
    check("TransactionCostModel", check_transaction_cost_model)
    check("RetrainingScheduler", check_retraining_scheduler)
    check("SignalFilter", check_signal_filter)
    check("RiskGuardian", check_risk_guardian)
    check("NN weights file exists", check_nn_weights_exist)
    check("NN weights loadable + predict", check_nn_weights_loadable)
    check("EquityEngine importable", check_equity_engine_import)
    check("Alpaca API connectivity", check_alpaca_connectivity)

    all_ok = report()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

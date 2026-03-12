"""
Validation script for core/ modules.
Runs in the project root so relative imports work correctly.
"""

import os
import sys
import traceback

# Make sure we run from the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)

PASS = "PASS"
FAIL = "FAIL"
results = []

def check(name, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"  {PASS}  {name}")
    except Exception as exc:
        results.append((FAIL, name))
        print(f"  {FAIL}  {name}")
        traceback.print_exc()

print("\n" + "="*60)
print("  ATNN core/ module validation")
print("="*60)

# ----------------------------------------------------------------
# 1. config.py
# ----------------------------------------------------------------
print("\n[1] core/config.py")

def test_config_imports():
    from core.config import (
        get_config, Config, IBKRConfig, DataConfig, RiskConfig,
        MLConfig, BacktestConfig, SystemConfig, StrategyConfig,
    )
check("imports", test_config_imports)

def test_config_get_config():
    from core.config import get_config
    cfg = get_config(reload=True)
    assert cfg.risk.max_position_pct == 0.20
    assert cfg.risk.max_drawdown_halt == -0.30
    assert cfg.risk.daily_loss_limit == -0.03
    assert cfg.risk.max_correlation == 0.85
    assert cfg.backtest.slippage_bps == 7.0
    assert cfg.backtest.train_window == 504
    assert cfg.backtest.test_window == 21
    assert cfg.backtest.min_windows == 12
    assert "SPY" in cfg.data.symbols
    assert "QQQ" in cfg.data.symbols
    assert "IWM" in cfg.data.symbols
    assert len(cfg.data.symbols) >= 50
    assert cfg.system.mode == "paper"
check("get_config() default values", test_config_get_config)

def test_config_env_override():
    import os
    os.environ["SYSTEM_MODE"] = "backtest"
    os.environ["RISK_MAX_POSITION_PCT"] = "0.03"
    from core.config import get_config
    cfg = get_config(reload=True)
    assert cfg.system.mode == "backtest"
    assert cfg.risk.max_position_pct == 0.03
    # Restore
    os.environ.pop("SYSTEM_MODE", None)
    os.environ.pop("RISK_MAX_POSITION_PCT", None)
    get_config(reload=True)
check("env var override", test_config_env_override)

def test_config_invalid_mode():
    import os
    os.environ["SYSTEM_MODE"] = "invalid"
    from core.config import get_config
    try:
        get_config(reload=True)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    finally:
        os.environ.pop("SYSTEM_MODE", None)
        get_config(reload=True)
check("invalid mode raises ValueError", test_config_invalid_mode)

def test_config_to_dict():
    from core.config import get_config
    cfg = get_config()
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert "risk" in d
    assert d["risk"]["max_position_pct"] == 0.20
check("to_dict() serialisation", test_config_to_dict)

# ----------------------------------------------------------------
# 2. logger.py
# ----------------------------------------------------------------
print("\n[2] core/logger.py")

def test_logger_imports():
    from core.logger import (
        TradeLogger, get_trade_logger,
        EVENT_SIGNAL, EVENT_ORDER, EVENT_FILL, EVENT_RISK_EVENT,
        EVENT_REGIME_CHANGE, EVENT_PERF_SNAPSHOT,
    )
check("imports", test_logger_imports)

def test_logger_create():
    from core.logger import TradeLogger
    log = TradeLogger(session_id="test-session", log_dir="/tmp/atnn_logs", echo_stdout=False)
    assert log.session_id == "test-session"
    log.close()
check("TradeLogger creation", test_logger_create)

def test_logger_log_signal():
    import json
    from core.logger import TradeLogger
    log = TradeLogger(session_id="test-sig", log_dir="/tmp/atnn_logs", echo_stdout=False)
    log.log_signal("momentum", "AAPL", "BUY", 0.85, {"z_score": 1.9})
    log.close()
check("log_signal()", test_logger_log_signal)

def test_logger_log_order():
    from core.logger import TradeLogger
    log = TradeLogger(session_id="test-ord", log_dir="/tmp/atnn_logs", echo_stdout=False)
    log.log_order("ord-001", "AAPL", "buy", 100, 175.0, "submitted")
    log.close()
check("log_order()", test_logger_log_order)

def test_logger_log_fill():
    from core.logger import TradeLogger
    log = TradeLogger(session_id="test-fill", log_dir="/tmp/atnn_logs", echo_stdout=False)
    log.log_fill("ord-001", 175.10, 100, 0.57)
    log.close()
check("log_fill()", test_logger_log_fill)

def test_logger_log_risk_event():
    from core.logger import TradeLogger
    log = TradeLogger(session_id="test-risk", log_dir="/tmp/atnn_logs", echo_stdout=False)
    log.log_risk_event("drawdown_reduce", {"drawdown_pct": -0.12})
    log.close()
check("log_risk_event()", test_logger_log_risk_event)

def test_logger_log_regime_change():
    from core.logger import TradeLogger
    log = TradeLogger(session_id="test-regime", log_dir="/tmp/atnn_logs", echo_stdout=False)
    log.log_regime_change("BULL", "BEAR", 0.91, {"vix": 28.5})
    log.close()
check("log_regime_change()", test_logger_log_regime_change)

def test_logger_performance_tracking():
    import math
    from core.logger import TradeLogger
    log = TradeLogger(session_id="test-perf", log_dir="/tmp/atnn_logs", echo_stdout=False)
    for v in [100_000, 101_000, 102_500, 101_800, 103_000]:
        log.update_portfolio_value(v)
    perf = log.compute_performance()
    assert perf["total_return"] > 0
    assert not math.isnan(perf["sharpe"])
    assert perf["max_drawdown"] <= 0
    log.log_perf_snapshot()
    log.close()
check("performance tracking + snapshot", test_logger_performance_tracking)

def test_logger_invalid_strength():
    from core.logger import TradeLogger
    log = TradeLogger(session_id="test-bad", log_dir="/tmp/atnn_logs", echo_stdout=False)
    try:
        log.log_signal("strat", "SPY", "BUY", 1.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    log.close()
check("log_signal() rejects strength > 1", test_logger_invalid_strength)

def test_logger_file_output():
    import json
    from datetime import date
    from pathlib import Path
    from core.logger import TradeLogger
    log = TradeLogger(session_id="test-file", log_dir="/tmp/atnn_logs", echo_stdout=False)
    log.log_signal("momentum", "SPY", "SELL", 0.6)
    log.close()

    today = date.today().isoformat()
    log_file = Path(f"/tmp/atnn_logs/trades_{today}.jsonl")
    assert log_file.exists(), f"Expected log file at {log_file}"
    lines = log_file.read_text().strip().split("\n")
    # Parse each line as JSON
    records = [json.loads(line) for line in lines if line.strip()]
    assert any(r.get("event") == "signal" for r in records)
check("JSONL file written and parseable", test_logger_file_output)

def test_get_trade_logger_singleton():
    from core.logger import get_trade_logger, _DEFAULT_LOGGER
    import core.logger as cl
    cl._DEFAULT_LOGGER = None  # reset
    log1 = get_trade_logger(session_id="sess-A", log_dir="/tmp/atnn_logs", echo_stdout=False)
    log2 = get_trade_logger(log_dir="/tmp/atnn_logs", echo_stdout=False)
    assert log1 is log2
    cl._DEFAULT_LOGGER = None  # clean up
check("get_trade_logger() singleton", test_get_trade_logger_singleton)

# ----------------------------------------------------------------
# 3. regime_detector.py
# ----------------------------------------------------------------
print("\n[3] core/regime_detector.py")

def test_regime_imports():
    from core.regime_detector import (
        RegimeDetector, RegimeState, Regime, VIXLevel,
        _FeatureBuilder, _compute_adx, _classify_vix,
    )
check("imports", test_regime_imports)

def _make_spy_data(n=300):
    import numpy as np, pandas as pd
    np.random.seed(42)
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    close = 400 * np.cumprod(1 + np.random.normal(0.0003, 0.012, n))
    high  = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low   = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    volume = np.random.randint(80_000_000, 120_000_000, n).astype(float)
    vix = np.abs(np.random.normal(18, 4, n))
    return pd.DataFrame({
        "date": dates, "close": close, "high": high,
        "low": low, "volume": volume, "vix": vix,
    }).set_index("date")

def test_regime_feature_builder():
    from core.regime_detector import _FeatureBuilder
    data = _make_spy_data()
    fb = _FeatureBuilder()
    df = fb.build(data)
    X, idx = fb.get_feature_matrix(df)
    assert X.shape[1] == 3
    assert len(X) > 200
check("_FeatureBuilder.build() + get_feature_matrix()", test_regime_feature_builder)

def test_regime_vix_classify():
    from core.regime_detector import _classify_vix, VIXLevel
    assert _classify_vix(10.0) == VIXLevel.LOW
    assert _classify_vix(20.0) == VIXLevel.NORMAL
    assert _classify_vix(30.0) == VIXLevel.ELEVATED
    assert _classify_vix(40.0) == VIXLevel.CRISIS
    import math
    assert _classify_vix(math.nan) == VIXLevel.UNKNOWN
check("_classify_vix() thresholds", test_regime_vix_classify)

def test_regime_adx():
    import pandas as pd, numpy as np
    from core.regime_detector import _compute_adx
    data = _make_spy_data(100)
    adx = _compute_adx(data["high"], data["low"], data["close"])
    assert isinstance(adx, pd.Series)
    valid = adx.dropna()
    assert len(valid) > 0
    assert valid.min() >= 0
    assert valid.max() <= 100
check("_compute_adx() valid range", test_regime_adx)

def test_regime_fit_predict():
    from core.regime_detector import RegimeDetector, Regime
    data = _make_spy_data(300)
    det = RegimeDetector()
    det.fit(data)
    assert det.is_fitted
    state = det.predict(data)
    assert state.regime in (Regime.BULL, Regime.SIDEWAYS, Regime.BEAR)
    assert 0.0 <= state.confidence <= 1.0
check("fit() + predict() returns valid RegimeState", test_regime_fit_predict)

def test_regime_predict_series():
    from core.regime_detector import RegimeDetector
    data = _make_spy_data(300)
    det = RegimeDetector()
    det.fit(data)
    series = det.predict_series(data)
    assert len(series) > 0
    assert "regime" in series.columns
    assert "confidence" in series.columns
check("predict_series() shape and columns", test_regime_predict_series)

def test_regime_insufficient_data():
    from core.regime_detector import RegimeDetector
    import pandas as pd, numpy as np
    small = _make_spy_data(30)  # < 60 days
    det = RegimeDetector()
    try:
        det.fit(small)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
check("fit() raises ValueError on < 60 bars", test_regime_insufficient_data)

def test_regime_predict_before_fit():
    from core.regime_detector import RegimeDetector
    det = RegimeDetector()
    try:
        det.predict(_make_spy_data())
        assert False, "Should raise RuntimeError"
    except RuntimeError:
        pass
check("predict() raises RuntimeError before fit", test_regime_predict_before_fit)

def test_regime_save_load():
    import tempfile, os
    from core.regime_detector import RegimeDetector
    data = _make_spy_data(300)
    det = RegimeDetector()
    det.fit(data)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        det.save(path)
        det2 = RegimeDetector.load(path)
        assert det2.is_fitted
        state2 = det2.predict(data)
        assert state2.regime == det.predict(data).regime
    finally:
        os.unlink(path)
check("save() / load() round-trip", test_regime_save_load)

def test_regime_unknown_returned_for_small_predict_set():
    from core.regime_detector import RegimeDetector, Regime
    data = _make_spy_data(300)
    det = RegimeDetector()
    det.fit(data)
    small = _make_spy_data(20)
    state = det.predict(small)
    assert state.regime == Regime.UNKNOWN
check("predict() returns UNKNOWN with < 60 valid bars after fit", test_regime_unknown_returned_for_small_predict_set)

def test_regime_crisis_flag():
    from core.regime_detector import RegimeDetector
    import numpy as np
    data = _make_spy_data(300)
    # Inject high VIX to trigger crisis
    data = data.copy()
    data["vix"] = 40.0
    det = RegimeDetector()
    det.fit(data)
    state = det.predict(data)
    assert state.is_crisis  # VIX >= 35 always sets crisis
check("is_crisis flag set when VIX >= 35", test_regime_crisis_flag)

# ----------------------------------------------------------------
# 4. risk_manager.py
# ----------------------------------------------------------------
print("\n[4] core/risk_manager.py")

def test_risk_imports():
    from core.risk_manager import (
        RiskManager, RiskAction, TradeApproval, PortfolioState,
    )
check("imports", test_risk_imports)

def _make_rm():
    from core.config import RiskConfig
    from core.logger import TradeLogger
    from core.risk_manager import RiskManager
    log = TradeLogger(session_id="test-rm", log_dir="/tmp/atnn_logs", echo_stdout=False)
    return RiskManager(RiskConfig(), log), log

def test_risk_check_position_size_ok():
    rm, log = _make_rm()
    # 15% of 100k = 15000; 70 shares @ 190 = 13300 → pass
    assert rm.check_position_size("AAPL", 70, 190.0, 100_000)
    log.close()
check("check_position_size() — within limit", test_risk_check_position_size_ok)

def test_risk_check_position_size_breach():
    rm, log = _make_rm()
    # 400 shares @ 50 = 20_000 = 20% > 15% limit
    assert not rm.check_position_size("AAPL", 400, 50.0, 100_000)
    log.close()
check("check_position_size() — breach denied", test_risk_check_position_size_breach)

def test_risk_check_position_size_errors():
    rm, log = _make_rm()
    try:
        rm.check_position_size("AAPL", 100, 50.0, -1)
        assert False
    except ValueError:
        pass
    try:
        rm.check_position_size("AAPL", -5, 50.0, 100_000)
        assert False
    except ValueError:
        pass
    log.close()
check("check_position_size() — invalid args raise ValueError", test_risk_check_position_size_errors)

def test_risk_check_drawdown_normal():
    rm, log = _make_rm()
    action = rm.check_drawdown(98_000, 100_000)
    from core.risk_manager import RiskAction
    assert action == RiskAction.NORMAL
    log.close()
check("check_drawdown() — NORMAL at -2 %", test_risk_check_drawdown_normal)

def test_risk_check_drawdown_reduce():
    rm, log = _make_rm()
    from core.risk_manager import RiskAction
    action = rm.check_drawdown(78_000, 100_000)  # -22% between -20% and -30%
    assert action == RiskAction.REDUCE
    log.close()
check("check_drawdown() — REDUCE at -22 %", test_risk_check_drawdown_reduce)

def test_risk_check_drawdown_halt():
    rm, log = _make_rm()
    from core.risk_manager import RiskAction
    action = rm.check_drawdown(68_000, 100_000)  # -32% beyond -30% halt
    assert action == RiskAction.HALT
    log.close()
check("check_drawdown() — HALT at -32 %", test_risk_check_drawdown_halt)

def test_risk_check_daily_loss_ok():
    rm, log = _make_rm()
    assert rm.check_daily_loss(-2_500, 100_000)  # -2.5 % < -3 % limit → ok
    log.close()
check("check_daily_loss() — within limit", test_risk_check_daily_loss_ok)

def test_risk_check_daily_loss_breach():
    rm, log = _make_rm()
    assert not rm.check_daily_loss(-3_500, 100_000)  # -3.5 % > -3 % limit
    log.close()
check("check_daily_loss() — breach denied", test_risk_check_daily_loss_breach)

def test_risk_correlation_check():
    import numpy as np, pandas as pd
    rm, log = _make_rm()
    np.random.seed(0)
    idx = pd.date_range("2025-01-01", periods=100)
    # Make NVDA and AAPL highly correlated
    base = np.random.normal(0, 0.01, 100)
    returns = pd.DataFrame({
        "AAPL": base + np.random.normal(0, 0.002, 100),
        "NVDA": base + np.random.normal(0, 0.002, 100),  # corr ~ 0.97
        "GS":   np.random.normal(0, 0.015, 100),          # low corr
    }, index=idx)
    # NVDA should be denied (too correlated with AAPL)
    positions = {"AAPL": 5000.0}
    assert not rm.check_correlation("NVDA", positions, returns)
    # GS should be allowed
    assert rm.check_correlation("GS", positions, returns)
    log.close()
check("check_correlation() — high/low correlation pairs", test_risk_correlation_check)

def test_risk_calculate_position_size():
    rm, log = _make_rm()
    shares = rm.calculate_position_size(
        signal_strength=0.8,
        volatility=0.20,
        portfolio_value=100_000,
        price=100.0,
    )
    assert shares > 0
    # Must not exceed 20 % of portfolio / price = 200 shares
    assert shares <= 200
    log.close()
check("calculate_position_size() — vol-inverse fallback", test_risk_calculate_position_size)

def test_risk_calculate_position_size_kelly():
    rm, log = _make_rm()
    shares = rm.calculate_position_size(
        signal_strength=0.7,
        volatility=0.25,
        portfolio_value=200_000,
        price=50.0,
        win_rate=0.55,
        avg_win_loss_ratio=1.5,
    )
    assert shares >= 0
    log.close()
check("calculate_position_size() — Kelly criterion path", test_risk_calculate_position_size_kelly)

def test_risk_approve_trade_approved():
    from core.risk_manager import PortfolioState
    rm, log = _make_rm()
    state = PortfolioState(
        equity=100_000,
        peak_equity=100_000,
        today_pnl=-500.0,
        positions={},
    )
    approval = rm.approve_trade("AAPL", "buy", 30, 100.0, state)
    assert approval.approved
    assert approval.reason == ""
    log.close()
check("approve_trade() — clean portfolio approved", test_risk_approve_trade_approved)

def test_risk_approve_trade_size_breach():
    from core.risk_manager import PortfolioState
    rm, log = _make_rm()
    state = PortfolioState(
        equity=100_000,
        peak_equity=100_000,
        today_pnl=-500.0,
        positions={},
    )
    # 500 shares @ 50 = 25 % notional > 20 % limit
    approval = rm.approve_trade("AAPL", "buy", 500, 50.0, state)
    assert not approval.approved
    assert "position_size" in approval.checks_failed
    log.close()
check("approve_trade() — size breach denied", test_risk_approve_trade_size_breach)

def test_risk_approve_trade_daily_loss_halt():
    from core.risk_manager import PortfolioState
    rm, log = _make_rm()
    state = PortfolioState(
        equity=100_000,
        peak_equity=100_000,
        today_pnl=-3_500.0,  # -3.5 % > -3 % limit
        positions={},
    )
    approval = rm.approve_trade("AAPL", "buy", 10, 100.0, state)
    assert not approval.approved
    assert "daily_loss" in approval.checks_failed
    log.close()
check("approve_trade() — daily loss halt", test_risk_approve_trade_daily_loss_halt)

def test_risk_approve_trade_drawdown_halt():
    from core.risk_manager import PortfolioState
    rm, log = _make_rm()
    state = PortfolioState(
        equity=68_000,      # -32 % drawdown exceeds -30% halt
        peak_equity=100_000,
        today_pnl=0.0,
        positions={},
    )
    approval = rm.approve_trade("AAPL", "buy", 10, 100.0, state)
    assert not approval.approved
    assert "drawdown_halt" in approval.checks_failed
    log.close()
check("approve_trade() — drawdown HALT gate", test_risk_approve_trade_drawdown_halt)

def test_risk_approve_trade_sector_exposure():
    from core.risk_manager import PortfolioState
    rm, log = _make_rm()
    # Tech sector already 32 %, adding 5 % would exceed 35 %
    state = PortfolioState(
        equity=100_000,
        peak_equity=100_000,
        today_pnl=0.0,
        positions={"AAPL": 32_000.0},
        sector_map={"AAPL": "Technology", "NVDA": "Technology"},
    )
    # Adding 5000 notional of NVDA (Tech) → 37 % > 35 %
    approval = rm.approve_trade("NVDA", "buy", 50, 100.0, state)
    assert not approval.approved
    assert "sector_exposure" in approval.checks_failed
    log.close()
check("approve_trade() — sector exposure breach", test_risk_approve_trade_sector_exposure)

def test_risk_invalid_side():
    from core.risk_manager import PortfolioState, RiskManager
    from core.config import RiskConfig
    from core.logger import TradeLogger
    log = TradeLogger(session_id="ts", log_dir="/tmp/atnn_logs", echo_stdout=False)
    rm = RiskManager(RiskConfig(), log)
    state = PortfolioState(100_000, 100_000, 0.0)
    try:
        rm.approve_trade("AAPL", "short", 10, 100.0, state)
        assert False
    except ValueError:
        pass
    log.close()
check("approve_trade() — invalid side raises ValueError", test_risk_invalid_side)

# ----------------------------------------------------------------
# 5. Cross-module integration
# ----------------------------------------------------------------
print("\n[5] Cross-module integration")

def test_core_package_import():
    import core
    assert hasattr(core, "get_config")
    assert hasattr(core, "TradeLogger")
    assert hasattr(core, "RegimeDetector")
    assert hasattr(core, "RiskManager")
check("from core import * works", test_core_package_import)

def test_risk_manager_uses_logger():
    """RiskManager should log risk events through the TradeLogger."""
    import json
    from datetime import date
    from pathlib import Path
    from core.config import RiskConfig
    from core.logger import TradeLogger
    from core.risk_manager import RiskManager, PortfolioState

    log = TradeLogger(session_id="integration", log_dir="/tmp/atnn_logs", echo_stdout=False)
    rm = RiskManager(RiskConfig(), log)
    # Trigger a drawdown_halt event
    from core.risk_manager import RiskAction
    action = rm.check_drawdown(68_000, 100_000)  # -32% beyond -30% halt
    assert action == RiskAction.HALT
    log.close()

    today = date.today().isoformat()
    log_file = Path(f"/tmp/atnn_logs/trades_{today}.jsonl")
    records = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]
    risk_events = [r for r in records if r.get("event") == "risk_event"]
    assert any(r.get("event_type") == "drawdown_halt" for r in risk_events)
check("RiskManager logs risk_event via TradeLogger to file", test_risk_manager_uses_logger)

def test_regime_detector_with_config():
    """RegimeDetector should integrate with config settings."""
    from core.config import get_config
    from core.regime_detector import RegimeDetector, Regime
    cfg = get_config()
    assert cfg.data.min_history_bars == 60
    data = _make_spy_data(300)
    det = RegimeDetector()
    det.fit(data)
    state = det.predict(data)
    assert state.regime in list(Regime)
check("RegimeDetector integrates with Config", test_regime_detector_with_config)

def test_full_pipeline():
    """Minimal end-to-end: regime → signal → risk approval."""
    import numpy as np, pandas as pd
    from core.config import get_config
    from core.logger import TradeLogger
    from core.regime_detector import RegimeDetector, Regime
    from core.risk_manager import RiskManager, PortfolioState

    cfg = get_config()
    log = TradeLogger(session_id="e2e", log_dir="/tmp/atnn_logs", echo_stdout=False)
    rm = RiskManager(cfg.risk, log)

    # 1. Build market data
    data = _make_spy_data(300)

    # 2. Detect regime
    det = RegimeDetector()
    det.fit(data)
    state = det.predict(data)
    log.log_regime_change("UNKNOWN", state.regime.value, state.confidence)

    # 3. Generate (mock) signal
    log.log_signal("momentum", "AAPL", "BUY", 0.75, {"regime": state.regime.value})

    # 4. Size position
    shares = rm.calculate_position_size(0.75, 0.20, 100_000, 180.0)

    # 5. Check approval
    portfolio = PortfolioState(
        equity=100_000, peak_equity=100_000, today_pnl=-300.0
    )
    approval = rm.approve_trade("AAPL", "buy", shares, 180.0, portfolio)

    # 6. Log order
    log.log_order("e2e-001", "AAPL", "buy", shares, 180.0, "pending_approval")

    assert state.regime in list(Regime)
    assert approval.approved or len(approval.checks_failed) > 0  # either outcome is valid
    log.close()
check("Full pipeline: regime → signal → risk approval → order log", test_full_pipeline)

# ----------------------------------------------------------------
# Summary
# ----------------------------------------------------------------
print("\n" + "="*60)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
print(f"  Results: {passed} passed / {failed} failed / {len(results)} total")
if failed:
    print("\n  FAILED checks:")
    for status, name in results:
        if status == FAIL:
            print(f"    - {name}")
print("="*60 + "\n")

if failed:
    sys.exit(1)

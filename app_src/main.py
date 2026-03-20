#!/usr/bin/env python3
"""
ATNN v2 — Algebraic Topology + Neural Network Trading System
=============================================================
Entry point for all operating modes.

Usage
-----
    python main.py live                       # Live/paper trading (connects to IBKR)
    python main.py backtest                   # Walk-forward backtest
    python main.py backtest --options         # Options-specific backtest
    python main.py train                      # Train NN models via walk-forward
    python main.py status                     # Show system status
    python main.py --config config/custom.yaml live   # Custom config
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from datetime import date, datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger("atnn")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atnn",
        description="ATNN v2 — Algebraic Topology + Neural Network Trading System",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to YAML config file (default: config/default.yaml)",
    )

    sub = parser.add_subparsers(dest="command", help="Operating mode")

    # live
    live_p = sub.add_parser("live", help="Run live/paper trading loop")
    live_p.add_argument("--dry-run", action="store_true",
                        help="Log signals only, do not execute trades")

    # backtest
    bt_p = sub.add_parser("backtest", help="Run walk-forward backtest")
    bt_p.add_argument("--options", action="store_true",
                       help="Run options-specific backtest")
    bt_p.add_argument("--start", default=None,
                       help="Start date (YYYY-MM-DD), default from config")
    bt_p.add_argument("--end", default=None,
                       help="End date (YYYY-MM-DD), default today")

    # train
    sub.add_parser("train", help="Train NN models via walk-forward")

    # status
    sub.add_parser("status", help="Show system status and config")

    return parser


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging to stdout."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


# ---------------------------------------------------------------------------
# Signal time helpers
# ---------------------------------------------------------------------------

def _parse_time(t: str) -> dt_time:
    """Parse 'HH:MM' to a time object."""
    h, m = t.split(":")
    return dt_time(int(h), int(m))


def _now_et():
    """Current datetime in Eastern Time."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York"))


async def _wait_until(target_time: dt_time) -> None:
    """Sleep until target_time ET (today). Returns immediately if past."""
    now = _now_et()
    target = now.replace(
        hour=target_time.hour, minute=target_time.minute, second=0, microsecond=0
    )
    delta = (target - now).total_seconds()
    if delta > 0:
        logger.info(f"Waiting {delta / 60:.1f} minutes until {target_time}...")
        await asyncio.sleep(delta)


# ---------------------------------------------------------------------------
# Broker init helper — shared by live and status
# ---------------------------------------------------------------------------

def _create_broker_components(cfg, enable_trading: bool = False):
    """Create IBKRClient and all broker modules.

    Returns
    -------
    dict with keys: client, data_feed, portfolio_mgr, equity_trader,
    option_trader, risk_monitor.  All values may be None if IBKR
    is not configured.
    """
    result = {
        "client": None,
        "data_feed": None,
        "portfolio_mgr": None,
        "equity_trader": None,
        "option_trader": None,
        "risk_monitor": None,
    }

    if not cfg.broker.is_configured():
        return result

    from broker.ibkr_client import IBKRClient, IBKRConfig
    from broker.data_feed import IBKRDataFeed
    from broker.portfolio_manager import PortfolioManager
    from broker.equity_trader import EquityTrader
    from broker.option_trader import OptionTrader
    from broker.risk_monitor import RiskMonitor, RiskConfig as RiskMonitorConfig

    ibkr_config = IBKRConfig(
        host=cfg.broker.host,
        port=cfg.broker.port,
        client_id=cfg.broker.client_id,
        account=cfg.broker.account,
        timeout=cfg.broker.timeout,
        max_reconnect_attempts=cfg.broker.max_reconnect_attempts,
    )
    client = IBKRClient(ibkr_config)

    data_feed = IBKRDataFeed(client)
    portfolio_mgr = PortfolioManager(client)
    equity_trader = EquityTrader(client, enabled=enable_trading)
    option_trader = OptionTrader(client, data_feed=data_feed, enabled=enable_trading)
    # Wire risk thresholds from YAML config (fraction → whole-number pct)
    risk_monitor_cfg = RiskMonitorConfig(
        max_daily_loss_pct=cfg.risk.daily_loss_flatten_pct * 100,   # 0.05 → 5.0
        reduce_exposure_pct=cfg.risk.daily_loss_reduce_pct * 100,   # 0.03 → 3.0
        max_drawdown_pct=cfg.risk.max_drawdown_halt_pct * 100,      # 0.15 → 15.0
        max_position_pct=cfg.risk.max_position_pct * 100,           # 0.05 → 5.0
    )
    risk_monitor = RiskMonitor(client, portfolio_mgr, config=risk_monitor_cfg)
    risk_monitor.register_traders(
        equity_trader=equity_trader,
        option_trader=option_trader,
    )

    result.update(
        client=client,
        data_feed=data_feed,
        portfolio_mgr=portfolio_mgr,
        equity_trader=equity_trader,
        option_trader=option_trader,
        risk_monitor=risk_monitor,
    )
    return result


# ---------------------------------------------------------------------------
# LIVE MODE
# ---------------------------------------------------------------------------


async def _run_live_async(cfg, dry_run: bool = False) -> None:
    """Async live trading loop — CONTINUOUS ROLLING SIGNALS.

    Runs signal generation + trade execution every cycle_interval_minutes
    from market open to close. No more single daily shot.

    Workflow per cycle (every 15 min by default):
    1. Fetch fresh market data from IBKR
    2. Compute TDA features → TDA signals
    3. Build NN features → NN predictions
    4. MetaAllocator → allocation weights
    5. SignalAggregator → ranked signals
    6. EnsembleRiskManager → sized positions (skip tickers already held)
    7. Execute trades with ATR-based dynamic brackets
    8. Monitor existing positions (emergency stops, partial profits)
    9. At EOD: reconcile, log, and check for periodic retraining
    """
    from core.logger import get_trade_logger
    from core.market_hours import MarketCalendar
    from core.kill_switch import KillSwitch, CircuitBreakerConfig
    from core.atr_brackets import calculate_brackets

    trade_log = get_trade_logger(log_level=cfg.system.log_level)
    market_cal = MarketCalendar()

    # --- Rolling signal cycle config ---
    cycle_interval = getattr(cfg.schedule, "cycle_interval_minutes", 15) * 60  # seconds
    min_signal_strength = getattr(cfg.ensemble, "min_signal_strength", 0.15)

    # --- Initialize enhancement modules ---
    nav_cache = None
    try:
        from core.nav_cache import NAVCache
        nav_cache = NAVCache()
        logger.info("NAVCache initialized")
    except Exception as e:
        logger.warning("NAVCache init failed (non-fatal): %s", e)

    data_cache = None
    try:
        from core.data_cache import MarketDataCache
        data_cache = MarketDataCache()
        logger.info("MarketDataCache initialized")
    except Exception as e:
        logger.warning("MarketDataCache init failed (non-fatal): %s", e)

    trade_journal = None
    try:
        from core.trade_journal import TradeJournal
        trade_journal = TradeJournal()
        logger.info("TradeJournal initialized")
    except Exception as e:
        logger.warning("TradeJournal init failed (non-fatal): %s", e)

    reporter = None
    try:
        from core.daily_report import DailyReporter
        reporter = DailyReporter()
        logger.info("DailyReporter initialized")
    except Exception as e:
        logger.warning("DailyReporter init failed (non-fatal): %s", e)

    # Graceful shutdown (asyncio.Event is mutable – references stay valid)
    _shutdown_event = asyncio.Event()

    def _handle_signal(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        _shutdown_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Parse schedule times
    recon_time = _parse_time(cfg.schedule.reconciliation_time)
    market_open_time = _parse_time(cfg.schedule.market_open)

    # Kill switch
    cached_nav = nav_cache.load(cfg.backtest.initial_capital) if nav_cache else cfg.backtest.initial_capital
    kill_switch = KillSwitch(
        config=CircuitBreakerConfig(
            max_drawdown_pct=-cfg.risk.max_drawdown_halt_pct,
            max_daily_loss_pct=-cfg.risk.daily_loss_flatten_pct,
        ),
        initial_equity=cached_nav,
    )

    # --- Connect to IBKR ---
    ib_connected = False
    broker = _create_broker_components(cfg, enable_trading=(not dry_run))

    if broker["client"] is not None:
        try:
            await broker["client"].connect()
            ib_connected = True
            logger.info("Connected to IBKR")
        except Exception as e:
            logger.warning(f"IBKR connection failed: {e}. Running in signal-only mode.")

    data_feed = broker["data_feed"]
    portfolio_mgr = broker["portfolio_mgr"]
    equity_trader = broker["equity_trader"]
    option_trader = broker["option_trader"]
    risk_monitor = broker["risk_monitor"]

    # Initialize peak NAV from cache
    if portfolio_mgr and nav_cache:
        cached_peak = nav_cache.load_peak_nav(fallback=cached_nav)
        if cached_peak > 0:
            portfolio_mgr.initialize_peak_nav(cached_peak)

    # --- Initialize TDA ---
    tda_extractor = None
    tda_strategy = None
    try:
        from tda import TDAFeatureExtractor
        from ensemble import TDADiffusionStrategy
        tda_extractor = TDAFeatureExtractor(
            ph_window=cfg.tda.ph_window,
            corr_window=cfg.tda.corr_window,
            diffusion_time=cfg.tda.diffusion_time,
        )
        tda_strategy = TDADiffusionStrategy()
        logger.info("TDA feature extractor + strategy initialized")
    except Exception as e:
        logger.warning(f"TDA init failed: {e}")

    # --- Load NN model ---
    nn_model = None
    nn_strategy = None
    nn_feature_engine = None
    model_dir = Path(cfg.system.model_dir)
    try:
        from nn import LSTMPredictor, AttentionLSTMPredictor, NNFeatureEngine
        from ensemble import NNDirectionalStrategy

        nn_feature_engine = NNFeatureEngine()
        nn_strategy = NNDirectionalStrategy()
        model_cls = AttentionLSTMPredictor if cfg.nn.model_type == "attention_lstm" else LSTMPredictor

        best_model = model_dir / "best_model.pt"
        model_files = sorted(model_dir.glob("*.pt")) + sorted(model_dir.glob("*.pth"))
        model_to_load = best_model if best_model.exists() else (model_files[-1] if model_files else None)

        if model_to_load:
            # Read input_size from model metadata or infer from checkpoint
            meta_path = model_dir / "model_meta.json"
            nn_input_size = 28  # default: 28 features
            if meta_path.exists():
                import json as _json
                with open(meta_path) as _mf:
                    _meta = _json.load(_mf)
                nn_input_size = _meta.get("input_size", nn_input_size)
                logger.info(f"Model metadata: input_size={nn_input_size}")
            else:
                _ckpt = torch.load(model_to_load, map_location="cpu", weights_only=True)
                if "bn.weight" in _ckpt:
                    nn_input_size = _ckpt["bn.weight"].shape[0]
                    logger.info(f"Inferred input_size={nn_input_size} from checkpoint")

            nn_model = model_cls(
                input_size=nn_input_size,
                hidden_size=cfg.nn.hidden_size,
                num_layers=cfg.nn.num_layers,
                dropout=cfg.nn.dropout,
            )
            nn_model.load_state_dict(torch.load(model_to_load, map_location="cpu", weights_only=True))
            nn_model.eval()
            logger.info(f"Loaded NN model from {model_to_load}")
        else:
            logger.info("No trained NN model found; signals will be TDA-only")
    except Exception as e:
        logger.warning(f"NN model load failed: {e}")

    # --- Initialize ensemble ---
    meta_allocator = None
    signal_aggregator = None
    ensemble_risk = None
    try:
        from ensemble import MetaAllocator, SignalAggregator, EnsembleRiskManager

        meta_allocator = MetaAllocator()
        signal_aggregator = SignalAggregator(
            agreement_bonus=1.0 + cfg.ensemble.agreement_bonus,
            disagreement_penalty=1.0 - cfg.ensemble.disagreement_penalty,
        )
        ensemble_risk = EnsembleRiskManager(
            max_position_pct=cfg.risk.max_position_pct * 100,
            kelly_multiplier=cfg.risk.kelly_fraction,
            max_risk_per_trade=getattr(cfg.risk.small_account, 'max_risk_per_trade', 500.0),
            max_equity_position=getattr(cfg.risk.small_account, 'max_equity_position', 1000.0),
        )
        logger.info("Ensemble components initialized")
    except Exception as e:
        logger.warning(f"Ensemble init failed: {e}")

    options_enabled = cfg.options.enabled and not dry_run
    equities_enabled = cfg.equities.enabled and not dry_run
    shorting_enabled = getattr(cfg.equities, 'allow_shorting', False)

    mode_str = "DRY-RUN" if dry_run else ("LIVE" if cfg.system.mode == "live" else "PAPER")
    print(f"\n{'=' * 60}")
    print(f"  ATNN v2 — {mode_str} MODE (CONTINUOUS)")
    print(f"  Symbols: {cfg.universe.symbols}")
    print(f"  Signal cycle: every {cycle_interval // 60} min")
    print(f"  Min signal strength: {min_signal_strength}")
    print(f"  Options: {'ENABLED' if options_enabled else 'DORMANT'}")
    print(f"  Equities: {'ENABLED' if equities_enabled else 'DORMANT'}")
    print(f"  Shorting: {'ENABLED (margin)' if shorting_enabled else 'DISABLED (cash account)'}")
    print(f"  IBKR: {'Connected' if ib_connected else 'Not connected'}")
    print(f"  Brackets: ATR-based dynamic")
    print(f"{'=' * 60}\n")

    _last_trading_date = None
    _days_since_retrain = 0
    _retrain_interval = 21
    _cycle_count = 0
    _daily_trades_executed = 0

    # Track which tickers we already hold to avoid duplicate entries
    _held_tickers: set = set()

    # Regime mapping for numeric regime codes
    _REGIME_MAP = {0: "NORMAL", 1: "STRESSED", 2: "CRASH"}

    while not _shutdown_event.is_set():
        # ---------------------------------------------------------------
        # Market hours gate
        # ---------------------------------------------------------------
        if not market_cal.is_market_open():
            next_open = market_cal.next_open()
            logger.info(f"Market closed. Next open: {next_open.strftime('%Y-%m-%d %H:%M ET')}")
            for _ in range(60):
                if _shutdown_event.is_set():
                    break
                await asyncio.sleep(1)
            continue

        today = date.today()
        now = _now_et()

        # ---------------------------------------------------------------
        # Daily reset
        # ---------------------------------------------------------------
        if _last_trading_date != today:
            _last_trading_date = today
            _days_since_retrain += 1
            _cycle_count = 0
            _daily_trades_executed = 0
            _held_tickers.clear()
            kill_switch.reset_daily(
                nav_cache.load(cfg.backtest.initial_capital) if nav_cache else cfg.backtest.initial_capital
            )
            logger.info(f"=== New trading day: {today} ===")

        # Kill switch check
        if not kill_switch.is_trading_allowed():
            kill_switch.check_cooldown_expired()
        if not kill_switch.is_trading_allowed():
            logger.warning(f"Trading halted: {kill_switch.block_reason}")
            await asyncio.sleep(60)
            continue

        # Wait for market open + 15 min buffer on first cycle of the day
        if _cycle_count == 0:
            await _wait_until(market_open_time)
            # Only wait the stabilization buffer if we're near market open
            _minutes_since_open = (now - now.replace(
                hour=market_open_time.hour, minute=market_open_time.minute, second=0
            )).total_seconds() / 60
            if _minutes_since_open < 20:
                logger.info("Waiting 15 min after open for data stabilization...")
                for _ in range(900):
                    if _shutdown_event.is_set():
                        break
                    await asyncio.sleep(1)
            else:
                logger.info("Mid-day start — skipping stabilization wait (%.0f min since open)", _minutes_since_open)

        # Check if we're past reconciliation time → go to EOD
        recon_dt = now.replace(hour=recon_time.hour, minute=recon_time.minute, second=0)
        if now >= recon_dt:
            # EOD reconciliation
            await _run_eod_reconciliation(
                cfg, portfolio_mgr, ib_connected, nav_cache, trade_journal,
                reporter, kill_switch, _held_tickers, _daily_trades_executed,
                tda_signals=pd.DataFrame(), nn_signals=pd.DataFrame(),
                actionable=pd.DataFrame(), sized_signals=[],
                current_regime="NORMAL", market_data=None, price_df=None,
                volume_df=None, tda_features=None, nn_model=nn_model,
                model_dir=model_dir, _days_since_retrain=_days_since_retrain,
                _retrain_interval=_retrain_interval,
            )
            _days_since_retrain = 0 if _days_since_retrain >= _retrain_interval else _days_since_retrain

            # Sleep until market close / next day
            logger.info("Trading day complete. Sleeping until next market open.")
            while not _shutdown_event.is_set() and market_cal.is_market_open():
                await asyncio.sleep(30)
            continue

        _cycle_count += 1
        logger.info(f"--- Signal cycle #{_cycle_count} ({now.strftime('%H:%M ET')}) ---")

        # Initialize cycle-level variables
        sized_signals = []
        actionable = pd.DataFrame()
        tda_signals = pd.DataFrame()
        nn_signals = pd.DataFrame()
        current_regime = "NORMAL"
        market_data = None
        price_df = None
        volume_df = None
        tda_features = None

        try:
            # ===========================================================
            # 1. Sync current positions → know what we already hold
            # ===========================================================
            if portfolio_mgr and ib_connected:
                try:
                    await portfolio_mgr.sync_positions()
                    equity_positions = list(portfolio_mgr.get_equity_positions())
                    _held_tickers = {
                        pos.contract.symbol
                        for pos in equity_positions
                        if hasattr(pos, 'contract') and hasattr(pos, 'position') and pos.position != 0
                    }
                    logger.info(f"Currently holding: {_held_tickers or 'nothing'}")
                except Exception as e:
                    logger.warning(f"Position sync failed: {e}")

            # ===========================================================
            # 2. Fetch market data
            # ===========================================================
            if data_feed and ib_connected:
                try:
                    market_data = await data_feed.get_historical_bars_multi(
                        symbols=cfg.universe.symbols,
                        duration="1 Y",
                        bar_size="1 day",
                    )
                    logger.info(f"Fetched data for {len(market_data)} symbols")
                except Exception as e:
                    logger.error(f"Data fetch failed: {e}")

            if data_cache and market_data:
                try:
                    data_cache.save_bars(market_data)
                except Exception as e:
                    logger.warning("Data cache save failed (non-fatal): %s", e)

            if not market_data:
                logger.warning("No market data — skipping cycle")
                await _interruptible_sleep(cycle_interval, _shutdown_event)
                continue

            # Build price/volume DataFrames
            closes = {}
            volumes = {}
            for sym, df in market_data.items():
                if df is not None and len(df) > 0:
                    close_col = next((c for c in df.columns if c.lower() == "close"), None)
                    vol_col = next((c for c in df.columns if c.lower() == "volume"), None)
                    if close_col:
                        closes[sym] = df[close_col]
                    if vol_col:
                        volumes[sym] = df[vol_col]

            if not closes:
                logger.warning("No valid close prices — skipping cycle")
                await _interruptible_sleep(cycle_interval, _shutdown_event)
                continue

            price_df = pd.DataFrame(closes)
            volume_df = pd.DataFrame(volumes) if volumes else None
            returns_df = price_df.pct_change().dropna()

            if data_cache:
                try:
                    data_cache.save_combined(price_df, volume_df)
                except Exception as e:
                    logger.warning("Data cache save_combined failed (non-fatal): %s", e)

            # ===========================================================
            # 3. TDA features → TDA signals
            # ===========================================================
            if tda_extractor and tda_strategy:
                try:
                    tda_features = tda_extractor.extract(returns_df)
                    if "regime" in tda_features.columns and len(tda_features) > 0:
                        last_regime = tda_features["regime"].iloc[-1]
                        if isinstance(last_regime, (int, float)):
                            current_regime = _REGIME_MAP.get(int(last_regime), "NORMAL")
                        else:
                            current_regime = str(last_regime) if str(last_regime) in _REGIME_MAP.values() else "NORMAL"

                    diffusion_residuals = tda_extractor.diffusion.generate_signals(
                        returns_df,
                        window=tda_extractor.corr_window,
                        diffusion_time=tda_extractor.diffusion_time,
                    )
                    if not diffusion_residuals.empty:
                        diffusion_residuals["regime"] = current_regime
                    tda_signals = tda_strategy.generate_signals(diffusion_residuals)
                    logger.info(f"TDA signals: {len(tda_signals)} | regime={current_regime}")
                except Exception as e:
                    logger.warning(f"TDA extraction failed: {e}")

            # ===========================================================
            # 4. NN features → NN predictions
            # ===========================================================
            if nn_model and nn_feature_engine and nn_strategy:
                try:
                    nn_features = nn_feature_engine.build_features(
                        price_df=price_df,
                        volume_df=volume_df,
                        tda_features_df=tda_features,
                    )
                    nn_signals = nn_strategy.generate_signals(
                        features=nn_features,
                        model=nn_model,
                        regime=current_regime,
                    )
                    logger.info(f"NN signals: {len(nn_signals)}")
                except Exception as e:
                    logger.warning(f"NN prediction failed: {e}")

            # ===========================================================
            # 5. Ensemble: allocate → aggregate → filter → size
            # ===========================================================
            if signal_aggregator and (not tda_signals.empty or not nn_signals.empty):
                try:
                    # Meta-allocator weights
                    if meta_allocator:
                        alloc = meta_allocator.allocate(
                            tda_signals=tda_signals,
                            nn_signals=nn_signals,
                            market_state={"regime": current_regime},
                        )
                        tda_w = alloc.tda_weight
                        nn_w = alloc.nn_weight
                        logger.info(f"Allocation: TDA={tda_w:.2f} NN={nn_w:.2f} — {alloc.reasoning}")
                    else:
                        tda_w, nn_w = 0.5, 0.5

                    # Aggregate
                    combined = signal_aggregator.aggregate(
                        tda_signals=tda_signals if not tda_signals.empty else pd.DataFrame(
                            columns=["ticker", "direction", "strength"]
                        ),
                        nn_signals=nn_signals if not nn_signals.empty else pd.DataFrame(
                            columns=["ticker", "direction", "strength"]
                        ),
                        tda_weight=tda_w,
                        nn_weight=nn_w,
                    )

                    # Use learned threshold if available, otherwise config
                    effective_threshold = min_signal_strength
                    if trade_journal:
                        learned = trade_journal.get_learned_params()
                        opt_thresh = learned.get("optimal_min_signal_strength")
                        if opt_thresh and opt_thresh > 0.10:
                            effective_threshold = opt_thresh
                            logger.info(f"Using learned signal threshold: {effective_threshold:.4f}")

                    # Filter
                    actionable = signal_aggregator.filter_signals(
                        min_strength=effective_threshold,
                    )

                    # --- POSITION-AWARE DEDUP ---
                    # Skip tickers we already hold (no doubling up)
                    if not actionable.empty and _held_tickers:
                        before = len(actionable)
                        actionable = actionable[~actionable["ticker"].isin(_held_tickers)]
                        skipped = before - len(actionable)
                        if skipped > 0:
                            logger.info(f"Skipped {skipped} signals for already-held tickers")

                    # --- TICKER BLACKLIST from online learning ---
                    if trade_journal and not actionable.empty:
                        blacklist = trade_journal.get_ticker_blacklist()
                        if blacklist:
                            before = len(actionable)
                            actionable = actionable[~actionable["ticker"].isin(blacklist)]
                            skipped = before - len(actionable)
                            if skipped > 0:
                                logger.info(f"Blacklisted {skipped} tickers: {blacklist}")

                    logger.info(f"Actionable signals: {len(actionable)}")

                    # Size positions
                    if ensemble_risk and not actionable.empty:
                        nav = nav_cache.load(cfg.backtest.initial_capital) if nav_cache else cfg.backtest.initial_capital
                        if portfolio_mgr and ib_connected:
                            try:
                                fetched_nav = await portfolio_mgr.get_nav()
                                if fetched_nav and fetched_nav > 0:
                                    nav = fetched_nav
                                    if nav_cache:
                                        try:
                                            nav_cache.save(nav, peak_nav=portfolio_mgr.peak_nav if portfolio_mgr else nav)
                                        except Exception:
                                            pass
                                else:
                                    logger.warning("NAV returned %.2f, using fallback $%.2f", fetched_nav, nav)
                            except Exception as e:
                                logger.warning("NAV fetch failed (%s), using fallback $%.2f", e, nav)

                        exposure = {"long_pct": 0.0, "short_pct": 0.0, "gross_pct": 0.0}
                        if portfolio_mgr and ib_connected:
                            try:
                                exp_data = await portfolio_mgr.get_total_exposure()
                                exposure = {
                                    "long_pct": exp_data.get("gross_exposure_pct", 0.0),
                                    "short_pct": exp_data.get("net_exposure_pct", 0.0),
                                    "gross_pct": exp_data.get("gross_exposure_pct", 0.0),
                                }
                            except Exception as e:
                                logger.warning("Exposure fetch failed (%s), using defaults", e)

                        kelly_params = {"win_rate": 0.52, "avg_win": 0.015, "avg_loss": 0.012}
                        if trade_journal:
                            try:
                                kelly_params = trade_journal.get_kelly_params()
                            except Exception as e:
                                logger.warning("Kelly params fetch failed (non-fatal): %s", e)

                        # Max concurrent positions check
                        max_concurrent = getattr(cfg.risk.small_account, 'max_concurrent_positions', 5)
                        slots_available = max_concurrent - len(_held_tickers)
                        if slots_available <= 0:
                            logger.info(f"Max concurrent positions ({max_concurrent}) reached — no new entries this cycle")
                            actionable = actionable.iloc[0:0]  # empty it

                        for _, sig in actionable.iterrows():
                            if slots_available <= 0:
                                break
                            ps = ensemble_risk.size_position(
                                signal={
                                    "ticker": sig["ticker"],
                                    "direction": sig["direction"],
                                    "strength": sig["final_strength"],
                                },
                                portfolio_value=nav,
                                current_exposure=exposure,
                                regime=current_regime,
                                win_rate=kelly_params["win_rate"],
                                avg_win=kelly_params["avg_win"],
                                avg_loss=kelly_params["avg_loss"],
                            )
                            if ps.position_value > 0:
                                sized_signals.append(ps)
                                slots_available -= 1
                                logger.info(
                                    f"  {ps.ticker} {ps.direction} "
                                    f"${ps.position_value:.2f} ({ps.position_pct:.2f}% NAV)"
                                    f"{' [CAPPED]' if ps.capped else ''}"
                                )
                except Exception as e:
                    logger.warning(f"Ensemble aggregation failed: {e}")

            # ===========================================================
            # 6. Execute trades with ATR-based brackets
            # ===========================================================
            _portfolio_state = None
            if kill_switch and sized_signals:
                try:
                    from core.risk_manager import PortfolioState
                    _ks_equity = nav
                    _ks_pnl = 0.0
                    if portfolio_mgr and ib_connected:
                        try:
                            _ks_pnl = await portfolio_mgr.get_daily_pnl()
                        except Exception:
                            pass
                    _portfolio_state = PortfolioState(
                        equity=_ks_equity,
                        peak_equity=portfolio_mgr.peak_nav if portfolio_mgr else _ks_equity,
                        today_pnl=_ks_pnl,
                    )
                except Exception as e:
                    logger.warning("Failed to build PortfolioState: %s", e)

            if sized_signals and not dry_run:
                for ps in sized_signals:
                    try:
                        # Kill switch pre-order check
                        if kill_switch and _portfolio_state:
                            if not kill_switch.pre_order_check(_portfolio_state):
                                logger.warning(
                                    "Kill switch blocked order for %s: %s",
                                    ps.ticker, kill_switch.block_reason,
                                )
                                continue

                        if equities_enabled and equity_trader:
                            last_price = price_df[ps.ticker].iloc[-1] if ps.ticker in price_df.columns else None
                            if last_price and last_price > 0:
                                qty = int(ps.position_value / last_price)

                                # Small account floor: at least 1 share if affordable
                                if qty < 1 and last_price <= nav * 0.50:
                                    qty = 1
                                    logger.info(
                                        f"Small account floor: bumping {ps.ticker} to 1 share "
                                        f"(${last_price:.2f}, {last_price/nav*100:.1f}% of NAV)"
                                    )
                                elif qty < 1:
                                    logger.info(
                                        f"Skipping {ps.ticker}: 1 share @ ${last_price:.2f} "
                                        f"exceeds 50% of NAV (${nav:.2f})"
                                    )
                                    continue

                                # Short selling gate
                                if ps.direction == "SHORT" and not shorting_enabled:
                                    logger.info(f"Skipping {ps.ticker} SHORT: shorting disabled (cash account)")
                                    continue

                                action = "BUY" if ps.direction == "LONG" else "SELL"

                                # --- ATR-BASED DYNAMIC BRACKETS ---
                                if action == "BUY":
                                    closes_series = price_df[ps.ticker].dropna()
                                    bracket = calculate_brackets(
                                        ticker=ps.ticker,
                                        entry_price=last_price,
                                        closes=closes_series,
                                        regime=current_regime,
                                        direction="LONG",
                                    )
                                    try:
                                        bracket_results = await equity_trader.place_bracket_order(
                                            symbol=ps.ticker,
                                            quantity=qty,
                                            action=action,
                                            limit_price=round(last_price * 1.005, 2),
                                            take_profit=bracket.take_profit_price,
                                            stop_loss=bracket.stop_loss_price,
                                        )
                                        _held_tickers.add(ps.ticker)
                                        _daily_trades_executed += 1
                                        logger.info(
                                            "BRACKET %s %d %s entry~$%.2f TP=$%.2f (%.1f%%) SL=$%.2f (%.1f%%) ATR=$%.2f [%s]",
                                            action, qty, ps.ticker, last_price,
                                            bracket.take_profit_price, bracket.take_profit_pct * 100,
                                            bracket.stop_loss_price, bracket.stop_loss_pct * 100,
                                            bracket.atr_value, current_regime,
                                        )
                                    except Exception as bracket_err:
                                        logger.warning("Bracket order failed (%s), falling back to market order", bracket_err)
                                        order_result = await equity_trader.place_market_order(
                                            symbol=ps.ticker, quantity=qty, action=action,
                                        )
                                        _held_tickers.add(ps.ticker)
                                        _daily_trades_executed += 1
                                        logger.info(f"MKT Order: {action} {qty} {ps.ticker} @ ~${last_price:.2f} → {order_result}")

                                elif action == "SELL" and shorting_enabled:
                                    # SHORT entry with inverted brackets
                                    closes_series = price_df[ps.ticker].dropna()
                                    bracket = calculate_brackets(
                                        ticker=ps.ticker,
                                        entry_price=last_price,
                                        closes=closes_series,
                                        regime=current_regime,
                                        direction="SHORT",
                                    )
                                    try:
                                        bracket_results = await equity_trader.place_bracket_order(
                                            symbol=ps.ticker,
                                            quantity=qty,
                                            action="SELL",
                                            limit_price=round(last_price * 0.995, 2),
                                            take_profit=bracket.take_profit_price,
                                            stop_loss=bracket.stop_loss_price,
                                        )
                                        _held_tickers.add(ps.ticker)
                                        _daily_trades_executed += 1
                                        logger.info(
                                            "SHORT BRACKET %d %s entry~$%.2f TP=$%.2f SL=$%.2f [%s]",
                                            qty, ps.ticker, last_price,
                                            bracket.take_profit_price, bracket.stop_loss_price,
                                            current_regime,
                                        )
                                    except Exception as bracket_err:
                                        logger.warning("Short bracket failed (%s), skipping", bracket_err)

                                # Record trade in journal
                                if trade_journal:
                                    try:
                                        trade_journal.record_trade(
                                            ticker=ps.ticker, action=action, quantity=qty,
                                            price=last_price, fill_price=last_price,
                                            strategy_source="TDA" if nn_model is None else "ENSEMBLE",
                                            signal_strength=ps.position_pct / 100.0,
                                            regime=current_regime,
                                        )
                                    except Exception as e:
                                        logger.warning("Trade journal record failed (non-fatal): %s", e)

                                if kill_switch:
                                    kill_switch.on_fill(0.0)

                    except Exception as e:
                        logger.error(f"Trade execution failed for {ps.ticker}: {e}")
            elif dry_run and sized_signals:
                logger.info(f"Dry-run: {len(sized_signals)} signals, no trades executed")
            elif not sized_signals:
                logger.info("No new signals this cycle")

            # ===========================================================
            # 7. Monitor existing positions (emergency stops)
            # ===========================================================
            if portfolio_mgr and ib_connected and equity_trader:
                try:
                    equity_positions = list(portfolio_mgr.get_equity_positions())
                    for pos in equity_positions:
                        symbol = pos.contract.symbol if hasattr(pos, 'contract') else None
                        current_price = pos.marketPrice if hasattr(pos, 'marketPrice') else None
                        avg_cost = pos.avgCost if hasattr(pos, 'avgCost') else None
                        pos_qty = pos.position if hasattr(pos, 'position') else 0

                        if not symbol or not current_price or current_price <= 0 or not avg_cost or avg_cost <= 0:
                            continue

                        change_pct = (current_price - avg_cost) / avg_cost

                        # Emergency stop: -5% from entry (backup to bracket SL)
                        if change_pct <= -0.05 and pos_qty > 0:
                            logger.warning(
                                "EMERGENCY STOP: %s down %.1f%% ($%.2f → $%.2f). Selling.",
                                symbol, change_pct * 100, avg_cost, current_price,
                            )
                            try:
                                result = await equity_trader.place_market_order(
                                    symbol=symbol, quantity=abs(pos_qty), action="SELL",
                                )
                                _held_tickers.discard(symbol)
                                if trade_journal:
                                    trade_journal.record_trade(
                                        ticker=symbol, action="SELL", quantity=abs(pos_qty),
                                        price=current_price, fill_price=current_price,
                                        strategy_source="EMERGENCY_STOP", regime=current_regime,
                                    )
                            except Exception as e:
                                logger.error("Emergency sell failed for %s: %s", symbol, e)

                        # Partial profit: +4% from entry, sell half
                        elif change_pct >= 0.04 and pos_qty > 1:
                            sell_qty = max(1, int(pos_qty / 2))
                            logger.info(
                                "PARTIAL PROFIT: %s up %.1f%%. Selling %d of %d shares.",
                                symbol, change_pct * 100, sell_qty, int(pos_qty),
                            )
                            try:
                                result = await equity_trader.place_market_order(
                                    symbol=symbol, quantity=sell_qty, action="SELL",
                                )
                                if trade_journal:
                                    trade_journal.record_trade(
                                        ticker=symbol, action="SELL", quantity=sell_qty,
                                        price=current_price, fill_price=current_price,
                                        strategy_source="PARTIAL_PROFIT", regime=current_regime,
                                    )
                            except Exception as e:
                                logger.error("Partial profit sell failed for %s: %s", symbol, e)

                    # NAV cache update
                    intraday_nav = await portfolio_mgr.get_nav()
                    if nav_cache and intraday_nav and intraday_nav > 0:
                        try:
                            nav_cache.save(intraday_nav, peak_nav=portfolio_mgr.peak_nav)
                        except Exception:
                            pass

                    # Kill switch intraday check
                    if intraday_nav and intraday_nav > 0:
                        try:
                            daily_pnl_check = await portfolio_mgr.get_daily_pnl()
                            pnl_pct = daily_pnl_check / intraday_nav * 100 if intraday_nav > 0 else 0.0
                            if pnl_pct <= -cfg.risk.daily_loss_flatten_pct * 100:
                                logger.warning("KILL SWITCH: Daily loss %.1f%%. Flattening all.", pnl_pct)
                                await equity_trader.flatten_all()
                                _held_tickers.clear()
                        except Exception as e:
                            logger.warning("Kill switch check failed: %s", e)

                except Exception as e:
                    logger.warning("Position monitoring error: %s", e)

            # ===========================================================
            # 8. Risk monitor
            # ===========================================================
            if risk_monitor and ib_connected:
                try:
                    risk_result = await risk_monitor.check_risk()
                    if hasattr(risk_result, "should_flatten") and risk_result.should_flatten:
                        logger.warning("Risk monitor triggered flatten!")
                        await risk_monitor.trigger_kill_switch("Risk limit breach")
                        _held_tickers.clear()
                except Exception as e:
                    logger.warning(f"Risk monitor failed: {e}")

        except Exception as e:
            logger.error(f"Signal cycle #{_cycle_count} failed: {e}", exc_info=True)

        # ===========================================================
        # Sleep until next cycle
        # ===========================================================
        logger.info(
            f"Cycle #{_cycle_count} complete. Trades today: {_daily_trades_executed}. "
            f"Holding: {_held_tickers or 'nothing'}. Next cycle in {cycle_interval // 60} min."
        )
        await _interruptible_sleep(cycle_interval, _shutdown_event)

    # ===================================================================
    # Graceful shutdown
    # ===================================================================
    logger.info("Shutting down ATNN v2...")
    if broker["client"] is not None:
        try:
            await broker["client"].disconnect()
        except Exception:
            pass
    if trade_journal:
        try:
            trade_journal.close()
        except Exception as e:
            logger.warning("Trade journal close failed (non-fatal): %s", e)
    trade_log.close()
    print("\nATNN v2 shut down cleanly.")


async def _interruptible_sleep(seconds: float, shutdown_event: asyncio.Event) -> None:
    """Sleep that can be interrupted by shutdown signal."""
    for _ in range(int(seconds)):
        if shutdown_event.is_set():
            break
        await asyncio.sleep(1)


async def _run_eod_reconciliation(
    cfg, portfolio_mgr, ib_connected, nav_cache, trade_journal,
    reporter, kill_switch, _held_tickers, _daily_trades_executed,
    tda_signals, nn_signals, actionable, sized_signals,
    current_regime, market_data, price_df, volume_df, tda_features,
    nn_model, model_dir, _days_since_retrain, _retrain_interval,
) -> None:
    """End-of-day reconciliation, reporting, and retrain check."""
    try:
        if portfolio_mgr and ib_connected:
            positions = await portfolio_mgr.sync_positions()
            nav = await portfolio_mgr.get_nav()
            daily_pnl = await portfolio_mgr.get_daily_pnl()
            logger.info(
                f"EOD: NAV=${nav:,.2f} | Daily P&L=${daily_pnl:,.2f} "
                f"| Positions={positions.get('position_count', 0)} "
                f"| Trades today={_daily_trades_executed}"
            )

            if nav_cache and nav and nav > 0:
                try:
                    nav_cache.save(nav, peak_nav=portfolio_mgr.peak_nav if portfolio_mgr else nav)
                except Exception as e:
                    logger.warning("EOD NAV cache save failed: %s", e)

            if trade_journal:
                try:
                    trade_journal.record_daily_stats(
                        nav, daily_pnl,
                        positions.get("position_count", 0),
                        len(sized_signals),
                    )
                except Exception as e:
                    logger.warning("Trade journal daily stats failed: %s", e)

            if reporter:
                try:
                    report = reporter.generate_report(
                        nav=nav,
                        daily_pnl=daily_pnl,
                        positions=trade_journal.get_open_positions() if trade_journal else [],
                        trades_today=trade_journal.get_recent_trades(limit=50) if trade_journal else [],
                        kelly_params=trade_journal.get_kelly_params() if trade_journal else {"win_rate": 0.52, "avg_win": 0.015, "avg_loss": 0.012},
                        regime=current_regime,
                        signals_generated=len(tda_signals) if not tda_signals.empty else 0,
                        signals_actionable=len(actionable) if not actionable.empty else 0,
                    )
                    reporter.log_report(report)
                except Exception as e:
                    logger.warning("Daily report generation failed: %s", e)
        else:
            logger.info("EOD: No IBKR connection for reconciliation")
    except Exception as e:
        logger.warning(f"EOD reconciliation failed: {e}")

    # Periodic retraining check
    try:
        needs_retrain = _days_since_retrain >= _retrain_interval
        if needs_retrain and market_data and price_df is not None:
            logger.info("=== Triggering periodic NN retrain ===")
            _run_retrain_inline(cfg, price_df, volume_df, tda_features)
            new_files = sorted(model_dir.glob("*.pt")) + sorted(model_dir.glob("*.pth"))
            if new_files and nn_model is not None:
                nn_model.load_state_dict(
                    torch.load(new_files[-1], map_location="cpu", weights_only=True)
                )
                nn_model.eval()
                logger.info(f"Reloaded NN model from {new_files[-1]}")
    except Exception as e:
        logger.warning(f"Periodic retrain check failed: {e}")



def run_live(cfg, dry_run: bool = False) -> None:
    """Synchronous wrapper for the async live loop."""
    asyncio.run(_run_live_async(cfg, dry_run=dry_run))

def _run_retrain_inline(cfg, price_df, volume_df, tda_features) -> None:
    """Retrain NN model inline using current data."""
    try:
        from nn import (
            WalkForwardTrainer, LSTMPredictor, AttentionLSTMPredictor,
            NNFeatureEngine,
        )
        from nn.data_loader import direction_labels

        model_cls = AttentionLSTMPredictor if cfg.nn.model_type == "attention_lstm" else LSTMPredictor

        # Build features + labels
        engine = NNFeatureEngine()
        features = engine.build_features(
            price_df=price_df,
            volume_df=volume_df,
            tda_features_df=tda_features,
        )
        if features.empty:
            logger.warning("No features for retraining — skipping")
            return

        # Create direction labels from returns
        returns = price_df.pct_change().dropna()
        # Use first symbol or mean returns for labels
        if len(returns.columns) > 0:
            avg_returns = returns.mean(axis=1)
        else:
            logger.warning("No return data for labels — skipping retrain")
            return

        target = direction_labels(avg_returns, threshold=cfg.nn.direction_threshold)

        # Align features and target
        common_idx = features.index.intersection(target.index)
        features = features.loc[common_idx]
        target = target.loc[common_idx]

        if len(features) < 100:
            logger.warning(f"Insufficient data for retrain: {len(features)} samples")
            return

        input_size = features.shape[1]

        trainer = WalkForwardTrainer(
            train_window=cfg.backtest.train_window,
            predict_horizon=cfg.backtest.test_window,
            max_epochs=cfg.nn.epochs,
            batch_size=cfg.nn.batch_size,
            lr=cfg.nn.learning_rate,
            patience=cfg.nn.early_stopping_patience,
            checkpoint_dir=cfg.system.model_dir,
        )

        result = trainer.train_walk_forward(
            features_df=features,
            target=target,
            model_class=model_cls,
            window=cfg.nn.sequence_length,
            input_size=input_size,
            hidden_size=cfg.nn.hidden_size,
            num_layers=cfg.nn.num_layers,
            dropout=cfg.nn.dropout,
        )

        if result.metrics_per_fold:
            avg_acc = np.mean([m.accuracy for m in result.metrics_per_fold])
            logger.info(
                f"Retrain complete: {len(result.metrics_per_fold)} folds, "
                f"avg accuracy={avg_acc:.4f}, best model={result.best_model_path}"
            )
        else:
            logger.warning("Retrain produced no folds")

    except Exception as e:
        logger.error(f"Retrain failed: {e}", exc_info=True)


def run_live(cfg, dry_run: bool = False) -> None:
    """Synchronous wrapper for the async live loop."""
    asyncio.run(_run_live_async(cfg, dry_run=dry_run))


# ---------------------------------------------------------------------------
# BACKTEST MODE
# ---------------------------------------------------------------------------

def run_backtest(cfg, options_mode: bool = False, start: str = None, end: str = None) -> None:
    """Run walk-forward backtest."""
    from core.logger import get_trade_logger

    trade_log = get_trade_logger(log_level=cfg.system.log_level)

    if options_mode:
        _run_options_backtest(cfg, start, end)
    else:
        _run_equity_backtest(cfg, start, end)

    trade_log.close()


def _run_equity_backtest(cfg, start: str = None, end: str = None) -> None:
    """Run walk-forward equity backtest using the backtest engine."""
    print(f"\n{'=' * 60}")
    print("  ATNN v2 — WALK-FORWARD BACKTEST")
    print(f"  Capital: ${cfg.backtest.initial_capital:,.2f}")
    print(f"  Train window: {cfg.backtest.train_window} days")
    print(f"  Test window: {cfg.backtest.test_window} days")
    print(f"{'=' * 60}\n")

    try:
        from backtest import WalkForwardOptimizer, BacktestReport

        optimizer = WalkForwardOptimizer(
            train_window=cfg.backtest.train_window,
            test_window=cfg.backtest.test_window,
            purge_gap=cfg.backtest.purge_gap,
            embargo_gap=cfg.backtest.embargo_gap,
        )

        logger.info("Walk-forward optimizer initialized")
        logger.info("Backtest requires historical data — run with IBKR connection or cached data")

        # Generate report stub
        report = BacktestReport()
        output_path = Path(cfg.system.data_dir) / "backtest_report.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Report would be saved to: {output_path}")

    except ImportError as e:
        logger.error(f"Backtest module import failed: {e}")
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)


def _run_options_backtest(cfg, start: str = None, end: str = None) -> None:
    """Run options-specific backtest."""
    print(f"\n{'=' * 60}")
    print("  ATNN v2 — OPTIONS BACKTEST")
    print(f"  Capital: ${cfg.backtest.initial_capital:,.2f}")
    print(f"  Strategies: {cfg.options.strategies}")
    print(f"{'=' * 60}\n")

    try:
        from backtest import OptionsBacktester

        backtester = OptionsBacktester(
            initial_capital=cfg.backtest.initial_capital,
            commission_per_contract=cfg.backtest.commission_per_contract,
        )
        logger.info("Options backtester initialized")
        logger.info("Options backtest requires historical options data")

    except ImportError as e:
        logger.error(f"Options backtest module import failed: {e}")
    except Exception as e:
        logger.error(f"Options backtest failed: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# TRAIN MODE
# ---------------------------------------------------------------------------

def run_train(cfg) -> None:
    """Run walk-forward NN training.

    Connects to IBKR to fetch historical data, builds features,
    and runs walk-forward training with the configured model type.
    """
    print(f"\n{'=' * 60}")
    print("  ATNN v2 — WALK-FORWARD NN TRAINING")
    print(f"  Model type: {cfg.nn.model_type}")
    print(f"  Hidden size: {cfg.nn.hidden_size}")
    print(f"  Epochs: {cfg.nn.epochs}")
    print(f"{'=' * 60}\n")

    try:
        from nn import (
            WalkForwardTrainer, LSTMPredictor, AttentionLSTMPredictor,
            NNFeatureEngine,
        )
        from nn.data_loader import direction_labels

        model_cls = AttentionLSTMPredictor if cfg.nn.model_type == "attention_lstm" else LSTMPredictor
        model_dir = Path(cfg.system.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Try to fetch data from IBKR
        price_df = None
        volume_df = None

        if cfg.broker.is_configured():
            logger.info("Fetching historical data from IBKR for training...")
            try:
                async def _fetch():
                    broker = _create_broker_components(cfg)
                    await broker["client"].connect()
                    data = await broker["data_feed"].get_historical_bars_multi(
                        symbols=cfg.universe.symbols,
                        duration="5 Y",
                        bar_size="1 day",
                    )
                    await broker["client"].disconnect()
                    return data

                market_data = asyncio.run(_fetch())
                closes = {}
                vols = {}
                for sym, df in market_data.items():
                    if df is not None and len(df) > 0:
                        close_col = next((c for c in df.columns if c.lower() == "close"), None)
                        vol_col = next((c for c in df.columns if c.lower() == "volume"), None)
                        if close_col:
                            closes[sym] = df[close_col]
                        if vol_col:
                            vols[sym] = df[vol_col]
                if closes:
                    price_df = pd.DataFrame(closes)
                    volume_df = pd.DataFrame(vols) if vols else None
                    logger.info(f"Fetched {len(price_df)} bars for {len(closes)} symbols")
            except Exception as e:
                logger.warning(f"IBKR data fetch failed: {e}")

        if price_df is None:
            logger.error("No data available for training. Connect to IBKR or provide cached data.")
            return

        # Build features
        engine = NNFeatureEngine()
        tda_features = None
        try:
            from tda import TDAFeatureExtractor
            tda_ext = TDAFeatureExtractor(
                ph_window=cfg.tda.ph_window,
                corr_window=cfg.tda.corr_window,
                diffusion_time=cfg.tda.diffusion_time,
            )
            returns_df = price_df.pct_change().dropna()
            tda_features = tda_ext.extract(returns_df)
        except Exception as e:
            logger.warning(f"TDA features skipped: {e}")

        features = engine.build_features(
            price_df=price_df,
            volume_df=volume_df,
            tda_features_df=tda_features,
        )

        # Create direction labels
        returns = price_df.pct_change().dropna()
        avg_returns = returns.mean(axis=1)
        target = direction_labels(avg_returns, threshold=cfg.nn.direction_threshold)

        # Align
        common_idx = features.index.intersection(target.index)
        features = features.loc[common_idx]
        target = target.loc[common_idx]

        logger.info(f"Training data: {len(features)} samples, {features.shape[1]} features")

        input_size = features.shape[1]
        trainer = WalkForwardTrainer(
            train_window=cfg.backtest.train_window,
            predict_horizon=cfg.backtest.test_window,
            max_epochs=cfg.nn.epochs,
            batch_size=cfg.nn.batch_size,
            lr=cfg.nn.learning_rate,
            patience=cfg.nn.early_stopping_patience,
            checkpoint_dir=str(model_dir),
        )

        result = trainer.train_walk_forward(
            features_df=features,
            target=target,
            model_class=model_cls,
            window=cfg.nn.sequence_length,
            input_size=input_size,
            hidden_size=cfg.nn.hidden_size,
            num_layers=cfg.nn.num_layers,
            dropout=cfg.nn.dropout,
        )

        if result.metrics_per_fold:
            avg_acc = np.mean([m.accuracy for m in result.metrics_per_fold])
            logger.info(
                f"Training complete: {len(result.metrics_per_fold)} folds, "
                f"avg accuracy={avg_acc:.4f}"
            )
            logger.info(f"Best model saved to: {result.best_model_path}")
        else:
            logger.warning("Training produced no folds — insufficient data?")

    except ImportError as e:
        logger.error(f"NN module import failed: {e}")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# STATUS MODE
# ---------------------------------------------------------------------------

def run_status(cfg) -> None:
    """Show system status — config, IBKR connection, positions, P&L."""
    print(f"\n{'=' * 60}")
    print("  ATNN v2 — SYSTEM STATUS")
    print(f"{'=' * 60}")

    # System info
    print(f"\n  System:")
    print(f"    Name:      {cfg.system.name}")
    print(f"    Mode:      {cfg.system.mode}")
    print(f"    Log level: {cfg.system.log_level}")
    print(f"    Data dir:  {cfg.system.data_dir}")
    print(f"    Model dir: {cfg.system.model_dir}")

    # Broker config
    print(f"\n  Broker:")
    print(f"    Host:      {cfg.broker.host}:{cfg.broker.port}")
    print(f"    Client ID: {cfg.broker.client_id}")
    print(f"    Account:   {cfg.broker.account or '(not set)'}")

    # Universe
    print(f"\n  Universe:")
    print(f"    Symbols:   {cfg.universe.symbols}")
    print(f"    Benchmark: {cfg.universe.benchmark}")

    # Risk
    print(f"\n  Risk Limits:")
    print(f"    Max position:     {cfg.risk.max_position_pct:.0%}")
    print(f"    Max sector:       {cfg.risk.max_sector_pct:.0%}")
    print(f"    Max gross exp:    {cfg.risk.max_gross_exposure:.0%}")
    print(f"    Kelly fraction:   {cfg.risk.kelly_fraction}")
    print(f"    Daily loss halt:  {cfg.risk.daily_loss_flatten_pct:.0%}")
    print(f"    Max DD halt:      {cfg.risk.max_drawdown_halt_pct:.0%}")

    # Trading engines
    print(f"\n  Trading Engines:")
    print(f"    Options:   {'ENABLED' if cfg.options.enabled else 'DORMANT'}")
    print(f"    Equities:  {'ENABLED' if cfg.equities.enabled else 'DORMANT'}")

    # Try IBKR connection
    print(f"\n  IBKR Connection:")
    if cfg.broker.is_configured():
        try:
            async def _status_check():
                broker = _create_broker_components(cfg)
                await broker["client"].connect()
                nav = await broker["portfolio_mgr"].get_nav()
                daily_pnl = await broker["portfolio_mgr"].get_daily_pnl()
                positions = await broker["portfolio_mgr"].sync_positions()
                await broker["client"].disconnect()
                return nav, daily_pnl, positions

            nav, daily_pnl, positions = asyncio.run(_status_check())
            print(f"    Status:    CONNECTED")
            print(f"    NAV:       ${nav:,.2f}")
            print(f"    Daily P&L: ${daily_pnl:,.2f}")
            print(f"    Positions: {positions.get('position_count', 0)}")
        except Exception as e:
            print(f"    Status:    NOT CONNECTED ({e})")
    else:
        print(f"    Status:    NOT CONFIGURED (no account set)")

    # NN model status
    print(f"\n  NN Model:")
    model_dir = Path(cfg.system.model_dir)
    model_files = sorted(model_dir.glob("*.pt")) + sorted(model_dir.glob("*.pth"))
    if model_files:
        print(f"    Latest:    {model_files[-1].name}")
        print(f"    Count:     {len(model_files)} model(s)")
    else:
        print(f"    Status:    No trained models found in {model_dir}")

    # Schedule
    print(f"\n  Schedule:")
    print(f"    Signal time:  {cfg.schedule.signal_time} ET")
    print(f"    Recon time:   {cfg.schedule.reconciliation_time} ET")
    print(f"    Market:       {cfg.schedule.market_open} - {cfg.schedule.market_close} ET")

    print(f"\n{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Load config
    from core.config import get_config
    cfg = get_config(config_path=args.config)

    # Override mode for backtest/live
    if args.command in ("live", "backtest"):
        cfg.system.mode = args.command

    setup_logging(cfg.system.log_level)

    logger.info(f"ATNN v2 starting — mode={args.command}")

    if args.command == "live":
        run_live(cfg, dry_run=getattr(args, "dry_run", False))
    elif args.command == "backtest":
        run_backtest(
            cfg,
            options_mode=getattr(args, "options", False),
            start=getattr(args, "start", None),
            end=getattr(args, "end", None),
        )
    elif args.command == "train":
        run_train(cfg)
    elif args.command == "status":
        run_status(cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

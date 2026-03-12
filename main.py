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
import logging
import signal
import sys
import time
from datetime import datetime, date, time as dt_time
from pathlib import Path
from typing import Optional

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


def _wait_until(target_time: dt_time) -> None:
    """Sleep until target_time ET (today). Returns immediately if past."""
    now = _now_et()
    target = now.replace(
        hour=target_time.hour, minute=target_time.minute, second=0, microsecond=0
    )
    delta = (target - now).total_seconds()
    if delta > 0:
        logger.info(f"Waiting {delta / 60:.1f} minutes until {target_time}...")
        time.sleep(delta)


# ---------------------------------------------------------------------------
# LIVE MODE
# ---------------------------------------------------------------------------

def run_live(cfg, dry_run: bool = False) -> None:
    """Run the live/paper trading loop.

    Workflow per trading day:
    1. Wait for signal_time
    2. Fetch market data from IBKR
    3. Compute TDA features
    4. Generate NN predictions
    5. Run ensemble -> sized signals
    6. Check risk constraints
    7. Execute trades (or log-only if dormant/dry-run)
    8. Wait for reconciliation_time
    9. Run EOD reconciliation
    10. Log daily summary
    """
    from core.logger import get_trade_logger
    from core.market_hours import MarketCalendar
    from core.kill_switch import KillSwitch, CircuitBreakerConfig

    trade_log = get_trade_logger(log_level=cfg.system.log_level)
    market_cal = MarketCalendar()

    # Graceful shutdown
    _shutdown = False

    def _handle_signal(signum, frame):
        nonlocal _shutdown
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        _shutdown = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Parse schedule times
    signal_time = _parse_time(cfg.schedule.signal_time)
    recon_time = _parse_time(cfg.schedule.reconciliation_time)

    # Kill switch
    kill_switch = KillSwitch(
        config=CircuitBreakerConfig(
            max_drawdown_pct=-cfg.risk.max_drawdown_halt_pct,
            max_daily_loss_pct=-cfg.risk.daily_loss_flatten_pct,
        ),
        initial_equity=cfg.backtest.initial_capital,
    )

    # Try to connect to IBKR
    ib_connected = False
    data_feed = None
    portfolio_mgr = None
    equity_trader = None
    option_trader = None
    risk_monitor = None

    if cfg.broker.is_configured():
        try:
            from broker.data_feed import IBKRDataFeed
            from broker.portfolio_manager import PortfolioManager
            from broker.equity_trader import EquityTrader
            from broker.option_trader import OptionTrader
            from broker.risk_monitor import RiskMonitor

            data_feed = IBKRDataFeed(
                host=cfg.broker.host,
                port=cfg.broker.port,
                client_id=cfg.broker.client_id,
            )
            portfolio_mgr = PortfolioManager(
                host=cfg.broker.host,
                port=cfg.broker.port,
                client_id=cfg.broker.client_id + 1,
                account=cfg.broker.account,
            )
            equity_trader = EquityTrader(
                host=cfg.broker.host,
                port=cfg.broker.port,
                client_id=cfg.broker.client_id + 2,
                account=cfg.broker.account,
            )
            option_trader = OptionTrader(
                host=cfg.broker.host,
                port=cfg.broker.port,
                client_id=cfg.broker.client_id + 3,
                account=cfg.broker.account,
            )
            risk_monitor = RiskMonitor(
                portfolio_manager=portfolio_mgr,
            )
            risk_monitor.register_traders(equity_trader, option_trader)
            ib_connected = True
            logger.info("Connected to IBKR")
        except Exception as e:
            logger.warning(f"IBKR connection failed: {e}. Running in signal-only mode.")

    # Initialize TDA
    tda_extractor = None
    try:
        from tda import TDAFeatureExtractor
        tda_extractor = TDAFeatureExtractor(
            ph_window=cfg.tda.ph_window,
            corr_window=cfg.tda.corr_window,
            diffusion_time=cfg.tda.diffusion_time,
        )
        logger.info("TDA feature extractor initialized")
    except Exception as e:
        logger.warning(f"TDA init failed: {e}")

    # Load NN model
    nn_model = None
    model_dir = Path(cfg.system.model_dir)
    try:
        from nn import LSTMPredictor, AttentionLSTMPredictor
        model_cls = AttentionLSTMPredictor if cfg.nn.model_type == "attention_lstm" else LSTMPredictor
        # Look for latest saved model
        model_files = sorted(model_dir.glob("*.pt")) + sorted(model_dir.glob("*.pth"))
        if model_files:
            import torch
            nn_model = model_cls(
                input_size=10,  # will be overridden by checkpoint
                hidden_size=cfg.nn.hidden_size,
                num_layers=cfg.nn.num_layers,
                dropout=cfg.nn.dropout,
            )
            nn_model.load_state_dict(torch.load(model_files[-1], map_location="cpu"))
            nn_model.eval()
            logger.info(f"Loaded NN model from {model_files[-1]}")
        else:
            logger.info("No trained NN model found; signals will be TDA-only")
    except Exception as e:
        logger.warning(f"NN model load failed: {e}")

    # Initialize ensemble
    meta_allocator = None
    signal_aggregator = None
    ensemble_risk = None
    try:
        from ensemble import MetaAllocator, SignalAggregator, EnsembleRiskManager
        meta_allocator = MetaAllocator(
            default_tda_weight=cfg.ensemble.default_tda_weight,
            default_nn_weight=cfg.ensemble.default_nn_weight,
        )
        signal_aggregator = SignalAggregator(
            min_signal_strength=cfg.ensemble.min_signal_strength,
            agreement_bonus=cfg.ensemble.agreement_bonus,
            disagreement_penalty=cfg.ensemble.disagreement_penalty,
        )
        ensemble_risk = EnsembleRiskManager(
            max_position_pct=cfg.risk.max_position_pct,
            kelly_fraction=cfg.risk.kelly_fraction,
        )
        logger.info("Ensemble components initialized")
    except Exception as e:
        logger.warning(f"Ensemble init failed: {e}")

    options_enabled = cfg.options.enabled and not dry_run
    equities_enabled = cfg.equities.enabled and not dry_run

    mode_str = "DRY-RUN" if dry_run else ("LIVE" if cfg.system.mode == "live" else "PAPER")
    print(f"\n{'=' * 60}")
    print(f"  ATNN v2 — {mode_str} MODE")
    print(f"  Symbols: {cfg.universe.symbols}")
    print(f"  Options: {'ENABLED' if options_enabled else 'DORMANT'}")
    print(f"  Equities: {'ENABLED' if equities_enabled else 'DORMANT'}")
    print(f"  IBKR: {'Connected' if ib_connected else 'Not connected'}")
    print(f"  Signal time: {cfg.schedule.signal_time} ET")
    print(f"{'=' * 60}\n")

    _last_trading_date = None

    while not _shutdown:
        # Market hours gate
        if not market_cal.is_market_open():
            next_open = market_cal.next_open()
            logger.info(f"Market closed. Next open: {next_open.strftime('%Y-%m-%d %H:%M ET')}")
            # Sleep in short intervals to check for shutdown
            for _ in range(60):
                if _shutdown:
                    break
                time.sleep(1)
            continue

        today = date.today()

        # Daily reset
        if _last_trading_date != today:
            _last_trading_date = today
            kill_switch.reset_daily(cfg.backtest.initial_capital)
            logger.info(f"=== New trading day: {today} ===")

        # Kill switch check
        if not kill_switch.is_trading_allowed():
            kill_switch.check_cooldown_expired()
        if not kill_switch.is_trading_allowed():
            logger.warning(f"Trading halted: {kill_switch.block_reason}")
            time.sleep(60)
            continue

        # Wait for signal time
        _wait_until(signal_time)
        if _shutdown:
            break

        logger.info("--- Signal generation cycle ---")

        try:
            # Fetch market data
            market_data = None
            if data_feed and ib_connected:
                try:
                    market_data = data_feed.get_historical_bars_multi(
                        symbols=cfg.universe.symbols,
                        duration="1 Y",
                        bar_size="1 day",
                    )
                    logger.info(f"Fetched data for {len(market_data)} symbols")
                except Exception as e:
                    logger.error(f"Data fetch failed: {e}")

            # Compute TDA features
            tda_features = None
            if tda_extractor and market_data:
                try:
                    import pandas as pd
                    # Build returns matrix from closes
                    closes = {}
                    for sym, df in market_data.items():
                        if df is not None and len(df) > 0:
                            close_col = next((c for c in df.columns if c.lower() == "close"), None)
                            if close_col:
                                closes[sym] = df[close_col]
                    if closes:
                        price_df = pd.DataFrame(closes)
                        returns_df = price_df.pct_change().dropna()
                        tda_features = tda_extractor.extract(returns_df)
                        logger.info(f"TDA features: {tda_features.columns.tolist()}")
                except Exception as e:
                    logger.warning(f"TDA feature extraction failed: {e}")

            # Generate NN predictions
            nn_predictions = None
            if nn_model and market_data:
                try:
                    logger.info("NN predictions: model loaded but inference skipped (no live pipeline yet)")
                except Exception as e:
                    logger.warning(f"NN prediction failed: {e}")

            # Run ensemble
            ensemble_signals = None
            if signal_aggregator and (tda_features is not None or nn_predictions is not None):
                try:
                    logger.info("Ensemble: combining available signals")
                    # In a full implementation, this would call signal_aggregator.aggregate()
                except Exception as e:
                    logger.warning(f"Ensemble aggregation failed: {e}")

            # Check risk constraints
            if ensemble_risk and ensemble_signals:
                try:
                    logger.info("Risk check: evaluating position sizes")
                except Exception as e:
                    logger.warning(f"Risk check failed: {e}")

            # Execute trades (or log only)
            if ensemble_signals and not dry_run:
                if options_enabled and option_trader:
                    logger.info("Options execution: would place trades here")
                if equities_enabled and equity_trader:
                    logger.info("Equity execution: would place trades here")
            else:
                logger.info("Signal-only mode: no trades executed")

            # Risk monitor check
            if risk_monitor and ib_connected:
                try:
                    risk_result = risk_monitor.check_risk()
                    logger.info(f"Risk check: {risk_result}")
                except Exception as e:
                    logger.warning(f"Risk monitor failed: {e}")

        except Exception as e:
            logger.error(f"Signal cycle failed: {e}", exc_info=True)

        # Wait for reconciliation time
        _wait_until(recon_time)
        if _shutdown:
            break

        # EOD reconciliation
        try:
            if portfolio_mgr and ib_connected:
                positions = portfolio_mgr.sync_positions()
                nav = portfolio_mgr.get_nav()
                daily_pnl = portfolio_mgr.get_daily_pnl()
                logger.info(f"EOD: NAV=${nav:,.2f} | Daily P&L=${daily_pnl:,.2f} | Positions={positions.get('position_count', 0)}")
            else:
                logger.info("EOD: No IBKR connection for reconciliation")
        except Exception as e:
            logger.warning(f"EOD reconciliation failed: {e}")

        # Sleep until next day
        logger.info("Trading day complete. Sleeping until next market open.")
        while not _shutdown and market_cal.is_market_open():
            time.sleep(30)

    # Graceful shutdown
    logger.info("Shutting down ATNN v2...")
    trade_log.close()
    print("\nATNN v2 shut down cleanly.")


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
    """Run walk-forward NN training."""
    print(f"\n{'=' * 60}")
    print("  ATNN v2 — WALK-FORWARD NN TRAINING")
    print(f"  Model type: {cfg.nn.model_type}")
    print(f"  Hidden size: {cfg.nn.hidden_size}")
    print(f"  Epochs: {cfg.nn.epochs}")
    print(f"{'=' * 60}\n")

    try:
        from nn import WalkForwardTrainer, LSTMPredictor, AttentionLSTMPredictor

        model_cls = AttentionLSTMPredictor if cfg.nn.model_type == "attention_lstm" else LSTMPredictor

        trainer = WalkForwardTrainer(
            model_class=model_cls,
            train_window=cfg.backtest.train_window,
            predict_horizon=cfg.backtest.test_window,
            max_epochs=cfg.nn.epochs,
            batch_size=cfg.nn.batch_size,
            lr=cfg.nn.learning_rate,
            patience=cfg.nn.early_stopping_patience,
        )
        logger.info("Walk-forward trainer initialized")
        logger.info("Training requires historical data — run with IBKR connection or cached data")

        # Save location
        model_dir = Path(cfg.system.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Models will be saved to: {model_dir}")

    except ImportError as e:
        logger.error(f"NN module import failed: {e}")
    except Exception as e:
        logger.error(f"Training setup failed: {e}", exc_info=True)


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
            from broker.portfolio_manager import PortfolioManager
            pm = PortfolioManager(
                host=cfg.broker.host,
                port=cfg.broker.port,
                client_id=cfg.broker.client_id + 10,
                account=cfg.broker.account,
            )
            nav = pm.get_nav()
            daily_pnl = pm.get_daily_pnl()
            positions = pm.sync_positions()
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

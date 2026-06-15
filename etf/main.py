"""
etf/main.py
===========
Command-line entry point for the ETF tactical-allocation engine.

Modes
-----
backtest : historical simulation on yfinance data, prints full metrics.
signal   : compute *today's* target weights from latest data (no trading).
paper    : connect to IBKR paper account and rebalance to target (dry-run safe).
live     : connect to IBKR live account and rebalance (requires explicit flag).

Examples
--------
    python -m etf.main --mode backtest --start 2007-01-01
    python -m etf.main --mode signal
    python -m etf.main --mode paper            # plans orders, dry-run by default
    python -m etf.main --mode paper --execute  # actually submits to paper acct
    python -m etf.main --mode live --execute --i-understand-the-risk
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Optional

import pandas as pd

from etf.config import ETFConfig, get_default_config
from etf.data import load_price_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("etf.main")


def _print_metrics(title: str, m, benchmark=None) -> None:
    print(f"\n{'='*64}\n{title}\n{'='*64}")
    rows = [
        ("Total return", f"{m.total_return:>10.2%}"),
        ("CAGR", f"{m.cagr:>10.2%}"),
        ("Annual vol", f"{m.annual_volatility:>10.2%}"),
        ("Sharpe", f"{m.sharpe:>10.2f}"),
        ("Sortino", f"{m.sortino:>10.2f}"),
        ("Max drawdown", f"{m.max_drawdown:>10.2%}"),
        ("Calmar", f"{m.calmar:>10.2f}"),
        ("Win rate (daily)", f"{m.win_rate:>10.2%}"),
        ("Profit factor", f"{m.profit_factor:>10.2f}"),
        ("VaR 95% (daily)", f"{m.var_95:>10.2%}"),
        ("CVaR 95% (daily)", f"{m.cvar_95:>10.2%}"),
        ("Avg gross exposure", f"{m.avg_gross_exposure:>10.2%}"),
        ("Annual turnover", f"{m.turnover_annual:>10.2f}x"),
        ("Alpha (ann.)", f"{m.alpha:>10.2%}"),
        ("Beta", f"{m.beta:>10.2f}"),
    ]
    for label, val in rows:
        print(f"  {label:<22}{val}")
    if benchmark is not None:
        print(f"  {'-'*40}")
        print(f"  {'Benchmark CAGR':<22}{benchmark.cagr:>10.2%}")
        print(f"  {'Benchmark Sharpe':<22}{benchmark.sharpe:>10.2f}")
        print(f"  {'Benchmark MaxDD':<22}{benchmark.max_drawdown:>10.2%}")


def cmd_backtest(cfg: ETFConfig, args) -> int:
    from etf.backtest import run_backtest

    if args.start:
        cfg.backtest.start = args.start
    if args.end:
        cfg.backtest.end = args.end

    logger.info("Loading data for %d symbols ...", len(cfg.all_symbols))
    prices = load_price_history(
        cfg.all_symbols, cfg.backtest.start, cfg.backtest.end, refresh=args.refresh
    )
    logger.info("Loaded %d rows %s -> %s", len(prices), prices.index.min().date(), prices.index.max().date())

    result = run_backtest(prices, cfg)
    _print_metrics("ETF ENGINE — BACKTEST", result.metrics, result.benchmark_metrics)

    print(f"\n  Rebalances: {len(result.rebalance_dates)}")
    last_w = result.weights_history.iloc[-1]
    held = last_w[last_w > 1e-4].sort_values(ascending=False)
    print("  Final target weights:")
    if held.empty:
        print("    (fully defensive — all cash)")
    for sym, w in held.items():
        print(f"    {sym:<6}{w:>8.2%}")

    if args.out:
        payload = {
            "metrics": result.metrics.as_dict(),
            "benchmark": result.benchmark_metrics.as_dict() if result.benchmark_metrics else None,
            "config": {
                "universe": cfg.risk_universe,
                "target_vol": cfg.risk.target_volatility,
                "top_k": cfg.signal.top_k,
                "rebalance_every": cfg.execution.rebalance_every,
            },
            "rebalances": len(result.rebalance_dates),
            "start": str(prices.index.min().date()),
            "end": str(prices.index.max().date()),
        }
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        logger.info("Wrote results to %s", args.out)
    return 0


def cmd_validate(cfg: ETFConfig, args) -> int:
    from etf.validation import run_validation

    if args.start:
        cfg.backtest.start = args.start
    if args.end:
        cfg.backtest.end = args.end

    logger.info("Loading data for %d symbols ...", len(cfg.all_symbols))
    prices = load_price_history(
        cfg.all_symbols, cfg.backtest.start, cfg.backtest.end, refresh=args.refresh
    )
    logger.info("Running Phase 0 validation battery (this runs many backtests) ...")
    report = run_validation(prices, cfg)
    print("\n" + report.summary())

    if args.out:
        payload = {
            "full_metrics": report.full_metrics.as_dict(),
            "deflated_sharpe": report.deflated_sharpe,
            "pbo": report.pbo,
            "cpcv_sharpe_median": float(report.cpcv_sharpes.mean()) if len(report.cpcv_sharpes) else None,
            "walk_forward": report.walk_forward.to_dict(orient="records"),
            "capacity_aum": (None if report.capacity.capacity_aum == float("inf")
                             else report.capacity.capacity_aum),
        }
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        logger.info("Wrote validation report to %s", args.out)
    return 0


def cmd_sleeves(cfg: ETFConfig, args) -> int:
    from etf.sleeve_analysis import analyze_sleeve_set, default_sleeves

    if args.start:
        cfg.backtest.start = args.start
    if args.end:
        cfg.backtest.end = args.end

    logger.info("Loading data for %d symbols ...", len(cfg.all_symbols))
    prices = load_price_history(
        cfg.all_symbols, cfg.backtest.start, cfg.backtest.end, refresh=args.refresh
    )
    sleeves = default_sleeves(cfg)
    # Research mode: append a candidate sleeve to the production roster so its
    # standalone OOS edge, correlation, and blend uplift can be measured before
    # any promotion decision.
    if getattr(args, "candidate", None):
        from etf.sleeves import CrossSectionalSleeve, TurnOfMonthSleeve
        candidates = {
            "seasonality": TurnOfMonthSleeve(cfg),
            "cross_sectional": CrossSectionalSleeve(cfg),
        }
        cand = candidates.get(args.candidate)
        if cand is None:
            logger.error("Unknown candidate sleeve: %s", args.candidate)
            return 2
        sleeves = list(sleeves) + [cand]
    logger.info("Backtesting %d sleeves: %s ...", len(sleeves), ", ".join(s.name for s in sleeves))
    report = analyze_sleeve_set(prices, sleeves, cfg)
    print("\n" + report.summary())

    if args.out:
        payload = {
            "names": report.names,
            "metrics": {n: report.metrics[n].as_dict() for n in report.names},
            "correlation": report.corr_matrix.round(4).to_dict(),
            "cpcv_median": report.cpcv_median,
            "cpcv_p_positive": report.cpcv_p_positive,
            "dsr": report.dsr,
            "combo_inv_vol": report.combo_inv_vol.as_dict(),
            "overlap_days": report.overlap_days,
        }
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        logger.info("Wrote sleeve analysis to %s", args.out)
    return 0


def cmd_portfolio(cfg: ETFConfig, args) -> int:
    """Phase 3: build the ERC-combined multi-sleeve book and report vs the gate."""
    import numpy as np

    from etf.portfolio import run_combined_backtest
    from etf.sleeve_analysis import default_sleeves
    from etf.validation import cpcv_oos_sharpes, deflated_sharpe_ratio

    if args.start:
        cfg.backtest.start = args.start
    if args.end:
        cfg.backtest.end = args.end
    if args.method:
        cfg.portfolio.method = args.method
    # Phase 4 return lever: raise the cap and turn on the book-level circuit
    # breaker. Defaults preserve the Phase 3 book (cap 1.0, no de-risk).
    if args.max_leverage is not None:
        cfg.portfolio.max_leverage = args.max_leverage
    if args.derisk:
        cfg.portfolio.dd_derisk = True
    if getattr(args, "vol_managed", False):
        cfg.vol_managed.enabled = True
    levered = cfg.portfolio.max_leverage > 1.0 or cfg.portfolio.dd_derisk
    phase = "PHASE 4 LEVERED BOOK" if levered else "PHASE 3 PORTFOLIO CONSTRUCTION"

    logger.info("Loading data for %d symbols ...", len(cfg.all_symbols))
    prices = load_price_history(
        cfg.all_symbols, cfg.backtest.start, cfg.backtest.end, refresh=args.refresh
    )
    sleeves = default_sleeves(cfg)
    sleeve_names = [s.name for s in sleeves]
    logger.info(
        "Combining %d sleeves via %s: %s",
        len(sleeves), cfg.portfolio.method, ", ".join(sleeve_names),
    )

    # --- Method comparison (equal / inverse_vol / erc) on identical sleeves ---
    comparison = {}
    for method in ("equal", "inverse_vol", "erc"):
        cfg.portfolio.method = method
        r = run_combined_backtest(prices, cfg, sleeves)
        comparison[method] = r

    print(f"\n{'='*72}\n{phase}  (vol target "
          f"{cfg.portfolio.target_volatility:.0%}, lev cap {cfg.portfolio.max_leverage:.1f}x, "
          f"DD-derisk {'ON' if cfg.portfolio.dd_derisk else 'OFF'})\n{'='*72}")
    hdr = f"{'method':<14}{'Sharpe':>8}{'Sortino':>9}{'CAGR':>9}{'Vol':>8}{'MaxDD':>9}{'Calmar':>8}{'PF':>7}"
    print(hdr)
    print("-" * len(hdr))
    for method in ("equal", "inverse_vol", "erc"):
        m = comparison[method].metrics
        print(f"{method:<14}{m.sharpe:>8.2f}{m.sortino:>9.2f}{m.cagr:>9.2%}"
              f"{m.annual_volatility:>8.2%}{m.max_drawdown:>9.2%}{m.calmar:>8.2f}{m.profit_factor:>7.2f}")

    # --- Focus on the configured method (default erc) for the gate + OOS ---
    primary = args.method or "erc"
    cfg.portfolio.method = primary
    res = comparison[primary]
    m = res.metrics

    # OOS robustness on the COMBINED return stream.
    combined_ret = res.returns.loc[res.gross_exposure[res.gross_exposure > 0].index]
    cpcv = cpcv_oos_sharpes(combined_ret, n_groups=8, k_test=2, purge=5)
    cpcv_median = float(np.median(cpcv)) if len(cpcv) else float("nan")
    cpcv_p_pos = float((cpcv > 0).mean()) if len(cpcv) else float("nan")
    cpcv_pp = cpcv_oos_sharpes(combined_ret, n_groups=8, k_test=2, purge=5, annualize=False)
    trials = cpcv_pp if len(cpcv_pp) else [0.0]
    dsr = deflated_sharpe_ratio(combined_ret, trial_sharpes=trials).get("dsr", float("nan"))

    print(f"\nOOS robustness of the {primary.upper()} book (CPCV 8C2 + DSR):")
    print(f"  median Sharpe = {cpcv_median:>6.2f}   "
          f"P(SR>0) = {cpcv_p_pos:>5.1%}   DSR = {dsr:>5.3f}")

    gate = {
        "OOS Sharpe >= 1.10": m.sharpe >= 1.10,
        "Max drawdown <= 18%": m.max_drawdown >= -0.18,
        "Calmar >= 0.80": m.calmar >= 0.80,
        "Profit factor >= 1.20": m.profit_factor >= 1.20,
    }
    print(f"\nPromotion gate ({primary.upper()} book, "
          f"{'levered' if levered else 'unlevered'}):")
    for label, ok in gate.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"\n  => {'GATE CLEARED' if all(gate.values()) else 'GATE NOT CLEARED'}")

    # Average ERC capital allocation across sleeves.
    avg_alloc = res.sleeve_weights.replace(0.0, np.nan).mean().fillna(0.0)
    print("\nAverage capital allocation (when deployed):")
    for name in sleeve_names:
        print(f"    {name:<18}{avg_alloc.get(name, 0.0):>8.2%}")
    print(f"    {'avg gross':<18}{res.gross_exposure.mean():>8.2%}")

    if args.out:
        payload = {
            "method": primary,
            "sleeves": sleeve_names,
            "max_leverage": cfg.portfolio.max_leverage,
            "dd_derisk": cfg.portfolio.dd_derisk,
            "comparison": {k: comparison[k].metrics.as_dict() for k in comparison},
            "metrics": m.as_dict(),
            "cpcv_median": cpcv_median,
            "cpcv_p_positive": cpcv_p_pos,
            "dsr": dsr,
            "gate": gate,
            "gate_cleared": all(gate.values()),
            "avg_allocation": {n: float(avg_alloc.get(n, 0.0)) for n in sleeve_names},
            "avg_gross_exposure": float(res.gross_exposure.mean()),
        }
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        logger.info("Wrote portfolio analysis to %s", args.out)
    return 0


def cmd_signal(cfg: ETFConfig, args) -> int:
    """Today's COMBINED multi-sleeve target weights (the live-traded book).

    Uses the same combiner allocation as the validated backtest, so the signal
    you see is exactly what paper/live will trade — no backtest/live drift.
    """
    from etf.portfolio import live_target_weights
    from etf.sleeve_analysis import default_sleeves

    if args.method:
        cfg.portfolio.method = args.method
    if args.max_leverage is not None:
        cfg.portfolio.max_leverage = args.max_leverage
    if args.derisk:
        cfg.portfolio.dd_derisk = True

    prices = load_price_history(cfg.all_symbols, cfg.backtest.start, None, refresh=args.refresh)
    sleeves = default_sleeves(cfg)
    alloc = live_target_weights(prices, cfg, sleeves)

    print(f"\n{'='*64}\nTODAY'S COMBINED TARGET WEIGHTS ({alloc.as_of.date()})\n{'='*64}")
    print(f"  Combiner: {cfg.portfolio.method}  |  vol-target {cfg.portfolio.target_volatility:.0%}"
          f"  |  lev cap {cfg.portfolio.max_leverage:.2f}x"
          f"  |  DD-derisk {'ON' if cfg.portfolio.dd_derisk else 'off'}")
    print("  Sleeve capital allocation:")
    for name, a in alloc.sleeve_alloc.items():
        print(f"    {name:<18}{a:>8.2%}")
    print(f"  Vol-target scale: {alloc.vol_scale:.2f}")
    print(f"  Gross exposure: {alloc.gross_exposure:.2%}  |  Cash: {alloc.cash_weight:.2%}")
    print("  Combined ETF weights:")
    for sym, w in sorted(alloc.weights.items(), key=lambda kv: -kv[1]):
        if abs(w) > 1e-4:
            print(f"    {sym:<6}{w:>8.2%}")
    return 0


async def _preflight(cfg: ETFConfig, args) -> int:
    """Read-only IBKR + data readiness check. Submits NOTHING.

    Verifies the four things that must be true before paper/live trading:
      1. price data loads and is fresh,
      2. today's combined target book computes (the validated combiner),
      3. IBKR connects and returns an account snapshot, and
      4. a live price is obtainable for at least one target symbol,
    then runs the pre-trade safety gate in report-only mode and prints a clear
    GO / NO-GO summary. This is the first thing to run against a new IBKR paper
    account.
    """
    from etf.ibkr_broker import IBKRETFBroker
    from etf.portfolio import live_target_weights
    from etf.safety import pretrade_safety_check
    from etf.sleeve_analysis import default_sleeves

    checks: list = []

    # 1. Data + 2. target book ------------------------------------------------
    prices = load_price_history(cfg.all_symbols, cfg.backtest.start, None, refresh=args.refresh)
    data_fresh = prices is not None and not prices.empty
    last_bar = prices.index[-1].date() if data_fresh else None
    checks.append((
        "Price data loads & non-empty",
        data_fresh,
        f"last bar {last_bar}, {prices.shape[1]} symbols" if data_fresh else "no data",
    ))

    target: dict = {}
    alloc = None
    if data_fresh:
        try:
            alloc = live_target_weights(prices, cfg, default_sleeves(cfg))
            target = {k: v for k, v in alloc.weights.items() if abs(v) > 1e-4}
            checks.append((
                "Combined target book computes",
                True,
                f"{len(target)} holdings, gross {alloc.gross_exposure:.1%}, cash {alloc.cash_weight:.1%}",
            ))
        except Exception as exc:
            checks.append(("Combined target book computes", False, str(exc)))

    # 3. IBKR connection + 4. account + price --------------------------------
    broker = IBKRETFBroker(cfg.ibkr, dry_run=True)  # read-only: never submits
    connected = await broker.connect()
    checks.append((
        "IBKR connection",
        connected,
        f"{cfg.ibkr.host}:{cfg.ibkr.port} client {cfg.ibkr.client_id}" if connected
        else f"could not connect to {cfg.ibkr.host}:{cfg.ibkr.port} (is Gateway/TWS running?)",
    ))

    account = None
    if connected:
        try:
            account = await broker.get_account()
            ok_acct = account is not None and account.equity > 0
            checks.append((
                "Account snapshot",
                ok_acct,
                f"equity ${account.equity:,.2f}, cash ${account.cash:,.2f}, "
                f"{len(account.positions)} positions" if ok_acct else "no account data",
            ))
            # Probe a live price for one target (or the benchmark as a fallback).
            # A missing quote only indicates a real problem (e.g. data permissions)
            # when the market is actually OPEN. When the market is closed (nights/
            # weekends/holidays) a paper account legitimately returns no quote, so we
            # mark the probe as a non-blocking SKIP rather than failing readiness.
            probe = next(iter(target), cfg.benchmark)
            px = await broker.get_price(probe)
            try:
                from core.market_hours import MarketCalendar
                market_open = MarketCalendar().is_market_open()
            except Exception:
                market_open = False
            if px is not None and px > 0:
                probe_ok, probe_detail = True, f"${px:.2f}"
            elif market_open:
                probe_ok = False
                probe_detail = "no quote during market hours (check market-data permissions)"
            else:
                probe_ok = None  # advisory: cannot quote while market closed
                probe_detail = "market closed — live quote check skipped (verifies when open)"
            checks.append((f"Live price probe ({probe})", probe_ok, probe_detail))
        except Exception as exc:
            checks.append(("Account snapshot", False, str(exc)))
        finally:
            await broker.disconnect()

    # 5. Pre-trade safety gate (report-only) ---------------------------------
    if alloc is not None:
        decision = pretrade_safety_check(
            cfg, current_drawdown=0.0, daily_pnl_pct=0.0,
            gross_exposure=alloc.gross_exposure, reconciliation_ok=True,
            data_is_fresh=data_fresh,
        )
        checks.append((
            "Pre-trade safety gate",
            decision.allowed,
            "clear" if decision.allowed else "; ".join(decision.reasons),
        ))

    print(f"\n{'='*64}\nETF PREFLIGHT — READINESS CHECK\n{'='*64}")
    all_ok = True
    for label, ok, detail in checks:
        # ok is True -> PASS, False -> FAIL (blocks readiness), None -> SKIP (advisory)
        if ok is False:
            all_ok = False
            tag = "FAIL"
        elif ok is None:
            tag = "SKIP"
        else:
            tag = "PASS"
        print(f"  [{tag}] {label:<32} {detail}")
    if target:
        print("\n  Planned (UN-submitted) target book:")
        for sym, w in sorted(target.items(), key=lambda kv: -kv[1]):
            print(f"    {sym:<6}{w:>8.2%}")
    print(f"\n  => {'READY for paper trading' if all_ok else 'NOT READY — resolve FAILs above'}")
    return 0 if all_ok else 5


def cmd_preflight(cfg: ETFConfig, args) -> int:
    return asyncio.run(_preflight(cfg, args))


async def _trade(cfg: ETFConfig, args, live: bool) -> int:
    from etf.ibkr_broker import IBKRETFBroker
    from etf.portfolio import live_target_weights
    from etf.safety import pretrade_safety_check
    from etf.state import (
        load_reconciliation_state,
        load_state,
        save_reconciliation_state,
        save_state,
        update_state,
    )
    from etf.sleeve_analysis import default_sleeves

    if args.method:
        cfg.portfolio.method = args.method
    if args.max_leverage is not None:
        cfg.portfolio.max_leverage = args.max_leverage
    if args.derisk:
        cfg.portfolio.dd_derisk = True

    prices = load_price_history(cfg.all_symbols, cfg.backtest.start, None, refresh=args.refresh)
    sleeves = default_sleeves(cfg)
    alloc = live_target_weights(prices, cfg, sleeves)
    target = {k: v for k, v in alloc.weights.items() if abs(v) > 1e-4}
    logger.info(
        "Combined target (%s): %s (cash %.1f%%, gross %.1f%%)",
        cfg.portfolio.method, target, 100 * alloc.cash_weight, 100 * alloc.gross_exposure,
    )

    # --- Early data-freshness gate (block before connecting) ------------
    data_fresh = prices is not None and not prices.empty
    if not data_fresh:
        logger.error("Pre-trade safety: market data is stale or missing — no orders.")
        return 4

    dry_run = not args.execute
    broker = IBKRETFBroker(cfg.ibkr, dry_run=dry_run)
    if not await broker.connect():
        logger.error("Could not connect to IBKR; aborting.")
        return 2
    try:
        # --- Update persistent equity state -> live drawdown / daily P&L ---
        # The kill-switch needs memory across cycles (peak + start-of-day equity)
        # that survives restarts. We read the broker's equity, advance the state,
        # and feed real drawdown / daily-P&L into the kill-switch. If no account
        # snapshot is available (e.g. dry-run with no gateway), fall back to the
        # safe 0.0 defaults — the data-freshness and gross-cap guards still apply.
        current_drawdown, daily_pnl_pct = 0.0, 0.0
        account = await broker.get_account()
        if account is not None and account.equity > 0:
            prev_state = load_state(cfg.execution.state_path)
            state, current_drawdown, daily_pnl_pct = update_state(prev_state, account.equity)
            save_state(state, cfg.execution.state_path)
            logger.info(
                "Equity state: equity $%.2f, peak $%.2f, drawdown %.2f%%, daily P&L %.2f%%",
                account.equity, state.peak_equity, 100 * current_drawdown, 100 * daily_pnl_pct,
            )
        else:
            logger.warning("No account snapshot — kill-switch drawdown/daily-P&L default to 0.")

        # --- Prior-cycle reconciliation memory (Phase 5) ----------------
        # If the previous cycle left an unresolved book mismatch, do NOT trade on
        # top of an inconsistent book — block until a human investigates and
        # resets (delete the recon-state file). A missing file => no prior cycle
        # => reconciled (a fresh deployment is free to trade).
        prior_recon = load_reconciliation_state(cfg.execution.recon_state_path)
        reconciliation_ok = True if prior_recon is None else prior_recon.ok
        if not reconciliation_ok:
            logger.warning(
                "Prior cycle left an unresolved reconciliation mismatch (%s) — "
                "blocking until reviewed/reset.", prior_recon.mismatches,
            )

        # --- Pre-trade kill-switch (Phase 5) ----------------------------
        decision = pretrade_safety_check(
            cfg,
            current_drawdown=current_drawdown,
            daily_pnl_pct=daily_pnl_pct,
            gross_exposure=alloc.gross_exposure,
            reconciliation_ok=reconciliation_ok,
            data_is_fresh=data_fresh,
        )
        if not decision.allowed:
            for reason in decision.reasons:
                logger.error("Pre-trade safety: %s", reason)
            if decision.halt:
                logger.error("KILL-SWITCH ENGAGED — trading halted, human reset required.")
            else:
                logger.error("Pre-trade safety blocked this cycle — no orders submitted.")
            return 4

        result = await broker.rebalance_to_weights(target, cfg)
        logger.info("Rebalance result: %s", result)
        # Give submitted orders time to fill before measuring/reconciling — else
        # positions haven't updated yet and reconciliation flags a spurious
        # mismatch that would block the next cycle. No-op in dry-run.
        settled = await broker.await_fills(cfg.execution.fill_timeout_seconds)
        if not settled:
            logger.warning(
                "Not all orders settled within %.0fs; reconciliation may flag a "
                "transient mismatch (will re-check next cycle).",
                cfg.execution.fill_timeout_seconds,
            )
        # Slippage telemetry: compare realised fills to the plan's assumptions.
        from etf.safety import compute_slippage, log_slippage
        fills = await broker.collect_fills()
        if fills:
            slip = compute_slippage(broker.last_orders, fills, cfg)
            log_slippage(slip, cfg.execution.slippage_log)
            level = logger.info if slip.within_tolerance else logger.warning
            level(
                "Slippage: avg %.2f bps (worst %.2f), cost $%.2f over $%.0f notional [%s budget]",
                slip.avg_slippage_bps, slip.worst_slippage_bps, slip.total_cost_usd,
                slip.total_notional, "within" if slip.within_tolerance else "OVER",
            )
        else:
            logger.info("Slippage telemetry: no realised fills to measure this cycle.")
        # Post-trade reconciliation: verify the live book matches intent.
        report = await broker.reconcile(target, cfg)
        if report is None:
            logger.warning("Reconciliation skipped (no account/price data).")
        elif report.ok:
            logger.info("Reconciliation OK: live book matches target within tolerance.")
            save_reconciliation_state(
                True, {}, cfg.execution.recon_state_path, as_of=report.as_of,
            )
        else:
            logger.warning(
                "Reconciliation MISMATCH (investigate before next cycle): %s",
                report.mismatches,
            )
            save_reconciliation_state(
                False, report.mismatches, cfg.execution.recon_state_path,
                as_of=report.as_of,
            )
    finally:
        await broker.disconnect()
    return 0


def cmd_paper(cfg: ETFConfig, args) -> int:
    return asyncio.run(_trade(cfg, args, live=False))


async def _run_loop(cfg: ETFConfig, args, live: bool) -> int:
    """Market-hours-aware scheduling loop (Phase 5 live runner).

    Wakes on a back-off schedule, and on each wake decides via the pure
    :func:`etf.runner.decide_action` whether today is a rebalance day inside the
    execution window with the cadence elapsed. When so, it runs exactly one
    fail-safe trade cycle (``_trade``) and advances the persisted cadence only
    if the cycle completed cleanly. ``--once`` performs a single check (ideal for
    a cron trigger) and exits; otherwise it loops until interrupted.
    """
    from etf.runner import (
        MarketCalendar,
        ScheduleState,
        decide_action,
        load_schedule_state,
        now_et,
        save_schedule_state,
    )

    cal = MarketCalendar()
    # --anytime widens the execution window to the whole session (still never
    # trades when the market is closed); otherwise use the last N minutes.
    window_minutes = 24 * 60 if args.anytime else args.window_minutes
    cadence_days = cfg.execution.rebalance_every

    while True:
        n = now_et()
        sched = load_schedule_state(cfg.execution.schedule_state_path)
        decision = decide_action(
            n, cal, sched,
            cadence_days=cadence_days,
            window_minutes=window_minutes,
            force=args.force,
        )
        mtc = decision.minutes_to_close
        logger.info(
            "Scheduler @ %s ET: trade=%s (session=%s, window=%s, cadence_elapsed=%s, "
            "min_to_close=%s) | %s",
            n.strftime("%Y-%m-%d %H:%M"), decision.should_trade, decision.is_trading_day,
            decision.in_execution_window, decision.cadence_elapsed,
            f"{mtc:.0f}" if mtc is not None else "n/a", "; ".join(decision.reasons),
        )

        if decision.should_trade:
            rc = await _trade(cfg, args, live)
            if rc == 0:
                save_schedule_state(
                    ScheduleState(last_rebalance_date=n.date().isoformat()),
                    cfg.execution.schedule_state_path,
                )
                logger.info("Rebalance cycle complete; cadence advanced to %s.", n.date())
            else:
                logger.warning(
                    "Trade cycle returned %d; cadence NOT advanced (will retry next wake).", rc
                )
            args.force = False  # a force is a one-shot; don't loop-trade

        if args.once:
            return 0

        logger.info("Sleeping %ds until next check.", decision.sleep_seconds)
        await asyncio.sleep(decision.sleep_seconds)


def cmd_run(cfg: ETFConfig, args) -> int:
    """Continuous (or single-shot via --once) market-hours-aware paper/live runner."""
    live = args.live
    if live and (not args.i_understand_the_risk or not args.execute):
        logger.error(
            "Live run requires both --execute and --i-understand-the-risk. "
            "Validate on paper for >= 20 trading days first (promotion gate)."
        )
        return 3
    try:
        return asyncio.run(_run_loop(cfg, args, live=live))
    except KeyboardInterrupt:
        logger.info("Runner interrupted by operator; shutting down cleanly.")
        return 0


def cmd_reset_safety(cfg: ETFConfig, args) -> int:
    """Clear persisted safety/scheduler state after a human has reviewed a halt
    or reconciliation mismatch (the documented runbook reset)."""
    import os as _os

    targets = {
        "reconciliation": cfg.execution.recon_state_path,
        "schedule": cfg.execution.schedule_state_path,
    }
    if args.reset_equity:
        targets["equity"] = cfg.execution.state_path
    for label, path in targets.items():
        try:
            if _os.path.exists(path):
                _os.remove(path)
                logger.info("Cleared %s state: %s", label, path)
            else:
                logger.info("No %s state to clear (%s).", label, path)
        except Exception as exc:
            logger.error("Failed to clear %s state %s: %s", label, path, exc)
            return 1
    logger.info("Safety state reset complete. Next cycle starts clean.")
    return 0


def cmd_live(cfg: ETFConfig, args) -> int:
    if not args.i_understand_the_risk or not args.execute:
        logger.error(
            "Live trading requires both --execute and --i-understand-the-risk. "
            "Validate on paper for >= 20 trading days first (promotion gate)."
        )
        return 3
    return asyncio.run(_trade(cfg, args, live=True))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ETF tactical-allocation engine")
    p.add_argument("--mode", choices=["backtest", "validate", "sleeves", "portfolio", "signal", "preflight", "paper", "live", "run", "reset-safety"], default="backtest")
    p.add_argument("--method", choices=["erc", "inverse_vol", "equal"], default=None, help="Portfolio combiner method (portfolio mode)")
    p.add_argument("--max-leverage", dest="max_leverage", type=float, default=None, help="Phase 4: raise the combined-book gross leverage cap (portfolio mode)")
    p.add_argument("--derisk", action="store_true", help="Phase 4: enable the combined-book drawdown circuit-breaker (portfolio mode)")
    p.add_argument("--candidate", choices=["seasonality", "cross_sectional"], default=None, help="Research: append a candidate sleeve to the roster for evaluation (sleeves mode)")
    p.add_argument("--vol-managed", dest="vol_managed", action="store_true", help="Apply the Moreira-Muir volatility-managed overlay to the equity-beta sleeves (portfolio mode)")
    p.add_argument("--start", default=None, help="Backtest start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="Backtest end date (YYYY-MM-DD)")
    p.add_argument("--refresh", action="store_true", help="Force re-download of price data")
    p.add_argument("--out", default=None, help="Write backtest results JSON to this path")
    p.add_argument("--execute", action="store_true", help="Actually submit orders (paper/live/run)")
    p.add_argument("--i-understand-the-risk", action="store_true", help="Required for live mode")
    # --- Live runner (mode=run) options ---------------------------------
    p.add_argument("--live", action="store_true", help="run mode: trade the live account (else paper). Still requires --execute and --i-understand-the-risk.")
    p.add_argument("--once", action="store_true", help="run mode: perform a single scheduling check (trade if due) then exit — ideal for a cron trigger.")
    p.add_argument("--window-minutes", dest="window_minutes", type=int, default=30, help="run mode: only trade within this many minutes before the close (default 30).")
    p.add_argument("--anytime", action="store_true", help="run mode: widen the execution window to the whole session (still never trades when the market is closed).")
    p.add_argument("--force", action="store_true", help="run mode: bypass the cadence gate for one immediate rebalance (market-open gate still applies).")
    p.add_argument("--reset-equity", dest="reset_equity", action="store_true", help="reset-safety mode: also clear the persisted equity high-water-mark state.")
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = get_default_config()
    dispatch = {
        "backtest": cmd_backtest,
        "validate": cmd_validate,
        "sleeves": cmd_sleeves,
        "portfolio": cmd_portfolio,
        "signal": cmd_signal,
        "preflight": cmd_preflight,
        "paper": cmd_paper,
        "live": cmd_live,
        "run": cmd_run,
        "reset-safety": cmd_reset_safety,
    }
    return dispatch[args.mode](cfg, args)


if __name__ == "__main__":
    sys.exit(main())

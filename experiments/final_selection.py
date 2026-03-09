"""
Final Selection: Test 3 finalist configs with deeper analysis.
Focus on drawdown recovery, consistency, and robustness.
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.disable(logging.WARNING)

import numpy as np
from vrp.config import get_config
from vrp.backtest import VRPBacktester

def analyze_config(name, config):
    bt = VRPBacktester(config)
    m = bt.run(start="2020-01-01", end="2025-12-31", verbose=False)
    
    # Year-by-year
    yearly = {}
    for t in bt.closed_trades:
        yr = t.close_date.year if t.close_date else t.entry_date.year
        if yr not in yearly:
            yearly[yr] = {"pnl": 0, "count": 0, "wins": 0}
        yearly[yr]["pnl"] += t.close_pnl
        yearly[yr]["count"] += 1
        if t.close_pnl > 0:
            yearly[yr]["wins"] += 1
    
    # Equity curve stats
    eq = np.array([e for _, e in bt.equity_curve])
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    
    # Underwater periods
    underwater = dd < 0
    if underwater.any():
        groups = np.split(underwater, np.where(np.diff(underwater.astype(int)))[0] + 1)
        durations = [len(g) for g in groups if g.any()]
        max_uw = max(durations) if durations else 0
        avg_uw = np.mean(durations) if durations else 0
    else:
        max_uw = 0
        avg_uw = 0
    
    # Monthly returns
    daily_eq = dict(bt.equity_curve)
    
    # Consecutive losers
    streak = 0
    max_losing_streak = 0
    for t in bt.closed_trades:
        if t.close_pnl <= 0:
            streak += 1
            max_losing_streak = max(max_losing_streak, streak)
        else:
            streak = 0
    
    # Capital deployed
    total_days = len(bt.equity_curve)
    days_with_positions = sum(1 for _, e in bt.equity_curve if len(bt.strategy.positions) > 0)
    
    # Expectancy
    expectancy = m.total_pnl / max(m.total_trades, 1)
    
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  Return: {m.total_return:+.1%} | Annual: {m.annual_return:+.1%} | Sharpe: {m.sharpe_ratio:.2f}")
    print(f"  MaxDD: {m.max_drawdown:.1%} | Calmar: {m.calmar_ratio:.2f} | Sortino: {m.sortino_ratio:.2f}")
    print(f"  Alpha: {m.alpha:+.1%} | Info Ratio: {m.information_ratio:.2f}")
    print(f"  Trades: {m.total_trades} | WR: {m.win_rate:.1%} | PF: {m.profit_factor:.2f}")
    print(f"  AvgWin: ${m.avg_win:+,.0f} | AvgLoss: ${m.avg_loss:+,.0f} | Payoff: {abs(m.avg_win/m.avg_loss) if m.avg_loss else 0:.2f}")
    print(f"  Expectancy: ${expectancy:+,.0f}/trade")
    print(f"  Max Losing Streak: {max_losing_streak}")
    print(f"  Max Underwater: {max_uw} days | Avg: {avg_uw:.0f} days")
    print(f"  Costs: comm=${m.total_commissions:,.0f} + slip=${m.total_slippage:,.0f} = ${m.total_commissions+m.total_slippage:,.0f}")
    print(f"  Year-by-Year:")
    all_positive = True
    for yr in sorted(yearly.keys()):
        y = yearly[yr]
        wr = y["wins"]/max(y["count"],1)*100
        marker = " ✓" if y["pnl"] > 0 else " ✗"
        if y["pnl"] <= 0:
            all_positive = False
        print(f"    {yr}: ${y['pnl']:>+8,.0f} ({y['count']:>3d} trades, {wr:>4.0f}% WR){marker}")
    print(f"  All years positive: {'YES' if all_positive else 'NO'}")
    
    return m, yearly


# ===================================================================
# FINALISTS
# ===================================================================

print("\n" + "="*70)
print("  FINAL SELECTION — 3 FINALISTS + BASELINE")
print("="*70)

# Baseline
baseline_cfg = get_config()
b_m, b_y = analyze_config("BASELINE (current production)", baseline_cfg)

# Finalist 1: Conservative upgrade (C from refinement)  
c1 = get_config()
c1.vix.min_vix = 16.0
c1.vix.standard_low = 20.0
c1.vix.low_vol_sizing_mult = 0.35
c1.spread.max_concurrent_positions = 3
c1.spread.max_total_risk_pct = 0.55
c1.spread.risk_per_trade = 0.22
c1.spread.profit_target_pct = 0.45
c1.spread.tight_profit_dte = 21
c1.spread.tight_profit_pct = 0.65
f1_m, f1_y = analyze_config("FINALIST 1: Conservative Alpha", c1)

# Finalist 2: Aggressive but controlled (G from refinement)
c2 = get_config()
c2.vix.min_vix = 15.0
c2.vix.standard_low = 20.0
c2.vix.low_vol_sizing_mult = 0.40
c2.spread.max_concurrent_positions = 3
c2.spread.max_total_risk_pct = 0.55
c2.spread.risk_per_trade = 0.22
c2.spread.profit_target_pct = 0.40
c2.spread.tight_profit_dte = 21
c2.spread.tight_profit_pct = 0.60
f2_m, f2_y = analyze_config("FINALIST 2: Aggressive Alpha", c2)

# Finalist 3: Maximum return with acceptable drawdown (A/B from refinement)
c3 = get_config()
c3.vix.min_vix = 15.0
c3.vix.standard_low = 20.0
c3.vix.low_vol_sizing_mult = 0.40
c3.spread.max_concurrent_positions = 3
c3.spread.max_total_risk_pct = 0.55
c3.spread.risk_per_trade = 0.20
c3.spread.profit_target_pct = 0.45
c3.spread.tight_profit_dte = 21
c3.spread.tight_profit_pct = 0.65
c3.spread.target_dte = 35
c3.spread.max_dte = 49
f3_m, f3_y = analyze_config("FINALIST 3: Maximum Alpha", c3)

# Final comparison
print(f"\n\n{'='*100}")
print(f"  FINAL HEAD-TO-HEAD")
print(f"{'='*100}")
print(f"{'Metric':<25} {'Baseline':>14} {'Conservative':>14} {'Aggressive':>14} {'Maximum':>14}")
print("-"*100)
rows = [
    ("Total Return", f"{b_m.total_return:+.1%}", f"{f1_m.total_return:+.1%}", f"{f2_m.total_return:+.1%}", f"{f3_m.total_return:+.1%}"),
    ("Annual Return", f"{b_m.annual_return:+.1%}", f"{f1_m.annual_return:+.1%}", f"{f2_m.annual_return:+.1%}", f"{f3_m.annual_return:+.1%}"),
    ("Sharpe", f"{b_m.sharpe_ratio:.2f}", f"{f1_m.sharpe_ratio:.2f}", f"{f2_m.sharpe_ratio:.2f}", f"{f3_m.sharpe_ratio:.2f}"),
    ("Sortino", f"{b_m.sortino_ratio:.2f}", f"{f1_m.sortino_ratio:.2f}", f"{f2_m.sortino_ratio:.2f}", f"{f3_m.sortino_ratio:.2f}"),
    ("Max Drawdown", f"{b_m.max_drawdown:.1%}", f"{f1_m.max_drawdown:.1%}", f"{f2_m.max_drawdown:.1%}", f"{f3_m.max_drawdown:.1%}"),
    ("Calmar", f"{b_m.calmar_ratio:.2f}", f"{f1_m.calmar_ratio:.2f}", f"{f2_m.calmar_ratio:.2f}", f"{f3_m.calmar_ratio:.2f}"),
    ("Alpha", f"{b_m.alpha:+.1%}", f"{f1_m.alpha:+.1%}", f"{f2_m.alpha:+.1%}", f"{f3_m.alpha:+.1%}"),
    ("Win Rate", f"{b_m.win_rate:.1%}", f"{f1_m.win_rate:.1%}", f"{f2_m.win_rate:.1%}", f"{f3_m.win_rate:.1%}"),
    ("Profit Factor", f"{b_m.profit_factor:.2f}", f"{f1_m.profit_factor:.2f}", f"{f2_m.profit_factor:.2f}", f"{f3_m.profit_factor:.2f}"),
    ("Total Trades", f"{b_m.total_trades}", f"{f1_m.total_trades}", f"{f2_m.total_trades}", f"{f3_m.total_trades}"),
    ("Total P&L", f"${b_m.total_pnl:+,.0f}", f"${f1_m.total_pnl:+,.0f}", f"${f2_m.total_pnl:+,.0f}", f"${f3_m.total_pnl:+,.0f}"),
    ("SPX Return", f"{b_m.spx_return:+.1%}", f"{f1_m.spx_return:+.1%}", f"{f2_m.spx_return:+.1%}", f"{f3_m.spx_return:+.1%}"),
]
for label, v1, v2, v3, v4 in rows:
    print(f"{label:<25} {v1:>14} {v2:>14} {v3:>14} {v4:>14}")
print("-"*100)

# Delta from baseline
print(f"\n  IMPROVEMENT OVER BASELINE:")
for name, fm in [("Conservative", f1_m), ("Aggressive", f2_m), ("Maximum", f3_m)]:
    ret_delta = fm.total_return - b_m.total_return
    sharpe_delta = fm.sharpe_ratio - b_m.sharpe_ratio
    pnl_delta = fm.total_pnl - b_m.total_pnl
    alpha_delta = fm.alpha - b_m.alpha
    dd_delta = fm.max_drawdown - b_m.max_drawdown
    print(f"  {name:>15}: Return {ret_delta:+.1%} | Sharpe {sharpe_delta:+.2f} | P&L ${pnl_delta:+,.0f} | Alpha {alpha_delta:+.1%} | DD {dd_delta:+.1%}")

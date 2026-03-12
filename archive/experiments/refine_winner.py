"""
Refine the winning configuration.

Top 3 candidates from round 1:
- #8 Full Alpha: +231.7%, Sharpe 0.76, MaxDD -25.2%
- #9 Balanced:   +185.5%, Sharpe 0.65, MaxDD -20.6%  
- #4 3 Positions:+166.3%, Sharpe 0.62, MaxDD -20.4%

Issues with #8: MaxDD -25.2% is higher than baseline -19.8%
Issues with #9: Calmar 0.93 is best but return lags #8 by 46pts

Goal: Find a config that pushes returns as high as possible
while keeping drawdown under -22%.

Also run year-by-year analysis to check for consistency.
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.disable(logging.WARNING)

from vrp.config import get_config
from vrp.backtest import VRPBacktester

def run_and_report(name, config, start="2020-01-01", end="2025-12-31"):
    bt = VRPBacktester(config)
    m = bt.run(start=start, end=end, verbose=False)
    
    # Year-by-year breakdown
    trades_by_year = {}
    for t in bt.closed_trades:
        yr = t.close_date.year if t.close_date else t.entry_date.year
        if yr not in trades_by_year:
            trades_by_year[yr] = {"pnl": 0, "count": 0, "wins": 0}
        trades_by_year[yr]["pnl"] += t.close_pnl
        trades_by_year[yr]["count"] += 1
        if t.close_pnl > 0:
            trades_by_year[yr]["wins"] += 1
    
    return m, trades_by_year, bt


def print_result(name, m, yearly):
    print(f"\n  {name}")
    print(f"  {'='*60}")
    print(f"  Return: {m.total_return:+.1%} | Annual: {m.annual_return:+.1%} | Sharpe: {m.sharpe_ratio:.2f}")
    print(f"  MaxDD: {m.max_drawdown:.1%} | Calmar: {m.calmar_ratio:.2f} | Sortino: {m.sortino_ratio:.2f}")
    print(f"  Trades: {m.total_trades} | WR: {m.win_rate:.1%} | PF: {m.profit_factor:.2f}")
    print(f"  PnL: ${m.total_pnl:+,.0f} | Alpha: {m.alpha:+.1%}")
    print(f"  AvgWin: ${m.avg_win:+,.0f} | AvgLoss: ${m.avg_loss:+,.0f}")
    print(f"  Year-by-Year:")
    for yr in sorted(yearly.keys()):
        y = yearly[yr]
        wr = y["wins"]/max(y["count"],1)*100
        print(f"    {yr}: ${y['pnl']:>+8,.0f} ({y['count']} trades, {wr:.0f}% WR)")


# ===================================================================
# REFINED EXPERIMENTS
# ===================================================================

configs = {}

# A) Full alpha with slightly tighter risk
c = get_config()
c.vix.min_vix = 15.0
c.vix.standard_low = 20.0
c.vix.low_vol_sizing_mult = 0.40
c.spread.max_concurrent_positions = 3
c.spread.max_total_risk_pct = 0.55  # tighter than 0.60
c.spread.risk_per_trade = 0.20
c.spread.profit_target_pct = 0.45
c.spread.tight_profit_dte = 21
c.spread.tight_profit_pct = 0.65
c.spread.entry_days = [4]  # Friday
c.spread.target_dte = 35
c.spread.max_dte = 49
configs["A: Full alpha (tighter risk)"] = c

# B) Same but no Friday restriction (any day)
c = get_config()
c.vix.min_vix = 15.0
c.vix.standard_low = 20.0
c.vix.low_vol_sizing_mult = 0.40
c.spread.max_concurrent_positions = 3
c.spread.max_total_risk_pct = 0.55
c.spread.risk_per_trade = 0.20
c.spread.profit_target_pct = 0.45
c.spread.tight_profit_dte = 21
c.spread.tight_profit_pct = 0.65
c.spread.target_dte = 35
c.spread.max_dte = 49
configs["B: Full alpha (any day)"] = c

# C) VIX 16 floor (more conservative) + 3 pos + dynamic exits
c = get_config()
c.vix.min_vix = 16.0
c.vix.standard_low = 20.0
c.vix.low_vol_sizing_mult = 0.35
c.spread.max_concurrent_positions = 3
c.spread.max_total_risk_pct = 0.55
c.spread.risk_per_trade = 0.22
c.spread.profit_target_pct = 0.45
c.spread.tight_profit_dte = 21
c.spread.tight_profit_pct = 0.65
configs["C: VIX 16 + 3pos + exits"] = c

# D) Like B but with stop loss at 1.75x (tighter)
c = get_config()
c.vix.min_vix = 15.0
c.vix.standard_low = 20.0
c.vix.low_vol_sizing_mult = 0.40
c.spread.max_concurrent_positions = 3
c.spread.max_total_risk_pct = 0.55
c.spread.risk_per_trade = 0.20
c.spread.profit_target_pct = 0.45
c.spread.tight_profit_dte = 21
c.spread.tight_profit_pct = 0.65
c.spread.stop_loss_multiple = 1.75
c.spread.target_dte = 35
c.spread.max_dte = 49
configs["D: B + tighter stops (1.75x)"] = c

# E) Like B but with 42 DTE (original) instead of 35
c = get_config()
c.vix.min_vix = 15.0
c.vix.standard_low = 20.0
c.vix.low_vol_sizing_mult = 0.40
c.spread.max_concurrent_positions = 3
c.spread.max_total_risk_pct = 0.55
c.spread.risk_per_trade = 0.20
c.spread.profit_target_pct = 0.45
c.spread.tight_profit_dte = 21
c.spread.tight_profit_pct = 0.65
configs["E: B + original 42 DTE"] = c

# F) Like E but with max DD halt at -12% (tighter risk mgmt) 
c = get_config()
c.vix.min_vix = 15.0
c.vix.standard_low = 20.0
c.vix.low_vol_sizing_mult = 0.40
c.spread.max_concurrent_positions = 3
c.spread.max_total_risk_pct = 0.55
c.spread.risk_per_trade = 0.20
c.spread.profit_target_pct = 0.45
c.spread.tight_profit_dte = 21
c.spread.tight_profit_pct = 0.65
c.risk.max_drawdown_halt = -0.12
c.risk.max_drawdown_reduce = -0.08
configs["F: E + tighter DD halt (-12%)"] = c

# G) Low VIX floor 15, 3 positions, faster profit target (40%), keep 42 DTE
c = get_config()
c.vix.min_vix = 15.0
c.vix.standard_low = 20.0
c.vix.low_vol_sizing_mult = 0.40
c.spread.max_concurrent_positions = 3
c.spread.max_total_risk_pct = 0.55
c.spread.risk_per_trade = 0.22
c.spread.profit_target_pct = 0.40
c.spread.tight_profit_dte = 21
c.spread.tight_profit_pct = 0.60
configs["G: VIX15 + 3pos + fast exits"] = c


print(f"\n{'='*60}")
print(f"  REFINEMENT ROUND: {len(configs)} variants")
print(f"{'='*60}")

all_results = []
for name, cfg in configs.items():
    print(f"\n  Running: {name}...", end=" ", flush=True)
    t0 = time.time()
    m, yearly, bt = run_and_report(name, cfg)
    print(f"done ({time.time()-t0:.1f}s)")
    print_result(name, m, yearly)
    all_results.append((name, m, yearly))

# Summary comparison
print(f"\n\n{'='*120}")
print(f"  REFINEMENT COMPARISON")
print(f"{'='*120}")
print(f"{'Name':<35} {'Return':>8} {'Annual':>8} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} {'Trades':>7} {'PnL':>10} {'Alpha':>8} {'Sortino':>8}")
print("-" * 120)
for name, m, _ in all_results:
    print(
        f"{name:<35} "
        f"{m.total_return:>+7.1%} "
        f"{m.annual_return:>+7.1%} "
        f"{m.sharpe_ratio:>7.2f} "
        f"{m.max_drawdown:>+7.1%} "
        f"{m.calmar_ratio:>7.2f} "
        f"{m.total_trades:>7d} "
        f"${m.total_pnl:>+9,.0f} "
        f"{m.alpha:>+7.1%} "
        f"{m.sortino_ratio:>8.2f}"
    )
print("-" * 120)

# Recommend: highest Sharpe with MaxDD > -23%
candidates = [(n,m,y) for n,m,y in all_results if m.max_drawdown > -0.23]
if candidates:
    best = max(candidates, key=lambda x: x[1].sharpe_ratio)
    print(f"\n  RECOMMENDED (Sharpe-optimal, MaxDD > -23%): {best[0]}")
    print(f"    Return: {best[1].total_return:+.1%} | Sharpe: {best[1].sharpe_ratio:.2f} | MaxDD: {best[1].max_drawdown:.1%} | Alpha: {best[1].alpha:+.1%}")
    
# Also find overall best return with MaxDD > -25%
candidates2 = [(n,m,y) for n,m,y in all_results if m.max_drawdown > -0.25]
if candidates2:
    best2 = max(candidates2, key=lambda x: x[1].total_return)
    print(f"  BEST RETURN (MaxDD > -25%): {best2[0]}")
    print(f"    Return: {best2[1].total_return:+.1%} | Sharpe: {best2[1].sharpe_ratio:.2f} | MaxDD: {best2[1].max_drawdown:.1%} | Alpha: {best2[1].alpha:+.1%}")

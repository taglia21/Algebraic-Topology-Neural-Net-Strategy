# Production Operations Guide

This guide summarizes the complete path from backtest research to live trading with quantitative safety gates at each phase.

## Overview

```
RESEARCH
  ↓
  backtest/metrics.py → compute OOS Sharpe, drawdown, Calmar, profit_factor
  ↓
  [Gate 1: research→paper]
  ├─ OOS Sharpe ≥ 1.10 ✓
  ├─ Max drawdown ≤ 18% ✓
  ├─ Calmar ≥ 0.80 ✓
  ├─ Profit factor ≥ 1.20 ✓
  ├─ No leakage ✓
  └─ All tests pass ✓
  ↓
PAPER TRADING (20+ days)
  ├─ Run in Alpaca paper mode
  ├─ Collect: trading days, slippage, rejections, reconciliation, halts
  ├─ Document: runbook + rollback steps
  │
  [Gate 2: paper→live]
  ├─ Paper days ≥ 20 ✓
  ├─ Slippage deviation ≤ 20% ✓
  ├─ Order rejection rate ≤ 1% ✓
  ├─ Reconciliation mismatches = 0 ✓
  ├─ Software defect halts = 0 ✓
  └─ Runbook documented ✓
  ↓
LIVE TRADING (small cap, 2 review cycles)
  ├─ Initial sizing ≤ 25% of target
  ├─ Run 2 × 5-10 day review cycles
  ├─ Monitor: stability metrics drift, incidents
  │
  [Gate 3: live scale-up]
  ├─ Sizing cap ≤ 25% ✓
  ├─ Review cycles ≥ 2 ✓
  ├─ Metrics drift ≤ 10% ✓
  └─ Sev-1 incidents = 0 ✓
  ↓
PRODUCTION (full target capital)
```

---

## Phase 1: Research → Paper

### Inputs
- Backtest results from `run_backtest.py` or `main.py --mode backtest`
- ML model validation from `ml/validation.py`

### Gate Criteria (in `scripts/check_promotion_gates.py`)

| Criterion | Threshold | Why |
|-----------|-----------|-----|
| OOS Sharpe | ≥ 1.10 | Risk-adjusted return stability |
| Max Drawdown | ≤ 18% | Psychological and capital preservation |
| Calmar Ratio | ≥ 0.80 | Return-to-risk efficiency |
| Profit Factor | ≥ 1.20 | Win rate quality (must beat 1.0) |
| Leakage Check | None found | No look-ahead or data snooping |
| Test Coverage | All pass | Code correctness |

### How to Validate

```bash
# Create evidence from backtest output
python -c "
import json
result = json.load(open('artifacts/backtest_result.json'))
evidence = {
    'oos_sharpe': result['metrics']['sharpe'],
    'max_drawdown': result['metrics']['max_drawdown'],
    'calmar': result['metrics']['calmar'],
    'profit_factor': result['metrics']['profit_factor'],
    'leakage_findings': False,  # manual check
    'all_touched_tests_pass': True,  # from pytest run
}
json.dump(evidence, open('templates/promotion_gate_evidence.research_to_paper.json', 'w'), indent=2)
"

# Validate gate
python scripts/check_promotion_gates.py \
  --gate research_to_paper \
  --input templates/promotion_gate_evidence.research_to_paper.json \
  --report-out artifacts/research_to_paper_gate_report.json

# Exit code 0 = pass, 2 = fail
echo $?
```

### If Gate Fails
- **Too low Sharpe?** → Reduce universe, improve signal, add regime filter
- **Too high drawdown?** → Tighten stop-losses, add correlation caps, reduce leverage
- **Low Calmar?** → Improve return without increasing drawdown; adjust position sizing
- **Profit factor < 1.20?** → Improve signal selectivity or reduce trade frequency
- **Leakage found?** → Audit feature engineering, validate CPCV logic
- **Tests fail?** → Fix code bugs before promotion

---

## Phase 2: Paper Trading

### Setup

1. **Configure credentials** in `.env`:
   ```env
   ALPACA_API_KEY=...
   ALPACA_API_SECRET=...
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   SYSTEM_MODE=paper
   ```

2. **Enable telemetry collection** in code:
   - Instrument execution manager to log daily trading
   - Track actual fill prices vs. modeled slippage
   - Count order rejections and reconciliation mismatches
   - Log circuit breaker halts with reason codes
   - See: [docs/TELEMETRY_COLLECTION.md](TELEMETRY_COLLECTION.md)

3. **Create runbook** (`operations/PAPER_TRADING_RUNBOOK.md`):
   ```markdown
   ## Paper Trading Runbook
   
   ### Start
   ```bash
   python main.py --mode paper
   ```
   
   ### Daily Checks
   - PnL > -2% (daily loss limit)
   - Exposures within limits
   - No reconciliation warnings
   
   ### Emergency Halt
   - Kill terminal with Ctrl+C
   - Or if kill switch triggered, system auto-halts
   
   ### Rollback
   - Switch `SYSTEM_MODE=backtest` in .env
   - Run backtest to verify consistency
   - Switch back when ready
   
   ### End Paper Phase
   - After 20+ trading days: collect telemetry
   - Run: python scripts/collect_paper_telemetry.py
   - Create PR with evidence.json
   ```

4. **Run paper trading:**
   ```bash
   python main.py --mode paper 2>&1 | tee operations/paper_trading.log
   # Keep running for 20+ trading days (4-5 calendar weeks)
   ```

### Metrics to Collect

See [docs/TELEMETRY_COLLECTION.md](TELEMETRY_COLLECTION.md) for detailed collection points.

- **Trading days:** Count unique trading dates logged
- **Slippage:** Compare filled prices to submission prices, compute mean in bps
- **Order rejection rate:** Rejected / Total submitted orders
- **Reconciliation mismatches:** Count unresolved position discrepancies per cycle
- **Software defect halts:** Count halts with "error:" or "exception:" in reason
- **Runbook documented:** Boolean; steps tested and verified

### Gate Criteria

| Criterion | Threshold | Why |
|-----------|-----------|-----|
| Paper days | ≥ 20 | Sufficient data for statistics |
| Slippage deviation | ≤ 20% relative | Reality vs. backtest model calibration |
| Order rejection rate | ≤ 1.0% | Broker/market conditions acceptable |
| Reconciliation mismatches | = 0 unresolved | System state truth matching broker truth |
| Software defect halts | = 0 | No code bugs causing emergency halts |
| Runbook documented | true | Operator knows how to run/stop/recover |

### How to Validate

```bash
# After 20+ days of paper trading, collect metrics
cp templates/promotion_gate_evidence.paper_to_live.example.json \
   templates/promotion_gate_evidence.paper_to_live.json

# Edit with your actual data (see TELEMETRY_COLLECTION.md)
# Then validate:
python scripts/check_promotion_gates.py \
  --gate paper_to_live \
  --input templates/promotion_gate_evidence.paper_to_live.json \
  --report-out artifacts/paper_to_live_gate_report.json

# If passes: create PR
git checkout -b feat/paper-to-live-promotion
git add templates/promotion_gate_evidence.paper_to_live.json
git commit -m "promotion: paper->live gate evidence"
git push -u origin feat/paper-to-live-promotion
gh pr create --base main --title "Promotion: paper→live gate evidence"

# CI auto-validates and blocks merge if gate fails
# Merge only after gate passes + review approval
```

### If Gate Fails
- **Not enough paper days?** → Continue paper trading until ≥ 20 days
- **Slippage too high?** → Check market conditions, order sizing, time-of-day patterns
- **High rejection rate?** → Investigate broker limits, liquidity, order types
- **Reconciliation mismatches?** → Debug broker API integration, timestamp handling
- **Software defect halts?** → Fix bugs, run more paper days
- **Runbook incomplete?** → Document and test all operational procedures

---

## Phase 3: Live Trading (Scale-Up)

### Setup

1. **Create live credentials**:
   ```env
   ALPACA_API_KEY=...  # LIVE key
   ALPACA_API_SECRET=...  # LIVE secret
   ALPACA_BASE_URL=https://api.alpaca.markets  # LIVE, not paper
   SYSTEM_MODE=live
   INITIAL_CAPITAL=25000  # ≤ 25% of target
   TARGET_CAPITAL=100000  # Full target
   ```

2. **Create runbook** (`operations/LIVE_TRADING_RUNBOOK.md`):
   ```markdown
   ## Live Trading Runbook
   
   ### Start (Scale Phase 1: Days 1-10)
   ```bash
   python main.py --mode live
   ```
   
   ### Monitor
   - Open: `tail -f operations/live_trading.log`
   - Dashboard: check PnL, exposures, orders
   - Daily review: pass/fail checklist
   
   ### Emergency Halt
   - Manual: Ctrl+C or close connections
   - Auto: Kill switch triggers at risk thresholds
   
   ### Scale Up to Phase 2 (Day 11+)
   - Review Phase 1 metrics, incidents, feedback
   - If all green: increase INITIAL_CAPITAL to 50%
   - Restart system with new config
   
   ### Full Scale (Day 21+)
   - After 2 successful phases: increase to TARGET_CAPITAL
   ```

3. **Run live with small cap:**
   ```bash
   # Phase 1: days 1-10 at 25% of target
   INITIAL_CAPITAL=25000 python main.py --mode live 2>&1 | tee operations/phase1.log
   
   # Phase 2: days 11-20 at 50% of target
   INITIAL_CAPITAL=50000 python main.py --mode live 2>&1 | tee operations/phase2.log
   
   # After review: full scale (if gates pass)
   INITIAL_CAPITAL=100000 python main.py --mode live
   ```

### Metrics to Collect

- **Initial sizing:** Percentage of target capital deployed in Phase 1 (should be ≤ 25%)
- **Review cycles:** Count of 5-10 day cycles with no Sev-1 incidents and passing review
- **Stability metrics:** Compare live Sharpe/Sortino/drawdown/Calmar to paper baselines; compute drift %
- **Sev-1 incidents:** Count critical incidents (system crash, large unintended loss, etc.)

### Gate Criteria

| Criterion | Threshold | Why |
|-----------|-----------|-----|
| Initial sizing | ≤ 25% | Limit exposure during learning phase |
| Review cycles | ≥ 2 | Multiple windows of stable operation |
| Metrics drift | ≤ 10% | Live performance matches paper expectations |
| Sev-1 incidents | = 0 | No critical system failures |

### How to Validate

```bash
# After 2 successful review cycles (20+ days), collect metrics
cp templates/promotion_gate_evidence.live_scale_up.example.json \
   templates/promotion_gate_evidence.live_scale_up.json

# Edit with your actual data
# Then validate:
python scripts/check_promotion_gates.py \
  --gate live_scale_up \
  --input templates/promotion_gate_evidence.live_scale_up.json \
  --report-out artifacts/live_scale_up_gate_report.json

# If passes: create PR for full-scale promotion
git checkout -b feat/live-scale-up-promotion
git add templates/promotion_gate_evidence.live_scale_up.json
git commit -m "promotion: live scale-up gate evidence"
git push -u origin feat/live-scale-up-promotion
gh pr create --base main --title "Promotion: live scale-up gate evidence"

# After gate passes + review: merge and proceed to full capital
```

### If Gate Fails
- **Sizing cap exceeded?** → Reduce initial capital to meet threshold
- **Not enough successful cycles?** → Continue live trading, repeat review cycles
- **Metrics drift > 10%?** → Investigate discrepancies between paper/live:
  - Order fill quality
  - Intraday volatility differences
  - Market regime shift
  - Execution timing
- **Sev-1 incidents?** → Resolve root causes, add monitoring/alerts

---

## Automation & Safety

### GitHub Actions CI
- Automatically validates gates when evidence PR is created
- Blocks merge if gate criteria not met
- Provides audit trail in workflow logs
- See: [docs/CI_AND_GOVERNANCE_SETUP.md](CI_AND_GOVERNANCE_SETUP.md)

### Local Pre-Commit Validation
- Optional: set up Git hooks to validate locally before commit
- See: [docs/PRE_COMMIT_HOOKS.md](PRE_COMMIT_HOOKS.md)

### Branch Protection
- Enforces PR workflow on main branch
- Requires 1+ approving review
- Blocks direct pushes
- Run: `bash scripts/enable_branch_protection.sh` (admin only)

---

## Troubleshooting

### "Gate validation failed, but my evidence looks correct"
1. Run validation locally: `python scripts/check_promotion_gates.py --gate <type> --input <file>`
2. Review gate thresholds in `scripts/check_promotion_gates.py`
3. Check evidence JSON field names and data types match expected schema
4. Adjust evidence or adjust gates (if justified)

### "Paper trading results worse than backtest"
- Common: slippage, rejections, market regime differences
- Check: execution fill quality, order timing, market hours
- Fix: tighten slippage model, adjust position sizing, add regime filter

### "Live trading metrics drifting from paper"
- Investigate: broker latency, intraday volatility, data provider differences
- Check: execution timing, reconciliation mismatches, kill switch triggers
- Measure: collect detailed metrics and compare paper→live at signal level

### "Incident during live trading"
1. Gather info: timestamp, PnL impact, orders/fills involved
2. Halt system: Ctrl+C or wait for auto-halt
3. Document: create `operations/INCIDENT_[DATE].md` with timeline and root cause
4. Fix: patch code, add tests, validate in backtest/paper before resuming
5. Restore: restart in paper mode until confidence restored

---

## Key Documents

| Document | Purpose |
|----------|---------|
| [CI_AND_GOVERNANCE_SETUP.md](CI_AND_GOVERNANCE_SETUP.md) | GitHub Actions workflow, branch protection |
| [PR_WORKFLOW.md](PR_WORKFLOW.md) | Feature branch → PR → review → merge |
| [TELEMETRY_COLLECTION.md](TELEMETRY_COLLECTION.md) | Where/how to instrument code for metrics |
| [PRE_COMMIT_HOOKS.md](PRE_COMMIT_HOOKS.md) | Local validation before commits |
| [../scripts/check_promotion_gates.py](../scripts/check_promotion_gates.py) | Gate thresholds and validation logic |
| [../templates/promotion_gate_evidence*.json](../templates/) | Evidence input schemas |

---

## Summary

**The system is now production-ready with quantitative safety gates.**

- ✅ Research → Paper gate: automated, CI-enforced
- ✅ Paper → Live gate: automated, CI-enforced
- ✅ Live Scale-Up gate: automated, CI-enforced
- ✅ All gates: explicit thresholds, deterministic validation, audit logs
- ✅ PR workflow: enforced via branch protection
- ✅ Telemetry: documented collection points
- ✅ Operations: runbooks and incident procedures

**Next step:** Merge this infrastructure, then proceed to paper trading (Phase 2).

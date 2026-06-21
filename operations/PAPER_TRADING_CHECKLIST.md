# Paper Trading Startup Checklist

This checklist guides you through the setup and execution of a 20+ day paper trading phase before promoting to live trading.

## Pre-Phase: Environment Setup (1 day)

### 1. Prepare Credentials
- [ ] Create Alpaca paper trading account at https://alpaca.markets
- [ ] Generate API key and secret from https://app.alpaca.markets/account/info
- [ ] Copy `.env.example` to `.env`
  ```bash
  cp .env.example .env
  ```
- [ ] Edit `.env` with:
  ```env
  ALPACA_API_KEY=<your_paper_key>
  ALPACA_API_SECRET=<your_paper_secret>
  ALPACA_BASE_URL=https://paper-api.alpaca.markets
  SYSTEM_MODE=paper
  INITIAL_CAPITAL=100000
  TARGET_CAPITAL=100000
  ```
- [ ] Test credentials:
  ```bash
  python -c "from equities.alpaca_broker import AlpacaBroker; b = AlpacaBroker(); print(f'Connected: {b.get_account().equity}')"
  ```

### 2. Prepare Runbook Document
- [ ] Create `operations/PAPER_TRADING_RUNBOOK.md` (see template below)
- [ ] Document daily checks, emergency procedures, and rollback steps

### 3. Prepare Telemetry Collection
- [ ] Review [docs/TELEMETRY_COLLECTION.md](../docs/TELEMETRY_COLLECTION.md)
- [ ] Confirm telemetry module is working:
  ```bash
  python -c "from equities.telemetry import get_telemetry; t = get_telemetry(); print(f'Telemetry ready')"
  ```

### 4. Verify Code Quality
- [ ] Run tests to ensure no regressions:
  ```bash
  python -m pytest tests/ -v --tb=short
  ```
- [ ] Run a quick backtest to verify logic:
  ```bash
  python run_backtest.py --small
  ```

---

## Phase: Paper Trading (20+ trading days)

### Daily Checks (Each Trading Day at Market Open)

**Before 9:30 AM ET:**
- [ ] Market open: Ensure system is running and connected
  ```bash
  python main.py --mode paper 2>&1 | tee -a operations/paper_trading_$(date +%Y%m%d).log
  ```
- [ ] Check Alpaca paper account dashboard for connection
- [ ] Verify initial account equity is as configured

**During Market Hours (9:30 AM - 4:00 PM ET):**
- [ ] Monitor console for errors or unexpected warnings
- [ ] Check system doesn't halt unexpectedly
- [ ] Spot-check a few orders filling correctly
- [ ] Ensure reconciliation matches broker positions

**After 4:00 PM ET (Market Close):**
- [ ] Stop system gracefully (Ctrl+C)
- [ ] Review log file for errors:
  ```bash
  tail -30 operations/paper_trading_$(date +%Y%m%d).log | grep -i "error\|warning\|halt"
  ```
- [ ] Record key metrics:
  - Daily PnL
  - Exposure (gross/net)
  - Largest positions
  - Any rejections or reconciliation warnings

### Weekly Review (Every Friday or End of Week)

- [ ] Aggregate metrics from all daily runs
- [ ] Check for patterns (e.g., consistent rejection source, slippage drift)
- [ ] Verify kill switch conditions aren't being triggered inappropriately
- [ ] Review halt logs for any software-related halts (vs. legitimate risk)

### End of Phase: Metrics Collection (After 20+ Trading Days)

- [ ] Calculate final metrics:
  ```python
  from equities.telemetry import get_telemetry
  tel = get_telemetry()
  evidence = tel.export_promotion_gate_evidence(
      path="templates/promotion_gate_evidence.paper_to_live.json",
      modeled_slippage_bps=7.0,
      runbook_documented=True
  )
  ```
- [ ] Verify metrics meet gate thresholds:
  ```bash
  python scripts/check_promotion_gates.py \
    --gate paper_to_live \
    --input templates/promotion_gate_evidence.paper_to_live.json \
    --report-out artifacts/paper_to_live_gate_report.json
  ```
- [ ] Review gate report:
  ```bash
  cat artifacts/paper_to_live_gate_report.json | python -m json.tool
  ```

---

## Emergency Procedures

### System Halts or Crashes

1. **Check logs:**
   ```bash
   tail -100 operations/paper_trading_$(date +%Y%m%d).log | grep -i "error\|exception\|halt"
   ```

2. **Identify root cause:**
   - Software defect? Check stack trace and file
   - Broker API error? Check Alpaca status
   - Data provider issue? Check yfinance/market data source
   - Risk condition? Check kill switch logs

3. **Recover:**
   - Fix issue (if code-related)
   - Wait for next market open
   - Restart system

4. **Document:**
   - Create `operations/INCIDENT_[DATE].md` with timeline and resolution

### Positions Out of Sync (Reconciliation Warning)

1. **Check reconciliation report:**
   ```bash
   # System logs reconciliation warnings automatically
   grep "Discrepancy" operations/paper_trading_*.log
   ```

2. **Investigate:**
   - Log into Alpaca web UI and compare positions
   - Check order fill times and quantities
   - Verify no stuck/cancelled orders

3. **Resolve:**
   - System will auto-correct most mismatches in next cycle
   - If persistent, manually verify in Alpaca UI
   - Record the mismatch for gate evidence (should be zero)

### Order Rejections Spiking

1. **Check reason:**
   ```bash
   grep "order_rejected\|rejection" operations/paper_trading_*.log | head -20
   ```

2. **Common causes:**
   - Alpaca position limits reached
   - Insufficient buying power
   - Symbol delisted or halted
   - Order type not supported

3. **Adjust:**
   - Reduce position cap or order size
   - Check market hours and regular trading hours (RTH)
   - Verify symbols are still tradeable

---

## Post-Phase: Validation & Promotion

### Validate Gate

```bash
python scripts/check_promotion_gates.py \
  --gate paper_to_live \
  --input templates/promotion_gate_evidence.paper_to_live.json
```

### If Gate Passes

1. Create feature branch:
   ```bash
   git checkout -b feat/paper-to-live-promotion
   ```

2. Commit evidence:
   ```bash
   git add templates/promotion_gate_evidence.paper_to_live.json
   git commit -m "promotion: paper->live gate evidence"
   git push -u origin feat/paper-to-live-promotion
   ```

3. Open PR:
   ```bash
   gh pr create --base main --title "Promotion: paper→live gate evidence" \
     --body "20+ days of paper trading. Gate validation report: $(cat artifacts/paper_to_live_gate_report.json)"
   ```

4. CI automatically validates; merge when gate passes + approval given

### If Gate Fails

- Review which thresholds weren't met
- Continue paper trading if the issue is resolvable
- Adjust configuration if needed
- Document findings in PR description before retrying

---

## Appendix: Runbook Template

Create `operations/PAPER_TRADING_RUNBOOK.md`:

```markdown
# Paper Trading Runbook

## System Start

Start paper trading in detached screen session:

\`\`\`bash
screen -S paper_trading -dm bash -c 'cd /workspaces/Algebraic-Topology-Neural-Net-Strategy && python main.py --mode paper 2>&1 | tee -a operations/paper_trading_$(date +%Y%m%d).log'
\`\`\`

Monitor in real-time:
\`\`\`bash
screen -r paper_trading
# Press Ctrl+D to detach without killing
\`\`\`

## System Stop

Kill the session:
\`\`\`bash
screen -S paper_trading -X kill
\`\`\`

Or from within session: Ctrl+C

## Daily Checklist

- Market open: system started and running
- Market close: review log for errors
- Weekly: aggregate metrics and review trends

## Emergency Halt

If system must stop immediately:
\`\`\`bash
pkill -9 -f "python main.py --mode paper"
\`\`\`

## Rollback to Backtest

If issues arise, verify logic in backtest before resuming paper:
\`\`\`bash
python run_backtest.py --small
\`\`\`

## Restart After Fix

1. Apply code fix
2. Run tests: `python -m pytest tests/ -v`
3. Run quick backtest: `python run_backtest.py --small`
4. Resume paper trading
```

---

## Key Files & References

| File | Purpose |
|------|---------|
| [.env.example](.env.example) | Credential template |
| [main.py](main.py) | Entry point for `--mode paper` |
| [docs/TELEMETRY_COLLECTION.md](../docs/TELEMETRY_COLLECTION.md) | Metric definitions |
| [docs/OPERATIONS_GUIDE.md](../docs/OPERATIONS_GUIDE.md) | Full operational guide |
| [scripts/check_promotion_gates.py](../scripts/check_promotion_gates.py) | Gate validator |
| [equities/telemetry.py](../equities/telemetry.py) | Telemetry collector |

---

## Success Criteria

Paper trading phase is complete when:

1. ✅ 20+ trading days elapsed
2. ✅ Paper→Live gate passes all 6 criteria:
   - Paper days ≥ 20
   - Slippage deviation ≤ 20%
   - Order rejection rate ≤ 1%
   - Reconciliation mismatches = 0
   - Software defect halts = 0
   - Runbook documented
3. ✅ PR merged with gate evidence
4. ✅ Ready to proceed to live scale-up phase

---

**Estimated Duration:** 4-5 calendar weeks (20 trading days)

**Start Date:** [to be filled in]
**Target End Date:** [to be calculated]

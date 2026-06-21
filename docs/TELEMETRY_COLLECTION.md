# Paper & Live Telemetry Collection Guide

This document specifies where and how to collect metrics required for promotion gate validation.

## Promotion Gate: Paper → Live

### Required Metrics & Collection Points

#### 1. **Paper Trading Days Elapsed**
- **Metric:** Total number of trading days in paper mode
- **Collection Point:** `equities/execution.py` → `ExecutionManager.on_bar()`
- **How:** Increment counter when processing each new trading day (0:00 UTC or market open)
- **Storage:** Save to stateful config or log file, one entry per date
- **Example:**
  ```python
  # In ExecutionManager or separate telemetry module
  paper_days = set()
  def on_bar(self, ...):
      paper_days.add(bar.timestamp.date())
  
  # Reporting:
  paper_trading_days = len(paper_days)
  ```

#### 2. **Realized vs. Modeled Slippage (basis points)**
- **Metric:** Mean slippage in basis points; compare realized fills to modeled assumptions
- **Collection Points:**
  - **Model:** `core/config.py` → `ExecutionConfig.slippage_bps` (modeled)
  - **Realized:** `equities/alpaca_broker.py` → `AlpacaBroker.submit_order()` (actual fills)
- **How:**
  ```python
  # In AlpacaBroker or ExecutionManager
  submitted_price = order.limit_price or market_price_at_submission
  filled_price = fill.price
  slippage_bps = (filled_price - submitted_price) / submitted_price * 10000
  realized_slippage_list.append(slippage_bps)
  
  mean_realized_slippage = np.mean(realized_slippage_list)
  ```
- **Reporting:**
  ```json
  {
    "modeled_slippage_bps": 7.0,
    "realized_slippage_bps": 8.1
  }
  ```

#### 3. **Order Rejection Rate (%)**
- **Metric:** Fraction of submitted orders that were rejected or not filled
- **Collection Point:** `equities/alpaca_broker.py` → `submit_order()`, `get_order_status()`
- **How:**
  ```python
  total_orders_submitted = count of submit_order() calls
  rejected_or_unfilled = count of orders with status in ['cancelled', 'rejected', 'expired']
  rejection_rate = rejected_or_unfilled / total_orders_submitted if total_orders_submitted > 0 else 0.0
  ```
- **Reporting:**
  ```json
  {
    "order_rejection_rate": 0.003
  }
  ```

#### 4. **Unresolved Reconciliation Mismatches**
- **Metric:** Count of position mismatches that could not be automatically resolved
- **Collection Point:** `core/reconciliation.py` → `Reconciler.reconcile()`
- **How:**
  ```python
  # In Reconciler class
  def reconcile(self, internal_positions):
      report = self._compare_positions(internal_positions)
      unresolved = [d for d in report.discrepancies if not d.resolved]
      return len(unresolved)
  ```
- **Reporting:**
  ```json
  {
    "unresolved_reconciliation_mismatches": 0
  }
  ```

#### 5. **Kill-Switch Halts Due to Software Defects**
- **Metric:** Count of hard halts triggered by code bugs (vs. legitimate risk conditions)
- **Collection Point:** `core/kill_switch.py` → `CircuitBreaker.check_halt()`
- **How:**
  - Log every halt with reason (e.g., "drawdown exceeded", "error: OrderAPI timeout", "vix spike")
  - Count those with reason containing "error:", "exception:", "defect:", etc.
  ```python
  software_defect_halts = [h for h in halt_log if "error:" in h.reason or "exception:" in h.reason]
  count = len(software_defect_halts)
  ```
- **Reporting:**
  ```json
  {
    "kill_switch_halts_due_to_software_defect": 0
  }
  ```

#### 6. **Runbook & Rollback Documented**
- **Metric:** Boolean; operator confirms runbook exists and rollback steps are tested
- **Collection Point:** Manual (operator checklist)
- **How:**
  - Create `operations/PAPER_TRADING_RUNBOOK.md` with:
    - How to start/stop paper trading
    - Emergency halt procedures
    - Reconciliation steps
    - Rollback to backtest/dry-run
  - Verify all steps documented and operator-tested
- **Reporting:**
  ```json
  {
    "runbook_and_rollback_documented": true
  }
  ```

---

## Promotion Gate: Live Scale-Up

### Required Metrics & Collection Points

#### 1. **Initial Live Sizing Cap (%)**
- **Metric:** Percentage of target capital allocation for first phase
- **Collection Point:** `core/config.py` → `PortfolioConfig.initial_live_capital`
- **How:**
  ```python
  initial_capital = config.initial_live_capital
  target_capital = config.target_capital
  initial_sizing_pct = (initial_capital / target_capital) * 100
  ```
- **Reporting:**
  ```json
  {
    "initial_live_sizing_cap_pct": 20
  }
  ```

#### 2. **Successful Review Cycles**
- **Metric:** Count of multi-day live trading windows with zero Sev-1 incidents and passing review
- **Collection Point:** Manual review checklist after each cycle
- **How:**
  - Run live for N days (typically 5-10 trading days = 1 cycle)
  - At cycle end: review PnL, drawdown, orders, reconciliation
  - If no Sev-1 incidents: mark cycle as "passed"
  - Repeat for at least 2 cycles
- **Reporting:**
  ```json
  {
    "successful_review_cycles": 2
  }
  ```

#### 3. **Stability Metrics Drift (%)**
- **Metric:** How much key metrics drifted from paper baselines (acceptable: ≤ 10%)
- **Collection Point:** Paper vs. live backtest/replay comparison
- **How:**
  ```python
  paper_sharpe = 1.13
  live_sharpe = 1.10
  drift_pct = abs(live_sharpe - paper_sharpe) / abs(paper_sharpe) * 100
  
  # Compute drift for multiple metrics and take max
  metrics_to_check = ['sharpe', 'sortino', 'max_drawdown', 'calmar', 'win_rate']
  drifts = [compute_drift(metric, paper, live) for metric in metrics_to_check]
  stability_drift_pct = max(drifts)
  ```
- **Reporting:**
  ```json
  {
    "stability_metrics_drift_pct": 8
  }
  ```

#### 4. **Sev-1 Incidents**
- **Metric:** Count of critical severity incidents (system crash, large unintended loss, data corruption)
- **Collection Point:** Incident log in `operations/LIVE_INCIDENTS.log`
- **How:**
  - Log every incident with severity: `[Sev-0|Sev-1|Sev-2|Sev-3]`
  - Count incidents with `[Sev-1]` tag
  - For scale-up gate: must be zero
- **Reporting:**
  ```json
  {
    "sev1_incidents": 0
  }
  ```

---

## Implementation Checklist

### For Paper Phase
- [ ] Instrument `ExecutionManager` to track trading days
- [ ] Instrument `AlpacaBroker.submit_order()` to record actual fills and compute slippage
- [ ] Add order rejection tracking to order status checks
- [ ] Instrument `Reconciler.reconcile()` to count unresolved mismatches
- [ ] Instrument `CircuitBreaker` to log halt reasons and count software defect halts
- [ ] Create `operations/PAPER_TRADING_RUNBOOK.md`
- [ ] Run paper trading for 20+ trading days
- [ ] Aggregate all metrics and export to JSON
- [ ] Create PR with `templates/promotion_gate_evidence.paper_to_live.json`
- [ ] CI auto-validates; merge when gate passes

### For Live Scale-Up Phase
- [ ] Set `initial_live_capital` to ≤ 25% of target in config
- [ ] Instrument live trading telemetry collection (same as paper)
- [ ] Create `operations/LIVE_TRADING_RUNBOOK.md` with scale-up timeline
- [ ] Run two 5-10 day review cycles
- [ ] Compare live vs. paper metrics for drift
- [ ] Log all incidents with severity tags
- [ ] Aggregate metrics and export to JSON
- [ ] Create PR with `templates/promotion_gate_evidence.live_scale_up.json`
- [ ] CI auto-validates; merge when gate passes

---

## Example: Paper Gate Collection Script

```python
# scripts/collect_paper_telemetry.py (recommended)

import json
from datetime import datetime, timezone
from pathlib import Path

def collect_paper_telemetry(
    execution_manager,
    broker,
    reconciler,
    circuit_breaker,
    modeled_slippage_bps: float = 7.0,
    runbook_documented: bool = False
) -> dict:
    """Aggregate paper trading telemetry for promotion gate."""
    
    # Count trading days
    paper_days = len(execution_manager.trading_days_log)
    
    # Compute slippage
    mean_realized_slippage = execution_manager.compute_mean_slippage_bps()
    
    # Order rejection rate
    total_orders = broker.total_orders_submitted
    rejected = broker.orders_rejected
    rejection_rate = (rejected / total_orders) if total_orders > 0 else 0.0
    
    # Reconciliation mismatches
    last_recon = reconciler.get_last_report()
    unresolved_mismatches = len([d for d in last_recon.discrepancies if not d.resolved])
    
    # Software defect halts
    defect_halts = [h for h in circuit_breaker.halt_log if "error:" in h.reason]
    defect_halt_count = len(defect_halts)
    
    # Build evidence
    evidence = {
        "paper_trading_days": paper_days,
        "modeled_slippage_bps": modeled_slippage_bps,
        "realized_slippage_bps": mean_realized_slippage,
        "order_rejection_rate": rejection_rate,
        "unresolved_reconciliation_mismatches": unresolved_mismatches,
        "kill_switch_halts_due_to_software_defect": defect_halt_count,
        "runbook_and_rollback_documented": runbook_documented,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    # Write to file
    output_path = Path("templates/promotion_gate_evidence.paper_to_live.json")
    output_path.write_text(json.dumps(evidence, indent=2))
    
    return evidence

if __name__ == "__main__":
    # Usage: python scripts/collect_paper_telemetry.py
    pass
```

---

## See Also
- [CI_AND_GOVERNANCE_SETUP.md](CI_AND_GOVERNANCE_SETUP.md) — Gate validation and CI workflow
- [../scripts/check_promotion_gates.py](../scripts/check_promotion_gates.py) — Gate thresholds and validation logic
- [../templates/promotion_gate_evidence.paper_to_live.example.json](../templates/promotion_gate_evidence.paper_to_live.example.json) — Example evidence file

# ETF Engine — Paper-Trading Operations Runbook

Operational guide for running the ETF tactical-allocation engine against an IBKR
paper account. The strategy logic is validated and frozen; this document covers
**operations only**: how to start, monitor, halt, and recover the live runner.

---

## 1. What the engine does

- Trades a 3-sleeve ETF book (trend, mean-reversion, defensive-carry) combined by
  the validated portfolio combiner with vol-targeting.
- Rebalances on a **low-frequency cadence** (`execution.rebalance_every`, default
  21 trading days). It does **not** trade daily.
- The runner (`--mode run`) wakes on a back-off schedule and trades exactly one
  cycle only when ALL hold: today is a trading session, the time is inside the
  execution window (last 30 min before the close by default), and the cadence
  has elapsed since the last successful rebalance.
- Every cycle is fail-safe: a pre-trade safety gate can BLOCK (skip the cycle) or
  HALT (kill-switch, requires human reset); post-trade reconciliation verifies the
  live book matches intent and persists the result so an unresolved mismatch
  blocks the next cycle.

---

## 2. First-time bring-up

### 2a. Headless droplet (recommended — 24/7/365)

On a fresh Ubuntu 22.04/24.04 droplet (>= 2 GB RAM; 4 GB comfortable), as root:

```bash
git clone https://github.com/taglia21/Algebraic-Topology-Neural-Net-Strategy.git /root/etf-engine
cd /root/etf-engine
bash etf/deploy/setup-droplet.sh      # installs deps, IB Gateway, IBC, both services
```

Then:

1. **Fill in credentials** in `/root/etf-engine/etf-engine.env` (root-only,
   chmod 600): `IBKR_USERNAME`, `IBKR_PASSWORD`, `IBKR_ACCOUNT=DU...`,
   `IBKR_TRADING_MODE=paper`, and `TWS_MAJOR_VRSN` (match the Gateway version
   directory the installer printed, e.g. `10.30` -> `1030`).
2. **Start the headless Gateway** (Xvfb + IBC auto-login; no GUI needed):
   ```bash
   systemctl start ibc-gateway
   journalctl -u ibc-gateway -f      # wait for IBC "Login has completed"
   ```
   IBC keeps Gateway alive 24/7 and triggers IBKR's daily soft-restart
   (`IBC_RESTART_TIME`, default 07:00 ET) — no re-login or 2FA needed.
3. **Preflight (read-only, submits nothing):**
   ```bash
   cd /root/etf-engine && python3 -m etf.main --mode preflight
   ```
   Every check should `PASS` — especially `IBKR connection`.
4. **Start the engine** (paper, submitting):
   ```bash
   systemctl start etf-engine
   journalctl -u etf-engine -f
   ```

Both services are `enabled`, so they auto-start on every reboot, and
`etf-engine` is ordered `After=ibc-gateway` so it never tries to trade before
Gateway is up.

### 2b. Local / manual Gateway (testing on your own machine)

1. **Install + log into IB Gateway** (paper) yourself; enable the API. Paper
   port is **4002** (Gateway) or **7497** (TWS).
2. **Configure** `etf-engine.env` (`IBKR_PORT`, `IBKR_ACCOUNT`, and a
   `ETF_IBKR_CLIENT_ID` distinct from the VRP engine). Leave
   `IBKR_MARKET_DATA_TYPE=3` (delayed) unless you have a real-time data
   subscription.
3. **Preflight (read-only, submits nothing):**
   ```bash
   python3 -m etf.main --mode preflight
   ```
   Confirms: price data loads & fresh, the combined target book computes, IBKR
   connects, the account snapshot returns, a live price is obtainable, and the
   pre-trade safety gate is clear. Resolve every `FAIL` before going further.
4. **Inspect today's intended book** (no broker contact):
   ```bash
   python3 -m etf.main --mode signal
   ```
5. **Start the runner** (paper, submitting):
   ```bash
   python3 -m etf.main --mode run --execute
   ```

> Run a dry-run shadow loop first by omitting `--execute` (it computes and logs
> the decision/target every cycle but submits nothing).

---

## 3. Daily monitoring

- **Logs:** `journalctl -u etf-engine -f`. Each wake logs a `Scheduler @ ... ET`
  line with `trade=`, `session=`, `window=`, `cadence_elapsed=`, `min_to_close=`.
- **On a rebalance cycle**, look for: `Combined target (...)`, `Rebalance result`,
  `Slippage: avg ... bps [within/OVER budget]`, and either
  `Reconciliation OK` or `Reconciliation MISMATCH`.
- **State files** (under `.etf_telemetry/`):
  - `reconciliation_state.json` — last reconciliation outcome (gates next cycle).
  - `schedule_state.json` — last rebalance date (drives the cadence).
  - `equity_state.json` (`execution.state_path`) — persistent equity/peak for the
    drawdown kill-switch.
  - `slippage_log` — per-cycle execution-quality telemetry.

### Promotion gate (paper -> live)
Do not enable `--live` until: >= 20 paper trading days, slippage deviation
<= 20% of modeled, order-rejection rate <= 1%, **zero** unresolved reconciliation
mismatches, and no software-caused kill-switch halts.

---

## 4. Halts & how to recover

The engine halts/blocks itself; recovery is a deliberate human action.

| Symptom in logs | Meaning | Action |
|---|---|---|
| `KILL-SWITCH: drawdown ...` / `daily loss ...` | Catastrophic trigger fired | Investigate market + book. Only after review, reset (below). |
| `BLOCK: unresolved reconciliation mismatch from prior cycle` | Last cycle's live book drifted from intent | Inspect `reconciliation_state.json` + the broker. Manually true up or reset. |
| `BLOCK: market/account data is stale or missing` | Data fail-safe | Check data feed / Gateway; resolves itself once data is fresh. |
| `Could not connect to IBKR` | Gateway down / API off | Restart Gateway, confirm port, re-run `--mode preflight`. |

### IB Gateway didn't come back (headless droplet)

```bash
systemctl status ibc-gateway          # is the service running?
journalctl -u ibc-gateway -n 100      # look for IBC login errors / 2FA prompts
systemctl restart ibc-gateway         # force a clean relaunch (Xvfb + IBC)
```
Common causes: wrong `IBKR_USERNAME`/`IBKR_PASSWORD` in the env file; a
mismatched `TWS_MAJOR_VRSN`; or IBKR requiring a one-time 2FA the first time from
a new IP. If 2FA blocks headless login, log into the IBKR website once from the
droplet's region/IP (or disable 2FA on the **paper** login) and restart.

### The reset (after human review only)
```bash
# Clear reconciliation + schedule state so the next cycle starts clean:
python3 -m etf.main --mode reset-safety

# Also reset persistent equity/peak (rarely needed; only on an intentional
# capital change or fresh start):
python3 -m etf.main --mode reset-safety --reset-equity
```
After a reset, the next eligible window will rebalance from a clean slate.

---

## 5. Stop / restart / update

```bash
systemctl stop etf-engine        # stop the runner (no positions are closed)
systemctl restart etf-engine     # safe: state persists across restarts
git pull --ff-only origin main && systemctl restart etf-engine   # update + restart
```

> The Gateway (`ibc-gateway`) and the engine (`etf-engine`) are independent
> services. Restarting the engine does NOT disturb the Gateway session;
> restarting `ibc-gateway` re-logs-in and the engine reconnects on its next wake.

Restarts are safe at any time: cadence, reconciliation, and equity state are all
persisted, so a restart never causes a double-rebalance or loss of the kill-switch
memory.

---

## 6. Useful one-off commands

```bash
# Single scheduling check then exit (ideal for a cron trigger instead of a daemon):
python3 -m etf.main --mode run --execute --once

# Force a rebalance now (bypasses cadence, NOT the market-open/window gates):
python3 -m etf.main --mode run --execute --once --force

# Widen the execution window to the whole session (still never trades when closed):
python3 -m etf.main --mode run --execute --anytime

# Read-only connectivity + readiness check (submits nothing):
python3 -m etf.main --mode preflight
```

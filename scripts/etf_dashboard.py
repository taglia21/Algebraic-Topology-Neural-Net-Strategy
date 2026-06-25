#!/usr/bin/env python3
"""ETF operations dashboard.

Professional monitoring surface for readiness, promotion gates, telemetry,
and recent order/fill activity.

Run:
    streamlit run scripts/etf_dashboard.py
"""

from __future__ import annotations

import glob
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]


@st.cache_data(ttl=20)
def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


@st.cache_data(ttl=20)
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except Exception:
        return rows
    return rows


def _latest_trade_log() -> Path | None:
    files = sorted(glob.glob(str(ROOT / "logs" / "trades_*.jsonl")))
    if not files:
        return None
    return Path(files[-1])


def _latest_phase3_artifact() -> Path | None:
    candidates = sorted(glob.glob(str(ROOT / "artifacts" / "etf_phase3_portfolio*.json")))
    if candidates:
        return Path(candidates[-1])
    default_path = ROOT / "etf_phase3_portfolio.json"
    if default_path.exists():
        return default_path
    return None


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.2f}%"


def _fmt_num(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.2f}"


def _status_chip(label: str, ok: bool | None) -> str:
    if ok is None:
        return f"{label}: UNKNOWN"
    return f"{label}: {'OK' if ok else 'BLOCKED'}"


def _render_header() -> None:
    st.set_page_config(page_title="ATNN ETF Ops Terminal", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at 10% 10%, #1a1e2a 0%, #090b10 55%);
            color: #f7f1d5;
        }
        .block-container {
            padding-top: 1rem;
        }
        .title {
            font-size: 2.0rem;
            font-weight: 700;
            color: #f6c244;
            letter-spacing: 0.04em;
        }
        .subtitle {
            color: #9aa6b2;
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }
        .panel {
            border: 1px solid #263346;
            border-radius: 10px;
            background: linear-gradient(180deg, #0f131b 0%, #0b0e14 100%);
            padding: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="title">ATNN ETF OPERATIONS TERMINAL</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Readiness, gate status, execution telemetry, and PnL state in one view.</div>',
        unsafe_allow_html=True,
    )


def _run_preflight() -> str:
    cmd = ["python", "-m", "etf.main", "--mode", "preflight"]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=180)
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def main() -> None:
    _render_header()

    gate_path = _latest_phase3_artifact()
    gate = _read_json(gate_path) if gate_path else None

    eq_state_path = ROOT / ".etf_telemetry" / "equity_state.json"
    recon_state_path = ROOT / ".etf_telemetry" / "reconciliation_state.json"
    sched_state_path = ROOT / ".etf_telemetry" / "schedule_state.json"
    slippage_path = ROOT / ".etf_telemetry" / "slippage.jsonl"

    eq_state = _read_json(eq_state_path)
    recon_state = _read_json(recon_state_path)
    sched_state = _read_json(sched_state_path)
    slippage_rows = _read_jsonl(slippage_path)

    trading_allowed = bool(gate and gate.get("gate_cleared", False))
    recon_ok = None if recon_state is None else bool(recon_state.get("ok", False))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Promotion Gate", "CLEARED" if trading_allowed else "BLOCKED")
    col2.metric("Reconciliation", "OK" if recon_ok else ("BLOCKED" if recon_ok is False else "UNKNOWN"))

    last_eq = float(eq_state["last_equity"]) if eq_state and "last_equity" in eq_state else None
    peak_eq = float(eq_state["peak_equity"]) if eq_state and "peak_equity" in eq_state else None
    dd = None
    if last_eq is not None and peak_eq and peak_eq > 0:
        dd = 1.0 - (last_eq / peak_eq)
    col3.metric("Last Equity", f"${last_eq:,.2f}" if last_eq is not None else "n/a")
    col4.metric("Current Drawdown", _fmt_pct(dd))

    chip_cols = st.columns(3)
    chip_cols[0].info(_status_chip("Execution Permission", trading_allowed))
    chip_cols[1].info(_status_chip("Prior Reconciliation", recon_ok))
    schedule_label = "UNKNOWN"
    if sched_state and sched_state.get("last_rebalance_date"):
        schedule_label = f"last rebalance: {sched_state['last_rebalance_date']}"
    chip_cols[2].info(f"Scheduler: {schedule_label}")

    st.markdown("### Promotion Gate Breakdown")
    if gate and isinstance(gate.get("gate"), dict):
        gate_df = pd.DataFrame(
            [{"Rule": k, "Pass": bool(v)} for k, v in gate["gate"].items()]
        )
        st.dataframe(gate_df, width="stretch", hide_index=True)

        m = gate.get("metrics", {})
        stat_cols = st.columns(6)
        stat_cols[0].metric("Sharpe", _fmt_num(m.get("sharpe")))
        stat_cols[1].metric("Sortino", _fmt_num(m.get("sortino")))
        stat_cols[2].metric("CAGR", _fmt_pct(m.get("cagr")))
        stat_cols[3].metric("Max DD", _fmt_pct(m.get("max_drawdown")))
        stat_cols[4].metric("Calmar", _fmt_num(m.get("calmar")))
        stat_cols[5].metric("Profit Factor", _fmt_num(m.get("profit_factor")))
    else:
        st.warning("No promotion gate JSON found yet. Run portfolio mode to generate evidence.")

    st.markdown("### Slippage And Fill Quality")
    if slippage_rows:
        slippage_df = pd.DataFrame(slippage_rows)
        if "as_of" in slippage_df.columns:
            slippage_df["as_of"] = pd.to_datetime(slippage_df["as_of"], errors="coerce")
            slippage_df = slippage_df.sort_values("as_of")
        fig = px.line(
            slippage_df,
            x="as_of",
            y="avg_slippage_bps",
            markers=True,
            title="Average Slippage (bps)",
        )
        fig.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig, width="stretch")

        c1, c2, c3 = st.columns(3)
        c1.metric("Cycles Logged", str(len(slippage_df)))
        c2.metric("Latest Avg Slippage", f"{slippage_df.iloc[-1]['avg_slippage_bps']:.2f} bps")
        c3.metric("Latest Cost", f"${slippage_df.iloc[-1].get('total_cost_usd', 0.0):,.2f}")
    else:
        st.info("No slippage telemetry yet (.etf_telemetry/slippage.jsonl not found or empty).")

    st.markdown("### Recent Trade Events")
    trade_path = _latest_trade_log()
    if trade_path:
        trade_rows = _read_jsonl(trade_path)
        if trade_rows:
            trades_df = pd.DataFrame(trade_rows)
            if "ts" in trades_df.columns:
                trades_df["ts"] = pd.to_datetime(trades_df["ts"], errors="coerce")
                trades_df = trades_df.sort_values("ts", ascending=False)

            show_cols = [c for c in ["ts", "event", "symbol", "side", "qty", "status", "fill_price", "level", "message"] if c in trades_df.columns]
            st.caption(f"Source: {trade_path.relative_to(ROOT)}")
            st.dataframe(trades_df[show_cols].head(50), width="stretch", hide_index=True)

            fills = trades_df[trades_df.get("event").astype(str).str.lower() == "fill"] if "event" in trades_df.columns else pd.DataFrame()
            if not fills.empty and "fill_price" in fills.columns and "fill_qty" in fills.columns:
                fills = fills.copy()
                fills["notional"] = fills["fill_price"].astype(float) * fills["fill_qty"].astype(float)
                st.metric("Approx Filled Notional (latest log)", f"${fills['notional'].sum():,.2f}")
        else:
            st.info("Trade log exists but is empty.")
    else:
        st.info("No trade log found under logs/trades_*.jsonl.")

    st.markdown("### Live Connectivity Probe")
    st.write("Run a preflight check from the dashboard to verify IBKR connectivity and execution readiness.")
    if st.button("Run Preflight Now"):
        with st.spinner("Running preflight..."):
            try:
                output = _run_preflight()
            except Exception as exc:
                output = f"Preflight failed to run: {exc}"
        ready = "READY for paper trading" in output
        st.success("Preflight returned READY") if ready else st.error("Preflight returned NOT READY")
        st.code(output[-8000:])

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.caption(f"Dashboard snapshot time: {now_utc}")


if __name__ == "__main__":
    main()

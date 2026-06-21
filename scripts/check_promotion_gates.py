#!/usr/bin/env python3
"""Validate promotion gates for research->paper->live progression.

This script enforces explicit quantitative and operational thresholds and
produces a deterministic pass/fail report for auditability.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateRule:
    field: str
    comparator: str
    threshold: float | int | bool
    description: str


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "pass"}:
            return True
        if lowered in {"false", "0", "no", "n", "fail"}:
            return False
    raise ValueError(f"Cannot coerce value to bool: {value!r}")


def _compare(actual: Any, rule: GateRule) -> bool:
    if rule.comparator == "ge":
        return float(actual) >= float(rule.threshold)
    if rule.comparator == "le":
        return float(actual) <= float(rule.threshold)
    if rule.comparator == "eq":
        return actual == rule.threshold
    if rule.comparator == "bool_is":
        return _to_bool(actual) is bool(rule.threshold)
    raise ValueError(f"Unsupported comparator: {rule.comparator}")


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _build_rules(gate: str, evidence: dict[str, Any]) -> tuple[list[GateRule], dict[str, Any]]:
    if gate == "research_to_paper":
        return [
            GateRule("oos_sharpe", "ge", 1.10, "OOS Sharpe >= 1.10"),
            GateRule("max_drawdown", "le", 0.18, "Max drawdown <= 18%"),
            GateRule("calmar", "ge", 0.80, "Calmar >= 0.80"),
            GateRule("profit_factor", "ge", 1.20, "Profit factor >= 1.20"),
            GateRule("leakage_findings", "bool_is", False, "No look-ahead/leakage findings"),
            GateRule("all_touched_tests_pass", "bool_is", True, "All touched tests pass"),
        ], {}

    if gate == "paper_to_live":
        modeled = float(evidence.get("modeled_slippage_bps", 0.0))
        realized = float(evidence.get("realized_slippage_bps", 0.0))
        denominator = abs(modeled) if abs(modeled) > 1e-12 else 1.0
        slippage_relative_deviation = abs(realized - modeled) / denominator
        derived = {"slippage_relative_deviation": slippage_relative_deviation}
        return [
            GateRule("paper_trading_days", "ge", 20, "Paper period >= 20 trading days"),
            GateRule(
                "slippage_relative_deviation",
                "le",
                0.20,
                "Paper/live-sim slippage deviation <= 20%",
            ),
            GateRule("order_rejection_rate", "le", 0.01, "Order rejection rate <= 1.0%"),
            GateRule(
                "unresolved_reconciliation_mismatches",
                "eq",
                0,
                "Reconciliation mismatch rate == 0 unresolved mismatches",
            ),
            GateRule(
                "kill_switch_halts_due_to_software_defect",
                "eq",
                0,
                "No kill switch hard halt caused by software defects",
            ),
            GateRule("runbook_and_rollback_documented", "bool_is", True, "Runbook and rollback documented"),
        ], derived

    if gate == "live_scale_up":
        return [
            GateRule("initial_live_sizing_cap_pct", "le", 25, "Initial live sizing cap <= 25%"),
            GateRule("successful_review_cycles", "ge", 2, "Two successful review cycles"),
            GateRule("stability_metrics_drift_pct", "le", 10, "Stability metrics drift <= 10%"),
            GateRule("sev1_incidents", "eq", 0, "No Sev-1 incidents"),
        ], {}

    raise ValueError(f"Unknown gate: {gate}")


def validate_gate(gate: str, evidence: dict[str, Any]) -> dict[str, Any]:
    rules, derived_values = _build_rules(gate, evidence)
    full_evidence = {**evidence, **derived_values}

    checks: list[dict[str, Any]] = []
    all_passed = True

    for rule in rules:
        present = rule.field in full_evidence
        if not present:
            checks.append(
                {
                    "field": rule.field,
                    "description": rule.description,
                    "status": "missing",
                    "expected": f"{rule.comparator} {rule.threshold}",
                    "actual": None,
                    "passed": False,
                }
            )
            all_passed = False
            continue

        actual = full_evidence[rule.field]
        passed = _compare(actual, rule)
        checks.append(
            {
                "field": rule.field,
                "description": rule.description,
                "status": "ok" if passed else "failed",
                "expected": f"{rule.comparator} {rule.threshold}",
                "actual": actual,
                "passed": passed,
            }
        )
        all_passed = all_passed and passed

    return {
        "gate": gate,
        "passed": all_passed,
        "checks": checks,
        "derived": derived_values,
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"Gate: {report['gate']}")
    print(f"Status: {'PASS' if report['passed'] else 'FAIL'}")
    derived = report.get("derived", {})
    if "slippage_relative_deviation" in derived:
        print(
            "Derived: slippage_relative_deviation="
            f"{_pct(float(derived['slippage_relative_deviation']))}"
        )
    print("Checks:")
    for check in report["checks"]:
        print(
            f"- [{ 'OK' if check['passed'] else 'X' }] "
            f"{check['description']} | field={check['field']} | "
            f"actual={check['actual']} | expected={check['expected']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate promotion gates from evidence JSON.")
    parser.add_argument(
        "--gate",
        required=True,
        choices=["research_to_paper", "paper_to_live", "live_scale_up"],
        help="Gate to validate.",
    )
    parser.add_argument("--input", required=True, help="Path to JSON evidence file.")
    parser.add_argument("--report-out", help="Optional path to write JSON report.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input evidence file not found: {input_path}")

    evidence = json.loads(input_path.read_text(encoding="utf-8"))
    report = validate_gate(args.gate, evidence)
    _print_report(report)

    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote report: {report_path}")

    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
V52 Team of Rivals - Multi-Agent Trading System
Based on: "If You Want Coherence, Orchestrate a Team of Rivals"
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.agents.orchestrator import ChiefOrchestrator
from src.agents.risk_team import RiskTeam
from src.agents.strategy_team import StrategyTeam
from src.agents.data_team import DataTeam
from src.agents.execution_team import ExecutionTeam
from src.agents.research_team import ResearchTeam


class TradingOffice:
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        print("="*60)
        print("INITIALIZING AI TRADING OFFICE")
        print("Team of Rivals Architecture")
        print("="*60)
        
        self.llm_model = llm_model
        print("\nInitializing departments...")
        
        self.orchestrator = ChiefOrchestrator(llm_model)
        print("  [OK] Chief Orchestrator")
        
        self.risk_team = RiskTeam(llm_model)
        print("  [OK] Risk Team (VETO authority)")
        
        self.strategy_team = StrategyTeam(llm_model)
        print("  [OK] Strategy Team")
        
        self.data_team = DataTeam(llm_model)
        print("  [OK] Data Team")
        
        self.execution_team = ExecutionTeam(llm_model)
        print("  [OK] Execution Team")
        
        self.research_team = ResearchTeam(llm_model)
        print("  [OK] Research Team")
        
        self.portfolio_state = {
            "equity": 100000,
            "drawdown": 0.0,
            "max_position": 0.0,
            "correlation": 0.0,
            "positions": {},
            "pnl_today": 0.0
        }
        
        self.meeting_history = []
        print("\n[OK] All departments initialized")
        print("="*60 + "\n")
    
    def morning_standup(self) -> Dict[str, Any]:
        print("\n" + "="*60)
        print("MORNING STANDUP - {}".format(datetime.now().strftime("%Y-%m-%d %H:%M")))
        print("="*60)
        
        reports = {
            "strategy": self.strategy_team.get_department_report(),
            "data": self.data_team.get_department_report(),
            "risk": self.risk_team.get_department_report(self.portfolio_state),
            "execution": self.execution_team.get_department_report(),
            "research": self.research_team.get_department_report()
        }
        
        summary = self.orchestrator.run_meeting("morning_standup", reports)
        
        print("\nDEPARTMENT STATUS:")
        for dept, report in reports.items():
            status = report.get("status", "UNKNOWN")
            print(f"  [{status}] {dept.upper()}")
        
        alerts = []
        if self.portfolio_state.get("drawdown", 0) > 0.05:
            alerts.append(f"Drawdown at {self.portfolio_state['drawdown']:.1%}")
        if reports["risk"].get("vetoes_today", 0) > 0:
            alerts.append(f"{reports['risk']['vetoes_today']} trade(s) vetoed")
        
        if alerts:
            print("\nALERTS:")
            for alert in alerts:
                print(f"  [!] {alert}")
        else:
            print("\n[OK] No critical alerts")
        
        meeting_record = {
            "type": "morning_standup",
            "timestamp": datetime.now().isoformat(),
            "reports": reports,
            "alerts": alerts
        }
        self.meeting_history.append(meeting_record)
        print("\n" + "="*60)
        return meeting_record
    
    def propose_trade(self, symbol: str, side: str, strategy: str,
                      quantity: int, expected_price: float) -> Dict[str, Any]:
        print(f"\nTRADE PROPOSAL: {side.upper()} {quantity} {symbol}")
        print(f"  Strategy: {strategy}")
        print(f"  Expected Price: ${expected_price:.2f}")
        
        trade_signal = {
            "symbol": symbol,
            "side": side,
            "strategy": strategy,
            "quantity": quantity,
            "expected_price": expected_price,
            "position_pct": (quantity * expected_price) / self.portfolio_state["equity"],
            "current_drawdown": self.portfolio_state.get("drawdown", 0),
            "portfolio_correlation": self.portfolio_state.get("correlation", 0),
            "backtest_sharpe": 1.5,
            "live_sharpe": 1.2,
        }
        
        print("\n  -> Data Team: Validating...")
        data_check = self.data_team.validate_data({
            "missing_pct": 0.001,
            "stale_count": 0,
            "lookahead_detected": False,
            "outlier_pct": 0.0001
        })
        
        if not data_check["approved"]:
            print("  [X] VETOED by Data Team")
            return {"approved": False, "vetoed_by": "Data Team"}
        print("  [OK] Data Team: Approved")
        
        print("  -> Risk Team: Reviewing...")
        risk_check = self.risk_team.review_proposed_trade(trade_signal)
        
        if not risk_check["approved"]:
            print("  [X] VETOED by Risk Team")
            for reason in risk_check.get("veto_reasons", []):
                print(f"      - {reason}")
            return {"approved": False, "vetoed_by": "Risk Team", "reasons": risk_check.get("veto_reasons")}
        print("  [OK] Risk Team: Approved")
        
        print("  -> Execution Team: Placing order...")
        execution_result = self.execution_team.execute_trade({
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "expected_price": expected_price
        })
        
        print(f"  [OK] Trade EXECUTED: {execution_result['status']}")
        return {"approved": True, "execution": execution_result}
    
    def eod_review(self) -> Dict[str, Any]:
        print("\n" + "="*60)
        print("END OF DAY REVIEW - {}".format(datetime.now().strftime("%Y-%m-%d %H:%M")))
        print("="*60)
        
        trading_log = {
            "pnl": self.portfolio_state.get("pnl_today", 0),
            "trades": self.execution_team.execution_log
        }
        
        print(f"\nTODAY'S P&L: ${trading_log['pnl']:.2f}")
        print(f"  Trades Executed: {len(trading_log['trades'])}")
        
        risk_report = self.risk_team.get_department_report(self.portfolio_state)
        print(f"  Vetoes: {risk_report.get('vetoes_today', 0)}")
        
        meeting_record = {
            "type": "eod_review",
            "timestamp": datetime.now().isoformat(),
            "trading_log": trading_log
        }
        self.meeting_history.append(meeting_record)
        print("\n" + "="*60)
        return meeting_record
    
    def run_demo(self):
        print("\n" + "*"*60)
        print("TEAM OF RIVALS DEMO")
        print("*"*60)
        
        self.morning_standup()
        
        print("\n" + "-"*40)
        print("TESTING TRADE APPROVAL PROCESS")
        print("-"*40)
        
        print("\n[Test 1] Reasonable trade - should PASS")
        self.propose_trade("AAPL", "buy", "momentum", 10, 180.00)
        
        print("\n[Test 2] Oversized position - should be VETOED")
        self.propose_trade("NVDA", "buy", "mean_reversion", 100, 800.00)
        
        self.eod_review()
        
        print("\n" + "*"*60)
        print("DEMO COMPLETE")
        print("*"*60)


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[!] WARNING: OPENAI_API_KEY not set")
        print("    Add to .env: OPENAI_API_KEY=sk-...")
        print("    Running in demo mode...\n")
    
    office = TradingOffice()
    office.run_demo()
    return office

if __name__ == "__main__":
    office = main()


from datetime import datetime
import json
from dotenv import load_dotenv
load_dotenv()

class RiskTeam:
    def __init__(self, llm_model="gpt-4o-mini"):
        self.llm_model = llm_model
        self.risk_limits = {
            "max_position_pct": 0.10,
            "max_drawdown_pct": 0.10,
            "max_correlation": 0.85,
            "sharpe_degradation_threshold": 0.50,
        }
        self.veto_log = []
    
    def check_position_size(self, position_pct):
        approved = position_pct <= self.risk_limits["max_position_pct"]
        return {"check": "position_size", "approved": approved, "current": position_pct,
                "reason": None if approved else f"Position {position_pct:.1%} exceeds {self.risk_limits['max_position_pct']:.1%} limit"}
    
    def check_drawdown(self, current_drawdown):
        approved = current_drawdown <= self.risk_limits["max_drawdown_pct"]
        return {"check": "drawdown", "approved": approved, "current": current_drawdown,
                "reason": None if approved else f"Drawdown {current_drawdown:.1%} exceeds limit"}
    
    def review_proposed_trade(self, trade_signal):
        checks = []
        checks.append(self.check_position_size(trade_signal.get("position_pct", 0)))
        checks.append(self.check_drawdown(trade_signal.get("current_drawdown", 0)))
        
        all_approved = all(check["approved"] for check in checks)
        failed_checks = [check for check in checks if not check["approved"]]
        
        decision = {
            "timestamp": datetime.now().isoformat(),
            "symbol": trade_signal.get("symbol"),
            "approved": all_approved,
            "checks": checks,
            "failed_checks": failed_checks,
            "vetoed_by": "Risk Team" if not all_approved else None,
            "veto_reasons": [check["reason"] for check in failed_checks] if failed_checks else None
        }
        
        if not all_approved:
            self.veto_log.append(decision)
        return decision
    
    def get_department_report(self, portfolio_state):
        return {
            "department": "Risk Team",
            "timestamp": datetime.now().isoformat(),
            "current_drawdown": portfolio_state.get("drawdown", 0),
            "vetoes_today": len([v for v in self.veto_log if v["timestamp"].startswith(datetime.now().strftime("%Y-%m-%d"))]),
            "status": "GREEN" if portfolio_state.get("drawdown", 0) < 0.05 else "YELLOW" if portfolio_state.get("drawdown", 0) < 0.10 else "RED"
        }

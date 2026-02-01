
from datetime import datetime
import json
from dotenv import load_dotenv
load_dotenv()

class DataTeam:
    def __init__(self, llm_model="gpt-4o-mini"):
        self.llm_model = llm_model
        self.validation_log = []
    
    def validate_data(self, data_info):
        checks = {
            "no_missing_data": data_info.get("missing_pct", 0) < 0.01,
            "no_stale_prices": data_info.get("stale_count", 0) == 0,
            "no_lookahead": data_info.get("lookahead_detected", False) == False,
        }
        approved = all(checks.values())
        result = {"timestamp": datetime.now().isoformat(), "approved": approved, "checks": checks}
        self.validation_log.append(result)
        return result
    
    def get_department_report(self):
        return {"department": "Data Team", "timestamp": datetime.now().isoformat(),
                "validations_today": len(self.validation_log), "status": "GREEN"}

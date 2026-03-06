import logging
import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ComponentHeartbeat:
    name: str
    last_seen: float
    status: str = "active"

class ProductionOrchestrator:
    """Manages system lifecycle and component health."""
    def __init__(self, heartbeat_timeout: float = 5.0):
        self.heartbeat_timeout = heartbeat_timeout
        self.components = {} # name -> ComponentHeartbeat
        self.is_running = False
        self._lock = threading.Lock()

    def register_component(self, name: str):
        with self._lock:
            self.components[name] = ComponentHeartbeat(name=name, last_seen=time.time())
            
    def pulse(self, name: str):
        with self._lock:
            if name in self.components:
                self.components[name].last_seen = time.time()
                self.components[name].status = "active"

    def monitor_loop(self):
        """Monitors components and flags failures."""
        while self.is_running:
            now = time.time()
            with self._lock:
                for name, comp in self.components.items():
                    if now - comp.last_seen > self.heartbeat_timeout:
                        comp.status = "stale"
                        logger.warning(f"COMPONENT STALE: {name} (last seen {now - comp.last_seen:.2f}s ago)")
                    if now - comp.last_seen > self.heartbeat_timeout * 2:
                        comp.status = "failed"
                        logger.error(f"COMPONENT FAILED: {name} - Triggering failover protocols")
            time.sleep(1.0)

    def start(self):
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Production Orchestrator started")

    def stop(self):
        self.is_running = False
        self.monitor_thread.join()
        logger.info("Production Orchestrator stopped")

class AIMetaController:
    """Adjusts system hyperparameters using AI-driven regime analysis."""
    def __init__(self):
        self.regime_map = {
            "high_vol": {"leverage_scale": 0.5, "stop_loss_mult": 1.5},
            "low_vol": {"leverage_scale": 1.2, "stop_loss_mult": 0.8},
            "trending": {"leverage_scale": 1.0, "stop_loss_mult": 1.0}
        }

    def get_optimal_params(self, market_regime: str) -> Dict[str, float]:
        """Returns adjusted parameters for the current market regime."""
        params = self.regime_map.get(market_regime, self.regime_map["trending"])
        logger.info(f"AI Meta-Controller: Applying {market_regime} parameters: {params}")
        return params


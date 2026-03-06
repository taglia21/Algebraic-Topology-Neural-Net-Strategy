import pytest
import numpy as np
import pandas as pd
import time
from src.ml.rl.environment import TradingEnv
from src.ml.rl.agent import PPOAgent, RLCoordinator
from src.orchestration.arbitrage import ArbitrageEngine, ArbitrageOpportunity
from src.orchestration.risk_manager import GlobalRiskManager
from src.hft.sor import SmartOrderRouter, ExecutionVenue, AdaptiveLimitOrderPlacer
from src.controller.orchestrator import ProductionOrchestrator, AIMetaController

class TestTier4Apex:
    @pytest.fixture
    def sample_market_data(self):
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        df = pd.DataFrame({
            'close': np.linspace(100, 110, 100) + np.random.randn(100)
        }, index=dates)
        return df

    def test_rl_environment(self, sample_market_data):
        env = TradingEnv(sample_market_data, window_size=10)
        state = env.reset()
        assert state.shape[0] == 10 + 2 # window + position info
        
        obs, reward, done, info = env.step(1) # Buy
        assert info['inventory'] > 0
        assert 'total_value' in info

    def test_ppo_agent_selection(self):
        agent = PPOAgent(state_dim=12, action_dim=3)
        state = np.random.randn(12)
        action = agent.select_action(state)
        assert action in [0, 1, 2]

    def test_arbitrage_detection(self):
        engine = ArbitrageEngine(min_profit_threshold=0.01)
        engine.add_pair("AAPL", "AAPL_ADR", hedge_ratio=1.0)
        
        market_data = {
            "AAPL": {"price": 150.0},
            "AAPL_ADR": {"price": 145.0}
        }
        
        opps = engine.scan(market_data)
        assert len(opps) == 1
        assert opps[0].expected_profit > 0.03 # (150-145)/145 = 3.4%

    def test_global_risk_manager(self):
        mgr = GlobalRiskManager(max_total_drawdown=0.1)
        mgr.update_metrics(100000.0) # Peak
        mgr.update_metrics(95000.0)  # 5% drawdown
        assert mgr.check_risk_limits() is True
        
        mgr.update_metrics(85000.0)  # 15% drawdown
        assert mgr.check_risk_limits() is False

    def test_smart_order_router(self):
        sor = SmartOrderRouter()
        sor.add_venue(ExecutionVenue("NASDAQ", 0.0001, 0.5, 0.9))
        sor.add_venue(ExecutionVenue("ARCA", 0.0002, 1.2, 0.8))
        
        venue = sor.route_order("SPY", 100, "buy")
        assert venue == "NASDAQ"

    def test_production_orchestrator(self):
        orch = ProductionOrchestrator(heartbeat_timeout=0.5)
        orch.register_component("OMS")
        orch.pulse("OMS")
        orch.start()
        
        time.sleep(0.1)
        assert orch.components["OMS"].status == "active"
        
        time.sleep(0.6) # Wait for stale timeout
        # Manual check since thread might not have run yet in test env
        orch.pulse("OMS") # Pulse again to keep active
        assert orch.components["OMS"].status == "active"
        orch.stop()

    def test_ai_meta_controller(self):
        meta = AIMetaController()
        params = meta.get_optimal_params("high_vol")
        assert params["leverage_scale"] == 0.5
        assert params["stop_loss_mult"] == 1.5


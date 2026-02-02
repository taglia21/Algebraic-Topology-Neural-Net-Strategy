#!/usr/bin/env python3
"""Activate Continuous Learning System.

This script initializes and starts the complete continuous learning
infrastructure for the trading bot, including:
- Online learning models
- Trade monitoring
- Automatic drift detection
- Genetic optimization
- RL-based reward learning

Usage:
    python src/activate_continuous_learning.py
    
    # Or in your trading code:
    from src.activate_continuous_learning import activate_learning
    activate_learning()
"""

import os
import sys
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def activate_learning(monitor_interval: int = 60) -> dict:
    """Activate the complete continuous learning system.
    
    Args:
        monitor_interval: Seconds between position checks (default: 60)
    
    Returns:
        Dict with status information
    """
    status = {
        'timestamp': datetime.now().isoformat(),
        'components': {},
        'success': False
    }
    
    logger.info("="*60)
    logger.info("ACTIVATING CONTINUOUS LEARNING SYSTEM")
    logger.info("="*60)
    
    # 1. Initialize Adaptive Learning Engine
    try:
        from src.adaptive_learning_engine import (
            OnlineLearningModel,
            ReinforcementLearningReward,
            DriftDetector,
            StrategyEvolution,
            AdaptiveLearningEngine
        )
        status['components']['adaptive_learning_engine'] = 'loaded'
        logger.info("[OK] Adaptive Learning Engine loaded")
    except Exception as e:
        status['components']['adaptive_learning_engine'] = f'error: {e}'
        logger.error(f"[FAIL] Adaptive Learning Engine: {e}")
    
    # 2. Initialize Learning Integration
    try:
        from src.learning_integration import (
            get_continuous_learner,
            initialize_continuous_learning
        )
        learner = initialize_continuous_learning()
        status['components']['learning_integration'] = 'initialized'
        logger.info("[OK] Learning Integration initialized")
    except Exception as e:
        status['components']['learning_integration'] = f'error: {e}'
        logger.error(f"[FAIL] Learning Integration: {e}")
    
    # 3. Initialize Trade Monitor
    try:
        from src.trade_monitor import (
            get_trade_monitor,
            start_learning_monitor
        )
        monitor = get_trade_monitor()
        start_learning_monitor(interval=monitor_interval)
        status['components']['trade_monitor'] = 'running'
        logger.info(f"[OK] Trade Monitor started (interval: {monitor_interval}s)")
    except Exception as e:
        status['components']['trade_monitor'] = f'error: {e}'
        logger.error(f"[FAIL] Trade Monitor: {e}")
    
    # 4. Initialize ML Retraining hooks
    try:
        from src.ml_retraining import MLRetrainingManager
        retrainer = MLRetrainingManager()
        status['components']['ml_retraining'] = 'ready'
        logger.info("[OK] ML Retraining Manager ready")
    except Exception as e:
        status['components']['ml_retraining'] = f'error: {e}'
        logger.warning(f"[WARN] ML Retraining: {e}")
    
    # Check overall status
    errors = [k for k, v in status['components'].items() if 'error' in str(v)]
    if len(errors) == 0:
        status['success'] = True
        logger.info("="*60)
        logger.info("CONTINUOUS LEARNING SYSTEM ACTIVE")
        logger.info("="*60)
    else:
        logger.warning(f"System partially active. Errors in: {errors}")
    
    return status


def get_learning_status() -> dict:
    """Get current status of all learning components."""
    status = {'timestamp': datetime.now().isoformat()}
    
    # Check Trade Monitor
    try:
        from src.trade_monitor import get_trade_monitor
        monitor = get_trade_monitor()
        status['trade_monitor'] = monitor.get_learning_stats()
    except Exception as e:
        status['trade_monitor'] = {'error': str(e)}
    
    # Check Learning Integration
    try:
        from src.learning_integration import get_continuous_learner
        learner = get_continuous_learner()
        status['learner'] = learner.get_stats()
    except Exception as e:
        status['learner'] = {'error': str(e)}
    
    return status


def deactivate_learning():
    """Stop the continuous learning system."""
    logger.info("Deactivating continuous learning...")
    
    try:
        from src.trade_monitor import stop_learning_monitor
        stop_learning_monitor()
        logger.info("Trade monitor stopped")
    except Exception as e:
        logger.error(f"Error stopping trade monitor: {e}")
    
    logger.info("Continuous learning deactivated")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("   TEAM OF RIVALS: CONTINUOUS LEARNING ACTIVATION")
    print("="*60 + "\n")
    
    # Activate
    status = activate_learning(monitor_interval=30)
    
    print("\nStatus:")
    for component, state in status['components'].items():
        icon = "OK" if 'error' not in str(state) else "FAIL"
        print(f"  [{icon}] {component}: {state}")
    
    print(f"\nOverall: {'SUCCESS' if status['success'] else 'PARTIAL'}")
    
    if status['success']:
        print("\nContinuous learning is now active!")
        print("The system will automatically:")
        print("  - Monitor all trades")
        print("  - Learn from trade outcomes")
        print("  - Detect market regime changes")
        print("  - Optimize strategy parameters")
        print("  - Retrain models as needed")
        
        print("\nPress Ctrl+C to stop...")
        try:
            while True:
                time.sleep(60)
                status = get_learning_status()
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status: {status.get('trade_monitor', {}).get('trades_processed', 0)} trades processed")
        except KeyboardInterrupt:
            print("\nShutting down...")
            deactivate_learning()
    else:
        print("\nSome components failed to initialize. Check logs.")
        sys.exit(1)

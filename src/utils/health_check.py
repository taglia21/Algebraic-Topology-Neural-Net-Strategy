#!/usr/bin/env python3
"""
Health Check API Endpoint
=========================

FastAPI server providing health monitoring for the trading bot.
Endpoint: http://localhost:8080/health

Returns:
{
    "status": "healthy" | "degraded" | "unhealthy",
    "last_trade_time": "2026-01-21T14:30:00Z",
    "portfolio_value": 105234.50,
    "sharpe_7d": 1.45,
    "max_dd_7d": 0.018,
    "api_status": {
        "polygon": "ok" | "error",
        "alpaca": "ok" | "error"
    },
    "uptime_seconds": 86400,
    "version": "2.1.0",
    "mode": "live"
}
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from fastapi import FastAPI, Response  # type: ignore[import-not-found]
    from fastapi.responses import JSONResponse  # type: ignore[import-not-found]
    import uvicorn  # type: ignore[import-not-found]
except ImportError:
    print("FastAPI not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)

# Configuration
HEALTH_CHECK_PORT = int(os.getenv("HEALTH_CHECK_PORT", "8080"))
STATE_DIR = PROJECT_ROOT / "state"
LOGS_DIR = PROJECT_ROOT / "logs"
STARTUP_TIME = time.time()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# STATE TRACKING
# =============================================================================

class HealthState:
    """Track health state across requests."""
    
    def __init__(self):
        self.last_trade_time: Optional[datetime] = None
        self.portfolio_value: float = 0.0
        self.daily_returns: list = []
        self.polygon_status: str = "unknown"
        self.alpaca_status: str = "unknown"
        self.last_polygon_check: float = 0
        self.last_alpaca_check: float = 0
        self.api_error_count: Dict[str, int] = {"polygon": 0, "alpaca": 0}
        self.mode: str = os.getenv("TRADING_MODE", "paper")
        
    def get_sharpe_7d(self) -> float:
        """Calculate 7-day rolling Sharpe ratio."""
        if len(self.daily_returns) < 2:
            return 0.0
        
        import numpy as np
        returns = np.array(self.daily_returns[-7:])
        if np.std(returns) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(252))
    
    def get_max_dd_7d(self) -> float:
        """Calculate 7-day max drawdown."""
        if len(self.daily_returns) < 1:
            return 0.0
        
        import numpy as np
        returns = np.array(self.daily_returns[-7:])
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        return float(np.max(drawdown))
    
    def load_from_state_files(self):
        """Load state from persistent files."""
        try:
            # Load last positions
            positions_file = STATE_DIR / "last_positions.json"
            if positions_file.exists():
                with open(positions_file) as f:
                    data = json.load(f)
                    if "timestamp" in data:
                        self.last_trade_time = datetime.fromisoformat(data["timestamp"])
                    if "portfolio_value" in data:
                        self.portfolio_value = data["portfolio_value"]
                    if "daily_returns" in data:
                        self.daily_returns = data["daily_returns"][-30:]  # Keep 30 days
                        
            # Load trading stats
            stats_file = STATE_DIR / "trading_stats.json"
            if stats_file.exists():
                with open(stats_file) as f:
                    data = json.load(f)
                    if "last_trade_time" in data:
                        self.last_trade_time = datetime.fromisoformat(data["last_trade_time"])
                        
        except Exception as e:
            logger.warning(f"Error loading state: {e}")


# Global state instance
health_state = HealthState()


# =============================================================================
# API CONNECTIVITY CHECKS
# =============================================================================

async def check_polygon_status() -> str:
    """Check Polygon API connectivity."""
    try:
        import httpx  # type: ignore[import-not-found]
        api_key = os.getenv("POLYGON_API_KEY_OTREP", "")
        
        if not api_key or api_key == "your_polygon_api_key_here":
            return "not_configured"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"https://api.polygon.io/v2/aggs/ticker/SPY/prev",
                params={"apiKey": api_key}
            )
            
            if response.status_code == 200:
                health_state.api_error_count["polygon"] = 0
                return "ok"
            else:
                health_state.api_error_count["polygon"] += 1
                return "error"
                
    except Exception as e:
        health_state.api_error_count["polygon"] += 1
        logger.error(f"Polygon check failed: {e}")
        return "error"


async def check_alpaca_status() -> str:
    """Check Alpaca API connectivity."""
    try:
        import httpx  # type: ignore[import-not-found]
        api_key = os.getenv("ALPACA_API_KEY", "")
        api_secret = os.getenv("ALPACA_SECRET_KEY", "")
        base_url = os.getenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
        
        if not api_key or api_key == "your_alpaca_api_key_here":
            return "not_configured"
        
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{base_url}/v2/account",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                health_state.portfolio_value = float(data.get("portfolio_value", 0))
                health_state.api_error_count["alpaca"] = 0
                return "ok"
            else:
                health_state.api_error_count["alpaca"] += 1
                return "error"
                
    except Exception as e:
        health_state.api_error_count["alpaca"] += 1
        logger.error(f"Alpaca check failed: {e}")
        return "error"


# =============================================================================
# DISCORD NOTIFICATIONS
# =============================================================================

async def send_discord_alert(title: str, message: str, color: int = 0xFF0000):
    """Send alert to Discord webhook."""
    try:
        import httpx  # type: ignore[import-not-found]
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
        
        if not webhook_url or webhook_url == "your_discord_webhook_url_here":
            return
        
        embed = {
            "title": f"🚨 {title}",
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "TDA Trading Bot Health Monitor"}
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(webhook_url, json={"embeds": [embed]})
            
    except Exception as e:
        logger.error(f"Discord alert failed: {e}")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(f"Health check server starting on port {HEALTH_CHECK_PORT}")
    health_state.load_from_state_files()
    yield
    logger.info("Health check server shutting down")


app = FastAPI(
    title="TDA Trading Bot Health API",
    description="Health monitoring endpoint for production trading bot",
    version="2.1.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check() -> JSONResponse:
    """
    Main health check endpoint.
    
    Returns comprehensive health status of the trading bot.
    """
    current_time = time.time()
    
    # Check APIs (rate limited to once per 60 seconds)
    if current_time - health_state.last_polygon_check > 60:
        health_state.polygon_status = await check_polygon_status()
        health_state.last_polygon_check = current_time
        
    if current_time - health_state.last_alpaca_check > 60:
        health_state.alpaca_status = await check_alpaca_status()
        health_state.last_alpaca_check = current_time
    
    # Load latest state
    health_state.load_from_state_files()
    
    # Determine overall status
    status = "healthy"
    
    # Check for degraded conditions
    if health_state.polygon_status == "error" or health_state.alpaca_status == "error":
        status = "degraded"
        
    if health_state.api_error_count["polygon"] > 5:
        status = "degraded"
        await send_discord_alert(
            "API Error Alert",
            f"Polygon API has failed {health_state.api_error_count['polygon']} consecutive times",
            color=0xFFA500
        )
        
    if health_state.api_error_count["alpaca"] > 5:
        status = "unhealthy"
        await send_discord_alert(
            "Critical API Error",
            f"Alpaca API has failed {health_state.api_error_count['alpaca']} consecutive times",
            color=0xFF0000
        )
    
    # Check max drawdown
    max_dd = health_state.get_max_dd_7d()
    if max_dd > 0.03:  # 3% drawdown threshold
        status = "degraded"
        await send_discord_alert(
            "Drawdown Alert",
            f"7-day max drawdown exceeded 3%: {max_dd:.2%}",
            color=0xFFA500
        )
    
    # Build response
    response = {
        "status": status,
        "last_trade_time": health_state.last_trade_time.isoformat() if health_state.last_trade_time else None,
        "portfolio_value": round(health_state.portfolio_value, 2),
        "sharpe_7d": round(health_state.get_sharpe_7d(), 3),
        "max_dd_7d": round(max_dd, 4),
        "api_status": {
            "polygon": health_state.polygon_status,
            "alpaca": health_state.alpaca_status,
        },
        "uptime_seconds": int(current_time - STARTUP_TIME),
        "version": "2.1.0",
        "mode": health_state.mode,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    
    # Set appropriate status code
    status_code = 200 if status == "healthy" else (503 if status == "unhealthy" else 200)
    
    return JSONResponse(content=response, status_code=status_code)


@app.get("/health/detailed")
async def detailed_health() -> JSONResponse:
    """Detailed health check with additional diagnostics."""
    import psutil  # type: ignore[import-not-found]
    
    # Get basic health
    basic_health = await health_check()
    basic_data = json.loads(basic_health.body)
    
    # Add system metrics
    basic_data["system"] = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
    }
    
    # Check log file sizes
    log_sizes = {}
    for log_file in LOGS_DIR.glob("*.log"):
        log_sizes[log_file.name] = log_file.stat().st_size / (1024 * 1024)  # MB
    basic_data["log_sizes_mb"] = log_sizes
    
    # Add recent errors from logs
    basic_data["recent_errors"] = await get_recent_errors()
    
    return JSONResponse(content=basic_data)


@app.get("/health/ready")
async def readiness_check() -> JSONResponse:
    """Kubernetes-style readiness probe."""
    # Check if bot is ready to receive traffic
    polygon_ok = health_state.polygon_status in ["ok", "unknown"]
    alpaca_ok = health_state.alpaca_status in ["ok", "unknown"]
    
    if polygon_ok and alpaca_ok:
        return JSONResponse(content={"ready": True}, status_code=200)
    else:
        return JSONResponse(content={"ready": False}, status_code=503)


@app.get("/health/live")
async def liveness_check() -> JSONResponse:
    """Kubernetes-style liveness probe."""
    # Simple check that the service is running
    return JSONResponse(content={"alive": True, "uptime": int(time.time() - STARTUP_TIME)})


@app.post("/health/alert-test")
async def test_alert() -> JSONResponse:
    """Send a test alert to Discord."""
    await send_discord_alert(
        "Test Alert",
        "This is a test alert from the health check system.",
        color=0x00FF00
    )
    return JSONResponse(content={"message": "Test alert sent"})


async def get_recent_errors(n: int = 5) -> list:
    """Get recent errors from log files."""
    errors = []
    try:
        trading_log = LOGS_DIR / "trading.log"
        if trading_log.exists():
            with open(trading_log, 'r') as f:
                lines = f.readlines()
                for line in reversed(lines[-1000:]):
                    if "ERROR" in line or "CRITICAL" in line:
                        errors.append(line.strip())
                        if len(errors) >= n:
                            break
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
    
    return errors


# =============================================================================
# STANDALONE SERVER
# =============================================================================

from enum import Enum

class HealthStatus(Enum):
    """Health status enum for external use."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheckServer:
    """Wrapper class for running health check server in background thread."""
    
    def __init__(self, port: int = 8080, version: str = "2.1.0"):
        self.port = port
        self.version = version
        self._thread: Optional[threading.Thread] = None
        self._server = None
        self._is_running = False
        
        # Update global state with version
        global health_state
        health_state.mode = version
        
    def start_background(self):
        """Start the health check server in a background thread."""
        import threading
        
        def run_server():
            import uvicorn  # type: ignore[import-not-found]
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=self.port,
                log_level="warning",
                access_log=False,
            )
            self._server = uvicorn.Server(config)
            self._is_running = True
            asyncio.run(self._server.serve())
        
        self._thread = threading.Thread(target=run_server, daemon=True)
        self._thread.start()
        logger.info(f"Health check server started on port {self.port}")
        
    def stop(self):
        """Stop the health check server."""
        if self._server:
            self._server.should_exit = True
            self._is_running = False
            logger.info("Health check server stopping")
            
    def update_status(self, status_info: Dict[str, Any]):
        """Update the health state with new information."""
        global health_state
        
        if "portfolio_value" in status_info:
            health_state.portfolio_value = status_info["portfolio_value"]
        if "last_trade_time" in status_info:
            health_state.last_trade_time = datetime.fromisoformat(status_info["last_trade_time"]) if isinstance(status_info["last_trade_time"], str) else status_info["last_trade_time"]
        if "max_drawdown_7d" in status_info:
            # Store as part of daily returns for calculation
            pass
            
    @property
    def is_running(self) -> bool:
        return self._is_running


import threading


def run_health_server():
    """Run the health check server."""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=HEALTH_CHECK_PORT,
        log_level="warning",  # Reduce noise
        access_log=False,
    )


if __name__ == "__main__":
    run_health_server()

"""
core/nav_cache.py
=================
Persistent NAV cache to avoid falling back to stale config values.
"""
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_PATH = Path("data/nav_cache.json")


class NAVCache:
    def __init__(self, cache_path: Path = _DEFAULT_CACHE_PATH):
        self._path = cache_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, nav: float) -> None:
        """Save current NAV to disk."""
        data = {
            "nav": nav,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            self._path.write_text(json.dumps(data))
            logger.debug("Cached NAV: $%.2f", nav)
        except Exception as e:
            logger.warning("Failed to cache NAV: %s", e)

    def load(self, fallback: float = 444.0) -> float:
        """Load last known NAV from disk, or return fallback."""
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                nav = float(data.get("nav", fallback))
                if nav > 0:
                    logger.info("Loaded cached NAV: $%.2f (from %s)", nav, data.get("timestamp", "unknown"))
                    return nav
        except Exception as e:
            logger.warning("Failed to load NAV cache: %s", e)
        return fallback

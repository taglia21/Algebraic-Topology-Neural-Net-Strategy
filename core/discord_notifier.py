"""
core/discord_notifier.py
========================
Discord webhook notifier for ATNN v2 trading bot.

Sends alerts to a Discord channel via webhook.
All sends are fire-and-forget with try/except — will never crash the bot.

Usage:
    from core.discord_notifier import DiscordNotifier
    notifier = DiscordNotifier()
    notifier.send_trade_alert("AAPL", "BUY", 10, 182.50)
"""

import json
import logging
import os
from datetime import datetime, timezone
from urllib import request
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

# Default webhook URL (DEREK webhook → #eod-review channel)
# Override via DISCORD_WEBHOOK_URL environment variable
_DEFAULT_WEBHOOK_URL = (
    "https://discord.com/api/webhooks/1482171912724545638/"
    "EiV03Fa7qhBj4VRXw9ItpR0w9l1-b6rI1kOwxPmeTM3ddDmf4g_uAghDRWtCW9SRF4M1"
)

# Discord embed color constants
COLOR_GREEN = 0x00FF00   # trades / success
COLOR_RED = 0xFF0000     # kill switch / errors
COLOR_BLUE = 0x0080FF    # daily reports / info
COLOR_ORANGE = 0xFF8C00  # warnings


class DiscordNotifier:
    """
    Fire-and-forget Discord webhook notifier.

    Parameters
    ----------
    webhook_url : str, optional
        Discord webhook URL. Defaults to DISCORD_WEBHOOK_URL env var or
        the built-in DEREK webhook URL.
    """

    BOT_NAME = "ATNN v2"
    BOT_AVATAR = None  # Could set to a URL for a custom avatar

    def __init__(self, webhook_url: str = None):
        self.webhook_url = (
            webhook_url
            or os.environ.get("DISCORD_WEBHOOK_URL")
            or _DEFAULT_WEBHOOK_URL
        )
        logger.debug("DiscordNotifier initialized (webhook=%s...)", self.webhook_url[:60])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_alert(self, title: str, message: str, color: int = COLOR_BLUE) -> None:
        """
        Send a generic embed alert.

        Parameters
        ----------
        title   : Embed title
        message : Embed description text
        color   : Embed sidebar color (hex int), default blue
        """
        embed = self._build_embed(title=title, description=message, color=color)
        self._send_embed(embed)

    def send_kill_switch_alert(self, reason: str) -> None:
        """Send a kill switch / circuit-breaker alert (red)."""
        embed = self._build_embed(
            title="🔴 KILL SWITCH ENGAGED",
            description=f"**Reason:** {reason}\n\nTrading has been halted. Manual review required.",
            color=COLOR_RED,
        )
        self._send_embed(embed)

    def send_trade_alert(
        self,
        ticker: str,
        action: str,
        qty: int,
        price: float,
    ) -> None:
        """
        Send a trade execution alert (green).

        Parameters
        ----------
        ticker : Stock symbol (e.g. "AAPL")
        action : "BUY" or "SELL"
        qty    : Number of shares
        price  : Execution / last price
        """
        direction_emoji = "🟢" if action.upper() == "BUY" else "🔴"
        embed = self._build_embed(
            title=f"{direction_emoji} Trade Executed: {action.upper()} {ticker}",
            description=(
                f"**Symbol:** {ticker}\n"
                f"**Action:** {action.upper()}\n"
                f"**Quantity:** {qty:,} shares\n"
                f"**Price:** ${price:,.2f}\n"
                f"**Notional:** ${qty * price:,.2f}"
            ),
            color=COLOR_GREEN,
        )
        self._send_embed(embed)

    def send_daily_report(self, report_text: str) -> None:
        """
        Send the EOD daily report as a Discord embed (blue).

        Long reports are automatically truncated to Discord's 4096-char limit.
        """
        # Discord embed description limit is 4096 characters
        max_len = 4000
        if len(report_text) > max_len:
            report_text = report_text[:max_len] + "\n…(truncated)"

        # Wrap in code block for monospace formatting
        description = f"```\n{report_text}\n```"

        embed = self._build_embed(
            title="📊 ATNN v2 — Daily Report",
            description=description,
            color=COLOR_BLUE,
        )
        self._send_embed(embed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_embed(self, title: str, description: str, color: int) -> dict:
        """Build a Discord embed dict."""
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": now_utc,
            "footer": {"text": self.BOT_NAME},
        }

    def _send_embed(self, embed: dict) -> None:
        """
        POST the embed to Discord. Fire-and-forget — logs errors but never raises.
        """
        payload = {
            "username": self.BOT_NAME,
            "embeds": [embed],
        }
        try:
            data = json.dumps(payload).encode("utf-8")
            req = request.Request(
                self.webhook_url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "ATNN-Bot/2.0",
                },
                method="POST",
            )
            with request.urlopen(req, timeout=5) as resp:
                status = resp.status
                if status not in (200, 204):
                    logger.warning(
                        "Discord webhook returned unexpected status %d", status
                    )
        except HTTPError as e:
            logger.warning("Discord webhook HTTP error: %s %s", e.code, e.reason)
        except URLError as e:
            logger.warning("Discord webhook URL error: %s", e.reason)
        except Exception as e:
            logger.warning("Discord webhook send failed (non-fatal): %s", e)

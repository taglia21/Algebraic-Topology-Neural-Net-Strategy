"""
vrp/notifications.py
====================
Trade notification system for the VRP Alpha Engine.

Supports:
- Slack webhooks (recommended: create a #vrp-trades channel)
- Email via SMTP
- Log-only (default fallback)

Configuration via environment variables:
    SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=your@gmail.com
    SMTP_PASSWORD=app-password
    NOTIFY_EMAIL=recipient@example.com
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)


class TradeNotifier:
    """Send trade notifications via Slack and/or email."""

    def __init__(self) -> None:
        self.slack_url: Optional[str] = os.environ.get("SLACK_WEBHOOK_URL")
        self.smtp_host: Optional[str] = os.environ.get("SMTP_HOST")
        self.smtp_port: int = int(os.environ.get("SMTP_PORT", "587"))
        self.smtp_user: Optional[str] = os.environ.get("SMTP_USER")
        self.smtp_pass: Optional[str] = os.environ.get("SMTP_PASSWORD")
        self.notify_email: Optional[str] = os.environ.get("NOTIFY_EMAIL")
        self._configured = bool(self.slack_url or self.smtp_host)

        if self._configured:
            channels = []
            if self.slack_url:
                channels.append("Slack")
            if self.smtp_host:
                channels.append("Email")
            logger.info(f"Notifications enabled: {', '.join(channels)}")
        else:
            logger.info("No notification channels configured (set SLACK_WEBHOOK_URL or SMTP_HOST)")

    @property
    def is_configured(self) -> bool:
        return self._configured

    def notify_trade_open(
        self,
        pos_id: str,
        short_strike: float,
        long_strike: float,
        expiry: str,
        quantity: int,
        credit: float,
        max_risk: float,
        spx: float,
        vix: float,
    ) -> None:
        """Notify on a new trade entry."""
        msg = (
            f"🟢 *NEW TRADE: {pos_id}*\n"
            f"SELL {short_strike:.0f}P / BUY {long_strike:.0f}P  exp {expiry}\n"
            f"Qty: {quantity}  |  Credit: ${credit:.0f}  |  Max Risk: ${max_risk:,.0f}\n"
            f"SPX: {spx:.0f}  |  VIX: {vix:.1f}"
        )
        self._send(msg, subject=f"VRP Trade Open: {pos_id}")

    def notify_trade_close(
        self,
        pos_id: str,
        reason: str,
        pnl: float,
        days_held: int,
    ) -> None:
        """Notify on trade exit."""
        icon = "🟢" if pnl > 0 else "🔴"
        msg = (
            f"{icon} *CLOSED: {pos_id}* — {reason}\n"
            f"P&L: ${pnl:+,.0f}  |  Held: {days_held}d"
        )
        self._send(msg, subject=f"VRP Trade Closed: {pos_id} ({reason})")

    def notify_risk_alert(self, alert: str, equity: float, drawdown: float) -> None:
        """Notify on risk management events (halt, drawdown, etc.)."""
        msg = (
            f"⚠️ *RISK ALERT*\n"
            f"{alert}\n"
            f"Equity: ${equity:,.0f}  |  Drawdown: {drawdown:.1%}"
        )
        self._send(msg, subject=f"VRP Risk Alert: {alert[:50]}")

    def notify_daily_summary(
        self,
        equity: float,
        daily_pnl: float,
        open_positions: int,
        total_risk: float,
        drawdown: float,
        spx: float,
        vix: float,
    ) -> None:
        """Send end-of-day summary."""
        pnl_icon = "📈" if daily_pnl >= 0 else "📉"
        msg = (
            f"{pnl_icon} *DAILY SUMMARY* — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n"
            f"Equity: ${equity:>10,.0f}  |  Day P&L: ${daily_pnl:+,.0f}\n"
            f"Open: {open_positions}  |  Risk: ${total_risk:,.0f}  |  DD: {drawdown:.1%}\n"
            f"SPX: {spx:.0f}  |  VIX: {vix:.1f}"
        )
        self._send(msg, subject="VRP Daily Summary")

    def notify_error(self, error_msg: str) -> None:
        """Notify on critical errors."""
        msg = f"🚨 *ERROR*\n{error_msg}"
        self._send(msg, subject=f"VRP Error: {error_msg[:50]}")

    def _send(self, message: str, subject: str = "VRP Notification") -> None:
        """Send via all configured channels."""
        logger.info(f"Notification: {subject}")

        if self.slack_url:
            self._send_slack(message)

        if self.smtp_host and self.notify_email:
            self._send_email(subject, message)

    def _send_slack(self, message: str) -> None:
        """Send to Slack webhook."""
        try:
            payload = json.dumps({"text": message}).encode("utf-8")
            req = Request(
                self.slack_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urlopen(req, timeout=10)
            logger.debug("Slack notification sent")
        except (URLError, Exception) as e:
            logger.warning(f"Slack notification failed: {e}")

    def _send_email(self, subject: str, body: str) -> None:
        """Send email via SMTP."""
        try:
            # Strip Slack markdown for email
            clean_body = body.replace("*", "").replace("_", "")

            msg = MIMEText(clean_body)
            msg["Subject"] = subject
            msg["From"] = self.smtp_user
            msg["To"] = self.notify_email

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)

            logger.debug("Email notification sent")
        except Exception as e:
            logger.warning(f"Email notification failed: {e}")

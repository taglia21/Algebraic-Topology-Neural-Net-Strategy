"""Discord integration for paper trading bot."""
import os
import requests

DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')

def send_message_to_discord(message: str) -> bool:
    """Send message to Discord webhook."""
    if not DISCORD_WEBHOOK_URL:
        print(f"[Discord] {message}")
        return True
    try:
        r = requests.post(DISCORD_WEBHOOK_URL, json={'content': message}, timeout=10)
        return r.status_code == 204
    except Exception as e:
        print(f"Discord error: {e}")
        return False

if __name__ == "__main__":
    send_message_to_discord("Test")

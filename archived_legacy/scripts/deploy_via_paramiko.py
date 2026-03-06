#!/usr/bin/env python3
"""
Deploy trading bot to DigitalOcean droplet via Paramiko SSH.

Connects to 134.209.40.95 as root, provisions Docker, clones/updates
the repo, writes .env, and starts the IBKR docker-compose stack.
"""

import sys
import time

import paramiko

HOST = "134.209.40.95"
USER = "root"
KEY_PATH = "~/.ssh/id_rsa_droplet"
TIMEOUT = 300  # seconds per command
BUILD_TIMEOUT = 900  # 15 min for docker build

ENV_CONTENTS = """\
IBKR_HOST=ib-gateway
IBKR_PORT=4002
IBKR_CLIENT_ID=1
IBKR_ACCOUNT=U22452226
TWS_USERID=aftartag21
TWS_PASSWORD=INTJ-t41
BROKER=ibkr
TRADING_MODE=live
SYMBOLS=SPY,QQQ,IWM,AAPL,MSFT
MAX_POSITION_SIZE=0.05
MAX_DAILY_LOSS=0.03
"""

COMMANDS = [
    ("Check OS", "echo OS: && cat /etc/os-release | head -3"),
    (
        "Install Docker",
        "command -v docker || (curl -fsSL https://get.docker.com | sh && systemctl enable docker && systemctl start docker)",
    ),
    ("Docker version", "docker --version"),
    (
        "Install git & compose plugin",
        "apt-get install -y git docker-compose-plugin 2>&1 | tail -5",
    ),
    ("Git version", "git --version"),
    (
        "Clone / update repo",
        '[ -d /opt/trading-bot/.git ] && (cd /opt/trading-bot && git fetch origin && git reset --hard origin/main) || git clone https://github.com/taglia21/Algebraic-Topology-Neural-Net-Strategy.git /opt/trading-bot',
    ),
    ("Latest commit", "cd /opt/trading-bot && git log --oneline -1"),
    (
        "Create directories",
        "mkdir -p /opt/trading-bot/logs /opt/trading-bot/state /opt/trading-bot/models",
    ),
    # .env is handled separately below
    (
        "Docker compose up",
        "cd /opt/trading-bot && docker compose --env-file .env -f deploy/docker-compose.ibkr.yml up -d --build 2>&1 | tail -20",
    ),
    (
        "Docker compose ps",
        "cd /opt/trading-bot && docker compose --env-file .env -f deploy/docker-compose.ibkr.yml ps",
    ),
]


def run_command(client: paramiko.SSHClient, label: str, cmd: str, timeout: int = TIMEOUT) -> int:
    """Execute a command, stream stdout/stderr, return exit code."""
    print(f"\n{'=' * 60}")
    print(f">>> [{label}]")
    print(f">>> {cmd}")
    print("-" * 60)

    _stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)

    # Stream stdout line by line
    for line in stdout:
        print(line, end="")

    # Print any stderr
    err = stderr.read().decode("utf-8", errors="replace")
    if err.strip():
        print(err, end="" if err.endswith("\n") else "\n")

    exit_code = stdout.channel.recv_exit_status()
    print(f"--- exit code: {exit_code}")
    return exit_code


def write_env_file(client: paramiko.SSHClient) -> int:
    """Write the .env file to the droplet via cat heredoc."""
    label = "Write .env"
    # Escape for shell: wrap in single-quoted heredoc delimiter
    cmd = f"cat > /opt/trading-bot/.env << 'ENVEOF'\n{ENV_CONTENTS}ENVEOF"
    return run_command(client, label, cmd)


def main() -> int:
    import os

    key_path = os.path.expanduser(KEY_PATH)
    if not os.path.exists(key_path):
        print(f"ERROR: SSH key not found at {key_path}")
        return 1

    print(f"Connecting to {USER}@{HOST} with key {key_path} ...")
    pkey = paramiko.RSAKey.from_private_key_file(key_path)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(
            hostname=HOST,
            username=USER,
            pkey=pkey,
            look_for_keys=False,
            allow_agent=False,
            timeout=30,
        )
        print(f"Connected to {HOST}\n")

        for label, cmd in COMMANDS:
            t = BUILD_TIMEOUT if "compose" in cmd and "up" in cmd else TIMEOUT
            rc = run_command(client, label, cmd, timeout=t)

            # Insert .env write after "Create directories" step
            if label == "Create directories":
                write_env_file(client)

            # Abort on critical failures (skip non-zero for apt/docker which may warn)
            if rc != 0 and label in ("Clone / update repo",):
                print(f"\nFATAL: '{label}' failed with exit code {rc}. Aborting.")
                return rc

    except paramiko.AuthenticationException:
        print("ERROR: Authentication failed. Check your SSH key.")
        return 1
    except paramiko.SSHException as e:
        print(f"ERROR: SSH error — {e}")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    finally:
        client.close()
        print(f"\nDisconnected from {HOST}")

    print("\n=== DEPLOYMENT COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

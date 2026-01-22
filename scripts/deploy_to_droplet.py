#!/usr/bin/env python3
"""
Deploy to Digital Ocean Droplet
================================

Automated deployment script for production trading bot.

Usage:
    python scripts/deploy_to_droplet.py --host <IP> --user root --key ~/.ssh/id_rsa
    
    # Or with environment variables:
    export DROPLET_HOST=123.456.789.012
    export DROPLET_USER=root
    export SSH_KEY_PATH=~/.ssh/id_rsa
    python scripts/deploy_to_droplet.py

Actions:
1. SSH to droplet
2. SCP deployment files
3. Run production_setup.sh
4. Start systemd service
5. Verify health endpoint
6. Setup cron for log rotation
"""

import os
import sys
import argparse
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


# =============================================================================
# CONFIGURATION
# =============================================================================

class DeployConfig:
    """Deployment configuration."""
    
    def __init__(self, args):
        self.host = args.host or os.getenv("DROPLET_HOST", "")
        self.user = args.user or os.getenv("DROPLET_USER", "root")
        self.ssh_key = Path(args.key or os.getenv("SSH_KEY_PATH", "~/.ssh/id_rsa")).expanduser()
        self.install_dir = "/opt/trading-bot"
        self.service_name = "trading_bot"
        self.health_port = 8080
        self.dry_run = args.dry_run
        self.skip_setup = args.skip_setup
        self.verbose = args.verbose
        
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.host:
            print("❌ Error: No host specified. Use --host or set DROPLET_HOST")
            return False
            
        if not self.ssh_key.exists():
            print(f"❌ Error: SSH key not found: {self.ssh_key}")
            return False
            
        return True


# =============================================================================
# SSH UTILITIES
# =============================================================================

class SSHClient:
    """Simple SSH client wrapper."""
    
    def __init__(self, config: DeployConfig):
        self.config = config
        self.ssh_opts = [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=30",
            "-i", str(config.ssh_key),
        ]
        
    def run_command(self, command: str, capture: bool = True) -> Tuple[int, str]:
        """Run command on remote server."""
        full_cmd = [
            "ssh",
            *self.ssh_opts,
            f"{self.config.user}@{self.config.host}",
            command
        ]
        
        if self.config.verbose:
            print(f"🔧 Running: {command}")
            
        if self.config.dry_run:
            print(f"[DRY RUN] Would run: {command}")
            return 0, ""
            
        try:
            result = subprocess.run(
                full_cmd,
                capture_output=capture,
                text=True,
                timeout=300
            )
            return result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return 1, "Command timed out"
        except Exception as e:
            return 1, str(e)
            
    def copy_file(self, local_path: Path, remote_path: str) -> bool:
        """Copy file to remote server."""
        full_cmd = [
            "scp",
            *self.ssh_opts,
            str(local_path),
            f"{self.config.user}@{self.config.host}:{remote_path}"
        ]
        
        if self.config.verbose:
            print(f"📤 Copying: {local_path} -> {remote_path}")
            
        if self.config.dry_run:
            print(f"[DRY RUN] Would copy: {local_path} -> {remote_path}")
            return True
            
        try:
            result = subprocess.run(full_cmd, capture_output=True, timeout=120)
            return result.returncode == 0
        except Exception as e:
            print(f"❌ SCP failed: {e}")
            return False
            
    def copy_directory(self, local_path: Path, remote_path: str) -> bool:
        """Copy directory to remote server."""
        full_cmd = [
            "scp",
            "-r",
            *self.ssh_opts,
            str(local_path),
            f"{self.config.user}@{self.config.host}:{remote_path}"
        ]
        
        if self.config.verbose:
            print(f"📤 Copying directory: {local_path} -> {remote_path}")
            
        if self.config.dry_run:
            print(f"[DRY RUN] Would copy directory: {local_path} -> {remote_path}")
            return True
            
        try:
            result = subprocess.run(full_cmd, capture_output=True, timeout=300)
            return result.returncode == 0
        except Exception as e:
            print(f"❌ SCP failed: {e}")
            return False


# =============================================================================
# DEPLOYMENT STEPS
# =============================================================================

def step_check_connectivity(ssh: SSHClient) -> bool:
    """Step 1: Check SSH connectivity."""
    print("\n" + "=" * 60)
    print("Step 1/7: Checking SSH connectivity")
    print("=" * 60)
    
    code, output = ssh.run_command("echo 'SSH connection successful' && uname -a")
    
    if code == 0:
        print(f"✅ Connected to {ssh.config.host}")
        print(f"   System: {output.strip().split(chr(10))[-1][:60]}")
        return True
    else:
        print(f"❌ SSH connection failed: {output}")
        return False


def step_copy_files(ssh: SSHClient) -> bool:
    """Step 2: Copy deployment files to server."""
    print("\n" + "=" * 60)
    print("Step 2/7: Copying deployment files")
    print("=" * 60)
    
    # Create temporary staging directory
    code, _ = ssh.run_command(f"mkdir -p /tmp/trading-bot-deploy")
    
    files_to_copy = [
        ("deploy/production_setup.sh", "/tmp/trading-bot-deploy/"),
        ("deploy/trading_bot.service", "/tmp/trading-bot-deploy/"),
        ("deploy/.env.template", "/tmp/trading-bot-deploy/"),
    ]
    
    # Check which files exist
    for local, remote in files_to_copy:
        local_path = PROJECT_ROOT / local
        if not local_path.exists():
            print(f"⚠️  Skipping missing file: {local}")
            continue
            
        if not ssh.copy_file(local_path, remote):
            print(f"❌ Failed to copy {local}")
            return False
        print(f"✅ Copied {local}")
        
    # Copy critical source files
    src_dirs = ["src/utils"]
    for src_dir in src_dirs:
        local_path = PROJECT_ROOT / src_dir
        if local_path.exists():
            ssh.run_command(f"mkdir -p /tmp/trading-bot-deploy/{src_dir}")
            if not ssh.copy_directory(local_path, f"/tmp/trading-bot-deploy/{src_dir}/"):
                print(f"⚠️  Warning: Could not copy {src_dir}")
            else:
                print(f"✅ Copied {src_dir}/")
                
    return True


def step_run_setup(ssh: SSHClient) -> bool:
    """Step 3: Run production setup script."""
    print("\n" + "=" * 60)
    print("Step 3/7: Running production setup")
    print("=" * 60)
    
    if ssh.config.skip_setup:
        print("⏭️  Skipping setup (--skip-setup flag set)")
        return True
    
    # Make setup script executable
    ssh.run_command("chmod +x /tmp/trading-bot-deploy/production_setup.sh")
    
    # Run setup in test mode first
    print("Running setup script (this may take 2-5 minutes)...")
    code, output = ssh.run_command(
        "cd /tmp/trading-bot-deploy && sudo bash production_setup.sh --test-mode 2>&1"
    )
    
    if code != 0:
        print(f"❌ Setup failed:")
        print(output[-2000:] if len(output) > 2000 else output)
        return False
        
    print("✅ Production setup completed")
    
    # Show verification output
    verification_lines = [l for l in output.split('\n') if '✅' in l or '❌' in l]
    for line in verification_lines[-10:]:
        print(f"   {line.strip()}")
        
    return True


def step_configure_env(ssh: SSHClient) -> bool:
    """Step 4: Configure environment variables."""
    print("\n" + "=" * 60)
    print("Step 4/7: Configuring environment")
    print("=" * 60)
    
    # Check if .env exists
    code, output = ssh.run_command(f"test -f {ssh.config.install_dir}/.env && echo 'exists'")
    
    if "exists" in output:
        print("✅ Environment file already configured")
        
        # Validate required variables
        code, output = ssh.run_command(
            f"grep -E '^(POLYGON_API_KEY|ALPACA_API_KEY|DISCORD_WEBHOOK)' {ssh.config.install_dir}/.env | head -3"
        )
        
        configured = 0
        for line in output.split('\n'):
            if line and 'your_' not in line.lower():
                configured += 1
                
        if configured < 3:
            print("⚠️  Warning: Some API keys may not be configured")
            print("   Please edit /opt/trading-bot/.env manually")
        else:
            print("✅ API keys appear to be configured")
            
    else:
        print("⚠️  Environment file not found")
        print("   You must configure /opt/trading-bot/.env before starting")
        
    return True


def step_start_service(ssh: SSHClient) -> bool:
    """Step 5: Start systemd service."""
    print("\n" + "=" * 60)
    print("Step 5/7: Starting systemd service")
    print("=" * 60)
    
    # Reload systemd
    ssh.run_command("sudo systemctl daemon-reload")
    
    # Stop service if running
    ssh.run_command(f"sudo systemctl stop {ssh.config.service_name} 2>/dev/null || true")
    
    # Start service
    code, output = ssh.run_command(f"sudo systemctl start {ssh.config.service_name}")
    
    if code != 0:
        print(f"❌ Failed to start service: {output}")
        return False
        
    # Wait a moment
    time.sleep(3)
    
    # Check status
    code, output = ssh.run_command(f"sudo systemctl status {ssh.config.service_name} --no-pager")
    
    if "active (running)" in output.lower():
        print("✅ Service started successfully")
        
        # Extract key info
        for line in output.split('\n'):
            if 'Active:' in line or 'Main PID:' in line:
                print(f"   {line.strip()}")
    else:
        print("⚠️  Service may not be running properly")
        print(output[-500:])
        
    return True


def step_verify_health(ssh: SSHClient) -> bool:
    """Step 6: Verify health endpoint."""
    print("\n" + "=" * 60)
    print("Step 6/7: Verifying health endpoint")
    print("=" * 60)
    
    # Wait for service to fully start
    print("Waiting for health endpoint to become available...")
    
    max_attempts = 10
    for attempt in range(max_attempts):
        time.sleep(3)
        
        code, output = ssh.run_command(
            f"curl -s http://localhost:{ssh.config.health_port}/health 2>/dev/null"
        )
        
        if code == 0 and output.strip():
            try:
                health = json.loads(output)
                print("✅ Health endpoint responding:")
                print(f"   Status: {health.get('status', 'unknown')}")
                print(f"   Uptime: {health.get('uptime_seconds', 0)} seconds")
                print(f"   Mode: {health.get('mode', 'unknown')}")
                print(f"   API Status:")
                print(f"     Polygon: {health.get('api_status', {}).get('polygon', 'unknown')}")
                print(f"     Alpaca: {health.get('api_status', {}).get('alpaca', 'unknown')}")
                
                return True
                
            except json.JSONDecodeError:
                print(f"   Attempt {attempt + 1}/{max_attempts}: Invalid JSON response")
        else:
            print(f"   Attempt {attempt + 1}/{max_attempts}: Endpoint not ready")
            
    print("⚠️  Health endpoint did not respond within timeout")
    print("   Check logs: sudo journalctl -u trading_bot -n 50")
    return False


def step_setup_monitoring(ssh: SSHClient) -> bool:
    """Step 7: Setup monitoring and alerts."""
    print("\n" + "=" * 60)
    print("Step 7/7: Setting up monitoring")
    print("=" * 60)
    
    # Create monitoring cron job
    cron_cmd = f"* * * * * curl -sf http://localhost:{ssh.config.health_port}/health > /dev/null || systemctl restart {ssh.config.service_name}"
    
    code, output = ssh.run_command(
        f"(crontab -l 2>/dev/null | grep -v 'health' ; echo '{cron_cmd}') | crontab -"
    )
    
    print("✅ Health check cron job configured (every minute)")
    
    # Verify crontab
    code, output = ssh.run_command("crontab -l 2>/dev/null | grep -c 'health' || echo '0'")
    
    if "1" in output or "2" in output:
        print("✅ Monitoring cron jobs active")
    else:
        print("⚠️  Monitoring may not be fully configured")
        
    return True


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def print_banner():
    """Print deployment banner."""
    print("")
    print("=" * 60)
    print("  TDA Trading Bot - Production Deployment")
    print("  V2.1 Live Trading System")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


def print_summary(config: DeployConfig, success: bool):
    """Print deployment summary."""
    print("\n" + "=" * 60)
    print("DEPLOYMENT SUMMARY")
    print("=" * 60)
    print(f"Target: {config.user}@{config.host}")
    print(f"Install Directory: {config.install_dir}")
    print(f"Service: {config.service_name}")
    print(f"Health Port: {config.health_port}")
    print(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if success:
        print("\n📋 Next Steps:")
        print(f"  1. SSH to server: ssh {config.user}@{config.host}")
        print(f"  2. Edit .env: nano {config.install_dir}/.env")
        print(f"  3. Check service: systemctl status {config.service_name}")
        print(f"  4. View logs: journalctl -u {config.service_name} -f")
        print(f"  5. Test health: curl http://{config.host}:{config.health_port}/health")
        
    print("=" * 60)


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(
        description="Deploy TDA Trading Bot to Digital Ocean Droplet"
    )
    parser.add_argument("--host", help="Droplet IP address")
    parser.add_argument("--user", default="root", help="SSH user (default: root)")
    parser.add_argument("--key", help="SSH private key path")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing")
    parser.add_argument("--skip-setup", action="store_true", help="Skip production_setup.sh")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Create config
    config = DeployConfig(args)
    
    if not config.validate():
        sys.exit(1)
        
    print(f"\n🎯 Deploying to: {config.user}@{config.host}")
    
    if config.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        
    # Create SSH client
    ssh = SSHClient(config)
    
    # Run deployment steps
    steps = [
        ("Check connectivity", step_check_connectivity),
        ("Copy files", step_copy_files),
        ("Run setup", step_run_setup),
        ("Configure env", step_configure_env),
        ("Start service", step_start_service),
        ("Verify health", step_verify_health),
        ("Setup monitoring", step_setup_monitoring),
    ]
    
    success = True
    for name, step_func in steps:
        try:
            if not step_func(ssh):
                print(f"\n❌ Step '{name}' failed. Stopping deployment.")
                success = False
                break
        except Exception as e:
            print(f"\n❌ Error in step '{name}': {e}")
            success = False
            break
            
    print_summary(config, success)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

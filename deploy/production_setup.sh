#!/bin/bash
# =============================================================================
# Production Setup Script for TDA Trading Bot
# =============================================================================
# Target: Ubuntu 22.04 LTS on Digital Ocean ($12/mo droplet, 2GB RAM, 1vCPU)
# Usage: sudo bash production_setup.sh [--test-mode]
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="/opt/trading-bot"
SERVICE_USER="tradingbot"
REPO_URL="https://github.com/taglia21/Algebraic-Topology-Neural-Net-Strategy.git"
PYTHON_VERSION="3.10"
TEST_MODE=${1:-""}

# Logging
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "Please run as root (sudo bash production_setup.sh)"
        exit 1
    fi
}

# Print banner
print_banner() {
    echo ""
    echo "=============================================="
    echo "  TDA Trading Bot - Production Setup"
    echo "  V2.1 Production System"
    echo "  $(date)"
    echo "=============================================="
    echo ""
}

# Step 1: System Update
system_update() {
    log_info "Step 1/10: Updating system packages..."
    apt update -qq
    apt upgrade -y -qq
    log_success "System updated"
}

# Step 2: Install Python and dependencies
install_python() {
    log_info "Step 2/10: Installing Python ${PYTHON_VERSION} and dependencies..."
    apt install -y -qq \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python3-pip \
        python3-dev \
        build-essential \
        git \
        curl \
        wget \
        htop \
        vim \
        jq \
        unzip \
        logrotate
    
    # Ensure python3 points to correct version
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 2>/dev/null || true
    
    log_success "Python ${PYTHON_VERSION} installed"
}

# Step 3: Configure Firewall
configure_firewall() {
    log_info "Step 3/10: Configuring firewall (ufw)..."
    apt install -y -qq ufw
    
    # Reset firewall rules
    ufw --force reset
    
    # Allow SSH (22)
    ufw allow 22/tcp comment 'SSH'
    
    # Allow health check endpoint (8080)
    ufw allow 8080/tcp comment 'Health Check API'
    
    # Enable firewall
    ufw --force enable
    
    log_success "Firewall configured (ports 22, 8080 open)"
}

# Step 4: Create service user
create_service_user() {
    log_info "Step 4/10: Creating service user '${SERVICE_USER}'..."
    
    if id "${SERVICE_USER}" &>/dev/null; then
        log_warning "User ${SERVICE_USER} already exists"
    else
        useradd -r -s /bin/false -m -d /home/${SERVICE_USER} ${SERVICE_USER}
        log_success "Service user created"
    fi
}

# Step 5: Clone repository
clone_repository() {
    log_info "Step 5/10: Cloning repository to ${INSTALL_DIR}..."
    
    if [ -d "${INSTALL_DIR}" ]; then
        log_warning "Directory ${INSTALL_DIR} already exists, updating..."
        cd ${INSTALL_DIR}
        git fetch origin
        git reset --hard origin/main
    else
        git clone ${REPO_URL} ${INSTALL_DIR}
    fi
    
    # Create required directories
    mkdir -p ${INSTALL_DIR}/logs
    mkdir -p ${INSTALL_DIR}/state
    mkdir -p ${INSTALL_DIR}/cache
    mkdir -p ${INSTALL_DIR}/results
    mkdir -p ${INSTALL_DIR}/backups
    
    # Set ownership
    chown -R ${SERVICE_USER}:${SERVICE_USER} ${INSTALL_DIR}
    
    log_success "Repository cloned/updated"
}

# Step 6: Create virtual environment and install dependencies
setup_venv() {
    log_info "Step 6/10: Setting up Python virtual environment..."
    
    cd ${INSTALL_DIR}
    
    # Create virtual environment
    python${PYTHON_VERSION} -m venv .venv
    
    # Activate and upgrade pip
    source .venv/bin/activate
    pip install --upgrade pip wheel setuptools -q
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt -q
        log_success "Python dependencies installed"
    else
        log_error "requirements.txt not found!"
        exit 1
    fi
    
    # Install additional production dependencies
    pip install fastapi uvicorn python-dotenv httpx aiofiles -q
    
    deactivate
    
    # Set ownership
    chown -R ${SERVICE_USER}:${SERVICE_USER} ${INSTALL_DIR}/.venv
    
    log_success "Virtual environment ready"
}

# Step 7: Configure environment variables
configure_environment() {
    log_info "Step 7/10: Configuring environment variables..."
    
    ENV_FILE="${INSTALL_DIR}/.env"
    
    if [ -f "${ENV_FILE}" ]; then
        log_warning ".env file already exists, skipping..."
        return
    fi
    
    # Copy template
    if [ -f "${INSTALL_DIR}/.env.template" ]; then
        cp ${INSTALL_DIR}/.env.template ${ENV_FILE}
    else
        # Create .env template
        cat > ${ENV_FILE} << 'EOF'
# =============================================================================
# TDA Trading Bot - Production Environment Variables
# =============================================================================
# REQUIRED: All marked [REQUIRED] must be set before starting

# Polygon.io API Key [REQUIRED]
POLYGON_API_KEY_OTREP=your_polygon_api_key_here

# Alpaca Trading API [REQUIRED]
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://api.alpaca.markets

# Discord Notifications [REQUIRED]
DISCORD_WEBHOOK_URL=your_discord_webhook_url_here

# Trading Configuration
TRADING_MODE=live
INITIAL_CAPITAL=100000
MAX_POSITION_SIZE=0.03
DRAWDOWN_LIMIT=0.05
EMERGENCY_STOP_LIMIT=0.08

# Health Check
HEALTH_CHECK_PORT=8080

# Logging
LOG_LEVEL=INFO

# Digital Ocean Spaces (for backups)
DO_SPACES_KEY=your_spaces_key_here
DO_SPACES_SECRET=your_spaces_secret_here
DO_SPACES_BUCKET=trading-bot-backups
DO_SPACES_REGION=nyc3
EOF
    fi
    
    chown ${SERVICE_USER}:${SERVICE_USER} ${ENV_FILE}
    chmod 600 ${ENV_FILE}
    
    if [ "$TEST_MODE" != "--test-mode" ]; then
        echo ""
        log_warning "IMPORTANT: You must configure the .env file with your API keys!"
        echo ""
        read -p "Press Enter to edit .env file now (or Ctrl+C to exit and edit later)..."
        nano ${ENV_FILE}
    fi
    
    log_success "Environment configuration complete"
}

# Step 8: Install systemd service
install_systemd_service() {
    log_info "Step 8/10: Installing systemd service..."
    
    # Copy service file
    cp ${INSTALL_DIR}/deploy/trading_bot.service /etc/systemd/system/
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable service to start on boot
    systemctl enable trading_bot.service
    
    log_success "Systemd service installed and enabled"
}

# Step 9: Configure log rotation
configure_logrotate() {
    log_info "Step 9/10: Configuring log rotation..."
    
    cat > /etc/logrotate.d/trading-bot << EOF
${INSTALL_DIR}/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 ${SERVICE_USER} ${SERVICE_USER}
    size 100M
    postrotate
        systemctl reload trading_bot.service 2>/dev/null || true
    endscript
}
EOF
    
    log_success "Log rotation configured (7 days, 100MB max)"
}

# Step 10: Setup backup cron job
setup_backup_cron() {
    log_info "Step 10/10: Setting up backup cron job..."
    
    # Create backup script
    cat > ${INSTALL_DIR}/scripts/backup_to_spaces.sh << 'EOF'
#!/bin/bash
# Daily backup to Digital Ocean Spaces
# Runs at 00:00 UTC via cron

BACKUP_DIR="/opt/trading-bot"
DATE=$(date +%Y%m%d)
BACKUP_FILE="/tmp/trading-bot-backup-${DATE}.tar.gz"

# Create backup archive
tar -czf ${BACKUP_FILE} \
    ${BACKUP_DIR}/logs \
    ${BACKUP_DIR}/state \
    ${BACKUP_DIR}/results \
    ${BACKUP_DIR}/cache/universe \
    2>/dev/null

# Upload to Digital Ocean Spaces (requires s3cmd configured)
if command -v s3cmd &> /dev/null; then
    source ${BACKUP_DIR}/.env
    s3cmd put ${BACKUP_FILE} s3://${DO_SPACES_BUCKET}/backups/ 2>/dev/null || true
fi

# Cleanup
rm -f ${BACKUP_FILE}

# Cleanup old local logs (keep 7 days)
find ${BACKUP_DIR}/logs -name "*.log.*.gz" -mtime +7 -delete 2>/dev/null || true
EOF
    
    chmod +x ${INSTALL_DIR}/scripts/backup_to_spaces.sh
    chown ${SERVICE_USER}:${SERVICE_USER} ${INSTALL_DIR}/scripts/backup_to_spaces.sh
    
    # Add cron job for daily backup at 00:00 UTC
    (crontab -l 2>/dev/null | grep -v "backup_to_spaces.sh"; echo "0 0 * * * ${INSTALL_DIR}/scripts/backup_to_spaces.sh") | crontab -
    
    log_success "Daily backup cron job configured (00:00 UTC)"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    echo ""
    echo "=============================================="
    echo "  Installation Verification"
    echo "=============================================="
    
    # Check Python
    if ${INSTALL_DIR}/.venv/bin/python --version &>/dev/null; then
        echo -e "✅ Python: $(${INSTALL_DIR}/.venv/bin/python --version)"
    else
        echo -e "❌ Python: NOT INSTALLED"
    fi
    
    # Check key packages
    for pkg in numpy pandas scipy tensorflow sklearn fastapi; do
        if ${INSTALL_DIR}/.venv/bin/python -c "import ${pkg}" 2>/dev/null; then
            echo -e "✅ ${pkg}: installed"
        else
            echo -e "❌ ${pkg}: NOT INSTALLED"
        fi
    done
    
    # Check directories
    for dir in logs state cache results; do
        if [ -d "${INSTALL_DIR}/${dir}" ]; then
            echo -e "✅ ${dir}/: exists"
        else
            echo -e "❌ ${dir}/: MISSING"
        fi
    done
    
    # Check firewall
    echo -e "✅ Firewall status:"
    ufw status | grep -E "(22|8080)" | head -2
    
    # Check service
    if systemctl is-enabled trading_bot.service &>/dev/null; then
        echo -e "✅ Systemd service: enabled"
    else
        echo -e "❌ Systemd service: NOT ENABLED"
    fi
    
    echo ""
}

# Print next steps
print_next_steps() {
    echo ""
    echo "=============================================="
    echo "  Setup Complete!"
    echo "=============================================="
    echo ""
    echo "Next steps:"
    echo "  1. Edit .env file with your API keys:"
    echo "     sudo nano ${INSTALL_DIR}/.env"
    echo ""
    echo "  2. Test the configuration:"
    echo "     sudo -u ${SERVICE_USER} ${INSTALL_DIR}/.venv/bin/python ${INSTALL_DIR}/production_launcher.py --mode=backtest --days=5"
    echo ""
    echo "  3. Start the trading bot:"
    echo "     sudo systemctl start trading_bot.service"
    echo ""
    echo "  4. Check status:"
    echo "     sudo systemctl status trading_bot.service"
    echo "     curl http://localhost:8080/health"
    echo ""
    echo "  5. View logs:"
    echo "     sudo journalctl -u trading_bot.service -f"
    echo "     tail -f ${INSTALL_DIR}/logs/trading.log"
    echo ""
    echo "=============================================="
}

# Main execution
main() {
    check_root
    print_banner
    
    if [ "$TEST_MODE" == "--test-mode" ]; then
        log_warning "Running in TEST MODE - some steps will be simulated"
    fi
    
    system_update
    install_python
    configure_firewall
    create_service_user
    clone_repository
    setup_venv
    configure_environment
    install_systemd_service
    configure_logrotate
    setup_backup_cron
    
    verify_installation
    print_next_steps
    
    log_success "Production setup complete!"
}

# Run main function
main "$@"

#!/bin/bash
################################################################################
# Production Deployment Script for DigitalOcean Droplet
# Deploys fixed trading modules to production environment
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}   Trading System Production Deployment${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# Configuration
DROPLET_IP="${DROPLET_IP:-}"
DROPLET_USER="${DROPLET_USER:-root}"
REMOTE_DIR="/opt/trading-system"
BACKUP_DIR="/opt/trading-system-backups"

# Check required environment variables
if [ -z "$DROPLET_IP" ]; then
    echo -e "${RED}ERROR: DROPLET_IP environment variable not set${NC}"
    echo "Usage: DROPLET_IP=your.droplet.ip ./deploy_to_droplet.sh"
    exit 1
fi

echo -e "${YELLOW}Deployment Configuration:${NC}"
echo "  Target: $DROPLET_USER@$DROPLET_IP"
echo "  Remote Directory: $REMOTE_DIR"
echo "  Backup Directory: $BACKUP_DIR"
echo ""

# Confirm deployment
read -p "Deploy to production? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo -e "${RED}Deployment cancelled${NC}"
    exit 0
fi

echo -e "${GREEN}[1/9] Creating local deployment package...${NC}"
DEPLOY_PACKAGE="deploy_$(date +%Y%m%d_%H%M%S).tar.gz"

tar -czf "$DEPLOY_PACKAGE" \
    src/risk_manager.py \
    src/position_sizer.py \
    src/multi_timeframe_analyzer.py \
    src/sentiment_analyzer.py \
    src/enhanced_trading_engine.py \
    requirements.txt \
    PRODUCTION_AUDIT_REPORT.md \
    FIXES_IMPLEMENTATION_COMPLETE.md \
    PRE_MARKET_CHECKLIST.md \
    run_tests_offline.sh \
    .env.production.example

echo -e "${GREEN}  ✓ Package created: $DEPLOY_PACKAGE${NC}"

echo -e "${GREEN}[2/9] Testing SSH connection...${NC}"
if ssh -o ConnectTimeout=10 $DROPLET_USER@$DROPLET_IP "echo 'Connection OK'" > /dev/null 2>&1; then
    echo -e "${GREEN}  ✓ SSH connection successful${NC}"
else
    echo -e "${RED}  ✗ SSH connection failed${NC}"
    echo -e "${RED}  Please check your SSH keys and droplet IP${NC}"
    exit 1
fi

echo -e "${GREEN}[3/9] Creating remote directories...${NC}"
ssh $DROPLET_USER@$DROPLET_IP "mkdir -p $REMOTE_DIR $BACKUP_DIR"
echo -e "${GREEN}  ✓ Remote directories ready${NC}"

echo -e "${GREEN}[4/9] Backing up existing installation...${NC}"
BACKUP_NAME="backup_$(date +%Y%m%d_%H%M%S).tar.gz"
ssh $DROPLET_USER@$DROPLET_IP "
    if [ -d $REMOTE_DIR/src ]; then
        cd $REMOTE_DIR && \
        tar -czf $BACKUP_DIR/$BACKUP_NAME src/ requirements.txt 2>/dev/null || true && \
        echo 'Backup created: $BACKUP_NAME'
    else
        echo 'No existing installation to backup'
    fi
"
echo -e "${GREEN}  ✓ Backup complete${NC}"

echo -e "${GREEN}[5/9] Uploading deployment package...${NC}"
scp "$DEPLOY_PACKAGE" $DROPLET_USER@$DROPLET_IP:$REMOTE_DIR/
echo -e "${GREEN}  ✓ Upload complete${NC}"

echo -e "${GREEN}[6/9] Extracting files on remote server...${NC}"
ssh $DROPLET_USER@$DROPLET_IP "
    cd $REMOTE_DIR && \
    tar -xzf $DEPLOY_PACKAGE && \
    rm $DEPLOY_PACKAGE && \
    echo 'Files extracted successfully'
"
echo -e "${GREEN}  ✓ Files extracted${NC}"

echo -e "${GREEN}[7/9] Installing Python dependencies...${NC}"
ssh $DROPLET_USER@$DROPLET_IP "
    cd $REMOTE_DIR && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt && \
    echo 'Dependencies installed'
"
echo -e "${GREEN}  ✓ Dependencies installed${NC}"

echo -e "${GREEN}[8/9] Setting up environment...${NC}"
ssh $DROPLET_USER@$DROPLET_IP "
    cd $REMOTE_DIR && \
    if [ ! -f .env.production ]; then
        cp .env.production.example .env.production
        echo '⚠️  Created .env.production - PLEASE UPDATE WITH REAL API KEYS!'
    fi
    chmod +x run_tests_offline.sh
    chmod 600 .env.production
"
echo -e "${GREEN}  ✓ Environment configured${NC}"

echo -e "${GREEN}[9/9] Running validation tests...${NC}"
ssh $DROPLET_USER@$DROPLET_IP "
    cd $REMOTE_DIR && \
    python3 -c 'import src.risk_manager as rm; print(\"✓ risk_manager imports OK\")' && \
    python3 -c 'import src.position_sizer as ps; print(\"✓ position_sizer imports OK\")' && \
    python3 -c 'import src.multi_timeframe_analyzer as mta; print(\"✓ multi_timeframe_analyzer imports OK\")' && \
    python3 -c 'import src.sentiment_analyzer as sa; print(\"✓ sentiment_analyzer imports OK\")' && \
    python3 -c 'import src.enhanced_trading_engine as ete; print(\"✓ enhanced_trading_engine imports OK\")'
" || {
    echo -e "${RED}  ✗ Import tests failed${NC}"
    echo -e "${YELLOW}  Check Python version and dependencies${NC}"
}

# Cleanup local package
rm "$DEPLOY_PACKAGE"

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}   DEPLOYMENT COMPLETE!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "${YELLOW}NEXT STEPS:${NC}"
echo "1. SSH into droplet: ssh $DROPLET_USER@$DROPLET_IP"
echo "2. Navigate to: cd $REMOTE_DIR"
echo "3. Update .env.production with real API keys"
echo "4. Run offline tests: ./run_tests_offline.sh"
echo "5. Review PRE_MARKET_CHECKLIST.md before market open"
echo ""
echo -e "${YELLOW}IMPORTANT:${NC}"
echo "  - Update API keys in .env.production"
echo "  - Test all modules before market open"
echo "  - Monitor logs during first trading session"
echo ""
echo -e "${GREEN}Deployment package backed up at: $BACKUP_DIR/$BACKUP_NAME${NC}"
echo ""

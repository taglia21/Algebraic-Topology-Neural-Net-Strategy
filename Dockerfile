# V28 Production Trading System
# Multi-stage build for optimized container size

# =============================================================================
# Stage 1: Build dependencies
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt requirements-v28.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-v28.txt


# =============================================================================
# Stage 2: Production image
# =============================================================================
FROM python:3.11-slim as production

# Labels
LABEL maintainer="TDA Trading Bot Team"
LABEL version="28.0"
LABEL description="V28 Production Trading System with Advanced Regime Detection"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    TZ=America/New_York \
    V28_ENV=production \
    V28_VERSION=28.0.0

# Create non-root user
RUN groupadd -r tradingbot && useradd -r -g tradingbot tradingbot

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libgomp1 \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/logs /app/state /app/cache /app/results /app/data && \
    chown -R tradingbot:tradingbot /app

# Copy application code
COPY --chown=tradingbot:tradingbot . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Switch to non-root user
USER tradingbot

# Expose ports
# 8080: REST/WebSocket API
# 8081: Metrics/Prometheus
EXPOSE 8080 8081

# Default command
CMD ["python", "run_v28_production.py", "--mode=live"]


# =============================================================================
# Stage 3: Development image
# =============================================================================
FROM production as development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    netcat-openbsd \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy \
    ipython

USER tradingbot

# Development command
CMD ["python", "-m", "pytest", "tests/", "-v"]


# =============================================================================
# Stage 4: Test runner
# =============================================================================
FROM development as test

USER tradingbot

# Run tests on build
RUN python -m pytest tests/ -v --tb=short || true

CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=.", "--cov-report=html"]

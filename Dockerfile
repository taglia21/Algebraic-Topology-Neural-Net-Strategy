# =============================================================================
# ATNN Quant Powerhouse — Production Docker Image
# =============================================================================
# Multi-stage build: slim Python 3.11 base with minimal attack surface.
#
# Build:
#   docker build -t atnn-quant .
#
# Run (paper trading):
#   docker run --env-file .env -v $(pwd)/data/cache:/app/data/cache atnn-quant --mode paper
#
# Run (backtest):
#   docker run --env-file .env atnn-quant --mode backtest --start 2023-01-01 --end 2025-12-31
# =============================================================================

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies (TA-Lib requires C library)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ make wget && \
    rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library
RUN wget -q https://github.com/TA-Lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
    tar -xzf ta-lib-0.6.4-src.tar.gz && \
    cd ta-lib-0.6.4 && \
    ./configure --prefix=/usr && \
    make -j$(nproc) && \
    make install && \
    cd .. && rm -rf ta-lib-0.6.4 ta-lib-0.6.4-src.tar.gz

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/cache logs models/lgbm data/cache/fundamentals

# Health check endpoint (checks that Python can import the system)
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "from core.config import get_config; get_config()" || exit 1

# Default: run backtest
ENTRYPOINT ["python", "main.py"]
CMD ["--mode", "backtest", "--start", "2023-01-01", "--end", "2025-12-31"]

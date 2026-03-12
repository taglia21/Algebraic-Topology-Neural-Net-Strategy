FROM python:3.11-slim

WORKDIR /app

# Install IB Gateway dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget unzip xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create necessary directories
RUN mkdir -p data logs models config

# Default: run in live mode with production config
CMD ["python", "main.py", "--config", "config/live.yaml", "live"]

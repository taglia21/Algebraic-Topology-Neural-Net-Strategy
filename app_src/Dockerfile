FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget unzip xvfb gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch FIRST (saves ~1.8GB vs CUDA version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# Install remaining deps (torch already satisfied, will skip)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data logs models config

CMD ["python", "main.py", "--config", "config/live.yaml", "live"]

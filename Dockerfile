# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface \
    PORT=8000

# Install curl for container healthcheck and runtime utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create non-root user and group for security
RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid 10001 --create-home --home-dir /app appuser && \
    mkdir -p /app/.cache/huggingface && \
    chown -R appuser:appgroup /app

# Copy dependency definitions first for optimal layer caching
COPY requirements.txt .

# Install PyTorch CPU wheel first to optimize build time and keep image lightweight
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download and cache embedding & reranker ML models into HF_HOME
COPY scripts/download_models.py scripts/download_models.py
RUN python scripts/download_models.py

# Copy application source code and entrypoint
COPY . .

# Set executable permission on entrypoint and ensure appuser ownership
RUN chmod +x /app/docker-entrypoint.sh && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose default HTTP port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT:-8000}/health || exit 1

# Production entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default empty CMD so entrypoint starts uvicorn on dynamic $PORT
CMD []

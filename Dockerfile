# syntax=docker/dockerfile:1
# Single Optimized Dockerfile for RAG App - Development & Production

# Build argument to determine environment
ARG BUILD_ENV=production

# =============================================================================
# Base Stage - Common dependencies and system setup
# =============================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies in a single layer with BuildKit cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        wget \
        libgl1-mesa-dri \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/*

# Create non-root user early
RUN useradd -m -u 1000 appuser

# =============================================================================
# Dependencies Stage - Install Python packages
# =============================================================================
FROM base as dependencies

WORKDIR /app

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies with BuildKit cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Conditionally install development dependencies
RUN if [ "$BUILD_ENV" = "development" ]; then \
        --mount=type=cache,target=/root/.cache/pip \
        pip install watchdog python-dotenv; \
    fi

# =============================================================================
# Final Stage - Application setup
# =============================================================================
FROM dependencies as final

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/data /app/latest_model /app/logs /app/storage && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Create startup script that adapts to environment
COPY --chown=appuser:appuser <<EOF /app/start.sh
#!/bin/bash
set -e

# Load .env file if it exists
if [ -f ".env" ]; then
    echo "📋 Loading environment variables from .env file..."
    set -a
    source .env
    set +a
    echo "✅ Environment variables loaded"
fi

echo "=========================================="
if [ "$BUILD_ENV" = "development" ]; then
    echo "🔧 DuraFlex Technical Assistant (DEV)"
else
    echo "🔧 DuraFlex Technical Assistant"
fi
echo "=========================================="
echo ""

# Check if index exists
if [ -d "latest_model" ] && [ -f "latest_model/docstore.json" ]; then
    echo "✅ RAG index found in latest_model/"
    echo "   📊 Indexed chunks: \$(python -c "import json; print(len(json.load(open('latest_model/docstore.json'))['docstore/data']))" 2>/dev/null || echo "unknown")"
    echo ""
else
    echo "⚠️  RAG Index Not Found! Running ingestion..."
    python ingest.py
    echo "✅ Ingestion complete!"
fi

echo "🔐 Login: admin/admin123 or tech1/tech123"
echo "🚀 Starting Streamlit server..."
echo ""

# Start Streamlit with environment-specific settings
if [ "$BUILD_ENV" = "development" ]; then
    # Development: Enable hot reloading
    exec python -m streamlit run app.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --server.runOnSave=true \
        --server.fileWatcherType=poll
else
    # Production: Optimized settings
    exec python -m streamlit run app.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false
fi
EOF

RUN chmod +x /app/start.sh

# Health check script (only for production)
COPY --chown=appuser:appuser <<EOF /app/healthcheck.sh
#!/bin/bash
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    exit 0
else
    exit 1
fi
EOF

RUN chmod +x /app/healthcheck.sh

EXPOSE 8501

# Conditional health check (only for production)
RUN if [ "$BUILD_ENV" = "production" ]; then \
        echo "HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 CMD /app/healthcheck.sh" >> /tmp/healthcheck; \
    fi

CMD ["/bin/bash", "/app/start.sh"]
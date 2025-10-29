# syntax=docker/dockerfile:1
# Single Optimized Dockerfile for RAG App - Development & Production

# Build argument to determine environment
ARG BUILD_ENV=production

# =============================================================================
# Base Stage - Common dependencies and system setup
# =============================================================================
FROM python:3.11-slim AS base

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
FROM base AS dependencies

# Redeclare ARG for this stage (required - ARGs don't persist across FROM)
ARG BUILD_ENV=production

WORKDIR /app

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies with BuildKit cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Conditionally install development dependencies
RUN if [ "$BUILD_ENV" = "development" ]; then \
        pip install watchdog python-dotenv; \
    fi

# =============================================================================
# Final Stage - Application setup
# =============================================================================
FROM dependencies AS final

# Redeclare ARG for this stage (required - ARGs don't persist across FROM)
ARG BUILD_ENV=production

# Set as ENV so it's available at runtime (for script conditionals)
ENV BUILD_ENV=${BUILD_ENV}

# Set HuggingFace cache directory to a location appuser can write to
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_DATASETS_CACHE=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories including cache directory for HuggingFace
# MUST create cache directory BEFORE switching to appuser
RUN mkdir -p /app/data /app/latest_model /app/logs /app/storage /app/.cache/huggingface && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app/.cache

# Switch to non-root user
USER appuser

# Create startup script that adapts to environment
# Using RUN heredoc instead of COPY heredoc for better compatibility
# Note: Single quotes around 'EOF' prevent variable expansion at build time
# BUILD_ENV will be evaluated at runtime via ENV variable
RUN printf '#!/bin/bash\nset -e\n\n# Load .env file if it exists\nif [ -f ".env" ]; then\n    echo "📋 Loading environment variables from .env file..."\n    set -a\n    # Use a while loop to load .env and strip carriage returns\n    while IFS= read -r line || [ -n "$line" ]; do\n        # Skip empty lines and comments\n        if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then\n            continue\n        fi\n        # Strip carriage returns and export\n        export "$(echo "$line" | tr -d '\''\r'\'')"\n    done < .env\n    set +a\n    echo "✅ Environment variables loaded"\nfi\n\necho "=========================================="\nif [ "$BUILD_ENV" = "development" ]; then\n    echo "🔧 DuraFlex Technical Assistant (DEV)"\nelse\n    echo "🔧 DuraFlex Technical Assistant"\nfi\necho "=========================================="\necho ""\n\n# Check if index exists\nif [ -d "latest_model" ] && [ -f "latest_model/docstore.json" ]; then\n    echo "✅ RAG index found in latest_model/"\n    echo "   📊 Indexed chunks: $(python -c '\''import json; print(len(json.load(open("latest_model/docstore.json"))["docstore/data"]))'\'' 2>/dev/null || echo '\''unknown'\'')"\n    echo ""\nelse\n    echo "⚠️  RAG Index Not Found! Running ingestion..."\n    python ingest.py\n    echo "✅ Ingestion complete!"\nfi\n\necho "🔐 Login: admin/admin123 or tech1/tech123"\necho "🚀 Starting Streamlit server..."\necho ""\n\n# Start Streamlit with environment-specific settings\nif [ "$BUILD_ENV" = "development" ]; then\n    # Development: Enable hot reloading\n    exec python -m streamlit run app.py \\\n        --server.port=8501 \\\n        --server.address=0.0.0.0 \\\n        --server.headless=true \\\n        --server.runOnSave=true \\\n        --server.fileWatcherType=poll\nelse\n    # Production: Optimized settings\n    exec python -m streamlit run app.py \\\n        --server.port=8501 \\\n        --server.address=0.0.0.0 \\\n        --server.headless=true \\\n        --server.enableCORS=false \\\n        --server.enableXsrfProtection=false\nfi\n' > /app/start.sh && chmod +x /app/start.sh

# Create health check script
RUN printf '#!/bin/bash\nif curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then\n    exit 0\nelse\n    exit 1\nfi\n' > /app/healthcheck.sh && chmod +x /app/healthcheck.sh

EXPOSE 8501

# Health check (always enabled for production)
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD /app/healthcheck.sh || exit 1

CMD ["/bin/bash", "/app/start.sh"]
